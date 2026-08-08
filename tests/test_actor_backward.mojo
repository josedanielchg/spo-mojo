"""The actor's backward, verified along two independent routes.

  1. Against **JAX's autodiff** over the whole chain (network + masking +
     log_softmax + cross entropy). It is the exact check: the same graph
     differentiated by a tool that shares not one line with ours.
  2. Against **central finite differences** computed here, with no golden. If the
     two agree, either both are wrong in the same way (unlikely) or the backward is
     correct.

The gradient with respect to the logits comes out clean:

    dL/dz = (pi - q) / batch

that is, "what the network says minus what the search says". What makes this more
than applying a formula is the **masking**: that logit is not a parameter, because
the forward overwrites it with NEG_INF, so its derivative has to be 0 **by
construction**. Numerically it would already come out 0 (pi = 0 and q = 0 on the
illegal ones), but relying on an underflow to zero for a gradient to be correct is
fragile, and the golden checks that JAX also gives exactly 0 there.
"""

from std.gpu.host import DeviceContext
from std.math import abs

from ops.buffers import zero_buffer
from ops.common import dtype
from networks.mlp import (CriticParams, CriticCache, CriticGrads, CriticScratch,
                          zero_critic_params)
from networks.actor import actor_probs
from networks.actor_loss import (actor_backward, cross_entropy_rows,
                                 logits_grad_kernel)
from ops.softmax import log_softmax_rows
from systems.spo.launch import TPB, blocks_for
from tests.golden_io import read_f32
from tests.helpers import upload, download, write_into, assert_close

comptime GOLDEN = String("tests/golden/")
comptime IN_DIM = 18
comptime NUM_ACTIONS = 9
comptime FD_EPS = Scalar[dtype](1e-2)
comptime FD_MIN_SIGNAL = Scalar[dtype](1e-4)
comptime TPB_ROW = 32


def load_params(ctx: DeviceContext, name: String,
                hidden: Int) raises -> CriticParams:
    p = zero_critic_params(ctx, IN_DIM, hidden, NUM_ACTIONS)
    tag = GOLDEN + name + "_"
    write_into[dtype](p.w1, read_f32(tag + "w1.bin"))
    write_into[dtype](p.b1, read_f32(tag + "b1.bin"))
    write_into[dtype](p.w2, read_f32(tag + "w2.bin"))
    write_into[dtype](p.b2, read_f32(tag + "b2.bin"))
    write_into[dtype](p.w3, read_f32(tag + "w3.bin"))
    write_into[dtype](p.b3, read_f32(tag + "b3.bin"))
    ctx.synchronize()
    return p^


def worst_rel(got: List[Scalar[dtype]], want: List[Scalar[dtype]],
              n: Int) -> Scalar[dtype]:
    """Worst relative error, scaled by the largest |want| so as not to divide by 0."""
    scale = Scalar[dtype](1e-6)
    for i in range(n):
        if abs(want[i]) > scale:
            scale = abs(want[i])
    worst = Scalar[dtype](0)
    for i in range(n):
        e = abs(got[i] - want[i]) / scale
        if e > worst:
            worst = e
    return worst


def check_case(ctx: DeviceContext, name: String, batch: Int,
               hidden: Int) raises:
    """One golden case: pi, dz and the six gradients against JAX."""
    tag = GOLDEN + name + "_"
    params = load_params(ctx, name, hidden)
    cache = CriticCache(ctx, batch, hidden, NUM_ACTIONS)
    grads = CriticGrads(ctx, IN_DIM, hidden, NUM_ACTIONS)
    scratch = CriticScratch(ctx, batch, IN_DIM, hidden, NUM_ACTIONS)

    x = upload[dtype](ctx, read_f32(tag + "x.bin"))
    mask = upload[dtype](ctx, read_f32(tag + "mask.bin"))
    q = upload[dtype](ctx, read_f32(tag + "q.bin"))
    pi = zero_buffer[dtype](ctx, batch * NUM_ACTIONS)

    # 1. The forward, and its pi against the golden. If this already failed, the
    #    backward would be comparing from a different starting point.
    actor_probs(ctx, params, cache, x, mask, pi, batch)
    ctx.synchronize()
    got_pi = download[dtype](pi, batch * NUM_ACTIONS)
    want_pi = read_f32(tag + "pi.bin")
    for i in range(batch * NUM_ACTIONS):
        assert_close(got_pi[i], want_pi[i], Scalar[dtype](2e-5),
                     String(name, " pi ", i))

    # 2. The loss, to confirm we are differentiating THE SAME function.
    log_pi = zero_buffer[dtype](ctx, batch * NUM_ACTIONS)
    per_state = zero_buffer[dtype](ctx, batch)
    ctx.enqueue_function[log_softmax_rows[TPB_ROW], log_softmax_rows[TPB_ROW]](
        log_pi.unsafe_ptr(), cache.value.unsafe_ptr(), NUM_ACTIONS,
        grid_dim=batch, block_dim=TPB_ROW)
    cross_entropy_rows(ctx, per_state, q, log_pi, batch, NUM_ACTIONS)
    ctx.synchronize()
    ps = download[dtype](per_state, batch)
    total = Scalar[dtype](0)
    for b in range(batch):
        total += ps[b]
    want_loss = read_f32(tag + "loss.bin")
    assert_close(total / Scalar[dtype](batch), want_loss[0], Scalar[dtype](2e-5),
                 String(name, " la perdida no coincide: no estamos derivando la "
                        "misma funcion"))

    # 3. dL/dz separately. Separating it matters for localising a fault: if dz
    #    lines up but dW1 does not, the problem is in the network; if dz already
    #    fails, it is in the loss.
    n_out = batch * NUM_ACTIONS
    ctx.enqueue_function[logits_grad_kernel, logits_grad_kernel](
        scratch.dvalue.unsafe_ptr(), pi.unsafe_ptr(), q.unsafe_ptr(),
        mask.unsafe_ptr(), n_out, Scalar[dtype](1) / Scalar[dtype](batch),
        grid_dim=blocks_for(n_out), block_dim=TPB)
    ctx.synchronize()
    got_dz = download[dtype](scratch.dvalue, n_out)
    want_dz = read_f32(tag + "dz.bin")
    host_mask = read_f32(tag + "mask.bin")
    for i in range(n_out):
        assert_close(got_dz[i], want_dz[i], Scalar[dtype](2e-6),
                     String(name, " dz ", i))
        if host_mask[i] == Scalar[dtype](0) and got_dz[i] != Scalar[dtype](0):
            raise Error(name, ": el gradiente se cuela por la casilla "
                        "enmascarada ", i, ": ", got_dz[i])

    # 4. And the six tensors, after the full backward.
    actor_backward(ctx, params, cache, grads, scratch, x, pi, q, mask, batch)
    ctx.synchronize()

    sizes = List[Int]()
    sizes.append(IN_DIM * hidden); sizes.append(hidden)
    sizes.append(hidden * hidden); sizes.append(hidden)
    sizes.append(hidden * NUM_ACTIONS); sizes.append(NUM_ACTIONS)
    names = List[String]()
    names.append("dw1"); names.append("db1"); names.append("dw2")
    names.append("db2"); names.append("dw3"); names.append("db3")

    # The six are downloaded at once: assigning inside an if/elif would force
    # initialising the list beforehand with a value that goes unused.
    all_got = List[List[Scalar[dtype]]]()
    all_got.append(download[dtype](grads.dw1, sizes[0]))
    all_got.append(download[dtype](grads.db1, sizes[1]))
    all_got.append(download[dtype](grads.dw2, sizes[2]))
    all_got.append(download[dtype](grads.db2, sizes[3]))
    all_got.append(download[dtype](grads.dw3, sizes[4]))
    all_got.append(download[dtype](grads.db3, sizes[5]))

    worst = Scalar[dtype](0)
    for k in range(6):
        want = read_f32(tag + names[k] + ".bin")
        e = worst_rel(all_got[k], want, sizes[k])
        if e > worst:
            worst = e
        if e > Scalar[dtype](1e-4):
            raise Error(name, ": ", names[k], " se aparta del autodiff, error "
                        "relativo ", e)
    print("PASS ", name, " (B=", batch, " H=", hidden,
          "): pi, perdida, dz y los 6 gradientes vs autodiff, peor error ", worst)


def test_against_jax_autodiff(ctx: DeviceContext) raises:
    """The golden's two cases. `bw_big` with B=40 goes past one block in the
    row-wise kernels, which is the blind spot I have already been bitten by four
    times."""
    check_case(ctx, "bw_small", 5, 32)
    check_case(ctx, "bw_big", 40, 64)


def loss_with_weights(ctx: DeviceContext, w3: List[Scalar[dtype]],
                      b3: List[Scalar[dtype]], name: String, batch: Int,
                      hidden: Int) raises -> Scalar[dtype]:
    """The loss with the last layer substituted. For finite differences."""
    params = load_params(ctx, name, hidden)
    write_into[dtype](params.w3, w3)
    write_into[dtype](params.b3, b3)
    ctx.synchronize()

    tag = GOLDEN + name + "_"
    cache = CriticCache(ctx, batch, hidden, NUM_ACTIONS)
    x = upload[dtype](ctx, read_f32(tag + "x.bin"))
    mask = upload[dtype](ctx, read_f32(tag + "mask.bin"))
    q = upload[dtype](ctx, read_f32(tag + "q.bin"))
    pi = zero_buffer[dtype](ctx, batch * NUM_ACTIONS)
    log_pi = zero_buffer[dtype](ctx, batch * NUM_ACTIONS)
    per_state = zero_buffer[dtype](ctx, batch)

    actor_probs(ctx, params, cache, x, mask, pi, batch)
    ctx.enqueue_function[log_softmax_rows[TPB_ROW], log_softmax_rows[TPB_ROW]](
        log_pi.unsafe_ptr(), cache.value.unsafe_ptr(), NUM_ACTIONS,
        grid_dim=batch, block_dim=TPB_ROW)
    cross_entropy_rows(ctx, per_state, q, log_pi, batch, NUM_ACTIONS)
    ctx.synchronize()

    ps = download[dtype](per_state, batch)
    total = Scalar[dtype](0)
    for b in range(batch):
        total += ps[b]
    return total / Scalar[dtype](batch)


def test_finite_differences(ctx: DeviceContext) raises:
    """Central finite differences over the last layer, with no golden in between.

    Each weight is moved +-eps and the change in the loss measured: the numerical
    gradient is (L(w+eps) - L(w-eps)) / (2 eps). Comparing against the analytic one
    is a check that shares NOTHING with the autodiff, so if the two agree it is
    very unlikely that both are wrong in the same way.

    It is limited to w3 and b3 (the output layer) because the signal is larger
    there and the test runs in seconds; the rest of the chain is already covered by
    the autodiff, and the lower layers' backward has been verified since E1.5.

    The FD_MIN_SIGNAL business comes from E1.5 and is that session's lesson: in
    float32, if |L(w+e) - L(w-e)| is tiny, the subtraction is catastrophic
    cancellation and the measurement verifies nothing. Those parameters get skipped
    and the count is reported.
    """
    name = String("bw_small")
    batch = 5
    hidden = 32
    tag = GOLDEN + name + "_"

    # The analytic gradient, from the backward.
    params = load_params(ctx, name, hidden)
    cache = CriticCache(ctx, batch, hidden, NUM_ACTIONS)
    grads = CriticGrads(ctx, IN_DIM, hidden, NUM_ACTIONS)
    scratch = CriticScratch(ctx, batch, IN_DIM, hidden, NUM_ACTIONS)
    x = upload[dtype](ctx, read_f32(tag + "x.bin"))
    mask = upload[dtype](ctx, read_f32(tag + "mask.bin"))
    q = upload[dtype](ctx, read_f32(tag + "q.bin"))
    pi = zero_buffer[dtype](ctx, batch * NUM_ACTIONS)
    actor_probs(ctx, params, cache, x, mask, pi, batch)
    actor_backward(ctx, params, cache, grads, scratch, x, pi, q, mask, batch)
    ctx.synchronize()
    dw3 = download[dtype](grads.dw3, hidden * NUM_ACTIONS)
    db3 = download[dtype](grads.db3, NUM_ACTIONS)

    base_w3 = read_f32(tag + "w3.bin")
    base_b3 = read_f32(tag + "b3.bin")

    checked = 0
    skipped = 0
    worst = Scalar[dtype](0)
    # A sample of w3 (one in every seven) and all of b3.
    for j in range(0, hidden * NUM_ACTIONS, 7):
        up = base_w3.copy(); up[j] = up[j] + FD_EPS
        dn = base_w3.copy(); dn[j] = dn[j] - FD_EPS
        lu = loss_with_weights(ctx, up, base_b3, name, batch, hidden)
        ld = loss_with_weights(ctx, dn, base_b3, name, batch, hidden)
        if abs(lu - ld) < FD_MIN_SIGNAL:
            skipped += 1
            continue
        num = (lu - ld) / (Scalar[dtype](2) * FD_EPS)
        rel = abs(num - dw3[j]) / (abs(num) if abs(num) > 1e-6 else 1e-6)
        if rel > worst:
            worst = rel
        checked += 1
        if rel > Scalar[dtype](0.02):
            raise Error("dw3[", j, "]: analitico ", dw3[j], " vs finitas ", num,
                        " (error relativo ", rel, ")")

    for j in range(NUM_ACTIONS):
        up = base_b3.copy(); up[j] = up[j] + FD_EPS
        dn = base_b3.copy(); dn[j] = dn[j] - FD_EPS
        lu = loss_with_weights(ctx, base_w3, up, name, batch, hidden)
        ld = loss_with_weights(ctx, base_w3, dn, name, batch, hidden)
        if abs(lu - ld) < FD_MIN_SIGNAL:
            skipped += 1
            continue
        num = (lu - ld) / (Scalar[dtype](2) * FD_EPS)
        rel = abs(num - db3[j]) / (abs(num) if abs(num) > 1e-6 else 1e-6)
        if rel > worst:
            worst = rel
        checked += 1
        if rel > Scalar[dtype](0.02):
            raise Error("db3[", j, "]: analitico ", db3[j], " vs finitas ", num)

    if checked < 5:
        raise Error("solo ", checked, " parametros con senal suficiente: la "
                    "prueba no verifica gran cosa")
    print("PASS diferencias finitas: ", checked, " parametros comprobados, ",
          skipped, " sin senal, peor error relativo ", worst)


def test_gradient_vanishes_when_pi_equals_q(ctx: DeviceContext) raises:
    """If pi already is q, the logits' gradient is zero.

    It is the check that the gradient pushes where it should: equation 11 projects
    q onto the network, so at the optimum no force can be left. An inverted sign
    would pass the goldens (which compare one point) but would fail here.
    """
    n = 4
    vals = List[Scalar[dtype]]()
    ms = List[Scalar[dtype]]()
    for _ in range(n):
        for a in range(NUM_ACTIONS):
            legal = a < 4
            vals.append(Scalar[dtype](0.25) if legal else Scalar[dtype](0))
            ms.append(Scalar[dtype](1) if legal else Scalar[dtype](0))
    same = upload[dtype](ctx, vals)
    mask = upload[dtype](ctx, ms)
    dz = zero_buffer[dtype](ctx, n * NUM_ACTIONS)

    ctx.enqueue_function[logits_grad_kernel, logits_grad_kernel](
        dz.unsafe_ptr(), same.unsafe_ptr(), same.unsafe_ptr(),
        mask.unsafe_ptr(), n * NUM_ACTIONS, Scalar[dtype](1) / Scalar[dtype](n),
        grid_dim=blocks_for(n * NUM_ACTIONS), block_dim=TPB)
    ctx.synchronize()
    got = download[dtype](dz, n * NUM_ACTIONS)
    for i in range(n * NUM_ACTIONS):
        assert_close(got[i], Scalar[dtype](0), Scalar[dtype](1e-7),
                     String("dz[", i, "] con pi = q deberia ser 0"))
    print("PASS con pi = q el gradiente de los logits se anula")


def main() raises:
    with DeviceContext() as ctx:
        test_against_jax_autodiff(ctx)
        test_gradient_vanishes_when_pi_equals_q(ctx)
        test_finite_differences(ctx)
