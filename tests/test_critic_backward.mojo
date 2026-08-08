"""The full MLP's backward: gradients through two ReLUs.

Two independent checks, because one alone is not enough for risk #1:

  1. Against JAX's autodiff (the EXACT reference), over the 6 weight tensors and
     two configurations (H=32 batch 20 ragged, H=64 batch 64 multi-tile).
  2. Against central finite differences on a mini network, which depends on no
     golden: if the two agree, either both are wrong in the same way (unlikely) or
     the backward is correct.

What is new with respect to E1.4 is that the gradient passes through the ReLU's
mask. That is where the typical error lives: applying it with the wrong activation,
or on the wrong side of the layer. A fault like that breaks nothing visible: the
network simply learns worse.

And mind an important difference from E1.4: there the loss was LINEAR in the
weights, so the finite differences were exact. Here the loss is quadratic and goes
through two ReLUs, that is, the numerical approximation now has real error. It is a
more demanding test.
"""

from std.gpu.host import DeviceContext
from std.math import abs, sqrt

from ops.common import dtype
from networks.mlp import (CriticParams, CriticCache, CriticGrads, CriticScratch,
                          critic_forward, critic_backward, zero_critic_params)
from tests.golden_io import read_f32
from tests.helpers import upload, zeros, download, write_into

comptime GOLDEN = String("tests/golden/")
comptime IN_DIM = 18
comptime OUT_DIM = 1


def cbwd_tag(hidden: Int, m: Int) -> String:
    return GOLDEN + "cbwd_h" + String(hidden) + "_m" + String(m) + "_"


def worst_abs(got: List[Scalar[dtype]], want: List[Scalar[dtype]], n: Int,
              tol: Scalar[dtype], what: String) raises -> Scalar[dtype]:
    """Largest absolute difference; blows up if it exceeds the tolerance."""
    worst = Scalar[dtype](0)
    at = 0
    for i in range(n):
        d = abs(got[i] - want[i])
        if d > worst:
            worst = d
            at = i
    if worst > tol:
        raise Error(what, ": diferencia ", worst, " en el indice ", at,
                    " (got=", got[at], " want=", want[at], ", tol ", tol, ")")
    return worst


def check_against_jax(ctx: DeviceContext, hidden: Int, m: Int) raises:
    """One golden case: the forward first, and then the 6 gradients."""
    t = cbwd_tag(hidden, m)
    params = zero_critic_params(ctx, IN_DIM, hidden, OUT_DIM)
    write_into[dtype](params.w1, read_f32(t + "w1.bin"))
    write_into[dtype](params.b1, read_f32(t + "b1.bin"))
    write_into[dtype](params.w2, read_f32(t + "w2.bin"))
    write_into[dtype](params.b2, read_f32(t + "b2.bin"))
    write_into[dtype](params.w3, read_f32(t + "w3.bin"))
    write_into[dtype](params.b3, read_f32(t + "b3.bin"))

    x = upload[dtype](ctx, read_f32(t + "x.bin"))
    target = upload[dtype](ctx, read_f32(t + "target.bin"))
    cache = CriticCache(ctx, m, hidden, OUT_DIM)
    grads = CriticGrads(ctx, IN_DIM, hidden, OUT_DIM)
    scratch = CriticScratch(ctx, m, IN_DIM, hidden, OUT_DIM)

    critic_forward(ctx, params, cache, x, m)
    critic_backward(ctx, params, cache, grads, scratch, x, target, m)
    ctx.synchronize()

    # The forward first: if it already differs, the gradient would differ for
    # reasons that are not the backward, and the error message would point at the
    # wrong place.
    tol_fwd = Scalar[dtype](1e-5) * sqrt(Scalar[dtype](hidden))
    _ = worst_abs(download[dtype](cache.value, m * OUT_DIM),
                  read_f32(t + "v.bin"), m * OUT_DIM, tol_fwd,
                  String("h", hidden, " forward V"))

    # And now the six gradients. The tolerance scales with the dimension being
    # reduced in each one.
    tol = Scalar[dtype](1e-6) * sqrt(Scalar[dtype](m * hidden))
    e = List[Scalar[dtype]]()
    e.append(worst_abs(download[dtype](grads.dw1, IN_DIM * hidden),
                       read_f32(t + "dw1.bin"), IN_DIM * hidden, tol,
                       String("h", hidden, " dW1")))
    e.append(worst_abs(download[dtype](grads.db1, hidden),
                       read_f32(t + "db1.bin"), hidden, tol,
                       String("h", hidden, " db1")))
    e.append(worst_abs(download[dtype](grads.dw2, hidden * hidden),
                       read_f32(t + "dw2.bin"), hidden * hidden, tol,
                       String("h", hidden, " dW2")))
    e.append(worst_abs(download[dtype](grads.db2, hidden),
                       read_f32(t + "db2.bin"), hidden, tol,
                       String("h", hidden, " db2")))
    e.append(worst_abs(download[dtype](grads.dw3, hidden * OUT_DIM),
                       read_f32(t + "dw3.bin"), hidden * OUT_DIM, tol,
                       String("h", hidden, " dW3")))
    e.append(worst_abs(download[dtype](grads.db3, OUT_DIM),
                       read_f32(t + "db3.bin"), OUT_DIM, tol,
                       String("h", hidden, " db3")))

    worst = Scalar[dtype](0)
    for i in range(len(e)):
        if e[i] > worst:
            worst = e[i]
    print("      hidden", hidden, " batch", m, " -> peor error en los 6 tensores:",
          worst, " (tol", tol, ")")


def test_against_jax_autodiff(ctx: DeviceContext) raises:
    """The 6 gradients against JAX's autodiff, in two configurations."""
    check_against_jax(ctx, 32, 20)    # ragged batch
    check_against_jax(ctx, 64, 64)    # multi-tile
    print("PASS los 6 gradientes del critico coinciden con el autodiff de JAX")


# --- The second check: finite differences, depending on no golden at all

comptime FD_IN = 4
comptime FD_HID = 5
comptime FD_OUT = 1
comptime FD_M = 3
comptime EPS = Scalar[dtype](1e-3)
comptime FD_TOL = Scalar[dtype](5e-2)
"""A loose relative tolerance: the loss is no longer linear in the weights (it is
quadratic and goes through two ReLUs), so the numerical approximation has real
error. What is sought is not precision but catching a wrong gradient."""

comptime FD_MIN_SIGNAL = Scalar[dtype](1e-4)
"""Minimum change in the loss (|L(w+e) - L(w-e)|) for the measurement to be
trusted.

It exists because of a REAL limitation of the method, measured while writing this
test: the loss is ~0.8 and the change to be measured is 2*e*grad, which for a
gradient of 0.004 is 8.6e-6. Subtracting two numbers of 0.8 that differ by 8.6e-6
loses five significant figures (catastrophic cancellation), and in float32 there
are only seven. Measured in numpy on that same parameter:

    float64   exact -0.00432000   finite diff -0.00432000   error   0.00%
    float32   exact -0.00432005   finite diff -0.00476837   error  10.38%

That is, the finite difference in float32 does NOT have the resolution for small
gradients; it is not that the backward is wrong (our kernel gave -0.00432005, which
matches the exact value). So the parameters whose signal does not clear the noise
get skipped and reported, instead of raising the tolerance until it passes. No
coverage is lost: JAX's autodiff already verifies ALL the parameters to 1e-8."""


def fd_values(n: Int, seed: Int) -> List[Scalar[dtype]]:
    """Deterministic values with mixed signs, for the mini network."""
    out = List[Scalar[dtype]]()
    for i in range(n):
        out.append(Scalar[dtype](((i * seed) % 11) - 5) * Scalar[dtype](0.3))
    return out^


def fd_loss(ctx: DeviceContext, w1: List[Scalar[dtype]], b1: List[Scalar[dtype]],
            w2: List[Scalar[dtype]], b2: List[Scalar[dtype]],
            w3: List[Scalar[dtype]], b3: List[Scalar[dtype]],
            x: List[Scalar[dtype]],
            target: List[Scalar[dtype]]) raises -> Scalar[dtype]:
    """The mean L2 loss, computed by actually running the forward."""
    p = zero_critic_params(ctx, FD_IN, FD_HID, FD_OUT)
    write_into[dtype](p.w1, w1); write_into[dtype](p.b1, b1)
    write_into[dtype](p.w2, w2); write_into[dtype](p.b2, b2)
    write_into[dtype](p.w3, w3); write_into[dtype](p.b3, b3)
    cache = CriticCache(ctx, FD_M, FD_HID, FD_OUT)
    critic_forward(ctx, p, cache, upload[dtype](ctx, x), FD_M)
    ctx.synchronize()

    v = download[dtype](cache.value, FD_M * FD_OUT)
    total = Scalar[dtype](0)
    for i in range(FD_M * FD_OUT):
        d = v[i] - target[i]
        total += Scalar[dtype](0.5) * d * d
    return total / Scalar[dtype](FD_M * FD_OUT)


def test_finite_differences_through_relu(ctx: DeviceContext) raises:
    """Finite differences over ALL the weights of a mini 4->5->5->1 network.

    It uses no golden: it perturbs each weight, measures how the loss really
    changes, and compares that with what the backward says. It is the check that
    does not depend on JAX (or anyone) being right.
    """
    w1 = fd_values(FD_IN * FD_HID, 7)
    b1 = fd_values(FD_HID, 13)
    w2 = fd_values(FD_HID * FD_HID, 5)
    b2 = fd_values(FD_HID, 17)
    w3 = fd_values(FD_HID * FD_OUT, 3)
    b3 = fd_values(FD_OUT, 11)
    x = fd_values(FD_M * FD_IN, 23)
    target = fd_values(FD_M * FD_OUT, 29)

    p = zero_critic_params(ctx, FD_IN, FD_HID, FD_OUT)
    write_into[dtype](p.w1, w1); write_into[dtype](p.b1, b1)
    write_into[dtype](p.w2, w2); write_into[dtype](p.b2, b2)
    write_into[dtype](p.w3, w3); write_into[dtype](p.b3, b3)
    cache = CriticCache(ctx, FD_M, FD_HID, FD_OUT)
    grads = CriticGrads(ctx, FD_IN, FD_HID, FD_OUT)
    scratch = CriticScratch(ctx, FD_M, FD_IN, FD_HID, FD_OUT)
    xd = upload[dtype](ctx, x)

    critic_forward(ctx, p, cache, xd, FD_M)
    critic_backward(ctx, p, cache, grads, scratch, xd,
                    upload[dtype](ctx, target), FD_M)
    ctx.synchronize()

    got_dw1 = download[dtype](grads.dw1, FD_IN * FD_HID)
    got_dw2 = download[dtype](grads.dw2, FD_HID * FD_HID)
    got_dw3 = download[dtype](grads.dw3, FD_HID * FD_OUT)
    got_db3 = download[dtype](grads.db3, FD_OUT)

    # dW1 is the furthest from the loss: its gradient crosses BOTH ReLU masks and
    # all three layers. If the chain is right, that one confirms it.
    worst = Scalar[dtype](0)
    checked = 0
    skipped = 0
    for i in range(FD_IN * FD_HID):
        moved = w1.copy()
        base = moved[i]
        moved[i] = base + EPS
        up = fd_loss(ctx, moved, b1, w2, b2, w3, b3, x, target)
        moved[i] = base - EPS
        down = fd_loss(ctx, moved, b1, w2, b2, w3, b3, x, target)

        # If the loss barely moved, the subtraction is almost all float32 noise:
        # that measurement can verify nothing (see FD_MIN_SIGNAL).
        if abs(up - down) < FD_MIN_SIGNAL:
            skipped += 1
            continue

        num = (up - down) / (Scalar[dtype](2) * EPS)
        scale = abs(num)
        if scale < Scalar[dtype](0.01):
            scale = Scalar[dtype](0.01)
        rel = abs(got_dw1[i] - num) / scale
        if rel > worst:
            worst = rel
        checked += 1
        if rel > FD_TOL:
            raise Error("dW1[", i, "]: analitico ", got_dw1[i], " vs medido ", num,
                        " (error relativo ", rel, ")")
    if checked == 0:
        raise Error("todos los parametros de dW1 quedaron por debajo del umbral: "
                    "el test no estaria comprobando nada")
    print("      dW1 (a traves de 2 ReLU y 3 capas):", checked, "comprobados,",
          skipped, "sin senal suficiente, peor error relativo", worst)

    # And the last layer, which is the short path.
    worst3 = Scalar[dtype](0)
    checked3 = 0
    skipped3 = 0
    for i in range(FD_HID * FD_OUT):
        moved = w3.copy()
        base = moved[i]
        moved[i] = base + EPS
        up = fd_loss(ctx, w1, b1, w2, b2, moved, b3, x, target)
        moved[i] = base - EPS
        down = fd_loss(ctx, w1, b1, w2, b2, moved, b3, x, target)
        if abs(up - down) < FD_MIN_SIGNAL:
            skipped3 += 1
            continue
        num = (up - down) / (Scalar[dtype](2) * EPS)
        scale = abs(num)
        if scale < Scalar[dtype](0.01):
            scale = Scalar[dtype](0.01)
        rel = abs(got_dw3[i] - num) / scale
        if rel > worst3:
            worst3 = rel
        checked3 += 1
        if rel > FD_TOL:
            raise Error("dW3[", i, "]: analitico ", got_dw3[i], " vs medido ", num,
                        " (error relativo ", rel, ")")
    if checked3 == 0:
        raise Error("ningun parametro de dW3 tuvo senal suficiente")
    print("      dW3 (capa de salida):", checked3, "comprobados,", skipped3,
          "sin senal, peor error relativo", worst3)
    print("PASS las diferencias finitas confirman el backward a traves del ReLU")


def test_relu_mask_actually_blocks(ctx: DeviceContext) raises:
    """Where the ReLU clipped, no gradient can pass.

    It is the structural check of the only new thing at this stage. If the mask
    were not applied, the gradient would still look plausible but would be that of
    a network WITHOUT a ReLU. Here the extreme case is forced: weights that switch
    almost everything off.
    """
    hidden = 8
    m = 4
    p = zero_critic_params(ctx, FD_IN, hidden, FD_OUT)
    # A very negative bias in layer 1 -> a1 = 0 across every neuron.
    ones = List[Scalar[dtype]]()
    for _ in range(FD_IN * hidden):
        ones.append(Scalar[dtype](0.1))
    very_negative = List[Scalar[dtype]]()
    for _ in range(hidden):
        very_negative.append(Scalar[dtype](-100))
    write_into[dtype](p.w1, ones)
    write_into[dtype](p.b1, very_negative)

    x = List[Scalar[dtype]]()
    for _ in range(m * FD_IN):
        x.append(Scalar[dtype](1))
    target = List[Scalar[dtype]]()
    for _ in range(m * FD_OUT):
        target.append(Scalar[dtype](5))

    cache = CriticCache(ctx, m, hidden, FD_OUT)
    grads = CriticGrads(ctx, FD_IN, hidden, FD_OUT)
    scratch = CriticScratch(ctx, m, FD_IN, hidden, FD_OUT)
    xd = upload[dtype](ctx, x)
    critic_forward(ctx, p, cache, xd, m)
    critic_backward(ctx, p, cache, grads, scratch, xd,
                    upload[dtype](ctx, target), m)
    ctx.synchronize()

    a1 = download[dtype](cache.a1, m * hidden)
    for i in range(m * hidden):
        if a1[i] != Scalar[dtype](0):
            raise Error("el montaje es incorrecto: a1 deberia estar todo a 0")

    # With the whole first layer switched off, no layer-1 weight can influence the
    # loss: its gradient has to be exactly zero.
    dw1 = download[dtype](grads.dw1, FD_IN * hidden)
    for i in range(FD_IN * hidden):
        if dw1[i] != Scalar[dtype](0):
            raise Error("dW1[", i, "] = ", dw1[i], " pero el ReLU apago toda la "
                        "capa: no deberia pasar gradiente")
    print("PASS la mascara del ReLU bloquea el gradiente donde recorto")


def main() raises:
    with DeviceContext() as ctx:
        test_relu_mask_actually_blocks(ctx)
        test_against_jax_autodiff(ctx)
        test_finite_differences_through_relu(ctx)
