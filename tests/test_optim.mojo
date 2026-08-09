"""Adam + global-norm clip against OPTAX's golden (Stoix's library).

Golden: tests/golden/gen/gen_adam.py, with `optax.chain(clip_by_global_norm(0.5),
adam(3e-4, eps=1e-5))`, which is literally ff_spo.py's configuration.

Two cases and three steps each:
    case0  small gradients, the global norm (0.354) does not reach the limit (0.5)
    case1  large gradients, norm 43.3 -> the clip bites

Three steps and not one because Adam's bias correction depends on the step number:
with a single step, an implementation WITHOUT the correction would give almost the
same thing.

The six tensors are compared together, which is how it is really used: the norm is
GLOBAL, that is, the clip's factor depends on all the gradients jointly. A
tensor-by-tensor test would not detect the mistake of clipping each one separately.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs, sqrt

from ops.common import dtype
from networks.optim import (AdamState, adam_step, sum_squares,
                            global_clip_scale)
from tests.golden_io import read_f32
from tests.helpers import upload, zeros, download, write_into

comptime GOLDEN = String("tests/golden/")
comptime IN_DIM = 4
comptime HIDDEN = 5
comptime OUT_DIM = 1
comptime LR = Scalar[dtype](3e-4)
comptime MAX_NORM = Scalar[dtype](0.5)
comptime STEPS = 3
comptime TOL = Scalar[dtype](2e-6)
"""A hard tolerance: the reference is exact and the operations are elementwise, so
only float32 rounding can differ."""


def check_tensor(ctx: DeviceContext, buf: DeviceBuffer[dtype], prefix: String,
                 name: String, step: Int, size: Int,
                 which: Int) raises -> Scalar[dtype]:
    """One tensor against optax's golden after step `step`."""
    got = download[dtype](buf, size)
    want = read_f32(prefix + name + "_p" + String(step) + ".bin")
    worst = Scalar[dtype](0)
    for j in range(size):
        d = abs(got[j] - want[j])
        if d > worst:
            worst = d
        if d > TOL:
            raise Error("case", which, " step ", step, " ", name, "[", j, "]: ",
                        got[j], " vs optax ", want[j], " (diff ", d, ")")
    return worst


def run_case(ctx: DeviceContext, which: Int) raises:
    """One case's three steps, comparing the 6 tensors after each one.

    The six are explicit and not in a list because `AdamState` owns device buffers
    and is not copyable, so it does not fit in a `List`. And it also leaves things
    the way the real learner will have them, working with named fields.
    """
    prefix = GOLDEN + "adam" + String(which) + "_"
    n_w1 = IN_DIM * HIDDEN
    n_b1 = HIDDEN
    n_w2 = HIDDEN * HIDDEN
    n_b2 = HIDDEN
    n_w3 = HIDDEN * OUT_DIM
    n_b3 = OUT_DIM

    p_w1 = upload[dtype](ctx, read_f32(prefix + "w1_p0.bin"))
    p_b1 = upload[dtype](ctx, read_f32(prefix + "b1_p0.bin"))
    p_w2 = upload[dtype](ctx, read_f32(prefix + "w2_p0.bin"))
    p_b2 = upload[dtype](ctx, read_f32(prefix + "b2_p0.bin"))
    p_w3 = upload[dtype](ctx, read_f32(prefix + "w3_p0.bin"))
    p_b3 = upload[dtype](ctx, read_f32(prefix + "b3_p0.bin"))

    g_w1 = upload[dtype](ctx, read_f32(prefix + "w1_g.bin"))
    g_b1 = upload[dtype](ctx, read_f32(prefix + "b1_g.bin"))
    g_w2 = upload[dtype](ctx, read_f32(prefix + "w2_g.bin"))
    g_b2 = upload[dtype](ctx, read_f32(prefix + "b2_g.bin"))
    g_w3 = upload[dtype](ctx, read_f32(prefix + "w3_g.bin"))
    g_b3 = upload[dtype](ctx, read_f32(prefix + "b3_g.bin"))

    s_w1 = AdamState(ctx, n_w1)
    s_b1 = AdamState(ctx, n_b1)
    s_w2 = AdamState(ctx, n_w2)
    s_b2 = AdamState(ctx, n_b2)
    s_w3 = AdamState(ctx, n_w3)
    s_b3 = AdamState(ctx, n_b3)

    for step in range(1, STEPS + 1):
        # 1. The GLOBAL norm: sum of squares over ALL SIX tensors together.
        total_sq = (sum_squares(ctx, g_w1, n_w1) + sum_squares(ctx, g_b1, n_b1)
                    + sum_squares(ctx, g_w2, n_w2) + sum_squares(ctx, g_b2, n_b2)
                    + sum_squares(ctx, g_w3, n_w3) + sum_squares(ctx, g_b3, n_b3))
        scale = global_clip_scale(total_sq, MAX_NORM)

        # 2. And the Adam step on each tensor, with THAT SAME factor.
        adam_step(ctx, p_w1, g_w1, s_w1, n_w1, LR, scale, step)
        adam_step(ctx, p_b1, g_b1, s_b1, n_b1, LR, scale, step)
        adam_step(ctx, p_w2, g_w2, s_w2, n_w2, LR, scale, step)
        adam_step(ctx, p_b2, g_b2, s_b2, n_b2, LR, scale, step)
        adam_step(ctx, p_w3, g_w3, s_w3, n_w3, LR, scale, step)
        adam_step(ctx, p_b3, g_b3, s_b3, n_b3, LR, scale, step)
        ctx.synchronize()

        # 3. Against optax, tensor by tensor.
        worst = check_tensor(ctx, p_w1, prefix, "w1", step, n_w1, which)
        for e in [check_tensor(ctx, p_b1, prefix, "b1", step, n_b1, which),
                  check_tensor(ctx, p_w2, prefix, "w2", step, n_w2, which),
                  check_tensor(ctx, p_b2, prefix, "b2", step, n_b2, which),
                  check_tensor(ctx, p_w3, prefix, "w3", step, n_w3, which),
                  check_tensor(ctx, p_b3, prefix, "b3", step, n_b3, which)]:
            if e > worst:
                worst = e
        if step == 1:
            print("      case", which, " norma global", sqrt(total_sq),
                  " clip factor", scale)
        print("        step", step, ": worst difference", worst)


def test_against_optax(ctx: DeviceContext) raises:
    """The two cases: without clipping and with clipping."""
    run_case(ctx, 0)
    run_case(ctx, 1)
    print("PASS adam + global clip match optax on both cases")


def test_global_norm_is_global(ctx: DeviceContext) raises:
    """The norm sums ALL the tensors, not each one on its own.

    It is the classic mistake when reimplementing the clip: clipping tensor by
    tensor changes the DIRECTION of the joint gradient; the global clip only
    changes its length. With three tensors of norm 3, 4 and 12, the global one is
    13 (not 12, nor 3+4+12).
    """
    a = List[Scalar[dtype]](); a.append(3.0)
    b = List[Scalar[dtype]](); b.append(4.0)
    c = List[Scalar[dtype]](); c.append(12.0)

    total = (sum_squares(ctx, upload[dtype](ctx, a), 1)
             + sum_squares(ctx, upload[dtype](ctx, b), 1)
             + sum_squares(ctx, upload[dtype](ctx, c), 1))
    norm = sqrt(total)
    if abs(norm - Scalar[dtype](13)) > Scalar[dtype](1e-5):
        raise Error("the global norm of (3,4,12) should be 13, gave ", norm)

    # And the factor clips exactly down to the limit, no further.
    scale = global_clip_scale(total, Scalar[dtype](6.5))
    if abs(scale - Scalar[dtype](0.5)) > Scalar[dtype](1e-6):
        raise Error("with norm 13 and limit 6.5 the factor should be 0.5, gave ",
                    scale)
    # Below the limit it touches nothing.
    if global_clip_scale(total, Scalar[dtype](20)) != Scalar[dtype](1):
        raise Error("if the norm fits, the factor has to be exactly 1")
    print("PASS the norm is global (3,4,12 -> 13) and the clip scales to the limit")


def test_sum_squares_multi_block(ctx: DeviceContext) raises:
    """The sum of squares with more elements than one block.

    With 1000 elements four blocks of 256 are needed, so the partial reduction and
    the final host-side sum both get exercised. All ones, so that the result is
    exactly the number of elements.
    """
    n = 1000
    ones = List[Scalar[dtype]]()
    for _ in range(n):
        ones.append(Scalar[dtype](1))
    got = sum_squares(ctx, upload[dtype](ctx, ones), n)
    if abs(got - Scalar[dtype](n)) > Scalar[dtype](1e-3):
        raise Error("the sum of 1000 ones squared should be 1000, gave ", got)
    print("PASS sum of squares correct across several blocks (n =", n, ")")


def main() raises:
    with DeviceContext() as ctx:
        test_sum_squares_multi_block(ctx)
        test_global_norm_is_global(ctx)
        test_against_optax(ctx)
