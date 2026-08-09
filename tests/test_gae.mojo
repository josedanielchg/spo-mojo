"""The truncated GAE against Stoix's REAL function.

Golden: tests/golden/gen/gen_gae.py, which imports
`batch_truncated_generalized_advantage_estimation` from Stoix's own repo.

Three cases:
    case0  no truncation, with episodes that end midway
    case1  WITH truncation: the branch that almost never gets exercised and where
           RL's classic silent bug lives
    case2  like real tic-tac-toe: games of 3-5 steps, reward only at the end

Besides the golden there are two hand checks that do not depend on Stoix being
right: the single-step case (where the GAE reduces to the TD error, computable by
hand) and the truncation one (where it is checked that it cuts the accumulator but
keeps its own delta, and that without truncating it DOES carry).
"""

from std.gpu.host import DeviceContext
from std.math import abs

from ops.common import dtype
from rl_utils.multistep import truncated_gae
from tests.golden_io import read_f32
from tests.helpers import upload, zeros, download

comptime GOLDEN = String("tests/golden/")
comptime GAMMA = Scalar[dtype](0.99)
comptime LAMBDA = Scalar[dtype](0.95)
comptime TOL = Scalar[dtype](1e-5)


def check_case(ctx: DeviceContext, which: Int, b: Int, t: Int) raises:
    """One golden case: advantages and targets."""
    p = GOLDEN + "gae" + String(which) + "_"
    n = b * t

    reward = upload[dtype](ctx, read_f32(p + "r.bin"))
    discount = upload[dtype](ctx, read_f32(p + "discount.bin"))
    v_tm1 = upload[dtype](ctx, read_f32(p + "v_tm1.bin"))
    v_t = upload[dtype](ctx, read_f32(p + "v_t.bin"))
    trunc = upload[dtype](ctx, read_f32(p + "trunc.bin"))
    want_adv = read_f32(p + "adv.bin")
    want_tgt = read_f32(p + "targets.bin")
    if len(want_adv) != n:
        raise Error("the golden for case ", which, " does not have the expected shape")

    adv = zeros[dtype](ctx, n)
    targets = zeros[dtype](ctx, n)
    truncated_gae(ctx, adv, targets, reward, discount, v_tm1, v_t, trunc,
                  b, t, LAMBDA)
    ctx.synchronize()

    got_adv = download[dtype](adv, n)
    got_tgt = download[dtype](targets, n)

    worst_a = Scalar[dtype](0)
    worst_t = Scalar[dtype](0)
    for i in range(n):
        da = abs(got_adv[i] - want_adv[i])
        dt = abs(got_tgt[i] - want_tgt[i])
        if da > worst_a:
            worst_a = da
        if dt > worst_t:
            worst_t = dt
        if da > TOL:
            raise Error("case", which, " ventaja[", i, "]: ", got_adv[i],
                        " vs stoix ", want_adv[i], " (diff ", da, ")")
        if dt > TOL:
            raise Error("case", which, " objetivo[", i, "]: ", got_tgt[i],
                        " vs stoix ", want_tgt[i], " (diff ", dt, ")")
    print("      case", which, " B=", b, " T=", t,
          " error max: ventajas", worst_a, " objetivos", worst_t)


def test_against_stoix(ctx: DeviceContext) raises:
    """The three cases against Stoix's implementation."""
    check_case(ctx, 0, 4, 16)     # no truncation
    check_case(ctx, 1, 4, 16)     # WITH truncation
    check_case(ctx, 2, 8, 32)     # like tic-tac-toe
    print("PASS the truncated GAE matches Stoix's on all three cases")


def test_single_step_is_td_error(ctx: DeviceContext) raises:
    """With T=1 the GAE reduces to the TD error, computable by hand.

    It depends on no golden: with a single step there is nothing to accumulate, so
    the advantage has to be exactly r + gamma*v_t - v_tm1.
    """
    r = List[Scalar[dtype]](); r.append(2.0)
    d = List[Scalar[dtype]](); d.append(GAMMA)
    vtm1 = List[Scalar[dtype]](); vtm1.append(0.5)
    vt = List[Scalar[dtype]](); vt.append(3.0)
    tr = List[Scalar[dtype]](); tr.append(0.0)

    adv = zeros[dtype](ctx, 1)
    targets = zeros[dtype](ctx, 1)
    truncated_gae(ctx, adv, targets, upload[dtype](ctx, r), upload[dtype](ctx, d),
                  upload[dtype](ctx, vtm1), upload[dtype](ctx, vt),
                  upload[dtype](ctx, tr), 1, 1, LAMBDA)
    ctx.synchronize()

    want_adv = Scalar[dtype](2.0) + GAMMA * Scalar[dtype](3.0) - Scalar[dtype](0.5)
    got_adv = download[dtype](adv, 1)[0]
    got_tgt = download[dtype](targets, 1)[0]
    if abs(got_adv - want_adv) > TOL:
        raise Error("with T=1 the advantage should be the TD error ", want_adv,
                    ", gave ", got_adv)
    if abs(got_tgt - (Scalar[dtype](0.5) + want_adv)) > TOL:
        raise Error("the target should be v_tm1 + advantage")
    print("PASS with a single step the GAE is exactly the TD error")


def test_truncation_cuts_the_accumulator(ctx: DeviceContext) raises:
    """On a truncated step, the advantage is ONLY its delta: it carries nothing from after.

    It is the fine detail of Stoix's implementation ("reset accumulator at
    truncation points while still using the current delta") and the one that
    separates a correct GAE from one that treats truncation as if it were terminal.

    Setup: two steps, the first truncated. Its advantage has to be its own delta,
    with no trace of the second step.
    """
    t_len = 2
    r = List[Scalar[dtype]](); r.append(1.0); r.append(10.0)
    d = List[Scalar[dtype]](); d.append(GAMMA); d.append(GAMMA)
    vtm1 = List[Scalar[dtype]](); vtm1.append(0.0); vtm1.append(0.0)
    vt = List[Scalar[dtype]](); vt.append(0.0); vt.append(0.0)
    tr = List[Scalar[dtype]](); tr.append(1.0); tr.append(0.0)   # step 0 truncated

    adv = zeros[dtype](ctx, t_len)
    targets = zeros[dtype](ctx, t_len)
    truncated_gae(ctx, adv, targets, upload[dtype](ctx, r), upload[dtype](ctx, d),
                  upload[dtype](ctx, vtm1), upload[dtype](ctx, vt),
                  upload[dtype](ctx, tr), 1, t_len, LAMBDA)
    ctx.synchronize()
    got = download[dtype](adv, t_len)

    # Step 1 (the last one) has nothing behind it: its advantage is its delta = 10.
    if abs(got[1] - Scalar[dtype](10)) > TOL:
        raise Error("the last step should be worth its delta (10), gave ", got[1])
    # Step 0 is truncated: its delta is 1, and it must NOT carry the 10 from behind.
    if abs(got[0] - Scalar[dtype](1)) > TOL:
        raise Error("the truncated step should be worth only its delta (1) and not "
                    "carry what follows; gave ", got[0])

    # And to show that the check discriminates: without truncating, the same setup
    # DOES carry.
    tr2 = List[Scalar[dtype]](); tr2.append(0.0); tr2.append(0.0)
    adv2 = zeros[dtype](ctx, t_len)
    targets2 = zeros[dtype](ctx, t_len)
    truncated_gae(ctx, adv2, targets2, upload[dtype](ctx, r),
                  upload[dtype](ctx, d), upload[dtype](ctx, vtm1),
                  upload[dtype](ctx, vt), upload[dtype](ctx, tr2), 1, t_len,
                  LAMBDA)
    ctx.synchronize()
    got2 = download[dtype](adv2, t_len)
    want2 = Scalar[dtype](1) + GAMMA * LAMBDA * Scalar[dtype](10)
    if abs(got2[0] - want2) > TOL:
        raise Error("without truncation, step 0 should carry: expected ", want2,
                    " and gave ", got2[0])
    print("PASS truncation cuts the accumulation but keeps its own delta")


def test_multiblock_batch(ctx: DeviceContext) raises:
    """More sequences than threads per block: the multi-block path.

    The golden's cases use B = 1, 4 and 8, all below TPB_GAE (128), that is, the
    GAE had never run across more than one block. It is the same blind spot that
    turned up in E1.4 with the backward.

    A setup checkable by hand: every sequence identical, with reward 1 at each
    step, values at 0 and no truncation. Then each delta is 1 and step t's
    advantage is the geometric sum 1 + (g*l) + (g*l)^2 + ... backwards. The LAST
    step is checked (which is exactly 1, accumulating nothing) and that ALL the
    sequences give the same: if some block computed too much or too little, some
    rows would differ from others.
    """
    b = 300          # > 128: three blocks
    t = 5
    n = b * t

    r = List[Scalar[dtype]]()
    d = List[Scalar[dtype]]()
    z = List[Scalar[dtype]]()
    tr = List[Scalar[dtype]]()
    for _ in range(n):
        r.append(Scalar[dtype](1))
        d.append(GAMMA)
        z.append(Scalar[dtype](0))
        tr.append(Scalar[dtype](0))

    adv = zeros[dtype](ctx, n)
    targets = zeros[dtype](ctx, n)
    truncated_gae(ctx, adv, targets, upload[dtype](ctx, r), upload[dtype](ctx, d),
                  upload[dtype](ctx, z), upload[dtype](ctx, z),
                  upload[dtype](ctx, tr), b, t, LAMBDA)
    ctx.synchronize()
    got = download[dtype](adv, n)

    # The last step of each sequence carries nothing: it is worth its delta = 1.
    # And going backwards the geometric series accumulates.
    want = List[Scalar[dtype]]()
    acc = Scalar[dtype](0)
    for _ in range(t):
        acc = Scalar[dtype](1) + GAMMA * LAMBDA * acc
        want.append(acc)          # want[k] = advantage k steps from the end

    for seq in range(b):
        for step in range(t):
            from_end = t - 1 - step
            expected = want[from_end]
            v = got[seq * t + step]
            if abs(v - expected) > TOL:
                raise Error("sequence ", seq, " step ", step, ": ", v,
                            " and should be ", expected)
    print("PASS multi-block: the", b, "sequences (3 blocks) give the same")


def main() raises:
    with DeviceContext() as ctx:
        test_single_step_is_td_error(ctx)
        test_truncation_cuts_the_accumulator(ctx)
        test_multiblock_batch(ctx)
        test_against_stoix(ctx)
