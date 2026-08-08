"""The Dirichlet noise at the root, against rlax's formula.

It mirrors Stoix's `apply_exploration_noise` (`ff_spo.py:119`), which calls
`rlax.add_dirichlet_noise`:

    noisy = (1 - fraction) * prior + fraction * noise,     noise ~ Dir(alpha)

With alpha = 1 (Stoix's default) the symmetric Dirichlet is uniform over the
simplex, and it is sampled by normalising exponentials: e_i = -ln(u_i),
noise = e / SUM(e). Here that computation is checked with known uniforms, so the
expected value is worked out by hand and does not depend on the generator.

What gets tested, and why each thing:
  - with fraction = 0 the logits come out **bit for bit identical** (it is Stoix's
    default, so the whole preceding suite depends on it being inert);
  - with fraction > 0 exactly the formula comes out;
  - a masked cell (NEG_INF) **stays masked**, or the noise would leave the search
    sampling illegal moves;
  - the noise sums to 1 over the actions (it is a Dirichlet, not just any vector);
  - and with more than one block of threads.
"""

from std.gpu.host import DeviceContext
from std.math import log, abs

from ops.common import dtype, NEG_INF
from systems.spo.root import add_dirichlet_noise_kernel
from systems.spo.launch import TPB, blocks_for
from tests.helpers import upload, download, assert_close

comptime TOL = Scalar[dtype](1e-5)
comptime N_ACT = 9


def run_noise(ctx: DeviceContext, logits: List[Scalar[dtype]],
              us: List[Scalar[dtype]], n_envs: Int,
              fraction: Scalar[dtype]) raises -> List[Scalar[dtype]]:
    lg = upload[dtype](ctx, logits)
    u = upload[dtype](ctx, us)
    ctx.enqueue_function[add_dirichlet_noise_kernel, add_dirichlet_noise_kernel](
        lg.unsafe_ptr(), u.unsafe_ptr(), n_envs, N_ACT, fraction,
        grid_dim=blocks_for(n_envs), block_dim=TPB)
    ctx.synchronize()
    return download[dtype](lg, n_envs * N_ACT)


def make_uniforms(n: Int, seed: Int) -> List[Scalar[dtype]]:
    """Deterministic uniforms in (0,1), without depending on the GPU's generator."""
    out = List[Scalar[dtype]]()
    x = seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) % 2147483648
        out.append(Scalar[dtype](x % 100000 + 1) / Scalar[dtype](100001))
    return out^


def test_fraction_zero_is_a_no_op(ctx: DeviceContext) raises:
    """With fraction = 0 the logits do not move AT ALL.

    It is Stoix's default, so the whole preceding suite (28 files) depends on this
    being inert. If the kernel touched the logits with fraction=0, it would have
    silently changed every result measured so far.
    """
    n_envs = 3
    logits = List[Scalar[dtype]]()
    for e in range(n_envs):
        for a in range(N_ACT):
            logits.append(Scalar[dtype](e) - Scalar[dtype](a) * Scalar[dtype](0.37))
    us = make_uniforms(n_envs * N_ACT, 11)

    got = run_noise(ctx, logits, us, n_envs, Scalar[dtype](0))
    for i in range(n_envs * N_ACT):
        if got[i] != logits[i]:
            raise Error("con fraction=0 el logit ", i, " cambio: ", logits[i],
                        " -> ", got[i])
    print("PASS con fraction = 0 el ruido es inerte (bit a bit)")


def test_matches_the_rlax_formula(ctx: DeviceContext) raises:
    """(1-f)*prior + f*noise, with the Dirichlet(1) computed by hand.

    The expected value is computed on the host with the SAME uniforms, so if the
    kernel normalised wrongly (by dividing by n_actions instead of by the sum, for
    instance) it would show: the noise would stop summing to 1.
    """
    n_envs = 2
    fraction = Scalar[dtype](0.25)
    logits = List[Scalar[dtype]]()
    for e in range(n_envs):
        for a in range(N_ACT):
            logits.append(Scalar[dtype](0.5) * Scalar[dtype](a)
                          - Scalar[dtype](e))
    us = make_uniforms(n_envs * N_ACT, 29)

    got = run_noise(ctx, logits, us, n_envs, fraction)

    for e in range(n_envs):
        total = Scalar[dtype](0)
        for a in range(N_ACT):
            total += -log(us[e * N_ACT + a])
        noise_sum = Scalar[dtype](0)
        for a in range(N_ACT):
            noise = (-log(us[e * N_ACT + a])) / total
            noise_sum += noise
            want = (Scalar[dtype](1) - fraction) * logits[e * N_ACT + a] \
                   + fraction * noise
            assert_close(got[e * N_ACT + a], want, TOL,
                         String("env ", e, " accion ", a))
        # The noise is a Dirichlet: its components sum to 1.
        assert_close(noise_sum, Scalar[dtype](1), TOL,
                     String("el ruido del env ", e, " deberia sumar 1"))
    print("PASS el ruido coincide con la formula de rlax (Dirichlet alpha=1)")


def test_masked_actions_stay_masked(ctx: DeviceContext) raises:
    """A masked cell stays effectively masked after the noise.

    Otherwise the search would sample illegal moves, and `ttt_apply` does not check
    legality: it would overwrite the rival's mark. The argument is that
    (1-f)*NEG_INF dominates any noise bounded in [0,1] as long as f < 1, but that
    has to be seen, not assumed.
    """
    n_envs = 1
    fraction = Scalar[dtype](0.5)
    logits = List[Scalar[dtype]]()
    for a in range(N_ACT):
        logits.append(NEG_INF if a % 2 == 0 else Scalar[dtype](1.0))
    us = make_uniforms(N_ACT, 7)

    got = run_noise(ctx, logits, us, n_envs, fraction)
    for a in range(N_ACT):
        if a % 2 == 0:
            if got[a] > Scalar[dtype](-1e30):
                raise Error("la accion tapada ", a, " dejo de estarlo: ", got[a])
        else:
            if got[a] <= Scalar[dtype](-1e30):
                raise Error("la accion legal ", a, " se tapo: ", got[a])
    print("PASS las acciones tapadas siguen tapadas con fraction = 0.5")


def test_multi_block(ctx: DeviceContext) raises:
    """70 envs: more than one block with TPB=32, and a non-round size.

    The `if e >= n_envs` guard is only tested if it is ever launched with a size
    that does not line up. It is the blind spot I have already been bitten by five
    times in this project.
    """
    n_envs = 70
    fraction = Scalar[dtype](0.3)
    logits = List[Scalar[dtype]]()
    for e in range(n_envs):
        for a in range(N_ACT):
            logits.append(Scalar[dtype](e % 5) - Scalar[dtype](a))
    us = make_uniforms(n_envs * N_ACT, 101)

    got = run_noise(ctx, logits, us, n_envs, fraction)
    # The last env is checked, which is the one falling in the partial block.
    e = n_envs - 1
    total = Scalar[dtype](0)
    for a in range(N_ACT):
        total += -log(us[e * N_ACT + a])
    for a in range(N_ACT):
        noise = (-log(us[e * N_ACT + a])) / total
        want = (Scalar[dtype](1) - fraction) * logits[e * N_ACT + a] \
               + fraction * noise
        assert_close(got[e * N_ACT + a], want, TOL,
                     String("env ", e, " accion ", a))
    print("PASS con 70 envs (varios bloques, tamano no redondo) sale igual")


def test_noise_flattens_a_peaked_prior(ctx: DeviceContext) raises:
    """The noise brings a very peaked prior closer to uniform.

    It is the effect it gets switched on for: in E2.6 we measured that a very
    confident learned prior leaves legal actions with very few particles. It is
    checked that the difference between the largest and the smallest logit GOES
    DOWN when the noise is applied, which is the operational definition of
    "flattening".
    """
    n_envs = 1
    logits = List[Scalar[dtype]]()
    for a in range(N_ACT):
        logits.append(Scalar[dtype](5) if a == 3 else Scalar[dtype](0))
    us = make_uniforms(N_ACT, 55)

    spread_before = Scalar[dtype](5)
    got = run_noise(ctx, logits, us, n_envs, Scalar[dtype](0.5))
    lo = got[0]; hi = got[0]
    for a in range(N_ACT):
        if got[a] < lo: lo = got[a]
        if got[a] > hi: hi = got[a]
    spread_after = hi - lo
    if spread_after >= spread_before:
        raise Error("el ruido deberia aplanar: rango ", spread_before, " -> ",
                    spread_after)
    print("PASS el ruido aplana un prior picudo: rango ", spread_before, " -> ",
          spread_after)


def main() raises:
    with DeviceContext() as ctx:
        test_fraction_zero_is_a_no_op(ctx)
        test_matches_the_rlax_formula(ctx)
        test_masked_actions_stay_masked(ctx)
        test_multi_block(ctx)
        test_noise_flattens_a_peaked_prior(ctx)
