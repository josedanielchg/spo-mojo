"""The search's seeding: root_fn.

Three levels of checking, from most exact to most statistical:
  1. the broadcast: each particle takes ITS env's state and value,
  2. with injected uniforms, the actions come out exact,
  3. with the real RNG, the frequencies follow the prior.
"""

from std.gpu.host import DeviceContext
from std.math import abs, log

from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform
from systems.spo.particles import Particles, StepOutputs
from systems.spo.spo_types import SPOConfig
from systems.spo.root import root_fn
from tests.helpers import upload, zeros, download, assert_close, assert_eq_int

comptime TPB = 32
comptime TOL = Scalar[dtype](1e-6)


def make_config(num_envs: Int, num_particles: Int, num_actions: Int,
                state_dim: Int) -> SPOConfig:
    """A small config tailored to each test, not the paper's."""
    return SPOConfig(
        num_envs=num_envs, num_particles=num_particles, num_actions=num_actions,
        state_dim=state_dim, search_depth=4, resample_period=4,
        temperature=0.5, search_gamma=1.0, search_gae_lambda=1.0)


def test_broadcast(ctx: DeviceContext) raises:
    """Each particle starts with its env's state and value, not its neighbour's."""
    num_envs = 3
    num_particles = 4
    state_dim = 2
    cfg = make_config(num_envs, num_particles, 2, state_dim)
    p_total = cfg.num_search_particles()

    # Clearly different states per env so that a mix-up shows: env e -> (10e, 10e+1)
    root_state = List[Scalar[dtype]]()
    for e in range(num_envs):
        root_state.append(Scalar[dtype](10 * e))
        root_state.append(Scalar[dtype](10 * e + 1))

    root_value = List[Scalar[dtype]]()
    for e in range(num_envs):
        root_value.append(Scalar[dtype](100 + e))

    root_logits = List[Scalar[dtype]]()
    for _ in range(num_envs * 2):
        root_logits.append(0.0)

    us = List[Scalar[dtype]]()
    for _ in range(p_total):
        us.append(0.1)

    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)
    root_fn(ctx, particles, outputs, cfg,
            upload[dtype](ctx, root_state), upload[dtype](ctx, root_logits),
            upload[dtype](ctx, root_value), upload[dtype](ctx, us))
    ctx.synchronize()

    got_state = download[dtype](particles.state, p_total * state_dim)
    got_value = download[dtype](particles.value, p_total)

    for p in range(p_total):
        e = p // num_particles
        assert_close(got_state[p * state_dim + 0], Scalar[dtype](10 * e), TOL,
                     String("estado[0] de la particula ", p))
        assert_close(got_state[p * state_dim + 1], Scalar[dtype](10 * e + 1), TOL,
                     String("estado[1] de la particula ", p))
        assert_close(got_value[p], Scalar[dtype](100 + e), TOL,
                     String("valor de la particula ", p))
    print("PASS broadcast de estado y valor a las particulas de cada env")


def test_exact_actions_with_injected_uniforms(ctx: DeviceContext) raises:
    """With fixed logits and hand-chosen uniforms, the actions are exact.

    Prior [0.25, 0.75] -> CDF [0.25, 1.0]. A uniform < 0.25 gives action 0 and
    anything above gives action 1.
    """
    num_envs = 1
    num_particles = 6
    cfg = make_config(num_envs, num_particles, 2, 1)
    p_total = cfg.num_search_particles()

    root_logits = List[Scalar[dtype]]()
    root_logits.append(log(Scalar[dtype](0.25)))
    root_logits.append(log(Scalar[dtype](0.75)))

    us = List[Scalar[dtype]]()
    want = List[Int]()
    us.append(0.0);   want.append(0)   # lower edge
    us.append(0.20);  want.append(0)   # inside [0, 0.25)
    us.append(0.249); want.append(0)   # right against the cut, below
    us.append(0.26);  want.append(1)   # past the cut
    us.append(0.90);  want.append(1)
    us.append(0.999); want.append(1)   # upper edge

    root_state = List[Scalar[dtype]]()
    root_state.append(0.0)
    root_value = List[Scalar[dtype]]()
    root_value.append(0.0)

    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)
    root_fn(ctx, particles, outputs, cfg,
            upload[dtype](ctx, root_state), upload[dtype](ctx, root_logits),
            upload[dtype](ctx, root_value), upload[dtype](ctx, us))
    ctx.synchronize()

    got = download[idx_dtype](particles.root_actions, p_total)
    for p in range(p_total):
        assert_eq_int(Int(got[p]), want[p], String("u=", us[p], " -> accion"))
    print("PASS acciones exactas con uniformes inyectados")


def test_log_probs_match_prior(ctx: DeviceContext) raises:
    """The prior_logits field stores log pi(a|s) of each particle's action."""
    num_envs = 1
    num_particles = 6
    cfg = make_config(num_envs, num_particles, 2, 1)
    p_total = cfg.num_search_particles()

    p0 = Scalar[dtype](0.25)
    p1 = Scalar[dtype](0.75)
    root_logits = List[Scalar[dtype]]()
    root_logits.append(log(p0))
    root_logits.append(log(p1))

    us = List[Scalar[dtype]]()
    us.append(0.1); us.append(0.2); us.append(0.24)
    us.append(0.3); us.append(0.5); us.append(0.9)

    root_state = List[Scalar[dtype]]()
    root_state.append(0.0)
    root_value = List[Scalar[dtype]]()
    root_value.append(0.0)

    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)
    root_fn(ctx, particles, outputs, cfg,
            upload[dtype](ctx, root_state), upload[dtype](ctx, root_logits),
            upload[dtype](ctx, root_value), upload[dtype](ctx, us))
    ctx.synchronize()

    actions = download[idx_dtype](particles.root_actions, p_total)
    logps = download[dtype](particles.prior_logits, p_total)
    for p in range(p_total):
        want = log(p0) if Int(actions[p]) == 0 else log(p1)
        assert_close(logps[p], want, Scalar[dtype](1e-5),
                     String("log_prob de la particula ", p))
    print("PASS prior_logits = log pi(a|s) de la accion elegida")


def test_action_frequencies_follow_prior(ctx: DeviceContext) raises:
    """With the real RNG, the fraction of particles per action follows the prior.

    The large sample is obtained with MANY ENVS, not with many particles per env:
    an env's particles have to fit in one block (see TPB_PARTICLES in
    launch.mojo). 1250 envs x 16 particles = 20000 samples, which is the same
    thing while respecting the search's real limit.
    """
    num_envs = 1250
    num_particles = 16
    cfg = make_config(num_envs, num_particles, 2, 1)
    p_total = cfg.num_search_particles()

    want_p1 = Scalar[dtype](0.75)
    root_logits = List[Scalar[dtype]]()
    for _ in range(num_envs):
        root_logits.append(log(Scalar[dtype](0.25)))
        root_logits.append(log(want_p1))

    root_state = List[Scalar[dtype]]()
    root_value = List[Scalar[dtype]]()
    for _ in range(num_envs):
        root_state.append(0.0)
        root_value.append(0.0)

    uniforms = zeros[dtype](ctx, p_total)
    ctx.enqueue_function[fill_uniform, fill_uniform](
        uniforms.unsafe_ptr(), UInt32(99), UInt32(0), p_total,
        grid_dim=(p_total + TPB - 1) // TPB, block_dim=TPB)

    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)
    root_fn(ctx, particles, outputs, cfg,
            upload[dtype](ctx, root_state), upload[dtype](ctx, root_logits),
            upload[dtype](ctx, root_value), uniforms)
    ctx.synchronize()

    actions = download[idx_dtype](particles.root_actions, p_total)
    ones = 0
    for p in range(p_total):
        a = Int(actions[p])
        if a < 0 or a > 1:
            raise Error("accion invalida en ", p, ": ", a)
        if a == 1:
            ones += 1

    frac = Scalar[dtype](ones) / Scalar[dtype](p_total)
    if abs(frac - want_p1) > 0.01:
        raise Error("fraccion de la accion 1: ", frac, ", esperada ", want_p1)
    print("PASS frecuencias de accion ~ prior (accion 1 en el", frac, "de las particulas)")


def test_accumulators_start_at_zero(ctx: DeviceContext) raises:
    """The seeding does not touch the accumulators: weight, gae, terminal and depth at zero.

    It matters because the SMC weight accumulates in place: if root_fn left
    garbage, the first depth would already start biased.
    """
    cfg = make_config(2, 8, 2, 1)
    p_total = cfg.num_search_particles()

    root_state = List[Scalar[dtype]]()
    root_value = List[Scalar[dtype]]()
    for _ in range(2):
        root_state.append(1.0)
        root_value.append(5.0)
    root_logits = List[Scalar[dtype]]()
    for _ in range(2 * 2):
        root_logits.append(0.0)
    us = List[Scalar[dtype]]()
    for i in range(p_total):
        us.append(Scalar[dtype](i) / Scalar[dtype](p_total))

    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)
    root_fn(ctx, particles, outputs, cfg,
            upload[dtype](ctx, root_state), upload[dtype](ctx, root_logits),
            upload[dtype](ctx, root_value), upload[dtype](ctx, us))
    ctx.synchronize()

    weights = download[dtype](particles.resample_td_weights, p_total)
    gae = download[dtype](particles.gae, p_total)
    terminal = download[idx_dtype](particles.terminal, p_total)
    depth = download[idx_dtype](particles.depth, p_total)
    for p in range(p_total):
        assert_close(weights[p], 0.0, TOL, String("peso inicial ", p))
        assert_close(gae[p], 0.0, TOL, String("gae inicial ", p))
        assert_eq_int(Int(terminal[p]), 0, String("terminal inicial ", p))
        assert_eq_int(Int(depth[p]), 0, String("depth inicial ", p))
    print("PASS los acumuladores arrancan a cero")


def main() raises:
    with DeviceContext() as ctx:
        test_broadcast(ctx)
        test_exact_actions_with_injected_uniforms(ctx)
        test_log_probs_match_prior(ctx)
        test_action_frequencies_follow_prior(ctx)
        test_accumulators_start_at_zero(ctx)
