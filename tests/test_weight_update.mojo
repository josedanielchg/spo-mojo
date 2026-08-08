"""SMC weight and forward GAE, with two depths worked out by hand.

Everything here can be computed on paper without a GPU, so the expected values are
literal constants and not a re-implementation of the kernel (which is the classic
trap: if the test repeats the code's formula, it checks that the code does what it
does, not that it does the right thing).

With search_gamma = search_gae_lambda = 1, the GAE's factor is discount^depth.
"""

from std.gpu.host import DeviceContext

from ops.common import dtype, idx_dtype
from systems.spo.particles import Particles, StepOutputs
from systems.spo.spo_types import SPOConfig
from systems.spo.weighting import update_particles
from tests.helpers import download, write_into, assert_close, assert_eq_int

comptime TOL = Scalar[dtype](1e-6)


def make_config(num_particles: Int) -> SPOConfig:
    return SPOConfig(
        num_envs=1, num_particles=num_particles, num_actions=2,
        state_dim=1, search_depth=4, resample_period=4,
        temperature=0.5, search_gamma=1.0, search_gae_lambda=1.0)


def set_step(outputs: StepOutputs, rewards: List[Scalar[dtype]],
             discounts: List[Scalar[dtype]],
             next_values: List[Scalar[dtype]]) raises:
    """Dictates the result of a model step, without running any model."""
    write_into[dtype](outputs.reward, rewards)
    write_into[dtype](outputs.discount, discounts)
    write_into[dtype](outputs.next_value, next_values)
    logits = List[Scalar[dtype]]()
    for _ in range(len(rewards)):
        logits.append(0.0)
    write_into[dtype](outputs.next_prior_logits, logits)


def test_two_depths_by_hand(ctx: DeviceContext) raises:
    """Two consecutive depths of a live particle.

    Depth 0:  V=10, r=1, V'=12  ->  td = 1 + 12 - 10 = 3
              weight = 0 + 3 = 3
              gae    = 0 + 3 * (1*1*1)^0 = 3 * 1 = 3
    Depth 1:  V=12, r=2, V'=11  ->  td = 2 + 11 - 12 = 1
              weight = 3 + 1 = 4
              gae    = 3 + 1 * 1^1 = 4
    """
    cfg = make_config(1)
    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    v0 = List[Scalar[dtype]]()
    v0.append(10.0)
    write_into[dtype](particles.value, v0)

    # Depth 0.
    r = List[Scalar[dtype]]();  r.append(1.0)
    d = List[Scalar[dtype]]();  d.append(1.0)     # still alive
    nv = List[Scalar[dtype]](); nv.append(12.0)
    set_step(outputs, r, d, nv)
    update_particles(ctx, particles, outputs, cfg)
    ctx.synchronize()

    assert_close(download[dtype](particles.resample_td_weights, 1)[0], 3.0, TOL,
                 "profundidad 0: peso")
    assert_close(download[dtype](particles.gae, 1)[0], 3.0, TOL,
                 "profundidad 0: gae")
    assert_close(download[dtype](particles.value, 1)[0], 12.0, TOL,
                 "profundidad 0: el valor nuevo releva al viejo")
    assert_eq_int(Int(download[idx_dtype](particles.depth, 1)[0]), 1,
                  "profundidad 0: depth++")

    # Depth 1.
    r2 = List[Scalar[dtype]]();  r2.append(2.0)
    d2 = List[Scalar[dtype]]();  d2.append(1.0)
    nv2 = List[Scalar[dtype]](); nv2.append(11.0)
    set_step(outputs, r2, d2, nv2)
    update_particles(ctx, particles, outputs, cfg)
    ctx.synchronize()

    assert_close(download[dtype](particles.resample_td_weights, 1)[0], 4.0, TOL,
                 "profundidad 1: peso acumulado")
    assert_close(download[dtype](particles.gae, 1)[0], 4.0, TOL,
                 "profundidad 1: gae acumulada")
    assert_eq_int(Int(download[idx_dtype](particles.depth, 1)[0]), 2,
                  "profundidad 1: depth++")
    print("PASS dos profundidades a mano: peso 4, gae 4")


def test_death_marks_terminal_and_then_freezes(ctx: DeviceContext) raises:
    """A particle that dies stops accumulating from the next depth onwards.

    The step that kills it does count, because it was not yet dead on entry, and
    from then on its weight stays frozen.

    Depth 0:  V=10, r=1, V'=0, discount=0 (it dies)
              td = 1 + 0 - 10 = -9 ; weight = -9 ; terminal <- 1
    Depth 1:  whatever; the mask cancels it -> weight stays at -9
    """
    cfg = make_config(1)
    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    v0 = List[Scalar[dtype]]()
    v0.append(10.0)
    write_into[dtype](particles.value, v0)

    r = List[Scalar[dtype]]();  r.append(1.0)
    d = List[Scalar[dtype]]();  d.append(0.0)     # discount 0 -> it dies
    nv = List[Scalar[dtype]](); nv.append(0.0)
    set_step(outputs, r, d, nv)
    update_particles(ctx, particles, outputs, cfg)
    ctx.synchronize()

    assert_close(download[dtype](particles.resample_td_weights, 1)[0], -9.0, TOL,
                 "el paso que mata si cuenta")
    assert_eq_int(Int(download[idx_dtype](particles.terminal, 1)[0]), 1,
                  "discount 0 tiene que marcar terminal")

    # Depth 1: it is already dead, so whatever comes gets ignored.
    r2 = List[Scalar[dtype]]();  r2.append(100.0)   # a huge reward on purpose
    d2 = List[Scalar[dtype]]();  d2.append(0.0)
    nv2 = List[Scalar[dtype]](); nv2.append(50.0)
    set_step(outputs, r2, d2, nv2)
    update_particles(ctx, particles, outputs, cfg)
    ctx.synchronize()

    assert_close(download[dtype](particles.resample_td_weights, 1)[0], -9.0, TOL,
                 "una particula muerta no puede seguir sumando peso")
    print("PASS la muerte cuenta una vez y luego congela el peso")


def test_dead_particle_gae_is_frozen_by_the_decay(ctx: DeviceContext) raises:
    """A dead particle's GAE freezes on its own, with no mask.

    Unlike the weight, the GAE carries no terminal mask (just as in Stoix). What
    freezes it is that its discount is 0 and the factor is discount^depth: at depth
    1 or more, 0^depth = 0 and the delta does not go in.

    This test exists because the absence of a mask looks like an oversight and is
    not.
    """
    cfg = make_config(1)
    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    # I start already at depth 1 and dead, with an accumulated gae of 7.
    v0 = List[Scalar[dtype]]();  v0.append(10.0)
    g0 = List[Scalar[dtype]]();  g0.append(7.0)
    dep = List[Scalar[idx_dtype]](); dep.append(Scalar[idx_dtype](1))
    ter = List[Scalar[idx_dtype]](); ter.append(Scalar[idx_dtype](1))
    write_into[dtype](particles.value, v0)
    write_into[dtype](particles.gae, g0)
    write_into[idx_dtype](particles.depth, dep)
    write_into[idx_dtype](particles.terminal, ter)

    r = List[Scalar[dtype]]();  r.append(100.0)
    d = List[Scalar[dtype]]();  d.append(0.0)      # dead -> discount 0
    nv = List[Scalar[dtype]](); nv.append(50.0)
    set_step(outputs, r, d, nv)
    update_particles(ctx, particles, outputs, cfg)
    ctx.synchronize()

    assert_close(download[dtype](particles.gae, 1)[0], 7.0, TOL,
                 "0^depth deberia anular el delta de una particula muerta")
    print("PASS la gae de una particula muerta se congela por el decay")


def test_depth_zero_always_counts(ctx: DeviceContext) raises:
    """At depth 0 the factor is discount^0 = 1, even with discount 0.

    It is the 0^0 = 1 edge case. It matters because if it were implemented with an
    exponentiation returning 0 for 0^0, the first step of every particle that dies
    immediately would disappear from the GAE.

    V=10, r=1, V'=0, discount=0 -> td = -9, and the gae has to be -9, not 0.
    """
    cfg = make_config(1)
    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    v0 = List[Scalar[dtype]]()
    v0.append(10.0)
    write_into[dtype](particles.value, v0)

    r = List[Scalar[dtype]]();  r.append(1.0)
    d = List[Scalar[dtype]]();  d.append(0.0)
    nv = List[Scalar[dtype]](); nv.append(0.0)
    set_step(outputs, r, d, nv)
    update_particles(ctx, particles, outputs, cfg)
    ctx.synchronize()

    assert_close(download[dtype](particles.gae, 1)[0], -9.0, TOL,
                 "el paso de profundidad 0 tiene que contar entero (0^0 = 1)")
    print("PASS la profundidad 0 cuenta entera aunque el discount sea 0")


def test_many_particles_are_independent(ctx: DeviceContext) raises:
    """Each particle keeps its own tally: the kernel does not mix neighbours."""
    n = 64
    cfg = make_config(n)
    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    values = List[Scalar[dtype]]()
    rewards = List[Scalar[dtype]]()
    discounts = List[Scalar[dtype]]()
    next_values = List[Scalar[dtype]]()
    for p in range(n):
        values.append(Scalar[dtype](p))
        rewards.append(Scalar[dtype](2 * p))
        discounts.append(1.0)
        next_values.append(Scalar[dtype](p) + 1.0)
    write_into[dtype](particles.value, values)
    set_step(outputs, rewards, discounts, next_values)

    update_particles(ctx, particles, outputs, cfg)
    ctx.synchronize()

    weights = download[dtype](particles.resample_td_weights, n)
    for p in range(n):
        # td = 2p + (p+1) - p = 2p + 1
        assert_close(weights[p], Scalar[dtype](2 * p + 1), TOL,
                     String("peso de la particula ", p))
    print("PASS", n, "particulas independientes")


def main() raises:
    with DeviceContext() as ctx:
        test_two_depths_by_hand(ctx)
        test_death_marks_terminal_and_then_freezes(ctx)
        test_dead_particle_gae_is_frozen_by_the_decay(ctx)
        test_depth_zero_always_counts(ctx)
        test_many_particles_are_independent(ctx)
