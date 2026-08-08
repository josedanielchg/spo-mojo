"""The resampling and ESS pieces, with inputs dictated by hand.

Nothing here runs a full search: the weights of interest are written straight into
the particles and the EXACT result is checked. When something fails, the failure
points at a specific kernel instead of at "the search".

The whole search's behaviour is tested separately, in test_search.mojo.
"""

from std.gpu.host import DeviceContext
from std.math import log

from ops.common import dtype, idx_dtype
from envs.toy_chain import NUM_ACTIONS, STATE_DIM
from systems.spo.particles import Particles, SearchScratch, SPOOutput
from systems.spo.spo_types import SPOConfig
from systems.spo.metrics import compute_ess_entropy
from systems.spo.resampling import resample
from tests.helpers import (upload, download, write_into, assert_close,
                           assert_eq_int)

comptime TOL = Scalar[dtype](1e-5)


def make_config(num_envs: Int, num_particles: Int, depth: Int,
                period: Int) -> SPOConfig:
    return SPOConfig(
        num_envs=num_envs, num_particles=num_particles, num_actions=NUM_ACTIONS,
        state_dim=STATE_DIM, search_depth=depth, resample_period=period,
        temperature=0.5, search_gamma=1.0, search_gae_lambda=1.0)


def test_resample_indices_are_exact(ctx: DeviceContext) raises:
    """With weights and uniforms dictated, the chosen indices are exact.

    Four particles with weights/temperature giving probabilities
    [0.1, 0.2, 0.3, 0.4] -> CDF [0.1, 0.3, 0.6, 1.0]. Since the temperature is 0.5,
    the weights to feed in are 0.5*log(p) plus any constant.
    """
    cfg = make_config(num_envs=1, num_particles=4, depth=1, period=1)
    p_total = cfg.num_search_particles()

    particles = Particles(ctx, cfg)
    scratch = SearchScratch(ctx, cfg)

    probs = List[Scalar[dtype]]()
    probs.append(0.1); probs.append(0.2); probs.append(0.3); probs.append(0.4)

    weights = List[Scalar[dtype]]()
    for n in range(4):
        weights.append(cfg.temperature * log(probs[n]))
    write_into[dtype](particles.resample_td_weights, weights)

    # I mark each particle with its index so as to see which one each slot copied.
    marks = List[Scalar[dtype]]()
    for n in range(4):
        marks.append(Scalar[dtype](n))
    write_into[dtype](particles.state, marks)

    us = List[Scalar[dtype]]()
    want = List[Int]()
    us.append(0.05); want.append(0)   # inside [0, 0.1)
    us.append(0.15); want.append(1)   # inside [0.1, 0.3)
    us.append(0.45); want.append(2)   # inside [0.3, 0.6)
    us.append(0.90); want.append(3)   # inside [0.6, 1.0)

    resample(ctx, particles, scratch, cfg, upload[dtype](ctx, us))
    ctx.synchronize()

    idx = download[idx_dtype](scratch.indices, p_total)
    state = download[dtype](particles.state, p_total)
    for n in range(4):
        assert_eq_int(Int(idx[n]), want[n], String("u=", us[n], " -> indice"))
        # and the actual gather brought that particle's state across
        assert_close(state[n], Scalar[dtype](want[n]), TOL,
                     String("el hueco ", n, " deberia tener el estado de ", want[n]))
    print("PASS indices de resampling exactos y gather correcto")


def test_resample_resets_weights_but_keeps_gae(ctx: DeviceContext) raises:
    """Stoix's critical detail: weights to zero, gae intact and NOT gathered.

    The gae is not reordered even though everything else is. Stoix does it on
    purpose (ff_spo.py:914) because the temperature loss needs the pre-resampling
    advantages.
    """
    cfg = make_config(num_envs=1, num_particles=4, depth=1, period=1)
    p_total = cfg.num_search_particles()

    particles = Particles(ctx, cfg)
    scratch = SearchScratch(ctx, cfg)

    # A huge weight on particle 2: almost every slot is going to copy it.
    weights = List[Scalar[dtype]]()
    weights.append(-100.0); weights.append(-100.0)
    weights.append(0.0); weights.append(-100.0)
    write_into[dtype](particles.resample_td_weights, weights)

    # a different gae in each particle, to notice if it gets reordered
    gae = List[Scalar[dtype]]()
    for n in range(4):
        gae.append(Scalar[dtype](10 + n))
    write_into[dtype](particles.gae, gae)

    us = List[Scalar[dtype]]()
    for _ in range(4):
        us.append(0.5)
    resample(ctx, particles, scratch, cfg, upload[dtype](ctx, us))
    ctx.synchronize()

    got_w = download[dtype](particles.resample_td_weights, p_total)
    got_gae = download[dtype](particles.gae, p_total)
    got_idx = download[idx_dtype](scratch.indices, p_total)

    for n in range(4):
        assert_close(got_w[n], 0.0, TOL,
                     String("el peso ", n, " deberia resetearse a 0"))
        assert_close(got_gae[n], Scalar[dtype](10 + n), TOL,
                     String("la gae ", n, " no deberia moverse"))
        # and the resampling DID indeed reorder (they all copied number 2)
        assert_eq_int(Int(got_idx[n]), 2,
                      String("el hueco ", n, " deberia copiar a la particula 2"))
    print("PASS resampling: pesos a 0, gae preservada sin reordenar")


def test_ess_uniform_vs_concentrated(ctx: DeviceContext) raises:
    """ESS = N with equal weights, and ESS -> 1 when one takes everything."""
    n_particles = 16
    cfg = make_config(num_envs=2, num_particles=n_particles, depth=1, period=99)

    particles = Particles(ctx, cfg)
    scratch = SearchScratch(ctx, cfg)
    output = SPOOutput(ctx, cfg)

    weights = List[Scalar[dtype]]()
    # env 0: all equal -> ESS = 16
    for _ in range(n_particles):
        weights.append(3.0)
    # env 1: one giant -> ESS ~ 1
    weights.append(100.0)
    for _ in range(n_particles - 1):
        weights.append(-100.0)
    write_into[dtype](particles.resample_td_weights, weights)

    compute_ess_entropy(ctx, particles, scratch, output, cfg, 0)
    ctx.synchronize()

    ess = download[dtype](output.ess, cfg.num_envs)
    entropy = download[dtype](output.entropy, cfg.num_envs)

    assert_close(ess[0], Scalar[dtype](n_particles), Scalar[dtype](1e-3),
                 "pesos uniformes -> ESS = N")
    assert_close(entropy[0], log(Scalar[dtype](n_particles)), Scalar[dtype](1e-4),
                 "pesos uniformes -> entropia = log N")
    assert_close(ess[1], 1.0, Scalar[dtype](1e-3),
                 "peso concentrado -> ESS = 1")
    assert_close(entropy[1], 0.0, Scalar[dtype](1e-4),
                 "peso concentrado -> entropia = 0")
    print("PASS ESS: uniforme", ess[0], "/ concentrado", ess[1])


def main() raises:
    with DeviceContext() as ctx:
        test_resample_indices_are_exact(ctx)
        test_resample_resets_weights_but_keeps_gae(ctx)
        test_ess_uniform_vs_concentrated(ctx)
