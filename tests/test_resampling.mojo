"""Las piezas del resampling y del ESS, con entradas dictadas a mano.

Nada de aqui corre una busqueda completa: se escriben los pesos que interesan
directamente en las particulas y se comprueba el resultado EXACTO. Cuando algo
falla, el fallo apunta a un kernel concreto en vez de a "la busqueda".

El comportamiento de la busqueda entera se prueba aparte, en test_search.mojo.
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
    """Con pesos y uniformes dictados, los indices elegidos son exactos.

    Cuatro particulas con pesos/temperatura que dan probabilidades
    [0.1, 0.2, 0.3, 0.4] -> CDF [0.1, 0.3, 0.6, 1.0]. Como la temperatura es 0.5,
    los pesos que hay que meter son 0.5*log(p) mas cualquier constante.
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

    # Marco cada particula con su indice para poder ver a quien copio cada hueco.
    marks = List[Scalar[dtype]]()
    for n in range(4):
        marks.append(Scalar[dtype](n))
    write_into[dtype](particles.state, marks)

    us = List[Scalar[dtype]]()
    want = List[Int]()
    us.append(0.05); want.append(0)   # dentro de [0, 0.1)
    us.append(0.15); want.append(1)   # dentro de [0.1, 0.3)
    us.append(0.45); want.append(2)   # dentro de [0.3, 0.6)
    us.append(0.90); want.append(3)   # dentro de [0.6, 1.0)

    resample(ctx, particles, scratch, cfg, upload[dtype](ctx, us))
    ctx.synchronize()

    idx = download[idx_dtype](scratch.indices, p_total)
    state = download[dtype](particles.state, p_total)
    for n in range(4):
        assert_eq_int(Int(idx[n]), want[n], String("u=", us[n], " -> indice"))
        # y el gather de verdad trajo el estado de esa particula
        assert_close(state[n], Scalar[dtype](want[n]), TOL,
                     String("el hueco ", n, " deberia tener el estado de ", want[n]))
    print("PASS indices de resampling exactos y gather correcto")


def test_resample_resets_weights_but_keeps_gae(ctx: DeviceContext) raises:
    """El detalle critico de Stoix: pesos a cero, gae intacta y SIN gatherear.

    La gae no se reordena aunque todo lo demas si. Stoix lo hace a proposito
    (ff_spo.py:914) porque el loss de la temperatura necesita las ventajas de
    antes de resamplear.
    """
    cfg = make_config(num_envs=1, num_particles=4, depth=1, period=1)
    p_total = cfg.num_search_particles()

    particles = Particles(ctx, cfg)
    scratch = SearchScratch(ctx, cfg)

    # Peso enorme en la particula 2: casi todos los huecos la van a copiar.
    weights = List[Scalar[dtype]]()
    weights.append(-100.0); weights.append(-100.0)
    weights.append(0.0); weights.append(-100.0)
    write_into[dtype](particles.resample_td_weights, weights)

    # gae distinta en cada particula, para notar si se reordena
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
        # y efectivamente el resampling SI reordeno (todos copiaron a la 2)
        assert_eq_int(Int(got_idx[n]), 2,
                      String("el hueco ", n, " deberia copiar a la particula 2"))
    print("PASS resampling: pesos a 0, gae preservada sin reordenar")


def test_ess_uniform_vs_concentrated(ctx: DeviceContext) raises:
    """ESS = N con pesos iguales, y ESS -> 1 cuando uno se lo lleva todo."""
    n_particles = 16
    cfg = make_config(num_envs=2, num_particles=n_particles, depth=1, period=99)

    particles = Particles(ctx, cfg)
    scratch = SearchScratch(ctx, cfg)
    output = SPOOutput(ctx, cfg)

    weights = List[Scalar[dtype]]()
    # env 0: todos iguales -> ESS = 16
    for _ in range(n_particles):
        weights.append(3.0)
    # env 1: uno gigante -> ESS ~ 1
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
