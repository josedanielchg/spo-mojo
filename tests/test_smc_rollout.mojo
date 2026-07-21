"""Resampling, ESS y la busqueda completa sobre el juguete.

El ultimo test es el que importa de toda la fase: partiendo de una politica que
no sabe nada (prior uniforme) y SIN entrenar nada, la busqueda tiene que
concentrar la masa en la accion buena. Si eso pasa, el E-step funciona.
"""

from std.gpu.host import DeviceContext
from std.math import abs, log

from ops.common import dtype, idx_dtype
from envs.toy_chain import (default_toy_chain, ToyChain,
                            ACTION_BAD, ACTION_GOOD, NUM_ACTIONS, STATE_DIM)
from systems.spo.spo_types import (SPOConfig, Particles, StepOutputs,
                                   SearchScratch, SPOOutput, SearchWorkspace)
from systems.spo.smc_search import resample, compute_ess_entropy, search
from tests.helpers import (upload, zeros, download, write_into, assert_close,
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


def test_ess_drops_and_recovers_after_resampling(ctx: DeviceContext) raises:
    """A lo largo del rollout el ESS baja y se recupera tras cada resampling.

    Con profundidad 8 y periodo 4 hay resampling despues de las profundidades 3
    y 7. Como el ESS se mide ANTES de resamplear, la profundidad 4 (la primera
    despues del reset) tiene que verse mas sana que la 3.
    """
    cfg = make_config(num_envs=4, num_particles=16, depth=8, period=4)
    toy = ToyChain(chain_length=20, horizon=20, value_scale=1.0)

    ws = SearchWorkspace(ctx, cfg)

    root_state = List[Scalar[dtype]]()
    for _ in range(cfg.num_envs):
        root_state.append(0.0)

    search[ToyChain](ctx, ws, cfg, toy, upload[dtype](ctx, root_state),
                     UInt32(2024))
    ctx.synchronize()

    ess = download[dtype](ws.output.ess, cfg.search_depth * cfg.num_envs)

    # Media por profundidad sobre los envs
    means = List[Scalar[dtype]]()
    for d in range(cfg.search_depth):
        total = Scalar[dtype](0)
        for e in range(cfg.num_envs):
            total += ess[d * cfg.num_envs + e]
        means.append(total / Scalar[dtype](cfg.num_envs))

    print("      ESS por profundidad:")
    for d in range(cfg.search_depth):
        print("        depth", d, "->", means[d])

    # Baja dentro del primer tramo...
    if means[3] >= means[0]:
        raise Error("el ESS deberia degradarse entre resamplings: d0=",
                    means[0], " d3=", means[3])
    # ...y se recupera justo despues del resampling de la profundidad 3.
    if means[4] <= means[3]:
        raise Error("el ESS deberia recuperarse tras el resampling: d3=",
                    means[3], " d4=", means[4])
    print("PASS el ESS cae entre resamplings y se recupera despues")


def test_search_improves_a_uniform_prior(ctx: DeviceContext) raises:
    """LA prueba de la fase: la busqueda mejora una politica que no sabe nada.

    El prior del juguete es uniforme (50/50) y no se entrena nada. Solo con
    simular, la politica mejorada q tiene que poner al menos el 80% de la masa
    en la accion buena.

    q se lee de la salida igual que la usara el M-step: el histograma de
    `sampled_actions` ponderado por `sampled_action_weights`.
    """
    cfg = make_config(num_envs=8, num_particles=16, depth=4, period=4)
    toy = default_toy_chain()

    ws = SearchWorkspace(ctx, cfg)

    root_state = List[Scalar[dtype]]()
    for _ in range(cfg.num_envs):
        root_state.append(0.0)      # todos en la casilla de salida

    search[ToyChain](ctx, ws, cfg, toy, upload[dtype](ctx, root_state),
                     UInt32(1234))
    ctx.synchronize()

    p_total = cfg.num_search_particles()
    actions = download[idx_dtype](ws.output.sampled_actions, p_total)
    weights = download[dtype](ws.output.sampled_action_weights, p_total)
    final_action = download[idx_dtype](ws.output.action, cfg.num_envs)

    worst = Scalar[dtype](1.0)
    for e in range(cfg.num_envs):
        q_good = Scalar[dtype](0)
        q_total = Scalar[dtype](0)
        for n in range(cfg.num_particles):
            p = e * cfg.num_particles + n
            q_total += weights[p]
            if Int(actions[p]) == ACTION_GOOD:
                q_good += weights[p]
        # Los pesos son un softmax por env, o sea que ya suman 1.
        assert_close(q_total, 1.0, Scalar[dtype](1e-4),
                     String("los pesos del env ", e, " deberian sumar 1"))
        if q_good < worst:
            worst = q_good

    print("      q(GOOD) minimo sobre", cfg.num_envs, "envs:", worst)
    if worst < 0.8:
        raise Error("la busqueda no mejoro el prior lo suficiente: q(GOOD)=", worst)

    # Y la accion que se ejecuta de verdad tiene que ser valida.
    for e in range(cfg.num_envs):
        a = Int(final_action[e])
        if a != ACTION_BAD and a != ACTION_GOOD:
            raise Error("accion final invalida en el env ", e, ": ", a)

    print("PASS la busqueda mejora el prior uniforme: q(GOOD) >=", worst)


def test_search_is_reproducible(ctx: DeviceContext) raises:
    """Misma semilla, misma busqueda. Sin esto no hay test que valga."""
    cfg = make_config(num_envs=4, num_particles=16, depth=4, period=4)
    toy = default_toy_chain()
    p_total = cfg.num_search_particles()

    root_state = List[Scalar[dtype]]()
    for _ in range(cfg.num_envs):
        root_state.append(0.0)

    first = List[Scalar[dtype]]()
    for run in range(2):
        ws = SearchWorkspace(ctx, cfg)
        search[ToyChain](ctx, ws, cfg, toy, upload[dtype](ctx, root_state),
                         UInt32(555))
        ctx.synchronize()
        w = download[dtype](ws.output.sampled_action_weights, p_total)
        if run == 0:
            first = w^
        else:
            for p in range(p_total):
                assert_close(w[p], first[p], TOL,
                             String("dos busquedas con la misma seed difieren en ", p))
    print("PASS la busqueda es reproducible con la misma seed")


def test_more_particles_is_not_worse(ctx: DeviceContext) raises:
    """Regresion: con N grande la busqueda tiene que seguir mejorando, no empeorar.

    Este test existe por un bug de verdad. Los kernels cuya fila es la dimension
    de particulas (resampling, ESS, softmax del readout) usan UN bloque por env,
    asi que N tiene que caber en el bloque. Con bloques de 32 y N=64 la busqueda
    devolvia q(GOOD)=0.75, PEOR que con N=16 (0.99), y sin avisar de nada.

    Lo caza el debug_assert del kernel, pero solo con -D ASSERT=all; por eso
    ademas hay una comprobacion en host (check_search_config) y este test.
    """
    toy = default_toy_chain()
    q = List[Scalar[dtype]]()

    counts = List[Int]()
    counts.append(4); counts.append(16); counts.append(64)

    for i in range(len(counts)):
        n = counts[i]
        cfg = make_config(num_envs=16, num_particles=n, depth=4, period=4)
        ws = SearchWorkspace(ctx, cfg)

        root_state = List[Scalar[dtype]]()
        for _ in range(cfg.num_envs):
            root_state.append(0.0)

        search[ToyChain](ctx, ws, cfg, toy, upload[dtype](ctx, root_state),
                         UInt32(4321))
        ctx.synchronize()

        p_total = cfg.num_search_particles()
        actions = download[idx_dtype](ws.output.sampled_actions, p_total)
        weights = download[dtype](ws.output.sampled_action_weights, p_total)
        total = Scalar[dtype](0)
        for p in range(p_total):
            if Int(actions[p]) == ACTION_GOOD:
                total += weights[p]
        q.append(total / Scalar[dtype](cfg.num_envs))
        print("      N =", n, "-> q(GOOD) =", q[i])

    if q[1] < q[0] or q[2] < q[1]:
        raise Error("mas particulas deberia mejorar (o al menos no empeorar): ",
                    q[0], " ", q[1], " ", q[2])
    print("PASS mas particulas no empeora la busqueda")


def main() raises:
    with DeviceContext() as ctx:
        test_resample_indices_are_exact(ctx)
        test_resample_resets_weights_but_keeps_gae(ctx)
        test_ess_uniform_vs_concentrated(ctx)
        test_ess_drops_and_recovers_after_resampling(ctx)
        test_search_improves_a_uniform_prior(ctx)
        test_more_particles_is_not_worse(ctx)
        test_search_is_reproducible(ctx)
