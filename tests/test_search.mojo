"""La busqueda SMC completa sobre el juguete: pruebas de COMPORTAMIENTO.

Aqui no se dicta nada, se corre `search()` de punta a punta y se mira lo que sale.
El tercero es el que importa de toda la fase 3: partiendo de una politica que no
sabe nada (prior uniforme) y SIN entrenar nada, la busqueda tiene que concentrar
la masa en la accion buena. Si eso pasa, el E-step funciona.

Las piezas sueltas (resampling, ESS) se prueban en test_resampling.mojo.
"""

from std.gpu.host import DeviceContext

from ops.common import dtype, idx_dtype
from envs.toy_chain import (default_toy_chain, ToyChain,
                            ACTION_BAD, ACTION_GOOD, NUM_ACTIONS, STATE_DIM)
from systems.spo.particles import SearchWorkspace
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from tests.helpers import upload, download, assert_close

comptime TOL = Scalar[dtype](1e-5)


def make_config(num_envs: Int, num_particles: Int, depth: Int,
                period: Int) -> SPOConfig:
    return SPOConfig(
        num_envs=num_envs, num_particles=num_particles, num_actions=NUM_ACTIONS,
        state_dim=STATE_DIM, search_depth=depth, resample_period=period,
        temperature=0.5, search_gamma=1.0, search_gae_lambda=1.0)


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
        test_ess_drops_and_recovers_after_resampling(ctx)
        test_search_improves_a_uniform_prior(ctx)
        test_more_particles_is_not_worse(ctx)
        test_search_is_reproducible(ctx)
