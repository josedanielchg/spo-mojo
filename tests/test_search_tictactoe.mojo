"""La busqueda SMC completa sobre Tic-Tac-Toe: pruebas de COMPORTAMIENTO.

Es el equivalente de test_search.mojo pero con el modelo de TTT: no se dicta
nada, se corre `search[TicTacToe]()` de punta a punta y se mira lo que sale. La
primera integracion de todo lo de la fase A con el nucleo SMC.

Lo que se comprueba, de mas basico a mas exigente:
  1. la busqueda corre y no produce NaN,
  2. la accion elegida es SIEMPRE legal (el prior enmascarado hace su trabajo),
  3. misma seed -> misma busqueda, bit a bit,
  4. el ESS se mantiene en rango sano [1, N].

Lo que NO se comprueba aqui: si la busqueda JUEGA bien. Eso necesita partidas de
verdad contra un rival, y va en la fase A' (el bucle real y la demo).
"""

from std.gpu.host import DeviceContext
from std.math import isnan

from ops.common import dtype, idx_dtype
from envs.tictactoe import (TicTacToe, default_tictactoe, NUM_CELLS,
                            NUM_ACTIONS, STATE_DIM, CELL_EMPTY)
from systems.spo.particles import SearchWorkspace
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from tests.helpers import upload, download, assert_close

comptime TOL = Scalar[dtype](1e-5)


def make_config(num_envs: Int, num_particles: Int, depth: Int,
                period: Int) -> SPOConfig:
    """Config de busqueda para TTT: 9 acciones, 9 floats de estado."""
    return SPOConfig(
        num_envs=num_envs, num_particles=num_particles, num_actions=NUM_ACTIONS,
        state_dim=STATE_DIM, search_depth=depth, resample_period=period,
        temperature=0.5, search_gamma=1.0, search_gae_lambda=1.0)


def board9(c0: Int, c1: Int, c2: Int, c3: Int, c4: Int,
           c5: Int, c6: Int, c7: Int, c8: Int) -> List[Scalar[dtype]]:
    """Un tablero legible: 9 codigos de casilla (1=X agente, -1=O rival, 0=vacia)."""
    out = List[Scalar[dtype]]()
    out.append(Scalar[dtype](c0)); out.append(Scalar[dtype](c1))
    out.append(Scalar[dtype](c2)); out.append(Scalar[dtype](c3))
    out.append(Scalar[dtype](c4)); out.append(Scalar[dtype](c5))
    out.append(Scalar[dtype](c6)); out.append(Scalar[dtype](c7))
    out.append(Scalar[dtype](c8))
    return out^


def mixed_roots(num_envs: Int) -> List[Scalar[dtype]]:
    """Estados raiz variados: vacio, media partida y casi lleno, en rotacion.

    Interesa mezclar posiciones porque cada una ejercita un camino distinto: desde
    el vacio la busqueda tiene 9 acciones y profundidad de sobra; desde casi lleno
    casi todas las particulas llegan a terminal en uno o dos pasos.
    """
    roots = List[Scalar[dtype]]()
    for e in range(num_envs):
        which = e % 3
        b = board9(0,0,0, 0,0,0, 0,0,0)              # vacio: le toca a X
        if which == 1:
            b = board9(1,0,-1, 0,1,0, -1,0,0)         # media partida
        elif which == 2:
            b = board9(1,-1,1, -1,1,0, -1,1,0)        # casi lleno: solo 5 y 8 libres
        for c in range(NUM_CELLS):
            roots.append(b[c])
    return roots^


def test_search_runs_without_nan(ctx: DeviceContext) raises:
    """La busqueda corre de punta a punta y ninguna salida trae NaN.

    Es el smoke de verdad: con el prior enmascarado hay NEG_INF en los logits de
    las casillas ocupadas, y ese es justo el sitio por donde se colaria un NaN
    (exp(-inf - -inf) en el softmax) si la convencion de enmascarado estuviera mal.
    """
    cfg = make_config(num_envs=6, num_particles=16, depth=6, period=3)
    model = default_tictactoe()
    ws = SearchWorkspace(ctx, cfg)
    p_total = cfg.num_search_particles()

    search[TicTacToe](ctx, ws, cfg, model,
                      upload[dtype](ctx, mixed_roots(cfg.num_envs)), UInt32(7))
    ctx.synchronize()

    weights = download[dtype](ws.output.sampled_action_weights, p_total)
    advantages = download[dtype](ws.output.sampled_advantages, p_total)
    values = download[dtype](ws.output.value, cfg.num_envs)
    ess = download[dtype](ws.output.ess, cfg.search_depth * cfg.num_envs)

    for p in range(p_total):
        if isnan(weights[p]):
            raise Error("peso NaN en la particula ", p)
        if isnan(advantages[p]):
            raise Error("ventaja NaN en la particula ", p)
    for e in range(cfg.num_envs):
        if isnan(values[e]):
            raise Error("valor NaN en el env ", e)
    for i in range(len(ess)):
        if isnan(ess[i]):
            raise Error("ESS NaN en el indice ", i)

    # Los pesos son un softmax por env: tienen que sumar 1.
    for e in range(cfg.num_envs):
        total = Scalar[dtype](0)
        for n in range(cfg.num_particles):
            total += weights[e * cfg.num_particles + n]
        assert_close(total, 1.0, Scalar[dtype](1e-4),
                     String("los pesos del env ", e, " deberian sumar 1"))
    print("PASS la busqueda corre sobre TTT sin NaN y los pesos suman 1")


def test_search_only_picks_legal_actions(ctx: DeviceContext) raises:
    """LA prueba de la fase A: la accion elegida nunca cae sobre una casilla ocupada.

    Es lo que justifica el prior enmascarado. Se comprueba tanto la accion final
    por env como TODAS las acciones raiz muestreadas: si una sola particula hubiera
    podido elegir una casilla ocupada, el enmascarado no estaria funcionando.
    """
    cfg = make_config(num_envs=6, num_particles=16, depth=6, period=3)
    model = default_tictactoe()
    ws = SearchWorkspace(ctx, cfg)
    p_total = cfg.num_search_particles()

    roots = mixed_roots(cfg.num_envs)
    search[TicTacToe](ctx, ws, cfg, model, upload[dtype](ctx, roots), UInt32(99))
    ctx.synchronize()

    final_action = download[idx_dtype](ws.output.action, cfg.num_envs)
    sampled = download[idx_dtype](ws.output.sampled_actions, p_total)

    for e in range(cfg.num_envs):
        a = Int(final_action[e])
        if a < 0 or a >= NUM_ACTIONS:
            raise Error("accion fuera de rango en el env ", e, ": ", a)
        if roots[e * NUM_CELLS + a] != CELL_EMPTY:
            raise Error("el env ", e, " eligio la casilla OCUPADA ", a)

    for e in range(cfg.num_envs):
        for n in range(cfg.num_particles):
            a = Int(sampled[e * cfg.num_particles + n])
            if a < 0 or a >= NUM_ACTIONS:
                raise Error("accion raiz fuera de rango en el env ", e, ": ", a)
            if roots[e * NUM_CELLS + a] != CELL_EMPTY:
                raise Error("la particula ", n, " del env ", e,
                            " muestreo la casilla OCUPADA ", a)
    print("PASS todas las acciones (finales y muestreadas) son legales")


def test_search_is_reproducible(ctx: DeviceContext) raises:
    """Misma semilla, misma busqueda, aunque el modelo sea estocastico.

    Importa mas que en el juguete: el step de TTT sortea la jugada del rival, asi
    que si el stream RNG del paso no fuera determinista el resultado cambiaria
    entre corridas.
    """
    cfg = make_config(num_envs=4, num_particles=16, depth=6, period=3)
    model = default_tictactoe()
    p_total = cfg.num_search_particles()
    roots = mixed_roots(cfg.num_envs)

    first = List[Scalar[dtype]]()
    first_actions = List[Scalar[idx_dtype]]()
    for run in range(2):
        ws = SearchWorkspace(ctx, cfg)
        search[TicTacToe](ctx, ws, cfg, model, upload[dtype](ctx, roots),
                          UInt32(31337))
        ctx.synchronize()
        w = download[dtype](ws.output.sampled_action_weights, p_total)
        a = download[idx_dtype](ws.output.action, cfg.num_envs)
        if run == 0:
            first = w^
            first_actions = a^
        else:
            for p in range(p_total):
                assert_close(w[p], first[p], TOL,
                             String("dos busquedas con la misma seed difieren en ", p))
            for e in range(cfg.num_envs):
                if Int(a[e]) != Int(first_actions[e]):
                    raise Error("la accion del env ", e, " cambio entre corridas")
    print("PASS la busqueda sobre TTT es reproducible (rival aleatorio incluido)")


def test_ess_stays_in_range(ctx: DeviceContext) raises:
    """El ESS se mantiene en [1, N] en todas las profundidades y envs.

    No exijo la sierra del juguete: en TTT muchas particulas llegan a terminal y
    congelan su peso, asi que la forma de la curva depende mucho de la posicion
    inicial. Lo que si tiene que cumplirse siempre es el rango: el ESS de N
    particulas no puede ser menor que 1 ni mayor que N.
    """
    cfg = make_config(num_envs=6, num_particles=16, depth=6, period=3)
    model = default_tictactoe()
    ws = SearchWorkspace(ctx, cfg)

    search[TicTacToe](ctx, ws, cfg, model,
                      upload[dtype](ctx, mixed_roots(cfg.num_envs)), UInt32(2718))
    ctx.synchronize()

    ess = download[dtype](ws.output.ess, cfg.search_depth * cfg.num_envs)
    lo = Scalar[dtype](1) - Scalar[dtype](1e-4)
    hi = Scalar[dtype](cfg.num_particles) + Scalar[dtype](1e-4)

    print("      ESS medio por profundidad:")
    for d in range(cfg.search_depth):
        total = Scalar[dtype](0)
        for e in range(cfg.num_envs):
            v = ess[d * cfg.num_envs + e]
            if v < lo or v > hi:
                raise Error("ESS fuera de rango en depth ", d, " env ", e, ": ", v)
            total += v
        print("        depth", d, "->", total / Scalar[dtype](cfg.num_envs))
    print("PASS el ESS se queda en [1,", cfg.num_particles, "] en todas las profundidades")


def main() raises:
    with DeviceContext() as ctx:
        test_search_runs_without_nan(ctx)
        test_search_only_picks_legal_actions(ctx)
        test_search_is_reproducible(ctx)
        test_ess_stays_in_range(ctx)
