"""El entorno REAL de Tic-Tac-Toe: reset, un turno de verdad y auto-reset.

Aqui no hay busqueda ni particulas: son partidas de verdad, un tablero por env.
La diferencia con el step de la busqueda es lo que sale (`done` en vez de discount
y bootstrap) y que las partidas terminadas se reinician solas.

Las reglas no se vuelven a probar: kernel de busqueda y kernel de entorno comparten
`ttt_advance`, y las reglas ya tienen sus tests en test_tictactoe.mojo. Lo que se
prueba aqui es lo que solo existe en el entorno real.
"""

from std.gpu.host import DeviceContext

from std.math import abs

from ops.common import dtype, idx_dtype
from envs.tictactoe import (ttt_reset_kernel, ttt_env_step_kernel,
                            ttt_auto_reset_kernel, ttt_random_policy_kernel,
                            NUM_CELLS, NUM_ACTIONS, STATE_DIM,
                            CELL_EMPTY, CELL_AGENT, CELL_RIVAL, TPB_TTT)
from envs.tictactoe_runner import play_random_games, MatchStats
from tests.helpers import (upload, zeros, download, filled, assert_close,
                           assert_eq_int)

comptime TOL = Scalar[dtype](1e-6)


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


def flatten(boards: List[List[Scalar[dtype]]]) -> List[Scalar[dtype]]:
    """Varios tableros seguidos, como los espera el device."""
    out = List[Scalar[dtype]]()
    for i in range(len(boards)):
        for c in range(NUM_CELLS):
            out.append(boards[i][c])
    return out^


def test_reset_clears_the_board(ctx: DeviceContext) raises:
    """Reset deja el tablero vacio, venga de donde venga."""
    boards = List[List[Scalar[dtype]]]()
    boards.append(board9(1,-1,1, -1,1,-1, 1,-1,1))   # lleno
    boards.append(board9(1,0,-1, 0,1,0, -1,0,0))     # media partida
    n = len(boards)

    state = upload[dtype](ctx, flatten(boards))
    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), n, grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()

    got = download[dtype](state, n * NUM_CELLS)
    for i in range(n * NUM_CELLS):
        assert_close(got[i], CELL_EMPTY, TOL, String("tras el reset, celda ", i))
    print("PASS reset deja el tablero vacio")


def test_env_step_reports_done(ctx: DeviceContext) raises:
    """Un turno real: recompensa del agente y `done` en los cuatro finales.

    Mismos escenarios que el step de la busqueda, pero mirando `done`: aqui lo que
    importa es si la partida acabo, para contarla y reiniciarla.
    """
    boards = List[List[Scalar[dtype]]]()
    acts = List[Scalar[idx_dtype]]()
    us = List[Scalar[dtype]]()
    exp_reward = List[Scalar[dtype]]()
    exp_done = List[Int]()
    names = List[String]()

    # Gana el agente completando la fila 0.
    boards.append(board9(1,1,0, -1,-1,0, 0,0,0)); acts.append(Scalar[idx_dtype](2))
    us.append(Scalar[dtype](0.5)); exp_reward.append(Scalar[dtype](1))
    exp_done.append(1); names.append("gana agente")
    # Empate: la jugada del agente llena el tablero sin linea.
    boards.append(board9(1,-1,1, 1,-1,-1, -1,1,0)); acts.append(Scalar[idx_dtype](8))
    us.append(Scalar[dtype](0.5)); exp_reward.append(Scalar[dtype](0.5))
    exp_done.append(1); names.append("empate al llenar")
    # Gana el rival: solo le queda la casilla 6 y con ella hace la columna 0.
    boards.append(board9(-1,1,0, -1,1,-1, 0,-1,1)); acts.append(Scalar[idx_dtype](2))
    us.append(Scalar[dtype](0.5)); exp_reward.append(Scalar[dtype](0))
    exp_done.append(1); names.append("gana rival")
    # La partida sigue: quedan huecos y nadie ha ganado.
    boards.append(board9(1,-1,1, -1,1,0, -1,0,0)); acts.append(Scalar[idx_dtype](5))
    us.append(Scalar[dtype](0.1)); exp_reward.append(Scalar[dtype](0))
    exp_done.append(0); names.append("sigue")

    n = len(boards)
    state = upload[dtype](ctx, flatten(boards))
    action = upload[idx_dtype](ctx, acts)
    u = upload[dtype](ctx, us)
    reward = zeros[dtype](ctx, n)
    done = filled[idx_dtype](ctx, n, Scalar[idx_dtype](-1))   # -1 = sin escribir

    ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
        state.unsafe_ptr(), action.unsafe_ptr(), u.unsafe_ptr(),
        reward.unsafe_ptr(), done.unsafe_ptr(), n,
        grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()

    got_r = download[dtype](reward, n)
    got_d = download[idx_dtype](done, n)
    for i in range(n):
        assert_close(got_r[i], exp_reward[i], TOL, String("reward '", names[i], "'"))
        assert_eq_int(Int(got_d[i]), exp_done[i], String("done '", names[i], "'"))
    print("PASS un turno real: recompensa y done en los cuatro finales")


def test_auto_reset_only_finished_games(ctx: DeviceContext) raises:
    """Solo los envs con done=1 empiezan de nuevo; los demas siguen su partida."""
    boards = List[List[Scalar[dtype]]]()
    boards.append(board9(1,1,1, -1,-1,0, 0,0,0))   # [0] termino -> se limpia
    boards.append(board9(1,0,-1, 0,1,0, -1,0,0))   # [1] sigue    -> intacto
    n = len(boards)

    state = upload[dtype](ctx, flatten(boards))
    dones = List[Scalar[idx_dtype]]()
    dones.append(Scalar[idx_dtype](1)); dones.append(Scalar[idx_dtype](0))
    done = upload[idx_dtype](ctx, dones)

    ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
        state.unsafe_ptr(), done.unsafe_ptr(), n, grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()

    got = download[dtype](state, n * NUM_CELLS)
    for c in range(NUM_CELLS):
        assert_close(got[c], CELL_EMPTY, TOL,
                     String("el env que termino deberia estar vacio, celda ", c))
        assert_close(got[NUM_CELLS + c], boards[1][c], TOL,
                     String("el env que sigue no deberia tocarse, celda ", c))
    print("PASS auto-reset: reinicia solo las partidas terminadas")


def test_random_policy_only_legal(ctx: DeviceContext) raises:
    """La politica aleatoria (la linea base) nunca juega sobre una casilla ocupada.

    Barre uniformes de 0 a ~1 sobre el mismo tablero para recorrer todas las
    casillas libres que puede elegir, y comprueba ademas que llega a elegir mas de
    una (si devolviera siempre la misma, seria legal pero no aleatoria).
    """
    b = board9(1,0,-1, 0,1,0, -1,0,0)     # libres: 1, 3, 5, 7, 8
    n = 10
    boards = List[List[Scalar[dtype]]]()
    us = List[Scalar[dtype]]()
    for i in range(n):
        boards.append(board9(1,0,-1, 0,1,0, -1,0,0))
        us.append(Scalar[dtype](i) / Scalar[dtype](n))

    state = upload[dtype](ctx, flatten(boards))
    u = upload[dtype](ctx, us)
    action = filled[idx_dtype](ctx, n, Scalar[idx_dtype](-1))

    ctx.enqueue_function[ttt_random_policy_kernel, ttt_random_policy_kernel](
        action.unsafe_ptr(), state.unsafe_ptr(), u.unsafe_ptr(), n,
        grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()

    got = download[idx_dtype](action, n)
    distinct = List[Int]()
    for i in range(n):
        a = Int(got[i])
        if a < 0 or a >= NUM_ACTIONS:
            raise Error("accion fuera de rango con u=", us[i], ": ", a)
        if b[a] != CELL_EMPTY:
            raise Error("la politica aleatoria eligio la casilla OCUPADA ", a)
        seen = False
        for j in range(len(distinct)):
            if distinct[j] == a:
                seen = True
        if not seen:
            distinct.append(a)

    if len(distinct) < 2:
        raise Error("la politica aleatoria devolvio siempre la misma casilla")
    print("PASS la politica aleatoria solo juega casillas libres (", len(distinct),
          "distintas )")


def test_random_baseline_matches_exact_odds(ctx: DeviceContext) raises:
    """LA prueba del bucle: la linea base reproduce las probabilidades EXACTAS.

    Con las dos partes jugando uniformemente al azar, las probabilidades de TTT se
    pueden calcular exactamente por recursion sobre todos los estados (no es una
    estimacion ni un numero de la literatura):

        gana X (agente)  0.5849      empate  0.1270      gana O  0.2881
        puntuacion media del agente = 0.5849 + 0.5*0.1270 = 0.6484

    Que salgan esas tres cifras valida de golpe TODO el bucle: el turno, el rival
    aleatorio, la deteccion de final, la clasificacion del resultado y el
    auto-reset. Un fallo en cualquiera de esas piezas movería las proporciones.

    La tolerancia es de 4 puntos porcentuales: con ~3000 partidas el error estandar
    ronda el 0.9%, asi que 4 puntos son mas de 4 sigma -- holgado para no fallar por
    ruido con otra semilla, y aun asi estrecho para cazar cualquier bug de verdad.
    """
    stats = play_random_games(ctx, 64, 200, UInt32(12345))
    n = stats.games()
    if n < 2000:
        raise Error("se esperaban miles de partidas en 200 turnos, salieron ", n)

    draw_rate = Scalar[dtype](stats.draws) / Scalar[dtype](n)
    loss_rate = Scalar[dtype](stats.losses) / Scalar[dtype](n)
    tol = Scalar[dtype](0.04)

    print("      partidas:", n, " gana X:", stats.win_rate(),
          " empate:", draw_rate, " gana O:", loss_rate,
          " score:", stats.score())

    if abs(stats.win_rate() - Scalar[dtype](0.5849)) > tol:
        raise Error("la tasa de victoria se aleja del valor exacto 0.5849: ",
                    stats.win_rate())
    if abs(draw_rate - Scalar[dtype](0.1270)) > tol:
        raise Error("la tasa de empate se aleja del valor exacto 0.1270: ", draw_rate)
    if abs(loss_rate - Scalar[dtype](0.2881)) > tol:
        raise Error("la tasa de derrota se aleja del valor exacto 0.2881: ", loss_rate)
    if abs(stats.score() - Scalar[dtype](0.6484)) > tol:
        raise Error("la puntuacion se aleja del valor exacto 0.6484: ", stats.score())

    # Y las tres categorias tienen que sumar exactamente las partidas contadas.
    assert_eq_int(stats.wins + stats.draws + stats.losses, n,
                  "victorias + empates + derrotas deberia ser el total")
    print("PASS la linea base reproduce las probabilidades exactas de TTT al azar")


def test_baseline_is_reproducible(ctx: DeviceContext) raises:
    """Misma semilla, mismo marcador. Sin esto no se puede comparar nada."""
    a = play_random_games(ctx, 32, 60, UInt32(777))
    b = play_random_games(ctx, 32, 60, UInt32(777))
    assert_eq_int(a.wins, b.wins, "las victorias deberian repetirse con la misma seed")
    assert_eq_int(a.draws, b.draws, "los empates deberian repetirse con la misma seed")
    assert_eq_int(a.losses, b.losses, "las derrotas deberian repetirse con la misma seed")

    # Y con otra semilla el marcador tiene que cambiar (si no, la seed no se usa).
    c = play_random_games(ctx, 32, 60, UInt32(4242))
    if c.wins == a.wins and c.draws == a.draws and c.losses == a.losses:
        raise Error("dos semillas distintas dieron el mismo marcador exacto")
    print("PASS la linea base es reproducible y depende de la semilla")


def main() raises:
    with DeviceContext() as ctx:
        test_reset_clears_the_board(ctx)
        test_env_step_reports_done(ctx)
        test_auto_reset_only_finished_games(ctx)
        test_random_policy_only_legal(ctx)
        test_random_baseline_matches_exact_odds(ctx)
        test_baseline_is_reproducible(ctx)
