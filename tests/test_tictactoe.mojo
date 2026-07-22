"""Layout del tablero de TTT: 9 floats por particula, sin solaparse.

A1a solo fija la convencion de almacenamiento. La prueba sube tableros DISTINTOS
para varias particulas y comprueba que el accesor lee la casilla correcta de la
particula correcta -- que el paso STATE_DIM es el bueno y una particula no ve las
casillas de su vecina (el mismo tipo de comprobacion que el broadcast de root_fn).
"""

from std.gpu.host import DeviceContext

from ops.common import dtype, idx_dtype
from envs.tictactoe import (ttt_read_cells_kernel, ttt_has_won_kernel,
                            ttt_legal_mask_kernel, ttt_apply_kernel,
                            NUM_CELLS, NUM_ACTIONS,
                            CELL_EMPTY, CELL_AGENT, CELL_RIVAL, TPB_TTT)
from tests.helpers import upload, zeros, download, assert_close

comptime TOL = Scalar[dtype](1e-6)


def board9(c0: Int, c1: Int, c2: Int, c3: Int, c4: Int,
           c5: Int, c6: Int, c7: Int, c8: Int) -> List[Scalar[dtype]]:
    """Un tablero legible: 9 codigos de casilla (1=X agente, -1=O rival, 0=vacia)
    en el orden 0..8. Todas las pruebas de TTT construyen tableros asi."""
    out = List[Scalar[dtype]]()
    out.append(Scalar[dtype](c0))
    out.append(Scalar[dtype](c1))
    out.append(Scalar[dtype](c2))
    out.append(Scalar[dtype](c3))
    out.append(Scalar[dtype](c4))
    out.append(Scalar[dtype](c5))
    out.append(Scalar[dtype](c6))
    out.append(Scalar[dtype](c7))
    out.append(Scalar[dtype](c8))
    return out^


def test_cell_codes(ctx: DeviceContext) raises:
    """Los codigos de casilla son simetricos y distinguibles.

    X (+1) y O (-1) suman 0 (la casilla vacia): la simetria respecto al 0 es lo
    que le viene bien a la red de la fase M. Y su diferencia es 2, o sea que no
    se confunden entre si.
    """
    assert_close(CELL_AGENT + CELL_RIVAL, CELL_EMPTY, TOL,
                 "X y O deberian ser simetricos respecto a la casilla vacia")
    assert_close(CELL_AGENT - CELL_RIVAL, Scalar[dtype](2), TOL,
                 "X y O deberian ser distinguibles")
    print("PASS codigos de casilla simetricos (X+O=vacia) y distinguibles")


def test_layout_roundtrip(ctx: DeviceContext) raises:
    """Tres tableros distintos, uno por particula, leidos por el accesor.

    Si el paso fuera distinto de STATE_DIM, o el accesor mezclara particulas,
    alguna casilla saldria con el valor de la vecina y el test lo cazaria.
    """
    b0 = board9( 1, 0,-1,   0, 1, 0,  -1, 0, 0)   # X . O / . X . / O . .
    b1 = board9( 0, 0, 0,   1, 1, 1,   0, 0, 0)   # fila del medio de X
    b2 = board9(-1,-1,-1,   0, 0, 0,   1, 1, 0)   # fila de O arriba

    boards = List[Scalar[dtype]]()
    for i in range(NUM_CELLS): boards.append(b0[i])
    for i in range(NUM_CELLS): boards.append(b1[i])
    for i in range(NUM_CELLS): boards.append(b2[i])

    n = 3
    state = upload[dtype](ctx, boards)
    cells_out = zeros[dtype](ctx, n * NUM_CELLS)
    ctx.enqueue_function[ttt_read_cells_kernel, ttt_read_cells_kernel](
        cells_out.unsafe_ptr(), state.unsafe_ptr(), n,
        grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()

    got = download[dtype](cells_out, n * NUM_CELLS)
    for p in range(n):
        for c in range(NUM_CELLS):
            assert_close(got[p * NUM_CELLS + c], boards[p * NUM_CELLS + c], TOL,
                         String("particula ", p, " casilla ", c))
    print("PASS layout: 9 casillas por particula, sin solaparse")


def run_has_won(ctx: DeviceContext, boards: List[Scalar[dtype]], n: Int,
                player: Scalar[dtype]) raises -> List[Scalar[dtype]]:
    """Corre ttt_has_won_kernel para `player` sobre n tableros y baja las flags."""
    state = upload[dtype](ctx, boards)
    won = zeros[dtype](ctx, n)
    ctx.enqueue_function[ttt_has_won_kernel, ttt_has_won_kernel](
        won.unsafe_ptr(), state.unsafe_ptr(), n, player,
        grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()
    return download[dtype](won, n)


def test_wins_on_every_line(ctx: DeviceContext) raises:
    """Las 8 lineas: un tablero por linea con X completandola, todas ganan."""
    wins = List[List[Scalar[dtype]]]()
    names = List[String]()
    wins.append(board9(1,1,1, 0,0,0, 0,0,0)); names.append("fila 0")
    wins.append(board9(0,0,0, 1,1,1, 0,0,0)); names.append("fila 1")
    wins.append(board9(0,0,0, 0,0,0, 1,1,1)); names.append("fila 2")
    wins.append(board9(1,0,0, 1,0,0, 1,0,0)); names.append("columna 0")
    wins.append(board9(0,1,0, 0,1,0, 0,1,0)); names.append("columna 1")
    wins.append(board9(0,0,1, 0,0,1, 0,0,1)); names.append("columna 2")
    wins.append(board9(1,0,0, 0,1,0, 0,0,1)); names.append("diagonal principal")
    wins.append(board9(0,0,1, 0,1,0, 1,0,0)); names.append("diagonal inversa")

    n = len(wins)
    batch = List[Scalar[dtype]]()
    for i in range(n):
        for c in range(NUM_CELLS):
            batch.append(wins[i][c])

    got = run_has_won(ctx, batch, n, CELL_AGENT)
    for i in range(n):
        assert_close(got[i], Scalar[dtype](1), TOL,
                     String("deberia ganar por la ", names[i]))
    print("PASS victoria en las 8 lineas (filas, columnas, diagonales)")


def test_no_false_win(ctx: DeviceContext) raises:
    """Tableros sin linea completa no cuentan como victoria."""
    boards = List[List[Scalar[dtype]]]()
    names = List[String]()
    boards.append(board9(0,0,0, 0,0,0, 0,0,0)); names.append("vacio")
    boards.append(board9(1,0,1, 0,1,0, 0,0,0)); names.append("X disperso sin linea")
    boards.append(board9(1,1,0, 0,0,0, 0,0,0)); names.append("dos de tres (falta una)")

    n = len(boards)
    batch = List[Scalar[dtype]]()
    for i in range(n):
        for c in range(NUM_CELLS):
            batch.append(boards[i][c])

    got = run_has_won(ctx, batch, n, CELL_AGENT)
    for i in range(n):
        assert_close(got[i], Scalar[dtype](0), TOL,
                     String("no deberia ser victoria: ", names[i]))
    print("PASS sin falsos positivos (vacio, disperso, dos de tres)")


def test_players_dont_cross(ctx: DeviceContext) raises:
    """La victoria de un jugador no cuenta para el otro.

    Tablero 0: O completa la fila 0 (X tiene fichas sueltas sin linea).
    Tablero 1: X completa la diagonal (O tiene fichas sueltas sin linea).
    """
    b0 = board9(-1,-1,-1,  1,1,0,  0,0,1)   # gana O (fila 0)
    b1 = board9( 1,0,-1,   0,1,-1,  0,0,1)   # gana X (diagonal 0,4,8)

    batch = List[Scalar[dtype]]()
    for c in range(NUM_CELLS): batch.append(b0[c])
    for c in range(NUM_CELLS): batch.append(b1[c])

    agent = run_has_won(ctx, batch, 2, CELL_AGENT)   # espera [0, 1]
    rival = run_has_won(ctx, batch, 2, CELL_RIVAL)   # espera [1, 0]

    assert_close(agent[0], Scalar[dtype](0), TOL, "la fila de O no es victoria de X")
    assert_close(agent[1], Scalar[dtype](1), TOL, "la diagonal de X si es victoria de X")
    assert_close(rival[0], Scalar[dtype](1), TOL, "la fila de O si es victoria de O")
    assert_close(rival[1], Scalar[dtype](0), TOL, "la diagonal de X no es victoria de O")
    print("PASS las fichas de X y O no se mezclan en el chequeo")


def run_legal_mask(ctx: DeviceContext, boards: List[Scalar[dtype]],
                   n: Int) raises -> List[Scalar[dtype]]:
    """Corre ttt_legal_mask_kernel sobre n tableros y baja la mascara [n, 9]."""
    state = upload[dtype](ctx, boards)
    mask = zeros[dtype](ctx, n * NUM_ACTIONS)
    ctx.enqueue_function[ttt_legal_mask_kernel, ttt_legal_mask_kernel](
        mask.unsafe_ptr(), state.unsafe_ptr(), n,
        grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()
    return download[dtype](mask, n * NUM_ACTIONS)


def test_legal_mask(ctx: DeviceContext) raises:
    """La mascara marca 1 en las casillas libres y 0 en las ocupadas."""
    boards_l = List[List[Scalar[dtype]]]()
    exp_l = List[List[Scalar[dtype]]]()
    # Con huecos: ocupadas 0,2,4,6 -> libres 1,3,5,7,8.
    boards_l.append(board9(1,0,-1, 0,1,0, -1,0,0))
    exp_l.append(   board9(0,1, 0, 1,0,1,  0,1,1))
    # Vacio: todo legal.
    boards_l.append(board9(0,0,0, 0,0,0, 0,0,0))
    exp_l.append(   board9(1,1,1, 1,1,1, 1,1,1))
    # Lleno: nada legal.
    boards_l.append(board9(1,-1,1, -1,1,-1, 1,-1,1))
    exp_l.append(   board9(0, 0,0,  0,0, 0, 0, 0,0))

    n = len(boards_l)
    batch = List[Scalar[dtype]]()
    for i in range(n):
        for c in range(NUM_CELLS): batch.append(boards_l[i][c])

    got = run_legal_mask(ctx, batch, n)
    for i in range(n):
        for c in range(NUM_ACTIONS):
            assert_close(got[i * NUM_ACTIONS + c], exp_l[i][c], TOL,
                         String("mascara legal tablero ", i, " casilla ", c))
    print("PASS mascara legal (huecos, vacio, lleno)")


def test_apply_changes_one_cell(ctx: DeviceContext) raises:
    """Aplicar una jugada cambia solo esa casilla; el resto queda igual."""
    # Dos particulas, misma ficha (X), casillas distintas: cada una cambia la suya.
    startA = board9(1,0,-1, 0,1,0, -1,0,0)   # libre en 1,3,5,7,8
    startB = board9(0,0,-1, 0,1,0, 0,0,0)    # libre en 0,1,3,5,6,7,8
    batch = List[Scalar[dtype]]()
    for c in range(NUM_CELLS): batch.append(startA[c])
    for c in range(NUM_CELLS): batch.append(startB[c])

    actions = List[Scalar[idx_dtype]]()
    actions.append(Scalar[idx_dtype](3))   # particula 0 -> casilla 3
    actions.append(Scalar[idx_dtype](8))   # particula 1 -> casilla 8

    state = upload[dtype](ctx, batch)
    action = upload[idx_dtype](ctx, actions)
    ctx.enqueue_function[ttt_apply_kernel, ttt_apply_kernel](
        state.unsafe_ptr(), action.unsafe_ptr(), 2, CELL_AGENT,
        grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()
    got = download[dtype](state, 2 * NUM_CELLS)

    expA = board9(1,0,-1, 1,1,0, -1,0,0)   # casilla 3 -> X
    expB = board9(0,0,-1, 0,1,0, 0,0,1)    # casilla 8 -> X
    for c in range(NUM_CELLS):
        assert_close(got[c], expA[c], TOL, String("apply X tablero 0 casilla ", c))
        assert_close(got[NUM_CELLS + c], expB[c], TOL,
                     String("apply X tablero 1 casilla ", c))

    # Una jugada de O para comprobar el argumento player.
    startC = board9(0,0,0, 0,0,0, 0,0,0)
    stateC = upload[dtype](ctx, startC)
    actionC = List[Scalar[idx_dtype]]()
    actionC.append(Scalar[idx_dtype](4))
    actC = upload[idx_dtype](ctx, actionC)
    ctx.enqueue_function[ttt_apply_kernel, ttt_apply_kernel](
        stateC.unsafe_ptr(), actC.unsafe_ptr(), 1, CELL_RIVAL,
        grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()
    gotC = download[dtype](stateC, NUM_CELLS)
    expC = board9(0,0,0, 0,-1,0, 0,0,0)    # casilla 4 -> O
    for c in range(NUM_CELLS):
        assert_close(gotC[c], expC[c], TOL, String("apply O casilla ", c))

    print("PASS aplicar jugada cambia solo esa casilla (X y O)")


def main() raises:
    with DeviceContext() as ctx:
        test_cell_codes(ctx)
        test_layout_roundtrip(ctx)
        test_wins_on_every_line(ctx)
        test_no_false_win(ctx)
        test_players_dont_cross(ctx)
        test_legal_mask(ctx)
        test_apply_changes_one_cell(ctx)
