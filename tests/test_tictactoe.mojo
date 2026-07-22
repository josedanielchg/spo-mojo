"""Layout del tablero de TTT: 9 floats por particula, sin solaparse.

A1a solo fija la convencion de almacenamiento. La prueba sube tableros DISTINTOS
para varias particulas y comprueba que el accesor lee la casilla correcta de la
particula correcta -- que el paso STATE_DIM es el bueno y una particula no ve las
casillas de su vecina (el mismo tipo de comprobacion que el broadcast de root_fn).
"""

from std.gpu.host import DeviceContext

from ops.common import dtype
from envs.tictactoe import (ttt_read_cells_kernel, NUM_CELLS,
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


def main() raises:
    with DeviceContext() as ctx:
        test_cell_codes(ctx)
        test_layout_roundtrip(ctx)
