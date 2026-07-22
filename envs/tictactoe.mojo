"""Tic-Tac-Toe como entorno para la busqueda SMC.

El tablero 3x3 se guarda como 9 floats, uno por casilla (STATE_DIM = 9):

    indices          codigo de cada casilla:
    0 1 2              0.0  = vacia
    3 4 5              1.0  = ficha del AGENTE (X, juega primero)
    6 7 8             -1.0  = ficha del RIVAL  (O, jugara al azar)

Por que 9 floats sueltos y no un bitboard empaquetado como en el MCTS: aqui no
hay un arbol que acumule miles de tableros -- la busqueda pisa las particulas en
el sitio en cada profundidad -- asi que empaquetar no ahorra nada que importe, y
ademas la red de la fase M querra el tablero justo asi, como numeros sueltos.

El agente es SIEMPRE X, y (fase A3) un "step" del modelo sera jugada del agente +
jugada aleatoria del rival, de modo que cada vez que la busqueda mira el estado le
toca a X: no hace falta guardar de quien es el turno.

Este fichero, como toy_chain, no importa la busqueda: solo los tipos y (mas
adelante) el contrato SearchModel. El E-step no sabe que hay TTT detras.
"""

from std.gpu import block_dim, block_idx, thread_idx

from ops.common import dtype, GlobalF32

# El tablero son 9 casillas: 9 floats de estado, 9 acciones (una por casilla).
comptime NUM_CELLS = 9
comptime STATE_DIM = 9
comptime NUM_ACTIONS = 9

# Codigo de cada casilla. El agente (X) es +1 y el rival (O) es -1 a proposito:
# quedan simetricos respecto al 0 (la casilla vacia), lo que le viene bien a la
# red de la fase M.
comptime CELL_EMPTY = Scalar[dtype](0)
comptime CELL_AGENT = Scalar[dtype](1)
comptime CELL_RIVAL = Scalar[dtype](-1)

comptime TPB_TTT = 32


def ttt_cell(state: GlobalF32, p: Int, c: Int) -> Scalar[dtype]:
    """La casilla c (0..8) de la particula p.

    Fija en un solo sitio la convencion de layout: las 9 casillas de una
    particula van seguidas, con paso STATE_DIM, para que un kernel por-particula
    lea las suyas y no las de la vecina.
    """
    return state[p * STATE_DIM + c]


def ttt_read_cells_kernel(cells_out: GlobalF32, state: GlobalF32,
                          n_particles: Int):
    """Copia las 9 casillas de cada particula a la salida usando ttt_cell.

    Todavia no hace logica de juego: existe para comprobar que el accesor indexa
    la casilla correcta de la particula correcta (que el paso STATE_DIM es el que
    toca y las particulas no se solapan). Un hilo por particula.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p < n_particles:
        for c in range(NUM_CELLS):
            cells_out[p * NUM_CELLS + c] = ttt_cell(state, p, c)
