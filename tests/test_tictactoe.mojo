"""Layout del tablero de TTT: 9 floats por particula, sin solaparse.

A1a solo fija la convencion de almacenamiento. La prueba sube tableros DISTINTOS
para varias particulas y comprueba que el accesor lee la casilla correcta de la
particula correcta -- que el paso STATE_DIM es el bueno y una particula no ve las
casillas de su vecina (el mismo tipo de comprobacion que el broadcast de root_fn).
"""

from std.gpu.host import DeviceContext

from ops.common import dtype, idx_dtype, NEG_INF
from envs.tictactoe import (ttt_read_cells_kernel, ttt_has_won_kernel,
                            ttt_legal_mask_kernel, ttt_apply_kernel,
                            ttt_outcome_kernel, ttt_prior_logits_kernel,
                            ttt_dynamics_kernel, TicTacToe, default_tictactoe,
                            NUM_CELLS, NUM_ACTIONS, STATE_DIM,
                            CELL_EMPTY, CELL_AGENT, CELL_RIVAL, TPB_TTT)
from systems.spo.particles import Particles, StepOutputs
from systems.spo.spo_types import SPOConfig
from tests.helpers import (upload, zeros, download, filled, write_into,
                           assert_close)

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


@fieldwise_init
struct Outcome(Movable):
    """Terminal y reward de cada tablero, ya en el host. Struct y no tupla porque
    en 1.0.0b1 una tupla de List no se deja construir."""
    var terminal: List[Scalar[dtype]]
    var reward: List[Scalar[dtype]]


def run_outcome(ctx: DeviceContext, boards: List[Scalar[dtype]],
                n: Int) raises -> Outcome:
    """Corre ttt_outcome_kernel sobre n tableros y baja terminal + reward."""
    state = upload[dtype](ctx, boards)
    term = zeros[dtype](ctx, n)
    rew = zeros[dtype](ctx, n)
    ctx.enqueue_function[ttt_outcome_kernel, ttt_outcome_kernel](
        term.unsafe_ptr(), rew.unsafe_ptr(), state.unsafe_ptr(), n,
        grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()
    return Outcome(download[dtype](term, n), download[dtype](rew, n))


def test_terminal_and_reward(ctx: DeviceContext) raises:
    """Los cuatro finales + un no-terminal, con su recompensa de agente."""
    boards_l = List[List[Scalar[dtype]]]()
    term_exp = List[Scalar[dtype]]()
    rew_exp = List[Scalar[dtype]]()
    names = List[String]()

    # Gana X (fila 0).
    boards_l.append(board9(1,1,1, -1,-1,0, 0,0,0))
    term_exp.append(Scalar[dtype](1)); rew_exp.append(Scalar[dtype](1))
    names.append("gana X")
    # Gana O (fila 0) -> derrota del agente.
    boards_l.append(board9(-1,-1,-1, 1,1,0, 0,0,1))
    term_exp.append(Scalar[dtype](1)); rew_exp.append(Scalar[dtype](0))
    names.append("gana O (derrota)")
    # Empate: tablero lleno sin ninguna linea.
    boards_l.append(board9(1,-1,1, 1,-1,-1, -1,1,1))
    term_exp.append(Scalar[dtype](1)); rew_exp.append(Scalar[dtype](0.5))
    names.append("empate")
    # No terminal: quedan huecos y nadie gano.
    boards_l.append(board9(1,0,-1, 0,1,0, -1,0,0))
    term_exp.append(Scalar[dtype](0)); rew_exp.append(Scalar[dtype](0))
    names.append("no terminal")
    # Gana X y ademas llena el tablero: es victoria, no empate.
    boards_l.append(board9(1,1,1, -1,-1,1, -1,1,-1))
    term_exp.append(Scalar[dtype](1)); rew_exp.append(Scalar[dtype](1))
    names.append("gana X en tablero lleno")

    n = len(boards_l)
    batch = List[Scalar[dtype]]()
    for i in range(n):
        for c in range(NUM_CELLS): batch.append(boards_l[i][c])

    out = run_outcome(ctx, batch, n)
    for i in range(n):
        assert_close(out.terminal[i], term_exp[i], TOL,
                     String("terminal incorrecto: ", names[i]))
        assert_close(out.reward[i], rew_exp[i], TOL,
                     String("reward incorrecto: ", names[i]))
    print("PASS terminal y recompensa (gana X / gana O / empate / no terminal / X en lleno)")


def run_prior(ctx: DeviceContext, boards: List[Scalar[dtype]],
              n: Int) raises -> List[Scalar[dtype]]:
    """Corre ttt_prior_logits_kernel sobre n estados raiz y baja los logits [n, 9]."""
    state = upload[dtype](ctx, boards)
    logits = zeros[dtype](ctx, n * NUM_ACTIONS)
    ctx.enqueue_function[ttt_prior_logits_kernel, ttt_prior_logits_kernel](
        logits.unsafe_ptr(), state.unsafe_ptr(), n,
        grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()
    return download[dtype](logits, n * NUM_ACTIONS)


def test_prior_masks_illegal(ctx: DeviceContext) raises:
    """El prior raiz: logit 0 en las casillas legales, NEG_INF en las ocupadas.

    Tras el softmax eso es una distribucion uniforme sobre las casillas libres y
    probabilidad 0 de jugar sobre una ficha ya puesta.
    """
    boards_l = List[List[Scalar[dtype]]]()
    boards_l.append(board9(1,0,-1, 0,1,0, -1,0,0))    # ocupadas 0,2,4,6
    boards_l.append(board9(0,0,0, 0,0,0, 0,0,0))       # vacio: todo legal
    boards_l.append(board9(1,-1,1, -1,1,-1, 1,-1,1))   # lleno: todo ilegal (degenerado)

    n = len(boards_l)
    batch = List[Scalar[dtype]]()
    for i in range(n):
        for c in range(NUM_CELLS): batch.append(boards_l[i][c])

    got = run_prior(ctx, batch, n)
    for e in range(n):
        for c in range(NUM_ACTIONS):
            want = Scalar[dtype](0)
            if boards_l[e][c] != CELL_EMPTY:
                want = NEG_INF
            assert_close(got[e * NUM_ACTIONS + c], want, TOL,
                         String("prior raiz env ", e, " casilla ", c))
    print("PASS prior raiz enmascarado (0 en legales, NEG_INF en ocupadas)")


@fieldwise_init
struct DynResult(Movable):
    """Salidas del step, ya en el host (struct porque en 1.0.0b1 una tupla de
    List no se deja construir)."""
    var state: List[Scalar[dtype]]
    var reward: List[Scalar[dtype]]
    var discount: List[Scalar[dtype]]
    var next_value: List[Scalar[dtype]]


def run_dynamics_at(ctx: DeviceContext, boards: List[Scalar[dtype]],
                    actions: List[Scalar[idx_dtype]],
                    uniforms: List[Scalar[dtype]],
                    depths: List[Scalar[idx_dtype]], n: Int,
                    reward_gamma: Scalar[dtype]) raises -> DynResult:
    """Corre ttt_dynamics_kernel con la profundidad y el descuento dados."""
    state = upload[dtype](ctx, boards)
    action = upload[idx_dtype](ctx, actions)
    us = upload[dtype](ctx, uniforms)
    depth = upload[idx_dtype](ctx, depths)
    reward = zeros[dtype](ctx, n)
    discount = zeros[dtype](ctx, n)
    next_value = zeros[dtype](ctx, n)
    ctx.enqueue_function[ttt_dynamics_kernel, ttt_dynamics_kernel](
        state.unsafe_ptr(), action.unsafe_ptr(), us.unsafe_ptr(),
        depth.unsafe_ptr(), reward.unsafe_ptr(), discount.unsafe_ptr(),
        next_value.unsafe_ptr(), n, reward_gamma,
        grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()
    return DynResult(download[dtype](state, n * NUM_CELLS),
                     download[dtype](reward, n), download[dtype](discount, n),
                     download[dtype](next_value, n))


def run_dynamics(ctx: DeviceContext, boards: List[Scalar[dtype]],
                 actions: List[Scalar[idx_dtype]], uniforms: List[Scalar[dtype]],
                 n: Int) raises -> DynResult:
    """El caso base: profundidad 0 y sin descuento, o sea la recompensa cruda."""
    depths = List[Scalar[idx_dtype]]()
    for _ in range(n):
        depths.append(Scalar[idx_dtype](0))
    return run_dynamics_at(ctx, boards, actions, uniforms, depths, n,
                           Scalar[dtype](1))


def test_step_all_paths(ctx: DeviceContext) raises:
    """Los caminos del step, con tableros/acciones/uniformes puestos a mano.

    El rival es aleatorio, asi que para los casos deterministas monto tableros
    donde O solo tiene una casilla legal; y para el caso 'sigue' pongo dos huecos
    y elijo el uniforme para saber cual le toca.
    """
    boards = List[List[Scalar[dtype]]]()
    acts = List[Scalar[idx_dtype]]()
    us = List[Scalar[dtype]]()
    exp_board = List[List[Scalar[dtype]]]()
    exp_reward = List[Scalar[dtype]]()
    exp_discount = List[Scalar[dtype]]()
    names = List[String]()

    # gana el agente: completa la fila 0.
    boards.append(board9(1,1,0, -1,-1,0, 0,0,0)); acts.append(Scalar[idx_dtype](2)); us.append(Scalar[dtype](0.5))
    exp_board.append(board9(1,1,1, -1,-1,0, 0,0,0)); exp_reward.append(Scalar[dtype](1)); exp_discount.append(Scalar[dtype](0)); names.append("gana agente")
    # empate: la jugada del agente llena el tablero sin linea.
    boards.append(board9(1,-1,1, 1,-1,-1, -1,1,0)); acts.append(Scalar[idx_dtype](8)); us.append(Scalar[dtype](0.5))
    exp_board.append(board9(1,-1,1, 1,-1,-1, -1,1,1)); exp_reward.append(Scalar[dtype](0.5)); exp_discount.append(Scalar[dtype](0)); names.append("empate al llenar")
    # gana el rival: tras la jugada del agente, O solo tiene la casilla 6 y con ella hace columna 0.
    boards.append(board9(-1,1,0, -1,1,-1, 0,-1,1)); acts.append(Scalar[idx_dtype](2)); us.append(Scalar[dtype](0.5))
    exp_board.append(board9(-1,1,1, -1,1,-1, -1,-1,1)); exp_reward.append(Scalar[dtype](0)); exp_discount.append(Scalar[dtype](0)); names.append("gana rival")
    # sigue: u=0.1 -> el rival toma la 1a casilla vacia (la 7).
    boards.append(board9(1,-1,1, -1,1,0, -1,0,0)); acts.append(Scalar[idx_dtype](5)); us.append(Scalar[dtype](0.1))
    exp_board.append(board9(1,-1,1, -1,1,1, -1,-1,0)); exp_reward.append(Scalar[dtype](0)); exp_discount.append(Scalar[dtype](1)); names.append("sigue u=0.1 -> O en 7")
    # sigue: u=0.9 -> el rival toma la 2a casilla vacia (la 8).
    boards.append(board9(1,-1,1, -1,1,0, -1,0,0)); acts.append(Scalar[idx_dtype](5)); us.append(Scalar[dtype](0.9))
    exp_board.append(board9(1,-1,1, -1,1,1, -1,0,-1)); exp_reward.append(Scalar[dtype](0)); exp_discount.append(Scalar[dtype](1)); names.append("sigue u=0.9 -> O en 8")

    n = len(boards)
    flat = List[Scalar[dtype]]()
    for i in range(n):
        for c in range(NUM_CELLS): flat.append(boards[i][c])

    out = run_dynamics(ctx, flat, acts, us, n)
    for p in range(n):
        for c in range(NUM_CELLS):
            assert_close(out.state[p * NUM_CELLS + c], exp_board[p][c], TOL,
                         String("tablero '", names[p], "' casilla ", c))
        assert_close(out.reward[p], exp_reward[p], TOL, String("reward '", names[p], "'"))
        assert_close(out.discount[p], exp_discount[p], TOL, String("discount '", names[p], "'"))
        assert_close(out.next_value[p], Scalar[dtype](0), TOL, String("next_value '", names[p], "'"))
    print("PASS step: gana agente / empate / gana rival / sigue (rival al azar 7 u 8)")


def test_step_discounts_reward_by_depth(ctx: DeviceContext) raises:
    """La recompensa se descuenta por profundidad: ganar YA vale mas que ganar tarde.

    Existe por un diagnostico concreto de la demo: sin descuento (gamma=1) el peso
    SMC es la suma de recompensas sin descontar, asi que una particula que gana en
    el paso 0 EMPATA con otra que gana en el paso 3, y el softmax del readout no
    puede distinguir "gane seguro" de "gane con suerte". Con gamma<1 el empate se
    rompe: se midio que q(jugada ganadora) sube de 0.24 a 0.9999 en una posicion
    con victoria inmediata.

    Mismo tablero ganador evaluado a cuatro profundidades: la recompensa tiene que
    salir gamma^profundidad.
    """
    n = 4
    gamma = Scalar[dtype](0.5)
    boards = List[Scalar[dtype]]()
    acts = List[Scalar[idx_dtype]]()
    us = List[Scalar[dtype]]()
    depths = List[Scalar[idx_dtype]]()
    for d in range(n):
        # X X . / -1 -1 . / . . .  -> la accion 2 completa la fila 0 y gana.
        b = board9(1,1,0, -1,-1,0, 0,0,0)
        for c in range(NUM_CELLS):
            boards.append(b[c])
        acts.append(Scalar[idx_dtype](2))
        us.append(Scalar[dtype](0.5))
        depths.append(Scalar[idx_dtype](d))

    out = run_dynamics_at(ctx, boards, acts, us, depths, n, gamma)

    want = Scalar[dtype](1)
    for d in range(n):
        assert_close(out.reward[d], want, TOL,
                     String("la victoria a profundidad ", d, " deberia valer gamma^", d))
        # Terminal en todas: el descuento no cambia el discount, solo la recompensa.
        assert_close(out.discount[d], Scalar[dtype](0), TOL,
                     String("una victoria es terminal, profundidad ", d))
        want *= gamma

    # Y con gamma=1 las cuatro valen lo mismo: es el empate que rompe el descuento.
    flat = run_dynamics_at(ctx, boards, acts, us, depths, n, Scalar[dtype](1))
    for d in range(n):
        assert_close(flat.reward[d], Scalar[dtype](1), TOL,
                     String("sin descuento toda victoria vale 1, profundidad ", d))
    print("PASS la recompensa se descuenta por profundidad (gamma^d)")


def ttt_config(num_envs: Int, num_particles: Int) -> SPOConfig:
    """Config pequena para probar el modelo TTT en aislamiento."""
    return SPOConfig(num_envs=num_envs, num_particles=num_particles,
                     num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                     search_depth=4, resample_period=4, temperature=0.5,
                     search_gamma=1.0, search_gae_lambda=1.0)


def test_model_eval_root(ctx: DeviceContext) raises:
    """El eval_root del modelo: prior enmascarado en la raiz + V puesto a 0."""
    model = default_tictactoe()
    num_envs = 2
    cfg = ttt_config(num_envs, 4)

    boards_list = List[List[Scalar[dtype]]]()
    boards_list.append(board9(1,0,-1, 0,1,0, -1,0,0))   # ocupadas 0,2,4,6
    boards_list.append(board9(0,0,0, 0,0,0, 0,0,0))      # vacio

    roots = List[Scalar[dtype]]()
    for e in range(num_envs):
        for c in range(NUM_CELLS): roots.append(boards_list[e][c])

    root_state = upload[dtype](ctx, roots)
    logits = zeros[dtype](ctx, num_envs * NUM_ACTIONS)
    # Relleno el valor a 99 para comprobar que eval_root lo PISA a 0 (no que ya
    # estuviera a 0 por casualidad).
    value = filled[dtype](ctx, num_envs, Scalar[dtype](99))
    model.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()

    got_logits = download[dtype](logits, num_envs * NUM_ACTIONS)
    got_value = download[dtype](value, num_envs)
    for e in range(num_envs):
        for c in range(NUM_ACTIONS):
            want = Scalar[dtype](0)
            if boards_list[e][c] != CELL_EMPTY:
                want = NEG_INF
            assert_close(got_logits[e * NUM_ACTIONS + c], want, TOL,
                         String("eval_root logit env ", e, " casilla ", c))
        assert_close(got_value[e], Scalar[dtype](0), TOL,
                     String("eval_root V env ", e, " deberia ser 0"))
    print("PASS eval_root del modelo: prior enmascarado + V=0")


def test_model_step(ctx: DeviceContext) raises:
    """Step del modelo: avanza el estado y rellena action_logits con el prior nuevo."""
    model = default_tictactoe()
    cfg = ttt_config(1, 1)
    p_total = cfg.num_search_particles()   # 1

    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    # El caso 'sigue' de A3b: el agente juega la 5, el rival (u=0.1) toma la 7.
    write_into[dtype](particles.state, board9(1,-1,1, -1,1,0, -1,0,0))
    acts = List[Scalar[idx_dtype]](); acts.append(Scalar[idx_dtype](5))
    write_into[idx_dtype](outputs.next_action, acts)
    us = List[Scalar[dtype]](); us.append(Scalar[dtype](0.1))
    step_us = upload[dtype](ctx, us)

    model.step(ctx, cfg, particles, outputs, step_us)
    ctx.synchronize()

    got_state = download[dtype](particles.state, p_total * NUM_CELLS)
    got_logits = download[dtype](outputs.action_logits, p_total * NUM_ACTIONS)

    # Tablero nuevo: X en 5, O en 7 -> solo queda libre la casilla 8.
    new_board = board9(1,-1,1, -1,1,1, -1,-1,0)
    for c in range(NUM_CELLS):
        assert_close(got_state[c], new_board[c], TOL,
                     String("step estado casilla ", c))
    # action_logits = prior enmascarado del tablero nuevo: 0 solo en la casilla 8.
    for c in range(NUM_ACTIONS):
        want = Scalar[dtype](0)
        if new_board[c] != CELL_EMPTY:
            want = NEG_INF
        assert_close(got_logits[c], want, TOL,
                     String("step action_logits casilla ", c))
    print("PASS step del modelo: avanza el estado y da el prior del estado nuevo")


def main() raises:
    with DeviceContext() as ctx:
        test_cell_codes(ctx)
        test_layout_roundtrip(ctx)
        test_wins_on_every_line(ctx)
        test_no_false_win(ctx)
        test_players_dont_cross(ctx)
        test_legal_mask(ctx)
        test_apply_changes_one_cell(ctx)
        test_terminal_and_reward(ctx)
        test_prior_masks_illegal(ctx)
        test_step_all_paths(ctx)
        test_step_discounts_reward_by_depth(ctx)
        test_model_eval_root(ctx)
        test_model_step(ctx)
