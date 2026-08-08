"""TTT board layout: 9 floats per particle, without overlapping.

A1a only pins down the storage convention. The test uploads DIFFERENT boards for
several particles and checks that the accessor reads the right cell of the right
particle -- that the STATE_DIM stride is the correct one and that a particle does
not see its neighbour's cells (the same kind of check as root_fn's broadcast).
"""

from std.gpu.host import DeviceContext

from ops.common import dtype, idx_dtype, NEG_INF
from envs.tictactoe import (ttt_read_cells_kernel, ttt_has_won_kernel,
                            ttt_legal_mask_kernel, ttt_apply_kernel,
                            ttt_outcome_kernel, ttt_prior_logits_kernel,
                            ttt_dynamics_kernel, ttt_encode_obs_kernel,
                            ttt_legal_mask_from_obs_kernel,
                            TicTacToe, default_tictactoe,
                            NUM_CELLS, NUM_ACTIONS, STATE_DIM, OBS_DIM,
                            CELL_EMPTY, CELL_AGENT, CELL_RIVAL, TPB_TTT)
from systems.spo.particles import Particles, StepOutputs
from systems.spo.spo_types import SPOConfig
from tests.helpers import (upload, zeros, download, filled, write_into,
                           assert_close)

comptime TOL = Scalar[dtype](1e-6)


def board9(c0: Int, c1: Int, c2: Int, c3: Int, c4: Int,
           c5: Int, c6: Int, c7: Int, c8: Int) -> List[Scalar[dtype]]:
    """A readable board: 9 cell codes (1=X agent, -1=O rival, 0=empty) in the order
    0..8. Every TTT test builds boards this way."""
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
    """The cell codes are symmetric and distinguishable.

    X (+1) and O (-1) sum to 0 (the empty cell): the symmetry about 0 is what suits
    the M-phase network. And their difference is 2, that is, they cannot be
    confused with one another.
    """
    assert_close(CELL_AGENT + CELL_RIVAL, CELL_EMPTY, TOL,
                 "X y O deberian ser simetricos respecto a la casilla vacia")
    assert_close(CELL_AGENT - CELL_RIVAL, Scalar[dtype](2), TOL,
                 "X y O deberian ser distinguibles")
    print("PASS codigos de casilla simetricos (X+O=vacia) y distinguibles")


def test_layout_roundtrip(ctx: DeviceContext) raises:
    """Three different boards, one per particle, read by the accessor.

    If the stride were other than STATE_DIM, or the accessor mixed particles up,
    some cell would come out with the neighbour's value and the test would catch it.
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
    """Runs ttt_has_won_kernel for `player` over n boards and brings the flags down."""
    state = upload[dtype](ctx, boards)
    won = zeros[dtype](ctx, n)
    ctx.enqueue_function[ttt_has_won_kernel, ttt_has_won_kernel](
        won.unsafe_ptr(), state.unsafe_ptr(), n, player,
        grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()
    return download[dtype](won, n)


def test_wins_on_every_line(ctx: DeviceContext) raises:
    """The 8 lines: one board per line with X completing it, all of them win."""
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
    """Boards with no completed line do not count as a win."""
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
    """One player's win does not count for the other.

    Board 0: O completes row 0 (X has loose marks with no line).
    Board 1: X completes the diagonal (O has loose marks with no line).
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
    """Runs ttt_legal_mask_kernel over n boards and brings the [n, 9] mask down."""
    state = upload[dtype](ctx, boards)
    mask = zeros[dtype](ctx, n * NUM_ACTIONS)
    ctx.enqueue_function[ttt_legal_mask_kernel, ttt_legal_mask_kernel](
        mask.unsafe_ptr(), state.unsafe_ptr(), n,
        grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()
    return download[dtype](mask, n * NUM_ACTIONS)


def test_legal_mask(ctx: DeviceContext) raises:
    """The mask marks 1 on the free cells and 0 on the occupied ones."""
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
    """Applying a move changes only that cell; the rest stays as it was."""
    # Two particles, same mark (X), different cells: each changes its own.
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

    # One O move, to check the player argument.
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
    """Terminal and reward of each board, already on the host. A struct and not a
    tuple because in 1.0.0b1 a tuple of Lists will not construct."""
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
    """The four endings + one non-terminal, with their agent reward."""
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
    # X wins and also fills the board: it is a win, not a draw.
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
    """Runs ttt_prior_logits_kernel over n root states and brings the [n, 9] logits down."""
    state = upload[dtype](ctx, boards)
    logits = zeros[dtype](ctx, n * NUM_ACTIONS)
    ctx.enqueue_function[ttt_prior_logits_kernel, ttt_prior_logits_kernel](
        logits.unsafe_ptr(), state.unsafe_ptr(), n,
        grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()
    return download[dtype](logits, n * NUM_ACTIONS)


def test_prior_masks_illegal(ctx: DeviceContext) raises:
    """The root prior: logit 0 on the legal cells, NEG_INF on the occupied ones.

    After the softmax that is a uniform distribution over the free cells and
    probability 0 of playing on a mark already placed.
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
    """The step's outputs, already on the host (a struct because in 1.0.0b1 a tuple
    of Lists will not construct)."""
    var state: List[Scalar[dtype]]
    var reward: List[Scalar[dtype]]
    var discount: List[Scalar[dtype]]
    var next_value: List[Scalar[dtype]]


def run_dynamics_at(ctx: DeviceContext, boards: List[Scalar[dtype]],
                    actions: List[Scalar[idx_dtype]],
                    uniforms: List[Scalar[dtype]],
                    depths: List[Scalar[idx_dtype]], n: Int,
                    reward_gamma: Scalar[dtype],
                    loss_penalty: Scalar[dtype] = 0) raises -> DynResult:
    """Runs ttt_dynamics_kernel with the given depth and discount."""
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
        next_value.unsafe_ptr(), n, reward_gamma, loss_penalty,
        grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()
    return DynResult(download[dtype](state, n * NUM_CELLS),
                     download[dtype](reward, n), download[dtype](discount, n),
                     download[dtype](next_value, n))


def run_dynamics(ctx: DeviceContext, boards: List[Scalar[dtype]],
                 actions: List[Scalar[idx_dtype]], uniforms: List[Scalar[dtype]],
                 n: Int) raises -> DynResult:
    """The base case: depth 0 and no discount, that is, the raw reward."""
    depths = List[Scalar[idx_dtype]]()
    for _ in range(n):
        depths.append(Scalar[idx_dtype](0))
    return run_dynamics_at(ctx, boards, actions, uniforms, depths, n,
                           Scalar[dtype](1))


def test_step_all_paths(ctx: DeviceContext) raises:
    """The step's paths, with boards/actions/uniforms set by hand.

    The rival is random, so for the deterministic cases I set up boards where O has
    only one legal cell; and for the 'goes on' case I leave two gaps and choose the
    uniform so as to know which one it takes.
    """
    boards = List[List[Scalar[dtype]]]()
    acts = List[Scalar[idx_dtype]]()
    us = List[Scalar[dtype]]()
    exp_board = List[List[Scalar[dtype]]]()
    exp_reward = List[Scalar[dtype]]()
    exp_discount = List[Scalar[dtype]]()
    names = List[String]()

    # the agent wins: it completes row 0.
    boards.append(board9(1,1,0, -1,-1,0, 0,0,0)); acts.append(Scalar[idx_dtype](2)); us.append(Scalar[dtype](0.5))
    exp_board.append(board9(1,1,1, -1,-1,0, 0,0,0)); exp_reward.append(Scalar[dtype](1)); exp_discount.append(Scalar[dtype](0)); names.append("gana agente")
    # draw: the agent's move fills the board with no line.
    boards.append(board9(1,-1,1, 1,-1,-1, -1,1,0)); acts.append(Scalar[idx_dtype](8)); us.append(Scalar[dtype](0.5))
    exp_board.append(board9(1,-1,1, 1,-1,-1, -1,1,1)); exp_reward.append(Scalar[dtype](0.5)); exp_discount.append(Scalar[dtype](0)); names.append("empate al llenar")
    # the rival wins: after the agent's move, O has only cell 6 and with it makes column 0.
    boards.append(board9(-1,1,0, -1,1,-1, 0,-1,1)); acts.append(Scalar[idx_dtype](2)); us.append(Scalar[dtype](0.5))
    exp_board.append(board9(-1,1,1, -1,1,-1, -1,-1,1)); exp_reward.append(Scalar[dtype](0)); exp_discount.append(Scalar[dtype](0)); names.append("gana rival")
    # goes on: u=0.1 -> the rival takes the 1st empty cell (7).
    boards.append(board9(1,-1,1, -1,1,0, -1,0,0)); acts.append(Scalar[idx_dtype](5)); us.append(Scalar[dtype](0.1))
    exp_board.append(board9(1,-1,1, -1,1,1, -1,-1,0)); exp_reward.append(Scalar[dtype](0)); exp_discount.append(Scalar[dtype](1)); names.append("sigue u=0.1 -> O en 7")
    # goes on: u=0.9 -> the rival takes the 2nd empty cell (8).
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
    """The reward is discounted by depth: winning NOW is worth more than winning late.

    It exists because of a concrete diagnosis from the demo: without a discount
    (gamma=1) the SMC weight is the sum of undiscounted rewards, so a particle that
    wins at step 0 TIES with one that wins at step 3, and the readout's softmax
    cannot tell "I win for sure" from "I won by luck". With gamma<1 the tie breaks:
    it was measured that q(winning move) goes from 0.24 to 0.9999 in a position with
    an immediate win.

    The same winning board evaluated at four depths: the reward has to come out as
    gamma^depth.
    """
    n = 4
    gamma = Scalar[dtype](0.5)
    boards = List[Scalar[dtype]]()
    acts = List[Scalar[idx_dtype]]()
    us = List[Scalar[dtype]]()
    depths = List[Scalar[idx_dtype]]()
    for d in range(n):
        # X X . / -1 -1 . / . . .  -> action 2 completes row 0 and wins.
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
        # Terminal in all of them: the discount does not change the discount field, only the reward.
        assert_close(out.discount[d], Scalar[dtype](0), TOL,
                     String("una victoria es terminal, profundidad ", d))
        want *= gamma

    # And with gamma=1 all four are worth the same: it is the tie the discount breaks.
    flat = run_dynamics_at(ctx, boards, acts, us, depths, n, Scalar[dtype](1))
    for d in range(n):
        assert_close(flat.reward[d], Scalar[dtype](1), TOL,
                     String("sin descuento toda victoria vale 1, profundidad ", d))
    print("PASS la recompensa se descuenta por profundidad (gamma^d)")


def ttt_config(num_envs: Int, num_particles: Int) -> SPOConfig:
    """A small config for testing the TTT model in isolation."""
    return SPOConfig(num_envs=num_envs, num_particles=num_particles,
                     num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                     search_depth=4, resample_period=4, temperature=0.5,
                     search_gamma=1.0, search_gae_lambda=1.0)


def test_model_eval_root(ctx: DeviceContext) raises:
    """The model's eval_root: masked prior at the root + V set to 0."""
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
    # I fill the value with 99 to check that eval_root OVERWRITES it with 0 (not
    # that it was already 0 by coincidence).
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
    """The model's step: it advances the state and fills action_logits with the new prior."""
    model = default_tictactoe()
    cfg = ttt_config(1, 1)
    p_total = cfg.num_search_particles()   # 1

    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    # A3b's 'goes on' case: the agent plays 5, the rival (u=0.1) takes 7.
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
    # action_logits = masked prior of the new board: 0 only on cell 8.
    for c in range(NUM_ACTIONS):
        want = Scalar[dtype](0)
        if new_board[c] != CELL_EMPTY:
            want = NEG_INF
        assert_close(got_logits[c], want, TOL,
                     String("step action_logits casilla ", c))
    print("PASS step del modelo: avanza el estado y da el prior del estado nuevo")


def run_encode(ctx: DeviceContext, boards: List[Scalar[dtype]],
               n: Int) raises -> List[Scalar[dtype]]:
    """Corre ttt_encode_obs_kernel y baja la observacion [n, OBS_DIM]."""
    state = upload[dtype](ctx, boards)
    obs = filled[dtype](ctx, n * OBS_DIM, Scalar[dtype](-1))   # -1 = sin escribir
    ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
        obs.unsafe_ptr(), state.unsafe_ptr(), n,
        grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()
    return download[dtype](obs, n * OBS_DIM)


def test_loss_penalty_separates_losing_from_continuing(ctx: DeviceContext) raises:
    """With `loss_penalty`, losing stops being worth the same as playing on.

    It is the problem the losses audit uncovered: `ttt_advance` returns reward 0
    whether the game is LOST or simply GOES ON, so the SMC weight cannot tell them
    apart and the search has no reason whatsoever to block a threat.

    Four particles, one per outcome, all at depth 0 so that the discount does not
    muddy the reading.
    """
    boards = List[Scalar[dtype]]()
    # p0 gana: X en 0 y 1, juega la 2.
    for b in board9(1,1,0, -1,-1,0, 0,0,0): boards.append(b)
    # p1 loses: O on 3 and 4, X plays 8 (does not block) and the rival finishes on 5.
    for b in board9(1,1,0, -1,-1,0, 0,0,0): boards.append(b)
    # p2 draws: a nearly full board with no lines, X closes the last cell.
    for b in board9(1,-1,1, 1,-1,-1, -1,1,0): boards.append(b)
    # p3 sigue: tablero vacio.
    for b in board9(0,0,0, 0,0,0, 0,0,0): boards.append(b)

    acts = List[Scalar[idx_dtype]]()
    acts.append(Scalar[idx_dtype](2))    # gana
    acts.append(Scalar[idx_dtype](8))    # no bloquea
    acts.append(Scalar[idx_dtype](8))    # empata
    acts.append(Scalar[idx_dtype](4))    # sigue
    us = List[Scalar[dtype]]()
    us.append(Scalar[dtype](0.1))
    # After X's move on 8, cells 2,5,6,7 are free; u=0.3 picks the second, that is
    # cell 5, which completes line 3-4-5 for the rival.
    us.append(Scalar[dtype](0.3))
    us.append(Scalar[dtype](0.1))
    us.append(Scalar[dtype](0.1))
    depths = List[Scalar[idx_dtype]]()
    for _ in range(4): depths.append(Scalar[idx_dtype](0))

    # Without a penalty: losing and going on give the same, which is exactly the problem.
    plain = run_dynamics_at(ctx, boards, acts, us, depths, 4, 1.0, 0)
    assert_close(plain.reward[0], Scalar[dtype](1), TOL, "sin castigo, ganar")
    assert_close(plain.reward[1], Scalar[dtype](0), TOL, "sin castigo, perder")
    assert_close(plain.reward[3], Scalar[dtype](0), TOL, "sin castigo, seguir")
    if plain.discount[1] != Scalar[dtype](0):
        raise Error("la particula 1 deberia haber perdido (discount 0)")

    # With penalty 1: games' +1 / 0 / -1 convention.
    pen = run_dynamics_at(ctx, boards, acts, us, depths, 4, 1.0, 1)
    assert_close(pen.reward[0], Scalar[dtype](1), TOL, "con castigo, ganar")
    assert_close(pen.reward[1], Scalar[dtype](-1), TOL, "con castigo, perder")
    assert_close(pen.reward[2], Scalar[dtype](0.5), TOL, "con castigo, empatar")
    assert_close(pen.reward[3], Scalar[dtype](0), TOL, "con castigo, seguir")

    # And the penalty is discounted by depth like any reward: losing later hurts
    # less, just as winning later rewards less.
    deep = List[Scalar[idx_dtype]]()
    for _ in range(4): deep.append(Scalar[idx_dtype](2))
    d2 = run_dynamics_at(ctx, boards, acts, us, deep, 4, 0.5, 1)
    assert_close(d2.reward[1], Scalar[dtype](-0.25), TOL,
                 "perder en la profundidad 2 con gamma 0.5")
    print("PASS loss_penalty separa perder de seguir, y se descuenta igual")


def test_encode_obs_two_planes(ctx: DeviceContext) raises:
    """The board is translated into two binary planes, computed by hand.

    It is what the critic and the actor will eat, so an error here would poison the
    whole M-step without anything failing: the network would simply learn wrongly.
    """
    # X . O / . X . / . . .
    b = board9(1,0,-1, 0,1,0, 0,0,0)
    got = run_encode(ctx, b, 1)

    # plano 0 = mis fichas (casillas 0 y 4), plano 1 = las suyas (casilla 2).
    want_mine = board9(1,0,0, 0,1,0, 0,0,0)
    want_theirs = board9(0,0,1, 0,0,0, 0,0,0)
    for c in range(NUM_CELLS):
        assert_close(got[c], want_mine[c], TOL,
                     String("plano propio, casilla ", c))
        assert_close(got[NUM_CELLS + c], want_theirs[c], TOL,
                     String("plano del rival, casilla ", c))
    print("PASS la observacion son dos planos binarios (mias / suyas)")


def test_encode_obs_edge_cases(ctx: DeviceContext) raises:
    """Empty, full, and the invariant that relates them.

    The invariant matters: a cell cannot be in both planes at once, and an occupied
    one has to be in exactly one. That catches a swap of planes or a badly written
    comparison.
    """
    boards = List[Scalar[dtype]]()
    empty = board9(0,0,0, 0,0,0, 0,0,0)
    full = board9(1,-1,1, -1,1,-1, 1,-1,1)
    for c in range(NUM_CELLS): boards.append(empty[c])
    for c in range(NUM_CELLS): boards.append(full[c])

    got = run_encode(ctx, boards, 2)

    # The empty one: all 18 values at zero (nothing in either plane).
    for i in range(OBS_DIM):
        assert_close(got[i], Scalar[dtype](0), TOL,
                     String("tablero vacio, valor ", i))

    # The full one and the invariant, over both boards.
    for t in range(2):
        base = t * OBS_DIM
        for c in range(NUM_CELLS):
            mine = got[base + c]
            theirs = got[base + NUM_CELLS + c]
            if mine + theirs > Scalar[dtype](1.5):
                raise Error("la casilla ", c, " del tablero ", t,
                            " esta en LOS DOS planos")
            # The source board comes out of the flat array, without copying lists.
            cell = boards[t * NUM_CELLS + c]
            occupied = Scalar[dtype](0) if cell == CELL_EMPTY else Scalar[dtype](1)
            assert_close(mine + theirs, occupied, TOL,
                         String("ocupada = en exactamente un plano, tablero ", t,
                                " casilla ", c))
    print("PASS bordes: vacio a cero, y cada casilla ocupada en un solo plano")


def test_encode_obs_no_overlap(ctx: DeviceContext) raises:
    """Several boards in a row: none overwrites its neighbour's observation.

    The same kind of check as A1a's layout, but over the OBS_DIM stride instead of
    STATE_DIM.
    """
    b0 = board9(1,1,1, 0,0,0, 0,0,0)      # solo mias, la fila de arriba
    b1 = board9(-1,-1,-1, 0,0,0, 0,0,0)   # solo suyas, la misma fila
    b2 = board9(0,0,0, 0,0,0, 0,0,1)      # una mia, la esquina

    boards = List[Scalar[dtype]]()
    for c in range(NUM_CELLS): boards.append(b0[c])
    for c in range(NUM_CELLS): boards.append(b1[c])
    for c in range(NUM_CELLS): boards.append(b2[c])
    got = run_encode(ctx, boards, 3)

    # Board 0: three ones in the own plane, nothing in the rival's.
    for c in range(3):
        assert_close(got[c], Scalar[dtype](1), TOL, String("t0 mia ", c))
    for c in range(NUM_CELLS):
        assert_close(got[NUM_CELLS + c], Scalar[dtype](0), TOL,
                     String("t0 no deberia tener fichas del rival, ", c))
    # Tablero 1: al reves, y en su propio hueco.
    for c in range(3):
        assert_close(got[OBS_DIM + NUM_CELLS + c], Scalar[dtype](1), TOL,
                     String("t1 suya ", c))
        assert_close(got[OBS_DIM + c], Scalar[dtype](0), TOL,
                     String("t1 no deberia tener fichas mias, ", c))
    # Board 2: only cell 8 in the own plane.
    assert_close(got[2 * OBS_DIM + 8], Scalar[dtype](1), TOL, "t2 esquina mia")
    total = Scalar[dtype](0)
    for i in range(OBS_DIM):
        total += got[2 * OBS_DIM + i]
    assert_close(total, Scalar[dtype](1), TOL, "t2 deberia tener una sola ficha")
    print("PASS tres tableros seguidos, cada observacion en su hueco")


def test_mask_from_obs_matches_mask_from_state(ctx: DeviceContext) raises:
    """The mask taken from the OBSERVATION matches the one taken from the board.

    The training buffer stores observations (18 floats), not states (9), so the
    actor needs to derive legality from there. The two routes HAVE to give the same
    thing: if they diverged, the actor would train with a different mask from the
    one it uses when playing, and would put probability on occupied cells without
    anything failing.

    It is tested with 40 boards so as to go past one block (TPB=32) and with a
    non-round size.
    """
    boards = List[Scalar[dtype]]()
    n = 40
    for i in range(n):
        # Varied, deterministic boards: cell c gets filled according to i and c.
        for c in range(NUM_CELLS):
            v = (i * 7 + c * 3) % 5
            if v == 0: boards.append(Scalar[dtype](1))
            elif v == 1: boards.append(Scalar[dtype](-1))
            else: boards.append(Scalar[dtype](0))

    state = upload[dtype](ctx, boards)
    obs = zeros[dtype](ctx, n * OBS_DIM)
    from_state = zeros[dtype](ctx, n * NUM_ACTIONS)
    from_obs = filled[dtype](ctx, n * NUM_ACTIONS, Scalar[dtype](-1))

    blocks = (n + TPB_TTT - 1) // TPB_TTT
    ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
        obs.unsafe_ptr(), state.unsafe_ptr(), n,
        grid_dim=blocks, block_dim=TPB_TTT)
    ctx.enqueue_function[ttt_legal_mask_kernel, ttt_legal_mask_kernel](
        from_state.unsafe_ptr(), state.unsafe_ptr(), n,
        grid_dim=blocks, block_dim=TPB_TTT)
    ctx.enqueue_function[ttt_legal_mask_from_obs_kernel,
                         ttt_legal_mask_from_obs_kernel](
        from_obs.unsafe_ptr(), obs.unsafe_ptr(), n,
        grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()

    a = download[dtype](from_state, n * NUM_ACTIONS)
    b = download[dtype](from_obs, n * NUM_ACTIONS)
    for i in range(n * NUM_ACTIONS):
        if a[i] != b[i]:
            raise Error("la mascara difiere en ", i, ": del estado ", a[i],
                        " y de la observacion ", b[i])
        # And against the board, not just one against the other: if both were
        # wrong in the same way, comparing them would not detect it.
        want = Scalar[dtype](1) if boards[(i // NUM_ACTIONS) * NUM_CELLS
                                         + (i % NUM_ACTIONS)] == 0 \
               else Scalar[dtype](0)
        if b[i] != want:
            raise Error("la mascara de la observacion en ", i, " deberia ser ",
                        want)
    print("PASS la mascara de la observacion coincide con la del tablero (40 filas)")


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
        test_encode_obs_two_planes(ctx)
        test_encode_obs_edge_cases(ctx)
        test_encode_obs_no_overlap(ctx)
        test_model_eval_root(ctx)
        test_model_step(ctx)
        test_loss_penalty_separates_losing_from_continuing(ctx)
        test_mask_from_obs_matches_mask_from_state(ctx)
