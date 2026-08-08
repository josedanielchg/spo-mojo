"""Tic-Tac-Toe as an environment for the SMC search.

The 3x3 board is stored as 9 floats, one per cell (STATE_DIM = 9):

    indices          code of each cell:
    0 1 2              0.0  = empty
    3 4 5              1.0  = AGENT's mark (X, moves first)
    6 7 8             -1.0  = RIVAL's mark (O, will play at random)

Why 9 loose floats instead of a packed bitboard like in the MCTS: there is no
tree here accumulating thousands of boards -- the search overwrites the particles
in place at every depth -- so packing saves nothing that matters, and besides the
M-phase network will want the board exactly like this, as loose numbers.

The agent is ALWAYS X, and (phase A3) one model "step" will be agent move +
random rival move, so that every time the search looks at the state it is X's
turn: there is no need to store whose turn it is.

This file, like toy_chain, does not import the search: only the types and (later)
the SearchModel contract. The E-step does not know there is TTT behind it.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from ops.common import dtype, idx_dtype, GlobalF32, GlobalI32, NEG_INF
from systems.spo.particles import Particles, StepOutputs
from systems.spo.search_model import SearchModel
from systems.spo.spo_types import SPOConfig

# The board is 9 cells: 9 state floats, 9 actions (one per cell).
comptime NUM_CELLS = 9
comptime STATE_DIM = 9
comptime NUM_ACTIONS = 9

# Code of each cell. The agent (X) is +1 and the rival (O) is -1 on purpose:
# they end up symmetric about 0 (the empty cell), which suits the M-phase
# network.
comptime CELL_EMPTY = Scalar[dtype](0)
comptime CELL_AGENT = Scalar[dtype](1)
comptime CELL_RIVAL = Scalar[dtype](-1)

comptime TPB_TTT = 32

# What the NETWORK sees (not the search): two planes of 9, own marks first and
# the rival's afterwards. It is the encoding pgx uses for tic-tac-toe
# (`observation.shape == (3, 3, 2)`), and we use the same one here so that the
# Mojo network and the Stoix one receive exactly the same thing and the
# comparison stays clean.
comptime NUM_PLANES = 2
comptime OBS_DIM = NUM_PLANES * NUM_CELLS   # 18


def ttt_cell(state: GlobalF32, p: Int, c: Int) -> Scalar[dtype]:
    """Cell c (0..8) of particle p.

    Pins the layout convention down in a single place: the 9 cells of one
    particle are contiguous, with stride STATE_DIM, so that a per-particle kernel
    reads its own cells and not its neighbour's.
    """
    return state[p * STATE_DIM + c]


def ttt_three(state: GlobalF32, p: Int, a: Int, b: Int, c: Int,
              player: Scalar[dtype]) -> Bool:
    """True if `player` occupies all three cells a, b, c of particle p.

    I compare cell != player and bail out as soon as one fails, instead of
    chaining three `and`s over SIMD-bool comparisons, which is what the kernel
    compiler swallows without complaint."""
    if ttt_cell(state, p, a) != player:
        return False
    if ttt_cell(state, p, b) != player:
        return False
    if ttt_cell(state, p, c) != player:
        return False
    return True


def ttt_has_won(state: GlobalF32, p: Int, player: Scalar[dtype]) -> Bool:
    """True if `player` (CELL_AGENT or CELL_RIVAL) completes any of the 8 lines.

    These are the same 8 lines as WIN_MASKS in the MCTS (3 rows, 3 columns, 2
    diagonals), here as index triples because the board is 9 floats and not a
    bitboard. The parallelism lives in the threads (one particle per thread), so
    inside the thread a plain OR over the 8 lines is the natural thing -- the
    SIMD-over-lines trick the CPU MCTS uses is not needed.
    """
    return (ttt_three(state, p, 0, 1, 2, player)      # rows
            or ttt_three(state, p, 3, 4, 5, player)
            or ttt_three(state, p, 6, 7, 8, player)
            or ttt_three(state, p, 0, 3, 6, player)    # columns
            or ttt_three(state, p, 1, 4, 7, player)
            or ttt_three(state, p, 2, 5, 8, player)
            or ttt_three(state, p, 0, 4, 8, player)    # diagonals
            or ttt_three(state, p, 2, 4, 6, player))


def ttt_has_won_kernel(won_out: GlobalF32, state: GlobalF32, n_particles: Int,
                       player: Scalar[dtype]):
    """1.0 if `player` won on its particle's board, 0.0 otherwise. One thread per
    particle."""
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p < n_particles:
        won_out[p] = Scalar[dtype](1) if ttt_has_won(state, p, player) \
                     else Scalar[dtype](0)


def ttt_is_legal(state: GlobalF32, p: Int, c: Int) -> Bool:
    """True if cell c of particle p is empty (legal move)."""
    if ttt_cell(state, p, c) == CELL_EMPTY:
        return True
    return False


def ttt_legal_mask_kernel(mask_out: GlobalF32, state: GlobalF32,
                          n_particles: Int):
    """Legal-action mask [n_particles, NUM_ACTIONS]: 1.0 if the cell is free,
    0.0 if it is taken. One thread per particle.

    The search will use it to mask the prior's illegal actions (putting -inf on
    the occupied ones) and the random rival to draw only among free cells.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p < n_particles:
        for c in range(NUM_ACTIONS):
            mask_out[p * NUM_ACTIONS + c] = Scalar[dtype](1) \
                if ttt_is_legal(state, p, c) else Scalar[dtype](0)


def ttt_apply(state: GlobalF32, p: Int, cell: Int, player: Scalar[dtype]):
    """Puts `player`'s mark on cell `cell` of particle p, in place.

    Does not check legality: it assumes the caller already picked a free cell
    (via the legal mask). It is the primitive the phase-A3 step will use for both
    the agent's move and the rival's.
    """
    state[p * STATE_DIM + cell] = player


def ttt_apply_kernel(state: GlobalF32, action: GlobalI32, n_particles: Int,
                     player: Scalar[dtype]):
    """Each particle puts a `player` mark on the cell its action names.
    One thread per particle, modifies the state in place."""
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p < n_particles:
        ttt_apply(state, p, Int(action[p]), player)


def ttt_is_full(state: GlobalF32, p: Int) -> Bool:
    """True if no empty cell is left (no legal moves)."""
    for c in range(NUM_CELLS):
        if ttt_cell(state, p, c) == CELL_EMPTY:
            return False
    return True


def ttt_is_terminal(state: GlobalF32, p: Int) -> Bool:
    """True if the game is over: somebody won or the board is full (draw)."""
    if ttt_has_won(state, p, CELL_AGENT):
        return True
    if ttt_has_won(state, p, CELL_RIVAL):
        return True
    if ttt_is_full(state, p):
        return True
    return False


def ttt_reward(state: GlobalF32, p: Int) -> Scalar[dtype]:
    """Reward of the board from the AGENT's point of view (X):

        X wins       -> +1.0
        draw         -> +0.5   (full board with no line)
        O wins       ->  0.0   (loss)
        non-terminal ->  0.0   (step reward: the game goes on)

    X-wins is checked before full: a winning move that also fills the board is a
    win, not a draw. `terminal` and `reward` are separate outputs: a loss and an
    intermediate step both give 0, and they are told apart by terminal.
    """
    if ttt_has_won(state, p, CELL_AGENT):
        return Scalar[dtype](1)
    if ttt_has_won(state, p, CELL_RIVAL):
        return Scalar[dtype](0)
    if ttt_is_full(state, p):
        return Scalar[dtype](0.5)
    return Scalar[dtype](0)


def ttt_outcome_kernel(terminal_out: GlobalF32, reward_out: GlobalF32,
                       state: GlobalF32, n_particles: Int):
    """Per particle: terminal (1.0/0.0) and the agent's reward. One thread per
    particle."""
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p < n_particles:
        terminal_out[p] = Scalar[dtype](1) if ttt_is_terminal(state, p) \
                          else Scalar[dtype](0)
        reward_out[p] = ttt_reward(state, p)


def ttt_prior_logits_kernel(logits_out: GlobalF32, state: GlobalF32,
                            n_envs: Int):
    """The search prior at the root states: [n_envs, NUM_ACTIONS].

    Uniform over the LEGAL cells (logit 0) and masked on the occupied ones
    (NEG_INF), so that the softmax gives probability 0 to playing on a mark that
    is already there. It is the analogue of the toy problem's uniform prior, but
    masked, because in TTT not every action is legal.

    I use NEG_INF (MIN_FINITE) and not a true -inf: if a board had no legal cell
    at all (should not happen at a root, but just in case), a whole row of -inf
    would give nan in the softmax; with MIN_FINITE it degenerates to uniform,
    which is harmless. One thread per env.
    """
    e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e < n_envs:
        for c in range(NUM_ACTIONS):
            logits_out[e * NUM_ACTIONS + c] = Scalar[dtype](0) \
                if ttt_is_legal(state, e, c) else NEG_INF


def ttt_random_legal_cell(state: GlobalF32, p: Int, u: Scalar[dtype]) -> Int:
    """The empty cell number floor(u*m) (m = number of empties), counting from 0.

    Uniform over the free cells, without lists: count how many there are and
    return the k-th. It is the analogue of the MCTS's random_set_bit, here over
    floats. u in [0,1). Returns -1 if there is none, which does not happen in the
    step (it is only called when the board is not full).
    """
    m = 0
    for c in range(NUM_CELLS):
        if ttt_is_legal(state, p, c):
            m += 1
    if m == 0:
        return -1
    k = Int(u * Scalar[dtype](m))
    if k >= m:            # guard in case u rounds exactly up to m
        k = m - 1
    j = 0
    for c in range(NUM_CELLS):
        if ttt_is_legal(state, p, c):
            if j == k:
                return c
            j += 1
    return -1             # unreachable with m > 0


@fieldwise_init
struct TTTOutcome(Copyable, Movable):
    """What comes out of advancing a board by one full turn. A struct of scalars,
    which a device function can indeed return (see docs/api_notes.md)."""
    var reward: Scalar[dtype]
    """AGENT's view: 1 win, 0.5 draw, 0 loss or still going."""
    var terminal: Scalar[dtype]
    """1.0 if the game ended, 0.0 if it goes on."""


def ttt_advance(state: GlobalF32, p: Int, action: Int,
                u: Scalar[dtype]) -> TTTOutcome:
    """One full turn: the agent (X) plays and the rival answers at random.

    Modifies p's board in place and returns reward and terminal:
      1. The agent plays `action`.
      2. If X wins or the board fills up -> terminal (the rival never plays).
      3. Otherwise the rival (O) plays a random legal cell, drawn with `u`.
      4. If O wins or the board fills up -> terminal.
    Each has_won/is_full is computed ONCE, in game order.

    This is the dynamics shared between the search (ttt_dynamics_kernel) and the
    real environment (ttt_env_step_kernel), just as cartpole_advance shared the
    physics: that way the rules live in a single place and cannot drift apart.

    Memory-safe by construction: `action` comes bounded to [0,9) and the rival
    only plays when the board is NOT full, so random_legal_cell always finds a
    free cell and never returns -1.
    """
    ttt_apply(state, p, action, CELL_AGENT)
    if ttt_has_won(state, p, CELL_AGENT):
        return TTTOutcome(Scalar[dtype](1), Scalar[dtype](1))     # agent wins
    if ttt_is_full(state, p):
        return TTTOutcome(Scalar[dtype](0.5), Scalar[dtype](1))   # draw

    cell = ttt_random_legal_cell(state, p, u)
    ttt_apply(state, p, cell, CELL_RIVAL)
    if ttt_has_won(state, p, CELL_RIVAL):
        return TTTOutcome(Scalar[dtype](0), Scalar[dtype](1))     # loss
    if ttt_is_full(state, p):
        return TTTOutcome(Scalar[dtype](0.5), Scalar[dtype](1))   # draw
    return TTTOutcome(Scalar[dtype](0), Scalar[dtype](0))         # goes on


def ttt_dynamics_kernel(state: GlobalF32, action: GlobalI32,
                        step_uniforms: GlobalF32, depth: GlobalI32,
                        reward_out: GlobalF32, discount_out: GlobalF32,
                        next_value_out: GlobalF32, n_particles: Int,
                        reward_gamma: Scalar[dtype],
                        loss_penalty: Scalar[dtype]):
    """The stochastic TTT step for the SEARCH: one turn + gamma folded in.

    Advances with `ttt_advance` and translates the result into the SMC core's
    contract: discount (0 if terminal, 1 if it goes on) and next_value (0: no
    critic yet).

    Like the toy problem, it does NOT check whether the particle was already
    dead: a terminal particle gets overwritten and it does not matter, because
    the SMC core already froze its weight. One thread per particle.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p >= n_particles:
        return

    out = ttt_advance(state, p, Int(action[p]), step_uniforms[p])
    # Discount by depth: winning NOW is worth more than winning later. Without
    # this, and with V=0, the SMC weight is the sum of undiscounted rewards, so a
    # particle that wins at step 0 ties with one that wins at step 3 -- and the
    # softmax cannot tell "I win for sure" from "I won by luck".
    g = Scalar[dtype](1)
    for _ in range(Int(depth[p])):
        g *= reward_gamma

    # Loss penalty. Without it, `ttt_advance` returns 0 whether the game is LOST
    # or simply GOES ON, so the SMC weight cannot tell the two apart and the
    # search has no reason whatsoever to block a threat. With loss_penalty > 0 a
    # loss becomes worth -loss_penalty, which is the usual game convention
    # (+1 / 0 / -1). With 0 the original behaviour is recovered.
    r = out.reward
    if loss_penalty != 0 and out.terminal != 0 and out.reward == 0:
        r = -loss_penalty

    reward_out[p] = r * g
    discount_out[p] = Scalar[dtype](1) - out.terminal
    next_value_out[p] = Scalar[dtype](0)


def ttt_seat_opens_first(e: Int) -> Bool:
    """EVEN environments are opened by the agent; ODD ones by the rival.

    The assignment goes by index and not by a draw. With an even number of
    environments, exactly half plays each side in every batch: there is no
    assignment variance adding to the measurement variance, and the two seats can
    be separated when aggregating because the env index says which is which.

    Playing second is NOT the same problem as playing first, and the exact
    references prove it: against the same uniform rival, score-maximising play
    scores 0.9974 opening and 0.9624 answering, and above all loses 0.00% in the
    first case against 0.42% in the second. When answering, maximising the score
    REQUIRES accepting losses; when opening, it does not.
    """
    return e % 2 == 0


def ttt_reset_kernel(state: GlobalF32, n_envs: Int):
    """Starts a new game: empty board, agent opens. One thread per env.

    Fixed-seat variant, kept as is: it is used by the demos and by the tests of
    the earlier phases, whose figures were measured with it. For the final
    comparison `ttt_reset_alt_kernel` is used instead.
    """
    e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e < n_envs:
        for c in range(NUM_CELLS):
            state[e * STATE_DIM + c] = CELL_EMPTY


def ttt_reset_alt_kernel(state: GlobalF32, u_open: GlobalF32, n_envs: Int):
    """Starts a new game with whichever seat the env is assigned.

    If the rival opens, it plays its cell right here. The rest of the system does
    not change at all: `ttt_advance` still does "the agent plays, the rival
    answers", the two-plane encoding already knows how to represent a board with
    a rival mark on it, and neither the search nor the networks notice anything.
    One thread per env.
    """
    e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e < n_envs:
        for c in range(NUM_CELLS):
            state[e * STATE_DIM + c] = CELL_EMPTY
        if not ttt_seat_opens_first(e):
            cell = ttt_random_legal_cell(state, e, u_open[e])
            ttt_apply(state, e, cell, CELL_RIVAL)


def ttt_env_step_kernel(state: GlobalF32, action: GlobalI32,
                        u_rival: GlobalF32, reward_out: GlobalF32,
                        done_out: GlobalI32, n_envs: Int):
    """One turn in the REAL environment: like the search's, but without gamma.

    It shares `ttt_advance` with the search kernel, so the rules are literally
    the same; the only thing that changes is the output: here what matters is
    `done` (to restart the game and count the result) instead of the discount and
    the bootstrap. One thread per env.
    """
    e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e < n_envs:
        out = ttt_advance(state, e, Int(action[e]), u_rival[e])
        reward_out[e] = out.reward
        done_out[e] = Scalar[idx_dtype](1) if out.terminal != Scalar[dtype](0) \
                      else Scalar[idx_dtype](0)


def ttt_auto_reset_kernel(state: GlobalF32, done: GlobalI32, n_envs: Int):
    """Envs that have just finished start a new game; the rest go on.

    It runs AFTER the host has read reward and done: the board is cleared, but
    the game's result is already recorded in its buffers. This is the REAL
    environment's auto-reset; inside the search there is no such thing (a
    terminal particle stays put and its weight is frozen by the terminal mask).
    One thread per env.

    Fixed-seat variant. See `ttt_auto_reset_alt_kernel`.
    """
    e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e < n_envs and Int(done[e]) != 0:
        for c in range(NUM_CELLS):
            state[e * STATE_DIM + c] = CELL_EMPTY


def ttt_auto_reset_alt_kernel(state: GlobalF32, done: GlobalI32,
                              u_open: GlobalF32, n_envs: Int):
    """Auto-reset with seat alternation.

    The env's seat does NOT change between games: env 3 always answers. That is
    what makes the assignment an exact stratification instead of a draw, and what
    lets the two seats be separated when aggregating: it is enough to look at the
    parity of the index. One thread per env.
    """
    e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e < n_envs and Int(done[e]) != 0:
        for c in range(NUM_CELLS):
            state[e * STATE_DIM + c] = CELL_EMPTY
        if not ttt_seat_opens_first(e):
            cell = ttt_random_legal_cell(state, e, u_open[e])
            ttt_apply(state, e, cell, CELL_RIVAL)


def ttt_random_policy_kernel(action_out: GlobalI32, state: GlobalF32,
                             uniforms: GlobalF32, n_envs: Int):
    """The agent's RANDOM policy: one random legal cell per env.

    It is the baseline the search is compared against: if planning does not win
    more games than this, the search is contributing nothing. One thread per env.
    """
    e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e < n_envs:
        action_out[e] = Scalar[idx_dtype](
            ttt_random_legal_cell(state, e, uniforms[e]))


def ttt_encode_obs_kernel(obs_out: GlobalF32, state: GlobalF32, n: Int):
    """Translates the board into the format the network eats: [n, 9] -> [n, OBS_DIM].

    From 9 coded cells (+1 mine / -1 theirs / 0 empty) to two binary planes laid
    one after the other:

        obs[0 .. 8]   1 if the cell is MINE,   0 otherwise
        obs[9 .. 17]  1 if the cell is THEIRS, 0 otherwise

    An empty cell stays 0 in BOTH planes, so "empty" is encoded by absence and no
    third plane is needed.

    Why two planes and not the 9 floats as they are: with a single number the
    network would have to work out on its own that -1, 0 and +1 are three
    CATEGORIES and not a scale with 0 in the middle. And above all, it is the
    encoding pgx uses, which is what the Stoix implementation will see: both
    networks receive the same thing.

    It serves equally for particles (during the search) and for envs (in the real
    loop): only `n` changes. One thread per board.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p >= n:
        return

    base = p * OBS_DIM
    for c in range(NUM_CELLS):
        v = ttt_cell(state, p, c)
        obs_out[base + c] = Scalar[dtype](1) if v == CELL_AGENT \
                            else Scalar[dtype](0)
        obs_out[base + NUM_CELLS + c] = Scalar[dtype](1) if v == CELL_RIVAL \
                                        else Scalar[dtype](0)


def ttt_read_cells_kernel(cells_out: GlobalF32, state: GlobalF32,
                          n_particles: Int):
    """Copies the 9 cells of each particle to the output using ttt_cell.

    It does no game logic yet: it exists to check that the accessor indexes the
    right cell of the right particle (that the STATE_DIM stride is the correct
    one and the particles do not overlap). One thread per particle.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p < n_particles:
        for c in range(NUM_CELLS):
            cells_out[p * NUM_CELLS + c] = ttt_cell(state, p, c)


struct TicTacToe(SearchModel, Copyable, Movable):
    """Tic-Tac-Toe as a search model: the agent (X) against a random rival.

    The rule is fixed and the agent is always X; there is no critic yet (V=0), so
    the search improves a masked uniform prior starting from zero value -- the
    "planner" mode of Milestone 1.

    Like toy_chain, it does not import the search: only the SearchModel contract
    and the types. The kernels it uses (A3a's prior, A3b's dynamics) are tested
    separately; here they are merely wired into the contract's two methods.
    """

    var reward_gamma: Scalar[dtype]
    """Depth discount on the reward. 1.0 = no discount (winning now is worth the
    same as winning later, which in a game with a terminal prize leaves the SMC
    weight nearly blind); <1 rewards winning fast."""

    var loss_penalty: Scalar[dtype]
    """What losing is worth, negated. 0 = the original convention (losing is
    worth 0, the same as not having settled the game); 1 = the game convention
    +1/0/-1. This is reward SHAPING inside the search: the real environment's
    reward, the one the scoreboard counts, is left untouched."""

    def __init__(out self, reward_gamma: Scalar[dtype],
                 loss_penalty: Scalar[dtype] = 0):
        self.reward_gamma = reward_gamma
        self.loss_penalty = loss_penalty

    def eval_root(self, ctx: DeviceContext, cfg: SPOConfig,
                  root_state: DeviceBuffer[dtype],
                  logits_out: DeviceBuffer[dtype],
                  value_out: DeviceBuffer[dtype]) raises:
        """The masked prior over the root states, and V=0 (no critic)."""
        blocks = (cfg.num_envs + TPB_TTT - 1) // TPB_TTT
        ctx.enqueue_function[ttt_prior_logits_kernel, ttt_prior_logits_kernel](
            logits_out.unsafe_ptr(), root_state.unsafe_ptr(), cfg.num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
        # V = 0: planner mode. It is overwritten explicitly because the workspace
        # is reused between searches and could carry stale values.
        value_out.enqueue_fill(0)

    def step(self, ctx: DeviceContext, cfg: SPOConfig, particles: Particles,
             outputs: StepOutputs, step_uniforms: DeviceBuffer[dtype]) raises:
        """Advances the P particles (agent + random rival) and evaluates the prior
        at the NEW state, which is where the next action will be sampled from."""
        p_total = cfg.num_search_particles()
        blocks = (p_total + TPB_TTT - 1) // TPB_TTT

        ctx.enqueue_function[ttt_dynamics_kernel, ttt_dynamics_kernel](
            particles.state.unsafe_ptr(), outputs.next_action.unsafe_ptr(),
            step_uniforms.unsafe_ptr(), particles.depth.unsafe_ptr(),
            outputs.reward.unsafe_ptr(), outputs.discount.unsafe_ptr(),
            outputs.next_value.unsafe_ptr(), p_total, self.reward_gamma,
            self.loss_penalty, grid_dim=blocks, block_dim=TPB_TTT)

        ctx.enqueue_function[ttt_prior_logits_kernel, ttt_prior_logits_kernel](
            outputs.action_logits.unsafe_ptr(), particles.state.unsafe_ptr(),
            p_total, grid_dim=blocks, block_dim=TPB_TTT)


def default_tictactoe() -> TicTacToe:
    """The TTT model **faithful to SPO**: no depth discount.

    This used to return `reward_gamma = 0.7`, and that was wrong as a DEFAULT.
    The 0.7 is a patch of ours from A6 (folding gamma^d into the reward) that
    Stoix does not apply: its `recurrent_fn` passes the environment's raw reward
    (`reward=next_timestep.reward`, ff_spo.py:341). Having it as the default made
    our deviation LOOK LIKE the method.

    And it turned out to be harmful as well: with the actor wired in,
    gamma_r = 1.0 gives 0.62% losses and gamma_r = 0.7 gives 2.65%. The patch
    existed to compensate for the missing policy; once the policy arrives, it
    gets in the way.

    The parameter is kept so that the A6 and E1.11 measurements can be
    reproduced, but it has to be passed by hand and in plain sight.
    """
    return TicTacToe(reward_gamma=1.0)


def ttt_legal_mask_from_obs_kernel(mask_out: GlobalF32, obs: GlobalF32,
                                   n: Int):
    """Legal mask read from the OBSERVATION instead of from the board.

    It exists because the training buffer stores observations (18 floats), not
    states (9). The state could be stored too, but that would duplicate
    information that is already there: in the two-plane encoding a cell is free
    if and only if it is 0 in BOTH.

    That the mask always comes from the same place the network sees matters: if
    it came through a separate channel it could drift out of sync and the actor
    would end up putting probability on marks already placed. One thread per row.
    """
    r = Int(block_dim.x * block_idx.x + thread_idx.x)
    if r >= n:
        return
    for c in range(NUM_CELLS):
        mine = obs[r * OBS_DIM + c]
        theirs = obs[r * OBS_DIM + NUM_CELLS + c]
        mask_out[r * NUM_ACTIONS + c] = Scalar[dtype](1) \
            if (mine == 0 and theirs == 0) else Scalar[dtype](0)
