"""The REAL Tic-Tac-Toe environment: reset, a real turn and auto-reset.

There is no search and no particles here: these are real games, one board per env.
The difference from the search's step is what comes out (`done` instead of discount
and bootstrap) and that finished games reset themselves.

The rules are not re-tested: the search kernel and the environment kernel share
`ttt_advance`, and the rules already have their tests in test_tictactoe.mojo. What
gets tested here is what only exists in the real environment.
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
    """A readable board: 9 cell codes (1=X agent, -1=O rival, 0=empty)."""
    out = List[Scalar[dtype]]()
    out.append(Scalar[dtype](c0)); out.append(Scalar[dtype](c1))
    out.append(Scalar[dtype](c2)); out.append(Scalar[dtype](c3))
    out.append(Scalar[dtype](c4)); out.append(Scalar[dtype](c5))
    out.append(Scalar[dtype](c6)); out.append(Scalar[dtype](c7))
    out.append(Scalar[dtype](c8))
    return out^


def flatten(boards: List[List[Scalar[dtype]]]) -> List[Scalar[dtype]]:
    """Several boards laid end to end, as the device expects them."""
    out = List[Scalar[dtype]]()
    for i in range(len(boards)):
        for c in range(NUM_CELLS):
            out.append(boards[i][c])
    return out^


def test_reset_clears_the_board(ctx: DeviceContext) raises:
    """Reset leaves the board empty, whatever it came from."""
    boards = List[List[Scalar[dtype]]]()
    boards.append(board9(1,-1,1, -1,1,-1, 1,-1,1))   # full
    boards.append(board9(1,0,-1, 0,1,0, -1,0,0))     # mid-game
    n = len(boards)

    state = upload[dtype](ctx, flatten(boards))
    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), n, grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()

    got = download[dtype](state, n * NUM_CELLS)
    for i in range(n * NUM_CELLS):
        assert_close(got[i], CELL_EMPTY, TOL, String("after the reset, cell ", i))
    print("PASS reset leaves the board empty")


def test_env_step_reports_done(ctx: DeviceContext) raises:
    """A real turn: the agent's reward and `done` across the four endings.

    The same scenarios as the search's step, but looking at `done`: here what
    matters is whether the game ended, so as to count it and restart it.
    """
    boards = List[List[Scalar[dtype]]]()
    acts = List[Scalar[idx_dtype]]()
    us = List[Scalar[dtype]]()
    exp_reward = List[Scalar[dtype]]()
    exp_done = List[Int]()
    names = List[String]()

    # The agent wins by completing row 0.
    boards.append(board9(1,1,0, -1,-1,0, 0,0,0)); acts.append(Scalar[idx_dtype](2))
    us.append(Scalar[dtype](0.5)); exp_reward.append(Scalar[dtype](1))
    exp_done.append(1); names.append("agent wins")
    # Draw: the agent's move fills the board with no line.
    boards.append(board9(1,-1,1, 1,-1,-1, -1,1,0)); acts.append(Scalar[idx_dtype](8))
    us.append(Scalar[dtype](0.5)); exp_reward.append(Scalar[dtype](0.5))
    exp_done.append(1); names.append("draw on filling")
    # The rival wins: only cell 6 is left and with it it completes column 0.
    boards.append(board9(-1,1,0, -1,1,-1, 0,-1,1)); acts.append(Scalar[idx_dtype](2))
    us.append(Scalar[dtype](0.5)); exp_reward.append(Scalar[dtype](0))
    exp_done.append(1); names.append("opponent wins")
    # The game goes on: there are gaps left and nobody has won.
    boards.append(board9(1,-1,1, -1,1,0, -1,0,0)); acts.append(Scalar[idx_dtype](5))
    us.append(Scalar[dtype](0.1)); exp_reward.append(Scalar[dtype](0))
    exp_done.append(0); names.append("sigue")

    n = len(boards)
    state = upload[dtype](ctx, flatten(boards))
    action = upload[idx_dtype](ctx, acts)
    u = upload[dtype](ctx, us)
    reward = zeros[dtype](ctx, n)
    done = filled[idx_dtype](ctx, n, Scalar[idx_dtype](-1))   # -1 = not written

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
    print("PASS one real turn: reward and done on all four endings")


def test_auto_reset_only_finished_games(ctx: DeviceContext) raises:
    """Only the envs with done=1 start over; the rest carry on with their game."""
    boards = List[List[Scalar[dtype]]]()
    boards.append(board9(1,1,1, -1,-1,0, 0,0,0))   # [0] ended  -> gets cleared
    boards.append(board9(1,0,-1, 0,1,0, -1,0,0))   # [1] carries on -> untouched
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
                     String("the env that ended should be empty, cell ", c))
        assert_close(got[NUM_CELLS + c], boards[1][c], TOL,
                     String("the env that continues should not be touched, cell ", c))
    print("PASS auto-reset: restarts only the finished games")


def test_random_policy_only_legal(ctx: DeviceContext) raises:
    """The random policy (the baseline) never plays on an occupied cell.

    It sweeps uniforms from 0 to ~1 over the same board so as to cover every free
    cell it can choose, and also checks that it does choose more than one (if it
    always returned the same one, it would be legal but not random).
    """
    b = board9(1,0,-1, 0,1,0, -1,0,0)     # free: 1, 3, 5, 7, 8
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
            raise Error("action out of range with u=", us[i], ": ", a)
        if b[a] != CELL_EMPTY:
            raise Error("the random policy chose the OCCUPIED cell ", a)
        seen = False
        for j in range(len(distinct)):
            if distinct[j] == a:
                seen = True
        if not seen:
            distinct.append(a)

    if len(distinct) < 2:
        raise Error("the random policy always returned the same cell")
    print("PASS the random policy only plays free cells (", len(distinct),
          "distinct )")


def test_random_baseline_matches_exact_odds(ctx: DeviceContext) raises:
    """THE loop's test: the baseline reproduces the EXACT probabilities.

    With both sides playing uniformly at random, TTT's probabilities can be
    computed exactly by recursion over all states (it is neither an estimate nor a
    number from the literature):

        X (agent) wins  0.5849      draw  0.1270      O wins  0.2881
        agent's mean score = 0.5849 + 0.5*0.1270 = 0.6484

    Those three figures coming out validates the WHOLE loop at once: the turn, the
    random rival, the end detection, the result classification and the auto-reset.
    A fault in any of those pieces would move the proportions.

    The tolerance is 4 percentage points: with ~3000 games the standard error is
    around 0.9%, so 4 points is more than 4 sigma -- generous enough not to fail on
    noise with another seed, and still tight enough to catch any real bug.
    """
    stats = play_random_games(ctx, 64, 200, UInt32(12345))
    n = stats.games()
    if n < 2000:
        raise Error("thousands of games were expected in 200 turns, got ", n)

    draw_rate = Scalar[dtype](stats.draws) / Scalar[dtype](n)
    loss_rate = Scalar[dtype](stats.losses) / Scalar[dtype](n)
    tol = Scalar[dtype](0.04)

    print("      games:", n, " X wins:", stats.win_rate(),
          " draw:", draw_rate, " O wins:", loss_rate,
          " score:", stats.score())

    if abs(stats.win_rate() - Scalar[dtype](0.5849)) > tol:
        raise Error("the win rate strays from the exact value 0.5849: ",
                    stats.win_rate())
    if abs(draw_rate - Scalar[dtype](0.1270)) > tol:
        raise Error("the draw rate strays from the exact value 0.1270: ", draw_rate)
    if abs(loss_rate - Scalar[dtype](0.2881)) > tol:
        raise Error("the loss rate strays from the exact value 0.2881: ", loss_rate)
    if abs(stats.score() - Scalar[dtype](0.6484)) > tol:
        raise Error("the score strays from the exact value 0.6484: ", stats.score())

    # And the three categories have to add up exactly to the games counted.
    assert_eq_int(stats.wins + stats.draws + stats.losses, n,
                  "wins + draws + losses should be the total")
    print("PASS the baseline reproduces the exact probabilities of random TTT")


def test_baseline_is_reproducible(ctx: DeviceContext) raises:
    """Same seed, same scoreboard. Without this nothing can be compared."""
    a = play_random_games(ctx, 32, 60, UInt32(777))
    b = play_random_games(ctx, 32, 60, UInt32(777))
    assert_eq_int(a.wins, b.wins, "the wins should repeat with the same seed")
    assert_eq_int(a.draws, b.draws, "the draws should repeat with the same seed")
    assert_eq_int(a.losses, b.losses, "the losses should repeat with the same seed")

    # And with another seed the scoreboard has to change (otherwise the seed is
    # not being used).
    c = play_random_games(ctx, 32, 60, UInt32(4242))
    if c.wins == a.wins and c.draws == a.draws and c.losses == a.losses:
        raise Error("two different seeds gave exactly the same scoreboard")
    print("PASS the baseline is reproducible and depends on the seed")


def main() raises:
    with DeviceContext() as ctx:
        test_reset_clears_the_board(ctx)
        test_env_step_reports_done(ctx)
        test_auto_reset_only_finished_games(ctx)
        test_random_policy_only_legal(ctx)
        test_random_baseline_matches_exact_odds(ctx)
        test_baseline_is_reproducible(ctx)
