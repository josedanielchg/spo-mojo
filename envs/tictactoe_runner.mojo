"""The harness that plays real games and keeps score.

Kept separate from `tictactoe.mojo` on purpose: the RULES live there (board, turn,
reset) and here lives whoever runs them in a loop and counts the results. The same
split the CartPole runner had.

For now only the BASELINE: the agent plays at random. It is the number the search
is compared against in phase A6 -- if planning does not win more games than
throwing a cell at random, the search is contributing nothing.

The tracking happens on the HOST: every turn reward and done are brought down to
the CPU and the finished games are recorded. It is host<->device traffic per step,
but this is not the hot loop (the M-step's learner will be), so clarity wins here.
Moving it to device is left as a future improvement.
"""

from std.gpu.host import DeviceContext

from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform
from envs.tictactoe import (ttt_reset_kernel, ttt_env_step_kernel,
                            ttt_auto_reset_kernel, ttt_random_policy_kernel,
                            STATE_DIM, TPB_TTT)

# RNG streams of the real environment. They live in high ranges so as not to cross
# the search's (root 0, actions 100+d, step 500+d, resampling 900+d, readout 7777),
# that way the games and the planning share no sequence even when they share a
# seed.
# Random-number streams at the ENVIRONMENT level. Each one is offset by a step
# counter which, over a long training run, reaches several tens of thousands: with
# 16 steps per round and 1172 rounds, it climbs to 18,752. The streams therefore
# have to be spaced far more than that, otherwise two of them end up drawing
# EXACTLY the same numbers.
#
# This is not hypothetical: spaced by 10,000, `RNG_RIVAL` and `RNG_READOUT`
# overlapped from round 625 onwards, and the rival then replayed the draws the
# readout had already used. It is the same fault as the one found in the reference
# implementation -- two independent uses sharing a source. A spacing of 10^6 leaves
# room for about fifty times more rounds than the longest campaign run here.
comptime RNG_POLICY = UInt32(1_000_000)
comptime RNG_RIVAL = UInt32(2_000_000)


@fieldwise_init
struct MatchStats(Copyable, Movable):
    """The scoreboard of a batch of games, from the agent's (X) point of view."""

    var wins: Int
    var draws: Int
    var losses: Int

    def games(self) -> Int:
        return self.wins + self.draws + self.losses

    def win_rate(self) -> Scalar[dtype]:
        """Fraction of games won. 0 if none was completed."""
        n = self.games()
        if n == 0:
            return Scalar[dtype](0)
        return Scalar[dtype](self.wins) / Scalar[dtype](n)

    def score(self) -> Scalar[dtype]:
        """Mean score per game: 1 win, 0.5 draw, 0 loss.

        It summarises the three figures in one number, and it is exactly the mean
        reward the agent sees, so it is what gets compared against the search.
        """
        n = self.games()
        if n == 0:
            return Scalar[dtype](0)
        return (Scalar[dtype](self.wins) + Scalar[dtype](0.5) * Scalar[dtype](self.draws)) \
               / Scalar[dtype](n)


def play_random_games(ctx: DeviceContext, num_envs: Int, num_steps: Int,
                      seed: UInt32) raises -> MatchStats:
    """Plays `num_steps` turns across `num_envs` parallel games, at random.

    The agent picks a free cell at random and so does the rival. Every game that
    finishes is recorded and the env starts another (auto-reset), so in
    `num_steps` turns roughly num_envs*num_steps/3.5 complete games come out.

    It returns the scoreboard. Games left unfinished when the turns run out are
    not counted; that slightly under-represents long games (one is left unfinished
    per env), so it is worth using plenty of turns for the bias to be negligible.
    """
    blocks = (num_envs + TPB_TTT - 1) // TPB_TTT

    state = zero_buffer[dtype](ctx, num_envs * STATE_DIM)
    action = zero_buffer[idx_dtype](ctx, num_envs)
    reward = zero_buffer[dtype](ctx, num_envs)
    done = zero_buffer[idx_dtype](ctx, num_envs)
    u_policy = zero_buffer[dtype](ctx, num_envs)
    u_rival = zero_buffer[dtype](ctx, num_envs)

    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), num_envs, grid_dim=blocks, block_dim=TPB_TTT)

    wins = 0
    draws = 0
    losses = 0

    for step in range(num_steps):
        # The agent's move: a random free cell.
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_policy.unsafe_ptr(), seed, RNG_POLICY + UInt32(step), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_random_policy_kernel, ttt_random_policy_kernel](
            action.unsafe_ptr(), state.unsafe_ptr(), u_policy.unsafe_ptr(),
            num_envs, grid_dim=blocks, block_dim=TPB_TTT)

        # The full turn (the rival answers inside the kernel).
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), seed, RNG_RIVAL + UInt32(step), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), action.unsafe_ptr(), u_rival.unsafe_ptr(),
            reward.unsafe_ptr(), done.unsafe_ptr(), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

        # Record the ones that finished. The reward tells the result: 1 / 0.5 / 0.
        with reward.map_to_host() as rh:
            with done.map_to_host() as dh:
                for e in range(num_envs):
                    if Int(dh[e]) != 0:
                        r = rh[e]
                        if r > Scalar[dtype](0.75):
                            wins += 1
                        elif r > Scalar[dtype](0.25):
                            draws += 1
                        else:
                            losses += 1

        # And the finished games start another one. It runs AFTER reading the
        # result.
        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)

    return MatchStats(wins, draws, losses)


def main() raises:
    """Runs the baseline and prints it, so it can be eyeballed.

    Exact reference (computed by recursion over all states, with both sides
    playing uniformly at random): X wins 58.49%, draws 12.70% and loses 28.81%,
    that is, a mean score of 0.6484. If these figures come out, the whole loop --
    turn, end detection, counting and auto-reset -- is right.
    """
    with DeviceContext() as ctx:
        stats = play_random_games(ctx, 64, 200, UInt32(12345))
        print("linea base (agente al azar vs rival al azar):")
        print("  partidas :", stats.games())
        print("  gana X   :", stats.wins, "(", stats.win_rate(), ")")
        print("  empate   :", stats.draws)
        print("  gana O   :", stats.losses)
        print("  score    :", stats.score(), " (exacto: 0.6484 )")
