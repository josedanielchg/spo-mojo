"""Benchmark of the SMC planner on tic-tac-toe, to compare with the MCTS.

    ./run.sh bench/bench_tictactoe.mojo [csv_path]

It plays games against the random rival while timing them, prints the summary with
Wilson intervals and writes a CSV row with the SAME schema as
`MCTS-mojo-tictactoe`, so that the two files can be concatenated.

It measures TWO times, because comparing a batched GPU with a serial CPU using a
single number would be misleading:

  * throughput: planning happens for NUM_ENVS games at once, so the amortised cost
    per decision is the total time divided by all the decisions.
  * latency: the same thing with a SINGLE env, which is what really compares with
    the MCTS's time per decision (it plays the games serially).

The timing excludes allocating the workspace and the initial reset, just as the
MCTS measures "only the play time".
"""

from std.gpu.host import DeviceContext
from std.sys import argv
from std.time import perf_counter_ns

from bench.metrics import PlannerMetrics, write_csv, fmt_fixed
from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform
from envs.tictactoe import (TicTacToe, ttt_reset_kernel, ttt_env_step_kernel,
                            ttt_auto_reset_kernel, NUM_ACTIONS, STATE_DIM,
                            TPB_TTT)
from envs.tictactoe_runner import RNG_RIVAL
from systems.spo.particles import SearchWorkspace
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search

comptime SEED = UInt32(20260724)
comptime NUM_ENVS = 64
comptime NUM_PARTICLES = 64
comptime DEPTH = 6
comptime PERIOD = 3
comptime TEMPERATURE = Scalar[dtype](0.02)
comptime REWARD_GAMMA = Scalar[dtype](0.7)
comptime NUM_STEPS = 60


@fieldwise_init
struct Timed(Movable):
    """A batch's scoreboard and time."""
    var wins: Int
    var draws: Int
    var losses: Int
    var moves: Int
    var decisions: Int
    var seconds: Float64


def run_games(ctx: DeviceContext, num_envs: Int, num_steps: Int) raises -> Timed:
    """Plays with the search and returns scoreboard + play time."""
    cfg = SPOConfig(num_envs=num_envs, num_particles=NUM_PARTICLES,
                    num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                    search_depth=DEPTH, resample_period=PERIOD,
                    temperature=TEMPERATURE, search_gamma=1.0,
                    search_gae_lambda=1.0)
    model = TicTacToe(reward_gamma=REWARD_GAMMA)
    ws = SearchWorkspace(ctx, cfg)
    blocks = (num_envs + TPB_TTT - 1) // TPB_TTT

    state = zero_buffer[dtype](ctx, num_envs * STATE_DIM)
    reward = zero_buffer[dtype](ctx, num_envs)
    done = zero_buffer[idx_dtype](ctx, num_envs)
    u_rival = zero_buffer[dtype](ctx, num_envs)

    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), num_envs, grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()          # the reset does not count as play time

    wins = 0
    draws = 0
    losses = 0
    moves = 0

    start = perf_counter_ns()
    for step in range(num_steps):
        search[TicTacToe](ctx, ws, cfg, model, state,
                          SEED ^ (UInt32(step) * 2654435761))
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), SEED, RNG_RIVAL + UInt32(step), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), ws.output.action.unsafe_ptr(),
            u_rival.unsafe_ptr(), reward.unsafe_ptr(), done.unsafe_ptr(),
            num_envs, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

        with reward.map_to_host() as rh:
            with done.map_to_host() as dh:
                for e in range(num_envs):
                    moves += 1
                    if Int(dh[e]) != 0:
                        r = rh[e]
                        if r > Scalar[dtype](0.75): wins += 1
                        elif r > Scalar[dtype](0.25): draws += 1
                        else: losses += 1

        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()
    seconds = Float64(perf_counter_ns() - start) / 1e9

    # One decision per env and per turn: the search decides for all of them at
    # once.
    return Timed(wins, draws, losses, moves, num_envs * num_steps, seconds)


def main() raises:
    args = argv()
    path = String(args[1]) if len(args) > 1 else String("results/bench_smc.csv")

    with DeviceContext() as ctx:
        print("planificador SMC sobre tres en raya (rival aleatorio)")
        print("  particulas", NUM_PARTICLES, " profundidad", DEPTH,
              " temperatura", TEMPERATURE, " descuento", REWARD_GAMMA)
        print()

        # A warm-up batch: the first call pays for kernel compilation, and that
        # cost is not part of what we want to measure.
        _ = run_games(ctx, NUM_ENVS, 3)

        batched = run_games(ctx, NUM_ENVS, NUM_STEPS)
        games = batched.wins + batched.draws + batched.losses
        steps_per_decision = NUM_PARTICLES * DEPTH

        m = PlannerMetrics(
            mode="smc_vs_random", games=games, iterations=NUM_PARTICLES,
            exploration=Float64(TEMPERATURE), seed=Int(SEED),
            total_runtime_s=batched.seconds, total_moves=batched.moves,
            decisions=batched.decisions,
            total_simulations=batched.decisions * steps_per_decision,
            x_wins=batched.wins, o_wins=batched.losses, draws=batched.draws)
        print(m.summary())

        # And the latency of a single decision, which is what compares with the
        # MCTS.
        single = run_games(ctx, 1, 40)
        latency = single.seconds / Float64(single.decisions)
        print()
        print("--- latencia (1 partida a la vez, comparable con el MCTS) ---")
        print("time/decision   : " + fmt_fixed(latency, 6) + " s")
        print("speedup por lote: x" + fmt_fixed(latency / m.avg_decision_time_s(), 1)
              + "  (planificar " + String(NUM_ENVS) + " partidas a la vez)")

        write_csv(m, path)
        print()
        print("csv escrito en", path)
