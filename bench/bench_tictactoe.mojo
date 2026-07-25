"""Benchmark del planificador SMC en tres en raya, para comparar con el MCTS.

    ./run.sh bench/bench_tictactoe.mojo [ruta_csv]

Corre partidas contra el rival aleatorio midiendo el tiempo, imprime el resumen con
intervalos de Wilson y escribe una fila CSV con el MISMO esquema que
`MCTS-mojo-tictactoe`, para poder concatenar los dos ficheros.

Mide DOS tiempos, porque comparar una GPU por lotes con una CPU en serie con un solo
numero seria enganoso:

  * throughput: se planifica para NUM_ENVS partidas a la vez, asi que el coste por
    decision repartido es el tiempo total entre todas las decisiones.
  * latencia: lo mismo con UN solo env, que es lo que de verdad se compara con el
    tiempo por decision del MCTS (que juega las partidas en serie).

El tiempo excluye la reserva del workspace y el reset inicial, igual que el MCTS
mide "only the play time".
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
    """Marcador y tiempo de una tanda."""
    var wins: Int
    var draws: Int
    var losses: Int
    var moves: Int
    var decisions: Int
    var seconds: Float64


def run_games(ctx: DeviceContext, num_envs: Int, num_steps: Int) raises -> Timed:
    """Juega con la busqueda y devuelve marcador + tiempo de juego."""
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
    ctx.synchronize()          # el reset no cuenta como tiempo de juego

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

    # Una decision por env y por turno: la busqueda decide para todos a la vez.
    return Timed(wins, draws, losses, moves, num_envs * num_steps, seconds)


def main() raises:
    args = argv()
    path = String(args[1]) if len(args) > 1 else String("results/bench_smc.csv")

    with DeviceContext() as ctx:
        print("planificador SMC sobre tres en raya (rival aleatorio)")
        print("  particulas", NUM_PARTICLES, " profundidad", DEPTH,
              " temperatura", TEMPERATURE, " descuento", REWARD_GAMMA)
        print()

        # Una tanda de calentamiento: la primera llamada paga la compilacion de los
        # kernels, y ese coste no es parte de lo que se quiere medir.
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

        # Y la latencia de una decision suelta, que es lo comparable con el MCTS.
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
