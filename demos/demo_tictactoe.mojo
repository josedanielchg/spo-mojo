"""Demo 2: the SMC search plays tic-tac-toe.

The question that settles phase A: does planning play better than throwing at
random? There is no trained network -- the prior is uniform over the legal cells
and V=0, so all the improvement comes from simulating games forward.

    ./run.sh demos/demo_tictactoe.mojo

The two references are EXACT, not estimated: in such a small game they are computed
by recursion over all states (`ttt_exact.py` / `ttt_optimal.py`).

    floor    random X    vs random O : wins 58.49%  draws 12.70%  loses 28.81%  score 0.6484
    ceiling  optimal X   vs random O : wins 99.48%  draws  0.52%  loses  0.00%  score 0.9974
    here     X = search  vs random O : wins 96.80%  draws  1.18%  loses  2.02%  score 0.9739

That is 1.50x random play, and 97.6% of the achievable maximum. Having the ceiling
matters: it says how much headroom really remains (little) and gives the reference
to compare the MCTS against. The most informative figure is the LOSSES one, because
optimal play NEVER loses: that 2% is rival threats the search did not block.

It prints four things:
  1. baseline vs search       the main result
  2. discount sweep           why the reward has to be discounted
  3. depth sweep              how much simulating more turns helps
  4. particle sweep           how much simulating more variants helps

The two findings it took to get here, both found with this demo (it used to give
0.997x, that is, the same as random):

  * A core bug: `root_fn` did not zero the accumulators, so on reusing the
    workspace the second search inherited `terminal = 1` and the mask froze ALL the
    weights. The search degenerated into choosing at random without failing at
    anything. It has a regression test in test_search.mojo.
  * The depth discount (`reward_gamma`): without it, winning on the first turn is
    worth the same as winning on the fourth, and the softmax cannot tell "I win for
    sure" from "I won by luck". It shows up in sweep 2: gamma=1 gives 0.81 and
    gamma=0.7 gives 0.97.

The parameters come from sweeping each one while measuring games: the temperature
matters most (0.5 -> 0.78, 0.1 -> 0.94, 0.02 -> 0.974, and there it saturates), the
discount has a flat valley between 0.5 and 0.7, and the resampling period is barely
noticeable.

A note on scale for reading the depth sweep: each model step is a full turn (agent
+ rival) and a game lasts at most 5 agent moves, so the improvement saturates as
soon as the depth covers the end of the game.
"""

from std.gpu.host import DeviceContext

from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform
from envs.tictactoe import (TicTacToe, ttt_reset_kernel,
                            ttt_env_step_kernel, ttt_auto_reset_kernel,
                            NUM_ACTIONS, STATE_DIM, TPB_TTT)
from envs.tictactoe_runner import play_random_games, MatchStats, RNG_RIVAL
from systems.spo.particles import SearchWorkspace
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search

comptime SEED = UInt32(20260724)
comptime NUM_ENVS = 64
"""Games in parallel. Each one is independent, so more envs = more games per turn
and a more stable measurement."""

comptime EXACT_RANDOM_SCORE = Scalar[dtype](0.6484)
"""The random agent's score, computed exactly (not measured)."""


def search_config(num_particles: Int, depth: Int, period: Int,
                  temperature: Scalar[dtype]) -> SPOConfig:
    """Search config for TTT.

    `search_gamma` stays at 1 because it only multiplies the bootstrap `V(s')`, and
    here V is 0: it has no effect at all. The discount that DOES matter is the
    model's (`reward_gamma`), which is applied to the reward by depth.
    """
    return SPOConfig(
        num_envs=NUM_ENVS, num_particles=num_particles, num_actions=NUM_ACTIONS,
        state_dim=STATE_DIM, search_depth=depth, resample_period=period,
        temperature=temperature, search_gamma=1.0, search_gae_lambda=1.0)


def play_search_games(ctx: DeviceContext, cfg: SPOConfig, model: TicTacToe,
                      num_steps: Int, seed: UInt32) raises -> MatchStats:
    """Plays real games using the SEARCH as the policy.

    On every turn: plan from the real state (one full search per env), execute the
    chosen action, record the game if it ended and reset it.

    It lives in demos/ and not in envs/ on purpose: this imports the search
    algorithm, and the project's rule is that an environment never depends on the
    algorithm. The application level (a demo) may depend on both.

    The `SearchWorkspace` is allocated ONCE outside the loop: here the search runs
    every turn, so allocating its buffers each time would be pure work for nothing.
    """
    ws = SearchWorkspace(ctx, cfg)
    num_envs = cfg.num_envs
    blocks = (num_envs + TPB_TTT - 1) // TPB_TTT

    state = zero_buffer[dtype](ctx, num_envs * STATE_DIM)
    reward = zero_buffer[dtype](ctx, num_envs)
    done = zero_buffer[idx_dtype](ctx, num_envs)
    u_rival = zero_buffer[dtype](ctx, num_envs)

    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), num_envs, grid_dim=blocks, block_dim=TPB_TTT)

    wins = 0
    draws = 0
    losses = 0

    for step in range(num_steps):
        # Plan from the real state. The search copies the board to its particles,
        # so it does not touch the environment's state.
        # The seed changes per turn (with a large odd multiplier) so that two turns
        # from the same board do not draw exactly the same variants.
        search[TicTacToe](ctx, ws, cfg, model, state,
                          seed ^ (UInt32(step) * 2654435761))

        # And execute the chosen action in the real environment.
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), seed, RNG_RIVAL + UInt32(step), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), ws.output.action.unsafe_ptr(),
            u_rival.unsafe_ptr(), reward.unsafe_ptr(), done.unsafe_ptr(),
            num_envs, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

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

        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)

    return MatchStats(wins, draws, losses)


def show(label: String, s: MatchStats) raises:
    """One scoreboard line, with the improvement over exact random play."""
    print("  ", label,
          "  partidas", s.games(),
          " gana", s.win_rate(),
          " empata", Scalar[dtype](s.draws) / Scalar[dtype](s.games()),
          " pierde", Scalar[dtype](s.losses) / Scalar[dtype](s.games()),
          " score", s.score(),
          " (x", s.score() / EXACT_RANDOM_SCORE, "vs azar )")


def main() raises:
    with DeviceContext() as ctx:
        print("=== 1. linea base vs busqueda ===")
        base = play_random_games(ctx, NUM_ENVS, 200, SEED)
        show("azar     ", base)
        print("            (exacto: gana 0.5849  empata 0.1270  pierde 0.2881  score 0.6484 )")

        cfg = search_config(num_particles=64, depth=6, period=3, temperature=0.02)
        srch = play_search_games(ctx, cfg, TicTacToe(reward_gamma=0.7), 60, SEED)
        show("busqueda ", srch)

        print()
        print("=== 2. barrido del descuento de recompensa (N=64, depth=6, temp=0.02) ===")
        print("    gamma=1 deja el peso SMC casi ciego: ganar ya vale igual que ganar despues")
        gammas = List[Scalar[dtype]]()
        gammas.append(1.0); gammas.append(0.9); gammas.append(0.7); gammas.append(0.5)
        for i in range(len(gammas)):
            c = search_config(num_particles=64, depth=6, period=3, temperature=0.02)
            s = play_search_games(ctx, c, TicTacToe(reward_gamma=gammas[i]), 60, SEED)
            show(String("gamma ", gammas[i], "  "), s)

        print()
        print("=== 3. barrido de profundidad (N=64, gamma=0.7) ===")
        depths = List[Int]()
        depths.append(1); depths.append(2); depths.append(3)
        depths.append(5); depths.append(8)
        for i in range(len(depths)):
            d = depths[i]
            c = search_config(num_particles=64, depth=d, period=3, temperature=0.02)
            s = play_search_games(ctx, c, TicTacToe(reward_gamma=0.7), 60, SEED)
            show(String("depth ", d, "  "), s)

        print()
        print("=== 4. barrido de particulas (depth=6, gamma=0.7) ===")
        counts = List[Int]()
        counts.append(4); counts.append(16); counts.append(64); counts.append(128)
        for i in range(len(counts)):
            n = counts[i]
            c = search_config(num_particles=n, depth=6, period=3, temperature=0.02)
            s = play_search_games(ctx, c, TicTacToe(reward_gamma=0.7), 60, SEED)
            show(String("N = ", n, "  "), s)
