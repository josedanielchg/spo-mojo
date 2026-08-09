"""J1/J2: the SPO-Mojo leg of the three-way comparison, with the TRAINED agent.

    ./run.sh bench/bench_agent_tictactoe.mojo [csv_path]

Replaces `bench/bench_tictactoe.mojo` as the comparison leg. That one measures the
**bare planner** (uniform prior, no network), which was all that existed in
Milestone 1; comparing that against two trained agents would be comparing
different things. That one still stands as the "no network" arm of the asymmetry
axis, and that is why this benchmark re-measures it in here, with the SAME
hyperparameters, instead of quoting its old CSV (which was moreover measured with
gamma_r = 0.7 and 64 particles).

**Trains and measures in the same run.** The Mojo implementation does not persist
weights, so there is no checkpoint to load: `train_run` returns the trained actor
precisely so that it can be measured without retraining. The trade-off is that the
agent measured is the one from THIS run; the seed is fixed so that it is
reproducible.

**Four rows, not one.** Two axes the plan asks to keep separate are crossed here:

  * readout protocol: `moda` (argmax of q, comparable with the MCTS's visit argmax
    and with SPO-Stoix's `moda` row) and `muestreada` (the draw from q, which is
    what the Stoix evaluator does);
  * time regime: `lote` (64 games at once: throughput) and `latencia` (a single
    game, which is the only thing comparable with the MCTS, which plays serially).

The 17-column schema has only one time slot, so the regime goes into the `mode`
label instead of a new column -- changing the schema would break concatenation
with the MCTS's CSV, which is already written.
"""

from std.gpu.host import DeviceContext
from std.sys import argv
from std.time import perf_counter_ns

from bench.metrics import PlannerMetrics, write_csv_rows, fmt_fixed
from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform
from envs.tictactoe import (TicTacToe, ttt_reset_alt_kernel, ttt_env_step_kernel,
                            ttt_auto_reset_alt_kernel, ttt_seat_opens_first,
                            NUM_ACTIONS, STATE_DIM, TPB_TTT)
from envs.tictactoe_actor import TicTacToeActor
from envs.tictactoe_runner import RNG_RIVAL
from systems.spo.particles import SearchWorkspace
from systems.spo.readout import readout_expected, readout_greedy, q_histogram
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from demos.train_spo import (train_run, ActorLearner, HIDDEN, SEARCH_DEPTH,
                             TEMPERATURE, NO_RESAMPLE, REWARD_GAMMA,
                             LOSS_PENALTY, NUM_PARTICLES, SEED as TRAIN_SEED)
from demos.train_critic import Critic

# Environment steps per training round: ROLLOUT (16) x NUM_ENVS (32).
comptime STEPS_PER_ROUND = 16 * 32

comptime BENCH_SEED = UInt32(20260805)
comptime BATCH_ENVS = 64
# 360 steps x 64 envs ~ 7000 games: the same sample as the SPO-Stoix leg, so that
# the Wilson intervals are comparable across legs.
comptime BATCH_STEPS = 360
# The latency row exists for the TIME, but its score is written to the CSV all
# the same, so it needs a sample that is not absurd: with 40 steps it produced 13
# games and a score of 1.0000, above the exact ceiling. With 2000 it gives ~650.
comptime LATENCY_STEPS = 2000
comptime RNG_BENCH_READOUT = UInt32(4_000_000)
comptime RNG_OPEN = UInt32(5_000_000)

# Exact ceilings, per seat, computed by recursion over all positions. They
# differ, and that is the point: when opening, maximising the score and never
# losing are the SAME policy; when answering, they are two different policies,
# and the first one accepts 0.42% losses.
comptime OPT_FIRST = 0.9974
comptime OPT_SECOND = 0.9624
comptime OPT_MEAN = 0.9799
comptime RANDOM_MEAN = 0.5000

# Exact references by recursion over all states, playing first.
comptime RANDOM_SCORE = 0.6484
comptime OPTIMAL_SCORE = 0.9974


@fieldwise_init
struct Timed(Movable):
    """Score and time of a campaign, SEPARATED BY SEAT.

    Index 0 denotes the games where the agent opens, index 1 those where the
    opponent opens. The two never mix in a common counter, for a reason that is
    not at all cosmetic: at equal campaign length, the seat where the opponent
    opens produces MORE games, since it consumes a cell and shortens the game.
    Measured: 12,233 games against 14,801. Aggregating the two would therefore
    weight the mean towards that seat --- which gives 0.4845 where the exact value
    of random play is 0.5000. The bias is not noise, and no error bar would flag
    it.

    The mean is therefore computed as the UNWEIGHTED mean of the two scores.
    """
    var wins: InlineArray[Int, 2]
    var draws: InlineArray[Int, 2]
    var losses: InlineArray[Int, 2]
    var envs: InlineArray[Int, 2]
    var moves: Int
    var decisions: Int
    var seconds: Float64

    def part_envs(self, k: Int) -> Float64:
        """Share of the environments holding seat k.

        It is this ratio, and not the ratio of games, that splits the time, the
        decisions and the moves. Each environment decides once per step, whatever
        its seat: the split there is therefore exactly the split of the
        environments. The split of the GAMES is different --- 3,251 against 3,600
        over the same campaign --- because the games where the opponent opens are
        shorter and more of them finish.
        """
        total = self.envs[0] + self.envs[1]
        return Float64(self.envs[k]) / Float64(total if total > 0 else 1)

    def games(self, k: Int) -> Int:
        return self.wins[k] + self.draws[k] + self.losses[k]

    def games_total(self) -> Int:
        return self.games(0) + self.games(1)

    def score(self, k: Int) -> Float64:
        n = self.games(k)
        d = n if n > 0 else 1
        return (Float64(self.wins[k]) + 0.5 * Float64(self.draws[k])) / Float64(d)

    def mean_score(self) -> Float64:
        """UNWEIGHTED mean of the two seats. See the note on the struct."""
        return 0.5 * (self.score(0) + self.score(1))

    def loss_pct(self, k: Int) -> Float64:
        n = self.games(k)
        return 100.0 * Float64(self.losses[k]) / Float64(n if n > 0 else 1)

    def mean_loss_pct(self) -> Float64:
        return 0.5 * (self.loss_pct(0) + self.loss_pct(1))


def play_timed(ctx: DeviceContext, actor: ActorLearner, critic: Critic,
               num_envs: Int, num_steps: Int, particles: Int, use_actor: Bool,
               greedy: Bool, spo_readout: Bool = False,
               gamma_r: Scalar[dtype] = REWARD_GAMMA,
               penalty: Scalar[dtype] = LOSS_PENALTY,
               depth: Int = SEARCH_DEPTH,
               temp: Scalar[dtype] = TEMPERATURE,
               period: Int = NO_RESAMPLE) raises -> Timed:
    """Plays against the random rival, timing the play.

    `use_actor` picks the search prior: the trained network or the uniform one.
    `greedy` picks the protocol: the mode of q or a draw from q.
    `spo_readout` picks WHICH q: the paper's (weighted histogram, equation 6) or
    our variant `readout_expected`, which averages per action before exponentiating.
    `gamma_r` is the depth discount on the reward, which Stoix does NOT have (it
    passes the raw reward). It is a parameter so that the weight-scale hypothesis
    can be falsified: with gamma_r < 1 the winning particles stop tying with one
    another, and since `exp(w/tau)` with small tau behaves like a maximum, the q of
    the faithful readout comes to measure "one of them won VERY early" instead of
    "how many of them win".
    `penalty` puts a loss at -penalty instead of at 0. Stoix does not have it:
    there, what tells a dead particle from a live one is that its bootstrap is 0.
    `depth`, `temp` and `period` are the three search knobs Stoix also exposes
    (`search_depth`, `temperature.fixed_temperature`, `resampling.period`). They
    are parameterised so that the SAME axes can be swept in both implementations
    and the SHAPE of the response compared, which is better evidence of
    equivalence than agreeing at a single point.

    The critic is ALWAYS wired in. This is not an option: SPO's E-step weight is
    `SUM_d r_d + gamma^(d+1) V(s_last) - V(root)`, so without V the particles that
    end on a non-terminal board contribute no value estimate at all. The default of
    `TicTacToeActor` and of `train_run` is `use_critic=False`, which is a VARIANT
    of ours and not SPO; inheriting it silently here left the whole Milestone 4
    comparison measuring a different algorithm from the one it claims to measure.
    """
    cfg = SPOConfig(num_envs=num_envs, num_particles=particles,
                    num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                    search_depth=depth, resample_period=period,
                    temperature=temp, search_gamma=1.0,
                    search_gae_lambda=1.0)
    model = TicTacToe(gamma_r, penalty)
    # `depth_disc` follows the same rule as E2's critic measurement: with
    # gamma_r < 1 the bootstrap is discounted by depth, gamma_r^(d+1) V.
    depth_disc = gamma_r < Scalar[dtype](1)
    amodel = TicTacToeActor(ctx, cfg.num_search_particles(), HIDDEN,
                            gamma_r, penalty, True, depth_disc)
    amodel.sync_from(ctx, actor.net.params)
    amodel.sync_critic_from(ctx, critic.online)
    ws = SearchWorkspace(ctx, cfg)
    blocks = (num_envs + TPB_TTT - 1) // TPB_TTT

    state = zero_buffer[dtype](ctx, num_envs * STATE_DIM)
    reward = zero_buffer[dtype](ctx, num_envs)
    done = zero_buffer[idx_dtype](ctx, num_envs)
    u_rival = zero_buffer[dtype](ctx, num_envs)
    u_readout = zero_buffer[dtype](ctx, num_envs)
    q_buf = zero_buffer[dtype](ctx, num_envs * NUM_ACTIONS)
    logits_buf = zero_buffer[dtype](ctx, num_envs * NUM_ACTIONS)

    u_open = zero_buffer[dtype](ctx, num_envs)
    ctx.enqueue_function[fill_uniform, fill_uniform](
        u_open.unsafe_ptr(), BENCH_SEED, RNG_OPEN, num_envs,
        grid_dim=blocks, block_dim=TPB_TTT)
    ctx.enqueue_function[ttt_reset_alt_kernel, ttt_reset_alt_kernel](
        state.unsafe_ptr(), u_open.unsafe_ptr(), num_envs,
        grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()          # the reset does not count as play time

    var wins = InlineArray[Int, 2](fill=0)
    var draws = InlineArray[Int, 2](fill=0)
    var losses = InlineArray[Int, 2](fill=0)
    var envs = InlineArray[Int, 2](fill=0)
    for e in range(num_envs):
        envs[0 if ttt_seat_opens_first(e) else 1] += 1
    moves = 0

    start = perf_counter_ns()
    for step in range(num_steps):
        sd = BENCH_SEED ^ (UInt32(step) * 2654435761)
        if use_actor:
            search[TicTacToeActor](ctx, ws, cfg, amodel, state, sd)
        else:
            search[TicTacToe](ctx, ws, cfg, model, state, sd)

        # The draw needs fresh numbers; the mode ignores them, but they are filled
        # in all the same so that both arms do the SAME work and the time stays
        # comparable across protocols.
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_readout.unsafe_ptr(), BENCH_SEED,
            RNG_BENCH_READOUT + UInt32(step), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
        if spo_readout:
            if greedy:
                # The mode of the weighted histogram. It only overwrites
                # `output.action`.
                readout_greedy(ctx, ws.particles, ws.output, cfg, q_buf)
            else:
                # The draw from q was already done by `search` (readout_weighted);
                # here q is merely materialised. Overwriting the action would mean
                # drawing twice.
                q_histogram(ctx, ws.particles, ws.output, cfg, q_buf)
        else:
            readout_expected(ctx, ws.particles, ws.output, cfg, logits_buf, q_buf,
                             u_readout, greedy)

        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), BENCH_SEED, RNG_RIVAL + UInt32(step),
            num_envs, grid_dim=blocks, block_dim=TPB_TTT)
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
                        # The seat is read from the parity of the index: that is
                        # the whole point of the stratified split.
                        k = 0 if ttt_seat_opens_first(e) else 1
                        r = rh[e]
                        if r > Scalar[dtype](0.75): wins[k] += 1
                        elif r > Scalar[dtype](0.25): draws[k] += 1
                        else: losses[k] += 1

        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_open.unsafe_ptr(), BENCH_SEED, RNG_OPEN + UInt32(step) + 1,
            num_envs, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_auto_reset_alt_kernel, ttt_auto_reset_alt_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), u_open.unsafe_ptr(), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()
    seconds = Float64(perf_counter_ns() - start) / 1e9

    # One decision per env and per turn: the search decides for all of them at
    # once.
    return Timed(wins, draws, losses, envs, moves, num_envs * num_steps, seconds)


def metrics_of(mode: String, t: Timed, particles: Int, k: Int) -> PlannerMetrics:
    """One CSV row for ONE seat. Never the two aggregated.

    Time, decisions and moves are split pro rata of the seat's ENVIRONMENTS, not
    of its games: each environment decides once per step whatever its seat. The
    two seats run in the same campaign and cannot be timed separately, but that
    split is exact, and the derived quantities (time per decision, simulations per
    second) do not even depend on it since the factor cancels out.
    """
    part = t.part_envs(k)
    return PlannerMetrics(
        mode=mode, games=t.games(k), iterations=particles,
        exploration=Float64(TEMPERATURE), seed=Int(BENCH_SEED),
        total_runtime_s=t.seconds * part,
        total_moves=Int(Float64(t.moves) * part),
        decisions=Int(Float64(t.decisions) * part),
        total_simulations=Int(Float64(t.decisions) * part) * particles * SEARCH_DEPTH,
        x_wins=t.wins[k], o_wins=t.losses[k], draws=t.draws[k])


def two_rows(mut rows: List[PlannerMetrics], mode: String, t: Timed,
                particles: Int) -> None:
    """Appends the rows of a campaign, one per seat.

    A seat with no game is skipped rather than written as zeros. The case arises
    for the latency measurement, which runs on ONE environment: the split being
    made on the parity of the index, a single environment covers only one seat.
    That row exists for the TIME, not for the score, so the limitation is
    inconsequential --- but it must be visible in the CSV rather than hidden
    behind a row of zeros that would look like a total defeat.
    """
    if t.games(0) > 0:
        rows.append(metrics_of(mode + "_1er", t, particles, 0))
    if t.games(1) > 0:
        rows.append(metrics_of(mode + "_2e", t, particles, 1))


def check_ceiling(mode: String, t: Timed) raises:
    """A score above the exact ceiling is not a good result: it is a bug.

    This is the guard that revealed the generator leak in SPO-Stoix (0.9997
    against a ceiling of 0.9974). The sigma is EMPIRICAL and not a worst-case
    bound: near the ceiling almost everything is a win, the real variance drops by
    a factor of a thousand, and the conservative bound would make the guard blind.

    Each seat has ITS OWN ceiling. Opening 0.9974, answering 0.9624: conflating
    them would let an impossible score through on the second seat's side.
    """
    for k in range(2):
        n = t.games(k)
        if n == 0:
            # Seat not covered: this is the normal case with a single
            # environment. Nothing to check, and above all nothing to report as an
            # anomaly.
            continue
        s = t.score(k)
        e_sq = (Float64(t.wins[k]) + 0.25 * Float64(t.draws[k])) / Float64(n)
        variance = e_sq - s * s
        if variance < 0.0:
            variance = 0.0
        se = (variance / Float64(n)) ** 0.5
        ceiling = OPT_FIRST if k == 0 else OPT_SECOND
        if s > ceiling + 4.0 * (se if se > 1e-12 else 1e-12):
            raise Error("`" + mode + "` scores " + fmt_fixed(s, 4)
                        + " on seat " + String(k) + ", above the exact ceiling "
                        + fmt_fixed(ceiling, 4)
                        + ". Impossible when playing cleanly: there is an information "
                        + "leak.")


def pad(s: String, width: Int, left: Bool = False) -> String:
    """Pads to `width`. Mojo has neither ljust nor rjust on String."""
    out = s
    while len(out) < width:
        out = (String(" ") + out) if left else (out + String(" "))
    return out


def show(mode: String, t: Timed) raises:
    """Prints the two seats and their unweighted mean."""
    if t.games(0) == 0 or t.games(1) == 0:
        k = 0 if t.games(0) > 0 else 1
        label = String("1st") if k == 0 else String("2nd")
        print("  " + pad(mode, 28) + " " + label + " only       "
              + fmt_fixed(t.score(k), 4) + "/" + fmt_fixed(t.loss_pct(k), 2) + "%"
              + "   n=" + String(t.games(k)))
        return
    print("  " + pad(mode, 28)
          + " 1st " + fmt_fixed(t.score(0), 4) + "/" + fmt_fixed(t.loss_pct(0), 2) + "%"
          + "  2nd " + fmt_fixed(t.score(1), 4) + "/" + fmt_fixed(t.loss_pct(1), 2) + "%"
          + "  mean " + fmt_fixed(t.mean_score(), 4)
          + "/" + fmt_fixed(t.mean_loss_pct(), 2) + "%"
          + "   n=" + String(t.games_total()))


def two(with_net: Timed, planner_run: Timed) raises -> String:
    """The two columns of a sweep: with and without network, mean over the seats."""
    return ("   with network " + fmt_fixed(with_net.mean_score(), 4)
            + " / " + fmt_fixed(with_net.mean_loss_pct(), 2) + "%"
            + "  without network " + fmt_fixed(planner_run.mean_score(), 4)
            + " / " + fmt_fixed(planner_run.mean_loss_pct(), 2) + "%")


def main() raises:
    args = argv()
    path = String(args[1]) if len(args) > 1 else String("results/bench_spo_mojo.csv")
    # Training budget in ROUNDS. Defaults to the trainer's own; it is passed on the
    # command line so that the 600,000 environment steps used to train the
    # SPO-Stoix leg can be matched (1172 rounds x 512 steps).
    rounds = Int(args[2]) if len(args) > 2 else 30
    # "fiel" = the paper's readout (weighted histogram). Anything else = our
    # variant. ONE ACTOR IS TRAINED PER READOUT: an actor distilled from one
    # readout's q is not the right prior for the other.
    spo_readout = (String(args[3]) == "faithful") if len(args) > 3 else False
    # Particles used for MEASURING. Training carries on with NUM_PARTICLES: raising
    # the search only at evaluation time is legitimate and is what gets compared
    # with the MCTS, which also picks its simulations per move at play time.
    n_part = Int(args[4]) if len(args) > 4 else NUM_PARTICLES
    # Depth discount on the reward. 1.0 = what Stoix does.
    gamma_r = Scalar[dtype](Float64(String(args[5]))) if len(args) > 5 \
              else REWARD_GAMMA
    # Loss penalty. 0.0 = what Stoix does.
    penalty = Scalar[dtype](Float64(String(args[6]))) if len(args) > 6 \
              else LOSS_PENALTY
    # TRAINING seed. Passed on the command line so that the same configuration can
    # be repeated with several seeds and the deviation BETWEEN them reported, which
    # is the uncertainty the Wilson interval does not capture.
    train_seed = UInt32(Int(String(args[7]))) if len(args) > 7 else TRAIN_SEED
    # "cabecera" measures only the comparison rows and skips the sweeps. It is what
    # is needed to repeat the SAME configuration with several seeds: the sweeps do
    # not change the conclusion and would multiply the cost by twenty.
    header_only = (String(args[8]) == "header") if len(args) > 8 else False
    # Resampling period for TRAINING and for the header rows. Defaults to
    # NO_RESAMPLE, which at depth 6 never fires: it is the configuration everything
    # up to here was measured with, and it does not change.
    #
    # It is exposed because the default setup turns resampling off whereas the
    # reference repository's default is 4, and one has to be able to check that the
    # loss floor is not a consequence of having turned it off. Training with
    # resampling and measuring with resampling is the only valid test: evaluating
    # an agent trained without it while resampling is on measures a mismatch, not
    # the effect of resampling.
    resample = Int(args[9]) if len(args) > 9 else NO_RESAMPLE

    with DeviceContext() as ctx:
        print("SPO-Mojo: TRAINED agent on tic-tac-toe (random opponent)")
        print("  particles: trains with", NUM_PARTICLES, " measures with", n_part)
        print("  depth", SEARCH_DEPTH, " temperature", TEMPERATURE)
        print("  gamma_r", gamma_r, " (Stoix passes the raw reward: 1.0)")
        print("  loss_penalty", penalty, " (Stoix does not have it: 0.0)")
        print("  training seed", train_seed)
        print("  exact references: random", RANDOM_SCORE,
              " optimal", OPTIMAL_SCORE)
        print()

        print("=== training (the Mojo implementation does not persist weights) ===")
        print("  rounds", rounds, " = ", rounds * STEPS_PER_ROUND,
              "environment steps")
        train_start = perf_counter_ns()
        print("  readout:", "faithful (weighted histogram)" if spo_readout
              else "variant (mean per action)")
        outcome = train_run(ctx, "spo-mojo", use_actor=True,
                            spo_readout=spo_readout, gamma_r=gamma_r,
                            penalty=penalty, use_critic=True,
                            depth_disc=gamma_r < Scalar[dtype](1),
                            rounds=rounds, seed=train_seed, period=resample)
        train_seconds = Float64(perf_counter_ns() - train_start) / 1e9
        print("  training time: " + fmt_fixed(train_seconds, 1)
              + " s  (paid ONCE; MCTS does not pay it, but pays search on"
              + " every move)")
        print()

        # Warm-up: the first call pays for kernel compilation and that cost is not
        # part of what we want to measure.
        _ = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS, 3, n_part,
                       True, True, spo_readout, gamma_r, penalty,
                       SEARCH_DEPTH, TEMPERATURE, resample)

        rows = List[PlannerMetrics]()

        print("=== batch of", BATCH_ENVS, "games (throughput) ===")
        greedy_run = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS, BATCH_STEPS,
                          n_part, True, True, spo_readout, gamma_r, penalty,
                          SEARCH_DEPTH, TEMPERATURE, resample)
        show("smc_agent_moda_lote", greedy_run)
        check_ceiling("smc_agent_moda_lote", greedy_run)
        two_rows(rows, "smc_agent_moda_lote", greedy_run, n_part)

        sampled_run = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS, BATCH_STEPS,
                                n_part, True, False, spo_readout, gamma_r, penalty,
                                SEARCH_DEPTH, TEMPERATURE, resample)
        show("smc_agent_muestreada_lote", sampled_run)
        check_ceiling("smc_agent_muestreada_lote", sampled_run)
        two_rows(rows, "smc_agent_muestreada_lote", sampled_run, n_part)

        # The arm WITHOUT the network, with the same hyperparameters: it is the
        # comparison term of the asymmetry axis (paying for training once versus
        # paying for search at every move).
        planner_run = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS, BATCH_STEPS,
                             n_part, False, True, spo_readout, gamma_r, penalty,
                             SEARCH_DEPTH, TEMPERATURE, resample)
        show("smc_planificador_moda_lote", planner_run)
        check_ceiling("smc_planificador_moda_lote", planner_run)
        two_rows(rows, "smc_planificador_moda_lote", planner_run, n_part)

        print()
        print("=== one game at a time (latency, comparable with MCTS) ===")
        lat = play_timed(ctx, outcome.actor, outcome.critic, 1, LATENCY_STEPS, n_part,
                         True, True, False, REWARD_GAMMA, LOSS_PENALTY,
                         SEARCH_DEPTH, TEMPERATURE, resample)
        latency = lat.seconds / Float64(lat.decisions)
        show("smc_agent_moda_latencia", lat)
        check_ceiling("smc_agent_moda_latencia", lat)
        two_rows(rows, "smc_agent_moda_latencia", lat, n_part)

        batched_cost = greedy_run.seconds / Float64(greedy_run.decisions)
        print("  latency         : " + fmt_fixed(latency, 6) + " s/decision")
        print("  amortised cost  : " + fmt_fixed(batched_cost, 6) + " s/decision")
        print("  batching gain   : x" + fmt_fixed(latency / batched_cost, 1)
              + "  (" + String(BATCH_ENVS) + " games at a time)")

        if header_only:
            write_csv_rows(rows, path)
            print()
            print("csv written to", path, " (header only)")
            return

        # --- The axis that really discriminates: search budget ---
        #
        # At 128 particles the score SATURATES (the agent and the network-less
        # planner give the same thing), so comparing strength at maximum budget
        # distinguishes nothing. What the learned prior should buy is reaching the
        # same level with LESS search, which is SPO's practical argument against
        # the MCTS: training is paid once and search is paid at every move.
        print()
        print("=== budget curve (same network, different particle counts) ===")
        print("  particles    with network                  without network")
        budgets = List[Int]()
        budgets.append(4)
        budgets.append(8)
        budgets.append(16)
        budgets.append(32)
        budgets.append(64)
        budgets.append(128)
        budgets.append(256)
        budgets.append(512)
        for b in budgets:
            with_net = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS, BATCH_STEPS, b,
                             True, True, spo_readout, gamma_r, penalty,
                             SEARCH_DEPTH, TEMPERATURE, resample)
            without_net = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS, BATCH_STEPS, b,
                             False, True, spo_readout, gamma_r, penalty,
                             SEARCH_DEPTH, TEMPERATURE, resample)
            check_ceiling("presupuesto_con_red_" + String(b), with_net)
            check_ceiling("presupuesto_sin_red_" + String(b), without_net)
            print("  " + pad(String(b), 10, True) + two(with_net, without_net))
            two_rows(rows, "presupuesto_con_red_" + String(b), with_net, b)
            two_rows(rows, "presupuesto_sin_red_" + String(b), without_net, b)

        # --- Sweep of the axes Stoix also exposes ---
        #
        # The agent is held FIXED and only the search moves, which is what can be
        # made identical in both implementations. What gets compared is not the
        # absolute values (they are two different agents) but the SHAPE: where it
        # saturates, where it crosses, in which direction it responds. Two
        # implementations of the same algorithm have to respond alike to the same
        # knobs.
        print()
        print("=== SWEEP: depth (particles fixed at", n_part, ") ===")
        depths = List[Int]()
        depths.append(1); depths.append(2); depths.append(3)
        depths.append(4); depths.append(6); depths.append(8)
        for d in depths:
            c = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS,
                           BATCH_STEPS, n_part, True, True, spo_readout,
                           gamma_r, penalty, d, TEMPERATURE, NO_RESAMPLE)
            sn = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS,
                            BATCH_STEPS, n_part, False, True, spo_readout,
                            gamma_r, penalty, d, TEMPERATURE, NO_RESAMPLE)
            check_ceiling("depth_" + String(d), c)
            check_ceiling("depth_nonet_" + String(d), sn)
            print("  depth       " + pad(String(d), 3, True) + two(c, sn))
            two_rows(rows, "barrido_profundidad_" + String(d), c, n_part)
            two_rows(rows, "barrido_profundidad_sinred_" + String(d),
                        sn, n_part)

        print()
        print("=== SWEEP: temperature ===")
        temps = List[Scalar[dtype]]()
        temps.append(0.005); temps.append(0.02); temps.append(0.1)
        temps.append(0.5); temps.append(1.0)
        for tv in temps:
            c = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS,
                           BATCH_STEPS, n_part, True, True, spo_readout,
                           gamma_r, penalty, SEARCH_DEPTH, tv, NO_RESAMPLE)
            sn = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS,
                            BATCH_STEPS, n_part, False, True, spo_readout,
                            gamma_r, penalty, SEARCH_DEPTH, tv, NO_RESAMPLE)
            check_ceiling("temperature", c)
            check_ceiling("temperature_nonet", sn)
            print("  tau " + fmt_fixed(Float64(tv), 3) + two(c, sn))
            lbl = fmt_fixed(Float64(tv), 3)
            two_rows(rows, "barrido_tau_" + lbl, c, n_part)
            two_rows(rows, "barrido_tau_sinred_" + lbl, sn, n_part)

        print()
        print("=== SWEEP: resampling period (99 = off) ===")
        periods = List[Int]()
        periods.append(1); periods.append(2); periods.append(4)
        periods.append(8); periods.append(99)
        for pr in periods:
            c = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS,
                           BATCH_STEPS, n_part, True, True, spo_readout,
                           gamma_r, penalty, SEARCH_DEPTH, TEMPERATURE, pr)
            sn = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS,
                            BATCH_STEPS, n_part, False, True, spo_readout,
                            gamma_r, penalty, SEARCH_DEPTH, TEMPERATURE, pr)
            check_ceiling("resampling", c)
            check_ceiling("resampling_nonet", sn)
            print("  period  " + pad(String(pr), 3, True) + two(c, sn))
            two_rows(rows, "barrido_remuestreo_" + String(pr), c, n_part)
            two_rows(rows, "barrido_remuestreo_sinred_" + String(pr), sn, n_part)

        write_csv_rows(rows, path)
        print()
        print("csv written to", path)
