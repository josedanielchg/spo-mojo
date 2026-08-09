"""E2.6: the three measurements that close the actor stage.

  1. **The network ALONE**, playing without searching. It is the distillation test:
     did any of the search's knowledge get into the weights? It is compared against
     exact random play (0.6484), computed by recursion.
  2. **The 2x2**: {faithful SPO, corrected} x {without actor, with actor}. The cell
     that demonstrates the EM loop is the second column: the search WITH the
     network beating the search WITHOUT it at the same budget.
  3. **The budget curve**: score against number of particles, with and without the
     network. With the network the same level should be reached sooner -- which is
     SPO's practical argument against MCTS (you pay for training once and then
     search less).

**Why the 2x2 and not a direct comparison.** Comparing "actor + corrected readout"
against "no actor + SPO readout" would mix two changes and the improvement could
not be attributed to either. With the four cells the actor's contribution (within
each row) and the readout's (between rows) get isolated.

**Each setup keeps ITS OWN hyperparameters**, which is the honest thing: SPO's runs
with resampling, gamma_r = 0.7 and no penalty (M1's); the corrected one runs without
resampling, gamma_r = 0.9 and penalty 1 (E1.11c's). It is not a pure readout
ablation, it is "the two systems we have, each at its best setting, with and without
a learned prior". And each trains ITS OWN actor with ITS OWN q: an actor trained
with one readout's q is not the right prior for the other.

Every game is played with the **mode** of q, not with a sample: we are measuring
strength, not exploring.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from bench.metrics import fmt_fixed, wilson_lo, wilson_hi
from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform
from envs.tictactoe import (TicTacToe, ttt_reset_kernel, ttt_env_step_kernel,
                            ttt_auto_reset_kernel, ttt_encode_obs_kernel,
                            NUM_ACTIONS, STATE_DIM, OBS_DIM, TPB_TTT)
from envs.tictactoe_actor import TicTacToeActor
from envs.tictactoe_runner import RNG_RIVAL
from networks.actor import Actor
from systems.spo.launch import TPB, blocks_for
from systems.spo.particles import SearchWorkspace
from systems.spo.readout import (readout_greedy, readout_expected,
                                 argmax_action_kernel)
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from tests.helpers import download
from demos.train_spo import (train_run, ActorLearner, HIDDEN, SEARCH_DEPTH,
                             TEMPERATURE, NO_RESAMPLE)

comptime EVAL_ENVS = 128
comptime EVAL_STEPS = 300
comptime SWEEP_STEPS = 100
comptime EVAL_SEED = UInt32(20260803)

# Exact references, by recursion over all states.
comptime RANDOM_SCORE = 0.6484
comptime OPTIMAL_SCORE = 0.9974

# The two setups, each with its own hyperparameters.
comptime SPO_PERIOD = 3
comptime SPO_GAMMA = Scalar[dtype](0.7)
comptime SPO_PENALTY = Scalar[dtype](0.0)
comptime FIX_GAMMA = Scalar[dtype](0.9)
comptime FIX_PENALTY = Scalar[dtype](1.0)
comptime EVAL_PARTICLES = 128


@fieldwise_init
struct Arm(Copyable, Movable):
    var name: String
    var wins: Int
    var draws: Int
    var losses: Int

    def games(self) -> Int:
        return self.wins + self.draws + self.losses

    def score(self) -> Float64:
        n = self.games()
        return (Float64(self.wins) + 0.5 * Float64(self.draws)) / Float64(n)


def pct(x: Float64) -> String:
    return fmt_fixed(x * 100.0, 2) + "%"


def show(a: Arm) raises:
    n = a.games()
    s = a.score()
    # The CI goes over the SCORE via wins and losses separately: the score is a
    # mean of 1/0.5/0 and its binomial CI does not apply as such, so the losses' CI
    # is reported (the column optimal play pins at 0).
    print("   ", a.name, "  n=", n,
          "  gana ", pct(Float64(a.wins) / Float64(n)),
          "  empata ", pct(Float64(a.draws) / Float64(n)),
          "  PIERDE ", pct(Float64(a.losses) / Float64(n)),
          " IC[", fmt_fixed(wilson_lo(a.losses, n) * 100.0, 2), ",",
          fmt_fixed(wilson_hi(a.losses, n) * 100.0, 2), "]",
          "  score ", fmt_fixed(s, 4))


@fieldwise_init
struct NetArm(Copyable, Movable):
    var arm: Arm
    var illegal: Int
    """How many times the network chose an occupied cell. It HAS to be 0."""


def play_network_only(ctx: DeviceContext, name: String, actor: Actor,
                      steps: Int) raises -> NetArm:
    """The network decides on its own: argmax of pi, with no search at all.

    It is the distillation test. If the network had learned nothing from the
    search, it would play like the initial prior (nearly uniform over the legal
    cells) and would score random play's 0.6484.
    """
    blocks = (EVAL_ENVS + TPB_TTT - 1) // TPB_TTT
    state = zero_buffer[dtype](ctx, EVAL_ENVS * STATE_DIM)
    obs = zero_buffer[dtype](ctx, EVAL_ENVS * OBS_DIM)
    action = zero_buffer[idx_dtype](ctx, EVAL_ENVS)
    reward = zero_buffer[dtype](ctx, EVAL_ENVS)
    done = zero_buffer[idx_dtype](ctx, EVAL_ENVS)
    u_rival = zero_buffer[dtype](ctx, EVAL_ENVS)

    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()

    wins = 0; draws = 0; losses = 0
    illegal = 0
    for step in range(steps):
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            obs.unsafe_ptr(), state.unsafe_ptr(), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        actor.forward(ctx, state, obs, EVAL_ENVS)
        # The move is the argmax of pi. The occupied ones come out at exactly 0,
        # so the argmax can never land on an illegal cell.
        ctx.enqueue_function[argmax_action_kernel, argmax_action_kernel](
            action.unsafe_ptr(), actor.probs.unsafe_ptr(), EVAL_ENVS,
            NUM_ACTIONS, grid_dim=blocks_for(EVAL_ENVS), block_dim=TPB)
        # BEFORE applying it: is it legal? `ttt_apply` does not check (its
        # docstring says so), so an illegal move would overwrite the rival's mark
        # and the score would come out inflated. It is COUNTED rather than assuming
        # the masking is enough.
        ctx.synchronize()
        with state.map_to_host() as sh:
            with action.map_to_host() as ah:
                for e in range(EVAL_ENVS):
                    if sh[e * STATE_DIM + Int(ah[e])] != Scalar[dtype](0):
                        illegal += 1

        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), EVAL_SEED, RNG_RIVAL + UInt32(step),
            EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), action.unsafe_ptr(), u_rival.unsafe_ptr(),
            reward.unsafe_ptr(), done.unsafe_ptr(), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

        with reward.map_to_host() as rh:
            with done.map_to_host() as dh:
                for e in range(EVAL_ENVS):
                    if Int(dh[e]) != 0:
                        r = rh[e]
                        if r > Scalar[dtype](0.75): wins += 1
                        elif r > Scalar[dtype](0.25): draws += 1
                        else: losses += 1
        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()
    return NetArm(Arm(name, wins, draws, losses), illegal)


def diagnose_prior(ctx: DeviceContext, actor: ActorLearner, spo_readout: Bool,
                   particles: Int, steps: Int) raises:
    """Does the learned prior collapse the search's diversity?

    It is the candidate explanation for why criterion 2 fails. If the prior is very
    peaked, `categorical_from_logits` almost always draws the same root action: the
    particles end up exploring ONE move and the search has nothing to compare. It is
    measured by counting how many DISTINCT root actions the particles sample, with
    the actor's prior and with a uniform prior, over THE SAME state and with THE
    SAME seed.
    """
    period = SPO_PERIOD if spo_readout else NO_RESAMPLE
    gamma_r = SPO_GAMMA if spo_readout else FIX_GAMMA
    penalty = SPO_PENALTY if spo_readout else FIX_PENALTY
    cfg = SPOConfig(num_envs=EVAL_ENVS, num_particles=particles,
                    num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                    search_depth=SEARCH_DEPTH, resample_period=period,
                    temperature=TEMPERATURE, search_gamma=1.0,
                    search_gae_lambda=1.0)
    model = TicTacToe(gamma_r, penalty)
    amodel = TicTacToeActor(ctx, cfg.num_search_particles(), HIDDEN, gamma_r,
                            penalty)
    amodel.sync_from(ctx, actor.net.params)
    ws = SearchWorkspace(ctx, cfg)
    q_buf = zero_buffer[dtype](ctx, EVAL_ENVS * NUM_ACTIONS)
    logits_buf = zero_buffer[dtype](ctx, EVAL_ENVS * NUM_ACTIONS)
    u_dummy = zero_buffer[dtype](ctx, EVAL_ENVS)

    blocks = (EVAL_ENVS + TPB_TTT - 1) // TPB_TTT
    state = zero_buffer[dtype](ctx, EVAL_ENVS * STATE_DIM)
    reward = zero_buffer[dtype](ctx, EVAL_ENVS)
    done = zero_buffer[idx_dtype](ctx, EVAL_ENVS)
    u_rival = zero_buffer[dtype](ctx, EVAL_ENVS)
    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()

    sum_a = Scalar[dtype](0); sum_u = Scalar[dtype](0)
    sum_legal = Scalar[dtype](0); count = 0
    # And the other hypothesis, specific to the MEAN readout: the per-action mean
    # has variance inverse to the number of particles sampling it. A peaked prior
    # leaves legal actions with 1 or 2 particles, and their mean is noise -- which
    # then competes on equal terms with that of an action sampled 40 times. SPO's
    # readout does not suffer from this because its sum of exponentials implicitly
    # weights by how many particles there are.
    thin_a = 0; thin_u = 0; min_a = Scalar[dtype](0); min_u = Scalar[dtype](0)

    for step in range(steps):
        sd = EVAL_SEED ^ (UInt32(step) * 2654435761)
        search[TicTacToeActor](ctx, ws, cfg, amodel, state, sd)
        ctx.synchronize()
        roots_a = download[idx_dtype](ws.particles.root_actions,
                                     cfg.num_search_particles())
        search[TicTacToe](ctx, ws, cfg, model, state, sd)
        ctx.synchronize()
        roots_u = download[idx_dtype](ws.particles.root_actions,
                                      cfg.num_search_particles())

        with state.map_to_host() as sh:
            for e in range(EVAL_ENVS):
                seen_a = List[Int](); seen_u = List[Int]()
                for _ in range(NUM_ACTIONS):
                    seen_a.append(0); seen_u.append(0)
                for n in range(particles):
                    seen_a[Int(roots_a[e * particles + n])] = 1
                    seen_u[Int(roots_u[e * particles + n])] = 1
                da = 0; du = 0; legal = 0
                for a in range(NUM_ACTIONS):
                    da += seen_a[a]; du += seen_u[a]
                    if sh[e * STATE_DIM + a] == Scalar[dtype](0):
                        legal += 1
                sum_a += Scalar[dtype](da); sum_u += Scalar[dtype](du)
                sum_legal += Scalar[dtype](legal); count += 1

                # Particle count per action, and how many actions are left
                # "thin" (fewer than 4 particles: their mean is unreliable).
                cnt_a = List[Int](); cnt_u = List[Int]()
                for _ in range(NUM_ACTIONS):
                    cnt_a.append(0); cnt_u.append(0)
                for n in range(particles):
                    cnt_a[Int(roots_a[e * particles + n])] += 1
                    cnt_u[Int(roots_u[e * particles + n])] += 1
                lo_a = particles; lo_u = particles
                for a in range(NUM_ACTIONS):
                    if cnt_a[a] > 0:
                        if cnt_a[a] < 4: thin_a += 1
                        if cnt_a[a] < lo_a: lo_a = cnt_a[a]
                    if cnt_u[a] > 0:
                        if cnt_u[a] < 4: thin_u += 1
                        if cnt_u[a] < lo_u: lo_u = cnt_u[a]
                min_a += Scalar[dtype](lo_a); min_u += Scalar[dtype](lo_u)

        # Advance with the search WITH the actor, so as to walk the positions
        # that agent actually visits.
        search[TicTacToeActor](ctx, ws, cfg, amodel, state, sd)
        if spo_readout:
            readout_greedy(ctx, ws.particles, ws.output, cfg, q_buf)
        else:
            readout_expected(ctx, ws.particles, ws.output, cfg, logits_buf,
                             q_buf, u_dummy, True)
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), EVAL_SEED, RNG_RIVAL + UInt32(step),
            EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), ws.output.action.unsafe_ptr(),
            u_rival.unsafe_ptr(), reward.unsafe_ptr(), done.unsafe_ptr(),
            EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

    c = Scalar[dtype](count)
    print("      acciones raiz DISTINTAS de", particles, "particulas:",
          " con prior ", sum_a / c, " | sin prior ", sum_u / c,
          " | legales ", sum_legal / c)
    print("      particulas de la accion MENOS muestreada:  con prior ",
          min_a / c, " | sin prior ", min_u / c)
    print("      acciones 'flacas' (<4 particulas) por posicion:  con prior ",
          Scalar[dtype](thin_a) / c, " | sin prior ",
          Scalar[dtype](thin_u) / c)


def play_search(ctx: DeviceContext, name: String, actor: ActorLearner,
                use_actor: Bool, spo_readout: Bool, particles: Int,
                steps: Int) raises -> Arm:
    """Plays with the search, with or without the learned prior, with one readout or
    the other.

    Each setup brings its own hyperparameters: SPO's with resampling and
    gamma_r 0.7, the corrected one without resampling, gamma_r 0.9 and penalty 1.
    """
    period = SPO_PERIOD if spo_readout else NO_RESAMPLE
    gamma_r = SPO_GAMMA if spo_readout else FIX_GAMMA
    penalty = SPO_PENALTY if spo_readout else FIX_PENALTY

    cfg = SPOConfig(num_envs=EVAL_ENVS, num_particles=particles,
                    num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                    search_depth=SEARCH_DEPTH, resample_period=period,
                    temperature=TEMPERATURE, search_gamma=1.0,
                    search_gae_lambda=1.0)
    model = TicTacToe(gamma_r, penalty)
    amodel = TicTacToeActor(ctx, cfg.num_search_particles(), HIDDEN, gamma_r,
                            penalty)
    amodel.sync_from(ctx, actor.net.params)
    ws = SearchWorkspace(ctx, cfg)
    q_buf = zero_buffer[dtype](ctx, EVAL_ENVS * NUM_ACTIONS)
    logits_buf = zero_buffer[dtype](ctx, EVAL_ENVS * NUM_ACTIONS)
    u_dummy = zero_buffer[dtype](ctx, EVAL_ENVS)

    blocks = (EVAL_ENVS + TPB_TTT - 1) // TPB_TTT
    state = zero_buffer[dtype](ctx, EVAL_ENVS * STATE_DIM)
    reward = zero_buffer[dtype](ctx, EVAL_ENVS)
    done = zero_buffer[idx_dtype](ctx, EVAL_ENVS)
    u_rival = zero_buffer[dtype](ctx, EVAL_ENVS)
    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()

    wins = 0; draws = 0; losses = 0
    for step in range(steps):
        sd = EVAL_SEED ^ (UInt32(step) * 2654435761)
        if use_actor:
            search[TicTacToeActor](ctx, ws, cfg, amodel, state, sd)
        else:
            search[TicTacToe](ctx, ws, cfg, model, state, sd)
        # The mode of q in both cases: strength is being measured, not exploration.
        if spo_readout:
            readout_greedy(ctx, ws.particles, ws.output, cfg, q_buf)
        else:
            readout_expected(ctx, ws.particles, ws.output, cfg, logits_buf,
                             q_buf, u_dummy, True)

        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), EVAL_SEED, RNG_RIVAL + UInt32(step),
            EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), ws.output.action.unsafe_ptr(),
            u_rival.unsafe_ptr(), reward.unsafe_ptr(), done.unsafe_ptr(),
            EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

        with reward.map_to_host() as rh:
            with done.map_to_host() as dh:
                for e in range(EVAL_ENVS):
                    if Int(dh[e]) != 0:
                        r = rh[e]
                        if r > Scalar[dtype](0.75): wins += 1
                        elif r > Scalar[dtype](0.25): draws += 1
                        else: losses += 1
        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()
    return Arm(name, wins, draws, losses)


def verdict(base: Arm, other: Arm) raises:
    """Compares two arms by their losses, with the usual rule: if the Wilson
    intervals overlap, no difference is claimed."""
    n1 = base.games(); n2 = other.games()
    hi2 = wilson_hi(other.losses, n2); lo1 = wilson_lo(base.losses, n1)
    lo2 = wilson_lo(other.losses, n2); hi1 = wilson_hi(base.losses, n1)
    p1 = Float64(base.losses) / Float64(n1)
    p2 = Float64(other.losses) / Float64(n2)
    print("      derrotas ", pct(p1), " -> ", pct(p2),
          "   score ", fmt_fixed(base.score(), 4), " -> ",
          fmt_fixed(other.score(), 4))
    if hi2 < lo1:
        print("      los IC NO se solapan y el nuevo esta por debajo: pierde menos.")
    elif lo2 > hi1:
        print("      los IC NO se solapan y el nuevo esta por ENCIMA: pierde mas.")
    else:
        print("      los IC se SOLAPAN: no se afirma diferencia en derrotas.")


def main() raises:
    with DeviceContext() as ctx:
        print("=== E2.6: la red sola, el 2x2, y la curva de presupuesto ===")
        print("   evaluacion:", EVAL_ENVS, "partidas a la vez x", EVAL_STEPS,
              "turnos, jugando la MODA de q")
        print("   referencias exactas: azar", RANDOM_SCORE, " optimo",
              OPTIMAL_SCORE, "(este ultimo pierde 0.00%)")
        print()

        # --- train one actor per setup, each with ITS OWN q ---
        print("--- 1. entrenamiento (un actor por montaje) ---")
        fixed = train_run(ctx, String("corregido, bucle EM"), True, False,
                          NO_RESAMPLE, FIX_GAMMA, FIX_PENALTY, EVAL_PARTICLES)
        spo = train_run(ctx, String("SPO fiel, bucle EM"), True, True,
                        SPO_PERIOD, SPO_GAMMA, SPO_PENALTY, EVAL_PARTICLES)

        print("--- 2. la red SOLA (destilacion: sin buscar nada) ---")
        net_fix = play_network_only(ctx, String("red del montaje corregido"),
                                    fixed.actor.net, EVAL_STEPS)
        net_spo = play_network_only(ctx, String("red del montaje SPO      "),
                                    spo.actor.net, EVAL_STEPS)
        show(net_fix.arm)
        show(net_spo.arm)
        print("    jugadas ILEGALES:  corregido ", net_fix.illegal, "  SPO ",
              net_spo.illegal, "   <- tienen que ser 0, o el score esta inflado")
        print("    referencia: jugar al azar da score", RANDOM_SCORE,
              "y pierde 28.81%")
        print()

        print("--- 3. el 2x2 (mismo presupuesto: N =", EVAL_PARTICLES, ") ---")
        a_spo_no = play_search(ctx, String("SPO fiel   sin actor"), spo.actor,
                               False, True, EVAL_PARTICLES, EVAL_STEPS)
        a_spo_yes = play_search(ctx, String("SPO fiel   CON actor"), spo.actor,
                                True, True, EVAL_PARTICLES, EVAL_STEPS)
        a_fix_no = play_search(ctx, String("corregido  sin actor"), fixed.actor,
                               False, False, EVAL_PARTICLES, EVAL_STEPS)
        a_fix_yes = play_search(ctx, String("corregido  CON actor"), fixed.actor,
                                True, False, EVAL_PARTICLES, EVAL_STEPS)
        show(a_spo_no); show(a_spo_yes); show(a_fix_no); show(a_fix_yes)
        print()
        print("--- 4. ¿la red mejora la busqueda que la entreno? ---")
        print("    (es la celda que demuestra el bucle EM)")
        print("    SPO fiel:")
        verdict(a_spo_no, a_spo_yes)
        print("    corregido:")
        verdict(a_fix_no, a_fix_yes)
        print()
        print("--- 4b. ¿por que el prior no ayuda a presupuesto alto? ---")
        print("    Candidata: un prior picudo colapsa la diversidad de acciones")
        print("    raiz, y la busqueda se queda sin nada que comparar.")
        print("    corregido:")
        diagnose_prior(ctx, fixed.actor, False, EVAL_PARTICLES, 40)
        print("    SPO fiel:")
        diagnose_prior(ctx, spo.actor, True, EVAL_PARTICLES, 40)
        print()

        print("--- 5. curva de presupuesto (montaje corregido) ---")
        print("    nuestro barrido sin red ya medido: 4->0.865  16->0.953  64->0.969")
        ns = List[Int]()
        ns.append(4); ns.append(16); ns.append(64); ns.append(128)
        for i in range(len(ns)):
            no = play_search(ctx, String("N=", ns[i], " sin actor"),
                             fixed.actor, False, False, ns[i], SWEEP_STEPS)
            yes = play_search(ctx, String("N=", ns[i], " CON actor"),
                              fixed.actor, True, False, ns[i], SWEEP_STEPS)
            print("    N=", ns[i], "  sin red score ",
                  fmt_fixed(no.score(), 4), " (pierde ",
                  pct(Float64(no.losses) / Float64(no.games())), ")",
                  "   con red score ", fmt_fixed(yes.score(), 4),
                  " (pierde ", pct(Float64(yes.losses) / Float64(yes.games())),
                  ")")
        print()
        print("    Si con red se alcanza antes el mismo nivel, ese es el argumento")
        print("    practico de SPO frente a MCTS: el entrenamiento se paga una vez")
        print("    y luego cada decision cuesta menos busqueda.")
