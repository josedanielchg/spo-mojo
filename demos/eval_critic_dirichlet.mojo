"""Stage 2 review: the critic reconnected and the Dirichlet noise.

They come out of two decisions taken while reviewing the stage:

**1. Reconnecting the critic.** V is in equation 10 of the paper

    A(s_t,a_t) = r_t + V(s_{t+1}) - V(s_t)

and in Stoix's `_critic_loss_fn`. Holding it at 0 was OUR deviation, not a choice
of the method. E1.11 measured that it did not help, but it measured it with a
critic trained on data from a planner that lost 2% and started from a uniform
prior. Now the data comes from a much stronger agent that visits other positions,
so that measurement does not apply and has to be repeated.

**2. Switching on the Dirichlet noise at the root.** It is not a deviation: Stoix
implements it (`apply_exploration_noise`, `ff_spo.py:119`, via
`rlax.add_dirichlet_noise`) and merely holds it at `fraction = 0.0`. In E2.6 we
measured that a very confident learned prior leaves legal actions with very few
particles (0.96 per position against 0.001 with a uniform prior), and the noise is
AlphaZero's mechanism for exactly that.

**The two setups being compared:**

  - **Literal SPO**: gamma_r = 1.0 (without our A6 discount), no loss penalty, with
    resampling, SPO's readout, critic with the contract's bootstrap
    (`discount * search_gamma * V`). It is SPO as it stands, and **we had never
    measured it with a genuinely trained critic**.
  - **Variant**: gamma_r = 0.9, penalty 1, no resampling, mean readout,
    depth-discounted bootstrap. It is E1.11c's setup, with ONE declared deviation.

Each setup trains its own actor and its own critic, and the cross
{critic yes/no} x {Dirichlet 0 / 0.25} is measured. Every game with the MODE of q.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from bench.metrics import fmt_fixed, wilson_lo, wilson_hi
from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform
from envs.tictactoe import (TicTacToe, ttt_reset_kernel, ttt_env_step_kernel,
                            ttt_auto_reset_kernel, NUM_ACTIONS, STATE_DIM,
                            TPB_TTT)
from envs.tictactoe_actor import TicTacToeActor
from envs.tictactoe_runner import RNG_RIVAL
from systems.spo.particles import SearchWorkspace
from systems.spo.readout import readout_greedy, readout_expected
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from demos.train_spo import (train_run, ActorLearner, HIDDEN, SEARCH_DEPTH,
                             TEMPERATURE, NO_RESAMPLE)
from demos.train_critic import Critic

comptime EVAL_ENVS = 128
comptime EVAL_STEPS = 300
comptime EVAL_SEED = UInt32(20260804)
comptime EVAL_PARTICLES = 128
comptime NOISE = Scalar[dtype](0.25)

# Literal SPO: without any of our additions.
comptime SPO_PERIOD = 3
comptime SPO_GAMMA = Scalar[dtype](1.0)
comptime SPO_PENALTY = Scalar[dtype](0.0)
# E1.11c's variant.
comptime VAR_GAMMA = Scalar[dtype](0.9)
comptime VAR_PENALTY = Scalar[dtype](1.0)


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
    print("   ", a.name, " n=", n,
          " gana ", pct(Float64(a.wins) / Float64(n)),
          " empata ", pct(Float64(a.draws) / Float64(n)),
          " PIERDE ", pct(Float64(a.losses) / Float64(n)),
          " IC[", fmt_fixed(wilson_lo(a.losses, n) * 100.0, 2), ",",
          fmt_fixed(wilson_hi(a.losses, n) * 100.0, 2), "]",
          " score ", fmt_fixed(a.score(), 4))


def verdict(base: Arm, other: Arm, what: String) raises:
    n1 = base.games(); n2 = other.games()
    hi2 = wilson_hi(other.losses, n2); lo1 = wilson_lo(base.losses, n1)
    lo2 = wilson_lo(other.losses, n2); hi1 = wilson_hi(base.losses, n1)
    print("      ", what, ": derrotas ",
          pct(Float64(base.losses) / Float64(n1)), " -> ",
          pct(Float64(other.losses) / Float64(n2)),
          "   score ", fmt_fixed(base.score(), 4), " -> ",
          fmt_fixed(other.score(), 4))
    if hi2 < lo1:
        print("          IC disjuntos, por debajo: MEJORA real.")
    elif lo2 > hi1:
        print("          IC disjuntos, por encima: EMPEORA de verdad.")
    else:
        print("          los IC se solapan: no se afirma diferencia.")


def play(ctx: DeviceContext, name: String, actor: ActorLearner,
         critic: Critic, spo_readout: Bool, use_critic: Bool,
         noise: Scalar[dtype], steps: Int,
         use_actor: Bool = True) raises -> Arm:
    """Plays with or without the actor's prior, with or without the critic, with or
    without noise.

    `use_actor = False` is needed for the control cell: without it the literal
    setup's improvement cannot be attributed to the actor.
    """
    period = SPO_PERIOD if spo_readout else NO_RESAMPLE
    gamma_r = SPO_GAMMA if spo_readout else VAR_GAMMA
    penalty = SPO_PENALTY if spo_readout else VAR_PENALTY
    # The depth-discounted bootstrap is ONLY needed if gamma_r < 1: it is what
    # puts the reward and the value on the same scale. With gamma_r = 1 (literal
    # SPO) the SearchModel's contract holds as it stands.
    depth_disc = gamma_r < Scalar[dtype](1)

    cfg = SPOConfig(num_envs=EVAL_ENVS, num_particles=EVAL_PARTICLES,
                    num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                    search_depth=SEARCH_DEPTH, resample_period=period,
                    temperature=TEMPERATURE, search_gamma=1.0,
                    search_gae_lambda=1.0, dirichlet_alpha=1.0,
                    dirichlet_fraction=noise)
    amodel = TicTacToeActor(ctx, cfg.num_search_particles(), HIDDEN, gamma_r,
                            penalty, use_critic, depth_disc)
    amodel.sync_from(ctx, actor.net.params)
    amodel.sync_critic_from(ctx, critic.online)
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
    umodel = TicTacToe(gamma_r, penalty)
    for step in range(steps):
        sd = EVAL_SEED ^ (UInt32(step) * 2654435761)
        if use_actor:
            search[TicTacToeActor](ctx, ws, cfg, amodel, state, sd)
        else:
            search[TicTacToe](ctx, ws, cfg, umodel, state, sd)
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


def main() raises:
    with DeviceContext() as ctx:
        print("=== Revision: critico reconectado + ruido de Dirichlet ===")
        print("   ", EVAL_ENVS, "partidas x", EVAL_STEPS, "turnos, moda de q,",
              " N =", EVAL_PARTICLES)
        print("   referencias exactas: azar 0.6484 | optimo 0.9974 (pierde 0.00%)")
        print()

        print("--- entrenamiento: cada montaje con su actor y su critico ---")
        # The critic goes in DURING training already: if the search that
        # generates the data does not use it, the q the actor learns is not that of
        # the system measured afterwards.
        spo = train_run(ctx, String("SPO literal"), True, True, SPO_PERIOD,
                        SPO_GAMMA, SPO_PENALTY, EVAL_PARTICLES, True, False, 0)
        var_ = train_run(ctx, String("variante E1.11c"), True, False,
                         NO_RESAMPLE, VAR_GAMMA, VAR_PENALTY, EVAL_PARTICLES,
                         True, True, 0)

        print("=== 0. la celda de control: SPO literal SIN actor ===")
        print("    Sin esta celda no se puede atribuir al actor la mejora del")
        print("    montaje literal, que es la afirmacion mas fuerte del bloque.")
        base_noactor = play(ctx, String("SPO literal, prior UNIFORME"),
                            spo.actor, spo.critic, True, False,
                            Scalar[dtype](0), EVAL_STEPS, False)
        base_noactor_c = play(ctx, String("  idem, CON critico        "),
                              spo.actor, spo.critic, True, True,
                              Scalar[dtype](0), EVAL_STEPS, False)
        show(base_noactor); show(base_noactor_c)
        print()

        print("=== 1. SPO literal (gamma_r=1, con remuestreo, readout de SPO) ===")
        s00 = play(ctx, String("sin critico, sin ruido "), spo.actor, spo.critic,
                   True, False, Scalar[dtype](0), EVAL_STEPS)
        s10 = play(ctx, String("CON critico, sin ruido "), spo.actor, spo.critic,
                   True, True, Scalar[dtype](0), EVAL_STEPS)
        s01 = play(ctx, String("sin critico, con ruido "), spo.actor, spo.critic,
                   True, False, NOISE, EVAL_STEPS)
        s11 = play(ctx, String("CON critico, con ruido "), spo.actor, spo.critic,
                   True, True, NOISE, EVAL_STEPS)
        show(s00); show(s10); show(s01); show(s11)
        print("    veredictos:")
        verdict(base_noactor, s00, String("anadir el ACTOR  "))
        verdict(s00, s10, String("anadir el critico"))
        verdict(s00, s01, String("anadir el ruido  "))
        verdict(s00, s11, String("los dos          "))
        print()

        print("=== 2. Variante (gamma_r=0.9, castigo, sin remuestreo, media) ===")
        v00 = play(ctx, String("sin critico, sin ruido "), var_.actor,
                   var_.critic, False, False, Scalar[dtype](0), EVAL_STEPS)
        v10 = play(ctx, String("CON critico, sin ruido "), var_.actor,
                   var_.critic, False, True, Scalar[dtype](0), EVAL_STEPS)
        v01 = play(ctx, String("sin critico, con ruido "), var_.actor,
                   var_.critic, False, False, NOISE, EVAL_STEPS)
        v11 = play(ctx, String("CON critico, con ruido "), var_.actor,
                   var_.critic, False, True, NOISE, EVAL_STEPS)
        show(v00); show(v10); show(v01); show(v11)
        print("    veredictos:")
        verdict(v00, v10, String("anadir el critico"))
        verdict(v00, v01, String("anadir el ruido  "))
        verdict(v00, v11, String("los dos          "))
        print()
        print("   El ruido y el critico son AMBOS parte de SPO (Stoix los tiene")
        print("   los dos; el ruido con fraction=0 por defecto). Activarlos no")
        print("   anade ninguna desviacion que defender.")
