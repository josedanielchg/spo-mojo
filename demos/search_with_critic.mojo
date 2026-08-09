"""E1.11: train the critic, plug it into the search and measure whether it helps.

The whole of stage 1 asks a single question: **do the losses go down?** The
critic-less planner loses 2.02% of its games as measured in A7; optimal play loses
0.00%. That 2% is rival threats the search did not see.

What the critic does exactly in the weight, taken from the code and not from
intuition. `update_particles_kernel` accumulates
`weights += r_d + next_value_d - value_d` and right afterwards does
`value_{d+1} = next_value_d`, so the sum TELESCOPES and a particle's final weight is

    weight = SUM_d r_d  +  (last bootstrap)  -  V(s_root)

And `root.mojo:49` hands the SAME V(s_root) to every particle of an env, so that
term is a per-env constant and the softmax cancels it entirely. That is, the critic
does NOT come in as a baseline: its only effect is the bootstrap of the particles
still ALIVE at the end of the search. Written out by cases, with V ~ c:

                        no critic            with critic
    win at d            gamma_r^d            gamma_r^d
    lose at d           0                    0
    stay alive          0                    c ~ 0.93

There is the problem. With `reward_gamma = 0.7`, winning at depth 1 is worth 0.7 and
staying alive is worth 0.93: **surviving scores higher than winning**, unless the
win is immediate. The search with a critic prefers not to settle the game. And it is
not a fault of the critic, it is a scale mismatch: the reward carries gamma_r^d
folded in (an A6 decision) and the value does not.

That is where the six setups measured below come from:

  1-4. the cross {no critic, with critic} x {gamma_r 0.7, 1.0}, which is the sweep
       the plan asked to re-measure now that the bootstrap has an effect;
  5.   the depth-discounted bootstrap (`depth_discounted`), which puts both halves
       of the sum on the same scale and leaves the desired order:

           win early  >  win late  >  survive  >  lose

       It is the configuration theory says should work, and it is here so as not to
       declare a negative result without having tried the favourable case.
  6.   the same setup 5 but with a "critic" that always returns the constant c. It
       separates LEVEL from INFORMATION: if it ties with the trained critic, then
       the network adds nothing beyond its mean, and the separation of 0.0072
       measured in E1.10 does not go far enough to change any decision.

Every arm plays the SAME games: the same number of envs, the same steps and the
same seed for the rival. The only thing that changes between them is the model.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns

from bench.metrics import fmt_fixed, wilson_lo, wilson_hi
from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform
from envs.tictactoe import (TicTacToe, ttt_reset_kernel, ttt_env_step_kernel,
                            ttt_auto_reset_kernel, ttt_encode_obs_kernel,
                            NUM_ACTIONS, STATE_DIM, OBS_DIM, TPB_TTT)
from envs.tictactoe_critic import TicTacToeCritic
from envs.tictactoe_runner import RNG_RIVAL
from networks.mlp import CriticCache, critic_forward
from rl_utils.buffer import TrajectoryBuffer
from systems.spo.particles import SearchWorkspace
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from systems.spo.readout import readout_greedy
from systems.spo.search_model import SearchModel
from tests.helpers import download, write_into
from demos.train_critic import (Critic, init_critic_weights, collect, update,
                                SEED, NUM_ENVS, HIDDEN, OUT_DIM, BATCH, ROLLOUT,
                                BUFFER_CAP, NUM_PARTICLES, SEARCH_DEPTH,
                                RESAMPLE_PERIOD, TEMPERATURE, REWARD_GAMMA)

# Evaluation is kept apart from training: more games and more envs, because what
# is wanted is a narrow confidence interval over a rate of ~2%.
comptime EVAL_ENVS = 128
comptime EVAL_STEPS = 400
comptime EVAL_SEED = UInt32(20260730)

comptime TRAIN_ROUNDS = 25
comptime UPDATES_PER_ROUND = 20


@fieldwise_init
struct Arm(Copyable, Movable):
    """One experiment arm's result."""
    var name: String
    var wins: Int
    var draws: Int
    var losses: Int
    var seconds: Float64

    def games(self) -> Int:
        return self.wins + self.draws + self.losses

    def score(self) -> Float64:
        n = self.games()
        return (Float64(self.wins) + 0.5 * Float64(self.draws)) / Float64(n)


def eval_config(reward_gamma: Scalar[dtype]) -> SPOConfig:
    """A7's search config, with EVAL_ENVS games at a time."""
    return SPOConfig(num_envs=EVAL_ENVS, num_particles=NUM_PARTICLES,
                     num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                     search_depth=SEARCH_DEPTH, resample_period=RESAMPLE_PERIOD,
                     temperature=TEMPERATURE, search_gamma=1.0,
                     search_gae_lambda=1.0)


def play[M: SearchModel](ctx: DeviceContext, name: String, model: M,
                         cfg: SPOConfig, steps: Int,
                         greedy: Bool = False) raises -> Arm:
    """Plays `steps` turns across EVAL_ENVS games and counts the scoreboard.

    It is A7's loop as it stands: search, play, let the rival answer, record the
    ones that end and reset the finished ones. The rival's seed depends only on the
    step, so every arm faces the same sequence of rivals.

    With `greedy` the move is the MODE of q instead of a sample from q. Sampling is
    the right thing for training (it is where the exploration comes from) but it
    deliberately injects suboptimal moves, so when measuring strength it has to be
    separated out.
    """
    ws = SearchWorkspace(ctx, cfg)
    q_buf = zero_buffer[dtype](ctx, EVAL_ENVS * NUM_ACTIONS)
    blocks = (EVAL_ENVS + TPB_TTT - 1) // TPB_TTT

    state = zero_buffer[dtype](ctx, EVAL_ENVS * STATE_DIM)
    reward = zero_buffer[dtype](ctx, EVAL_ENVS)
    done = zero_buffer[idx_dtype](ctx, EVAL_ENVS)
    u_rival = zero_buffer[dtype](ctx, EVAL_ENVS)

    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()

    wins = 0
    draws = 0
    losses = 0
    start = perf_counter_ns()
    for step in range(steps):
        search[M](ctx, ws, cfg, model, state,
                  EVAL_SEED ^ (UInt32(step) * 2654435761))
        if greedy:
            readout_greedy(ctx, ws.particles, ws.output, cfg, q_buf)
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), EVAL_SEED, RNG_RIVAL + UInt32(step), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
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
    seconds = Float64(perf_counter_ns() - start) / 1e9
    return Arm(name, wins, draws, losses, seconds)


def pct(x: Float64) -> String:
    return fmt_fixed(x * 100.0, 2) + "%"


def show(arm: Arm) raises:
    """One table row, with the Wilson CI over the LOSSES.

    The interval goes on the losses and not on the wins because it is the column
    that answers the question: optimal play loses 0.00%, so the losses directly
    measure what the search did not see.
    """
    n = arm.games()
    lo = wilson_lo(arm.losses, n)
    hi = wilson_hi(arm.losses, n)
    print("  ", arm.name,
          "  n=", n,
          "  gana ", pct(Float64(arm.wins) / Float64(n)),
          "  empata ", pct(Float64(arm.draws) / Float64(n)),
          "  PIERDE ", pct(Float64(arm.losses) / Float64(n)),
          " IC[", fmt_fixed(lo * 100.0, 2), ", ", fmt_fixed(hi * 100.0, 2), "]",
          "  score ", fmt_fixed(arm.score(), 4))


def verdict(base: Arm, other: Arm) raises:
    """Compares two arms by their loss rate and says whether anything can be claimed.

    The rule we set ourselves: if the Wilson intervals OVERLAP, no improvement is
    claimed. Not overlapping is a stricter condition than the formal two-proportion
    test, so sticking to it is conservative and easy to defend.
    """
    n1 = base.games(); n2 = other.games()
    lo1 = wilson_lo(base.losses, n1); hi1 = wilson_hi(base.losses, n1)
    lo2 = wilson_lo(other.losses, n2); hi2 = wilson_hi(other.losses, n2)
    p1 = Float64(base.losses) / Float64(n1)
    p2 = Float64(other.losses) / Float64(n2)

    print("   ", other.name, " vs ", base.name, ":")
    print("      derrotas ", pct(p1), " -> ", pct(p2))
    if hi2 < lo1:
        print("      los intervalos NO se solapan y el nuevo esta por debajo:",
              " se puede afirmar que pierde menos.")
    elif lo2 > hi1:
        print("      los intervalos NO se solapan y el nuevo esta por ENCIMA:",
              " pierde mas, la diferencia es real.")
    else:
        print("      los intervalos se SOLAPAN: no se afirma diferencia.")


def mean_value(ctx: DeviceContext, mut critic: Critic, cfg: SPOConfig,
               model: TicTacToe) raises -> Scalar[dtype]:
    """The critic's mean V over boards from real games.

    It is the `c` of the header's reasoning: the reference level the rewards are
    going to be compared against. It gets printed so that the arithmetic can be
    followed by hand in the report.
    """
    blocks = (EVAL_ENVS + TPB_TTT - 1) // TPB_TTT
    cache = CriticCache(ctx, EVAL_ENVS, HIDDEN, OUT_DIM)
    ws = SearchWorkspace(ctx, cfg)
    state = zero_buffer[dtype](ctx, EVAL_ENVS * STATE_DIM)
    obs = zero_buffer[dtype](ctx, EVAL_ENVS * OBS_DIM)
    reward = zero_buffer[dtype](ctx, EVAL_ENVS)
    done = zero_buffer[idx_dtype](ctx, EVAL_ENVS)
    u_rival = zero_buffer[dtype](ctx, EVAL_ENVS)

    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)

    total = Scalar[dtype](0)
    count = 0
    for step in range(30):
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            obs.unsafe_ptr(), state.unsafe_ptr(), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        critic_forward(ctx, critic.online, cache, obs, EVAL_ENVS)
        ctx.synchronize()
        vs = download[dtype](cache.value, EVAL_ENVS)
        for e in range(EVAL_ENVS):
            total += vs[e]
            count += 1

        search[TicTacToe](ctx, ws, cfg, model, state,
                          EVAL_SEED ^ (UInt32(step) * 40503))
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), EVAL_SEED, RNG_RIVAL + UInt32(step), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), ws.output.action.unsafe_ptr(),
            u_rival.unsafe_ptr(), reward.unsafe_ptr(), done.unsafe_ptr(),
            EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()
    return total / Scalar[dtype](count)


def train(ctx: DeviceContext, mut critic: Critic) raises:
    """E1.10's training, as it stands: the critic learns from the search WITHOUT a
    critic, which is the only policy there is so far."""
    cfg = SPOConfig(num_envs=NUM_ENVS, num_particles=NUM_PARTICLES,
                    num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                    search_depth=SEARCH_DEPTH, resample_period=RESAMPLE_PERIOD,
                    temperature=TEMPERATURE, search_gamma=1.0,
                    search_gae_lambda=1.0)
    model = TicTacToe(reward_gamma=REWARD_GAMMA)
    ws = SearchWorkspace(ctx, cfg)
    buf = TrajectoryBuffer(BUFFER_CAP, ROLLOUT, OBS_DIM)

    blocks = (NUM_ENVS + TPB_TTT - 1) // TPB_TTT
    state = zero_buffer[dtype](ctx, NUM_ENVS * STATE_DIM)
    obs_buf = zero_buffer[dtype](ctx, NUM_ENVS * OBS_DIM)
    next_obs_buf = zero_buffer[dtype](ctx, NUM_ENVS * OBS_DIM)
    reward = zero_buffer[dtype](ctx, NUM_ENVS)
    done = zero_buffer[idx_dtype](ctx, NUM_ENVS)
    u_rival = zero_buffer[dtype](ctx, NUM_ENVS)
    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), NUM_ENVS, grid_dim=blocks, block_dim=TPB_TTT)

    step = 0
    last = Scalar[dtype](0)
    for round_idx in range(TRAIN_ROUNDS):
        _ = collect(ctx, buf, cfg, model, ws, state, obs_buf, next_obs_buf,
                    reward, done, u_rival, SEED, round_idx)
        for _ in range(UPDATES_PER_ROUND):
            step += 1
            last = update(ctx, critic, buf, step, SEED)
    print("   entrenado:", TRAIN_ROUNDS * UPDATES_PER_ROUND,
          "pasos de gradiente, perdida final", last)


def main() raises:
    with DeviceContext() as ctx:
        print("=== E1.11: el critico entra en la busqueda ===")
        print("   evaluacion:", EVAL_ENVS, "partidas a la vez x", EVAL_STEPS,
              "turnos, misma semilla en todos los brazos")
        print()

        # 1. Entrenar el critico (E1.10).
        print("--- 1. entrenamiento del critico ---")
        critic = Critic(ctx, BATCH * ROLLOUT)
        init_critic_weights(ctx, critic, SEED)
        train(ctx, critic)

        cfg_train = SPOConfig(num_envs=EVAL_ENVS, num_particles=NUM_PARTICLES,
                              num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                              search_depth=SEARCH_DEPTH,
                              resample_period=RESAMPLE_PERIOD,
                              temperature=TEMPERATURE, search_gamma=1.0,
                              search_gae_lambda=1.0)
        c = mean_value(ctx, critic, cfg_train, TicTacToe(REWARD_GAMMA))
        print("   V medio sobre posiciones reales: c =", c)
        print("   peso de una particula segun como acabe (V(raiz) se cancela):")
        print("      ganar en d=0   ", Scalar[dtype](1))
        print("      ganar en d=1   ", Scalar[dtype](0.7),
              "   <- con reward_gamma=0.7")
        print("      ganar en d=3   ", Scalar[dtype](0.343))
        print("      seguir viva    ", c, "  <- esto es lo que anade el critico")
        print("      perder         ", Scalar[dtype](0))
        print()

        # 2. Los brazos.
        max_batch = EVAL_ENVS * NUM_PARTICLES
        m_c07 = TicTacToeCritic(ctx, max_batch, HIDDEN, Scalar[dtype](0.7))
        m_c10 = TicTacToeCritic(ctx, max_batch, HIDDEN, Scalar[dtype](1.0))
        m_dd = TicTacToeCritic(ctx, max_batch, HIDDEN, Scalar[dtype](0.7),
                               depth_discounted=True)
        m_c07.sync_from(ctx, critic.online)
        m_c10.sync_from(ctx, critic.online)
        m_dd.sync_from(ctx, critic.online)

        # The control that separates LEVEL from INFORMATION: a "critic" that
        # returns the constant c and nothing else. With w3 = 0 the network ignores
        # the input and outputs b3. If this arm ties with the trained critic, then
        # what the trained one adds is only its mean, and the separation of 0.0072
        # measured in E1.10 does not go far enough to move any decision.
        m_const = TicTacToeCritic(ctx, max_batch, HIDDEN, Scalar[dtype](0.7),
                                  depth_discounted=True)
        b3 = List[Scalar[dtype]](); b3.append(c)
        write_into[dtype](m_const.params.b3, b3)   # w1,w2,w3 se quedan a cero
        ctx.synchronize()

        # Warm-up: the first search of each kind pays for compilation.
        _ = play[TicTacToe](ctx, "warmup", TicTacToe(REWARD_GAMMA),
                            eval_config(REWARD_GAMMA), 3)
        _ = play[TicTacToeCritic](ctx, "warmup", m_c07, eval_config(0.7), 3)

        print("--- 2. seis montajes, las mismas partidas ---")
        a_base = play[TicTacToe](ctx, "sin critico  gamma_r=0.7",
                                 TicTacToe(Scalar[dtype](0.7)),
                                 eval_config(0.7), EVAL_STEPS)
        show(a_base)
        a_g1 = play[TicTacToe](ctx, "sin critico  gamma_r=1.0",
                               TicTacToe(Scalar[dtype](1.0)),
                               eval_config(1.0), EVAL_STEPS)
        show(a_g1)
        a_c07 = play[TicTacToeCritic](ctx, "CON critico  gamma_r=0.7", m_c07,
                                      eval_config(0.7), EVAL_STEPS)
        show(a_c07)
        a_c10 = play[TicTacToeCritic](ctx, "CON critico  gamma_r=1.0", m_c10,
                                      eval_config(1.0), EVAL_STEPS)
        show(a_c10)
        a_dd = play[TicTacToeCritic](ctx, "CON critico  escala coherente", m_dd,
                                     eval_config(0.7), EVAL_STEPS)
        show(a_dd)
        a_const = play[TicTacToeCritic](ctx, "V constante  escala coherente",
                                        m_const, eval_config(0.7), EVAL_STEPS)
        show(a_const)
        print()

        print("--- 2b. los mismos, pero jugando la MODA de q en vez de una muestra ---")
        g_base = play[TicTacToe](ctx, "sin critico  gamma_r=0.7  CODICIOSO",
                                 TicTacToe(Scalar[dtype](0.7)),
                                 eval_config(0.7), EVAL_STEPS, greedy=True)
        show(g_base)
        g_dd = play[TicTacToeCritic](ctx, "CON critico  coherente   CODICIOSO",
                                     m_dd, eval_config(0.7), EVAL_STEPS,
                                     greedy=True)
        show(g_dd)
        print()

        print("--- 3. veredicto (IC de Wilson al 95% sobre las derrotas) ---")
        verdict(a_base, a_c07)
        verdict(a_base, a_c10)
        verdict(a_base, a_g1)
        verdict(a_base, a_dd)
        print()
        print("--- 4. ¿aporta el critico algo mas que su media? ---")
        verdict(a_dd, a_const)
        print()
        print("--- 5. ¿cuanto de las derrotas era el muestreo de q? ---")
        verdict(a_base, g_base)
        verdict(g_base, g_dd)
        print()
        print("   referencia exacta: el juego optimo pierde 0.00%;")
        print("   jugar al azar pierde 28.81%.")
