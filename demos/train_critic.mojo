"""The critic's training loop: play, store, learn.

    ./run.sh demos/train_critic.mojo

It brings together everything from stage 1. Two alternating phases, as in Stoix:

    ACT      play games with the search and store the transitions
    LEARN    sample from the buffer and take gradient steps on the critic

The learning phase's order is that of ff_spo.py's `_critic_loss_fn`, and there is
one detail that is easy to confuse: **TWO networks are used**.

    pred   = ONLINE critic (obs)              <- what gets trained
    v_tm1  = TARGET critic (obs)              |
    v_t    = TARGET critic (bootstrap_obs)    |-- to compute the targets
    targets = GAE(reward, (1-done)*gamma, lambda, v_tm1, v_t, truncated)
    loss   = mean of 0.5*(pred - targets)^2

If the online one were used for the targets, the critic would chase a target moving
at the same time as itself: that is why the slow copy exists (E1.7's EMA).

At this stage the search still runs with V = 0. The critic learns by WATCHING the
games that search generates, but does not feed it yet; plugging it in is E1.11, and
that is where whether it helps gets measured.

Two things get measured, and the second matters more:

  1. **does the loss go down?** A critic that does not reduce its error has learned
     nothing.
  2. **does its value SEPARATE the games won from the games lost?** Because the
     loss would also go down if the critic merely predicted the mean every time.

The second measurement does not use correlation but the difference of means by
outcome, and that is on purpose: the search wins ~97% of the games, so the outcome
is nearly constant and correlating against something nearly constant measures
noise. With classes this imbalanced, comparing group means holds up better.

An expectation worth having before looking at the numbers: the rival plays AT
RANDOM, so a particular game's outcome has a lot of luck in it. A good position can
end in a loss if the rival gets it right. The critic should predict the EXPECTED
value, not the particular outcome, so the separation between wins and losses will be
small by construction.

That is why the separation is reported with its STANDARD ERROR and in sigmas: with
classes this imbalanced (~97% wins) a small difference of means can be pure chance.
Measuring with few samples it came out at 1.7 sigmas (not claimable) and with 6.7
times more, 2.5 sigmas (claimable): the effect was there, the data to see it was
missing.

Results with the current configuration (25 rounds, 500 updates):

    loss              0.41  ->  0.017           (25 times smaller)
    mean V            0.00  ->  0.9346          (matches the expected value)
    separation        0.0009 -> 0.0072 +/- 0.0029  (2.47 sigmas: real)

The separation is REAL but small: V stays nearly constant across positions. Whether
that is enough for the search to see threats is another question, and it gets
answered in E1.11 by measuring the 2.02% loss rate, not by assuming.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.copy import copy_kernel
from ops.rng import fill_uniform
from envs.tictactoe import (TicTacToe, ttt_reset_kernel, ttt_env_step_kernel,
                            ttt_auto_reset_kernel, ttt_encode_obs_kernel,
                            NUM_ACTIONS, STATE_DIM, OBS_DIM, TPB_TTT)
from envs.tictactoe_runner import RNG_RIVAL
from networks.mlp import (CriticParams, CriticCache, CriticGrads, CriticScratch,
                          critic_forward, critic_backward, zero_critic_params)
from networks.optim import (AdamState, adam_step, ema_update, sum_squares,
                            global_clip_scale)
from rl_utils.buffer import TrajectoryBuffer
from rl_utils.multistep import truncated_gae
from systems.spo.particles import SearchWorkspace
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from tests.helpers import upload, download, write_into

# --- configuration (Stoix's values where applicable) ---
comptime SEED = UInt32(20260729)
comptime NUM_ENVS = 32
comptime ROLLOUT = 16
"""Steps per env in each acting phase. With games of ~4 turns, each sequence
covers about 4 complete games."""

comptime HIDDEN = 64
comptime OUT_DIM = 1
comptime GAMMA = Scalar[dtype](0.99)
comptime GAE_LAMBDA = Scalar[dtype](0.95)
comptime CRITIC_LR = Scalar[dtype](3e-4)
comptime MAX_GRAD_NORM = Scalar[dtype](0.5)
comptime TAU = Scalar[dtype](0.005)
comptime BATCH = 16
comptime BUFFER_CAP = 256

# The search that generates the games: the config tuned in A6/A7.
comptime NUM_PARTICLES = 64
comptime SEARCH_DEPTH = 6
comptime RESAMPLE_PERIOD = 3
comptime TEMPERATURE = Scalar[dtype](0.02)
comptime REWARD_GAMMA = Scalar[dtype](0.7)


struct Critic(Movable):
    """The whole critic: both networks, the working buffers and Adam."""

    var online: CriticParams
    var target: CriticParams
    var cache: CriticCache
    """For the online network's forward (the one being trained)."""
    var tcache: CriticCache
    """For the target network's two forwards."""
    var grads: CriticGrads
    var scratch: CriticScratch

    var a_w1: AdamState
    var a_b1: AdamState
    var a_w2: AdamState
    var a_b2: AdamState
    var a_w3: AdamState
    var a_b3: AdamState

    def __init__(out self, ctx: DeviceContext, max_batch: Int) raises:
        self.online = zero_critic_params(ctx, OBS_DIM, HIDDEN, OUT_DIM)
        self.target = zero_critic_params(ctx, OBS_DIM, HIDDEN, OUT_DIM)
        self.cache = CriticCache(ctx, max_batch, HIDDEN, OUT_DIM)
        self.tcache = CriticCache(ctx, max_batch, HIDDEN, OUT_DIM)
        self.grads = CriticGrads(ctx, OBS_DIM, HIDDEN, OUT_DIM)
        self.scratch = CriticScratch(ctx, max_batch, OBS_DIM, HIDDEN, OUT_DIM)
        self.a_w1 = AdamState(ctx, OBS_DIM * HIDDEN)
        self.a_b1 = AdamState(ctx, HIDDEN)
        self.a_w2 = AdamState(ctx, HIDDEN * HIDDEN)
        self.a_b2 = AdamState(ctx, HIDDEN)
        self.a_w3 = AdamState(ctx, HIDDEN * OUT_DIM)
        self.a_b3 = AdamState(ctx, OUT_DIM)


def init_critic_weights(ctx: DeviceContext, mut critic: Critic,
                        seed: UInt32) raises:
    """He-style initial weights and zero biases; the target starts identical.

    That both networks start IDENTICAL matters: if they differed, the targets would
    be biased from the very first update and it would be hard to see why.
    """
    fan_ins = List[Int]()
    fan_ins.append(OBS_DIM); fan_ins.append(HIDDEN); fan_ins.append(HIDDEN)
    sizes = List[Int]()
    sizes.append(OBS_DIM * HIDDEN); sizes.append(HIDDEN * HIDDEN)
    sizes.append(HIDDEN * OUT_DIM)

    for layer in range(3):
        n = sizes[layer]
        scale = Scalar[dtype](1) / Scalar[dtype](fan_ins[layer]) ** 0.5
        vals = List[Scalar[dtype]]()
        buf = zero_buffer[dtype](ctx, n)
        ctx.enqueue_function[fill_uniform, fill_uniform](
            buf.unsafe_ptr(), seed, UInt32(500 + layer), n,
            grid_dim=(n + 255) // 256, block_dim=256)
        ctx.synchronize()
        u = download[dtype](buf, n)
        for i in range(n):
            # From U(0,1) to U(-scale, scale): centred at zero and on He's scale.
            vals.append((u[i] * Scalar[dtype](2) - Scalar[dtype](1)) * scale)

        if layer == 0:
            write_into[dtype](critic.online.w1, vals)
            write_into[dtype](critic.target.w1, vals)
        elif layer == 1:
            write_into[dtype](critic.online.w2, vals)
            write_into[dtype](critic.target.w2, vals)
        else:
            write_into[dtype](critic.online.w3, vals)
            write_into[dtype](critic.target.w3, vals)
    ctx.synchronize()


def collect(ctx: DeviceContext, mut buf: TrajectoryBuffer, cfg: SPOConfig,
            model: TicTacToe, ws: SearchWorkspace, state: DeviceBuffer[dtype],
            obs_buf: DeviceBuffer[dtype], next_obs_buf: DeviceBuffer[dtype],
            reward: DeviceBuffer[dtype], done: DeviceBuffer[idx_dtype],
            u_rival: DeviceBuffer[dtype], seed: UInt32,
            round_idx: Int) raises -> Scalar[dtype]:
    """Plays ROLLOUT turns across NUM_ENVS games and stores the sequences.

    It returns the mean score of the finished games, so that one can see the search
    is still playing just as well while the critic learns.

    An important detail: `next_obs` is captured AFTER the step but BEFORE the
    auto-reset. It is Stoix's `bootstrap_obs`: if it were taken after the reset,
    the bootstrap would be looking at an empty board from a new game.
    """
    blocks = (NUM_ENVS + TPB_TTT - 1) // TPB_TTT

    # Per-env history, on the host: [T, ...] for each one.
    obs_hist = List[Scalar[dtype]]()
    next_hist = List[Scalar[dtype]]()
    rew_hist = List[Scalar[dtype]]()
    done_hist = List[Scalar[dtype]]()
    for _ in range(NUM_ENVS * ROLLOUT * OBS_DIM):
        obs_hist.append(Scalar[dtype](0))
        next_hist.append(Scalar[dtype](0))
    for _ in range(NUM_ENVS * ROLLOUT):
        rew_hist.append(Scalar[dtype](0))
        done_hist.append(Scalar[dtype](0))

    wins = 0
    draws = 0
    losses = 0

    for t in range(ROLLOUT):
        # The observation BEFORE deciding.
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            obs_buf.unsafe_ptr(), state.unsafe_ptr(), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)

        step_seed = seed ^ (UInt32(round_idx * 1000 + t) * 2654435761)
        search[TicTacToe](ctx, ws, cfg, model, state, step_seed)

        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), seed, RNG_RIVAL + UInt32(round_idx * 64 + t),
            NUM_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), ws.output.action.unsafe_ptr(),
            u_rival.unsafe_ptr(), reward.unsafe_ptr(), done.unsafe_ptr(),
            NUM_ENVS, grid_dim=blocks, block_dim=TPB_TTT)

        # The next observation, BEFORE the auto-reset.
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            next_obs_buf.unsafe_ptr(), state.unsafe_ptr(), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

        o = download[dtype](obs_buf, NUM_ENVS * OBS_DIM)
        no = download[dtype](next_obs_buf, NUM_ENVS * OBS_DIM)
        r = download[dtype](reward, NUM_ENVS)
        d = download[idx_dtype](done, NUM_ENVS)
        for e in range(NUM_ENVS):
            for k in range(OBS_DIM):
                obs_hist[(e * ROLLOUT + t) * OBS_DIM + k] = o[e * OBS_DIM + k]
                next_hist[(e * ROLLOUT + t) * OBS_DIM + k] = no[e * OBS_DIM + k]
            rew_hist[e * ROLLOUT + t] = r[e]
            done_hist[e * ROLLOUT + t] = Scalar[dtype](1) if Int(d[e]) != 0 \
                                         else Scalar[dtype](0)
            if Int(d[e]) != 0:
                if r[e] > Scalar[dtype](0.75): wins += 1
                elif r[e] > Scalar[dtype](0.25): draws += 1
                else: losses += 1

        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)

    # Each env contributes a whole sequence.
    zeros_t = List[Scalar[dtype]]()
    for _ in range(ROLLOUT):
        zeros_t.append(Scalar[dtype](0))      # truncated: en TTT nunca ocurre
    for e in range(NUM_ENVS):
        seq_obs = List[Scalar[dtype]]()
        seq_next = List[Scalar[dtype]]()
        seq_r = List[Scalar[dtype]]()
        seq_d = List[Scalar[dtype]]()
        for t in range(ROLLOUT):
            for k in range(OBS_DIM):
                seq_obs.append(obs_hist[(e * ROLLOUT + t) * OBS_DIM + k])
                seq_next.append(next_hist[(e * ROLLOUT + t) * OBS_DIM + k])
            seq_r.append(rew_hist[e * ROLLOUT + t])
            seq_d.append(done_hist[e * ROLLOUT + t])
        buf.add(seq_obs, seq_r, seq_d, zeros_t, seq_next)

    games = wins + draws + losses
    if games == 0:
        return Scalar[dtype](0)
    return (Scalar[dtype](wins) + Scalar[dtype](0.5) * Scalar[dtype](draws)) \
           / Scalar[dtype](games)


def update(ctx: DeviceContext, mut critic: Critic, buf: TrajectoryBuffer,
           step: Int, seed: UInt32) raises -> Scalar[dtype]:
    """One gradient step on the critic. Returns the loss BEFORE the step."""
    idx = buf.sample_indices(BATCH, seed, UInt32(step))
    n = BATCH * ROLLOUT          # filas de red: cada paso de cada secuencia

    obs = upload[dtype](ctx, buf.gather(idx))
    boot = upload[dtype](ctx, buf.gather_bootstrap(idx))
    reward = upload[dtype](ctx, buf.gather_steps(idx, 0))
    done_host = buf.gather_steps(idx, 1)
    trunc = upload[dtype](ctx, buf.gather_steps(idx, 2))

    # discount = (1 - done) * gamma, tal cual lo monta ff_spo._critic_loss_fn.
    disc_host = List[Scalar[dtype]]()
    for i in range(n):
        disc_host.append((Scalar[dtype](1) - done_host[i]) * GAMMA)
    discount = upload[dtype](ctx, disc_host)

    # 1. The TARGET's values, which are what build the targets.
    v_tm1 = zero_buffer[dtype](ctx, n)
    v_t = zero_buffer[dtype](ctx, n)
    critic_forward(ctx, critic.target, critic.tcache, obs, n)
    ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
        v_tm1.unsafe_ptr(), critic.tcache.value.unsafe_ptr(), n,
        grid_dim=(n + 255) // 256, block_dim=256)
    critic_forward(ctx, critic.target, critic.tcache, boot, n)
    ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
        v_t.unsafe_ptr(), critic.tcache.value.unsafe_ptr(), n,
        grid_dim=(n + 255) // 256, block_dim=256)

    # 2. The targets, with the truncated GAE.
    adv = zero_buffer[dtype](ctx, n)
    targets = zero_buffer[dtype](ctx, n)
    truncated_gae(ctx, adv, targets, reward, discount, v_tm1, v_t, trunc,
                  BATCH, ROLLOUT, GAE_LAMBDA)

    # 3. The ONLINE network's prediction, which is the one being trained.
    critic_forward(ctx, critic.online, critic.cache, obs, n)
    ctx.synchronize()

    pred = download[dtype](critic.cache.value, n)
    tgt = download[dtype](targets, n)
    loss = Scalar[dtype](0)
    for i in range(n):
        d = pred[i] - tgt[i]
        loss += Scalar[dtype](0.5) * d * d
    loss /= Scalar[dtype](n)

    # 4. Gradients, GLOBAL clip over the six tensors, and Adam.
    critic_backward(ctx, critic.online, critic.cache, critic.grads,
                    critic.scratch, obs, targets, n)
    ctx.synchronize()

    total_sq = (sum_squares(ctx, critic.grads.dw1, OBS_DIM * HIDDEN)
                + sum_squares(ctx, critic.grads.db1, HIDDEN)
                + sum_squares(ctx, critic.grads.dw2, HIDDEN * HIDDEN)
                + sum_squares(ctx, critic.grads.db2, HIDDEN)
                + sum_squares(ctx, critic.grads.dw3, HIDDEN * OUT_DIM)
                + sum_squares(ctx, critic.grads.db3, OUT_DIM))
    scale = global_clip_scale(total_sq, MAX_GRAD_NORM)

    adam_step(ctx, critic.online.w1, critic.grads.dw1, critic.a_w1,
              OBS_DIM * HIDDEN, CRITIC_LR, scale, step)
    adam_step(ctx, critic.online.b1, critic.grads.db1, critic.a_b1,
              HIDDEN, CRITIC_LR, scale, step)
    adam_step(ctx, critic.online.w2, critic.grads.dw2, critic.a_w2,
              HIDDEN * HIDDEN, CRITIC_LR, scale, step)
    adam_step(ctx, critic.online.b2, critic.grads.db2, critic.a_b2,
              HIDDEN, CRITIC_LR, scale, step)
    adam_step(ctx, critic.online.w3, critic.grads.dw3, critic.a_w3,
              HIDDEN * OUT_DIM, CRITIC_LR, scale, step)
    adam_step(ctx, critic.online.b3, critic.grads.db3, critic.a_b3,
              OUT_DIM, CRITIC_LR, scale, step)

    # 5. And the target moves a little towards the online one.
    ema_update(ctx, critic.target.w1, critic.online.w1, OBS_DIM * HIDDEN, TAU)
    ema_update(ctx, critic.target.b1, critic.online.b1, HIDDEN, TAU)
    ema_update(ctx, critic.target.w2, critic.online.w2, HIDDEN * HIDDEN, TAU)
    ema_update(ctx, critic.target.b2, critic.online.b2, HIDDEN, TAU)
    ema_update(ctx, critic.target.w3, critic.online.w3, HIDDEN * OUT_DIM, TAU)
    ema_update(ctx, critic.target.b3, critic.online.b3, OUT_DIM, TAU)
    ctx.synchronize()

    return loss


def evaluate(ctx: DeviceContext, mut critic: Critic, cfg: SPOConfig,
             model: TicTacToe, ws: SearchWorkspace, state: DeviceBuffer[dtype],
             obs_buf: DeviceBuffer[dtype], reward: DeviceBuffer[dtype],
             done: DeviceBuffer[idx_dtype], u_rival: DeviceBuffer[dtype],
             eval_cache: CriticCache, steps: Int,
             seed: UInt32) raises -> Scalar[dtype]:
    """Does V predict the game's outcome? Returns the correlation.

    The loss going down does NOT prove the critic has learned anything useful: it
    could be predicting the mean every time and it would go down all the same. The
    real test is whether its value SEPARATES the games that are won from those that
    are not.

    It plays, records V(s) at each step, and when the game ends assigns the real
    outcome (1 / 0.5 / 0) to all of its steps. At the end the Pearson correlation
    between the two series is computed. Near 0 = the critic distinguishes nothing;
    positive and high = it predicts.
    """
    blocks = (NUM_ENVS + TPB_TTT - 1) // TPB_TTT

    # Per env: the predicted values of the game in progress.
    pending = List[Scalar[dtype]]()
    pending_count = List[Int]()
    for _ in range(NUM_ENVS * 16):
        pending.append(Scalar[dtype](0))
    for _ in range(NUM_ENVS):
        pending_count.append(0)

    vs = List[Scalar[dtype]]()      # V(s) de pasos de partidas ya terminadas
    outs = List[Scalar[dtype]]()    # y el resultado real de su partida

    for t in range(steps):
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            obs_buf.unsafe_ptr(), state.unsafe_ptr(), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        # The value the critic gives the CURRENT board.
        critic_forward(ctx, critic.online, eval_cache, obs_buf, NUM_ENVS)
        search[TicTacToe](ctx, ws, cfg, model, state,
                          seed ^ (UInt32(9000 + t) * 2654435761))
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), seed, RNG_RIVAL + UInt32(3000 + t), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), ws.output.action.unsafe_ptr(),
            u_rival.unsafe_ptr(), reward.unsafe_ptr(), done.unsafe_ptr(),
            NUM_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

        v = download[dtype](eval_cache.value, NUM_ENVS)
        r = download[dtype](reward, NUM_ENVS)
        d = download[idx_dtype](done, NUM_ENVS)
        for e in range(NUM_ENVS):
            c = pending_count[e]
            if c < 16:
                pending[e * 16 + c] = v[e]
                pending_count[e] = c + 1
            if Int(d[e]) != 0:
                # The game ended: all of its steps share the outcome.
                for k in range(pending_count[e]):
                    vs.append(pending[e * 16 + k])
                    outs.append(r[e])
                pending_count[e] = 0

        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)

    n = len(vs)
    if n < 10:
        print("      (muy pocas partidas para evaluar)")
        return Scalar[dtype](0)

    # Mean V split by outcome. This is used and NOT the correlation because the
    # classes are VERY imbalanced: the search wins ~96%, so the outcome is nearly
    # constant and correlating against something nearly constant measures noise,
    # not predictive power. Comparing group means does hold up under the imbalance.
    sum_win = Scalar[dtype](0); n_win = 0
    sum_draw = Scalar[dtype](0); n_draw = 0
    sum_loss = Scalar[dtype](0); n_loss = 0
    for i in range(n):
        if outs[i] > Scalar[dtype](0.75):
            sum_win += vs[i]; n_win += 1
        elif outs[i] > Scalar[dtype](0.25):
            sum_draw += vs[i]; n_draw += 1
        else:
            sum_loss += vs[i]; n_loss += 1

    mw = sum_win / Scalar[dtype](n_win) if n_win > 0 else Scalar[dtype](0)
    md = sum_draw / Scalar[dtype](n_draw) if n_draw > 0 else Scalar[dtype](0)
    ml = sum_loss / Scalar[dtype](n_loss) if n_loss > 0 else Scalar[dtype](0)

    # Each group's deviation, to know whether the difference of means is real or
    # noise. With only ~50 lost positions out of ~1900, a small separation can be
    # pure chance, and claiming it without checking would be inventing a result.
    var_w = Scalar[dtype](0)
    var_l = Scalar[dtype](0)
    for i in range(n):
        if outs[i] > Scalar[dtype](0.75):
            dw = vs[i] - mw
            var_w += dw * dw
        elif outs[i] <= Scalar[dtype](0.25):
            dl = vs[i] - ml
            var_l += dl * dl
    if n_win > 1:
        var_w /= Scalar[dtype](n_win - 1)
    if n_loss > 1:
        var_l /= Scalar[dtype](n_loss - 1)

    print("      posiciones:", n, " -> ganadas", n_win, "(V medio", mw, ")",
          " empatadas", n_draw, "(", md, ")", " perdidas", n_loss, "(", ml, ")")

    if n_loss < 2 or n_win < 2:
        return Scalar[dtype](0)

    # Standard error of the difference of means, and how many sigmas it is.
    se = (var_w / Scalar[dtype](n_win) + var_l / Scalar[dtype](n_loss)) ** 0.5
    diff = mw - ml
    sigmas = diff / se if se > Scalar[dtype](0) else Scalar[dtype](0)
    print("      separacion", diff, " +/-", se, " -> ", sigmas, "sigmas")
    if sigmas > Scalar[dtype](2):
        print("      (>2 sigmas: la separacion es real, no ruido)")
    else:
        print("      (<2 sigmas: NO se puede afirmar que separe)")
    return diff


def main() raises:
    with DeviceContext() as ctx:
        print("=== entrenamiento del critico sobre tres en raya ===")
        print("  envs", NUM_ENVS, " rollout", ROLLOUT, " red", HIDDEN,
              " batch", BATCH, " lr", CRITIC_LR)
        print()

        cfg = SPOConfig(num_envs=NUM_ENVS, num_particles=NUM_PARTICLES,
                        num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                        search_depth=SEARCH_DEPTH,
                        resample_period=RESAMPLE_PERIOD,
                        temperature=TEMPERATURE, search_gamma=1.0,
                        search_gae_lambda=1.0)
        model = TicTacToe(reward_gamma=REWARD_GAMMA)
        ws = SearchWorkspace(ctx, cfg)

        critic = Critic(ctx, BATCH * ROLLOUT)
        init_critic_weights(ctx, critic, SEED)
        eval_cache = CriticCache(ctx, NUM_ENVS, HIDDEN, OUT_DIM)
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

        # The correlation BEFORE training: with random weights it should be ~0.
        print("  ANTES de entrenar:")
        corr0 = evaluate(ctx, critic, cfg, model, ws, state, obs_buf, reward,
                         done, u_rival, eval_cache, 400, SEED)
        print("      separacion V(ganadas) - V(perdidas):", corr0)
        print()

        step = 0
        for round_idx in range(25):
            score = collect(ctx, buf, cfg, model, ws, state, obs_buf,
                            next_obs_buf, reward, done, u_rival, SEED, round_idx)

            first = Scalar[dtype](0)
            last = Scalar[dtype](0)
            for e in range(20):
                step += 1
                l = update(ctx, critic, buf, step, SEED)
                if e == 0:
                    first = l
                last = l
            print("  ronda", round_idx, " secuencias", buf.size(),
                  " score de la busqueda", score,
                  " perdida", first, "->", last)

        print()
        print("  DESPUES de entrenar:")
        corr1 = evaluate(ctx, critic, cfg, model, ws, state, obs_buf, reward,
                         done, u_rival, eval_cache, 400, SEED)
        print("      separacion V(ganadas) - V(perdidas):", corr1)
        print()
        print("  separacion antes", corr0, " -> despues", corr1)
        print("  Que la perdida baje no basta: un critico que prediga siempre la")
        print("  media tambien la bajaria. Lo que dice si aprendio algo util es si")
        print("  su valor SEPARA las partidas que se ganan de las que se pierden.")
