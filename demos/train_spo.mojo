"""E2.4: the complete learning loop. Actor and critic, both training.

Up to here the critic learned on its own (E1.10) and the actor existed but nobody
called it. This brings them together and closes half the EM loop:

    E-step   the search plans and produces q, the improved policy
    M-step   the critic learns to predict returns  (L2 loss, E1.10)
             the actor learns to imitate q         (equation 11, E2.2/E2.3)

The return leg is still missing: having the search's prior come from the actor
instead of being uniform. That is E2.5, and until then the search does not know
there is an actor -- here the actor only watches. That this is deliberate matters:
it allows measuring whether the actor LEARNS before putting it in charge of
deciding, and if something looks odd in E2.5 we already know it does not come from
here.

**Its own optimiser for each network**, as in Stoix, which has separate `actor_lr`
and `critic_lr` (both at 3e-4 in its config) and chains the norm clip inside each
optax. That is: two Adam states, two independent global clips. Mixing them would
mean one's norm clipped the other's gradients.

**What gets reported and why.** The raw cross entropy is useless as a curve: it
decomposes into H(q) + KL(q||pi) and H(q) does NOT depend on the actor, so the floor
moves when q changes. The **KL** is reported, whose zero means "the actor reproduces
what the search says" and is comparable across configurations. See the long comment
in networks/actor_loss.mojo.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from std.math import log

from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.copy import copy_kernel
from ops.rng import fill_uniform
from ops.softmax import log_softmax_rows
from envs.tictactoe import (TicTacToe, ttt_reset_alt_kernel, ttt_env_step_kernel,
                            ttt_auto_reset_alt_kernel, ttt_encode_obs_kernel,
                            ttt_legal_mask_from_obs_kernel, NUM_ACTIONS,
                            NUM_CELLS, STATE_DIM, OBS_DIM, TPB_TTT)
from envs.tictactoe_actor import TicTacToeActor
from envs.tictactoe_runner import RNG_RIVAL
from networks.actor import Actor, actor_probs
from networks.actor_loss import (actor_backward, cross_entropy_rows,
                                 entropy_rows, kl_rows)
from networks.mlp import (CriticCache, CriticGrads, CriticScratch,
                          critic_forward, critic_backward)
from networks.optim import (AdamState, adam_step, ema_update, sum_squares,
                            global_clip_scale)
from rl_utils.buffer import TrajectoryBuffer
from rl_utils.multistep import truncated_gae
from systems.spo.particles import SearchWorkspace
from systems.spo.readout import readout_expected, q_histogram
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from tests.helpers import upload, download, write_into
from demos.train_critic import Critic, init_critic_weights

comptime SEED = UInt32(20260802)
comptime NUM_ENVS = 32
comptime ROLLOUT = 16
comptime HIDDEN = 64
comptime GAMMA = Scalar[dtype](0.99)
comptime GAE_LAMBDA = Scalar[dtype](0.95)
comptime CRITIC_LR = Scalar[dtype](3e-4)
comptime ACTOR_LR = Scalar[dtype](3e-4)
comptime MAX_GRAD_NORM = Scalar[dtype](0.5)
comptime TAU = Scalar[dtype](0.005)
comptime BATCH = 16
comptime BUFFER_CAP = 256

# The search, with the setup left after E1.11c: no resampling, mean readout, and
# the loss penalty. It is the one that plays at 0.00% losses.
comptime NUM_PARTICLES = 128
comptime SEARCH_DEPTH = 6
comptime NO_RESAMPLE = 99
comptime TEMPERATURE = Scalar[dtype](0.02)
comptime REWARD_GAMMA = Scalar[dtype](0.9)
comptime LOSS_PENALTY = Scalar[dtype](1.0)

# Its own stream for the readout's action draw. RNG_POLICY (20000) and RNG_RIVAL
# (30000) are already taken; 40000 collides with neither.
comptime RNG_READOUT = UInt32(3_000_000)
# The opponent's opening move when it is the one that starts. A separate stream:
# it must correlate neither with the policy nor with the opponent's replies during
# the game, otherwise the seat and the play would be linked.
comptime RNG_OPEN = UInt32(5_000_000)

comptime TRAIN_ROUNDS = 30
comptime UPDATES_PER_ROUND = 80


struct ActorLearner(Movable):
    """The actor with everything needed to train it."""

    var net: Actor
    var grads: CriticGrads
    var scratch: CriticScratch
    var a_w1: AdamState
    var a_b1: AdamState
    var a_w2: AdamState
    var a_b2: AdamState
    var a_w3: AdamState
    var a_b3: AdamState

    def __init__(out self, ctx: DeviceContext, max_batch: Int) raises:
        self.net = Actor(ctx, max_batch, HIDDEN)
        self.grads = CriticGrads(ctx, OBS_DIM, HIDDEN, NUM_ACTIONS)
        self.scratch = CriticScratch(ctx, max_batch, OBS_DIM, HIDDEN,
                                     NUM_ACTIONS)
        self.a_w1 = AdamState(ctx, OBS_DIM * HIDDEN)
        self.a_b1 = AdamState(ctx, HIDDEN)
        self.a_w2 = AdamState(ctx, HIDDEN * HIDDEN)
        self.a_b2 = AdamState(ctx, HIDDEN)
        self.a_w3 = AdamState(ctx, HIDDEN * NUM_ACTIONS)
        self.a_b3 = AdamState(ctx, NUM_ACTIONS)


def init_actor_weights(ctx: DeviceContext, mut actor: ActorLearner,
                       seed: UInt32) raises:
    """He-style weights and zero biases, same as the critic.

    With zero biases and centred weights, the initial policy is practically uniform
    over the legal cells -- which is exactly the prior the search has been working
    with so far. That is, in E2.5 the actor will come in without a sudden jump.
    """
    fan_ins = List[Int]()
    fan_ins.append(OBS_DIM); fan_ins.append(HIDDEN); fan_ins.append(HIDDEN)
    sizes = List[Int]()
    sizes.append(OBS_DIM * HIDDEN); sizes.append(HIDDEN * HIDDEN)
    sizes.append(HIDDEN * NUM_ACTIONS)

    for layer in range(3):
        n = sizes[layer]
        scale = Scalar[dtype](1) / Scalar[dtype](fan_ins[layer]) ** 0.5
        buf = zero_buffer[dtype](ctx, n)
        ctx.enqueue_function[fill_uniform, fill_uniform](
            buf.unsafe_ptr(), seed, UInt32(900 + layer), n,
            grid_dim=(n + 255) // 256, block_dim=256)
        ctx.synchronize()
        u = download[dtype](buf, n)
        vals = List[Scalar[dtype]]()
        for i in range(n):
            vals.append((u[i] * Scalar[dtype](2) - Scalar[dtype](1)) * scale)
        if layer == 0:
            write_into[dtype](actor.net.params.w1, vals)
        elif layer == 1:
            write_into[dtype](actor.net.params.w2, vals)
        else:
            write_into[dtype](actor.net.params.w3, vals)
    ctx.synchronize()


def collect(ctx: DeviceContext, mut buf: TrajectoryBuffer, cfg: SPOConfig,
            model: TicTacToe, amodel: TicTacToeActor, use_actor: Bool,
            spo_readout: Bool, ws: SearchWorkspace, state: DeviceBuffer[dtype],
            obs_buf: DeviceBuffer[dtype], next_obs_buf: DeviceBuffer[dtype],
            q_buf: DeviceBuffer[dtype], logits_buf: DeviceBuffer[dtype],
            reward: DeviceBuffer[dtype], done: DeviceBuffer[idx_dtype],
            u_rival: DeviceBuffer[dtype], u_readout: DeviceBuffer[dtype],
            u_open: DeviceBuffer[dtype], seed: UInt32,
            round_idx: Int) raises -> Scalar[dtype]:
    """Plays ROLLOUT turns and stores observations, rewards AND the q.

    The difference from E1.10's `collect` is that q: it is the actor's target, and
    it has to be captured AT THE MOMENT, because the search's workspace is reused
    and on the next turn it is gone.

    As there, `next_obs` is taken AFTER the step but BEFORE the auto-reset: if it
    were taken afterwards, the bootstrap would be looking at an empty board from
    another game.
    """
    blocks = (NUM_ENVS + TPB_TTT - 1) // TPB_TTT

    obs_hist = List[Scalar[dtype]]()
    next_hist = List[Scalar[dtype]]()
    q_hist = List[Scalar[dtype]]()
    rew_hist = List[Scalar[dtype]]()
    done_hist = List[Scalar[dtype]]()
    for _ in range(NUM_ENVS * ROLLOUT * OBS_DIM):
        obs_hist.append(Scalar[dtype](0))
        next_hist.append(Scalar[dtype](0))
    for _ in range(NUM_ENVS * ROLLOUT * NUM_ACTIONS):
        q_hist.append(Scalar[dtype](0))
    for _ in range(NUM_ENVS * ROLLOUT):
        rew_hist.append(Scalar[dtype](0))
        done_hist.append(Scalar[dtype](0))

    finished = 0
    score_sum = Scalar[dtype](0)

    for t in range(ROLLOUT):
        stream = UInt32(round_idx * ROLLOUT + t)

        # The observation BEFORE moving: it is the one that goes with this q.
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            obs_buf.unsafe_ptr(), state.unsafe_ptr(), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)

        # With `use_actor`, the search starts from the network's prior instead of
        # uniform: it is what closes the EM loop.
        if use_actor:
            search[TicTacToeActor](ctx, ws, cfg, amodel, state,
                                   seed ^ (stream * 2654435761))
        else:
            search[TicTacToe](ctx, ws, cfg, model, state,
                              seed ^ (stream * 2654435761))
        # The variant's readout leaves a dense q in q_buf AND picks the action.
        # It draws from q (greedy=False) instead of taking the mode: that draw IS
        # SPO's exploration, and without it the buffer would only ever see one line
        # of play. FRESH uniforms: reusing the ones the search left would make the
        # draw correlated with the particle sampling.
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_readout.unsafe_ptr(), seed, RNG_READOUT + stream, NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        if spo_readout:
            # SPO's q as it stands: aggregate the per-particle weights by action.
            # The action executed is the one `search` already chose (drawn from q).
            q_histogram(ctx, ws.particles, ws.output, cfg, q_buf)
        else:
            readout_expected(ctx, ws.particles, ws.output, cfg, logits_buf,
                             q_buf, u_readout, False)
        ctx.synchronize()

        obs_now = download[dtype](obs_buf, NUM_ENVS * OBS_DIM)
        q_now = download[dtype](q_buf, NUM_ENVS * NUM_ACTIONS)

        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), seed, RNG_RIVAL + stream, NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), ws.output.action.unsafe_ptr(),
            u_rival.unsafe_ptr(), reward.unsafe_ptr(), done.unsafe_ptr(),
            NUM_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            next_obs_buf.unsafe_ptr(), state.unsafe_ptr(), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

        next_now = download[dtype](next_obs_buf, NUM_ENVS * OBS_DIM)
        rew_now = download[dtype](reward, NUM_ENVS)
        done_now = download[idx_dtype](done, NUM_ENVS)

        for e in range(NUM_ENVS):
            for k in range(OBS_DIM):
                obs_hist[(e * ROLLOUT + t) * OBS_DIM + k] = obs_now[e * OBS_DIM + k]
                next_hist[(e * ROLLOUT + t) * OBS_DIM + k] = next_now[e * OBS_DIM + k]
            for k in range(NUM_ACTIONS):
                q_hist[(e * ROLLOUT + t) * NUM_ACTIONS + k] = \
                    q_now[e * NUM_ACTIONS + k]
            rew_hist[e * ROLLOUT + t] = rew_now[e]
            d = Scalar[dtype](1) if Int(done_now[e]) != 0 else Scalar[dtype](0)
            done_hist[e * ROLLOUT + t] = d
            if d != 0:
                finished += 1
                score_sum += rew_now[e]

        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_open.unsafe_ptr(), seed, RNG_OPEN + stream, NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_auto_reset_alt_kernel, ttt_auto_reset_alt_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), u_open.unsafe_ptr(), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)

    # One sequence per env.
    for e in range(NUM_ENVS):
        seq_obs = List[Scalar[dtype]]()
        seq_next = List[Scalar[dtype]]()
        seq_q = List[Scalar[dtype]]()
        seq_r = List[Scalar[dtype]]()
        seq_d = List[Scalar[dtype]]()
        seq_tr = List[Scalar[dtype]]()
        for t in range(ROLLOUT):
            for k in range(OBS_DIM):
                seq_obs.append(obs_hist[(e * ROLLOUT + t) * OBS_DIM + k])
                seq_next.append(next_hist[(e * ROLLOUT + t) * OBS_DIM + k])
            for k in range(NUM_ACTIONS):
                seq_q.append(q_hist[(e * ROLLOUT + t) * NUM_ACTIONS + k])
            seq_r.append(rew_hist[e * ROLLOUT + t])
            seq_d.append(done_hist[e * ROLLOUT + t])
            seq_tr.append(Scalar[dtype](0))
        buf.add(seq_obs, seq_r, seq_d, seq_tr, seq_next, seq_q)

    return score_sum / Scalar[dtype](finished) if finished > 0 \
        else Scalar[dtype](0)


def states_from_buffer(buf: TrajectoryBuffer, n_want: Int, seed: UInt32,
                       stream: UInt32) raises -> List[Scalar[dtype]]:
    """`n_want` boards taken from the buffer, reconstructed from the observation.

    It is needed because the KL's floor has to be measured over the SAME state
    distribution the training uses. Measuring it over fresh positions gives a number
    that is not comparable -- it happened to me: I came out with a floor of 1.21 and
    an actor at 0.94, that is, the actor "below the floor", which is impossible and
    only meant I was comparing two different things.
    """
    idx = buf.sample_indices(n_want, seed, stream)
    obs = buf.gather(idx)
    span = buf.t_len * buf.obs_dim
    out = List[Scalar[dtype]]()
    for k in range(n_want):
        # A random step within the sequence, so as not to always take t=0 (which
        # is almost always the empty board).
        t = Int((seed + UInt32(k) * 2654435761) % UInt32(buf.t_len))
        base = k * span + t * buf.obs_dim
        for c in range(NUM_CELLS):
            mine = obs[base + c]
            theirs = obs[base + NUM_CELLS + c]
            if mine != 0:
                out.append(Scalar[dtype](1))
            elif theirs != 0:
                out.append(Scalar[dtype](-1))
            else:
                out.append(Scalar[dtype](0))
    return out^


def measure_q_noise(ctx: DeviceContext, cfg: SPOConfig, model: TicTacToe,
                    amodel: TicTacToeActor, use_actor: Bool,
                    ws: SearchWorkspace, state: DeviceBuffer[dtype],
                    q_buf: DeviceBuffer[dtype], logits_buf: DeviceBuffer[dtype],
                    u_readout: DeviceBuffer[dtype], reps: Int,
                    seed: UInt32) raises -> Scalar[dtype]:
    """The KL's FLOOR: how much noise q carries as a target.

    It is needed for the same reason H(q) had to be separated from the cross
    entropy. The q the actor imitates comes out of a search with RANDOM particles:
    the same position searched twice gives different qs. An actor is a deterministic
    function of the state, so the best it can learn is the MEAN of those qs, and its
    KL will not drop below the noise separating them.

    **How it is computed matters, and I got it wrong twice before finding it.** The
    floor is E_k[KL(q_k || q_mean)], and with pi = q_mean that is

        floor = H(q_mean) - E_k[H(q_k)]

    That second form is the one used here, and it is the only stable one. Computing
    it as a direct KL requires estimating q_mean well, and with q nearly one-hot over
    5-9 cells that calls for a great many repetitions: if q_k's cell appears in few
    of the others, the log of a tiny probability inflates the result without bound
    (with leave-one-out and 12 repetitions I got a floor of 1.33, above the actor's
    KL, that is, impossible). With the difference of entropies there are no
    logarithms of tiny values and the number is stable.

    A bias remains: H(q_mean) is estimated with `reps` samples and comes out
    somewhat low, so the floor does too. It is mitigated by raising reps, and main()
    measures it with two different values so that any movement in the number shows.
    """
    all_q = List[Scalar[dtype]]()
    n = cfg.num_envs * NUM_ACTIONS
    blocks = (cfg.num_envs + TPB_TTT - 1) // TPB_TTT

    for k in range(reps):
        # With THE SAME model that generated the training data. Measuring it with
        # another gives a floor from a different q distribution and the number is not
        # comparable: it happened to me again here -- a "138% of what is reducible"
        # came out, impossible, and this was the cause.
        if use_actor:
            search[TicTacToeActor](ctx, ws, cfg, amodel, state,
                                   seed ^ (UInt32(7919 + k) * 2654435761))
        else:
            search[TicTacToe](ctx, ws, cfg, model, state,
                              seed ^ (UInt32(7919 + k) * 2654435761))
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_readout.unsafe_ptr(), seed, UInt32(50000 + k), cfg.num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
        readout_expected(ctx, ws.particles, ws.output, cfg, logits_buf, q_buf,
                         u_readout, False)
        ctx.synchronize()
        qk = download[dtype](q_buf, n)
        for i in range(n):
            all_q.append(qk[i])

    # E_k[H(q_k)]: the mean entropy of the individual qs.
    mean_h = Scalar[dtype](0)
    for k in range(reps):
        for e in range(cfg.num_envs):
            acc = Scalar[dtype](0)
            for a in range(NUM_ACTIONS):
                w = all_q[k * n + e * NUM_ACTIONS + a]
                if w > Scalar[dtype](0):
                    acc += w * log(w)
            mean_h += -acc
    mean_h /= Scalar[dtype](reps * cfg.num_envs)

    # H(q_mean): the entropy of the mean.
    h_mean = Scalar[dtype](0)
    for e in range(cfg.num_envs):
        acc = Scalar[dtype](0)
        for a in range(NUM_ACTIONS):
            m = Scalar[dtype](0)
            for k in range(reps):
                m += all_q[k * n + e * NUM_ACTIONS + a]
            m /= Scalar[dtype](reps)
            if m > Scalar[dtype](0):
                acc += m * log(m)
        h_mean += -acc
    h_mean /= Scalar[dtype](cfg.num_envs)

    # By Jensen H(mean) >= mean(H), so this cannot come out negative.
    return h_mean - mean_h


@fieldwise_init
struct Report(Copyable, Movable):
    """What comes out of one learning step."""
    var critic_loss: Scalar[dtype]
    var cross_entropy: Scalar[dtype]
    var entropy_q: Scalar[dtype]
    var kl: Scalar[dtype]
    var actor_gnorm: Scalar[dtype]
    var actor_clip: Scalar[dtype]


def update(ctx: DeviceContext, mut critic: Critic, mut actor: ActorLearner,
           buf: TrajectoryBuffer, step: Int, seed: UInt32) raises -> Report:
    """One gradient step on BOTH networks, each with its own optimiser."""
    idx = buf.sample_indices(BATCH, seed, UInt32(step))
    n = BATCH * ROLLOUT

    obs = upload[dtype](ctx, buf.gather(idx))
    boot = upload[dtype](ctx, buf.gather_bootstrap(idx))
    q = upload[dtype](ctx, buf.gather_q(idx))
    reward = upload[dtype](ctx, buf.gather_steps(idx, 0))
    done_host = buf.gather_steps(idx, 1)
    trunc = upload[dtype](ctx, buf.gather_steps(idx, 2))

    disc_host = List[Scalar[dtype]]()
    for i in range(n):
        disc_host.append((Scalar[dtype](1) - done_host[i]) * GAMMA)
    discount = upload[dtype](ctx, disc_host)

    # ---------------- the critic, same as in E1.10 ----------------
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

    adv = zero_buffer[dtype](ctx, n)
    targets = zero_buffer[dtype](ctx, n)
    truncated_gae(ctx, adv, targets, reward, discount, v_tm1, v_t, trunc,
                  BATCH, ROLLOUT, GAE_LAMBDA)

    critic_forward(ctx, critic.online, critic.cache, obs, n)
    ctx.synchronize()
    pred = download[dtype](critic.cache.value, n)
    tgt = download[dtype](targets, n)
    c_loss = Scalar[dtype](0)
    for i in range(n):
        d = pred[i] - tgt[i]
        c_loss += Scalar[dtype](0.5) * d * d
    c_loss /= Scalar[dtype](n)

    critic_backward(ctx, critic.online, critic.cache, critic.grads,
                    critic.scratch, obs, targets, n)
    ctx.synchronize()

    c_sq = (sum_squares(ctx, critic.grads.dw1, OBS_DIM * HIDDEN)
            + sum_squares(ctx, critic.grads.db1, HIDDEN)
            + sum_squares(ctx, critic.grads.dw2, HIDDEN * HIDDEN)
            + sum_squares(ctx, critic.grads.db2, HIDDEN)
            + sum_squares(ctx, critic.grads.dw3, HIDDEN)
            + sum_squares(ctx, critic.grads.db3, 1))
    c_scale = global_clip_scale(c_sq, MAX_GRAD_NORM)

    adam_step(ctx, critic.online.w1, critic.grads.dw1, critic.a_w1,
              OBS_DIM * HIDDEN, CRITIC_LR, c_scale, step)
    adam_step(ctx, critic.online.b1, critic.grads.db1, critic.a_b1,
              HIDDEN, CRITIC_LR, c_scale, step)
    adam_step(ctx, critic.online.w2, critic.grads.dw2, critic.a_w2,
              HIDDEN * HIDDEN, CRITIC_LR, c_scale, step)
    adam_step(ctx, critic.online.b2, critic.grads.db2, critic.a_b2,
              HIDDEN, CRITIC_LR, c_scale, step)
    adam_step(ctx, critic.online.w3, critic.grads.dw3, critic.a_w3,
              HIDDEN, CRITIC_LR, c_scale, step)
    adam_step(ctx, critic.online.b3, critic.grads.db3, critic.a_b3,
              1, CRITIC_LR, c_scale, step)

    ema_update(ctx, critic.target.w1, critic.online.w1, OBS_DIM * HIDDEN, TAU)
    ema_update(ctx, critic.target.b1, critic.online.b1, HIDDEN, TAU)
    ema_update(ctx, critic.target.w2, critic.online.w2, HIDDEN * HIDDEN, TAU)
    ema_update(ctx, critic.target.b2, critic.online.b2, HIDDEN, TAU)
    ema_update(ctx, critic.target.w3, critic.online.w3, HIDDEN, TAU)
    ema_update(ctx, critic.target.b3, critic.online.b3, 1, TAU)

    # ---------------- the actor ----------------
    # The mask comes from the observation, which is all the buffer stores.
    ctx.enqueue_function[ttt_legal_mask_from_obs_kernel,
                         ttt_legal_mask_from_obs_kernel](
        actor.net.mask.unsafe_ptr(), obs.unsafe_ptr(), n,
        grid_dim=(n + 255) // 256, block_dim=256)
    actor_probs(ctx, actor.net.params, actor.net.cache, obs, actor.net.mask,
                actor.net.probs, n)

    # Loss and diagnostic. The cross entropy decomposes into H(q) + KL, and only
    # the KL depends on the actor.
    log_pi = zero_buffer[dtype](ctx, n * NUM_ACTIONS)
    ce = zero_buffer[dtype](ctx, n)
    hq = zero_buffer[dtype](ctx, n)
    kl = zero_buffer[dtype](ctx, n)
    ctx.enqueue_function[log_softmax_rows[32], log_softmax_rows[32]](
        log_pi.unsafe_ptr(), actor.net.cache.value.unsafe_ptr(), NUM_ACTIONS,
        grid_dim=n, block_dim=32)
    cross_entropy_rows(ctx, ce, q, log_pi, n, NUM_ACTIONS)
    entropy_rows(ctx, hq, q, n, NUM_ACTIONS)
    kl_rows(ctx, kl, ce, hq, n)
    ctx.synchronize()

    ce_h = download[dtype](ce, n)
    hq_h = download[dtype](hq, n)
    kl_h = download[dtype](kl, n)
    ce_m = Scalar[dtype](0); hq_m = Scalar[dtype](0); kl_m = Scalar[dtype](0)
    for i in range(n):
        ce_m += ce_h[i]; hq_m += hq_h[i]; kl_m += kl_h[i]
    ce_m /= Scalar[dtype](n); hq_m /= Scalar[dtype](n); kl_m /= Scalar[dtype](n)

    actor_backward(ctx, actor.net.params, actor.net.cache, actor.grads,
                   actor.scratch, obs, actor.net.probs, q, actor.net.mask, n)
    ctx.synchronize()

    # Its own GLOBAL clip: the actor's norm must not clip the critic nor the other
    # way round. It is what Stoix does by chaining the clip inside each optax.
    a_sq = (sum_squares(ctx, actor.grads.dw1, OBS_DIM * HIDDEN)
            + sum_squares(ctx, actor.grads.db1, HIDDEN)
            + sum_squares(ctx, actor.grads.dw2, HIDDEN * HIDDEN)
            + sum_squares(ctx, actor.grads.db2, HIDDEN)
            + sum_squares(ctx, actor.grads.dw3, HIDDEN * NUM_ACTIONS)
            + sum_squares(ctx, actor.grads.db3, NUM_ACTIONS))
    a_scale = global_clip_scale(a_sq, MAX_GRAD_NORM)
    a_norm = a_sq ** 0.5

    adam_step(ctx, actor.net.params.w1, actor.grads.dw1, actor.a_w1,
              OBS_DIM * HIDDEN, ACTOR_LR, a_scale, step)
    adam_step(ctx, actor.net.params.b1, actor.grads.db1, actor.a_b1,
              HIDDEN, ACTOR_LR, a_scale, step)
    adam_step(ctx, actor.net.params.w2, actor.grads.dw2, actor.a_w2,
              HIDDEN * HIDDEN, ACTOR_LR, a_scale, step)
    adam_step(ctx, actor.net.params.b2, actor.grads.db2, actor.a_b2,
              HIDDEN, ACTOR_LR, a_scale, step)
    adam_step(ctx, actor.net.params.w3, actor.grads.dw3, actor.a_w3,
              HIDDEN * NUM_ACTIONS, ACTOR_LR, a_scale, step)
    adam_step(ctx, actor.net.params.b3, actor.grads.db3, actor.a_b3,
              NUM_ACTIONS, ACTOR_LR, a_scale, step)
    ctx.synchronize()

    return Report(c_loss, ce_m, hq_m, kl_m, a_norm, a_scale)


@fieldwise_init
struct ArmResult(Copyable, Movable):
    var name: String
    var kl_first: Scalar[dtype]
    var kl_last: Scalar[dtype]
    var floor: Scalar[dtype]
    var critic_first: Scalar[dtype]
    var critic_last: Scalar[dtype]
    var score: Scalar[dtype]
    var hq_first: Scalar[dtype]
    var hq_last: Scalar[dtype]


struct TrainOutcome(Movable):
    """What comes out of training: the metrics AND the actor, so it can be measured.

    They go together because `ActorLearner` owns `DeviceBuffer`s and is only
    `Movable`: if training did not return it, one would have to retrain in order to
    measure.
    """
    var result: ArmResult
    var actor: ActorLearner
    var critic: Critic

    def __init__(out self, var result: ArmResult, var actor: ActorLearner,
                 var critic: Critic):
        # Explicit init: `@fieldwise_init` will not do because `ActorLearner` owns
        # DeviceBuffers and it has to be TRANSFERRED, not copied.
        self.result = result^
        self.actor = actor^
        self.critic = critic^


def train_run(ctx: DeviceContext, name: String, use_actor: Bool,
              spo_readout: Bool = False, period: Int = NO_RESAMPLE,
              gamma_r: Scalar[dtype] = REWARD_GAMMA,
              penalty: Scalar[dtype] = LOSS_PENALTY,
              particles: Int = NUM_PARTICLES, use_critic: Bool = False,
              depth_disc: Bool = False,
              dirichlet: Scalar[dtype] = 0,
              temp: Scalar[dtype] = TEMPERATURE,
              rounds: Int = TRAIN_ROUNDS,
              seed: UInt32 = SEED) raises -> TrainOutcome:
    """A complete arm: trains actor and critic, with or without a learned prior.

    Both arms share seed, config and number of steps. The only thing that changes is
    where the search's prior comes from, so the difference can be attributed to that
    and to nothing else.

    `rounds` is the DATA budget: each round plays ROLLOUT turns across NUM_ENVS
    environments, that is, ROLLOUT*NUM_ENVS environment steps. It is a parameter and
    not a constant because Milestone 4's comparison needs to match this budget with
    SPO-Stoix's, and the default has to stay as it always was so as not to silently
    move what the other experiments measure.

    `seed` governs ALL the training's randomness: initialisation of both networks,
    rollouts, the readout's draw and the buffer's sampling. It is a parameter so
    that the variability BETWEEN SEEDS can be measured, which is a different
    uncertainty from that of the number of games played: the Wilson interval says
    how many games were played, not how much the result depends on training's
    randomness.
    """
    cfg = SPOConfig(num_envs=NUM_ENVS, num_particles=particles,
                    num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                    search_depth=SEARCH_DEPTH, resample_period=period,
                    temperature=temp, search_gamma=1.0,
                    search_gae_lambda=1.0, dirichlet_alpha=1.0,
                    dirichlet_fraction=dirichlet)
    model = TicTacToe(gamma_r, penalty)
    ws = SearchWorkspace(ctx, cfg)

    n_rows = BATCH * ROLLOUT
    critic = Critic(ctx, n_rows)
    init_critic_weights(ctx, critic, seed)
    actor = ActorLearner(ctx, n_rows)
    init_actor_weights(ctx, actor, seed)
    amodel = TicTacToeActor(ctx, cfg.num_search_particles(), HIDDEN,
                            gamma_r, penalty, use_critic, depth_disc)
    amodel.sync_from(ctx, actor.net.params)
    amodel.sync_critic_from(ctx, critic.online)
    buf = TrajectoryBuffer(BUFFER_CAP, ROLLOUT, OBS_DIM, NUM_ACTIONS)

    blocks = (NUM_ENVS + TPB_TTT - 1) // TPB_TTT
    state = zero_buffer[dtype](ctx, NUM_ENVS * STATE_DIM)
    obs_buf = zero_buffer[dtype](ctx, NUM_ENVS * OBS_DIM)
    next_obs_buf = zero_buffer[dtype](ctx, NUM_ENVS * OBS_DIM)
    q_buf = zero_buffer[dtype](ctx, NUM_ENVS * NUM_ACTIONS)
    logits_buf = zero_buffer[dtype](ctx, NUM_ENVS * NUM_ACTIONS)
    reward = zero_buffer[dtype](ctx, NUM_ENVS)
    done = zero_buffer[idx_dtype](ctx, NUM_ENVS)
    u_rival = zero_buffer[dtype](ctx, NUM_ENVS)
    u_readout = zero_buffer[dtype](ctx, NUM_ENVS)
    u_open = zero_buffer[dtype](ctx, NUM_ENVS)
    # The agent opens in the even-indexed environments and answers in the odd ones.
    # It therefore learns both seats, which is indispensable: the final measurements
    # evaluate it in both, and an agent trained on one side only would be judged
    # there on a situation it has never seen.
    ctx.enqueue_function[fill_uniform, fill_uniform](
        u_open.unsafe_ptr(), seed, RNG_OPEN, NUM_ENVS,
        grid_dim=blocks, block_dim=TPB_TTT)
    ctx.enqueue_function[ttt_reset_alt_kernel, ttt_reset_alt_kernel](
        state.unsafe_ptr(), u_open.unsafe_ptr(), NUM_ENVS,
        grid_dim=blocks, block_dim=TPB_TTT)

    print("--- brazo:", name, "---")
    print("  ronda   score    critico      H(q)       KL      |g_actor|")
    step = 0
    first = Report(0, 0, 0, 0, 0, 0)
    last = Report(0, 0, 0, 0, 0, 0)
    last_score = Scalar[dtype](0)
    for round_idx in range(rounds):
        score = collect(ctx, buf, cfg, model, amodel, use_actor, spo_readout,
                        ws, state,
                        obs_buf, next_obs_buf, q_buf, logits_buf, reward, done,
                        u_rival, u_readout, u_open, seed, round_idx)
        last_score = score
        for e in range(UPDATES_PER_ROUND):
            step += 1
            r = update(ctx, critic, actor, buf, step, seed)
            if round_idx == 0 and e == 0:
                first = r.copy()
            last = r.copy()
        # HERE is where the loop closes: the next round's search will see the actor
        # that has just been trained. Without this line the M-step would train a
        # network nobody uses.
        if use_actor:
            # BOTH networks get refreshed: the EM loop closes over the actor, and
            # the critic also has to be the current one or the search would evaluate
            # with a V from 80 gradient steps ago.
            amodel.sync_from(ctx, actor.net.params)
            amodel.sync_critic_from(ctx, critic.online)
            ctx.synchronize()
        report_every = rounds // 8 if rounds >= 48 else 6
        if round_idx % report_every == 0 or round_idx == rounds - 1:
            print("   ", round_idx, "  ", score, "  ", last.critic_loss,
                  "  ", last.entropy_q, "  ", last.kl, "  ", last.actor_gnorm)

    st_vals = states_from_buffer(buf, NUM_ENVS, seed, UInt32(999))
    write_into[dtype](state, st_vals)
    ctx.synchronize()
    floor = measure_q_noise(ctx, cfg, model, amodel, use_actor, ws, state,
                            q_buf, logits_buf, u_readout, 48, seed)
    print()
    res = ArmResult(name, first.kl, last.kl, floor, first.critic_loss,
                    last.critic_loss, last_score, first.entropy_q,
                    last.entropy_q)
    return TrainOutcome(res^, actor^, critic^)


def show(r: ArmResult) raises:
    reducible = r.kl_first - r.floor
    frac = (r.kl_first - r.kl_last) / reducible if reducible > 0 \
           else Scalar[dtype](0)
    print("  ", r.name)
    print("      critico ", r.critic_first, " -> ", r.critic_last)
    print("      KL ", r.kl_first, " -> ", r.kl_last, "   suelo ", r.floor,
          "   recorrido ", frac * Scalar[dtype](100), "% de lo reducible")
    print("      score de la busqueda al final: ", r.score)
    print("      H(q) ", r.hq_first, " -> ", r.hq_last,
          "   <- si esto cambia, las KL de los dos brazos miden objetivos "
          "distintos")


def main() raises:
    with DeviceContext() as ctx:
        print("=== E2.5: el prior del actor entra en la busqueda ===")
        print("   envs", NUM_ENVS, " rollout", ROLLOUT, " red", HIDDEN,
              " batch", BATCH, " lr", ACTOR_LR)
        print("   busqueda: N", NUM_PARTICLES, " profundidad", SEARCH_DEPTH,
              " sin remuestreo, readout de media")
        print("   Los dos brazos comparten semilla y pasos; lo unico que cambia")
        print("   es de donde sale el prior.")
        print()

        uniform = train_run(ctx, String("prior UNIFORME (E2.4)"), False)
        learned = train_run(ctx, String("prior del ACTOR (bucle EM cerrado)"),
                            True)

        print("=== comparacion ===")
        show(uniform.result)
        show(learned.result)
        print()
        print("   El score es contra rival aleatorio, con la accion sorteada de q")
        print("   (no la moda), asi que no es comparable con el 0.9936 de E1.11c,")
        print("   que se midio jugando la moda. La medicion buena, con partidas")
        print("   suficientes y las cuatro celdas del 2x2, es E2.6.")
