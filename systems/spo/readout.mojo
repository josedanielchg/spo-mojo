"""Phase 5 of the search: reading out. This is where the improved policy q comes from.

It is the bridge to the M-step, and the thing to understand is that `q` is not an
array of probabilities: it is TWO things together.

    sampled_actions         the N root actions that survived, with repetitions if
                            resampling copied some of them several times
    sampled_action_weights  softmax(weight/temperature) of each one

Their weighted histogram IS q, equation 6 of the paper. The action executed in the
real environment is drawn from there -- it is sampled, the argmax is not taken,
and the exploration comes out of that sampling.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from ops.common import dtype, idx_dtype, GlobalF32, GlobalI32
from ops.copy import copy_kernel
from ops.rng import categorical_from_logits
from ops.softmax import softmax_rows
from systems.spo.launch import TPB, TPB_PARTICLES, blocks_for
from systems.spo.particles import Particles, SearchScratch, SPOOutput
from systems.spo.resampling import resample_logits_kernel
from systems.spo.spo_types import SPOConfig


def select_action_kernel(action_out: GlobalI32, root_actions: GlobalI32,
                         chosen: GlobalI32, num_envs: Int, num_particles: Int):
    """The env's final action is the ROOT action of the drawn particle."""
    env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env < num_envs:
        action_out[env] = root_actions[env * num_particles + Int(chosen[env])]


def mean_over_particles_kernel(out_mean: GlobalF32, values: GlobalF32,
                               num_envs: Int, num_particles: Int):
    """Mean per env. num_particles is 16, so one thread per env is plenty."""
    env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= num_envs:
        return
    total = Scalar[dtype](0)
    for n in range(num_particles):
        total += values[env * num_particles + n]
    out_mean[env] = total / Scalar[dtype](num_particles)


def q_histogram_kernel(q_out: GlobalF32, root_actions: GlobalI32,
                       weights: GlobalF32, num_envs: Int, num_particles: Int,
                       num_actions: Int):
    """q[env, a] = sum of the weights of the particles whose root action is `a`.

    It is equation 6 of the paper written out as an actual array:
    `readout_weighted` leaves q implicit (actions + weights, with repetitions)
    because that is enough to sample from, but to take the maximum you have to
    aggregate per action first.

    One thread per (env, action), and each one walks its env's particles. No
    atomics: each thread writes into its own slot.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= num_envs * num_actions:
        return
    env = i // num_actions
    a = i % num_actions
    total = Scalar[dtype](0)
    for n in range(num_particles):
        if Int(root_actions[env * num_particles + n]) == a:
            total += weights[env * num_particles + n]
    q_out[i] = total


def argmax_action_kernel(action_out: GlobalI32, q: GlobalF32, num_envs: Int,
                         num_actions: Int):
    """The action with the most q mass. One thread per env.

    Ties resolved by the lowest index (strict `>`), which is deterministic and is
    what `jnp.argmax` does.
    """
    env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= num_envs:
        return
    best = 0
    best_q = q[env * num_actions]
    for a in range(1, num_actions):
        v = q[env * num_actions + a]
        if v > best_q:
            best_q = v
            best = a
    action_out[env] = Scalar[idx_dtype](best)


def q_histogram(ctx: DeviceContext, particles: Particles, output: SPOOutput,
                cfg: SPOConfig, q_buf: DeviceBuffer[dtype]) raises:
    """SPO's q as a dense array [num_envs, num_actions], WITHOUT touching the action.

    `readout_weighted` leaves q implicit (root actions + weights, with
    repetitions) because that is enough to sample from. The M-step, on the other
    hand, needs q as a vector: it is what the cross entropy of equation 11 eats.
    Aggregating per action is exactly the regrouping `SUM_n w_n = SUM_a q(a)` --
    it changes nothing, only the shape.

    It does not overwrite `output.action`: the action was already chosen by
    `search` drawing from q, which is what SPO does. It serves to train the actor
    with SPO's q without changing how the game is played.
    """
    n_cells = cfg.num_envs * cfg.num_actions
    ctx.enqueue_function[q_histogram_kernel, q_histogram_kernel](
        q_buf.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        output.sampled_action_weights.unsafe_ptr(), cfg.num_envs,
        cfg.num_particles, cfg.num_actions,
        grid_dim=blocks_for(n_cells), block_dim=TPB)


def readout_greedy(ctx: DeviceContext, particles: Particles, output: SPOOutput,
                   cfg: SPOConfig, q_buf: DeviceBuffer[dtype]) raises:
    """The MODE of q instead of a sample from q. For EVALUATING, not for training.

    It is called AFTER `search`, and the only thing it does is overwrite
    `output.action`; everything else the search produced (the weights, the
    advantages, the value) is left intact, so the M-step would still see exactly
    the same thing.

    Why it is needed: `readout_weighted` draws the action from q, and that draw IS
    the algorithm's exploration. Perfect while learning, but when measuring
    strength it deliberately injects suboptimal moves and the measurement comes
    out worse than the agent can actually play. Separating the two is standard in
    RL (sample to train, mode to evaluate) and here it is also needed to compare
    fairly against an MCTS, which picks its move by the visit maximum.

    `q_buf` is [num_envs, num_actions].
    """
    q_histogram(ctx, particles, output, cfg, q_buf)
    ctx.enqueue_function[argmax_action_kernel, argmax_action_kernel](
        output.action.unsafe_ptr(), q_buf.unsafe_ptr(), cfg.num_envs,
        cfg.num_actions, grid_dim=blocks_for(cfg.num_envs), block_dim=TPB)


def action_mean_logits_kernel(logits_out: GlobalF32, root_actions: GlobalI32,
                              raw_weights: GlobalF32, num_envs: Int,
                              num_particles: Int, num_actions: Int,
                              temperature: Scalar[dtype]):
    """logits[env, a] = (mean of the weights of `a`'s particles) / temperature.

    The difference with `q_histogram_kernel` is where the exponential goes in, and
    that difference changes everything. Equation 6 does

        q(a)  =  SUM_{p in a}  exp(weight_p / tau)         <- exponential first

    and here we do

        q(a)  ~  exp( MEAN_{p in a}(weight_p) / tau )      <- mean first

    With the sum of exponentials a bad particle contributes ~0, but then it was
    already contributing ~0 compared with a good one: it never SUBTRACTS. That is
    why the action is judged by its best particles and the risk is invisible. With
    the mean, a particle that loses drags its action down in proportion to how
    frequent it is.

    Formally: it is the difference between estimating E[exp(A/tau)] and
    exp(E[A]/tau). They coincide if the environment is DETERMINISTIC (the paper's
    are); with a random rival they do not, and the Jensen gap is an optimistic
    bias.

    An action no particle tried is marked with -inf so that its q is 0.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= num_envs * num_actions:
        return
    env = i // num_actions
    a = i % num_actions
    total = Scalar[dtype](0)
    count = 0
    for n in range(num_particles):
        if Int(root_actions[env * num_particles + n]) == a:
            total += raw_weights[env * num_particles + n]
            count += 1
    if count == 0:
        logits_out[i] = Scalar[dtype](-1e30)
    else:
        logits_out[i] = (total / Scalar[dtype](count)) / temperature


def readout_expected(ctx: DeviceContext, particles: Particles,
                     output: SPOOutput, cfg: SPOConfig,
                     logits_buf: DeviceBuffer[dtype],
                     q_buf: DeviceBuffer[dtype],
                     uniforms: DeviceBuffer[dtype], greedy: Bool) raises:
    """Readout that averages per action BEFORE exponentiating. A VARIANT, not SPO.

    It departs from equation 6 of the paper on purpose. It is here because the
    audit in `demos/audit_blunders.mojo` showed that the original readout cannot
    penalise risk (see `action_mean_logits_kernel`), and the only way to PROVE
    that this is the cause is to change it and see whether blocking goes up.

    It leaves `output.action` and `q_buf` [num_envs, num_actions]. It touches
    neither `sampled_action_weights` nor `sampled_advantages`, so what the M-step
    would see is still SPO's.

    `greedy` chooses between the mode (evaluate) and a sample (train/explore).
    """
    n_cells = cfg.num_envs * cfg.num_actions

    ctx.enqueue_function[action_mean_logits_kernel, action_mean_logits_kernel](
        logits_buf.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        particles.resample_td_weights.unsafe_ptr(), cfg.num_envs,
        cfg.num_particles, cfg.num_actions, cfg.temperature,
        grid_dim=blocks_for(n_cells), block_dim=TPB)

    ctx.enqueue_function[softmax_rows[TPB_PARTICLES], softmax_rows[TPB_PARTICLES]](
        q_buf.unsafe_ptr(), logits_buf.unsafe_ptr(), cfg.num_actions,
        grid_dim=cfg.num_envs, block_dim=TPB_PARTICLES)

    if greedy:
        ctx.enqueue_function[argmax_action_kernel, argmax_action_kernel](
            output.action.unsafe_ptr(), q_buf.unsafe_ptr(), cfg.num_envs,
            cfg.num_actions, grid_dim=blocks_for(cfg.num_envs), block_dim=TPB)
    else:
        # Here the categorical is over ACTIONS, not over particles: the index that
        # comes out already IS the move, without going through root_actions.
        ctx.enqueue_function[categorical_from_logits[TPB_PARTICLES],
                             categorical_from_logits[TPB_PARTICLES]](
            output.action.unsafe_ptr(), logits_buf.unsafe_ptr(),
            uniforms.unsafe_ptr(), cfg.num_actions,
            grid_dim=cfg.num_envs, block_dim=TPB_PARTICLES)


def readout_weighted(ctx: DeviceContext, particles: Particles,
                     scratch: SearchScratch, output: SPOOutput, cfg: SPOConfig,
                     uniforms: DeviceBuffer[dtype]) raises:
    """Reads the search's result. Mirrors Stoix's `readout_weighted`.

    `uniforms` only needs num_envs values (one per env), but a buffer of P is
    passed for convenience: the first num_envs are read.
    """
    p_total = cfg.num_search_particles()
    blocks_p = blocks_for(p_total)

    # logits = weight / temperature, the same ones resampling uses
    ctx.enqueue_function[resample_logits_kernel, resample_logits_kernel](
        scratch.resample_logits.unsafe_ptr(),
        particles.resample_td_weights.unsafe_ptr(),
        p_total, cfg.temperature, grid_dim=blocks_p, block_dim=TPB)

    # normalised weights of each root action
    ctx.enqueue_function[softmax_rows[TPB_PARTICLES], softmax_rows[TPB_PARTICLES]](
        output.sampled_action_weights.unsafe_ptr(),
        scratch.resample_logits.unsafe_ptr(), cfg.num_particles,
        grid_dim=cfg.num_envs, block_dim=TPB_PARTICLES)

    # one particle per env, drawn with those weights
    ctx.enqueue_function[categorical_from_logits[TPB_PARTICLES],
                         categorical_from_logits[TPB_PARTICLES]](
        output.action.unsafe_ptr(), scratch.resample_logits.unsafe_ptr(),
        uniforms.unsafe_ptr(), cfg.num_particles,
        grid_dim=cfg.num_envs, block_dim=TPB_PARTICLES)

    # ...and its root action is the one executed. output.action is reused as the
    # destination: the particle index goes in and the action comes out.
    ctx.enqueue_function[select_action_kernel, select_action_kernel](
        output.action.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        output.action.unsafe_ptr(), cfg.num_envs, cfg.num_particles,
        grid_dim=blocks_for(cfg.num_envs), block_dim=TPB)

    ctx.enqueue_function[copy_kernel[idx_dtype], copy_kernel[idx_dtype]](
        output.sampled_actions.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        p_total, grid_dim=blocks_p, block_dim=TPB)

    ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
        output.sampled_advantages.unsafe_ptr(), particles.gae.unsafe_ptr(),
        p_total, grid_dim=blocks_p, block_dim=TPB)

    # The root's value, not the one at the end of the rollout: it is what Stoix
    # puts into SPOOutput.value (jnp.mean(root.particle_values, axis=-1)).
    ctx.enqueue_function[mean_over_particles_kernel, mean_over_particles_kernel](
        output.value.unsafe_ptr(), output.root_values.unsafe_ptr(),
        cfg.num_envs, cfg.num_particles,
        grid_dim=blocks_for(cfg.num_envs), block_dim=TPB)
