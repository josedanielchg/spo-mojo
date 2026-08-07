"""Phase 2 of the search: weighing. Equation 10 of the paper, in log-space.

It is the heart of the SMC: after the model advances the particles, this is where
it is decided how much evidence each one has accumulated in favour of its root
action.

    weight <- weight + (r + V' - V)      with a mask for dead particles

The paper writes it as a product, `w <- w * exp(A/eta)`. Storing the SUM of
advantages and deferring the `exp(.../eta)` to the resampling's softmax is
mathematically the same and far more stable: adding does not overflow,
exponentiating does.

And along the way the forward GAE is accumulated, which is a second tally with a
different destination: the weights decide which particles survive, the GAE feeds
the M-step's temperature loss. Do not confuse them.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext

from ops.common import dtype, idx_dtype, GlobalF32, GlobalI32
from systems.spo.launch import TPB, blocks_for
from systems.spo.particles import Particles, StepOutputs
from systems.spo.spo_types import SPOConfig


def update_particles_kernel(
        # particle state, updated in place
        weights: GlobalF32, gae: GlobalF32, value: GlobalF32,
        terminal: GlobalI32, depth: GlobalI32, prior_logits: GlobalF32,
        # what the model's step produced
        reward: GlobalF32, discount: GlobalF32, next_value: GlobalF32,
        next_prior_logits: GlobalF32,
        # config
        n_particles: Int, search_gamma: Scalar[dtype],
        search_gae_lambda: Scalar[dtype]):
    """Closes one depth: SMC weight, GAE, and the state handover.

    It fuses three things that in Stoix are separate functions
    (`smc_weight_update_fn`, `calculate_gae` and `update_particles`) because with
    one thread per particle all three are the same pass: the old values are read
    and the new ones written.

    The order inside the thread is the only delicate part: the TD error and the
    GAE need the OLD V(s) and depth, so they are computed before overwriting them.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p >= n_particles:
        return

    old_value = value[p]
    old_depth = Int(depth[p])
    was_terminal = Int(terminal[p]) != 0
    step_discount = discount[p]

    # The TD error is not multiplied by gamma because it already comes folded
    # inside next_value, which is the bootstrap_value the model returned. Stoix
    # comments the same: "We do not multiply by discount as we do it in the
    # recurrent_fn".
    td_error = reward[p] + next_value[p] - old_value

    # The terminal mask freezes the weight of particles that are already dead,
    # because what happens to them after dying should not change their evidence.
    mask = Scalar[dtype](0) if was_terminal else Scalar[dtype](1)
    weights[p] = weights[p] + td_error * mask

    # The GAE runs the other way round from usual. Normally it is computed
    # backwards in time, but here the search moves forwards and cannot look into
    # the future, so each depth adds its delta discounted by
    # (gamma*lambda*discount)^depth.
    #
    # Note that there is no terminal mask, just as in Stoix. What freezes a dead
    # particle is that its discount is 0 and then the factor vanishes on its own.
    # The exception is depth 0, where the exponent is 0 and the factor is 1 no
    # matter what, and that is correct: the first step always counts in full even
    # if the particle dies on it.
    decay_base = search_gamma * search_gae_lambda * step_discount
    decay = Scalar[dtype](1)
    for _ in range(old_depth):
        decay *= decay_base
    gae[p] = gae[p] + td_error * decay

    # And the handover: the new becomes the current.
    value[p] = next_value[p]
    prior_logits[p] = next_prior_logits[p]
    depth[p] = Scalar[idx_dtype](old_depth + 1)
    # `terminal` is sticky: once dead, dead. A discount of 0 kills it, whether it
    # was a real death or a truncation.
    if was_terminal or step_discount == 0.0:
        terminal[p] = Scalar[idx_dtype](1)


def update_particles(ctx: DeviceContext, particles: Particles,
                     outputs: StepOutputs, cfg: SPOConfig) raises:
    """Enqueues the closing of one depth."""
    p_total = cfg.num_search_particles()
    ctx.enqueue_function[update_particles_kernel, update_particles_kernel](
        particles.resample_td_weights.unsafe_ptr(), particles.gae.unsafe_ptr(),
        particles.value.unsafe_ptr(), particles.terminal.unsafe_ptr(),
        particles.depth.unsafe_ptr(), particles.prior_logits.unsafe_ptr(),
        outputs.reward.unsafe_ptr(), outputs.discount.unsafe_ptr(),
        outputs.next_value.unsafe_ptr(), outputs.next_prior_logits.unsafe_ptr(),
        p_total, cfg.search_gamma, cfg.search_gae_lambda,
        grid_dim=blocks_for(p_total), block_dim=TPB)
