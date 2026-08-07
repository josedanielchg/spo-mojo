"""Phase 1 of the search: seeding. Stoix's `root_fn`.

Starting from one state per environment, it leaves N particles per environment
ready for depth 0: all in the same state and with the same value, but each with
ITS OWN action drawn from the prior. That draw is the only thing separating them
at the start; everything else (that they diverge, that some turn out better) comes
afterwards.

`sample_next_actions` also lives here. It does not belong to the root, but it is
the other generic half of the step: the model leaves the logits of the new state
and this draws the action for the next depth. They sit together because they are
the same gesture -- sample an action and record its log-prob -- at two different
moments.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import exp, log

from ops.common import dtype, idx_dtype, NEG_INF, GlobalF32, GlobalI32
from ops.copy import copy_kernel
from ops.rng import categorical_from_logits
from systems.spo.launch import TPB, blocks_for, check_search_config
from systems.spo.particles import Particles, StepOutputs, SPOOutput
from systems.spo.spo_types import SPOConfig


def broadcast_state_kernel(particle_state: GlobalF32, root_state: GlobalF32,
                           n_particles: Int, num_particles: Int, state_dim: Int):
    """Copies the env's state to each of its N particles.
    This is Stoix's `broadcast_tree`. One thread per (particle, state component).
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    total = n_particles * state_dim
    if i >= total:
        return

    p = i // state_dim          # which particle
    d = i % state_dim           # which component of its state
    env = p // num_particles    # which env it comes from

    particle_state[p * state_dim + d] = root_state[env * state_dim + d]


def broadcast_value_kernel(particle_value: GlobalF32, root_value: GlobalF32,
                           n_particles: Int, num_particles: Int):
    """V(s_root) replicated to the env's N particles. They all start with the
    same value because they are all in the same state."""
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p < n_particles:
        particle_value[p] = root_value[p // num_particles]


def broadcast_logits_kernel(particle_logits: GlobalF32, root_logits: GlobalF32,
                            n_particles: Int, num_particles: Int, num_actions: Int):
    """Replicates the env's prior logits to its N particles.

    This is done in order to reuse phase 2's `categorical_from_logits`, which
    samples one action per ROW. By replicating, each particle has its own row and
    draws its own action; without replicating we would have to write a kernel that
    draws N samples from the same row, that is, new code where tested code already
    exists.

    From depth 1 onwards this replication is no longer needed: each particle is in
    a different state and has its own logits.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    total = n_particles * num_actions
    if i >= total:
        return

    p = i // num_actions
    a = i % num_actions
    env = p // num_particles

    particle_logits[p * num_actions + a] = root_logits[env * num_actions + a]


def log_prob_of_action_kernel(log_prob_out: GlobalF32, logits: GlobalF32,
                              actions: GlobalI32, n_particles: Int,
                              num_actions: Int):
    """log pi(a|s) of the action each particle happened to get.

    This function computes what probability the policy gave to the action each
    particle chose. It does not choose the action. The action was already chosen
    by categorical_from_logits.

    log_softmax(logits)[a] = logits[a] - logsumexp(logits), with the max
    subtracted to avoid overflow (same recipe as ops/softmax.mojo, but here the
    loop is single-threaded over num_actions, which is 2 or 4; setting up a whole
    block for that would waste it).

    In Stoix this is `pi.log_prob(sampled_actions)`.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p >= n_particles:
        return

    base = p * num_actions

    biggest = NEG_INF
    for a in range(num_actions):
        v = logits[base + a]
        if v > biggest:
            biggest = v

    total = Scalar[dtype](0)
    for a in range(num_actions):
        total += exp(logits[base + a] - biggest)

    chosen = Int(actions[p])
    log_prob_out[p] = logits[base + chosen] - (biggest + log(total))


def add_dirichlet_noise_kernel(logits: GlobalF32, u: GlobalF32, n_envs: Int,
                               n_actions: Int, fraction: Scalar[dtype]):
    """Dirichlet exploration noise at the root. One thread per env.

    Mirrors Stoix's `apply_exploration_noise` (`ff_spo.py:119`), which calls
    `rlax.add_dirichlet_noise` and does

        noisy = (1 - fraction) * prior + fraction * noise,   noise ~ Dir(alpha)

    **A detail worth saying out loud:** Stoix passes it `pi.logits`, that is, it
    mixes a vector of LOGITS with a vector of PROBABILITIES (the Dirichlet sums to
    1). The rlax docstring says "prior policy vector", so the semantics do not
    quite line up. With `fraction = 0` (its default) it makes no difference
    because the term vanishes; with fraction > 0 the noise contributes at most
    `fraction` to a logit. It is reproduced as is because that is what the
    reference does, and the oddity is noted rather than "fixed" on our own.

    With alpha = 1 the symmetric Dirichlet is uniform over the simplex, and it is
    sampled by normalising exponentials: e_i = -ln(u_i), noise = e / SUM(e). It is
    exact, with no Gamma sampler. alpha = 1 is Stoix's default.

    Masked cells (NEG_INF) stay masked: (1-f)*NEG_INF dominates any noise bounded
    in [0,1] as long as f < 1.
    """
    e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= n_envs:
        return
    base = e * n_actions
    total = Scalar[dtype](0)
    for a in range(n_actions):
        # -ln(u) with u in (0,1]. Bounded from below so as not to ask for log(0).
        uu = u[base + a]
        if uu < Scalar[dtype](1e-7):
            uu = Scalar[dtype](1e-7)
        total += -log(uu)
    if total <= Scalar[dtype](0):
        return
    for a in range(n_actions):
        uu = u[base + a]
        if uu < Scalar[dtype](1e-7):
            uu = Scalar[dtype](1e-7)
        noise = (-log(uu)) / total
        logits[base + a] = (Scalar[dtype](1) - fraction) * logits[base + a] \
                           + fraction * noise


def root_fn(ctx: DeviceContext, particles: Particles, outputs: StepOutputs,
            cfg: SPOConfig, root_state: DeviceBuffer[dtype],
            root_logits: DeviceBuffer[dtype], root_value: DeviceBuffer[dtype],
            uniforms: DeviceBuffer[dtype]) raises:
    """Seeds the search: N particles per env, one action each.

    Equivalent to Stoix's `root_fn`

    Inputs (computed by the model over the ROOT states, one per env):
        root_state  [num_envs, state_dim]
        root_logits [num_envs, num_actions]  policy scores for each action
        root_value  [num_envs]               V(s_root)
        uniforms    [P]                      random numbers to draw actions with

    Output: `particles` is left ready for depth 0. The fields that accumulate
    (weight, gae, terminal, depth) are zeroed HERE, as in Stoix's
    `init_particles`.

    Zeroing them explicitly is not decoration: the `SearchWorkspace` is allocated
    once and reused for every search, so without this reset the second search
    would inherit the first one's state. And the field that hurts most is
    `terminal`: if it arrives as 1, the mask in `update_particles` freezes the
    weight from depth 0 and NO particle accumulates anything. The weights all stay
    at zero, the readout's softmax comes out uniform and the search degenerates
    into picking at random among the root actions -- silently, without failing.
    """
    check_search_config(cfg)
    p_total = cfg.num_search_particles()

    # 0. Accumulators to zero. See the why in the docstring.
    particles.resample_td_weights.enqueue_fill(0)
    particles.gae.enqueue_fill(0)
    particles.terminal.enqueue_fill(0)
    particles.depth.enqueue_fill(0)

    # 1. We copy each environment's root state to all of its particles. Each
    # particle needs its own copy in order to simulate a different future.
    state_elems = p_total * cfg.state_dim   # Total components we copy.
    ctx.enqueue_function[broadcast_state_kernel, broadcast_state_kernel](
        particles.state.unsafe_ptr(), root_state.unsafe_ptr(),
        p_total, cfg.num_particles, cfg.state_dim,
        grid_dim=blocks_for(state_elems), block_dim=TPB)

    # 2. We also copy the value V(root state). The particles of one and the same
    # environment start with the same value because they still share a state.
    ctx.enqueue_function[broadcast_value_kernel, broadcast_value_kernel](
        particles.value.unsafe_ptr(), root_value.unsafe_ptr(),
        p_total, cfg.num_particles,
        grid_dim=blocks_for(p_total), block_dim=TPB)

    # 3. We copy to each particle the scores of every action given by the current
    # policy. They are logits, not probabilities nor old policies.
    logit_elems = p_total * cfg.num_actions
    ctx.enqueue_function[broadcast_logits_kernel, broadcast_logits_kernel](
        outputs.action_logits.unsafe_ptr(), root_logits.unsafe_ptr(),
        p_total, cfg.num_particles, cfg.num_actions,
        grid_dim=blocks_for(logit_elems), block_dim=TPB)

    # 4. We draw each particle's first action. They all start from the same
    # distribution, but they use different random numbers and may pick different
    # actions. The result is stored in root_actions.
    ctx.enqueue_function[categorical_from_logits[TPB], categorical_from_logits[TPB]](
        particles.root_actions.unsafe_ptr(), outputs.action_logits.unsafe_ptr(),
        uniforms.unsafe_ptr(), cfg.num_actions,
        grid_dim=p_total, block_dim=TPB)

    # 5. For each particle we store log(probability) of the action it has just
    # chosen under the original policy. This does not choose the action again.
    ctx.enqueue_function[log_prob_of_action_kernel, log_prob_of_action_kernel](
        particles.prior_logits.unsafe_ptr(), outputs.action_logits.unsafe_ptr(),
        particles.root_actions.unsafe_ptr(), p_total, cfg.num_actions,
        grid_dim=blocks_for(p_total), block_dim=TPB)

    # 6. We set up the first step: the root action is the one executed at depth 0.
    # root_actions is kept until the end; next_action will keep changing to name
    # the action to be executed at each depth.
    ctx.enqueue_function[copy_kernel[idx_dtype], copy_kernel[idx_dtype]](
        outputs.next_action.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        p_total, grid_dim=blocks_for(p_total), block_dim=TPB)


def sample_next_actions(ctx: DeviceContext, outputs: StepOutputs, cfg: SPOConfig,
                        uniforms: DeviceBuffer[dtype]) raises:
    """The generic half of recurrent_fn: choosing the next depth's action.

    In Stoix the `recurrent_fn` does four things: advance the environment,
    evaluate actor and critic at the new state, sample the next action, and fold
    in gamma/truncation. The model-dependent ones (advance, evaluate, fold) live
    in the model's kernel; these two, which are the same for everyone, live here.

    In comes `outputs.action_logits` [P, num_actions] (already written by the
    model over the NEW state) and out come `next_action` and `next_prior_logits`.

    `uniforms` has to bring P fresh values: one per particle. Reusing the previous
    depth's would correlate the trajectories.
    """
    p_total = cfg.num_search_particles()

    ctx.enqueue_function[categorical_from_logits[TPB], categorical_from_logits[TPB]](
        outputs.next_action.unsafe_ptr(), outputs.action_logits.unsafe_ptr(),
        uniforms.unsafe_ptr(), cfg.num_actions,
        grid_dim=p_total, block_dim=TPB)

    ctx.enqueue_function[log_prob_of_action_kernel, log_prob_of_action_kernel](
        outputs.next_prior_logits.unsafe_ptr(), outputs.action_logits.unsafe_ptr(),
        outputs.next_action.unsafe_ptr(), p_total, cfg.num_actions,
        grid_dim=blocks_for(p_total), block_dim=TPB)


def snapshot_root_values(ctx: DeviceContext, particles: Particles,
                         output: SPOOutput, cfg: SPOConfig) raises:
    """Saves V(s_root) right after seeding, before the rollout overwrites it.

    It is needed because `particles.value` moves along with the particles, but the
    public output reports the ROOT's value."""
    p_total = cfg.num_search_particles()
    ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
        output.root_values.unsafe_ptr(), particles.value.unsafe_ptr(), p_total,
        grid_dim=blocks_for(p_total), block_dim=TPB)
