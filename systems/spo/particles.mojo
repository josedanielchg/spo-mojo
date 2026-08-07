"""The search's containers: the swarm, the step, the scratch and the output.

They all follow the same pattern: struct-of-arrays with flat `DeviceBuffer`s of
P = envs*particles elements, indexed by `p = env * num_particles + n`. Stoix uses
[NumEnvs, NumParticles, ...] arrays and lets JAX do the tree_map; here the flat
index makes most kernels a map of one thread per particle.

Four containers and one bundle:

    Particles       the swarm's state, what survives from one depth to the next
    StepOutputs     what ONE model step produces, rewritten every time
    SearchScratch   the resampling's intermediate buffers
    SPOOutput       the search's public result
    SearchWorkspace the previous four plus the uniforms, allocated once
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from systems.spo.spo_types import SPOConfig


struct Particles(Movable):
    """The P hypothetical trajectories: the swarm's state.

    The names are Stoix's on purpose (see `Particles` in ff_spo.py).
    """

    var state: DeviceBuffer[dtype]
    """[P, state_dim] the simulator's full state, not just the observation."""

    var root_actions: DeviceBuffer[idx_dtype]
    """[P] the depth-0 action. It is the only thing that ends up being executed."""

    var resample_td_weights: DeviceBuffer[dtype]
    """[P] sum of TD errors since the last resampling. Low weight, unpromising trajectory"""

    var prior_logits: DeviceBuffer[dtype]
    """[P] The log-probability of the chosen action is stored. log pi(chosen action | state)
    It will serve later to compare the original policy with the improved policy during the M-step."""

    var value: DeviceBuffer[dtype]
    """[P] V(s) of the particle's current state."""

    var terminal: DeviceBuffer[idx_dtype]
    """[P] 1 if the particle is already dead. It is sticky: once 1 it never goes back to 0."""

    var depth: DeviceBuffer[idx_dtype]
    """[P] how many steps it has advanced."""

    var gae: DeviceBuffer[dtype]
    """[P] advantage accumulated forwards. It feeds the temperature loss."""

    def __init__(out self, ctx: DeviceContext, config: SPOConfig) raises:
        p = config.num_search_particles()
        self.state = zero_buffer[dtype](ctx, p * config.state_dim)
        self.root_actions = zero_buffer[idx_dtype](ctx, p)
        self.resample_td_weights = zero_buffer[dtype](ctx, p)
        self.prior_logits = zero_buffer[dtype](ctx, p)
        self.value = zero_buffer[dtype](ctx, p)
        self.terminal = zero_buffer[idx_dtype](ctx, p)
        self.depth = zero_buffer[idx_dtype](ctx, p)
        self.gae = zero_buffer[dtype](ctx, p)


struct StepOutputs(Movable):
    """What one model step returns: Stoix's `SPORecurrentFnOutput`.

    They are working buffers rewritten at every depth; they sit outside Particles
    because they are not part of the particle's state, only of the step.
    """

    var reward: DeviceBuffer[dtype]
    """[P] the step's reward."""

    var discount: DeviceBuffer[dtype]
    """[P] Stoix's rec_discount: discount*(1-truncated). At 0 it marks "this
    particle stopped simulating", whether it really died or was truncated."""

    var next_value: DeviceBuffer[dtype]
    """[P] Stoix's bootstrap_value: discount_real * search_gamma * V(s').
    Mind the difference with the field above: on a truncation the discount is 0
    but this one is not, because the truncated state does have a future and its
    value has to be carried along."""

    var next_action: DeviceBuffer[idx_dtype]
    """[P] the action the particle will execute at the NEXT depth."""

    var next_prior_logits: DeviceBuffer[dtype]
    """[P] log-prob of that action under the prior."""

    var action_logits: DeviceBuffer[dtype]
    """[P, num_actions] the prior's logits at the new state. It is where
    next_action is sampled from, and it is stored because the sampling happens in
    another kernel."""

    def __init__(out self, ctx: DeviceContext, config: SPOConfig) raises:
        p = config.num_search_particles()
        self.reward = zero_buffer[dtype](ctx, p)
        self.discount = zero_buffer[dtype](ctx, p)
        self.next_value = zero_buffer[dtype](ctx, p)
        self.next_action = zero_buffer[idx_dtype](ctx, p)
        self.next_prior_logits = zero_buffer[dtype](ctx, p)
        self.action_logits = zero_buffer[dtype](ctx, p * config.num_actions)


struct SearchScratch(Movable):
    """The resampling's auxiliary buffers.

    Resampling is a gather: particle i becomes a copy of particle idx[i]. Doing it
    in place would be a textbook race, because one thread can write its
    destination before another has read that very slot, so an intermediate buffer
    per field is needed.

    Only the six fields that get copied are here. `resample_td_weights` is not
    needed because it is reset to zero, and neither is `gae` because it is
    preserved without reordering (see the note in resampling.mojo).
    """

    var state: DeviceBuffer[dtype]
    var root_actions: DeviceBuffer[idx_dtype]
    var prior_logits: DeviceBuffer[dtype]
    var value: DeviceBuffer[dtype]
    var terminal: DeviceBuffer[idx_dtype]
    var depth: DeviceBuffer[idx_dtype]

    var indices: DeviceBuffer[idx_dtype]
    """[P] which particle each slot copies."""

    var resample_logits: DeviceBuffer[dtype]
    """[P] the SMC weights divided by the temperature."""

    def __init__(out self, ctx: DeviceContext, config: SPOConfig) raises:
        p = config.num_search_particles()
        self.state = zero_buffer[dtype](ctx, p * config.state_dim)
        self.root_actions = zero_buffer[idx_dtype](ctx, p)
        self.prior_logits = zero_buffer[dtype](ctx, p)
        self.value = zero_buffer[dtype](ctx, p)
        self.terminal = zero_buffer[idx_dtype](ctx, p)
        self.depth = zero_buffer[idx_dtype](ctx, p)
        self.indices = zero_buffer[idx_dtype](ctx, p)
        self.resample_logits = zero_buffer[dtype](ctx, p)


struct SPOOutput(Movable):
    """The public result of one search. Mirrors SPOOutput from spo_types.py."""

    var action: DeviceBuffer[idx_dtype]
    """[num_envs] the action actually executed in the environment. One per env only"""

    var sampled_actions: DeviceBuffer[idx_dtype]
    """[P] the N root actions that survived. Their histogram is the improved
    policy q, which is exactly what the M-step tries to imitate."""

    var sampled_action_weights: DeviceBuffer[dtype]
    """[P] the weight of each one: softmax(w/temperature) per env."""

    var value: DeviceBuffer[dtype]
    """[num_envs] the mean of V(s_root) over the particles."""

    var sampled_advantages: DeviceBuffer[dtype]
    """[P] each particle's gae. It feeds the temperature loss."""

    var root_values: DeviceBuffer[dtype]
    """[P] copy of V(s_root) per particle, saved before the rollout overwrites
    particles.value."""

    var ess: DeviceBuffer[dtype]
    """[search_depth, num_envs] effective sample size at each depth."""

    var entropy: DeviceBuffer[dtype]
    """[search_depth, num_envs] entropy of the weights at each depth."""

    def __init__(out self, ctx: DeviceContext, config: SPOConfig) raises:
        p = config.num_search_particles()
        metrics = config.search_depth * config.num_envs
        self.action = zero_buffer[idx_dtype](ctx, config.num_envs)
        self.sampled_actions = zero_buffer[idx_dtype](ctx, p)
        self.sampled_action_weights = zero_buffer[dtype](ctx, p)
        self.value = zero_buffer[dtype](ctx, config.num_envs)
        self.sampled_advantages = zero_buffer[dtype](ctx, p)
        self.root_values = zero_buffer[dtype](ctx, p)
        self.ess = zero_buffer[dtype](ctx, metrics)
        self.entropy = zero_buffer[dtype](ctx, metrics)


struct SearchWorkspace(Movable):
    """All the memory one search needs, allocated ONCE.

    Each search used to allocate its own uniform and root buffers on entry and
    release them on exit. On the toy problem it made no difference, but in phase 8
    the search runs at every environment step, and allocating device memory
    thousands of times is pure work for nothing. Here the workspace is built once
    and reused across every call to `search`.
    """

    var particles: Particles
    var outputs: StepOutputs
    var scratch: SearchScratch
    var output: SPOOutput

    var root_logits: DeviceBuffer[dtype]
    """[num_envs, num_actions] the prior evaluated at the root states."""

    var root_value: DeviceBuffer[dtype]
    """[num_envs] V(s_root)."""

    var u_action: DeviceBuffer[dtype]
    """[P] uniforms to draw actions with. It is refilled at every depth; it can be
    reused without fear because the stream is unique and executes in order: the
    next fill cannot overtake the kernel that read the previous one."""

    var u_step: DeviceBuffer[dtype]
    """[P] uniforms for the model's stochastic transition (e.g. the random
    rival's move in TTT). One per particle, refilled at every depth from its own
    stream. A deterministic model ignores them."""

    var u_noise: DeviceBuffer[dtype]
    """[num_envs, num_actions] uniforms of the Dirichlet noise at the root. It
    gets its own buffer so that the noise does not share a sequence with the
    action sampling: if they shared one, the exploration the noise is trying to
    add would be correlated with what was already sampled."""

    var u_resample: DeviceBuffer[dtype]
    """[P] resampling uniforms, in a separate buffer so that action sampling and
    resampling do not share a sequence."""

    def __init__(out self, ctx: DeviceContext, cfg: SPOConfig) raises:
        p = cfg.num_search_particles()
        self.particles = Particles(ctx, cfg)
        self.outputs = StepOutputs(ctx, cfg)
        self.scratch = SearchScratch(ctx, cfg)
        self.output = SPOOutput(ctx, cfg)
        self.root_logits = zero_buffer[dtype](ctx, cfg.num_envs * cfg.num_actions)
        self.root_value = zero_buffer[dtype](ctx, cfg.num_envs)
        self.u_action = zero_buffer[dtype](ctx, p)
        self.u_step = zero_buffer[dtype](ctx, p)
        self.u_noise = zero_buffer[dtype](ctx, cfg.num_envs * cfg.num_actions)
        self.u_resample = zero_buffer[dtype](ctx, p)
