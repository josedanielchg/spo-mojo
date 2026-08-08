"""Toy MDP: a corridor with two actions. The first SearchModel.

It serves to validate the SMC search before the networks or CartPole exist. It is
deterministic and its optimal policy is known by heart, so the tests can check
exact numbers instead of tolerances.

    positions:    0 --> 1 --> 2 --> ... --> L
    action GOOD (1): advances one cell and gives reward 1
    action BAD  (0): kills the episode right there, reward 0

Two different ways for an episode to end, and that is precisely the point:

  * A true TERMINAL (action BAD): there is no future. The real discount is 0, so
    the bootstrap_value is 0 too.
  * TRUNCATION (reaching the horizon): the episode is cut off by a time limit, but
    the state did have a future. The real discount is 1, the `rec_discount` the
    search sees is 0 (the particle stops simulating) but the bootstrap_value is
    not 0: it carries search_gamma * V(s').

Confusing those two cases is RL's classic silent bug, and it is the reason the toy
problem has both paths from day one. That is why the horizon (`horizon`) is
SMALLER than the corridor's length (`chain_length`): when truncating at cell
`horizon` there are still cells ahead and V(s') comes out non-zero, which is what
lets the test notice the difference.

Analytic value: (L - pos) good cells remain, each with reward 1, and search_gamma
is 1, so V(pos) = value_scale * (L - pos). With value_scale = 0 one gets V == 0,
which is the "no critic" case the CartPole demo will use.

This file does NOT import the search: only the types and the `SearchModel`
contract. The environment has no reason to know SPO exists, just as in Stoix,
where `envs/` does not know the algorithm and it is `ff_spo.py` that builds the
search on top.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from ops.common import dtype, idx_dtype, GlobalF32, GlobalI32
from systems.spo.particles import Particles, StepOutputs
from systems.spo.search_model import SearchModel
from systems.spo.spo_types import SPOConfig

# The two actions. The order matters for the tests: BAD is 0.
comptime ACTION_BAD = 0
comptime ACTION_GOOD = 1
comptime NUM_ACTIONS = 2

# The corridor is one-dimensional: the state is just the position.
comptime STATE_DIM = 1

comptime TPB_TOY = 32


def toy_value(pos: Scalar[dtype], chain_length: Int,
              value_scale: Scalar[dtype]) -> Scalar[dtype]:
    """V(pos) = value_scale * cells remaining. A device function: it is called
    both by the value kernel and by the recurrent one."""
    remaining = Scalar[dtype](chain_length) - pos
    if remaining < 0.0:
        remaining = 0.0
    return value_scale * remaining


def toy_value_kernel(value_out: GlobalF32, state: GlobalF32, n_particles: Int,
                     chain_length: Int, value_scale: Scalar[dtype]):
    """V(s) of each particle. One thread per particle."""
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n_particles:
        value_out[i] = toy_value(state[i * STATE_DIM], chain_length, value_scale)


def toy_policy_logits_kernel(logits_out: GlobalF32, state: GlobalF32,
                             n_particles: Int):
    """The prior at each particle's state.

    The toy problem's prior is UNIFORM, and that is deliberate: if the search
    improves a policy that knows nothing, the improvement comes from the search
    and from nothing else. All logits equal -> uniform softmax (the value does not
    matter, the softmax is invariant to adding a constant).
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n_particles:
        for a in range(NUM_ACTIONS):
            logits_out[i * NUM_ACTIONS + a] = Scalar[dtype](0)


def toy_recurrent_kernel(state: GlobalF32, action: GlobalI32,
                         reward_out: GlobalF32, discount_out: GlobalF32,
                         next_value_out: GlobalF32,
                         n_particles: Int, chain_length: Int, horizon: Int,
                         value_scale: Scalar[dtype], search_gamma: Scalar[dtype]):
    """Advances one cell and folds in gamma. The model's dynamics.

    It is the equivalent of Stoix's `recurrent_fn` for this model, including the
    two lines that are hardest to get right:

        rec_discount    = discount * (1 - truncated)
        bootstrap_value = discount * search_gamma * V(s')

    The state is updated in place. Dead particles stay put: there is no auto-reset
    inside the search, because a terminal particle no longer counts (the SMC
    core's terminal mask freezes its weight) and leaving it still makes the tests
    deterministic.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= n_particles:
        return

    pos = state[i * STATE_DIM]
    a = Int(action[i])

    # First the environment's dynamics.
    reward = Scalar[dtype](0)
    next_pos = pos
    discount_real = Scalar[dtype](1)   # the environment's "real" discount
    last = False                       # Stoix's timestep.last()

    if a == ACTION_GOOD:
        next_pos = pos + 1.0
        reward = 1.0
        # Truncation: time ran out, but the state had a future.
        if next_pos >= Scalar[dtype](horizon):
            last = True
            discount_real = 1.0
    else:
        # A true terminal: there is no future to value. reward stays at 0.
        last = True
        discount_real = 0.0

    # And now the same thing Stoix does with the timestep: something is truncated
    # if it ended but its real discount is not 0. A true terminal comes with
    # discount 0, so it never counts as truncated.
    truncated = last and discount_real != 0.0

    # The discount the search sees: 0 as soon as the particle stops simulating,
    # whether by death or by truncation. The SMC core uses it to mark terminal. It
    # is Stoix's `discount * (1 - truncated)`.
    rec_discount = discount_real * (Scalar[dtype](0) if truncated else Scalar[dtype](1))

    # The bootstrapping value uses the REAL discount, not the one above: that is
    # why a truncation keeps its value and a terminal does not.
    bootstrap_value = discount_real * search_gamma * toy_value(
        next_pos, chain_length, value_scale)

    state[i * STATE_DIM] = next_pos
    reward_out[i] = reward
    discount_out[i] = rec_discount
    next_value_out[i] = bootstrap_value


@fieldwise_init
struct ToyChain(SearchModel, Copyable, Movable):
    """The corridor as a search model.

    The struct stays on the host ALWAYS: what goes down to the GPU are its three
    loose numbers, inside the `enqueue_function` calls of its own methods. That is
    why the search can be generic over the model without any kernel crossing the
    boundary (see the contract in search_model.mojo).
    """

    var chain_length: Int
    """How many good cells there are in total. It defines the analytic value."""

    var horizon: Int
    """At which cell the episode is truncated. Smaller than chain_length so that
    truncation leaves future value uncollected."""

    var value_scale: Scalar[dtype]
    """1.0 = exact analytic value, 0.0 = V==0 (the no-critic case)."""

    def eval_root(self, ctx: DeviceContext, cfg: SPOConfig,
                  root_state: DeviceBuffer[dtype],
                  logits_out: DeviceBuffer[dtype],
                  value_out: DeviceBuffer[dtype]) raises:
        """The prior and V(s) over the root states, one per environment."""
        blocks = (cfg.num_envs + TPB_TOY - 1) // TPB_TOY

        ctx.enqueue_function[toy_policy_logits_kernel, toy_policy_logits_kernel](
            logits_out.unsafe_ptr(), root_state.unsafe_ptr(), cfg.num_envs,
            grid_dim=blocks, block_dim=TPB_TOY)

        ctx.enqueue_function[toy_value_kernel, toy_value_kernel](
            value_out.unsafe_ptr(), root_state.unsafe_ptr(), cfg.num_envs,
            self.chain_length, self.value_scale,
            grid_dim=blocks, block_dim=TPB_TOY)

    def step(self, ctx: DeviceContext, cfg: SPOConfig, particles: Particles,
             outputs: StepOutputs, step_uniforms: DeviceBuffer[dtype]) raises:
        """Advances the P particles by one depth.

        Two kernels: the dynamics (which already folds in gamma and the
        truncation) and the prior evaluated at the NEW state. Drawing the next
        action is not the model's business: that is handled by
        `sample_next_actions`, which is generic.

        It does not touch `particles.value`, and that matters: the TD error of the
        weight update needs the old V(s), and the new one stays in
        `outputs.next_value` until update_particles moves it.

        The toy problem is deterministic: it receives `step_uniforms` because of
        the contract but ignores them (only models with a stochastic transition
        use them).
        """
        p_total = cfg.num_search_particles()
        blocks = (p_total + TPB_TOY - 1) // TPB_TOY

        ctx.enqueue_function[toy_recurrent_kernel, toy_recurrent_kernel](
            particles.state.unsafe_ptr(), outputs.next_action.unsafe_ptr(),
            outputs.reward.unsafe_ptr(), outputs.discount.unsafe_ptr(),
            outputs.next_value.unsafe_ptr(), p_total,
            self.chain_length, self.horizon, self.value_scale, cfg.search_gamma,
            grid_dim=blocks, block_dim=TPB_TOY)

        ctx.enqueue_function[toy_policy_logits_kernel, toy_policy_logits_kernel](
            outputs.action_logits.unsafe_ptr(), particles.state.unsafe_ptr(),
            p_total, grid_dim=blocks, block_dim=TPB_TOY)


def default_toy_chain() -> ToyChain:
    """L=8, truncation at 4: on truncating 4 cells remain, that is V(4) = 4 != 0."""
    return ToyChain(chain_length=8, horizon=4, value_scale=1.0)
