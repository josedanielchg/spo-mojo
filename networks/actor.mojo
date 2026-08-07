"""The actor: the network that picks the move. 18 -> H -> H -> 9, with masking.

It is the piece the system was missing. Until now the search started from a
UNIFORM prior over the legal cells; the actor replaces it with a learned
distribution, and that closes the paper's EM loop:

    E-step   the search plans and produces q, the improved policy
    M-step   the actor learns to imitate q
    and on the next round the search starts from the actor, not from uniform

The NETWORK is exactly the same as the critic's except for the output: 9 logits
instead of 1 value. That is why the MLP is not reimplemented here --
`critic_forward` is already generic in `out_dim` and has been verified against
goldens and against autodiff since E1.3/E1.5. Reusing it is not laziness: it is
not re-risking code that already passes three independent checks.

What is genuinely new lives below, and it is the **masking**. In tic-tac-toe not
every action is legal, and a network has no way of knowing that on its own: if it
is not masked, it will learn to put probability on occupied cells and the search
will spend particles on impossible moves. Stoix brings none of this -- its
environments have no illegal actions -- so it is one of the pieces that have to be
added to take SPO to a board game.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from ops.buffers import zero_buffer
from ops.common import dtype, GlobalF32
from ops.softmax import softmax_rows, log_softmax_rows
from envs.tictactoe import NUM_ACTIONS, NEG_INF, OBS_DIM, ttt_legal_mask_kernel
from networks.mlp import (CriticParams, CriticCache, critic_forward,
                          zero_critic_params)
from systems.spo.launch import TPB, blocks_for

# The weights and activations are the same types as the critic's: same MLP,
# different output. The aliases exist so that the M-step's code reads as what it
# is ("the actor's weights") without duplicating structs.
comptime ActorParams = CriticParams
comptime ActorCache = CriticCache

comptime TPB_ACTOR = 32


def zero_actor_params(ctx: DeviceContext, hidden: Int) raises -> ActorParams:
    """Actor weights at zero: OBS_DIM -> hidden -> hidden -> NUM_ACTIONS."""
    return zero_critic_params(ctx, OBS_DIM, hidden, NUM_ACTIONS)


def mask_logits_kernel(logits: GlobalF32, mask: GlobalF32, n: Int,
                       num_actions: Int):
    """Masks the illegal actions' logits in place. One thread per (row, action).

    `mask` is 1.0 on the legal ones and 0.0 on the occupied ones, which is what
    `ttt_legal_mask_kernel` produces.

    NEG_INF (the most negative finite float32) is used and not a true -inf, for
    the same reason as `ttt_prior_logits_kernel`: if a whole row were masked (full
    board), with -inf the softmax would give nan and the nan would propagate
    silently to everything it touches afterwards. With MIN_FINITE the row
    degenerates to uniform, which in a terminal position makes no difference
    because nobody is going to play that action.

    In place on purpose: the logits buffer coming out of the MLP is already
    exactly the right size and no other one is needed.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= n * num_actions:
        return
    if mask[i] == Scalar[dtype](0):
        logits[i] = NEG_INF


def actor_logits(ctx: DeviceContext, params: ActorParams, cache: ActorCache,
                 obs: DeviceBuffer[dtype], mask: DeviceBuffer[dtype],
                 m: Int) raises:
    """Masked logits of m boards. They end up in `cache.value` [m, NUM_ACTIONS].

    Two steps: the MLP (already verified) and the masking. It does no softmax --
    whoever needs it asks for it separately, because the M-step works in log space
    and applying it here would force undoing it.
    """
    critic_forward(ctx, params, cache, obs, m)
    n_cells = m * params.out_dim
    ctx.enqueue_function[mask_logits_kernel, mask_logits_kernel](
        cache.value.unsafe_ptr(), mask.unsafe_ptr(), m, params.out_dim,
        grid_dim=blocks_for(n_cells), block_dim=TPB)


def actor_probs(ctx: DeviceContext, params: ActorParams, cache: ActorCache,
                obs: DeviceBuffer[dtype], mask: DeviceBuffer[dtype],
                probs_out: DeviceBuffer[dtype], m: Int) raises:
    """The actor's policy: pi(a|s) for m boards, already masked.

    The occupied cells come out with probability exactly 0, because
    exp(NEG_INF - max) underflows to 0 in float32. It is not an approximation that
    depends on the scale: NEG_INF is ~1e38 away from a logit's typical maximum,
    and exp of that is exactly zero.
    """
    actor_logits(ctx, params, cache, obs, mask, m)
    ctx.enqueue_function[softmax_rows[TPB_ACTOR], softmax_rows[TPB_ACTOR]](
        probs_out.unsafe_ptr(), cache.value.unsafe_ptr(), params.out_dim,
        grid_dim=m, block_dim=TPB_ACTOR)


def actor_log_probs(ctx: DeviceContext, params: ActorParams, cache: ActorCache,
                    obs: DeviceBuffer[dtype], mask: DeviceBuffer[dtype],
                    log_probs_out: DeviceBuffer[dtype], m: Int) raises:
    """log pi(a|s) for m boards. It is what the M-step's loss eats.

    It is not computed as log(actor_probs): a very low logit underflows the
    softmax to 0 and its log would be -inf even though the exact value was
    representable. The cross entropy weights precisely the small terms, so the
    log-softmax is computed directly (see `log_softmax_rows`).

    On the illegal cells a very negative finite value comes out (NEG_INF
    unchanged). The loss kernel skips those terms because their q is 0.
    """
    actor_logits(ctx, params, cache, obs, mask, m)
    ctx.enqueue_function[log_softmax_rows[TPB_ACTOR], log_softmax_rows[TPB_ACTOR]](
        log_probs_out.unsafe_ptr(), cache.value.unsafe_ptr(), params.out_dim,
        grid_dim=m, block_dim=TPB_ACTOR)


struct Actor(Movable):
    """The actor with its buffers, ready to be used from the search or the learner.

    It keeps the mask and the probabilities allocated for `max_batch` boards,
    which is what is needed both at the root (num_envs) and during training
    (batch * rollout).
    """

    var params: ActorParams
    var cache: ActorCache
    var mask: DeviceBuffer[dtype]
    """[max_batch, NUM_ACTIONS] 1 legal, 0 occupied."""
    var probs: DeviceBuffer[dtype]
    """[max_batch, NUM_ACTIONS] the policy, after the masked softmax."""
    var log_probs: DeviceBuffer[dtype]
    """[max_batch, NUM_ACTIONS] log pi, for the M-step's loss."""
    var hidden: Int
    var max_batch: Int
    """How many boards fit. Asking for more would write outside the buffers, so
    the methods check it instead of corrupting memory silently."""

    def __init__(out self, ctx: DeviceContext, max_batch: Int,
                 hidden: Int) raises:
        self.params = zero_actor_params(ctx, hidden)
        self.cache = ActorCache(ctx, max_batch, hidden, NUM_ACTIONS)
        self.mask = zero_buffer[dtype](ctx, max_batch * NUM_ACTIONS)
        self.probs = zero_buffer[dtype](ctx, max_batch * NUM_ACTIONS)
        self.log_probs = zero_buffer[dtype](ctx, max_batch * NUM_ACTIONS)
        self.hidden = hidden
        self.max_batch = max_batch

    def _check(self, m: Int) raises:
        """A loud error instead of an out-of-range write.

        Without this, asking for more boards than were allocated would overwrite
        someone else's memory and the symptom would show up much later, in another
        buffer. It costs one host-side comparison per call.
        """
        if m > self.max_batch:
            raise Error("el actor se reservo para ", self.max_batch,
                        " tableros y se le piden ", m)

    def mask_from_state(self, ctx: DeviceContext, state: DeviceBuffer[dtype],
                        m: Int) raises:
        """Fills the mask by reading the board. Legality is NOT passed in from
        outside: it is derived from the state, which is the only source of truth."""
        self._check(m)
        ctx.enqueue_function[ttt_legal_mask_kernel, ttt_legal_mask_kernel](
            self.mask.unsafe_ptr(), state.unsafe_ptr(), m,
            grid_dim=(m + TPB_ACTOR - 1) // TPB_ACTOR, block_dim=TPB_ACTOR)

    def forward(self, ctx: DeviceContext, state: DeviceBuffer[dtype],
                obs: DeviceBuffer[dtype], m: Int) raises:
        """From the board to the policy: mask, MLP, masking and softmax.

        `obs` has to arrive with the two-plane observation already computed (it is
        produced by `ttt_encode_obs_kernel`); the actor does not compute it so as
        not to duplicate that conversion, which already lives in the environment.
        """
        self._check(m)
        self.mask_from_state(ctx, state, m)
        actor_probs(ctx, self.params, self.cache, obs, self.mask, self.probs, m)

    def forward_log(self, ctx: DeviceContext, state: DeviceBuffer[dtype],
                    obs: DeviceBuffer[dtype], m: Int) raises:
        """From the board to log pi, which is what the M-step needs."""
        self._check(m)
        self.mask_from_state(ctx, state, m)
        actor_log_probs(ctx, self.params, self.cache, obs, self.mask,
                        self.log_probs, m)
