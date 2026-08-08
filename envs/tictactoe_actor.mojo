"""Tic-Tac-Toe with the actor's PRIOR. This is where the EM loop closes.

Until now the search started from a uniform prior over the legal cells: it had no
opinion at all about where to look. This model replaces that with whatever the
network says, and with it the paper's loop closes:

    E-step   the search plans starting from pi and produces q, better than pi
    M-step   the actor learns to imitate q, that is, pi moves towards q
    and on the next round the search starts from a better pi

Without this piece the M-step trains a network nobody uses. With it, each
iteration improves the next one's starting point, which is where the method's
compounding improvement comes from.

**How the prior comes in, exactly.** It was checked in `weighting.mojo:83`: the
`prior_logits` are only HANDED OVER from one depth to the next, they never appear
in the SMC weight. That is, the prior does not change how a particle is scored,
only WHICH ACTIONS get sampled. It is equation 10's simplification in the paper:
when the proposal distribution is the policy itself, the ratio pi/pi cancels and
what is left is `w propto w * exp(A/eta)`. And it is exactly the role AlphaZero's
prior plays: steering the search, not valuing it.

The weights are a frozen copy via `sync_from`, just as in `tictactoe_critic.mojo`
and for the same reason: two owners of the same `DeviceBuffer` in Mojo is a mess,
and the copy makes it clear at which moment the search starts seeing the new
actor.

**The critic is optional but is ON by default**, and that is a correction with
respect to this file's first version. V is in equation 10 of the paper

    A(s_t,a_t) = r_t + V(s_{t+1}) - V(s_t)

and in Stoix's `_critic_loss_fn`, so having it disconnected was a deviation of
ours, not a choice of the method. E1.11 measured that it did not help, but it
measured it with a critic trained on data from a PLANNER that lost 2% and started
from a uniform prior. Now the data comes from a much stronger agent that visits
other positions, so the old measurement does not apply and has to be repeated.

The critic's two bootstrap modes are E1.11's:
  - `discount * search_gamma * V(s')`, the literal contract of the SearchModel and
    of Stoix;
  - `discount * gamma_r^(d+1) * V(s')`, which is needed if `reward_gamma` < 1
    because then the reward carries gamma_r^d folded in and the value has to be on
    the same scale (see `bootstrap_depth_kernel`).
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from ops.buffers import zero_buffer
from ops.common import dtype
from ops.copy import copy_kernel
from envs.tictactoe import (ttt_dynamics_kernel, ttt_encode_obs_kernel,
                            ttt_legal_mask_kernel, NUM_ACTIONS, OBS_DIM,
                            TPB_TTT)
from envs.tictactoe_critic import bootstrap_kernel, bootstrap_depth_kernel
from networks.actor import ActorParams, ActorCache, actor_logits, zero_actor_params
from networks.mlp import (CriticParams, CriticCache, critic_forward,
                          zero_critic_params)
from systems.spo.particles import Particles, StepOutputs
from systems.spo.search_model import SearchModel
from systems.spo.spo_types import SPOConfig


struct TicTacToeActor(SearchModel, Movable):
    """TTT where the search's prior is set by a trained network."""

    var reward_gamma: Scalar[dtype]
    var loss_penalty: Scalar[dtype]

    var params: ActorParams
    """Frozen copy of the actor's weights. Refreshed with `sync_from`."""
    var cache: ActorCache
    var obs: DeviceBuffer[dtype]
    """[max_batch, OBS_DIM] the encoded board."""
    var mask: DeviceBuffer[dtype]
    """[max_batch, NUM_ACTIONS] 1 legal, 0 occupied."""
    var hidden: Int
    var max_batch: Int

    var critic: CriticParams
    """The critic's weights, also as a frozen copy."""
    var ccache: CriticCache
    var use_critic: Bool
    """Whether V comes from the network or stays at 0."""
    var depth_discounted: Bool
    """Whether the bootstrap carries gamma_r^(d+1) instead of only `search_gamma`.
    It is needed when reward_gamma < 1; see `bootstrap_depth_kernel`."""

    def __init__(out self, ctx: DeviceContext, max_batch: Int, hidden: Int,
                 reward_gamma: Scalar[dtype],
                 loss_penalty: Scalar[dtype] = 0,
                 use_critic: Bool = False,
                 depth_discounted: Bool = False) raises:
        """`max_batch` has to cover the largest use: num_envs * num_particles."""
        self.reward_gamma = reward_gamma
        self.loss_penalty = loss_penalty
        self.hidden = hidden
        self.max_batch = max_batch
        self.use_critic = use_critic
        self.depth_discounted = depth_discounted
        self.params = zero_actor_params(ctx, hidden)
        self.cache = ActorCache(ctx, max_batch, hidden, NUM_ACTIONS)
        self.obs = zero_buffer[dtype](ctx, max_batch * OBS_DIM)
        self.mask = zero_buffer[dtype](ctx, max_batch * NUM_ACTIONS)
        self.critic = zero_critic_params(ctx, OBS_DIM, hidden, 1)
        self.ccache = CriticCache(ctx, max_batch, hidden, 1)

    def sync_critic_from(self, ctx: DeviceContext, src: CriticParams) raises:
        """Brings in the weights of the critic being trained."""
        if src.in_dim != OBS_DIM or src.hidden != self.hidden or src.out_dim != 1:
            raise Error("el critico no tiene la forma del modelo: ", src.in_dim,
                        "x", src.hidden, "x", src.out_dim)
        h = self.hidden
        self._copy(ctx, self.critic.w1, src.w1, OBS_DIM * h)
        self._copy(ctx, self.critic.b1, src.b1, h)
        self._copy(ctx, self.critic.w2, src.w2, h * h)
        self._copy(ctx, self.critic.b2, src.b2, h)
        self._copy(ctx, self.critic.w3, src.w3, h)
        self._copy(ctx, self.critic.b3, src.b3, 1)

    def sync_from(self, ctx: DeviceContext, src: CriticParams) raises:
        """Brings in the weights of the actor being trained.

        Calling it between iterations of the EM loop is precisely what makes the
        next E-step start from a better policy. If it were not called, the search
        would keep the constructor's zeros -- a uniform prior in disguise, and the
        loop would not close.
        """
        if src.in_dim != OBS_DIM or src.hidden != self.hidden \
                or src.out_dim != NUM_ACTIONS:
            raise Error("el actor que se intenta copiar no tiene la forma del "
                        "modelo: ", src.in_dim, "x", src.hidden, "x",
                        src.out_dim)
        h = self.hidden
        self._copy(ctx, self.params.w1, src.w1, OBS_DIM * h)
        self._copy(ctx, self.params.b1, src.b1, h)
        self._copy(ctx, self.params.w2, src.w2, h * h)
        self._copy(ctx, self.params.b2, src.b2, h)
        self._copy(ctx, self.params.w3, src.w3, h * NUM_ACTIONS)
        self._copy(ctx, self.params.b3, src.b3, NUM_ACTIONS)

    def _copy(self, ctx: DeviceContext, dst: DeviceBuffer[dtype],
              src: DeviceBuffer[dtype], n: Int) raises:
        ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
            dst.unsafe_ptr(), src.unsafe_ptr(), n,
            grid_dim=(n + 255) // 256, block_dim=256)

    def _prior(self, ctx: DeviceContext, state: DeviceBuffer[dtype],
               out_logits: DeviceBuffer[dtype], m: Int) raises:
        """The actor's logits over m boards, masked, into `out_logits`.

        Three steps: the mask and the observation come out of the STATE (the only
        source of truth about what is legal), and the network sets the logits.
        """
        if m > self.max_batch:
            raise Error("el modelo se reservo para ", self.max_batch,
                        " tableros y se le piden ", m)
        blocks = (m + TPB_TTT - 1) // TPB_TTT

        ctx.enqueue_function[ttt_legal_mask_kernel, ttt_legal_mask_kernel](
            self.mask.unsafe_ptr(), state.unsafe_ptr(), m,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            self.obs.unsafe_ptr(), state.unsafe_ptr(), m,
            grid_dim=blocks, block_dim=TPB_TTT)
        actor_logits(ctx, self.params, self.cache, self.obs, self.mask, m)

        n_cells = m * NUM_ACTIONS
        ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
            out_logits.unsafe_ptr(), self.cache.value.unsafe_ptr(), n_cells,
            grid_dim=(n_cells + 255) // 256, block_dim=256)

    def eval_root(self, ctx: DeviceContext, cfg: SPOConfig,
                  root_state: DeviceBuffer[dtype],
                  logits_out: DeviceBuffer[dtype],
                  value_out: DeviceBuffer[dtype]) raises:
        """The NETWORK's prior at the root states, and V from the critic (or 0)."""
        self._prior(ctx, root_state, logits_out, cfg.num_envs)
        if not self.use_critic:
            # It is overwritten explicitly because the workspace is reused between
            # searches and could carry stale values (the A6 bug).
            value_out.enqueue_fill(0)
            return
        # `_prior` already left the encoded observation in self.obs, so there is no
        # need to recompute it.
        critic_forward(ctx, self.critic, self.ccache, self.obs, cfg.num_envs)
        n = cfg.num_envs
        ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
            value_out.unsafe_ptr(), self.ccache.value.unsafe_ptr(), n,
            grid_dim=(n + 255) // 256, block_dim=256)

    def step(self, ctx: DeviceContext, cfg: SPOConfig, particles: Particles,
             outputs: StepOutputs, step_uniforms: DeviceBuffer[dtype]) raises:
        """Advances the particles and evaluates the network's prior at the NEW state."""
        p_total = cfg.num_search_particles()
        blocks = (p_total + TPB_TTT - 1) // TPB_TTT

        ctx.enqueue_function[ttt_dynamics_kernel, ttt_dynamics_kernel](
            particles.state.unsafe_ptr(), outputs.next_action.unsafe_ptr(),
            step_uniforms.unsafe_ptr(), particles.depth.unsafe_ptr(),
            outputs.reward.unsafe_ptr(), outputs.discount.unsafe_ptr(),
            outputs.next_value.unsafe_ptr(), p_total, self.reward_gamma,
            self.loss_penalty, grid_dim=blocks, block_dim=TPB_TTT)

        self._prior(ctx, particles.state, outputs.action_logits, p_total)

        if self.use_critic:
            # self.obs already holds the NEW state encoded, from `_prior`.
            critic_forward(ctx, self.critic, self.ccache, self.obs, p_total)
            if self.depth_discounted:
                ctx.enqueue_function[bootstrap_depth_kernel,
                                     bootstrap_depth_kernel](
                    outputs.next_value.unsafe_ptr(),
                    outputs.discount.unsafe_ptr(),
                    self.ccache.value.unsafe_ptr(),
                    particles.depth.unsafe_ptr(), p_total, self.reward_gamma,
                    grid_dim=blocks, block_dim=TPB_TTT)
            else:
                ctx.enqueue_function[bootstrap_kernel, bootstrap_kernel](
                    outputs.next_value.unsafe_ptr(),
                    outputs.discount.unsafe_ptr(),
                    self.ccache.value.unsafe_ptr(), p_total, cfg.search_gamma,
                    grid_dim=blocks, block_dim=TPB_TTT)
