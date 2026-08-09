"""Tic-Tac-Toe as a search model, but with a CRITIC instead of V = 0.

The same environment as `tictactoe.mojo` -- same rules, same kernels -- with the
only difference being where the value comes from:

    TicTacToe        V(s) = 0 always            the pure planner
    TicTacToeCritic  V(s) = whatever the net says   <- this one

Why it matters: with V = 0 the search only sees anything if the game ENDS within
its depth. If it does not end, every particle weighs the same and the search is
blind. A critic puts a value on the intermediate positions, so in principle it
should be able to tell "this looks bad" before the game is over.

It gets its own file so as not to touch `TicTacToe`, which is already tested and
still useful: the comparison between the two IS the E1.11 experiment.

The bootstrap follows the SearchModel contract, just like Stoix's `recurrent_fn`:

    next_value = discount_real * search_gamma * V(s')

that is, a particle that has just died carries no future value (its discount is 0)
and one that is still alive does.

The weights are a frozen COPY, not a reference to the training ones: the model
owns its buffers and refreshes them with `sync_from`. It costs six copies of a few
thousand floats, it happens once, and in exchange there are no two owners of the
same buffer and no doubt about whether the search is using half-updated weights.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from ops.buffers import zero_buffer
from ops.common import dtype, GlobalF32, GlobalI32
from ops.copy import copy_kernel
from envs.tictactoe import (ttt_prior_logits_kernel, ttt_dynamics_kernel,
                            ttt_encode_obs_kernel, OBS_DIM, TPB_TTT)
from networks.mlp import (CriticParams, CriticCache, critic_forward,
                          zero_critic_params)
from systems.spo.particles import Particles, StepOutputs
from systems.spo.search_model import SearchModel
from systems.spo.spo_types import SPOConfig


def bootstrap_kernel(next_value: GlobalF32, discount: GlobalF32,
                     value: GlobalF32, n: Int, search_gamma: Scalar[dtype]):
    """The bootstrap: next_value = discount * search_gamma * V(s'), one thread per
    particle.

    The discount already carries whether the particle died folded in, so
    multiplying by it is enough for a terminal state to carry no future.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p < n:
        next_value[p] = discount[p] * search_gamma * value[p]


def bootstrap_depth_kernel(next_value: GlobalF32, discount: GlobalF32,
                           value: GlobalF32, depth: GlobalI32, n: Int,
                           reward_gamma: Scalar[dtype]):
    """The bootstrap discounted by DEPTH: discount * gamma^(d+1) * V(s').

    It exists because of a scale mismatch that shows up when the search's weights
    are summed. The core does `weights += r_d + next_value_d - value_d` and then
    `value_d+1 = next_value_d`, so the sum telescopes:

        total weight = SUM_d r_d  +  (final bootstrap)  -  V(s_root)

    The reward `ttt_dynamics_kernel` produces already comes multiplied by gamma^d.
    If the bootstrap does NOT carry the same factor, the two halves of that sum
    are on different scales: with gamma=0.7 a win at depth 3 is worth 0.343 while
    a particle that is still alive carries V ~ 0.93. The search then concludes
    that winning is worse than not settling the game, which is exactly the
    opposite of what we want.

    With gamma^(d+1) the two halves are back on the same scale and the weight
    becomes the D-step discounted return with bootstrap, minus a constant:

        win at d        ->  gamma^d - V(s_root)
        stay alive      ->  gamma^D * V(s_D) - V(s_root)
        lose at d       ->  0 - V(s_root)

    that is, winning early > winning late > surviving > losing, which is the right
    order.

    `depth[p]` is the CURRENT depth (the core increments it afterwards), so the
    state being jumped to is at d+1: hence the exponent.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p >= n:
        return
    g = Scalar[dtype](1)
    for _ in range(Int(depth[p]) + 1):
        g *= reward_gamma
    next_value[p] = discount[p] * g * value[p]


struct TicTacToeCritic(SearchModel, Movable):
    """TTT with the value a trained network gives.

    It carries the weights and their working buffers inside. The struct lives on
    the host; what goes down to the GPU are the buffers, just as the toy problem
    sent down its three numbers.

    It is NOT `Copyable` (it owns `DeviceBuffer`s), unlike `TicTacToe`. The generic
    search accepts it all the same because it only reads it.
    """

    var reward_gamma: Scalar[dtype]
    """Depth discount on the reward, as in TicTacToe."""

    var params: CriticParams
    """The frozen copy of the weights. Refreshed with `sync_from`."""
    var cache: CriticCache
    """The forward's activations. Allocated for the largest use (P particles)."""
    var obs: DeviceBuffer[dtype]
    """[max_batch, OBS_DIM] the encoded board, reused on every call."""
    var hidden: Int

    var loss_penalty: Scalar[dtype]
    """What losing is worth, negated. See `TicTacToe.loss_penalty`."""

    var depth_discounted: Bool
    """Whether the bootstrap carries gamma^(d+1) (a scale consistent with the
    reward) or only `search_gamma` (the SearchModel's literal contract, like
    Stoix). See `bootstrap_depth_kernel` for why there are two."""

    def __init__(out self, ctx: DeviceContext, max_batch: Int, hidden: Int,
                 reward_gamma: Scalar[dtype],
                 depth_discounted: Bool = False,
                 loss_penalty: Scalar[dtype] = 0) raises:
        """Allocates everything for `max_batch` boards at a time.

        `max_batch` has to cover the larger of the two uses: `num_envs` at the
        root and `num_envs * num_particles` in the step. That is, the second.
        """
        self.reward_gamma = reward_gamma
        self.hidden = hidden
        self.depth_discounted = depth_discounted
        self.loss_penalty = loss_penalty
        self.params = zero_critic_params(ctx, OBS_DIM, hidden, 1)
        self.cache = CriticCache(ctx, max_batch, hidden, 1)
        self.obs = zero_buffer[dtype](ctx, max_batch * OBS_DIM)

    def sync_from(self, ctx: DeviceContext, src: CriticParams) raises:
        """Copies `src`'s six tensors into the model's weights, on the GPU.

        It is called when the search should see the current critic. In E1.11 once
        is enough, after training; in the full M-step loop it would be called
        every iteration.
        """
        if src.in_dim != OBS_DIM or src.hidden != self.hidden or src.out_dim != 1:
            raise Error("the critic being copied does not have the model's "
                        "shape: ", src.in_dim, "x", src.hidden, "x", src.out_dim)

        h = self.hidden
        self._copy(ctx, self.params.w1, src.w1, OBS_DIM * h)
        self._copy(ctx, self.params.b1, src.b1, h)
        self._copy(ctx, self.params.w2, src.w2, h * h)
        self._copy(ctx, self.params.b2, src.b2, h)
        self._copy(ctx, self.params.w3, src.w3, h)
        self._copy(ctx, self.params.b3, src.b3, 1)

    def _copy(self, ctx: DeviceContext, dst: DeviceBuffer[dtype],
              src: DeviceBuffer[dtype], n: Int) raises:
        ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
            dst.unsafe_ptr(), src.unsafe_ptr(), n,
            grid_dim=(n + 255) // 256, block_dim=256)

    def eval_root(self, ctx: DeviceContext, cfg: SPOConfig,
                  root_state: DeviceBuffer[dtype],
                  logits_out: DeviceBuffer[dtype],
                  value_out: DeviceBuffer[dtype]) raises:
        """Masked prior (same as without a critic) and V(s) from the network."""
        blocks = (cfg.num_envs + TPB_TTT - 1) // TPB_TTT

        ctx.enqueue_function[ttt_prior_logits_kernel, ttt_prior_logits_kernel](
            logits_out.unsafe_ptr(), root_state.unsafe_ptr(), cfg.num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)

        # The board in the format the network eats, and its value.
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            self.obs.unsafe_ptr(), root_state.unsafe_ptr(), cfg.num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
        critic_forward(ctx, self.params, self.cache, self.obs, cfg.num_envs)

        # V(s_root) goes straight to the output: there is no discount to fold in
        # here.
        ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
            value_out.unsafe_ptr(), self.cache.value.unsafe_ptr(), cfg.num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)

    def step(self, ctx: DeviceContext, cfg: SPOConfig, particles: Particles,
             outputs: StepOutputs, step_uniforms: DeviceBuffer[dtype]) raises:
        """Advances the particles and sets the bootstrap with the network's value."""
        p_total = cfg.num_search_particles()
        blocks = (p_total + TPB_TTT - 1) // TPB_TTT

        # 1. The usual dynamics. It leaves next_value at 0; it is overwritten in
        # step 3.
        ctx.enqueue_function[ttt_dynamics_kernel, ttt_dynamics_kernel](
            particles.state.unsafe_ptr(), outputs.next_action.unsafe_ptr(),
            step_uniforms.unsafe_ptr(), particles.depth.unsafe_ptr(),
            outputs.reward.unsafe_ptr(), outputs.discount.unsafe_ptr(),
            outputs.next_value.unsafe_ptr(), p_total, self.reward_gamma,
            self.loss_penalty, grid_dim=blocks, block_dim=TPB_TTT)

        # 2. V(s') over the NEW state.
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            self.obs.unsafe_ptr(), particles.state.unsafe_ptr(), p_total,
            grid_dim=blocks, block_dim=TPB_TTT)
        critic_forward(ctx, self.params, self.cache, self.obs, p_total)

        # 3. The bootstrap, with the discount already folded in and whichever
        # gamma applies.
        if self.depth_discounted:
            ctx.enqueue_function[bootstrap_depth_kernel, bootstrap_depth_kernel](
                outputs.next_value.unsafe_ptr(), outputs.discount.unsafe_ptr(),
                self.cache.value.unsafe_ptr(), particles.depth.unsafe_ptr(),
                p_total, self.reward_gamma,
                grid_dim=blocks, block_dim=TPB_TTT)
        else:
            ctx.enqueue_function[bootstrap_kernel, bootstrap_kernel](
                outputs.next_value.unsafe_ptr(), outputs.discount.unsafe_ptr(),
                self.cache.value.unsafe_ptr(), p_total, cfg.search_gamma,
                grid_dim=blocks, block_dim=TPB_TTT)

        # 4. And the prior at the new state, which is where the action is sampled
        # from.
        ctx.enqueue_function[ttt_prior_logits_kernel, ttt_prior_logits_kernel](
            outputs.action_logits.unsafe_ptr(), particles.state.unsafe_ptr(),
            p_total, grid_dim=blocks, block_dim=TPB_TTT)
