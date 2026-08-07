"""The full SMC search: SPO's E-step. Mirrors the SPO class in ff_spo.py.

This file only ORCHESTRATES. Each phase lives in its own module and they are
called here in order, which is how the algorithm reads:

    root.mojo         seed         N particles per env, one action each
    (the model)       advance      model.step(), the only environment-specific bit
    weighting.mojo    weigh        weight <- weight + (r + V' - V)
    metrics.mojo      measure      ESS and entropy, before resampling
    resampling.mojo   resample     every `resample_period` depths
    readout.mojo      read out     the improved policy q and the action to execute

Correspondence with Stoix, in case a side-by-side comparison is needed:

    search[M]()        = SPO.search + SPO.rollout
    smc_depth_close()  = the second half of one_step_rollout
    root.mojo          = make_root_fn
    weighting.mojo     = smc_weight_update_fn + calculate_gae + update_particles
    resampling.mojo    = get_resample_logits + resample
    metrics.mojo       = calculate_ess_and_entropy
    readout.mojo       = readout_weighted

`search[M]()` is the ONLY copy of the algorithm: there is no per-environment
version. What changes between models are `SearchModel`'s two methods.

Index convention throughout the E-step: particle p = env * num_particles + n. It
is flat on purpose, so that most kernels are a map of one thread per particle and
`env = p // num_particles` comes out with one division.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from ops.common import dtype
from ops.rng import fill_uniform
from systems.spo.launch import TPB, blocks_for
from systems.spo.metrics import compute_ess_entropy
from systems.spo.particles import (Particles, StepOutputs, SearchScratch,
                                   SPOOutput, SearchWorkspace)
from systems.spo.readout import readout_weighted
from systems.spo.resampling import resample
from systems.spo.root import root_fn, add_dirichlet_noise_kernel, sample_next_actions, snapshot_root_values
from systems.spo.search_model import SearchModel
from systems.spo.spo_types import SPOConfig
from systems.spo.weighting import update_particles


# RNG streams. Each use gets its own so that they do not share a sequence: if the
# action draw and the resampling draw came out of the same stream, the decisions
# would end up correlated. They live here and not in the model because the
# convention is a property of the SEARCH: that way the toy problem and CartPole
# are comparable with the same seed.
comptime RNG_ROOT = UInt32(0)
comptime RNG_ACTION = UInt32(100)
comptime RNG_STEP = UInt32(500)
comptime RNG_RESAMPLE = UInt32(900)
comptime RNG_READOUT = UInt32(7777)
comptime RNG_NOISE = UInt32(3333)
"""The Dirichlet noise's own stream: it shares a sequence with none of the
others."""


def smc_depth_close(ctx: DeviceContext, particles: Particles,
                    outputs: StepOutputs, scratch: SearchScratch,
                    output: SPOOutput, cfg: SPOConfig, depth: Int,
                    resample_uniforms: DeviceBuffer[dtype]) raises:
    """Everything that comes after the model's step within one depth.

    It is the generic part of Stoix's `one_step_rollout`: it serves equally for
    the toy problem, CartPole or the MLP.

    The order matters: the ESS is measured with the weights already updated but
    BEFORE resampling, which is where the collapse that justifies resampling shows
    up. Measuring it afterwards would always give a healthy number and would say
    nothing.
    """
    update_particles(ctx, particles, outputs, cfg)
    compute_ess_entropy(ctx, particles, scratch, output, cfg, depth)

    # Stoix's 'period' mode: resample when (depth+1) is a multiple of the period.
    if (depth + 1) % cfg.resample_period == 0:
        resample(ctx, particles, scratch, cfg, resample_uniforms)


def search[M: SearchModel](ctx: DeviceContext, ws: SearchWorkspace,
                           cfg: SPOConfig, model: M,
                           root_state: DeviceBuffer[dtype],
                           seed: UInt32) raises:
    """One complete SMC search: from the root states to the action to execute.

    The result is left in `ws.output`: the action per environment, the improved
    policy q (support + weights), the advantages for the temperature loss and the
    per-depth metrics.
    """
    p_total = cfg.num_search_particles()
    blocks_p = blocks_for(p_total)

    # The model evaluated at the root, one state per environment.
    model.eval_root(ctx, cfg, root_state, ws.root_logits, ws.root_value)

    # Exploration noise at the root, in the same place as Stoix: on the
    # prior_logits, BEFORE drawing the particles' actions. With
    # `dirichlet_fraction = 0` (Stoix's default) nothing is enqueued and the
    # search is bit for bit the one from before.
    if cfg.dirichlet_fraction > 0:
        n_cells = cfg.num_envs * cfg.num_actions
        ctx.enqueue_function[fill_uniform, fill_uniform](
            ws.u_noise.unsafe_ptr(), seed, RNG_NOISE, n_cells,
            grid_dim=blocks_for(n_cells), block_dim=TPB)
        ctx.enqueue_function[add_dirichlet_noise_kernel,
                             add_dirichlet_noise_kernel](
            ws.root_logits.unsafe_ptr(), ws.u_noise.unsafe_ptr(), cfg.num_envs,
            cfg.num_actions, cfg.dirichlet_fraction,
            grid_dim=blocks_for(cfg.num_envs), block_dim=TPB)

    ctx.enqueue_function[fill_uniform, fill_uniform](
        ws.u_action.unsafe_ptr(), seed, RNG_ROOT, p_total,
        grid_dim=blocks_p, block_dim=TPB)

    root_fn(ctx, ws.particles, ws.outputs, cfg, root_state, ws.root_logits,
            ws.root_value, ws.u_action)
    snapshot_root_values(ctx, ws.particles, ws.output, cfg)

    # The depth loop lives on the host but only enqueues kernels: there is not a
    # single synchronize inside, because the host does not need to read anything
    # until the end and the stream already executes them in order.
    for d in range(cfg.search_depth):
        ctx.enqueue_function[fill_uniform, fill_uniform](
            ws.u_step.unsafe_ptr(), seed, RNG_STEP + UInt32(d), p_total,
            grid_dim=blocks_p, block_dim=TPB)
        ctx.enqueue_function[fill_uniform, fill_uniform](
            ws.u_action.unsafe_ptr(), seed, RNG_ACTION + UInt32(d), p_total,
            grid_dim=blocks_p, block_dim=TPB)

        # The only two lines of the whole search that depend on the model.
        model.step(ctx, cfg, ws.particles, ws.outputs, ws.u_step)
        sample_next_actions(ctx, ws.outputs, cfg, ws.u_action)

        ctx.enqueue_function[fill_uniform, fill_uniform](
            ws.u_resample.unsafe_ptr(), seed, RNG_RESAMPLE + UInt32(d), p_total,
            grid_dim=blocks_p, block_dim=TPB)
        smc_depth_close(ctx, ws.particles, ws.outputs, ws.scratch, ws.output,
                        cfg, d, ws.u_resample)

    # And finally the action that actually gets executed.
    ctx.enqueue_function[fill_uniform, fill_uniform](
        ws.u_action.unsafe_ptr(), seed, RNG_READOUT, p_total,
        grid_dim=blocks_p, block_dim=TPB)
    readout_weighted(ctx, ws.particles, ws.scratch, ws.output, cfg, ws.u_action)
