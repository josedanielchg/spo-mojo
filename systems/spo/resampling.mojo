"""Phase 3 of the search: resampling. It turns weight into multiplicity.

Promising particles get copied several times and bad ones disappear. Afterwards
they all have the same weight again, and the information about which one was
better no longer lives in the weights but in HOW MANY COPIES there are of each.
That is what avoids degeneracy: without resampling, after a few depths a single
particle takes all the mass and the other fifteen are expensive noise.

Four chained kernels:

    resample_logits    weight / temperature
    resample_indices   softmax -> CDF -> N draws per env
    gather_particles   dst[i] = src[idx[i]], into an intermediate buffer
    copy_back          scratch -> particles, and weights to zero
"""

from std.builtin.debug_assert import debug_assert
from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import exp
from std.memory import stack_allocation, AddressSpace

from ops.common import dtype, idx_dtype, NEG_INF, GlobalF32, GlobalI32
from ops.reductions import block_reduce_max
from ops.scan import block_scan_inclusive
from systems.spo.launch import TPB, TPB_PARTICLES, blocks_for
from systems.spo.particles import Particles, SearchScratch
from systems.spo.spo_types import SPOConfig


def resample_logits_kernel(logits_out: GlobalF32, weights: GlobalF32,
                           n_particles: Int, temperature: Scalar[dtype]):
    """logits = weights / temperature, Stoix's `get_resample_logits`.

    The temperature decides how aggressive the search is. With a low temperature
    only the best particles survive and the ESS collapses; with a high one the
    mass is spread almost evenly and the search barely improves on the prior. It
    is the same eta that the M-step will end up learning.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p < n_particles:
        logits_out[p] = weights[p] / temperature


def resample_indices_kernel[TPB_P: Int](
        indices_out: GlobalI32, logits: GlobalF32, uniforms: GlobalF32,
        num_particles: Int):
    """Draws N indices per env, with probability softmax(logits) (inverse CDF).

    One block per env and one thread per particle. The CDF is built once in shared
    memory and then each of the N threads does its own search with its own
    uniform, that is, N samples from the same CDF. That is the difference with
    `categorical_from_logits`, which draws a single sample per row.

    As in phase 2's sampling, the CDF is left unnormalised and what gets scaled is
    the uniform.
    """
    debug_assert(num_particles <= TPB_P,
                 "resample: an env's particles must fit in the block")

    shared = stack_allocation[TPB_P, Scalar[dtype],
                              address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    env = Int(block_idx.x)
    base = env * num_particles

    # stable softmax -> CDF
    shared[tid] = logits[base + tid] if tid < num_particles else NEG_INF
    barrier()
    row_max = block_reduce_max[TPB_P](shared, tid)

    shared[tid] = exp(logits[base + tid] - row_max) if tid < num_particles else Scalar[dtype](0)
    barrier()
    block_scan_inclusive[TPB_P](shared, tid)

    if tid >= num_particles:
        return

    total = shared[num_particles - 1]
    target = uniforms[base + tid] * total

    # Linear scan. With 16 particles a binary search does not pay off, and this
    # way the code says exactly what it does.
    chosen = num_particles - 1     # the last one by default, in case rounding
                                   # leaves the target right on the edge
    for n in range(num_particles):
        cdf_prev = shared[n - 1] if n > 0 else Scalar[dtype](0)
        if cdf_prev <= target and target < shared[n]:
            chosen = n
            break

    indices_out[base + tid] = Scalar[idx_dtype](chosen)


def gather_particles_kernel(
        # destination (the scratch buffers)
        dst_state: GlobalF32, dst_root_actions: GlobalI32,
        dst_prior_logits: GlobalF32, dst_value: GlobalF32,
        dst_terminal: GlobalI32, dst_depth: GlobalI32,
        # source (the real particles)
        src_state: GlobalF32, src_root_actions: GlobalI32,
        src_prior_logits: GlobalF32, src_value: GlobalF32,
        src_terminal: GlobalI32, src_depth: GlobalI32,
        indices: GlobalI32, n_particles: Int, num_particles: Int,
        state_dim: Int):
    """dst[i] = src[indices[i]], with the indices inside the same env.

    It goes to scratch and not in place: if it wrote on top, one thread could
    overwrite a slot another has not read yet. It is the classic gather race.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p >= n_particles:
        return

    env = p // num_particles
    src = env * num_particles + Int(indices[p])

    for d in range(state_dim):
        dst_state[p * state_dim + d] = src_state[src * state_dim + d]
    dst_root_actions[p] = src_root_actions[src]
    dst_prior_logits[p] = src_prior_logits[src]
    dst_value[p] = src_value[src]
    dst_terminal[p] = src_terminal[src]
    dst_depth[p] = src_depth[src]


def copy_back_kernel(
        dst_state: GlobalF32, dst_root_actions: GlobalI32,
        dst_prior_logits: GlobalF32, dst_value: GlobalF32,
        dst_terminal: GlobalI32, dst_depth: GlobalI32, dst_weights: GlobalF32,
        src_state: GlobalF32, src_root_actions: GlobalI32,
        src_prior_logits: GlobalF32, src_value: GlobalF32,
        src_terminal: GlobalI32, src_depth: GlobalI32,
        n_particles: Int, state_dim: Int):
    """Returns the scratch to the particles and resets the weight to zero.

    The reset is part of resampling: once the copies have been handed out, all the
    particles are equally well regarded again.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p >= n_particles:
        return

    for d in range(state_dim):
        dst_state[p * state_dim + d] = src_state[p * state_dim + d]
    dst_root_actions[p] = src_root_actions[p]
    dst_prior_logits[p] = src_prior_logits[p]
    dst_value[p] = src_value[p]
    dst_terminal[p] = src_terminal[p]
    dst_depth[p] = src_depth[p]
    dst_weights[p] = Scalar[dtype](0)


def resample(ctx: DeviceContext, particles: Particles, scratch: SearchScratch,
             cfg: SPOConfig, uniforms: DeviceBuffer[dtype]) raises:
    """Full resampling. Mirrors Stoix's `resample`.

    The gae is not touched, and that is worth noticing. Stoix gathers everything
    and then rewrites the gae with the pre-resampling one, so in practice it stays
    exactly as it was. The reason, according to their comment, is that the
    temperature loss needs the pre-resampling advantages in order to aim the KL
    correctly; the price is that the gae only covers up to the last resampling.

    Here it comes out simpler: it is enough not to copy it, which is the same
    thing and saves a buffer.
    """
    p_total = cfg.num_search_particles()
    blocks = blocks_for(p_total)

    ctx.enqueue_function[resample_logits_kernel, resample_logits_kernel](
        scratch.resample_logits.unsafe_ptr(),
        particles.resample_td_weights.unsafe_ptr(),
        p_total, cfg.temperature, grid_dim=blocks, block_dim=TPB)

    ctx.enqueue_function[resample_indices_kernel[TPB_PARTICLES],
                         resample_indices_kernel[TPB_PARTICLES]](
        scratch.indices.unsafe_ptr(), scratch.resample_logits.unsafe_ptr(),
        uniforms.unsafe_ptr(), cfg.num_particles,
        grid_dim=cfg.num_envs, block_dim=TPB_PARTICLES)

    ctx.enqueue_function[gather_particles_kernel, gather_particles_kernel](
        scratch.state.unsafe_ptr(), scratch.root_actions.unsafe_ptr(),
        scratch.prior_logits.unsafe_ptr(), scratch.value.unsafe_ptr(),
        scratch.terminal.unsafe_ptr(), scratch.depth.unsafe_ptr(),
        particles.state.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        particles.prior_logits.unsafe_ptr(), particles.value.unsafe_ptr(),
        particles.terminal.unsafe_ptr(), particles.depth.unsafe_ptr(),
        scratch.indices.unsafe_ptr(), p_total, cfg.num_particles, cfg.state_dim,
        grid_dim=blocks, block_dim=TPB)

    ctx.enqueue_function[copy_back_kernel, copy_back_kernel](
        particles.state.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        particles.prior_logits.unsafe_ptr(), particles.value.unsafe_ptr(),
        particles.terminal.unsafe_ptr(), particles.depth.unsafe_ptr(),
        particles.resample_td_weights.unsafe_ptr(),
        scratch.state.unsafe_ptr(), scratch.root_actions.unsafe_ptr(),
        scratch.prior_logits.unsafe_ptr(), scratch.value.unsafe_ptr(),
        scratch.terminal.unsafe_ptr(), scratch.depth.unsafe_ptr(),
        p_total, cfg.state_dim, grid_dim=blocks, block_dim=TPB)
