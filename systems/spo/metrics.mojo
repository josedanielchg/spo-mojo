"""Phase 4 of the search: measuring. ESS and entropy of the weights, per environment.

They change nothing about the search: they are the diagnostic. But they are THE
diagnostic, and without them there is no way of knowing whether the swarm is
healthy or whether fifteen of the sixteen particles are noise.

    ESS = 1 / sum(w_i^2)   how many particles really contribute
    entropy = -sum(w log w)

With uniform weights the ESS is N, because they all count equally; when one weight
takes everything it drops to 1 and the search has collapsed. In the demo it can be
seen falling between resamplings and recovering right afterwards, which is exactly
the curve one expects from SMC.

In Stoix this is also the trigger for the `ess` resampling mode, which is not
implemented here (we use the `period` mode).
"""

from std.gpu import block_idx, thread_idx, barrier
from std.gpu.host import DeviceContext
from std.math import exp, log
from std.memory import stack_allocation, AddressSpace

from ops.common import dtype, NEG_INF, GlobalF32
from ops.reductions import block_reduce_max, block_reduce_sum
from systems.spo.launch import TPB, TPB_PARTICLES, blocks_for
from systems.spo.particles import Particles, SearchScratch, SPOOutput
from systems.spo.resampling import resample_logits_kernel
from systems.spo.spo_types import SPOConfig


def ess_entropy_kernel[TPB_P: Int](ess_out: GlobalF32, entropy_out: GlobalF32,
                                   logits: GlobalF32, num_particles: Int):
    """ESS and entropy of the normalised weights. One block per env, one thread per particle."""
    shared = stack_allocation[TPB_P, Scalar[dtype],
                              address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    env = Int(block_idx.x)
    base = env * num_particles

    active = tid < num_particles

    # stable softmax of the logits
    shared[tid] = logits[base + tid] if active else NEG_INF
    barrier()
    row_max = block_reduce_max[TPB_P](shared, tid)

    e = exp(logits[base + tid] - row_max) if active else Scalar[dtype](0)
    shared[tid] = e
    barrier()
    total = block_reduce_sum[TPB_P](shared, tid)

    w = e / total if active else Scalar[dtype](0)

    # sum(w^2) -> ESS
    shared[tid] = w * w
    barrier()
    sum_sq = block_reduce_sum[TPB_P](shared, tid)

    # -sum(w log w) -> entropy. The TINY avoids log(0) on the particles that were
    # left without mass; Stoix uses the same trick with finfo.tiny.
    TINY = Scalar[dtype](1e-30)
    shared[tid] = -w * log(w + TINY) if active else Scalar[dtype](0)
    barrier()
    ent = block_reduce_sum[TPB_P](shared, tid)

    if tid == 0:
        ess_out[env] = Scalar[dtype](1) / sum_sq
        entropy_out[env] = ent


def compute_ess_entropy(ctx: DeviceContext, particles: Particles,
                        scratch: SearchScratch, output: SPOOutput,
                        cfg: SPOConfig, depth: Int) raises:
    """Measures ESS and entropy with the current weights and stores them in row `depth`."""
    p_total = cfg.num_search_particles()
    ctx.enqueue_function[resample_logits_kernel, resample_logits_kernel](
        scratch.resample_logits.unsafe_ptr(),
        particles.resample_td_weights.unsafe_ptr(),
        p_total, cfg.temperature,
        grid_dim=blocks_for(p_total), block_dim=TPB)

    row = depth * cfg.num_envs
    ctx.enqueue_function[ess_entropy_kernel[TPB_PARTICLES],
                         ess_entropy_kernel[TPB_PARTICLES]](
        output.ess.unsafe_ptr() + row, output.entropy.unsafe_ptr() + row,
        scratch.resample_logits.unsafe_ptr(), cfg.num_particles,
        grid_dim=cfg.num_envs, block_dim=TPB_PARTICLES)
