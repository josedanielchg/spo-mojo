"""Fase 4 de la busqueda: medir. ESS y entropia de los pesos, por entorno.

No cambian nada de la busqueda: son el diagnostico. Pero son EL diagnostico, y
sin ellos no hay forma de saber si el enjambre esta sano o si quince de las
dieciseis particulas son ruido.

    ESS = 1 / sum(w_i^2)   cuantas particulas aportan de verdad
    entropia = -sum(w log w)

Con pesos uniformes el ESS vale N, porque todas cuentan igual; cuando un peso se
lo lleva todo baja a 1 y la busqueda ha colapsado. En la demo se ve caer entre
resamplings y recuperarse justo despues, que es exactamente la curva que uno
espera de SMC.

En Stoix esto es tambien el disparador del modo de resampling por `ess`, que aqui
no esta implementado (usamos el modo `period`).
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
    """ESS y entropia de los pesos normalizados. Un bloque por env, un hilo por particula."""
    shared = stack_allocation[TPB_P, Scalar[dtype],
                              address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    env = Int(block_idx.x)
    base = env * num_particles

    active = tid < num_particles

    # softmax estable de los logits
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

    # -sum(w log w) -> entropia. El TINY evita log(0) en las particulas que se
    # quedaron sin masa; Stoix usa el mismo truco con finfo.tiny.
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
    """Mide ESS y entropia con los pesos de ahora y los guarda en la fila `depth`."""
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
