"""Fase 3 de la busqueda: remuestrear. Convierte peso en multiplicidad.

Las particulas prometedoras se copian varias veces y las malas desaparecen.
Despues todas vuelven a tener el mismo peso, y la informacion de cual era mejor ya
no vive en los pesos sino en CUANTAS COPIAS hay de cada una. Eso es lo que evita
la degeneracion: sin resamplear, al cabo de unas profundidades una sola particula
se lleva toda la masa y las otras quince son ruido caro.

Cuatro kernels en cadena:

    resample_logits    peso / temperatura
    resample_indices   softmax -> CDF -> N sorteos por env
    gather_particles   dst[i] = src[idx[i]], a un buffer intermedio
    copy_back          scratch -> particulas, y pesos a cero
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
    """logits = pesos / temperatura, el `get_resample_logits` de Stoix.

    La temperatura decide lo agresiva que es la busqueda. Con una temperatura
    baja solo sobreviven las mejores particulas y el ESS se hunde; con una alta
    se reparte casi por igual y la busqueda apenas mejora al prior. Es el mismo
    eta que el M-step acabara aprendiendo.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p < n_particles:
        logits_out[p] = weights[p] / temperature


def resample_indices_kernel[TPB_P: Int](
        indices_out: GlobalI32, logits: GlobalF32, uniforms: GlobalF32,
        num_particles: Int):
    """Sortea N indices por env, con probabilidad softmax(logits) (CDF inversa).

    Un bloque por env y un hilo por particula. El CDF se construye una sola vez
    en shared memory y luego cada uno de los N hilos hace su busqueda con su
    propio uniforme, o sea N muestras del mismo CDF. Esa es la diferencia con
    `categorical_from_logits`, que saca una sola muestra por fila.

    Como en el muestreo de la fase 2, el CDF se deja sin normalizar y lo que se
    escala es el uniforme.
    """
    debug_assert(num_particles <= TPB_P,
                 "resample: las particulas de un env tienen que caber en el bloque")

    shared = stack_allocation[TPB_P, Scalar[dtype],
                              address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    env = Int(block_idx.x)
    base = env * num_particles

    # softmax estable -> CDF
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

    # Barrido lineal. Con 16 particulas no compensa una busqueda binaria y asi
    # el codigo dice exactamente lo que hace.
    chosen = num_particles - 1     # por defecto el ultimo, por si el redondeo
                                   # deja el target justo en el borde
    for n in range(num_particles):
        cdf_prev = shared[n - 1] if n > 0 else Scalar[dtype](0)
        if cdf_prev <= target and target < shared[n]:
            chosen = n
            break

    indices_out[base + tid] = Scalar[idx_dtype](chosen)


def gather_particles_kernel(
        # destino (los buffers de scratch)
        dst_state: GlobalF32, dst_root_actions: GlobalI32,
        dst_prior_logits: GlobalF32, dst_value: GlobalF32,
        dst_terminal: GlobalI32, dst_depth: GlobalI32,
        # origen (las particulas de verdad)
        src_state: GlobalF32, src_root_actions: GlobalI32,
        src_prior_logits: GlobalF32, src_value: GlobalF32,
        src_terminal: GlobalI32, src_depth: GlobalI32,
        indices: GlobalI32, n_particles: Int, num_particles: Int,
        state_dim: Int):
    """dst[i] = src[indices[i]], con los indices dentro del mismo env.

    Va a scratch y no in-place: si escribiera encima, un hilo podria pisar un
    hueco que otro todavia no ha leido. Es la carrera clasica del gather.
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
    """Devuelve el scratch a las particulas y resetea el peso a cero.

    El reset es parte del resampling: tras repartir las copias, todas las
    particulas vuelven a estar igual de bien consideradas.
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
    """Resampling completo. Espeja `resample` de Stoix.

    La gae no se toca, y conviene fijarse en eso. Stoix hace el gather de todo
    y luego reescribe la gae con la de antes del resampling, asi que en la
    practica se queda tal cual estaba. El motivo, segun su comentario, es que el
    loss de la temperatura necesita las ventajas de antes de resamplear para
    apuntar bien al KL; el precio es que la gae solo cubre hasta el ultimo
    resampling.

    Aqui sale mas simple: basta con no copiarla, que es lo mismo y ahorra un
    buffer.
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
