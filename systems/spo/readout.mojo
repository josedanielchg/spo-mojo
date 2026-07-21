"""Fase 5 de la busqueda: leer. Aqui sale la politica mejorada q.

Es el puente al M-step, y lo que hay que entender es que `q` no es un array de
probabilidades: son DOS cosas juntas.

    sampled_actions         las N acciones raiz que sobrevivieron, con
                            repeticiones si el resampling copio alguna varias veces
    sampled_action_weights  softmax(peso/temperatura) de cada una

Su histograma ponderado ES q, la ecuacion 6 del paper. La accion que se ejecuta
en el entorno real se sortea de ahi -- se muestrea, no se coge el argmax, y de
ese muestreo sale la exploracion.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from ops.common import dtype, idx_dtype, GlobalF32, GlobalI32
from ops.copy import copy_kernel
from ops.rng import categorical_from_logits
from ops.softmax import softmax_rows
from systems.spo.launch import TPB, TPB_PARTICLES, blocks_for
from systems.spo.particles import Particles, SearchScratch, SPOOutput
from systems.spo.resampling import resample_logits_kernel
from systems.spo.spo_types import SPOConfig


def select_action_kernel(action_out: GlobalI32, root_actions: GlobalI32,
                         chosen: GlobalI32, num_envs: Int, num_particles: Int):
    """La accion final del env es la accion RAIZ de la particula sorteada."""
    env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env < num_envs:
        action_out[env] = root_actions[env * num_particles + Int(chosen[env])]


def mean_over_particles_kernel(out_mean: GlobalF32, values: GlobalF32,
                               num_envs: Int, num_particles: Int):
    """Media por env. num_particles es 16, asi que un hilo por env va sobrado."""
    env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= num_envs:
        return
    total = Scalar[dtype](0)
    for n in range(num_particles):
        total += values[env * num_particles + n]
    out_mean[env] = total / Scalar[dtype](num_particles)


def readout_weighted(ctx: DeviceContext, particles: Particles,
                     scratch: SearchScratch, output: SPOOutput, cfg: SPOConfig,
                     uniforms: DeviceBuffer[dtype]) raises:
    """Lee el resultado de la busqueda. Espeja `readout_weighted` de Stoix.

    `uniforms` solo necesita num_envs valores (uno por env), pero se pasa un
    buffer de P por comodidad: se leen los primeros num_envs.
    """
    p_total = cfg.num_search_particles()
    blocks_p = blocks_for(p_total)

    # logits = peso / temperatura, los mismos que usa el resampling
    ctx.enqueue_function[resample_logits_kernel, resample_logits_kernel](
        scratch.resample_logits.unsafe_ptr(),
        particles.resample_td_weights.unsafe_ptr(),
        p_total, cfg.temperature, grid_dim=blocks_p, block_dim=TPB)

    # pesos normalizados de cada accion raiz
    ctx.enqueue_function[softmax_rows[TPB_PARTICLES], softmax_rows[TPB_PARTICLES]](
        output.sampled_action_weights.unsafe_ptr(),
        scratch.resample_logits.unsafe_ptr(), cfg.num_particles,
        grid_dim=cfg.num_envs, block_dim=TPB_PARTICLES)

    # una particula por env, sorteada con esos pesos
    ctx.enqueue_function[categorical_from_logits[TPB_PARTICLES],
                         categorical_from_logits[TPB_PARTICLES]](
        output.action.unsafe_ptr(), scratch.resample_logits.unsafe_ptr(),
        uniforms.unsafe_ptr(), cfg.num_particles,
        grid_dim=cfg.num_envs, block_dim=TPB_PARTICLES)

    # ...y su accion raiz es la que se ejecuta. Se reusa output.action como
    # destino: entra el indice de particula y sale la accion.
    ctx.enqueue_function[select_action_kernel, select_action_kernel](
        output.action.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        output.action.unsafe_ptr(), cfg.num_envs, cfg.num_particles,
        grid_dim=blocks_for(cfg.num_envs), block_dim=TPB)

    ctx.enqueue_function[copy_kernel[idx_dtype], copy_kernel[idx_dtype]](
        output.sampled_actions.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        p_total, grid_dim=blocks_p, block_dim=TPB)

    ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
        output.sampled_advantages.unsafe_ptr(), particles.gae.unsafe_ptr(),
        p_total, grid_dim=blocks_p, block_dim=TPB)

    # El valor de la raiz, no el del final del rollout: es lo que Stoix mete en
    # SPOOutput.value (jnp.mean(root.particle_values, axis=-1)).
    ctx.enqueue_function[mean_over_particles_kernel, mean_over_particles_kernel](
        output.value.unsafe_ptr(), output.root_values.unsafe_ptr(),
        cfg.num_envs, cfg.num_particles,
        grid_dim=blocks_for(cfg.num_envs), block_dim=TPB)
