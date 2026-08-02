"""La busqueda SMC completa: el E-step de SPO. Espeja la clase SPO de ff_spo.py.

Este fichero solo ORQUESTA. Cada fase vive en su propio modulo y aqui se llaman
en orden, que es como se lee el algoritmo:

    root.mojo         sembrar      N particulas por env, una accion cada una
    (el modelo)       avanzar      model.step(), lo unico especifico del entorno
    weighting.mojo    pesar        peso <- peso + (r + V' - V)
    metrics.mojo      medir        ESS y entropia, antes de resamplear
    resampling.mojo   remuestrear  cada `resample_period` profundidades
    readout.mojo      leer         la politica mejorada q y la accion a ejecutar

Correspondencia con Stoix, por si hace falta comparar lado a lado:

    search[M]()        = SPO.search + SPO.rollout
    smc_depth_close()  = la segunda mitad de one_step_rollout
    root.mojo          = make_root_fn
    weighting.mojo     = smc_weight_update_fn + calculate_gae + update_particles
    resampling.mojo    = get_resample_logits + resample
    metrics.mojo       = calculate_ess_and_entropy
    readout.mojo       = readout_weighted

`search[M]()` es la UNICA copia del algoritmo: no hay una version por entorno.
Lo que cambia entre modelos son los dos metodos de `SearchModel`.

Convencion de indices en todo el E-step: la particula p = env * num_particles + n.
Es plana a proposito, asi la mayoria de kernels son un map de un hilo por
particula y `env = p // num_particles` sale con una division.
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


# Streams del RNG. Cada uso tiene el suyo para que no compartan secuencia: si el
# sorteo de acciones y el del resampling salieran del mismo stream, las decisiones
# quedarian correlacionadas. Viven aqui y no en el modelo porque la convencion es
# una propiedad de la BUSQUEDA: asi el juguete y CartPole son comparables con la
# misma semilla.
comptime RNG_ROOT = UInt32(0)
comptime RNG_ACTION = UInt32(100)
comptime RNG_STEP = UInt32(500)
comptime RNG_RESAMPLE = UInt32(900)
comptime RNG_READOUT = UInt32(7777)
comptime RNG_NOISE = UInt32(3333)
"""Stream propio del ruido de Dirichlet: no comparte secuencia con
ninguno de los otros."""


def smc_depth_close(ctx: DeviceContext, particles: Particles,
                    outputs: StepOutputs, scratch: SearchScratch,
                    output: SPOOutput, cfg: SPOConfig, depth: Int,
                    resample_uniforms: DeviceBuffer[dtype]) raises:
    """Todo lo que va despues del paso del modelo en una profundidad.

    Es la parte generica de `one_step_rollout` de Stoix: sirve igual para el
    juguete, CartPole o el MLP.

    El orden importa: el ESS se mide con los pesos ya actualizados pero ANTES de
    resamplear, que es donde se ve el colapso que justifica el resampling. Medirlo
    despues daria siempre un numero sano y no diria nada.
    """
    update_particles(ctx, particles, outputs, cfg)
    compute_ess_entropy(ctx, particles, scratch, output, cfg, depth)

    # Modo 'period' de Stoix: resamplea cuando (depth+1) es multiplo del periodo.
    if (depth + 1) % cfg.resample_period == 0:
        resample(ctx, particles, scratch, cfg, resample_uniforms)


def search[M: SearchModel](ctx: DeviceContext, ws: SearchWorkspace,
                           cfg: SPOConfig, model: M,
                           root_state: DeviceBuffer[dtype],
                           seed: UInt32) raises:
    """Una busqueda SMC completa: de los estados raiz a la accion a ejecutar.

    El resultado queda en `ws.output`: la accion por entorno, la politica mejorada
    q (soporte + pesos), las ventajas para el loss de la temperatura y las
    metricas por profundidad.
    """
    p_total = cfg.num_search_particles()
    blocks_p = blocks_for(p_total)

    # El modelo evaluado en la raiz, un estado por entorno.
    model.eval_root(ctx, cfg, root_state, ws.root_logits, ws.root_value)

    # Ruido de exploracion en la raiz, en el mismo sitio que Stoix: sobre los
    # prior_logits, ANTES de sortear las acciones de las particulas. Con
    # `dirichlet_fraction = 0` (el defecto de Stoix) no se encola nada y la
    # busqueda es bit a bit la de antes.
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

    # El bucle de profundidad vive en el host pero solo encola kernels: no hay ni
    # un synchronize dentro, porque el host no necesita leer nada hasta el final y
    # el stream ya los ejecuta en orden.
    for d in range(cfg.search_depth):
        ctx.enqueue_function[fill_uniform, fill_uniform](
            ws.u_step.unsafe_ptr(), seed, RNG_STEP + UInt32(d), p_total,
            grid_dim=blocks_p, block_dim=TPB)
        ctx.enqueue_function[fill_uniform, fill_uniform](
            ws.u_action.unsafe_ptr(), seed, RNG_ACTION + UInt32(d), p_total,
            grid_dim=blocks_p, block_dim=TPB)

        # Las dos unicas lineas que dependen del modelo de toda la busqueda.
        model.step(ctx, cfg, ws.particles, ws.outputs, ws.u_step)
        sample_next_actions(ctx, ws.outputs, cfg, ws.u_action)

        ctx.enqueue_function[fill_uniform, fill_uniform](
            ws.u_resample.unsafe_ptr(), seed, RNG_RESAMPLE + UInt32(d), p_total,
            grid_dim=blocks_p, block_dim=TPB)
        smc_depth_close(ctx, ws.particles, ws.outputs, ws.scratch, ws.output,
                        cfg, d, ws.u_resample)

    # Y por ultimo la accion que se ejecuta de verdad.
    ctx.enqueue_function[fill_uniform, fill_uniform](
        ws.u_action.unsafe_ptr(), seed, RNG_READOUT, p_total,
        grid_dim=blocks_p, block_dim=TPB)
    readout_weighted(ctx, ws.particles, ws.scratch, ws.output, cfg, ws.u_action)
