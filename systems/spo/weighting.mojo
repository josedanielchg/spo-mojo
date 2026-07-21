"""Fase 2 de la busqueda: pesar. La ecuacion 10 del paper, en log-space.

Es el corazon del SMC: despues de que el modelo avance las particulas, aqui se
decide cuanta evidencia ha acumulado cada una a favor de su accion raiz.

    peso <- peso + (r + V' - V)      con mascara de particulas muertas

El paper lo escribe como un producto, `w <- w * exp(A/eta)`. Guardar la SUMA de
ventajas y diferir el `exp(.../eta)` al softmax del resampling es lo mismo
matematicamente y mucho mas estable: sumar no desborda, exponenciar si.

Y de paso se acumula la GAE hacia adelante, que es una segunda cuenta con otro
destino: los pesos deciden que particulas sobreviven, la GAE alimenta el loss de
la temperatura del M-step. No confundirlas.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext

from ops.common import dtype, idx_dtype, GlobalF32, GlobalI32
from systems.spo.launch import TPB, blocks_for
from systems.spo.particles import Particles, StepOutputs
from systems.spo.spo_types import SPOConfig


def update_particles_kernel(
        # estado de la particula, se actualiza in-place
        weights: GlobalF32, gae: GlobalF32, value: GlobalF32,
        terminal: GlobalI32, depth: GlobalI32, prior_logits: GlobalF32,
        # lo que produjo el paso del modelo
        reward: GlobalF32, discount: GlobalF32, next_value: GlobalF32,
        next_prior_logits: GlobalF32,
        # config
        n_particles: Int, search_gamma: Scalar[dtype],
        search_gae_lambda: Scalar[dtype]):
    """Cierra una profundidad: peso SMC, GAE, y el relevo de estado.

    Funde tres cosas que en Stoix son funciones separadas (`smc_weight_update_fn`,
    `calculate_gae` y `update_particles`) porque con un hilo por particula las
    tres son la misma pasada: se leen los valores viejos y se escriben los nuevos.

    El orden dentro del hilo es lo unico delicado: el error TD y la GAE necesitan
    el V(s) y la profundidad VIEJOS, asi que se calculan antes de pisarlos.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p >= n_particles:
        return

    old_value = value[p]
    old_depth = Int(depth[p])
    was_terminal = Int(terminal[p]) != 0
    step_discount = discount[p]

    # El error TD no se multiplica por gamma porque ya viene plegado dentro de
    # next_value, que es el bootstrap_value que devolvio el modelo. Stoix lo
    # comenta igual: "We do not multiply by discount as we do it in the
    # recurrent_fn".
    td_error = reward[p] + next_value[p] - old_value

    # La mascara terminal congela el peso de las particulas ya muertas, porque
    # lo que les pase despues de morir no deberia cambiar su evidencia.
    mask = Scalar[dtype](0) if was_terminal else Scalar[dtype](1)
    weights[p] = weights[p] + td_error * mask

    # La GAE va al reves de lo habitual. Normalmente se calcula hacia atras en
    # el tiempo, pero aqui la busqueda avanza y no se puede mirar al futuro, asi
    # que cada profundidad anade su delta descontado por
    # (gamma*lambda*discount)^profundidad.
    #
    # Fijate en que no hay mascara terminal, igual que en Stoix. Lo que congela a
    # una particula muerta es que su discount vale 0 y entonces el factor se
    # anula solo. La excepcion es la profundidad 0, donde el exponente es 0 y el
    # factor vale 1 pase lo que pase, y eso es correcto: el primer paso siempre
    # cuenta entero aunque la particula muera en el.
    decay_base = search_gamma * search_gae_lambda * step_discount
    decay = Scalar[dtype](1)
    for _ in range(old_depth):
        decay *= decay_base
    gae[p] = gae[p] + td_error * decay

    # Y el relevo: lo nuevo pasa a ser lo actual.
    value[p] = next_value[p]
    prior_logits[p] = next_prior_logits[p]
    depth[p] = Scalar[idx_dtype](old_depth + 1)
    # `terminal` es pegajoso: una vez muerta, muerta. Un discount de 0 la mata,
    # tanto si fue muerte real como truncacion.
    if was_terminal or step_discount == 0.0:
        terminal[p] = Scalar[idx_dtype](1)


def update_particles(ctx: DeviceContext, particles: Particles,
                     outputs: StepOutputs, cfg: SPOConfig) raises:
    """Encola el cierre de una profundidad."""
    p_total = cfg.num_search_particles()
    ctx.enqueue_function[update_particles_kernel, update_particles_kernel](
        particles.resample_td_weights.unsafe_ptr(), particles.gae.unsafe_ptr(),
        particles.value.unsafe_ptr(), particles.terminal.unsafe_ptr(),
        particles.depth.unsafe_ptr(), particles.prior_logits.unsafe_ptr(),
        outputs.reward.unsafe_ptr(), outputs.discount.unsafe_ptr(),
        outputs.next_value.unsafe_ptr(), outputs.next_prior_logits.unsafe_ptr(),
        p_total, cfg.search_gamma, cfg.search_gae_lambda,
        grid_dim=blocks_for(p_total), block_dim=TPB)
