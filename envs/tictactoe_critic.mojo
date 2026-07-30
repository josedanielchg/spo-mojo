"""Tic-Tac-Toe como modelo de busqueda, pero con un CRITICO en vez de V = 0.

Mismo entorno que `tictactoe.mojo` — mismas reglas, mismos kernels — con la unica
diferencia de donde sale el valor:

    TicTacToe        V(s) = 0 siempre           el planificador puro
    TicTacToeCritic  V(s) = lo que diga la red   <- esto

Por que importa: con V = 0 la busqueda solo ve algo si la partida TERMINA dentro de
su profundidad. Si no termina, todas las particulas pesan igual y la busqueda esta
ciega. Un critico pone valor a las posiciones intermedias, asi que en principio
deberia distinguir "esto pinta mal" antes de que la partida acabe.

Va en su propio fichero para no tocar `TicTacToe`, que ya esta probado y sigue
siendo util: la comparacion entre los dos ES el experimento de E1.11.

El bootstrap sigue el contrato del SearchModel, igual que el `recurrent_fn` de
Stoix:

    next_value = discount_real * search_gamma * V(s')

o sea que una particula que acaba de morir no arrastra valor futuro (su discount es
0) y una que sigue viva si.

Los pesos son una COPIA congelada, no una referencia a los del entrenamiento: el
modelo es dueno de sus buffers y se refresca con `sync_from`. Cuesta seis copias de
unos miles de floats, se hace una vez, y a cambio no hay dos duenos del mismo
buffer ni dudas sobre si la busqueda esta usando pesos a medio actualizar.
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
    """El bootstrap: next_value = discount * search_gamma * V(s'), un hilo por
    particula.

    El discount ya trae plegado si la particula murio, asi que multiplicar por el
    basta para que un estado terminal no arrastre futuro.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p < n:
        next_value[p] = discount[p] * search_gamma * value[p]


def bootstrap_depth_kernel(next_value: GlobalF32, discount: GlobalF32,
                           value: GlobalF32, depth: GlobalI32, n: Int,
                           reward_gamma: Scalar[dtype]):
    """El bootstrap descontado por PROFUNDIDAD: discount * gamma^(d+1) * V(s').

    Existe por un desajuste de escalas que se ve sumando los pesos de la busqueda.
    El nucleo hace `weights += r_d + next_value_d - value_d` y despues
    `value_d+1 = next_value_d`, asi que la suma telescopa:

        peso total = SUMA_d r_d  +  (bootstrap final)  -  V(s_raiz)

    La recompensa que produce `ttt_dynamics_kernel` ya viene multiplicada por
    gamma^d. Si el bootstrap NO lleva el mismo factor, las dos mitades de esa suma
    estan en escalas distintas: con gamma=0.7 una victoria en la profundidad 3
    vale 0.343 mientras que una particula que sigue viva arrastra V ~ 0.93. La
    busqueda concluye entonces que ganar es peor que no resolver la partida, que
    es exactamente lo contrario de lo que se quiere.

    Con gamma^(d+1) las dos mitades vuelven a la misma escala y el peso pasa a ser
    el retorno descontado a D pasos con bootstrap, menos una constante:

        ganar en d      ->  gamma^d - V(s_raiz)
        seguir vivo     ->  gamma^D * V(s_D) - V(s_raiz)
        perder en d     ->  0 - V(s_raiz)

    o sea ganar pronto > ganar tarde > sobrevivir > perder, que es el orden
    correcto.

    `depth[p]` es la profundidad ACTUAL (el nucleo la incrementa despues), asi que
    el estado al que se salta esta en d+1: de ahi el exponente.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p >= n:
        return
    g = Scalar[dtype](1)
    for _ in range(Int(depth[p]) + 1):
        g *= reward_gamma
    next_value[p] = discount[p] * g * value[p]


struct TicTacToeCritic(SearchModel, Movable):
    """TTT con el valor que da una red entrenada.

    Lleva dentro los pesos y sus buffers de trabajo. El struct vive en el host; lo
    que baja a la GPU son los buffers, igual que el juguete bajaba sus tres numeros.

    NO es `Copyable` (posee `DeviceBuffer`), a diferencia de `TicTacToe`. La
    busqueda generica lo acepta igual porque solo lo lee.
    """

    var reward_gamma: Scalar[dtype]
    """Descuento de la recompensa por profundidad, como en TicTacToe."""

    var params: CriticParams
    """La copia congelada de los pesos. Se refresca con `sync_from`."""
    var cache: CriticCache
    """Activaciones del forward. Reservadas para el mayor uso (P particulas)."""
    var obs: DeviceBuffer[dtype]
    """[max_batch, OBS_DIM] el tablero codificado, reutilizado en cada llamada."""
    var hidden: Int

    var depth_discounted: Bool
    """Si el bootstrap lleva gamma^(d+1) (escala coherente con la recompensa) o
    solo `search_gamma` (el contrato literal del SearchModel, como Stoix). Ver
    `bootstrap_depth_kernel` para por que hay dos."""

    def __init__(out self, ctx: DeviceContext, max_batch: Int, hidden: Int,
                 reward_gamma: Scalar[dtype],
                 depth_discounted: Bool = False) raises:
        """Reserva todo para `max_batch` tableros a la vez.

        `max_batch` tiene que cubrir el uso mayor de los dos: `num_envs` en la raiz
        y `num_envs * num_particles` en el step. O sea, el segundo.
        """
        self.reward_gamma = reward_gamma
        self.hidden = hidden
        self.depth_discounted = depth_discounted
        self.params = zero_critic_params(ctx, OBS_DIM, hidden, 1)
        self.cache = CriticCache(ctx, max_batch, hidden, 1)
        self.obs = zero_buffer[dtype](ctx, max_batch * OBS_DIM)

    def sync_from(self, ctx: DeviceContext, src: CriticParams) raises:
        """Copia los seis tensores de `src` a los pesos del modelo, en la GPU.

        Se llama cuando se quiere que la busqueda vea al critico actual. En E1.11
        basta una vez, despues de entrenar; en el bucle completo del M-step se
        llamaria cada iteracion.
        """
        if src.in_dim != OBS_DIM or src.hidden != self.hidden or src.out_dim != 1:
            raise Error("el critico que se intenta copiar no tiene la forma del "
                        "modelo: ", src.in_dim, "x", src.hidden, "x", src.out_dim)

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
        """Prior enmascarado (igual que sin critico) y V(s) de la red."""
        blocks = (cfg.num_envs + TPB_TTT - 1) // TPB_TTT

        ctx.enqueue_function[ttt_prior_logits_kernel, ttt_prior_logits_kernel](
            logits_out.unsafe_ptr(), root_state.unsafe_ptr(), cfg.num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)

        # El tablero en el formato que come la red, y su valor.
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            self.obs.unsafe_ptr(), root_state.unsafe_ptr(), cfg.num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
        critic_forward(ctx, self.params, self.cache, self.obs, cfg.num_envs)

        # V(s_raiz) va tal cual a la salida: aqui no hay discount que plegar.
        ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
            value_out.unsafe_ptr(), self.cache.value.unsafe_ptr(), cfg.num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)

    def step(self, ctx: DeviceContext, cfg: SPOConfig, particles: Particles,
             outputs: StepOutputs, step_uniforms: DeviceBuffer[dtype]) raises:
        """Avanza las particulas y pone el bootstrap con el valor de la red."""
        p_total = cfg.num_search_particles()
        blocks = (p_total + TPB_TTT - 1) // TPB_TTT

        # 1. La dinamica de siempre. Deja next_value a 0; se pisa en el paso 3.
        ctx.enqueue_function[ttt_dynamics_kernel, ttt_dynamics_kernel](
            particles.state.unsafe_ptr(), outputs.next_action.unsafe_ptr(),
            step_uniforms.unsafe_ptr(), particles.depth.unsafe_ptr(),
            outputs.reward.unsafe_ptr(), outputs.discount.unsafe_ptr(),
            outputs.next_value.unsafe_ptr(), p_total, self.reward_gamma,
            grid_dim=blocks, block_dim=TPB_TTT)

        # 2. V(s') sobre el estado NUEVO.
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            self.obs.unsafe_ptr(), particles.state.unsafe_ptr(), p_total,
            grid_dim=blocks, block_dim=TPB_TTT)
        critic_forward(ctx, self.params, self.cache, self.obs, p_total)

        # 3. El bootstrap, con el discount ya plegado y el gamma que toque.
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

        # 4. Y el prior en el estado nuevo, de donde se muestrea la accion.
        ctx.enqueue_function[ttt_prior_logits_kernel, ttt_prior_logits_kernel](
            outputs.action_logits.unsafe_ptr(), particles.state.unsafe_ptr(),
            p_total, grid_dim=blocks, block_dim=TPB_TTT)
