"""Tic-Tac-Toe con el PRIOR del actor. Aqui se cierra el bucle EM.

Hasta ahora la busqueda partia de un prior uniforme sobre las casillas legales: no
tenia ninguna opinion sobre por donde mirar. Este modelo lo sustituye por lo que
diga la red, y con eso el bucle del paper se cierra:

    E-step   la busqueda planifica partiendo de pi y produce q, mejor que pi
    M-step   el actor aprende a imitar q, o sea pi se acerca a q
    y la vuelta siguiente la busqueda parte de un pi mejor

Sin esta pieza el M-step entrena una red que nadie usa. Con ella, cada iteracion
mejora el punto de partida de la siguiente, que es de donde sale la mejora
compuesta del metodo.

**Como entra el prior, exactamente.** Se comprobo en `weighting.mojo:83`: los
`prior_logits` solo se RELEVAN de una profundidad a la siguiente, nunca aparecen en
el peso SMC. O sea que el prior no cambia como se puntua una particula, solo QUE
ACCIONES se muestrean. Es la simplificacion de la ecuacion 10 del paper: cuando la
distribucion propuesta es la propia politica, el cociente pi/pi se cancela y queda
`w propto w * exp(A/eta)`. Y es exactamente el papel que juega el prior de
AlphaZero: dirigir la busqueda, no valorarla.

Los pesos son una copia congelada con `sync_from`, igual que en
`tictactoe_critic.mojo` y por la misma razon: dos duenos del mismo `DeviceBuffer`
en Mojo es un lio, y la copia deja claro en que momento la busqueda empieza a ver
al actor nuevo.

**El critico es opcional pero por defecto ENTRA**, y eso es una correccion respecto
a la primera version de este fichero. V esta en la ecuacion 10 del paper

    A(s_t,a_t) = r_t + V(s_{t+1}) - V(s_t)

y en `_critic_loss_fn` de Stoix, asi que tenerlo desconectado era una desviacion
nuestra, no una eleccion del metodo. E1.11 midio que no ayudaba, pero lo midio con
un critico entrenado con datos de un PLANIFICADOR que perdia el 2% y partia de
prior uniforme. Ahora los datos vienen de un agente mucho mas fuerte que visita
otras posiciones, asi que la medida vieja no aplica y hay que repetirla.

Los dos modos de bootstrap del critico son los de E1.11:
  - `discount * search_gamma * V(s')`, el contrato literal del SearchModel y de
    Stoix;
  - `discount * gamma_r^(d+1) * V(s')`, que hace falta si `reward_gamma` < 1 porque
    entonces la recompensa lleva gamma_r^d plegado y el valor tiene que estar en la
    misma escala (ver `bootstrap_depth_kernel`).
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from ops.buffers import zero_buffer
from ops.common import dtype
from ops.copy import copy_kernel
from envs.tictactoe import (ttt_dynamics_kernel, ttt_encode_obs_kernel,
                            ttt_legal_mask_kernel, NUM_ACTIONS, OBS_DIM,
                            TPB_TTT)
from envs.tictactoe_critic import bootstrap_kernel, bootstrap_depth_kernel
from networks.actor import ActorParams, ActorCache, actor_logits, zero_actor_params
from networks.mlp import (CriticParams, CriticCache, critic_forward,
                          zero_critic_params)
from systems.spo.particles import Particles, StepOutputs
from systems.spo.search_model import SearchModel
from systems.spo.spo_types import SPOConfig


struct TicTacToeActor(SearchModel, Movable):
    """TTT donde el prior de la busqueda lo pone una red entrenada."""

    var reward_gamma: Scalar[dtype]
    var loss_penalty: Scalar[dtype]

    var params: ActorParams
    """Copia congelada de los pesos del actor. Se refresca con `sync_from`."""
    var cache: ActorCache
    var obs: DeviceBuffer[dtype]
    """[max_batch, OBS_DIM] el tablero codificado."""
    var mask: DeviceBuffer[dtype]
    """[max_batch, NUM_ACTIONS] 1 legal, 0 ocupada."""
    var hidden: Int
    var max_batch: Int

    var critic: CriticParams
    """Los pesos del critico, tambien en copia congelada."""
    var ccache: CriticCache
    var use_critic: Bool
    """Si V sale de la red o se queda en 0."""
    var depth_discounted: Bool
    """Si el bootstrap lleva gamma_r^(d+1) en vez de solo `search_gamma`. Hace
    falta cuando reward_gamma < 1; ver `bootstrap_depth_kernel`."""

    def __init__(out self, ctx: DeviceContext, max_batch: Int, hidden: Int,
                 reward_gamma: Scalar[dtype],
                 loss_penalty: Scalar[dtype] = 0,
                 use_critic: Bool = False,
                 depth_discounted: Bool = False) raises:
        """`max_batch` tiene que cubrir el uso mayor: num_envs * num_particles."""
        self.reward_gamma = reward_gamma
        self.loss_penalty = loss_penalty
        self.hidden = hidden
        self.max_batch = max_batch
        self.use_critic = use_critic
        self.depth_discounted = depth_discounted
        self.params = zero_actor_params(ctx, hidden)
        self.cache = ActorCache(ctx, max_batch, hidden, NUM_ACTIONS)
        self.obs = zero_buffer[dtype](ctx, max_batch * OBS_DIM)
        self.mask = zero_buffer[dtype](ctx, max_batch * NUM_ACTIONS)
        self.critic = zero_critic_params(ctx, OBS_DIM, hidden, 1)
        self.ccache = CriticCache(ctx, max_batch, hidden, 1)

    def sync_critic_from(self, ctx: DeviceContext, src: CriticParams) raises:
        """Trae los pesos del critico que se esta entrenando."""
        if src.in_dim != OBS_DIM or src.hidden != self.hidden or src.out_dim != 1:
            raise Error("el critico no tiene la forma del modelo: ", src.in_dim,
                        "x", src.hidden, "x", src.out_dim)
        h = self.hidden
        self._copy(ctx, self.critic.w1, src.w1, OBS_DIM * h)
        self._copy(ctx, self.critic.b1, src.b1, h)
        self._copy(ctx, self.critic.w2, src.w2, h * h)
        self._copy(ctx, self.critic.b2, src.b2, h)
        self._copy(ctx, self.critic.w3, src.w3, h)
        self._copy(ctx, self.critic.b3, src.b3, 1)

    def sync_from(self, ctx: DeviceContext, src: CriticParams) raises:
        """Trae los pesos del actor que se esta entrenando.

        Llamarlo entre iteraciones del bucle EM es justo lo que hace que el
        siguiente E-step parta de una politica mejor. Si no se llamara, la busqueda
        se quedaria con los ceros del constructor -- prior uniforme disfrazado, y
        el bucle no se cerraria.
        """
        if src.in_dim != OBS_DIM or src.hidden != self.hidden \
                or src.out_dim != NUM_ACTIONS:
            raise Error("el actor que se intenta copiar no tiene la forma del "
                        "modelo: ", src.in_dim, "x", src.hidden, "x",
                        src.out_dim)
        h = self.hidden
        self._copy(ctx, self.params.w1, src.w1, OBS_DIM * h)
        self._copy(ctx, self.params.b1, src.b1, h)
        self._copy(ctx, self.params.w2, src.w2, h * h)
        self._copy(ctx, self.params.b2, src.b2, h)
        self._copy(ctx, self.params.w3, src.w3, h * NUM_ACTIONS)
        self._copy(ctx, self.params.b3, src.b3, NUM_ACTIONS)

    def _copy(self, ctx: DeviceContext, dst: DeviceBuffer[dtype],
              src: DeviceBuffer[dtype], n: Int) raises:
        ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
            dst.unsafe_ptr(), src.unsafe_ptr(), n,
            grid_dim=(n + 255) // 256, block_dim=256)

    def _prior(self, ctx: DeviceContext, state: DeviceBuffer[dtype],
               out_logits: DeviceBuffer[dtype], m: Int) raises:
        """Los logits del actor sobre m tableros, enmascarados, en `out_logits`.

        Tres pasos: la mascara y la observacion salen del ESTADO (la unica fuente
        de verdad sobre que es legal), y la red pone los logits.
        """
        if m > self.max_batch:
            raise Error("el modelo se reservo para ", self.max_batch,
                        " tableros y se le piden ", m)
        blocks = (m + TPB_TTT - 1) // TPB_TTT

        ctx.enqueue_function[ttt_legal_mask_kernel, ttt_legal_mask_kernel](
            self.mask.unsafe_ptr(), state.unsafe_ptr(), m,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            self.obs.unsafe_ptr(), state.unsafe_ptr(), m,
            grid_dim=blocks, block_dim=TPB_TTT)
        actor_logits(ctx, self.params, self.cache, self.obs, self.mask, m)

        n_cells = m * NUM_ACTIONS
        ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
            out_logits.unsafe_ptr(), self.cache.value.unsafe_ptr(), n_cells,
            grid_dim=(n_cells + 255) // 256, block_dim=256)

    def eval_root(self, ctx: DeviceContext, cfg: SPOConfig,
                  root_state: DeviceBuffer[dtype],
                  logits_out: DeviceBuffer[dtype],
                  value_out: DeviceBuffer[dtype]) raises:
        """El prior de la RED en los estados raiz, y V del critico (o 0)."""
        self._prior(ctx, root_state, logits_out, cfg.num_envs)
        if not self.use_critic:
            # Se pisa explicitamente porque el workspace se reutiliza entre
            # busquedas y podria traer valores viejos (el bug de A6).
            value_out.enqueue_fill(0)
            return
        # `_prior` ya dejo la observacion codificada en self.obs, asi que no hace
        # falta recalcularla.
        critic_forward(ctx, self.critic, self.ccache, self.obs, cfg.num_envs)
        n = cfg.num_envs
        ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
            value_out.unsafe_ptr(), self.ccache.value.unsafe_ptr(), n,
            grid_dim=(n + 255) // 256, block_dim=256)

    def step(self, ctx: DeviceContext, cfg: SPOConfig, particles: Particles,
             outputs: StepOutputs, step_uniforms: DeviceBuffer[dtype]) raises:
        """Avanza las particulas y evalua el prior de la red en el estado NUEVO."""
        p_total = cfg.num_search_particles()
        blocks = (p_total + TPB_TTT - 1) // TPB_TTT

        ctx.enqueue_function[ttt_dynamics_kernel, ttt_dynamics_kernel](
            particles.state.unsafe_ptr(), outputs.next_action.unsafe_ptr(),
            step_uniforms.unsafe_ptr(), particles.depth.unsafe_ptr(),
            outputs.reward.unsafe_ptr(), outputs.discount.unsafe_ptr(),
            outputs.next_value.unsafe_ptr(), p_total, self.reward_gamma,
            self.loss_penalty, grid_dim=blocks, block_dim=TPB_TTT)

        self._prior(ctx, particles.state, outputs.action_logits, p_total)

        if self.use_critic:
            # self.obs ya tiene el estado NUEVO codificado, de `_prior`.
            critic_forward(ctx, self.critic, self.ccache, self.obs, p_total)
            if self.depth_discounted:
                ctx.enqueue_function[bootstrap_depth_kernel,
                                     bootstrap_depth_kernel](
                    outputs.next_value.unsafe_ptr(),
                    outputs.discount.unsafe_ptr(),
                    self.ccache.value.unsafe_ptr(),
                    particles.depth.unsafe_ptr(), p_total, self.reward_gamma,
                    grid_dim=blocks, block_dim=TPB_TTT)
            else:
                ctx.enqueue_function[bootstrap_kernel, bootstrap_kernel](
                    outputs.next_value.unsafe_ptr(),
                    outputs.discount.unsafe_ptr(),
                    self.ccache.value.unsafe_ptr(), p_total, cfg.search_gamma,
                    grid_dim=blocks, block_dim=TPB_TTT)
