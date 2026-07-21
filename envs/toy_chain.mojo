"""MDP de juguete: un pasillo con dos acciones. El primer SearchModel.

Sirve para validar la busqueda SMC antes de que existan las redes o CartPole. Es
deterministico y su politica optima se sabe de memoria, asi que los tests pueden
comprobar numeros exactos en vez de tolerancias.

    posiciones:   0 --> 1 --> 2 --> ... --> L
    accion GOOD (1): avanza una casilla y da recompensa 1
    accion BAD  (0): mata el episodio ahi mismo, recompensa 0

Dos formas distintas de que un episodio acabe, y esa es justo la gracia:

  * TERMINAL de verdad (accion BAD): no hay futuro. discount real = 0, asi que el
    bootstrap_value tambien es 0.
  * TRUNCACION (llegar al horizonte): el episodio se corta por un limite de
    tiempo, pero el estado si tenia futuro. El discount real es 1, el `rec_discount`
    que ve la busqueda es 0 (la particula deja de simular) pero el
    bootstrap_value no es 0: arrastra search_gamma * V(s').

Confundir esos dos casos es el bug silencioso clasico de RL, y es la razon de que
el juguete tenga los dos caminos desde el primer dia. Por eso el horizonte
(`horizon`) es MENOR que el largo del pasillo (`chain_length`): al truncar en la
casilla `horizon` todavia quedan casillas por delante y V(s') sale distinto de
cero, que es lo que hace que el test note la diferencia.

Valor analitico: quedan (L - pos) casillas buenas, cada una con recompensa 1, y
search_gamma es 1, asi que V(pos) = value_scale * (L - pos). Con value_scale = 0
se obtiene V == 0, que es el caso "sin critico" que usara la demo de CartPole.

Este fichero NO importa la busqueda: solo los tipos y el contrato `SearchModel`.
El entorno no tiene por que saber que existe SPO, igual que en Stoix, donde
`envs/` no conoce el algoritmo y es `ff_spo.py` quien monta la busqueda encima.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from ops.common import dtype, idx_dtype, GlobalF32, GlobalI32
from systems.spo.particles import Particles, StepOutputs
from systems.spo.search_model import SearchModel
from systems.spo.spo_types import SPOConfig

# Las dos acciones. El orden importa para los tests: BAD es la 0.
comptime ACTION_BAD = 0
comptime ACTION_GOOD = 1
comptime NUM_ACTIONS = 2

# El pasillo es unidimensional: el estado es solo la posicion.
comptime STATE_DIM = 1

comptime TPB_TOY = 32


def toy_value(pos: Scalar[dtype], chain_length: Int,
              value_scale: Scalar[dtype]) -> Scalar[dtype]:
    """V(pos) = value_scale * casillas que quedan. Funcion de device: la llaman
    tanto el kernel de valor como el recurrente."""
    remaining = Scalar[dtype](chain_length) - pos
    if remaining < 0.0:
        remaining = 0.0
    return value_scale * remaining


def toy_value_kernel(value_out: GlobalF32, state: GlobalF32, n_particles: Int,
                     chain_length: Int, value_scale: Scalar[dtype]):
    """V(s) de cada particula. Un hilo por particula."""
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n_particles:
        value_out[i] = toy_value(state[i * STATE_DIM], chain_length, value_scale)


def toy_policy_logits_kernel(logits_out: GlobalF32, state: GlobalF32,
                             n_particles: Int):
    """El prior en el estado de cada particula.

    El prior del juguete es UNIFORME, y eso es deliberado: si la busqueda mejora
    una politica que no sabe nada, la mejora viene de la busqueda y de nada mas.
    Logits todos iguales -> softmax uniforme (da igual el valor, el softmax es
    invariante a sumar una constante).
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n_particles:
        for a in range(NUM_ACTIONS):
            logits_out[i * NUM_ACTIONS + a] = Scalar[dtype](0)


def toy_recurrent_kernel(state: GlobalF32, action: GlobalI32,
                         reward_out: GlobalF32, discount_out: GlobalF32,
                         next_value_out: GlobalF32,
                         n_particles: Int, chain_length: Int, horizon: Int,
                         value_scale: Scalar[dtype], search_gamma: Scalar[dtype]):
    """Avanza una casilla y pliega gamma. La dinamica del modelo.

    Es el equivalente del `recurrent_fn` de Stoix para este modelo, incluyendo las
    dos lineas que mas cuesta acertar:

        rec_discount    = discount * (1 - truncated)
        bootstrap_value = discount * search_gamma * V(s')

    El estado se actualiza in-place. Las particulas muertas se quedan quietas: no
    hay auto-reset dentro de la busqueda, porque una particula terminal ya no
    vuelve a contar (la mascara terminal del nucleo SMC congela su peso) y
    dejarla quieta hace los tests deterministas.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= n_particles:
        return

    pos = state[i * STATE_DIM]
    a = Int(action[i])

    # Primero la dinamica del entorno.
    reward = Scalar[dtype](0)
    next_pos = pos
    discount_real = Scalar[dtype](1)   # el discount "de verdad" del entorno
    last = False                       # el timestep.last() de Stoix

    if a == ACTION_GOOD:
        next_pos = pos + 1.0
        reward = 1.0
        # Truncacion: se acabo el tiempo, pero el estado tenia futuro.
        if next_pos >= Scalar[dtype](horizon):
            last = True
            discount_real = 1.0
    else:
        # Terminal de verdad: no hay futuro que valorar. reward se queda en 0.
        last = True
        discount_real = 0.0

    # Y ahora lo mismo que hace Stoix con el timestep: algo esta truncado si
    # termino pero su discount real no es 0. Un terminal de verdad trae discount
    # 0, asi que nunca cuenta como truncado.
    truncated = last and discount_real != 0.0

    # El discount que ve la busqueda: 0 en cuanto la particula deja de simular,
    # sea por muerte o por truncacion. El nucleo SMC lo usa para marcar terminal.
    # Es el `discount * (1 - truncated)` de Stoix.
    rec_discount = discount_real * (Scalar[dtype](0) if truncated else Scalar[dtype](1))

    # El valor para bootstrapping usa el discount REAL, no el de arriba: por eso
    # una truncacion conserva su valor y un terminal no.
    bootstrap_value = discount_real * search_gamma * toy_value(
        next_pos, chain_length, value_scale)

    state[i * STATE_DIM] = next_pos
    reward_out[i] = reward
    discount_out[i] = rec_discount
    next_value_out[i] = bootstrap_value


@fieldwise_init
struct ToyChain(SearchModel, Copyable, Movable):
    """El pasillo como modelo de busqueda.

    El struct se queda SIEMPRE en el host: lo que baja a la GPU son sus tres
    numeros sueltos, dentro de los `enqueue_function` de sus propios metodos. Por
    eso la busqueda puede ser generica sobre el modelo sin que ningun kernel
    cruce la frontera (ver el contrato en search_model.mojo).
    """

    var chain_length: Int
    """Cuantas casillas buenas hay en total. Define el valor analitico."""

    var horizon: Int
    """En que casilla se trunca el episodio. Menor que chain_length para que la
    truncacion deje valor futuro sin recoger."""

    var value_scale: Scalar[dtype]
    """1.0 = valor analitico exacto, 0.0 = V==0 (el caso sin critico)."""

    def eval_root(self, ctx: DeviceContext, cfg: SPOConfig,
                  root_state: DeviceBuffer[dtype],
                  logits_out: DeviceBuffer[dtype],
                  value_out: DeviceBuffer[dtype]) raises:
        """El prior y V(s) sobre los estados raiz, uno por entorno."""
        blocks = (cfg.num_envs + TPB_TOY - 1) // TPB_TOY

        ctx.enqueue_function[toy_policy_logits_kernel, toy_policy_logits_kernel](
            logits_out.unsafe_ptr(), root_state.unsafe_ptr(), cfg.num_envs,
            grid_dim=blocks, block_dim=TPB_TOY)

        ctx.enqueue_function[toy_value_kernel, toy_value_kernel](
            value_out.unsafe_ptr(), root_state.unsafe_ptr(), cfg.num_envs,
            self.chain_length, self.value_scale,
            grid_dim=blocks, block_dim=TPB_TOY)

    def step(self, ctx: DeviceContext, cfg: SPOConfig, particles: Particles,
             outputs: StepOutputs) raises:
        """Avanza las P particulas una profundidad.

        Dos kernels: la dinamica (que ya pliega gamma y la truncacion) y el prior
        evaluado en el estado NUEVO. Sortear la accion siguiente no es cosa del
        modelo: de eso se encarga `sample_next_actions`, que es generico.

        No toca `particles.value`, y eso importa: el error TD de la actualizacion
        de pesos necesita el V(s) viejo, y el nuevo se queda en
        `outputs.next_value` hasta que update_particles lo mueva.
        """
        p_total = cfg.num_search_particles()
        blocks = (p_total + TPB_TOY - 1) // TPB_TOY

        ctx.enqueue_function[toy_recurrent_kernel, toy_recurrent_kernel](
            particles.state.unsafe_ptr(), outputs.next_action.unsafe_ptr(),
            outputs.reward.unsafe_ptr(), outputs.discount.unsafe_ptr(),
            outputs.next_value.unsafe_ptr(), p_total,
            self.chain_length, self.horizon, self.value_scale, cfg.search_gamma,
            grid_dim=blocks, block_dim=TPB_TOY)

        ctx.enqueue_function[toy_policy_logits_kernel, toy_policy_logits_kernel](
            outputs.action_logits.unsafe_ptr(), particles.state.unsafe_ptr(),
            p_total, grid_dim=blocks, block_dim=TPB_TOY)


def default_toy_chain() -> ToyChain:
    """L=8, truncacion en 4: al truncar quedan 4 casillas, o sea V(4) = 4 != 0."""
    return ToyChain(chain_length=8, horizon=4, value_scale=1.0)
