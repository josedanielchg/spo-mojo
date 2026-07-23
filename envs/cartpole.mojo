"""CartPole-v1: la fisica exacta de gymnax, vectorizada.

Reproduce paso a paso `step_env` de
gymnax/environments/classic_control/cartpole.py. El estado son 5 numeros:
(x, x_dot, theta, theta_dot, time). El `time` va como float pero es exacto
(los enteros hasta 2^24 lo son, y el maximo es 500).

Por etapas:
  1.2  la dinamica (integracion de Euler)                    -> cartpole_advance
  2.1  terminacion + discount + reward + flag de truncacion  -> cartpole_recurrent_kernel
  2.2  reset + auto-reset                                     -> cartpole_reset_kernel
  3.1  el struct SearchModel (enchufa CartPole a la busqueda) -> struct CartPole

Como el juguete, este fichero implementa el contrato `SearchModel` importando solo
los tipos de datos y el contrato, NO el algoritmo de busqueda. El entorno no sabe
que existe SPO; es la busqueda quien lo usa.

Las constantes son los EnvParams de gymnax. Ojo con la precision: en Python son
float64, pero JAX las mantiene en float32 al operar con el estado (float32) por
weak typing, asi que aqui van en float32 y el resultado cuadra con el golden.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sin, cos, abs

from ops.common import dtype, idx_dtype, GlobalF32, GlobalI32
from ops.rng import fill_uniform
from systems.spo.particles import Particles, StepOutputs
from systems.spo.search_model import SearchModel
from systems.spo.spo_types import SPOConfig

comptime TPB_CART = 32

comptime STATE_DIM = 5
comptime NUM_ACTIONS = 2

# Las dos acciones: empujar el carro a izquierda o derecha.
comptime ACTION_LEFT = 0
comptime ACTION_RIGHT = 1

# --- gymnax EnvParams (CartPole-v1) ---
comptime GRAVITY = Scalar[dtype](9.8)
comptime MASSPOLE = Scalar[dtype](0.1)
comptime TOTAL_MASS = Scalar[dtype](1.1)          # masscart + masspole
comptime LENGTH = Scalar[dtype](0.5)              # media longitud del palo
comptime POLEMASS_LENGTH = Scalar[dtype](0.05)    # masspole * length
comptime FORCE_MAG = Scalar[dtype](10.0)
comptime TAU = Scalar[dtype](0.02)                # paso de integracion (segundos)
comptime FOUR_THIRDS = Scalar[dtype](4.0 / 3.0)

# --- condiciones de fin de episodio ---
comptime X_THRESHOLD = Scalar[dtype](2.4)                  # el carro se sale de la pista
comptime THETA_THRESHOLD = Scalar[dtype](0.2094395102393195)  # 12 * 2*pi/360, el palo cae
comptime MAX_STEPS = 500                                   # limite de pasos (truncacion)

# --- reset: estado inicial U(-0.05, 0.05) en las 4 componentes dinamicas ---
comptime RESET_MINVAL = Scalar[dtype](-0.05)
comptime RESET_RANGE = Scalar[dtype](0.1)                  # maxval - minval


@fieldwise_init
struct CartState(Copyable, Movable):
    """El estado del carro, para poder devolverlo de una funcion de device.
    Es lo que permite compartir la fisica entre el kernel de fisica y el
    recurrente sin duplicar las formulas."""
    var x: Scalar[dtype]
    var x_dot: Scalar[dtype]
    var theta: Scalar[dtype]
    var theta_dot: Scalar[dtype]
    var time: Scalar[dtype]


def cartpole_advance(x: Scalar[dtype], x_dot: Scalar[dtype], theta: Scalar[dtype],
                     theta_dot: Scalar[dtype], time: Scalar[dtype],
                     action: Int) -> CartState:
    """Un paso de la dinamica, sin tocar terminacion. Espeja `step_env` de gymnax
    (lineas 61-90). Funcion de device: la llaman los dos kernels de abajo."""
    # force = +FORCE_MAG si la accion es 1, -FORCE_MAG si es 0. Con la misma
    # expresion que gymnax (no con un if) para que el float32 salga identico.
    a = Scalar[dtype](action)
    force = FORCE_MAG * a - FORCE_MAG * (Scalar[dtype](1) - a)

    costheta = cos(theta)
    sintheta = sin(theta)

    temp = (force + POLEMASS_LENGTH * (theta_dot * theta_dot) * sintheta) / TOTAL_MASS
    thetaacc = (GRAVITY * sintheta - costheta * temp) / (
        LENGTH * (FOUR_THIRDS - MASSPOLE * (costheta * costheta) / TOTAL_MASS))
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

    # Integracion de Euler (la unica que ofrece gymnax).
    return CartState(
        x + TAU * x_dot,
        x_dot + TAU * xacc,
        theta + TAU * theta_dot,
        theta_dot + TAU * thetaacc,
        time + Scalar[dtype](1),
    )


def cartpole_out_of_bounds(x: Scalar[dtype], theta: Scalar[dtype]) -> Bool:
    """Muerte REAL: el carro se salio o el palo cayo. Sin futuro que valorar.
    Es la parte de `is_terminal` de gymnax que NO es el limite de pasos."""
    return abs(x) > X_THRESHOLD or abs(theta) > THETA_THRESHOLD


def cartpole_is_terminal(x: Scalar[dtype], theta: Scalar[dtype],
                         time: Scalar[dtype]) -> Bool:
    """El `is_terminal` completo de gymnax: fuera de pista, palo caido, o limite
    de pasos. Los tres cuentan como 'done'."""
    return cartpole_out_of_bounds(x, theta) or Int(time) >= MAX_STEPS


def cartpole_physics_kernel(state_out: GlobalF32, state_in: GlobalF32,
                            action: GlobalI32, n_particles: Int):
    """Solo la dinamica, a un buffer de salida. Es lo que prueba el golden A.

    In-place seguro: lee los 5 componentes antes de escribir ninguno, asi que
    state_out puede ser el mismo buffer que state_in.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= n_particles:
        return

    base = i * STATE_DIM
    ns = cartpole_advance(state_in[base + 0], state_in[base + 1],
                          state_in[base + 2], state_in[base + 3],
                          state_in[base + 4], Int(action[i]))
    state_out[base + 0] = ns.x
    state_out[base + 1] = ns.x_dot
    state_out[base + 2] = ns.theta
    state_out[base + 3] = ns.theta_dot
    state_out[base + 4] = ns.time


def cartpole_recurrent_kernel(
        state: GlobalF32, action: GlobalI32,
        reward_out: GlobalF32, discount_out: GlobalF32, next_value_out: GlobalF32,
        n_particles: Int, value_scale: Scalar[dtype], search_gamma: Scalar[dtype],
        truncate_on_limit: Int):
    """El paso completo del modelo: fisica + terminacion + plegado de gamma.

    Mismo patron que `toy_recurrent_kernel`: actualiza `state` in-place y escribe
    reward / rec_discount / bootstrap. La unica diferencia especifica de CartPole
    es que el limite de 500 pasos puede tratarse como muerte (flag A) o como
    truncacion con futuro (flag B).

      reward         = 1 - prev_terminal   (gymnax: si entras ya muerto, 0)
      rec_discount   = discount_real * (1 - truncated)   (0 en cuanto para la particula)
      bootstrap      = discount_real * search_gamma * V(s')

    Sobre V: el modelo de la demo no tiene critico, asi que V es una CONSTANTE
    (value_scale), 0 en la demo. value_scale > 0 solo existe para los tests, para
    que el bootstrap no sea trivialmente 0 y se pueda ver el efecto del flag.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= n_particles:
        return

    base = i * STATE_DIM
    x = state[base + 0]
    x_dot = state[base + 1]
    theta = state[base + 2]
    theta_dot = state[base + 3]
    time = state[base + 4]

    # 1. El reward mira el estado de ENTRADA (1 - prev_terminal de gymnax).
    prev_terminal = cartpole_is_terminal(x, theta, time)
    reward = Scalar[dtype](0) if prev_terminal else Scalar[dtype](1)

    # 2. La dinamica. gymnax integra pase lo que pase, aunque ya estuviera muerta.
    ns = cartpole_advance(x, x_dot, theta, theta_dot, time, Int(action[i]))

    # 3. Clasificar el fin del episodio en el estado NUEVO.
    dead = cartpole_out_of_bounds(ns.x, ns.theta)   # muerte real, sin futuro
    hit_limit = Int(ns.time) >= MAX_STEPS           # limite de pasos
    last = dead or hit_limit                        # = done de gymnax

    # 4. discount_real: 0 si murio de verdad; en el limite depende del flag.
    #    La muerte real manda sobre el limite (si el palo cayo, da igual el reloj).
    discount_real = Scalar[dtype](1)
    if dead:
        discount_real = Scalar[dtype](0)
    elif hit_limit:
        discount_real = Scalar[dtype](1) if truncate_on_limit != 0 else Scalar[dtype](0)

    # 5. El plegado, igual que en el juguete y en Stoix.
    truncated = last and discount_real != 0.0
    rec_discount = discount_real * (Scalar[dtype](0) if truncated else Scalar[dtype](1))
    bootstrap = discount_real * search_gamma * value_scale   # V = value_scale (constante)

    state[base + 0] = ns.x
    state[base + 1] = ns.x_dot
    state[base + 2] = ns.theta
    state[base + 3] = ns.theta_dot
    state[base + 4] = ns.time
    reward_out[i] = reward
    discount_out[i] = rec_discount
    next_value_out[i] = bootstrap


def write_reset_state(state: GlobalF32, base: Int, uniforms: GlobalF32, ubase: Int):
    """Escribe un estado inicial: U(-0.05, 0.05) en las 4 componentes dinamicas
    y time = 0. Espeja `reset_env` de gymnax. Funcion de device: la comparten el
    reset incondicional y el auto-reset. `uniforms` trae 4 valores en [0,1)."""
    state[base + 0] = RESET_MINVAL + uniforms[ubase + 0] * RESET_RANGE
    state[base + 1] = RESET_MINVAL + uniforms[ubase + 1] * RESET_RANGE
    state[base + 2] = RESET_MINVAL + uniforms[ubase + 2] * RESET_RANGE
    state[base + 3] = RESET_MINVAL + uniforms[ubase + 3] * RESET_RANGE
    state[base + 4] = Scalar[dtype](0)


def cartpole_reset_kernel(state_out: GlobalF32, uniforms: GlobalF32, n_envs: Int):
    """Resetea TODOS los envs. Para sembrar el estado inicial del bucle.

    `uniforms` es [n_envs, 4] en [0,1), generados aparte con fill_uniform (misma
    convencion que el resto: el kernel recibe los uniformes, no llama al RNG)."""
    e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= n_envs:
        return
    write_reset_state(state_out, e * STATE_DIM, uniforms, e * 4)


def cartpole_auto_reset_kernel(state: GlobalF32, done: GlobalI32,
                               uniforms: GlobalF32, n_envs: Int):
    """Resetea SOLO los envs cuyo episodio termino (done != 0), dejando los demas
    intactos. Es lo que usa el bucle de entorno real tras cada paso: una partida
    que acaba empieza otra desde un estado nuevo, y las que siguen vivas continuan.

    Ojo: esto es el auto-reset del ENTORNO REAL, no de la busqueda. Dentro de la
    busqueda las particulas muertas se quedan quietas (la mascara terminal congela
    su peso), igual que en el juguete."""
    e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= n_envs:
        return
    if Int(done[e]) == 0:
        return
    write_reset_state(state, e * STATE_DIM, uniforms, e * 4)


def cartpole_policy_logits_kernel(logits_out: GlobalF32, n_rows: Int):
    """El prior de la demo: UNIFORME (todos los logits a 0).

    Es deliberado, igual que en el juguete: si la busqueda mejora una politica
    que no sabe nada, la mejora viene de la busqueda y de nada mas. Cuando en la
    fase 5 exista el actor, este kernel se cambia por el forward de la red."""
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n_rows:
        for a in range(NUM_ACTIONS):
            logits_out[i * NUM_ACTIONS + a] = Scalar[dtype](0)


def cartpole_value_kernel(value_out: GlobalF32, n_rows: Int,
                          value_scale: Scalar[dtype]):
    """V(s) = value_scale, constante. 0 en la demo (sin critico): ahi los pesos
    SMC degeneran al retorno acumulado y la busqueda planifica sin ninguna red.
    value_scale > 0 solo lo usan los tests."""
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n_rows:
        value_out[i] = value_scale


@fieldwise_init
struct CartPole(SearchModel, Copyable, Movable):
    """CartPole como modelo de busqueda: el `SearchModel` de la demo 2.

    Enchufa a la busqueda que ya existe sin tocar ni una linea del nucleo SMC:
    solo hay que saber evaluar la raiz y dar un paso. El struct se queda en el
    host; sus dos campos bajan como argumentos de los kernels.

    En la demo va con value_scale = 0 (V ≡ 0, sin red) y el flag por defecto (A).
    """

    var value_scale: Scalar[dtype]
    """El valor constante del modelo. 0 en la demo; > 0 solo para tests."""

    var truncate_on_step_limit: Bool
    """El paso 500 como truncacion con futuro (True, opcion B) o como muerte
    (False, opcion A = fiel a gymnax). Ver cartpole_recurrent_kernel."""

    def eval_root(self, ctx: DeviceContext, cfg: SPOConfig,
                  root_state: DeviceBuffer[dtype],
                  logits_out: DeviceBuffer[dtype],
                  value_out: DeviceBuffer[dtype]) raises:
        """El prior y V(s) sobre los estados raiz, uno por entorno.

        El prior es uniforme y V es constante, asi que ninguno de los dos mira
        `root_state` -- pero la firma lo recibe porque el contrato es generico y
        el actor/critico de la fase 5 si lo necesitaran."""
        blocks = (cfg.num_envs + TPB_CART - 1) // TPB_CART

        ctx.enqueue_function[cartpole_policy_logits_kernel, cartpole_policy_logits_kernel](
            logits_out.unsafe_ptr(), cfg.num_envs,
            grid_dim=blocks, block_dim=TPB_CART)

        ctx.enqueue_function[cartpole_value_kernel, cartpole_value_kernel](
            value_out.unsafe_ptr(), cfg.num_envs, self.value_scale,
            grid_dim=blocks, block_dim=TPB_CART)

    def step(self, ctx: DeviceContext, cfg: SPOConfig, particles: Particles,
             outputs: StepOutputs) raises:
        """Avanza las P particulas una profundidad: fisica + terminacion + plegado.

        Dos kernels: el recurrente (que ya hace todo el trabajo de las etapas 1-2)
        y el prior en el estado NUEVO. Sortear la accion siguiente no es cosa del
        modelo: de eso se encarga `sample_next_actions`, que es generico.

        No toca `particles.value`: el error TD necesita el V viejo, y el nuevo se
        queda en `outputs.next_value` hasta que update_particles lo mueva."""
        p_total = cfg.num_search_particles()
        blocks = (p_total + TPB_CART - 1) // TPB_CART
        flag = 1 if self.truncate_on_step_limit else 0

        ctx.enqueue_function[cartpole_recurrent_kernel, cartpole_recurrent_kernel](
            particles.state.unsafe_ptr(), outputs.next_action.unsafe_ptr(),
            outputs.reward.unsafe_ptr(), outputs.discount.unsafe_ptr(),
            outputs.next_value.unsafe_ptr(), p_total,
            self.value_scale, cfg.search_gamma, flag,
            grid_dim=blocks, block_dim=TPB_CART)

        ctx.enqueue_function[cartpole_policy_logits_kernel, cartpole_policy_logits_kernel](
            outputs.action_logits.unsafe_ptr(), p_total,
            grid_dim=blocks, block_dim=TPB_CART)


def default_cartpole() -> CartPole:
    """La config de la demo: V ≡ 0 (sin red) y el paso 500 como muerte (opcion A)."""
    return CartPole(value_scale=0.0, truncate_on_step_limit=False)


# --- El entorno REAL: donde las acciones se ejecutan de verdad -------------
#
# Hasta aqui todo era imaginacion dentro de la busqueda. Estos dos ultimos
# kernels son el entorno de verdad: la accion elegida se ejecuta, se acumula el
# retorno y el episodio se reinicia al terminar. Es lo que usa la demo (y en la
# fase 8, el bucle de actuacion del learner).
#
# Diferencia con cartpole_recurrent_kernel: aquel pliega gamma y el bootstrap
# para la BUSQUEDA; este solo da reward y done, que es lo que necesita jugar de
# verdad. Comparten la fisica (cartpole_advance) y la terminacion.

def cartpole_env_step_kernel(state: GlobalF32, action: GlobalI32,
                             reward_out: GlobalF32, done_out: GlobalI32,
                             n_envs: Int):
    """Un paso del entorno REAL. Un hilo por env.

    reward = 1 - prev_terminal (gymnax): el paso que termina el episodio TAMBIEN
    da 1, porque prev_terminal mira el estado de entrada, no el de salida. Por eso
    el retorno de un episodio es el numero de pasos que aguanto."""
    e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= n_envs:
        return

    base = e * STATE_DIM
    x = state[base + 0]
    x_dot = state[base + 1]
    theta = state[base + 2]
    theta_dot = state[base + 3]
    time = state[base + 4]

    prev_terminal = cartpole_is_terminal(x, theta, time)
    reward = Scalar[dtype](0) if prev_terminal else Scalar[dtype](1)

    ns = cartpole_advance(x, x_dot, theta, theta_dot, time, Int(action[e]))
    done = cartpole_is_terminal(ns.x, ns.theta, ns.time)

    state[base + 0] = ns.x
    state[base + 1] = ns.x_dot
    state[base + 2] = ns.theta
    state[base + 3] = ns.theta_dot
    state[base + 4] = ns.time
    reward_out[e] = reward
    done_out[e] = Scalar[idx_dtype](1) if done else Scalar[idx_dtype](0)


def random_actions_kernel(actions_out: GlobalI32, uniforms: GlobalF32, n: Int):
    """Politica aleatoria uniforme: accion 1 si u >= 0.5, si no 0. Es la linea
    base contra la que se mide la busqueda en la demo (la busqueda tiene que
    jugar mucho mejor que tirar una moneda cada paso)."""
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n:
        actions_out[i] = Scalar[idx_dtype](1) if uniforms[i] >= 0.5 else Scalar[idx_dtype](0)


def cartpole_step_envs(ctx: DeviceContext, state: DeviceBuffer[dtype],
                       actions: DeviceBuffer[idx_dtype],
                       reward: DeviceBuffer[dtype], done: DeviceBuffer[idx_dtype],
                       reset_u: DeviceBuffer[dtype], n_envs: Int,
                       seed: UInt32, reset_stream: UInt32) raises:
    """Un paso del entorno real para los n_envs, con auto-reset de los terminados.

    Deja `reward` y `done` listos para que el host acumule el retorno. El
    auto-reset va DESPUES del paso: los envs que acaban de terminar empiezan otro
    episodio desde un estado nuevo; los vivos siguen. `reset_stream` tiene que
    variar por paso para que los reinicios no repitan el mismo estado inicial."""
    blocks = (n_envs + TPB_CART - 1) // TPB_CART

    ctx.enqueue_function[cartpole_env_step_kernel, cartpole_env_step_kernel](
        state.unsafe_ptr(), actions.unsafe_ptr(), reward.unsafe_ptr(),
        done.unsafe_ptr(), n_envs, grid_dim=blocks, block_dim=TPB_CART)

    ctx.enqueue_function[fill_uniform, fill_uniform](
        reset_u.unsafe_ptr(), seed, reset_stream, n_envs * 4,
        grid_dim=(n_envs * 4 + TPB_CART - 1) // TPB_CART, block_dim=TPB_CART)

    ctx.enqueue_function[cartpole_auto_reset_kernel, cartpole_auto_reset_kernel](
        state.unsafe_ptr(), done.unsafe_ptr(), reset_u.unsafe_ptr(), n_envs,
        grid_dim=blocks, block_dim=TPB_CART)
