"""CartPole-v1: la fisica exacta de gymnax, vectorizada.

Reproduce paso a paso `step_env` de
gymnax/environments/classic_control/cartpole.py. El estado son 5 numeros:
(x, x_dot, theta, theta_dot, time). El `time` va como float pero es exacto
(los enteros hasta 2^24 lo son, y el maximo es 500).

Esta etapa es SOLO la dinamica (la integracion de Euler). La terminacion, el
discount y el reward vienen en la etapa siguiente; el struct SearchModel en la de
despues. El fichero va creciendo por etapas.

Las constantes son los EnvParams de gymnax. Ojo con la precision: en Python son
float64, pero JAX las mantiene en float32 al operar con el estado (float32) por
weak typing, asi que aqui van en float32 y el resultado cuadra con el golden.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import sin, cos

from ops.common import dtype, idx_dtype, GlobalF32, GlobalI32

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


def cartpole_physics_kernel(state_out: GlobalF32, state_in: GlobalF32,
                            action: GlobalI32, n_particles: Int):
    """Un paso de la dinamica de CartPole. Un hilo por particula.

    Espeja `step_env` de gymnax (lineas 61-90). NO comprueba terminacion: gymnax
    integra la fisica pase lo que pase -- una particula ya caida sigue
    evolucionando-- y la terminacion solo afecta al reward y al done, que se
    calculan aparte (etapa siguiente).

    In-place seguro: cada hilo lee sus 5 componentes en registros ANTES de
    escribir ninguna, asi que state_out puede ser el mismo buffer que state_in.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= n_particles:
        return

    base = i * STATE_DIM
    x = state_in[base + 0]
    x_dot = state_in[base + 1]
    theta = state_in[base + 2]
    theta_dot = state_in[base + 3]
    time = state_in[base + 4]

    # force = +FORCE_MAG si la accion es 1, -FORCE_MAG si es 0. Se calcula con la
    # misma expresion que gymnax (no con un if) para que el float32 salga identico.
    a = Scalar[dtype](Int(action[i]))
    force = FORCE_MAG * a - FORCE_MAG * (Scalar[dtype](1) - a)

    costheta = cos(theta)
    sintheta = sin(theta)

    # La dinamica del pendulo invertido sobre el carro, tal cual gymnax.
    temp = (force + POLEMASS_LENGTH * (theta_dot * theta_dot) * sintheta) / TOTAL_MASS
    thetaacc = (GRAVITY * sintheta - costheta * temp) / (
        LENGTH * (FOUR_THIRDS - MASSPOLE * (costheta * costheta) / TOTAL_MASS))
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

    # Integracion de Euler (la unica que ofrece gymnax).
    state_out[base + 0] = x + TAU * x_dot
    state_out[base + 1] = x_dot + TAU * xacc
    state_out[base + 2] = theta + TAU * theta_dot
    state_out[base + 3] = theta_dot + TAU * thetaacc
    state_out[base + 4] = time + Scalar[dtype](1)
