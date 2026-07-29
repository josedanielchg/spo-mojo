"""La GAE truncada: los objetivos que aprende el critico.

Port de `batch_truncated_generalized_advantage_estimation` de
`stoix/utils/multistep.py`. Es el UNICO sitio donde se usa la GAE de libreria; no
confundirla con la `calculate_gae` de dentro de la busqueda, que es otra cosa
(esa va hacia ADELANTE y alimenta el dual de temperatura).

El problema que resuelve: ¿a que numero deberia apuntar el critico en cada
posicion? La respuesta ingenua ("la recompensa que llego al final") tiene mucha
varianza; la otra ingenua ("lo que ya predice el critico") no aporta informacion.
La GAE mezcla las dos con un parametro lambda, y lo hace acumulando hacia atras:

    delta_t = r + gamma*v_t - v_tm1                 el error de un paso
    acc     = delta + gamma*lambda*acc*(1 - trunc)  acumulado hacia ATRAS
    target  = v_tm1 + acc                           lo que aprendera el critico

`discount` ya viene con el done plegado: en Stoix se calcula como
`(1 - done) * gamma`, asi que en el ultimo paso de un episodio vale 0 y corta el
bootstrap solo.

**El detalle que hay que respetar**, y esta comentado igual en Stoix: en un paso
truncado el acumulador se resetea (`* (1 - truncation)`) PERO el delta de ese
mismo paso si se usa. Truncar no borra la informacion del paso, corta la
propagacion hacia atras. Confundirlo es el bug silencioso clasico de RL: el
entrenamiento sigue corriendo, solo aprende objetivos mal calculados.

En tres en raya la truncacion nunca se dispara (las partidas acaban solas en 5
jugadas como mucho, no hay limite de tiempo), pero se implementa igual: es parte
de la funcion de referencia y el test la ejercita.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from ops.common import dtype, GlobalF32

comptime TPB_GAE = 128


def gae_kernel(adv_out: GlobalF32, target_out: GlobalF32, reward: GlobalF32,
               discount: GlobalF32, v_tm1: GlobalF32, v_t: GlobalF32,
               truncation: GlobalF32, batch: Int, t_len: Int,
               lambda_: Scalar[dtype]):
    """Una secuencia por hilo, recorrida hacia atras en el tiempo.

    El paralelismo esta en el BATCH y no en el tiempo, porque la recurrencia es
    secuencial por naturaleza (cada paso necesita el acumulado del siguiente).
    Con T = 32 el bucle es corto y no compensa complicarlo con un scan paralelo.

    Layout batch-major [B, T], que es el que usa Stoix por defecto
    (`time_major=False`).
    """
    b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= batch:
        return

    acc = Scalar[dtype](0)
    for i in range(t_len):
        t = t_len - 1 - i               # hacia atras
        idx = b * t_len + t

        delta = reward[idx] + discount[idx] * v_t[idx] - v_tm1[idx]
        # El delta SI entra aunque el paso este truncado; lo que se corta es el
        # acumulado que venia de mas adelante.
        acc = delta + discount[idx] * lambda_ * acc \
              * (Scalar[dtype](1) - truncation[idx])

        adv_out[idx] = acc
        target_out[idx] = v_tm1[idx] + acc


def truncated_gae(ctx: DeviceContext, adv: DeviceBuffer[dtype],
                  targets: DeviceBuffer[dtype], reward: DeviceBuffer[dtype],
                  discount: DeviceBuffer[dtype], v_tm1: DeviceBuffer[dtype],
                  v_t: DeviceBuffer[dtype], truncation: DeviceBuffer[dtype],
                  batch: Int, t_len: Int, lambda_: Scalar[dtype]) raises:
    """Ventajas y objetivos del critico para un batch de secuencias [B, T]."""
    ctx.enqueue_function[gae_kernel, gae_kernel](
        adv.unsafe_ptr(), targets.unsafe_ptr(), reward.unsafe_ptr(),
        discount.unsafe_ptr(), v_tm1.unsafe_ptr(), v_t.unsafe_ptr(),
        truncation.unsafe_ptr(), batch, t_len, lambda_,
        grid_dim=(batch + TPB_GAE - 1) // TPB_GAE, block_dim=TPB_GAE)
