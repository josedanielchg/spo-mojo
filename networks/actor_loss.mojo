"""La perdida del M-step: entropia cruzada ponderada (ecuacion 11 del paper).

    max_theta  E_{s~mu} [ E_{a~q(.|s)} [ log pi(a|s,theta) ] ]

o sea, minimizar  -SUM_a q(a) log pi(a)  promediado sobre los estados. El paper la
describe como "projecting the non-parametric policy q back to the space of
parametrisable policies", igual que hace AlphaZero: la busqueda produce una politica
mejorada y la red aprende a imitarla.

**Forma densa y forma de particulas son la misma cantidad.** Stoix la implementa
como estimador Monte Carlo sobre las N particulas,

    loss = -SUM_n w_n log pi(a_n)            (compute_cross_entropy_loss)

y como las acciones raiz se repiten entre particulas, agrupando por accion

    SUM_n w_n log pi(a_n) = SUM_a ( SUM_{n: a_n=a} w_n ) log pi(a) = SUM_a q(a) log pi(a)

No es una aproximacion: es la misma suma reagrupada. El golden lo demuestra
llamando a la funcion de Stoix de verdad y comparando (diff ~1e-8).

Aqui se usa la densa por dos razones. Nuestro readout ya produce q como vector
[B, 9], asi que sumar sobre 9 acciones sale mas barato que recolectar log-probs de
512 particulas con repeticiones; y al no muestrear, no arrastra varianza de
muestreo. En tres en raya el espacio de acciones es diminuto, asi que la densa es
siempre viable -- en un entorno con miles de acciones habria que volver a la de
particulas, y por eso el golden guarda las dos.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import log
from std.gpu.host import DeviceContext, DeviceBuffer

from ops.common import dtype, GlobalF32
from systems.spo.launch import TPB, blocks_for


def cross_entropy_rows_kernel(loss_out: GlobalF32, q: GlobalF32,
                              log_pi: GlobalF32, n_rows: Int,
                              num_actions: Int):
    """loss[fila] = -SUM_a q[a] * log_pi[a]. Un hilo por estado.

    Un hilo por fila y no por (fila, accion) porque num_actions es 9: repartirlo
    costaria mas en reduccion que en aritmetica.

    **El `if w != 0` no es una optimizacion, es correccion.** En una casilla
    ilegal q vale 0 y log_pi vale menos infinito (o el finito mas negativo, segun
    como se haya enmascarado). El producto 0 * (-inf) es NaN en IEEE, y un NaN
    aqui se propagaria callado a la perdida, al gradiente y a los pesos, sin que
    nada falle ruidosamente. Saltarse el termino da el valor correcto -- que es 0,
    porque una accion con probabilidad objetivo nula no contribuye -- sea cual sea
    la representacion del enmascarado.
    """
    row = Int(block_dim.x * block_idx.x + thread_idx.x)
    if row >= n_rows:
        return
    acc = Scalar[dtype](0)
    base = row * num_actions
    for a in range(num_actions):
        w = q[base + a]
        if w != Scalar[dtype](0):
            acc += w * log_pi[base + a]
    loss_out[row] = -acc


def cross_entropy_rows(ctx: DeviceContext, loss_out: DeviceBuffer[dtype],
                       q: DeviceBuffer[dtype], log_pi: DeviceBuffer[dtype],
                       n_rows: Int, num_actions: Int) raises:
    """La perdida de cada estado, sin promediar.

    Se deja sin promediar a proposito: el promedio es una reduccion que el bucle
    de entrenamiento ya hace en host para reportar (igual que el critico en
    E1.10), y el gradiente necesita el factor 1/n plegado en otro sitio.
    """
    ctx.enqueue_function[cross_entropy_rows_kernel, cross_entropy_rows_kernel](
        loss_out.unsafe_ptr(), q.unsafe_ptr(), log_pi.unsafe_ptr(), n_rows,
        num_actions, grid_dim=blocks_for(n_rows), block_dim=TPB)


# ---------------------------------------------------------------------------
# El diagnostico: separar el suelo de lo que de verdad aprende la red.
#
# La entropia cruzada se descompone exactamente en
#
#     H(q, pi)  =  H(q)  +  KL(q || pi)
#
# y H(q) NO depende de pi. Es el suelo: cuando el actor llega a q, la KL vale 0 y
# la perdida se queda en H(q), que puede ser cualquier cosa.
#
# Por que importa reportar la KL y no la perdida cruda. En el bucle real q la
# produce la busqueda y CAMBIA entre iteraciones, asi que el suelo se mueve: la
# perdida puede subir con el actor aprendiendo mejor, solo porque q se volvio mas
# dispersa. Y al comparar dos readouts es peor todavia -- medimos H(q) = 1.178 con
# el readout de SPO y ~0 con la variante (su q es casi one-hot), o sea 1.18 nats
# de diferencia solo en el suelo. Las perdidas crudas de los dos brazos no serian
# comparables.
#
# La KL si: su cero significa "el actor reproduce exactamente lo que dice la
# busqueda", y ese cero es el mismo en todas las configuraciones.
# ---------------------------------------------------------------------------


def entropy_rows_kernel(h_out: GlobalF32, q: GlobalF32, n_rows: Int,
                        num_actions: Int):
    """H(q)[fila] = -SUM_a q[a] log q[a]. Un hilo por estado.

    El `if w > 0` es el convenio 0*log(0) = 0, y ademas evita el NaN: log(0) es
    -inf y 0 * (-inf) da NaN. Con acciones ilegales q vale 0 exacto, asi que este
    caso ocurre SIEMPRE, no es defensivo.
    """
    row = Int(block_dim.x * block_idx.x + thread_idx.x)
    if row >= n_rows:
        return
    acc = Scalar[dtype](0)
    base = row * num_actions
    for a in range(num_actions):
        w = q[base + a]
        if w > Scalar[dtype](0):
            acc += w * log(w)
    h_out[row] = -acc


def kl_rows_kernel(kl_out: GlobalF32, cross_entropy: GlobalF32,
                   entropy: GlobalF32, n_rows: Int):
    """KL(q||pi) = H(q,pi) - H(q). Un hilo por estado."""
    row = Int(block_dim.x * block_idx.x + thread_idx.x)
    if row < n_rows:
        kl_out[row] = cross_entropy[row] - entropy[row]


def entropy_rows(ctx: DeviceContext, h_out: DeviceBuffer[dtype],
                 q: DeviceBuffer[dtype], n_rows: Int,
                 num_actions: Int) raises:
    """H(q) de cada estado: el suelo que la perdida no puede bajar."""
    ctx.enqueue_function[entropy_rows_kernel, entropy_rows_kernel](
        h_out.unsafe_ptr(), q.unsafe_ptr(), n_rows, num_actions,
        grid_dim=blocks_for(n_rows), block_dim=TPB)


def kl_rows(ctx: DeviceContext, kl_out: DeviceBuffer[dtype],
            cross_entropy: DeviceBuffer[dtype], entropy: DeviceBuffer[dtype],
            n_rows: Int) raises:
    """La KL a partir de la entropia cruzada y la entropia, ya calculadas."""
    ctx.enqueue_function[kl_rows_kernel, kl_rows_kernel](
        kl_out.unsafe_ptr(), cross_entropy.unsafe_ptr(), entropy.unsafe_ptr(),
        n_rows, grid_dim=blocks_for(n_rows), block_dim=TPB)
