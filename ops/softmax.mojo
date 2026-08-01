"""Softmax estable y logsumexp por filas.

El truco es restar el maximo de la fila antes de exponenciar. exp(1000) es inf
y inf/inf es nan, pero exp(1000 - 1000) = 1. Matematicamente da igual, porque
el factor exp(-m) aparece en el numerador y en el denominador y se cancela:

    exp(x_i - m) / SUM_j exp(x_j - m)  ==  exp(x_i) / SUM_j exp(x_j)

Numericamente es la diferencia entre funcionar y devolver nan. El test tiene una
fila de +1000 y otra de -1000 justo para vigilar esto.

En SPO se usa en dos sitios: la cabeza categorica del actor, y la conversion de
los pesos SMC w/eta en probabilidades de resampling.

Layout: un bloque por fila (bloque = env, hilo = particula).
TPB tiene que ser potencia de dos. row_size puede ser mayor que TPB.
"""

from std.builtin.debug_assert import debug_assert
from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.math import exp, log
from std.memory import stack_allocation, AddressSpace

from ops.common import dtype, NEG_INF, SharedF32, GlobalF32
from ops.reductions import block_reduce_sum, block_reduce_max


def _row_max[TPB: Int](shared: SharedF32, a_ptr: GlobalF32,
                       base: Int, row_size: Int, tid: Int) -> Scalar[dtype]:
    """Maximo de la fila, devuelto a todos los hilos. Deja shared reutilizable."""
    acc = NEG_INF
    i = tid
    while i < row_size:
        v = a_ptr[base + i]
        if v > acc:
            acc = v
        i += TPB
    shared[tid] = acc
    barrier()
    return block_reduce_max[TPB](shared, tid)


def _row_sum_exp[TPB: Int](shared: SharedF32, a_ptr: GlobalF32,
                           base: Int, row_size: Int, tid: Int,
                           row_max: Scalar[dtype]) -> Scalar[dtype]:
    """SUM_i exp(a[i] - row_max), devuelto a todos los hilos."""
    acc = Scalar[dtype](0)
    i = tid
    while i < row_size:
        acc += exp(a_ptr[base + i] - row_max)
        i += TPB
    shared[tid] = acc
    barrier()
    return block_reduce_sum[TPB](shared, tid)


def softmax_rows[TPB: Int](out_ptr: GlobalF32, a_ptr: GlobalF32, row_size: Int):
    """out = softmax(a) por filas."""
    debug_assert(Int(block_dim.x) == TPB, "softmax_rows: block_dim tiene que ser TPB")

    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    row = Int(block_idx.x)
    base = row * row_size

    row_max = _row_max[TPB](shared, a_ptr, base, row_size, tid)
    row_sum = _row_sum_exp[TPB](shared, a_ptr, base, row_size, tid, row_max)

    # Se recalcula el exp en vez de guardarlo de la pasada anterior. Guardarlo
    # pediria otro array en shared de tamano row_size, y row_size puede ser mayor
    # que TPB (que es justo lo que shared mide). Prefiero pagar el exp dos veces
    # que meter un limite artificial; si algun dia duele, se perfila primero.
    i = tid
    while i < row_size:
        out_ptr[base + i] = exp(a_ptr[base + i] - row_max) / row_sum
        i += TPB


def logsumexp_rows[TPB: Int](out_ptr: GlobalF32, a_ptr: GlobalF32, row_size: Int):
    """out[fila] = log(SUM exp(a)), un escalar por fila.

    Identidad usada: log(SUM exp(x)) = m + log(SUM exp(x - m)).
    En SPO aparece en el temperature loss del M-step (ecuacion 7 del paper).
    """
    debug_assert(Int(block_dim.x) == TPB, "logsumexp_rows: block_dim tiene que ser TPB")

    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    row = Int(block_idx.x)
    base = row * row_size

    row_max = _row_max[TPB](shared, a_ptr, base, row_size, tid)
    row_sum = _row_sum_exp[TPB](shared, a_ptr, base, row_size, tid, row_max)

    if tid == 0:
        out_ptr[row] = row_max + log(row_sum)


def log_softmax_rows[TPB: Int](out_ptr: GlobalF32, a_ptr: GlobalF32,
                               row_size: Int):
    """out = log(softmax(a)) por filas, calculado sin pasar por el softmax.

    Identidad: log softmax(a)[i] = a[i] - (m + log(SUM exp(a - m))), con m el
    maximo de la fila. Es el mismo denominador que `logsumexp_rows`.

    Por que no `log(softmax(a))` y ya: el softmax de un logit muy bajo desborda a
    0 y su log seria -inf aunque el valor exacto fuera perfectamente
    representable (-40, pongamos). Calcularlo directo conserva esos valores, que
    es justo lo que necesita una entropia cruzada: los terminos con probabilidad
    pequena son los que mas pesan en el log.

    En las casillas ENMASCARADAS la entrada es NEG_INF (el float32 finito mas
    negativo) y la salida se queda ahi: restarle un denominador de orden 1 no la
    mueve, porque a esa magnitud el ULP es ~1e31. Sale un valor finito y muy
    negativo, no un -inf, que es lo que conviene para no propagar NaN.
    """
    debug_assert(Int(block_dim.x) == TPB,
                 "log_softmax_rows: block_dim tiene que ser TPB")

    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    row = Int(block_idx.x)
    base = row * row_size

    row_max = _row_max[TPB](shared, a_ptr, base, row_size, tid)
    row_sum = _row_sum_exp[TPB](shared, a_ptr, base, row_size, tid, row_max)
    log_denom = row_max + log(row_sum)

    i = tid
    while i < row_size:
        out_ptr[base + i] = a_ptr[base + i] - log_denom
        i += TPB
