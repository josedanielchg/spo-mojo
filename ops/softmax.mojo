"""Softmax estable y logsumexp por filas (Puzzle 18).

El truco es restar el maximo de la fila antes de exponenciar: exp(1000) es inf,
pero exp(1000 - 1000) = 1. Matematicamente da igual porque el factor exp(-m) se
cancela arriba y abajo; numericamente es la diferencia entre funcionar y no.

En SPO esto se usa dos veces: para la cabeza categorica del actor y para
convertir los pesos SMC w/eta en probabilidades de resampling.

Layout: un bloque por fila (bloque = env, hilo = particula).
TPB tiene que ser potencia de dos.
"""

from std.gpu import block_idx, thread_idx, barrier
from std.math import exp, log
from std.memory import stack_allocation, AddressSpace

comptime dtype = DType.float32
comptime NEG_INF = Scalar[dtype](-3.4028235e38)


def _row_max[TPB: Int](shared: UnsafePointer[Scalar[dtype], MutAnyOrigin, address_space = AddressSpace.SHARED],
                       a_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
                       base: Int, row_size: Int, tid: Int) -> Scalar[dtype]:
    acc = NEG_INF
    i = tid
    while i < row_size:
        v = a_ptr[base + i]
        if v > acc:
            acc = v
        i += TPB
    shared[tid] = acc
    barrier()

    stride = TPB // 2
    while stride > 0:
        if tid < stride:
            other = shared[tid + stride]
            if other > shared[tid]:
                shared[tid] = other
        barrier()
        stride //= 2

    m = shared[0]
    # Todos tienen que haber leido shared[0] antes de que nadie lo pise.
    barrier()
    return m


def _row_sum_exp[TPB: Int](shared: UnsafePointer[Scalar[dtype], MutAnyOrigin, address_space = AddressSpace.SHARED],
                           a_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
                           base: Int, row_size: Int, tid: Int,
                           row_max: Scalar[dtype]) -> Scalar[dtype]:
    acc = Scalar[dtype](0)
    i = tid
    while i < row_size:
        acc += exp(a_ptr[base + i] - row_max)
        i += TPB
    shared[tid] = acc
    barrier()

    stride = TPB // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        barrier()
        stride //= 2

    s = shared[0]
    barrier()
    return s


def softmax_rows[TPB: Int](out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
                           a_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
                           row_size: Int):
    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = thread_idx.x
    base = block_idx.x * row_size

    row_max = _row_max[TPB](shared, a_ptr, base, row_size, tid)
    row_sum = _row_sum_exp[TPB](shared, a_ptr, base, row_size, tid, row_max)

    i = tid
    while i < row_size:
        out_ptr[base + i] = exp(a_ptr[base + i] - row_max) / row_sum
        i += TPB


def logsumexp_rows[TPB: Int](out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
                             a_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
                             row_size: Int):
    """log(sum(exp(x))) = m + log(sum(exp(x - m))). Un valor por fila."""
    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = thread_idx.x
    base = block_idx.x * row_size

    row_max = _row_max[TPB](shared, a_ptr, base, row_size, tid)
    row_sum = _row_sum_exp[TPB](shared, a_ptr, base, row_size, tid, row_max)

    if tid == 0:
        out_ptr[block_idx.x] = row_max + log(row_sum)
