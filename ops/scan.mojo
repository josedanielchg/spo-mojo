"""Prefix sum por filas, inclusivo y exclusivo (Hillis-Steele, Puzzle 14).

En SPO esto es el CDF del resampling: partiendo de los pesos normalizados, el
prefix sum inclusivo da los cortes contra los que se compara el uniforme.
El exclusivo sirve para posiciones de escritura (el caso del histograma).

Limite: row_size <= TPB (un bloque por fila, sin scan multi-bloque). Para SPO
sobra, porque la fila es el numero de particulas (16).

Hillis-Steele hace log2(TPB) pasadas y cada una toca todos los hilos, asi que
hace mas trabajo total que un scan de Blelloch; para 16-32 elementos da igual y
el codigo es la mitad de largo.
"""

from std.gpu import block_idx, thread_idx, barrier
from std.memory import stack_allocation, AddressSpace

comptime dtype = DType.float32


def inclusive_scan_rows[TPB: Int](out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
                                  a_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
                                  row_size: Int):
    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = thread_idx.x
    base = block_idx.x * row_size

    shared[tid] = a_ptr[base + tid] if tid < row_size else Scalar[dtype](0)
    barrier()

    offset = 1
    while offset < TPB:
        # Disciplina read -> barrier -> write (Puzzle 10): si escribo shared[tid]
        # antes de que todos hayan leido shared[tid-offset], me como el valor viejo
        # de otro hilo. El barrier de en medio es el que evita la carrera.
        val = shared[tid - offset] if tid >= offset else Scalar[dtype](0)
        barrier()
        shared[tid] += val
        barrier()
        offset *= 2

    if tid < row_size:
        out_ptr[base + tid] = shared[tid]


def exclusive_scan_rows[TPB: Int](out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
                                  a_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
                                  row_size: Int):
    """Igual que el inclusivo pero desplazado uno a la derecha, con 0 al inicio."""
    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = thread_idx.x
    base = block_idx.x * row_size

    shared[tid] = a_ptr[base + tid] if tid < row_size else Scalar[dtype](0)
    barrier()

    offset = 1
    while offset < TPB:
        val = shared[tid - offset] if tid >= offset else Scalar[dtype](0)
        barrier()
        shared[tid] += val
        barrier()
        offset *= 2

    if tid < row_size:
        # out[0] = 0, out[i] = inclusive[i-1]
        out_ptr[base + tid] = shared[tid - 1] if tid > 0 else Scalar[dtype](0)
