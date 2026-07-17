"""Reducciones por fila: sum, max y argmax.

Layout: un bloque por fila, un hilo por elemento (el patron "axis sum" del Puzzle 15).
Es el layout que va a usar SPO: bloque = env, hilo = particula.

Si la fila es mas larga que el bloque, cada hilo acumula a saltos de TPB antes de
entrar a la reduccion en arbol (Puzzle 12), asi que row_size puede ser cualquiera.

TPB tiene que ser potencia de dos (la reduccion en arbol va partiendo a la mitad).
"""

from std.gpu import block_idx, thread_idx, barrier, WARP_SIZE
from std.gpu.primitives.warp import shuffle_xor
from std.memory import stack_allocation, AddressSpace

comptime dtype = DType.float32
comptime idx_dtype = DType.int32

# Un float32 lo bastante negativo como para perder cualquier comparacion.
comptime NEG_INF = Scalar[dtype](-3.4028235e38)


def sum_rows[TPB: Int](out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
                       a_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
                       row_size: Int):
    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = thread_idx.x
    base = block_idx.x * row_size

    # Fase 1: cada hilo se come su parte de la fila.
    acc = Scalar[dtype](0)
    i = tid
    while i < row_size:
        acc += a_ptr[base + i]
        i += TPB
    shared[tid] = acc
    barrier()

    # Fase 2: reduccion en arbol. stride es uniforme, asi que el barrier lo ven
    # todos los hilos del bloque (la regla del Puzzle 10).
    stride = TPB // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        barrier()
        stride //= 2

    if tid == 0:
        out_ptr[block_idx.x] = shared[0]


def warp_sum_rows(out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
                  a_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
                  row_size: Int):
    """Variante butterfly del Puzzle 26: un warp por fila, sin shared memory.

    Pide row_size <= WARP_SIZE, que es exactamente el caso de SPO (16 particulas).
    Lanzar con block_dim=WARP_SIZE.
    """
    tid = thread_idx.x
    base = block_idx.x * row_size

    v = a_ptr[base + tid] if tid < row_size else Scalar[dtype](0)

    # shuffle_xor intercambia con el lane tid^offset: tras log2(WARP_SIZE) pasos
    # todos los lanes tienen la suma total. No hace falta barrier: dentro de un
    # warp los hilos van en lockstep.
    offset = WARP_SIZE // 2
    while offset > 0:
        v += shuffle_xor(v, UInt32(offset))
        offset //= 2

    if tid == 0:
        out_ptr[block_idx.x] = v


def max_rows[TPB: Int](out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
                       a_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
                       row_size: Int):
    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = thread_idx.x
    base = block_idx.x * row_size

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

    if tid == 0:
        out_ptr[block_idx.x] = shared[0]


def argmax_rows[TPB: Int](out_ptr: UnsafePointer[Scalar[idx_dtype], MutAnyOrigin],
                          a_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
                          row_size: Int):
    """Con empates gana el indice menor (decidido asi para que sea determinista)."""
    shared_val = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    shared_idx = stack_allocation[TPB, Scalar[idx_dtype], address_space = AddressSpace.SHARED]()
    tid = thread_idx.x
    base = block_idx.x * row_size

    best = NEG_INF
    best_i = Scalar[idx_dtype](-1)
    i = tid
    while i < row_size:
        v = a_ptr[base + i]
        # Estricto: si empata, se queda el que ya tenia (que viene de un i menor).
        if v > best:
            best = v
            best_i = Scalar[idx_dtype](i)
        i += TPB
    shared_val[tid] = best
    shared_idx[tid] = best_i
    barrier()

    stride = TPB // 2
    while stride > 0:
        if tid < stride:
            other = shared_val[tid + stride]
            mine = shared_val[tid]
            # Solo cedo si el otro es estrictamente mayor, o si empata y su
            # indice es menor. Con -1 (hilo sin datos) nunca gana, porque su
            # valor es NEG_INF.
            take = other > mine
            if other == mine:
                take = shared_idx[tid + stride] < shared_idx[tid]
            if take:
                shared_val[tid] = other
                shared_idx[tid] = shared_idx[tid + stride]
        barrier()
        stride //= 2

    if tid == 0:
        out_ptr[block_idx.x] = shared_idx[0]
