"""Reducciones por fila: sum, max y argmax.

Layout: un bloque por fila, un hilo por elemento (el patron "axis sum" del
Puzzle 15). Es el layout que va a usar SPO: bloque = env, hilo = particula.

Este modulo tiene dos niveles:

  1. Los *primitivos de bloque* (`block_reduce_sum`, `block_reduce_max`), que
     reducen un array que ya esta en shared memory. Los reutilizan softmax.mojo
     y rng.mojo, que necesitan reducir a mitad de camino de otra cosa.
  2. Los *kernels* (`sum_rows`, `max_rows`, `argmax_rows`), que cargan la fila
     desde memoria global y llaman al primitivo.

Si la fila es mas larga que el bloque, cada hilo acumula a saltos de TPB antes
de reducir (Puzzle 12), asi que row_size puede ser cualquiera.

TPB tiene que ser potencia de dos: el arbol va partiendo el rango a la mitad y
si TPB no es potencia de dos se queda un trozo sin combinar.
"""

from std.builtin.debug_assert import debug_assert
from std.gpu import block_dim, block_idx, thread_idx, barrier, WARP_SIZE
from std.gpu.primitives.warp import shuffle_xor
from std.memory import stack_allocation, AddressSpace

from ops.common import dtype, idx_dtype, NEG_INF, SharedF32, GlobalF32, GlobalI32


# ---------------------------------------------------------------------------
# Primitivos de bloque: reducen shared[0..TPB) y devuelven el resultado a TODOS
# los hilos (no solo al 0), que es lo que hace falta cuando el valor reducido se
# usa despues, como el maximo en el softmax.
#
# Contrato de los dos:
#   Antes  : shared[0..TPB) relleno y con un barrier() hecho.
#   Despues: shared queda machacado y se puede volver a rellenar; salen con un
#            barrier() para que nadie lo pise mientras otro aun lo lee.
# ---------------------------------------------------------------------------

def block_reduce_sum[TPB: Int](shared: SharedF32, tid: Int) -> Scalar[dtype]:
    stride = TPB // 2
    while stride > 0:
        # El que escribe (tid < stride) nunca es el que otro esta leyendo
        # (tid + stride >= stride), asi que un solo barrier por ronda basta.
        # Ojo: esto NO vale para el scan, donde si se solapan (ver scan.mojo).
        if tid < stride:
            shared[tid] += shared[tid + stride]
        barrier()
        stride //= 2

    total = shared[0]
    barrier()
    return total


def block_reduce_max[TPB: Int](shared: SharedF32, tid: Int) -> Scalar[dtype]:
    stride = TPB // 2
    while stride > 0:
        if tid < stride:
            other = shared[tid + stride]
            if other > shared[tid]:
                shared[tid] = other
        barrier()
        stride //= 2

    biggest = shared[0]
    barrier()
    return biggest


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def sum_rows[TPB: Int](out_ptr: GlobalF32, a_ptr: GlobalF32, row_size: Int):
    """out[fila] = suma de la fila. Lanzar con grid_dim=num_filas, block_dim=TPB."""
    # shared se dimensiona con TPB, asi que lanzar con mas hilos escribiria fuera.
    debug_assert(Int(block_dim.x) == TPB, "sum_rows: block_dim tiene que ser TPB")

    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    row = Int(block_idx.x)
    base = row * row_size

    # Paso 1: cada hilo se come su parte de la fila (a saltos de TPB).
    acc = Scalar[dtype](0)
    i = tid
    while i < row_size:
        acc += a_ptr[base + i]
        i += TPB
    shared[tid] = acc
    barrier()

    # Paso 2: los TPB parciales se combinan en arbol.
    total = block_reduce_sum[TPB](shared, tid)

    if tid == 0:
        out_ptr[row] = total


def max_rows[TPB: Int](out_ptr: GlobalF32, a_ptr: GlobalF32, row_size: Int):
    """out[fila] = maximo de la fila."""
    debug_assert(Int(block_dim.x) == TPB, "max_rows: block_dim tiene que ser TPB")

    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    row = Int(block_idx.x)
    base = row * row_size

    # Los hilos que no alcanzan ningun elemento (row_size < TPB) se quedan en
    # NEG_INF, que pierde contra cualquier valor real.
    acc = NEG_INF
    i = tid
    while i < row_size:
        v = a_ptr[base + i]
        if v > acc:
            acc = v
        i += TPB
    shared[tid] = acc
    barrier()

    biggest = block_reduce_max[TPB](shared, tid)

    if tid == 0:
        out_ptr[row] = biggest


def argmax_rows[TPB: Int](out_ptr: GlobalI32, a_ptr: GlobalF32, row_size: Int):
    """out[fila] = indice del maximo. Con empates gana el indice MENOR.

    La regla de desempate no es un capricho: en una reduccion en arbol el orden
    en que se combinan los elementos depende de TPB, asi que sin una regla
    explicita el resultado cambiaria al cambiar el tamano del bloque. Justo el
    tipo de no-determinismo que hace que un test falle un dia de cada diez.

    No usa block_reduce_max porque hay que arrastrar el indice junto al valor.
    """
    debug_assert(Int(block_dim.x) == TPB, "argmax_rows: block_dim tiene que ser TPB")

    shared_val = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    shared_idx = stack_allocation[TPB, Scalar[idx_dtype], address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    row = Int(block_idx.x)
    base = row * row_size

    best = NEG_INF
    best_i = Scalar[idx_dtype](-1)
    i = tid
    while i < row_size:
        v = a_ptr[base + i]
        # Comparacion estricta: si empata se queda el que ya tenia, que viene de
        # un i menor porque el bucle sube.
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
            # Cedo si el otro es estrictamente mayor; si empatan, gana el indice
            # menor. Un hilo sin datos lleva NEG_INF y nunca gana (su -1 no puede
            # colarse). Como cada mitad ya cumple "el indice menor de los que
            # empatan en el maximo", combinarlas asi lo mantiene.
            take = other > mine
            if other == mine:
                take = shared_idx[tid + stride] < shared_idx[tid]
            if take:
                shared_val[tid] = other
                shared_idx[tid] = shared_idx[tid + stride]
        barrier()
        stride //= 2

    if tid == 0:
        out_ptr[row] = shared_idx[0]


def warp_sum_rows(out_ptr: GlobalF32, a_ptr: GlobalF32, row_size: Int):
    """Suma por fila con butterfly de warp (Puzzle 26): ni shared memory ni barrier.

    Dentro de un warp los 32 lanes van en lockstep y el valor viaja por
    registros, asi que sobra la sincronizacion. Sale bastante mas corto que la
    version en arbol, a cambio de exigir row_size <= WARP_SIZE.

    Para SPO esa restriccion no molesta: son 16 particulas, medio warp.
    Lanzar con block_dim=WARP_SIZE.
    """
    debug_assert(row_size <= Int(WARP_SIZE),
                 "warp_sum_rows: la fila tiene que caber en un warp")
    debug_assert(Int(block_dim.x) == Int(WARP_SIZE),
                 "warp_sum_rows: block_dim tiene que ser WARP_SIZE")

    tid = Int(thread_idx.x)
    row = Int(block_idx.x)
    base = row * row_size

    # Los lanes sobrantes aportan 0, que es el neutro de la suma.
    v = a_ptr[base + tid] if tid < row_size else Scalar[dtype](0)

    # shuffle_xor(v, offset) trae el valor del lane tid^offset. Tras
    # log2(WARP_SIZE) rondas TODOS los lanes tienen la suma total (por eso se
    # llama butterfly y no reduccion: no colapsa hacia el lane 0).
    offset = Int(WARP_SIZE) // 2
    while offset > 0:
        v += shuffle_xor(v, UInt32(offset))
        offset //= 2

    if tid == 0:
        out_ptr[row] = v
