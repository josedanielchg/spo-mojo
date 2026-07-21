"""Prefix sum por filas, inclusivo y exclusivo.

En SPO esto acaba siendo el CDF del resampling: partiendo de los pesos, el
prefix sum inclusivo da los cortes contra los que se compara el uniforme. El
exclusivo sirve para calcular posiciones de escritura, como en un histograma.

Igual que en reductions.mojo, primero va el primitivo de bloque y luego los
kernels que lo envuelven.

El limite es row_size <= TPB. Un scan de verdad para arrays mas grandes que un
bloque necesita tres pasadas (scan por bloque, scan de los totales, y sumar el
offset), y aqui no hace falta porque la fila es el numero de particulas, 16.

El algoritmo es Hillis-Steele: log2(TPB) pasadas y cada una toca todos los
hilos, o sea O(n log n) sumas frente a las O(n) de un scan de Blelloch. Para
16-32 elementos el trabajo de mas no se nota y el codigo es la mitad de largo.
"""

from std.builtin.debug_assert import debug_assert
from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.memory import stack_allocation, AddressSpace

from ops.common import dtype, SharedF32, GlobalF32


def block_scan_inclusive[TPB: Int](shared: SharedF32, tid: Int):
    """Prefix sum inclusivo in-place sobre shared[0..TPB).

    Antes  : shared relleno y con barrier() hecho.
    Despues: shared[i] = suma de shared[0..i], y barrier() hecho.
    """
    offset = 1
    while offset < TPB:
        # Aqui si hacen falta dos barriers por ronda, al contrario que en la
        # reduccion en arbol. Motivo: el hilo tid escribe shared[tid] mientras
        # el hilo tid+offset quiere leer ese MISMO shared[tid]. Los rangos de
        # lectura y escritura se solapan, asi que hay que separar en el tiempo
        # "todos leen" de "todos escriben".
        #
        # Sin el barrier de en medio el bug es de los peores: no revienta, solo
        # da un resultado distinto segun como el scheduler ordene los warps.
        val = shared[tid - offset] if tid >= offset else Scalar[dtype](0)
        barrier()
        shared[tid] += val
        barrier()
        offset *= 2


def inclusive_scan_rows[TPB: Int](out_ptr: GlobalF32, a_ptr: GlobalF32, row_size: Int):
    """out[i] = a[0] + ... + a[i], por filas."""
    debug_assert(row_size <= TPB, "inclusive_scan_rows: row_size tiene que caber en TPB")
    debug_assert(Int(block_dim.x) == TPB, "inclusive_scan_rows: block_dim tiene que ser TPB")

    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    base = Int(block_idx.x) * row_size

    # Los hilos sobrantes cargan 0 (neutro de la suma) para que el scan del
    # bloque entero salga bien sin casos especiales.
    shared[tid] = a_ptr[base + tid] if tid < row_size else Scalar[dtype](0)
    barrier()

    block_scan_inclusive[TPB](shared, tid)

    if tid < row_size:
        out_ptr[base + tid] = shared[tid]


def exclusive_scan_rows[TPB: Int](out_ptr: GlobalF32, a_ptr: GlobalF32, row_size: Int):
    """out[0] = 0, out[i] = a[0] + ... + a[i-1].

    Es el inclusivo desplazado un puesto a la derecha. El uso tipico es convertir
    conteos en offsets de escritura: cada bucket sabe donde empieza su tramo.
    """
    debug_assert(row_size <= TPB, "exclusive_scan_rows: row_size tiene que caber en TPB")
    debug_assert(Int(block_dim.x) == TPB, "exclusive_scan_rows: block_dim tiene que ser TPB")

    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    base = Int(block_idx.x) * row_size

    shared[tid] = a_ptr[base + tid] if tid < row_size else Scalar[dtype](0)
    barrier()

    block_scan_inclusive[TPB](shared, tid)

    if tid < row_size:
        out_ptr[base + tid] = shared[tid - 1] if tid > 0 else Scalar[dtype](0)
