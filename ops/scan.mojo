"""Prefix sum por filas, inclusivo y exclusivo.

En SPO usamos el inclusivo para construir la CDF del resampling. La CDF divide
los pesos de las partículas en intervalos. Generamos un número aleatorio y
elegimos la partícula cuyo intervalo contiene ese número.

Cada bloque GPU procesa una fila y cada hilo procesa una posición. Toda la fila
debe caber dentro del bloque, por eso row_size no puede ser mayor que TPB. En
SPO cada fila contiene las partículas de un entorno, normalmente 16, así que
caben dentro de un bloque de 32 o más hilos.

Primero se implementa block_scan_inclusive, que trabaja directamente sobre la
memoria compartida de un bloque. Después se implementan inclusive_scan_rows y
exclusive_scan_rows, que cargan las filas, llaman al primitivo y escriben los
resultados.
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
