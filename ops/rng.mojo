"""RNG contador y muestreo categorico por CDF inversa.

Sin estado global: el valor sale de hash(seed, stream, counter). Es la misma idea
que las keys de JAX y tiene dos ventajas para SPO:
  - reproducible aunque los hilos corran en cualquier orden (el hilo i usa
    counter = i, no hay una secuencia compartida que dependa del scheduling),
  - se puede "splitear" en streams independientes por uso (uno para la accion
    raiz, otro para el resampling...) sin que se pisen.

El mixer es el finalizador lowbias32 (dos rondas de xor-shift/multiply). No es
critptografico ni falta que hace: solo tiene que pasar los tests estadisticos
del muestreo.

Los kernels que muestrean reciben los uniformes YA generados como input. Asi el
test puede inyectar uniformes a mano y comprobar indices exactos.
"""

from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.math import exp
from std.memory import stack_allocation, AddressSpace

comptime dtype = DType.float32
comptime idx_dtype = DType.int32

# 1 / 2^24: los 24 bits altos son justo la mantisa de un float32, asi que
# el uniforme sale en [0,1) sin huecos ni repeticiones raras.
comptime INV_2P24 = Scalar[dtype](5.9604645e-8)


@fieldwise_init
struct RngKey(Copyable, Movable):
    """Estilo key de JAX: se pasa por valor y se derivan streams hijos."""

    var seed: UInt32
    var stream: UInt32

    def split(self, which: UInt32) -> RngKey:
        """Deriva un stream hijo. Mismo seed, stream distinto."""
        return RngKey(self.seed, mix32(self.stream ^ (which + 0x9E3779B9)))


def mix32(x_in: UInt32) -> UInt32:
    """Finalizador lowbias32."""
    x = x_in
    x ^= x >> 16
    x *= 0x7FEB352D
    x ^= x >> 15
    x *= 0x846CA68B
    x ^= x >> 16
    return x


def rand_bits(seed: UInt32, stream: UInt32, counter: UInt32) -> UInt32:
    return mix32(seed ^ mix32(stream ^ mix32(counter)))


def rand_uniform(seed: UInt32, stream: UInt32, counter: UInt32) -> Scalar[dtype]:
    """Uniforme en [0, 1)."""
    return Scalar[dtype]((rand_bits(seed, stream, counter) >> 8)) * INV_2P24


def fill_uniform(out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
                 seed: UInt32, stream: UInt32, n: Int):
    """El hilo i usa counter = i: el resultado no depende del orden de ejecucion."""
    i = block_dim.x * block_idx.x + thread_idx.x
    if i < n:
        out_ptr[i] = rand_uniform(seed, stream, UInt32(i))


def categorical_from_logits[TPB: Int](
        out_ptr: UnsafePointer[Scalar[idx_dtype], MutAnyOrigin],
        logits_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        u_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        row_size: Int):
    """Una muestra categorica por fila, por CDF inversa.

    softmax estable -> prefix sum -> primer indice con cdf > u.

    Trabaja con el CDF SIN normalizar y escala el uniforme por el total
    (u * total) en vez de dividir cada elemento: una division menos por hilo y
    se evita el caso total==0.

    Pide row_size <= TPB. Lanzar con block_dim=TPB, grid_dim=num_rows.
    """
    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = thread_idx.x
    row = block_idx.x
    base = row * row_size

    # --- max de la fila (para el softmax estable) ---
    NEG_INF = Scalar[dtype](-3.4028235e38)
    shared[tid] = logits_ptr[base + tid] if tid < row_size else NEG_INF
    barrier()
    stride = TPB // 2
    while stride > 0:
        if tid < stride:
            other = shared[tid + stride]
            if other > shared[tid]:
                shared[tid] = other
        barrier()
        stride //= 2
    row_max = shared[0]
    barrier()

    # --- exp(x - max), sin normalizar ---
    shared[tid] = exp(logits_ptr[base + tid] - row_max) if tid < row_size else Scalar[dtype](0)
    barrier()

    # --- prefix sum inclusivo -> CDF sin normalizar ---
    offset = 1
    while offset < TPB:
        val = shared[tid - offset] if tid >= offset else Scalar[dtype](0)
        barrier()
        shared[tid] += val
        barrier()
        offset *= 2

    total = shared[row_size - 1]
    target = u_ptr[row] * total

    # --- primer indice cuyo tramo [cdf_prev, cdf) contiene a target ---
    if tid < row_size:
        cdf = shared[tid]
        cdf_prev = shared[tid - 1] if tid > 0 else Scalar[dtype](0)
        hit = cdf_prev <= target and target < cdf
        # Red de seguridad: si el redondeo deja target justo en (o pasado) el
        # final, se lo queda el ultimo. Sin esto ninguna fila escribiria nada.
        if tid == row_size - 1 and target >= cdf:
            hit = True
        if hit:
            out_ptr[row] = Scalar[idx_dtype](tid)
