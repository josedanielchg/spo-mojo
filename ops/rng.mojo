"""RNG contador y muestreo categorico por CDF inversa.

Sin estado global: el valor sale de hash(seed, stream, counter), como las keys
de JAX. Dos ventajas para SPO:

  - Reproducible pase lo que pase con el scheduler. El hilo i usa counter = i,
    asi que no hay una secuencia compartida cuyo resultado dependa del orden en
    que se ejecuten los hilos. Con un RNG con estado, dos corridas con la misma
    seed darian numeros distintos a cada hilo.
  - Se "splitea" en streams independientes por uso (uno para la accion raiz,
    otro para el resampling, otro para el reset del entorno) sin que se pisen.

El mixer es el finalizador lowbias32: dos rondas de xor-shift + multiplicacion.
No es criptografico ni falta que hace; solo tiene que pasar los tests
estadisticos del muestreo (media, varianza y chi2 estan en test_rng.mojo).

Nota de diseno importante: los kernels que muestrean reciben los uniformes YA
generados como input, en vez de llamar al RNG por dentro. Asi el test puede
inyectar uniformes a mano y comprobar indices EXACTOS, sin depender de
tolerancias estadisticas.
"""

from std.builtin.debug_assert import debug_assert
from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.math import exp
from std.memory import stack_allocation, AddressSpace

from ops.common import dtype, idx_dtype, NEG_INF, GlobalF32, GlobalI32
from ops.reductions import block_reduce_max
from ops.scan import block_scan_inclusive

# 1 / 2^24. Me quedo con los 24 bits altos porque son justo los que caben en la
# mantisa de un float32: asi cada valor representable en [0,1) sale con la misma
# probabilidad. Si usara los 32 bits, la conversion a float redondearia y algunos
# valores saldrian el doble de veces que otros.
comptime INV_2P24 = Scalar[dtype](5.9604645e-8)

# Proporcion aurea en 32 bits. Es la constante de dispersion de rigor (la misma
# que usa Fibonacci hashing): mezcla bien los bits altos y bajos.
comptime GOLDEN32 = UInt32(0x9E3779B9)


def mix32(x_in: UInt32) -> UInt32:
    """Finalizador lowbias32: mezcla los 32 bits para que un contador 0,1,2,...
    se convierta en algo que parezca aleatorio."""
    x = x_in
    x ^= x >> 16
    x *= 0x7FEB352D
    x ^= x >> 15
    x *= 0x846CA68B
    x ^= x >> 16
    return x


def rand_bits(seed: UInt32, stream: UInt32, counter: UInt32) -> UInt32:
    """Los tres se mezclan en cascada para que cambiar cualquiera de ellos
    cambie el resultado entero, no solo unos pocos bits."""
    return mix32(seed ^ mix32(stream ^ mix32(counter)))


def rand_uniform(seed: UInt32, stream: UInt32, counter: UInt32) -> Scalar[dtype]:
    """Uniforme en [0, 1). El 1.0 nunca sale, que es lo que quiere la CDF inversa."""
    return Scalar[dtype]((rand_bits(seed, stream, counter) >> 8)) * INV_2P24


@fieldwise_init
struct RngKey(Copyable, Movable):
    """Contabilidad de streams al estilo key de JAX, para el lado host.

    Los kernels reciben (seed, stream) por separado en vez de un RngKey: pasar
    structs como argumento de kernel pide que sean DevicePassable y no aporta
    nada aqui. Esto es el libro de cuentas de quien usa que stream.
    """

    var seed: UInt32
    var stream: UInt32

    def split(self, which: UInt32) -> RngKey:
        """Deriva un stream hijo: mismo seed, stream distinto e independiente."""
        return RngKey(self.seed, mix32(self.stream ^ (which + GOLDEN32)))


def fill_uniform(out_ptr: GlobalF32, seed: UInt32, stream: UInt32, n: Int):
    """Rellena out[0..n) con uniformes. Un hilo por valor, con su guard."""
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n:
        # counter = i (el indice global), no un contador compartido: de ahi
        # viene la reproducibilidad.
        out_ptr[i] = rand_uniform(seed, stream, UInt32(i))


def categorical_from_logits[TPB: Int](out_ptr: GlobalI32, logits_ptr: GlobalF32,
                                      u_ptr: GlobalF32, row_size: Int):
    """Una muestra categorica por fila, por CDF inversa.

    Receta: softmax estable -> prefix sum (= CDF) -> primer indice cuyo tramo
    contiene al uniforme. Es la composicion de las tres piezas de la fase 2.

    Detalle que ahorra trabajo: el CDF se deja SIN normalizar y en vez de dividir
    cada elemento por el total se escala el uniforme (u * total). Sale una
    division por bloque en lugar de una por hilo, y de paso desaparece el caso
    total == 0.

    Pide row_size <= TPB. Lanzar con grid_dim=num_filas, block_dim=TPB.
    """
    debug_assert(row_size <= TPB, "categorical: row_size tiene que caber en TPB")
    debug_assert(Int(block_dim.x) == TPB, "categorical: block_dim tiene que ser TPB")

    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    row = Int(block_idx.x)
    base = row * row_size

    # Primero el maximo de la fila, que es lo que hace estable al softmax.
    shared[tid] = logits_ptr[base + tid] if tid < row_size else NEG_INF
    barrier()
    row_max = block_reduce_max[TPB](shared, tid)

    # Ahora exp(x - max), sin normalizar. Los hilos sobrantes ponen 0 para no
    # aportar masa de probabilidad.
    shared[tid] = exp(logits_ptr[base + tid] - row_max) if tid < row_size else Scalar[dtype](0)
    barrier()

    # El prefix sum convierte eso en un CDF, tambien sin normalizar.
    block_scan_inclusive[TPB](shared, tid)

    total = shared[row_size - 1]
    target = u_ptr[row] * total

    # Y cada hilo mira si el target cae en su tramo [cdf_prev, cdf). Exactamente
    # uno acierta, asi que no hay carrera al escribir out[row]. Un elemento con
    # probabilidad 0 tiene cdf_prev == cdf, o sea un tramo vacio, y por tanto no
    # puede salir elegido, que es justo lo que queremos.
    if tid < row_size:
        cdf = shared[tid]
        cdf_prev = shared[tid - 1] if tid > 0 else Scalar[dtype](0)
        hit = cdf_prev <= target and target < cdf

        # Red de seguridad por redondeo: el CDF se acumula sumando floats, asi
        # que el ultimo tramo puede quedarse un ulp por debajo de total y dejar
        # al target fuera de todos los tramos. Sin esto la fila no escribiria
        # nada y out[row] se quedaria con basura.
        if tid == row_size - 1 and target >= cdf:
            hit = True

        if hit:
            out_ptr[row] = Scalar[idx_dtype](tid)
