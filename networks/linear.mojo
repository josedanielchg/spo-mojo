"""La capa lineal: y = x @ W + b, con la matmul tileada del Puzzle 16.

Es el ladrillo de todas las redes del M-step (el critico y el actor son pilas de
estas). Por eso vale la pena hacerla bien una vez: cada capa de cada forward, y
mas adelante las dos matmuls de cada backward, pasan por aqui.

    x  [M, K]   M = cuantos tableros a la vez (particulas o envs)
                K = entradas de la capa
    W  [K, N]   N = salidas de la capa
    b  [N]
    y  [M, N]

Por que tileado y no la version ingenua: en la ingenua cada hilo lee su fila de x
y su columna de W desde memoria global, asi que cada elemento de x se vuelve a
leer N veces en todo el grid. Tileando, un bloque carga UNA vez su trozo de x y
de W en shared memory y los reutiliza TILE veces. Es el mismo cambio que el
Puzzle 16 mide como el paso de ~1.5% del pico a algo razonable: no cambia las
cuentas, cambia cuantas veces se cruza la memoria.

Los guards no son adorno: nuestras dimensiones reales (18 entradas, batch de
particulas) casi nunca son multiplos del tile, asi que los bordes se pisan
siempre. Se cargan ceros fuera de rango para que el acumulador no vea basura.
"""

from std.gpu import barrier, block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import stack_allocation, AddressSpace

from ops.common import dtype, GlobalF32

comptime TILE = 16
"""Lado del tile cuadrado. 16x16 = 256 hilos por bloque, que es un tamano comodo
y deja cada tile de shared memory en 1 KB (16*16*4 bytes)."""


def linear_forward_kernel[T: Int](y: GlobalF32, x: GlobalF32, w: GlobalF32,
                                  bias: GlobalF32, m: Int, k: Int, n: Int):
    """y = x @ W + b. Un hilo por elemento de salida, bloques de TxT.

    El bucle recorre la dimension K a saltos de T. En cada vuelta el bloque
    entero coopera para cargar un tile de x y otro de W en shared memory, y
    despues cada hilo acumula el producto parcial de su fila por su columna.

    Los dos `barrier()` estan FUERA de cualquier condicional, que es la regla del
    Puzzle 9: si un hilo no llega a un barrier al que llegan los demas, el bloque
    se cuelga para siempre. El primero espera a que el tile este cargado antes de
    usarlo; el segundo espera a que todos hayan terminado de usarlo antes de
    sobrescribirlo en la vuelta siguiente. Quitar el segundo es el bug clasico:
    funciona casi siempre y falla de vez en cuando.
    """
    x_tile = stack_allocation[T * T, Scalar[dtype],
                              address_space = AddressSpace.SHARED]()
    w_tile = stack_allocation[T * T, Scalar[dtype],
                              address_space = AddressSpace.SHARED]()

    ty = Int(thread_idx.y)
    tx = Int(thread_idx.x)
    row = Int(block_idx.y) * T + ty     # que fila de la salida
    col = Int(block_idx.x) * T + tx     # que columna de la salida

    acc = Scalar[dtype](0)
    num_tiles = (k + T - 1) // T

    for t in range(num_tiles):
        # Cargar el tile de x: la fila `row`, columnas [t*T, t*T+T).
        xc = t * T + tx
        x_tile[ty * T + tx] = x[row * k + xc] if (row < m and xc < k) \
                              else Scalar[dtype](0)
        # Y el de W: filas [t*T, t*T+T), la columna `col`.
        wr = t * T + ty
        w_tile[ty * T + tx] = w[wr * n + col] if (wr < k and col < n) \
                              else Scalar[dtype](0)
        barrier()

        # El producto parcial. Los ceros de relleno no aportan nada, asi que no
        # hace falta acortar el bucle en los bordes.
        for i in range(T):
            acc += x_tile[ty * T + i] * w_tile[i * T + tx]
        barrier()

    if row < m and col < n:
        y[row * n + col] = acc + bias[col]


def linear_forward(ctx: DeviceContext, y: DeviceBuffer[dtype],
                   x: DeviceBuffer[dtype], w: DeviceBuffer[dtype],
                   bias: DeviceBuffer[dtype], m: Int, k: Int, n: Int) raises:
    """Encola la capa: un bloque de TILExTILE por cada tile de la salida."""
    grid_x = (n + TILE - 1) // TILE
    grid_y = (m + TILE - 1) // TILE
    ctx.enqueue_function[linear_forward_kernel[TILE],
                         linear_forward_kernel[TILE]](
        y.unsafe_ptr(), x.unsafe_ptr(), w.unsafe_ptr(), bias.unsafe_ptr(),
        m, k, n, grid_dim=(grid_x, grid_y), block_dim=(TILE, TILE))
