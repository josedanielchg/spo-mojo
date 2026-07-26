"""El MLP del critico: una pila de capas lineales con ReLU en medio.

    x  [M, in_dim]  ──linear──►  ──relu──►  ──linear──►  ──relu──►  ──linear──►  V  [M, out_dim]
                        h1                      h2                     (sin relu:
                                                                        V puede ser
                                                                        negativo)

Es generico en las dimensiones a proposito: `networks/` no importa nada de `envs/`,
igual que la busqueda no importa el entorno. Quien lo construya (el learner) decide
que tamano tiene; para tres en raya seran 18 -> 64 -> 64 -> 1.

Las activaciones se guardan en un `CriticCache` en vez de tirarlas. Todavia no hace
falta para el forward, pero el backward manual las necesita (la tecnica del Puzzle
22: cachear el forward para no recomputarlo al derivar), y reservarlas aqui evita
tener que cambiar la firma despues.

Un detalle que ahorra un buffer: la mascara del ReLU se puede derivar de la
activacion YA aplicada (post-relu > 0 si y solo si pre-relu > 0), asi que el relu
se hace in-place y no hace falta guardar el valor previo.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from ops.buffers import zero_buffer
from ops.common import dtype, GlobalF32
from networks.linear import linear_forward

comptime TPB_NET = 256
"""Hilos por bloque para los kernels elementwise de las redes."""


def relu_kernel(a: GlobalF32, n: Int):
    """ReLU in-place: a[i] = max(a[i], 0). Un hilo por elemento."""
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n:
        if a[i] < Scalar[dtype](0):
            a[i] = Scalar[dtype](0)


def relu(ctx: DeviceContext, a: DeviceBuffer[dtype], n: Int) raises:
    """Encola el ReLU sobre los n primeros elementos del buffer."""
    ctx.enqueue_function[relu_kernel, relu_kernel](
        a.unsafe_ptr(), n,
        grid_dim=(n + TPB_NET - 1) // TPB_NET, block_dim=TPB_NET)


@fieldwise_init
struct CriticParams(Movable):
    """Los pesos del critico. El struct vive en el host; los buffers, en la GPU."""

    var w1: DeviceBuffer[dtype]
    """[in_dim, hidden]"""
    var b1: DeviceBuffer[dtype]
    """[hidden]"""
    var w2: DeviceBuffer[dtype]
    """[hidden, hidden]"""
    var b2: DeviceBuffer[dtype]
    """[hidden]"""
    var w3: DeviceBuffer[dtype]
    """[hidden, out_dim]"""
    var b3: DeviceBuffer[dtype]
    """[out_dim]"""

    var in_dim: Int
    var hidden: Int
    var out_dim: Int


def zero_critic_params(ctx: DeviceContext, in_dim: Int, hidden: Int,
                       out_dim: Int) raises -> CriticParams:
    """Pesos a cero, para rellenarlos despues (desde un golden o desde un init)."""
    return CriticParams(
        w1=zero_buffer[dtype](ctx, in_dim * hidden),
        b1=zero_buffer[dtype](ctx, hidden),
        w2=zero_buffer[dtype](ctx, hidden * hidden),
        b2=zero_buffer[dtype](ctx, hidden),
        w3=zero_buffer[dtype](ctx, hidden * out_dim),
        b3=zero_buffer[dtype](ctx, out_dim),
        in_dim=in_dim, hidden=hidden, out_dim=out_dim)


struct CriticCache(Movable):
    """Las activaciones de un forward, que el backward volvera a leer.

    Se reservan para el batch mas grande que se vaya a usar; un forward con menos
    filas simplemente usa el principio de cada buffer.
    """

    var a1: DeviceBuffer[dtype]
    """[M, hidden] despues del ReLU de la primera capa."""
    var a2: DeviceBuffer[dtype]
    """[M, hidden] despues del ReLU de la segunda."""
    var value: DeviceBuffer[dtype]
    """[M, out_dim] la salida, sin ReLU."""

    def __init__(out self, ctx: DeviceContext, max_batch: Int, hidden: Int,
                 out_dim: Int) raises:
        self.a1 = zero_buffer[dtype](ctx, max_batch * hidden)
        self.a2 = zero_buffer[dtype](ctx, max_batch * hidden)
        self.value = zero_buffer[dtype](ctx, max_batch * out_dim)


def critic_forward(ctx: DeviceContext, params: CriticParams,
                   cache: CriticCache, x: DeviceBuffer[dtype], m: Int) raises:
    """V(s) para m entradas. Deja las activaciones en `cache`.

    Tres capas encoladas en el mismo stream, asi que se ejecutan en orden sin
    necesidad de sincronizar entre ellas (la leccion del Puzzle 14).
    """
    linear_forward(ctx, cache.a1, x, params.w1, params.b1,
                   m, params.in_dim, params.hidden)
    relu(ctx, cache.a1, m * params.hidden)

    linear_forward(ctx, cache.a2, cache.a1, params.w2, params.b2,
                   m, params.hidden, params.hidden)
    relu(ctx, cache.a2, m * params.hidden)

    # La ultima capa NO lleva ReLU: V es un valor con signo, y recortarlo a 0
    # impediria representar posiciones malas.
    linear_forward(ctx, cache.value, cache.a2, params.w3, params.b3,
                   m, params.hidden, params.out_dim)
