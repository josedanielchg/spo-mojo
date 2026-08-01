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
from networks.linear import linear_forward, linear_backward

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


# ---------------------------------------------------------------------------
# El backward del critico. Encadena el de la capa lineal (que ya esta verificado
# contra autodiff) a traves de las dos mascaras de ReLU:
#
#     L  ──►  dV  ──►  capa3  ──►  da2  ──►  relu'  ──►  capa2  ──►  da1
#                                                            ──►  relu'  ──►  capa1
#
# El gradiente del ReLU es lo unico nuevo aqui, y es facil: la derivada de
# max(x,0) vale 1 donde x era positivo y 0 donde no. Como el ReLU se aplico
# in-place en el forward, se usa la activacion GUARDADA para saberlo
# (post-relu > 0 si y solo si pre-relu > 0), sin necesidad de haber guardado el
# valor previo.
# ---------------------------------------------------------------------------


def value_loss_grad_kernel(dv: GlobalF32, value: GlobalF32, target: GlobalF32,
                           n: Int, scale: Scalar[dtype]):
    """dL/dV para L = media de 0.5*(V - target)^2, o sea (V - target)/n.

    Es la misma perdida que usa Stoix para el critico (`rlax.l2_loss` seguido de
    `.mean()`); el 0.5 esta para que la derivada salga limpia, sin un 2 delante.
    `scale` es 1/n, calculado en el host para no dividir en cada hilo.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n:
        dv[i] = (value[i] - target[i]) * scale


def relu_backward_kernel(grad: GlobalF32, activation: GlobalF32, n: Int):
    """Pasa el gradiente por el ReLU, in-place: grad *= (activacion > 0).

    Donde el ReLU recorto, la salida no dependia de la entrada, asi que por ahi no
    pasa gradiente. Ojo con el borde: en exactamente 0 la derivada no existe y se
    toma 0, que es la convencion habitual (y coincide con el `<` del forward).
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n:
        if activation[i] <= Scalar[dtype](0):
            grad[i] = Scalar[dtype](0)


struct CriticGrads(Movable):
    """Los gradientes de los pesos, con la misma forma que `CriticParams`."""

    var dw1: DeviceBuffer[dtype]
    var db1: DeviceBuffer[dtype]
    var dw2: DeviceBuffer[dtype]
    var db2: DeviceBuffer[dtype]
    var dw3: DeviceBuffer[dtype]
    var db3: DeviceBuffer[dtype]

    def __init__(out self, ctx: DeviceContext, in_dim: Int, hidden: Int,
                 out_dim: Int) raises:
        self.dw1 = zero_buffer[dtype](ctx, in_dim * hidden)
        self.db1 = zero_buffer[dtype](ctx, hidden)
        self.dw2 = zero_buffer[dtype](ctx, hidden * hidden)
        self.db2 = zero_buffer[dtype](ctx, hidden)
        self.dw3 = zero_buffer[dtype](ctx, hidden * out_dim)
        self.db3 = zero_buffer[dtype](ctx, out_dim)


struct CriticScratch(Movable):
    """Los gradientes intermedios que viajan hacia atras entre capas.

    Son buffers de trabajo, no resultados: `dvalue` es lo que entra por arriba y
    `da2`/`da1` lo que va bajando. `dx` se calcula pero nadie lo usa (la entrada
    no es un parametro entrenable); se reserva porque el backward de la capa
    siempre lo produce y tener dos caminos distintos costaria mas que un buffer.
    """

    var dvalue: DeviceBuffer[dtype]
    var da2: DeviceBuffer[dtype]
    var da1: DeviceBuffer[dtype]
    var dx: DeviceBuffer[dtype]

    def __init__(out self, ctx: DeviceContext, max_batch: Int, in_dim: Int,
                 hidden: Int, out_dim: Int) raises:
        self.dvalue = zero_buffer[dtype](ctx, max_batch * out_dim)
        self.da2 = zero_buffer[dtype](ctx, max_batch * hidden)
        self.da1 = zero_buffer[dtype](ctx, max_batch * hidden)
        self.dx = zero_buffer[dtype](ctx, max_batch * in_dim)


def critic_backward(ctx: DeviceContext, params: CriticParams,
                    cache: CriticCache, grads: CriticGrads,
                    scratch: CriticScratch, x: DeviceBuffer[dtype],
                    target: DeviceBuffer[dtype], m: Int) raises:
    """Gradientes de los 6 tensores de pesos para la perdida L2 contra `target`.

    Da por hecho que `cache` viene de un `critic_forward` con la MISMA `x` y los
    mismos pesos: el backward reutiliza esas activaciones en vez de recomputarlas
    (Puzzle 22). Si se llamara con un cache viejo, los gradientes saldrian mal sin
    que nada fallara.
    """
    n_out = m * params.out_dim

    # 1. Por donde entra todo: la derivada de la perdida respecto a V.
    ctx.enqueue_function[value_loss_grad_kernel, value_loss_grad_kernel](
        scratch.dvalue.unsafe_ptr(), cache.value.unsafe_ptr(),
        target.unsafe_ptr(), n_out, Scalar[dtype](1) / Scalar[dtype](n_out),
        grid_dim=(n_out + TPB_NET - 1) // TPB_NET, block_dim=TPB_NET)

    # 2..6: el resto no depende de QUE perdida sea, solo del gradiente que baja.
    backward_from_dvalue(ctx, params, cache, grads, scratch, x, m)


def backward_from_dvalue(ctx: DeviceContext, params: CriticParams,
                         cache: CriticCache, grads: CriticGrads,
                         scratch: CriticScratch, x: DeviceBuffer[dtype],
                         m: Int) raises:
    """El backward de la red a partir de `scratch.dvalue`, ya calculado.

    Se separo de `critic_backward` cuando llego el actor: la red es la misma y el
    camino hacia atras tambien, lo unico que cambia es POR DONDE ENTRA el
    gradiente. El critico entra con d(L2)/dV = (V - target)/n y el actor con
    dL/dlogits = (pi - q)/n. Todo lo de aqui abajo es identico.

    Sigue dando por hecho que `cache` viene de un forward con la MISMA x y los
    mismos pesos.
    """
    n_hidden = m * params.hidden

    # 2. Tercera capa: su entrada fue a2.
    linear_backward(ctx, grads.dw3, grads.db3, scratch.da2,
                    cache.a2, params.w3, scratch.dvalue,
                    m, params.hidden, params.out_dim)

    # 3. El ReLU de la segunda capa, sobre el gradiente que acaba de bajar.
    ctx.enqueue_function[relu_backward_kernel, relu_backward_kernel](
        scratch.da2.unsafe_ptr(), cache.a2.unsafe_ptr(), n_hidden,
        grid_dim=(n_hidden + TPB_NET - 1) // TPB_NET, block_dim=TPB_NET)

    # 4. Segunda capa: su entrada fue a1.
    linear_backward(ctx, grads.dw2, grads.db2, scratch.da1,
                    cache.a1, params.w2, scratch.da2,
                    m, params.hidden, params.hidden)

    # 5. El ReLU de la primera.
    ctx.enqueue_function[relu_backward_kernel, relu_backward_kernel](
        scratch.da1.unsafe_ptr(), cache.a1.unsafe_ptr(), n_hidden,
        grid_dim=(n_hidden + TPB_NET - 1) // TPB_NET, block_dim=TPB_NET)

    # 6. Primera capa: su entrada fue la observacion.
    linear_backward(ctx, grads.dw1, grads.db1, scratch.dx,
                    x, params.w1, scratch.da1,
                    m, params.in_dim, params.hidden)
