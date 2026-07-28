"""Adam y el clip por norma global. El paso de "corregir" del aprendizaje.

El backward dice EN QUE DIRECCION mover cada peso; esto decide CUANTO. Sin un
optimizador, un gradiente grande daria un salto enorme y uno pequeno no movería
nada, y el entrenamiento seria inestable.

Adam mantiene dos medias moviles por peso:

    m  la direccion reciente          (media del gradiente)
    v  cuanto se mueve ese peso       (media del gradiente al cuadrado)

y avanza `lr * m / (sqrt(v) + eps)`. La division es lo que lo hace robusto: un
peso cuyo gradiente oscila mucho (v grande) se mueve poco, y uno con gradiente
pequeno pero constante avanza igual. Las dos medias arrancan en cero, asi que al
principio subestiman; por eso se corrigen dividiendo por (1 - beta^t), y esa
correccion depende del NUMERO DE PASO.

Antes de Adam va el clip por norma global, exactamente como en Stoix:

    optax.chain(clip_by_global_norm(max_norm), adam(lr, eps=1e-5))

"Global" quiere decir que la norma se calcula sobre TODOS los tensores juntos, no
tensor a tensor. Es el error tipico al reimplementarlo: recortar cada tensor por
separado cambia la DIRECCION del gradiente conjunto, y esto solo cambia su
longitud.

Ojo con eps: Stoix pone 1e-5 explicitamente, no el 1e-8 de optax por defecto.
"""

from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt
from std.memory import stack_allocation, AddressSpace

from ops.buffers import zero_buffer
from ops.common import dtype, GlobalF32

comptime TPB_OPT = 256
"""Hilos por bloque de los kernels del optimizador."""

comptime BETA1 = Scalar[dtype](0.9)
comptime BETA2 = Scalar[dtype](0.999)
comptime ADAM_EPS = Scalar[dtype](1e-5)
"""Los valores de Stoix. b1/b2 son los defaults de optax; eps NO (optax trae
1e-8, Stoix lo sube a 1e-5)."""


def sum_squares_kernel[TPB: Int](partials: GlobalF32, x: GlobalF32, n: Int):
    """Suma de cuadrados por bloque: `partials[bloque] = suma de x[i]^2`.

    Reduccion en arbol en shared memory (Puzzle 12). Cada bloque deja un parcial
    y el host los suma: son unos pocos floats de trafico y evita tener que
    coordinar bloques, que en GPU exige otro kernel (Puzzle 14).
    """
    shared = stack_allocation[TPB, Scalar[dtype],
                              address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    i = Int(block_dim.x * block_idx.x + tid)

    v = x[i] if i < n else Scalar[dtype](0)
    shared[tid] = v * v
    barrier()

    stride = TPB // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        barrier()
        stride //= 2

    if tid == 0:
        partials[Int(block_idx.x)] = shared[0]


def adam_step_kernel(param: GlobalF32, grad: GlobalF32, m: GlobalF32,
                     v: GlobalF32, n: Int, lr: Scalar[dtype],
                     grad_scale: Scalar[dtype], bc1: Scalar[dtype],
                     bc2: Scalar[dtype]):
    """Un paso de Adam, elementwise.

    `grad_scale` es el factor del clip global (1.0 si no recorta), aplicado aqui
    para no tener que reescribir los gradientes en otra pasada.

    `bc1` y `bc2` son 1/(1 - beta^t), las correcciones de sesgo, calculadas en el
    host porque dependen del paso y no del elemento.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= n:
        return

    g = grad[i] * grad_scale
    mi = BETA1 * m[i] + (Scalar[dtype](1) - BETA1) * g
    vi = BETA2 * v[i] + (Scalar[dtype](1) - BETA2) * g * g
    m[i] = mi
    v[i] = vi

    m_hat = mi * bc1
    v_hat = vi * bc2
    param[i] = param[i] - lr * m_hat / (sqrt(v_hat) + ADAM_EPS)


struct AdamState(Movable):
    """Los dos momentos de un tensor de pesos. Arrancan a cero, como en optax."""

    var m: DeviceBuffer[dtype]
    var v: DeviceBuffer[dtype]
    var size: Int

    def __init__(out self, ctx: DeviceContext, size: Int) raises:
        self.m = zero_buffer[dtype](ctx, size)
        self.v = zero_buffer[dtype](ctx, size)
        self.size = size


def sum_squares(ctx: DeviceContext, x: DeviceBuffer[dtype],
                n: Int) raises -> Scalar[dtype]:
    """Suma de cuadrados de un buffer, cerrando la reduccion en el host."""
    blocks = (n + TPB_OPT - 1) // TPB_OPT
    partials = zero_buffer[dtype](ctx, blocks)
    ctx.enqueue_function[sum_squares_kernel[TPB_OPT],
                         sum_squares_kernel[TPB_OPT]](
        partials.unsafe_ptr(), x.unsafe_ptr(), n,
        grid_dim=blocks, block_dim=TPB_OPT)
    ctx.synchronize()

    total = Scalar[dtype](0)
    with partials.map_to_host() as h:
        for i in range(blocks):
            total += h[i]
    return total


def global_clip_scale(total_sq: Scalar[dtype],
                      max_norm: Scalar[dtype]) -> Scalar[dtype]:
    """El factor del clip: 1 si la norma cabe, max_norm/norma si se pasa.

    Se calcula a partir de la suma de cuadrados de TODOS los tensores, no de uno.
    """
    norm = sqrt(total_sq)
    if norm <= max_norm:
        return Scalar[dtype](1)
    return max_norm / norm


def adam_step(ctx: DeviceContext, param: DeviceBuffer[dtype],
              grad: DeviceBuffer[dtype], state: AdamState, n: Int,
              lr: Scalar[dtype], grad_scale: Scalar[dtype], step: Int) raises:
    """Actualiza un tensor. `step` empieza en 1 (como en optax)."""
    b1t = Scalar[dtype](1)
    b2t = Scalar[dtype](1)
    for _ in range(step):
        b1t *= BETA1
        b2t *= BETA2
    bc1 = Scalar[dtype](1) / (Scalar[dtype](1) - b1t)
    bc2 = Scalar[dtype](1) / (Scalar[dtype](1) - b2t)

    ctx.enqueue_function[adam_step_kernel, adam_step_kernel](
        param.unsafe_ptr(), grad.unsafe_ptr(), state.m.unsafe_ptr(),
        state.v.unsafe_ptr(), n, lr, grad_scale, bc1, bc2,
        grid_dim=(n + TPB_OPT - 1) // TPB_OPT, block_dim=TPB_OPT)
