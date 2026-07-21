"""Copia plana device-a-device, parametrizada por dtype.

Antes habia dos kernels identicos, `copy_actions_kernel` (int32) y
`copy_f32_kernel` (float32), porque el tipo del puntero forma parte de la firma.
Parametrizando por dtype es un solo kernel: `copy_kernel[idx_dtype]` para
acciones y `copy_kernel[dtype]` para valores.

En la busqueda hace falta mas de lo que parece, porque hay campos que viven en
dos sitios a la vez: `root_actions` guarda la accion de profundidad 0 para
siempre mientras `next_action` se pisa en cada profundidad, y la salida publica
se lleva su propia copia de la gae y de las acciones raiz.
"""

from std.gpu import block_dim, block_idx, thread_idx


def copy_kernel[dt: DType](dst: UnsafePointer[Scalar[dt], MutAnyOrigin],
                           src: UnsafePointer[Scalar[dt], MutAnyOrigin],
                           n: Int):
    """dst[i] = src[i]. Un hilo por elemento, con su guard."""
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n:
        dst[i] = src[i]
