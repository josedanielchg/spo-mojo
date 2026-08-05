"""Smoke test: el kernel map del Puzzle 1 con el guard del Puzzle 3.

Confirma que DeviceContext funciona en esta maquina y que el guard `if i < size`
aguanta un tamano que no es multiplo del block size (33 con TPB=32). Es el
primer kernel del proyecto y el que se mira cuando algo raro pasa: si este falla,
el problema es del entorno, no del codigo.
"""

from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from ops.common import dtype, GlobalF32
from tests.helpers import upload, zeros, download, assert_close

comptime TPB = 32
comptime TOL = Scalar[dtype](1e-6)


def add_ten(out_ptr: GlobalF32, a_ptr: GlobalF32, size: Int):
    # El guard es el punto del test: el numero de hilos lanzados se redondea
    # hacia arriba al tamano del bloque, asi que casi nunca coincide con el
    # tamano de los datos. Sin el `if` los hilos de mas escriben fuera.
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < size:
        out_ptr[i] = a_ptr[i] + 10.0


def check(ctx: DeviceContext, size: Int) raises:
    data = List[Scalar[dtype]]()
    for i in range(size):
        data.append(Scalar[dtype](i))

    a = upload[dtype](ctx, data)
    o = zeros[dtype](ctx, size)

    blocks = (size + TPB - 1) // TPB
    ctx.enqueue_function[add_ten, add_ten](
        o.unsafe_ptr(), a.unsafe_ptr(), size, grid_dim=blocks, block_dim=TPB)
    ctx.synchronize()

    got = download[dtype](o, size)
    for i in range(size):
        assert_close(got[i], data[i] + 10.0, TOL, String("add_ten size=", size, " i=", i))
    print("PASS add_ten size", size, "(grid", blocks, "x block", TPB, ")")


def main() raises:
    with DeviceContext() as ctx:
        print("device:", ctx.name())
        check(ctx, 4)    # cabe de sobra en un bloque
        check(ctx, 33)   # ragged: 2 bloques, 31 hilos de mas en el segundo
