"""Smoke test: Puzzle 1's map kernel with Puzzle 3's guard.

It confirms that DeviceContext works on this machine and that the `if i < size`
guard holds up for a size that is not a multiple of the block size (33 with
TPB=32). It is the project's first kernel and the one to look at when something
odd happens: if this one fails, the problem is the environment, not the code.
"""

from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from ops.common import dtype, GlobalF32
from tests.helpers import upload, zeros, download, assert_close

comptime TPB = 32
comptime TOL = Scalar[dtype](1e-6)


def add_ten(out_ptr: GlobalF32, a_ptr: GlobalF32, size: Int):
    # The guard is the point of the test: the number of threads launched is
    # rounded up to the block size, so it almost never matches the size of the
    # data. Without the `if`, the extra threads write out of bounds.
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
        check(ctx, 4)    # fits in one block with room to spare
        check(ctx, 33)   # ragged: 2 blocks, 31 extra threads in the second
