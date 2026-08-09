"""Tests for ops/reductions.mojo against host-side loops.

The sizes are not random:
  16   -> half a warp row, SPO's real case (16 particles)
  32   -> exactly one warp / one whole block
  33   -> the "ragged" case: longer than the block and not a multiple. This is
          where badly written guards fall over and where the TPB-strided loop has
          to go round with mismatched threads.
  1024 -> 32 elements per thread, so that the accumulation loop is noticeable.
"""

from std.gpu.host import DeviceContext
from std.gpu import WARP_SIZE

from ops.common import dtype, idx_dtype
from ops.reductions import sum_rows, warp_sum_rows, max_rows, argmax_rows
from tests.helpers import upload, zeros, download, assert_close, assert_eq_int

comptime TPB = 32

# The data are small integers, and float32 represents every integer up to 2^24
# exactly. That is, the reference sums and the GPU's coincide BIT for BIT even
# though the accumulation order differs: the tolerance is unnecessary, but I leave
# it in case I ever change the pattern to non-integer values.
comptime TOL = Scalar[dtype](1e-4)


def fill_pattern(rows: Int, row_size: Int) -> List[Scalar[dtype]]:
    """A deterministic, non-monotonic pattern.

    The `% 31` makes the values repeat every 31 columns, so with row_size=1024
    there are plenty of ties at the maximum: the argmax test becomes a test of the
    tie-breaking rule as well, for free.
    """
    data = List[Scalar[dtype]]()
    for r in range(rows):
        for c in range(row_size):
            data.append(Scalar[dtype]((r * 7 + c * 13) % 31) - 15.0)
    return data^


def host_sum(data: List[Scalar[dtype]], row: Int, row_size: Int) -> Scalar[dtype]:
    total = Scalar[dtype](0)
    for c in range(row_size):
        total += data[row * row_size + c]
    return total


def host_argmax(data: List[Scalar[dtype]], row: Int, row_size: Int) -> Int:
    """Host-side reference. Strict comparison -> the lower index stays, the same
    rule argmax_rows promises."""
    best = data[row * row_size]
    best_i = 0
    for c in range(row_size):
        if data[row * row_size + c] > best:
            best = data[row * row_size + c]
            best_i = c
    return best_i


def check_sum(ctx: DeviceContext, rows: Int, row_size: Int) raises:
    data = fill_pattern(rows, row_size)
    a = upload[dtype](ctx, data)
    o = zeros[dtype](ctx, rows)

    ctx.enqueue_function[sum_rows[TPB], sum_rows[TPB]](
        o.unsafe_ptr(), a.unsafe_ptr(), row_size, grid_dim=rows, block_dim=TPB)
    ctx.synchronize()

    got = download[dtype](o, rows)
    for r in range(rows):
        assert_close(got[r], host_sum(data, r, row_size), TOL,
                     String("sum_rows row_size=", row_size, " row=", r))
    print("PASS sum_rows row_size", row_size)


def check_warp_sum(ctx: DeviceContext, rows: Int, row_size: Int) raises:
    """The warp butterfly has to give exactly the same as the tree version with
    shared memory. They are two different routes to the same number."""
    data = fill_pattern(rows, row_size)
    a = upload[dtype](ctx, data)
    o = zeros[dtype](ctx, rows)

    ctx.enqueue_function[warp_sum_rows, warp_sum_rows](
        o.unsafe_ptr(), a.unsafe_ptr(), row_size,
        grid_dim=rows, block_dim=WARP_SIZE)
    ctx.synchronize()

    got = download[dtype](o, rows)
    for r in range(rows):
        assert_close(got[r], host_sum(data, r, row_size), TOL,
                     String("warp_sum_rows row_size=", row_size, " row=", r))
    print("PASS warp_sum_rows row_size", row_size)


def check_max(ctx: DeviceContext, rows: Int, row_size: Int) raises:
    data = fill_pattern(rows, row_size)
    a = upload[dtype](ctx, data)
    o = zeros[dtype](ctx, rows)

    ctx.enqueue_function[max_rows[TPB], max_rows[TPB]](
        o.unsafe_ptr(), a.unsafe_ptr(), row_size, grid_dim=rows, block_dim=TPB)
    ctx.synchronize()

    got = download[dtype](o, rows)
    for r in range(rows):
        want = data[r * row_size + host_argmax(data, r, row_size)]
        assert_close(got[r], want, TOL,
                     String("max_rows row_size=", row_size, " row=", r))
    print("PASS max_rows row_size", row_size)


def check_argmax(ctx: DeviceContext, rows: Int, row_size: Int) raises:
    data = fill_pattern(rows, row_size)
    a = upload[dtype](ctx, data)
    o = zeros[idx_dtype](ctx, rows)

    ctx.enqueue_function[argmax_rows[TPB], argmax_rows[TPB]](
        o.unsafe_ptr(), a.unsafe_ptr(), row_size, grid_dim=rows, block_dim=TPB)
    ctx.synchronize()

    got = download[idx_dtype](o, rows)
    for r in range(rows):
        assert_eq_int(Int(got[r]), host_argmax(data, r, row_size),
                      String("argmax_rows row_size=", row_size, " row=", r))
    print("PASS argmax_rows row_size", row_size)


def check_argmax_all_tied(ctx: DeviceContext) raises:
    """Extreme case: the whole row has the same value. Index 0 has to come out.

    With 33 elements and blocks of 32, thread 0 sees columns 0 and 32 (both tied),
    so it also checks the tie-breaking inside the strided loop, not only the tree's.
    """
    row_size = 33
    data = List[Scalar[dtype]]()
    for _ in range(row_size):
        data.append(Scalar[dtype](5.0))

    a = upload[dtype](ctx, data)
    o = zeros[idx_dtype](ctx, 1)

    ctx.enqueue_function[argmax_rows[TPB], argmax_rows[TPB]](
        o.unsafe_ptr(), a.unsafe_ptr(), row_size, grid_dim=1, block_dim=TPB)
    ctx.synchronize()

    got = download[idx_dtype](o, 1)
    assert_eq_int(Int(got[0]), 0, "argmax with the whole row tied")
    print("PASS argmax_rows ties -> lowest index")


def main() raises:
    with DeviceContext() as ctx:
        rows = 4
        for row_size in [16, 32, 33, 1024]:
            check_sum(ctx, rows, row_size)
            check_max(ctx, rows, row_size)
            check_argmax(ctx, rows, row_size)

        # The butterfly is only valid if the row fits in a warp.
        for row_size in [16, 32]:
            check_warp_sum(ctx, rows, row_size)

        check_argmax_all_tied(ctx)
