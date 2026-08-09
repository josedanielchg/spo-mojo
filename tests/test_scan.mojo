"""Prefix sum against the host-side accumulation.

Besides the general case, two checks with intent:
  - the last element of the inclusive scan has to be the row's sum (otherwise the
    scan has dropped elements along the way),
  - the exclusive one gives write positions with no gaps and no overlaps, which is
    what it is actually used for.
"""

from std.gpu.host import DeviceContext

from ops.common import dtype
from ops.scan import inclusive_scan_rows, exclusive_scan_rows
from tests.helpers import upload, zeros, download, assert_close

comptime TPB = 32
comptime TOL = Scalar[dtype](1e-4)


def make_data(rows: Int, row_size: Int) -> List[Scalar[dtype]]:
    """Values in 1..7. All positive on purpose: that way the running total is
    strictly increasing and an off-by-one jumps out."""
    data = List[Scalar[dtype]]()
    for r in range(rows):
        for c in range(row_size):
            data.append(Scalar[dtype]((r * 3 + c) % 7) + 1.0)
    return data^


def check_scans(ctx: DeviceContext, rows: Int, row_size: Int) raises:
    data = make_data(rows, row_size)
    a = upload[dtype](ctx, data)
    inc = zeros[dtype](ctx, rows * row_size)
    exc = zeros[dtype](ctx, rows * row_size)

    ctx.enqueue_function[inclusive_scan_rows[TPB], inclusive_scan_rows[TPB]](
        inc.unsafe_ptr(), a.unsafe_ptr(), row_size, grid_dim=rows, block_dim=TPB)
    ctx.enqueue_function[exclusive_scan_rows[TPB], exclusive_scan_rows[TPB]](
        exc.unsafe_ptr(), a.unsafe_ptr(), row_size, grid_dim=rows, block_dim=TPB)
    ctx.synchronize()

    got_inc = download[dtype](inc, rows * row_size)
    got_exc = download[dtype](exc, rows * row_size)

    for r in range(rows):
        running = Scalar[dtype](0)
        for c in range(row_size):
            i = r * row_size + c
            # the exclusive one is what was accumulated before this element
            assert_close(got_exc[i], running, TOL,
                         String("exclusive row=", r, " col=", c))
            running += data[i]
            # and the inclusive one what was accumulated afterwards
            assert_close(got_inc[i], running, TOL,
                         String("inclusive row=", r, " col=", c))
        assert_close(got_inc[r * row_size + row_size - 1], running, TOL,
                     String("the last of the inclusive is not the sum, row=", r))
    print("PASS scans row_size", row_size)


def check_histogram_positions(ctx: DeviceContext) raises:
    """The exclusive scan's classic use: counts -> write offsets.

    With counts [3, 0, 2, 1] the offsets have to be [0, 3, 3, 5]: each bucket knows
    where its stretch begins and does not overwrite its neighbour. The 0 in the
    second bucket is the interesting bit, because it leaves two repeated offsets
    (3 and 3) -- correct, that bucket writes nothing.
    """
    counts = List[Scalar[dtype]]()
    counts.append(3.0)
    counts.append(0.0)
    counts.append(2.0)
    counts.append(1.0)

    want = List[Scalar[dtype]]()
    want.append(0.0)
    want.append(3.0)
    want.append(3.0)
    want.append(5.0)

    n = len(counts)
    a = upload[dtype](ctx, counts)
    o = zeros[dtype](ctx, n)

    ctx.enqueue_function[exclusive_scan_rows[TPB], exclusive_scan_rows[TPB]](
        o.unsafe_ptr(), a.unsafe_ptr(), n, grid_dim=1, block_dim=TPB)
    ctx.synchronize()

    got = download[dtype](o, n)
    for i in range(n):
        assert_close(got[i], want[i], TOL, String("histogram offset ", i))
    print("PASS exclusive scan -> histogram offsets")


def main() raises:
    with DeviceContext() as ctx:
        # row_size=1 checks that the scan does not fall over on a trivial row, and
        # 17 that the leftover threads (which load 0) contribute nothing.
        for row_size in [1, 4, 16, 17, 32]:
            check_scans(ctx, 4, row_size)
        check_histogram_positions(ctx)
