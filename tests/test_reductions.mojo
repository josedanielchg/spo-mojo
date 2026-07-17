"""Pruebas de ops/reductions.mojo contra bucles en host.

Tamanos 16/32/33/1024: el 33 es el caso "ragged" (mas larga que el bloque y no
multiplo), que es donde se rompen los guards mal escritos.
"""

from std.gpu.host import DeviceContext
from std.math import abs

from std.gpu import WARP_SIZE

from ops.reductions import sum_rows, warp_sum_rows, max_rows, argmax_rows, dtype, idx_dtype

comptime TPB = 32
comptime TOL = Scalar[dtype](1e-4)


def fill_pattern(ctx: DeviceContext, rows: Int, row_size: Int) raises -> List[Scalar[dtype]]:
    """Valores deterministas y no monotonos, para que el argmax no caiga siempre al final."""
    host = List[Scalar[dtype]]()
    for r in range(rows):
        for c in range(row_size):
            x = Scalar[dtype]((r * 7 + c * 13) % 31) - 15.0
            host.append(x)
    return host^


def check_sum(ctx: DeviceContext, rows: Int, row_size: Int) raises:
    data = fill_pattern(ctx, rows, row_size)

    a = ctx.enqueue_create_buffer[dtype](rows * row_size)
    a.enqueue_fill(0)
    with a.map_to_host() as h:
        for i in range(rows * row_size):
            h[i] = data[i]
    o = ctx.enqueue_create_buffer[dtype](rows)
    o.enqueue_fill(0)

    ctx.enqueue_function[sum_rows[TPB], sum_rows[TPB]](
        o.unsafe_ptr(), a.unsafe_ptr(), row_size, grid_dim=rows, block_dim=TPB)
    ctx.synchronize()

    with o.map_to_host() as h:
        for r in range(rows):
            expected = Scalar[dtype](0)
            for c in range(row_size):
                expected += data[r * row_size + c]
            if abs(h[r] - expected) > TOL:
                raise Error("sum_rows row_size=", row_size, " row=", r,
                            " got=", h[r], " expected=", expected)
    print("PASS sum_rows row_size", row_size)


def check_warp_sum(ctx: DeviceContext, rows: Int, row_size: Int) raises:
    """El warp_sum tiene que dar lo mismo que el sum_rows de shared memory."""
    data = fill_pattern(ctx, rows, row_size)

    a = ctx.enqueue_create_buffer[dtype](rows * row_size)
    a.enqueue_fill(0)
    with a.map_to_host() as h:
        for i in range(rows * row_size):
            h[i] = data[i]
    o = ctx.enqueue_create_buffer[dtype](rows)
    o.enqueue_fill(0)

    ctx.enqueue_function[warp_sum_rows, warp_sum_rows](
        o.unsafe_ptr(), a.unsafe_ptr(), row_size, grid_dim=rows, block_dim=WARP_SIZE)
    ctx.synchronize()

    with o.map_to_host() as h:
        for r in range(rows):
            expected = Scalar[dtype](0)
            for c in range(row_size):
                expected += data[r * row_size + c]
            if abs(h[r] - expected) > TOL:
                raise Error("warp_sum_rows row_size=", row_size, " row=", r,
                            " got=", h[r], " expected=", expected)
    print("PASS warp_sum_rows row_size", row_size)


def check_max(ctx: DeviceContext, rows: Int, row_size: Int) raises:
    data = fill_pattern(ctx, rows, row_size)

    a = ctx.enqueue_create_buffer[dtype](rows * row_size)
    a.enqueue_fill(0)
    with a.map_to_host() as h:
        for i in range(rows * row_size):
            h[i] = data[i]
    o = ctx.enqueue_create_buffer[dtype](rows)
    o.enqueue_fill(0)

    ctx.enqueue_function[max_rows[TPB], max_rows[TPB]](
        o.unsafe_ptr(), a.unsafe_ptr(), row_size, grid_dim=rows, block_dim=TPB)
    ctx.synchronize()

    with o.map_to_host() as h:
        for r in range(rows):
            expected = data[r * row_size]
            for c in range(row_size):
                if data[r * row_size + c] > expected:
                    expected = data[r * row_size + c]
            if abs(h[r] - expected) > TOL:
                raise Error("max_rows row_size=", row_size, " row=", r,
                            " got=", h[r], " expected=", expected)
    print("PASS max_rows row_size", row_size)


def check_argmax(ctx: DeviceContext, rows: Int, row_size: Int) raises:
    data = fill_pattern(ctx, rows, row_size)

    a = ctx.enqueue_create_buffer[dtype](rows * row_size)
    a.enqueue_fill(0)
    with a.map_to_host() as h:
        for i in range(rows * row_size):
            h[i] = data[i]
    o = ctx.enqueue_create_buffer[idx_dtype](rows)
    o.enqueue_fill(0)

    ctx.enqueue_function[argmax_rows[TPB], argmax_rows[TPB]](
        o.unsafe_ptr(), a.unsafe_ptr(), row_size, grid_dim=rows, block_dim=TPB)
    ctx.synchronize()

    with o.map_to_host() as h:
        for r in range(rows):
            best = data[r * row_size]
            best_i = 0
            for c in range(row_size):
                if data[r * row_size + c] > best:
                    best = data[r * row_size + c]
                    best_i = c
            if Int(h[r]) != best_i:
                raise Error("argmax_rows row_size=", row_size, " row=", r,
                            " got=", Int(h[r]), " expected=", best_i)
    print("PASS argmax_rows row_size", row_size)


def check_argmax_ties(ctx: DeviceContext) raises:
    """Todo empatado: tiene que ganar el indice 0."""
    row_size = 33
    a = ctx.enqueue_create_buffer[dtype](row_size)
    a.enqueue_fill(5)
    o = ctx.enqueue_create_buffer[idx_dtype](1)
    o.enqueue_fill(0)

    ctx.enqueue_function[argmax_rows[TPB], argmax_rows[TPB]](
        o.unsafe_ptr(), a.unsafe_ptr(), row_size, grid_dim=1, block_dim=TPB)
    ctx.synchronize()

    with o.map_to_host() as h:
        if Int(h[0]) != 0:
            raise Error("argmax con empates: esperaba 0, salio ", Int(h[0]))
    print("PASS argmax_rows empates -> indice menor")


def main() raises:
    with DeviceContext() as ctx:
        rows = 4
        for row_size in [16, 32, 33, 1024]:
            check_sum(ctx, rows, row_size)
            check_max(ctx, rows, row_size)
            check_argmax(ctx, rows, row_size)
        # El warp butterfly solo aplica cuando la fila cabe en un warp,
        # que es el caso de SPO (16 particulas).
        for row_size in [16, 32]:
            check_warp_sum(ctx, rows, row_size)
        check_argmax_ties(ctx)
