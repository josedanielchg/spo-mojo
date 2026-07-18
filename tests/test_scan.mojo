"""Prefix sum contra el acumulado en host.

Ademas del caso general, dos comprobaciones con intencion:
 - el ultimo del inclusivo = la suma de la fila,
 - el exclusivo da posiciones de escritura sin huecos ni solapes (histograma).
"""

from std.gpu.host import DeviceContext
from std.math import abs

from ops.scan import inclusive_scan_rows, exclusive_scan_rows, dtype

comptime TPB = 32
comptime TOL = Scalar[dtype](1e-4)


def make_data(rows: Int, row_size: Int) -> List[Scalar[dtype]]:
    data = List[Scalar[dtype]]()
    for r in range(rows):
        for c in range(row_size):
            data.append(Scalar[dtype]((r * 3 + c) % 7) + 1.0)
    return data^


def check_scans(ctx: DeviceContext, rows: Int, row_size: Int) raises:
    data = make_data(rows, row_size)

    a = ctx.enqueue_create_buffer[dtype](rows * row_size)
    a.enqueue_fill(0)
    with a.map_to_host() as h:
        for i in range(rows * row_size):
            h[i] = data[i]

    inc = ctx.enqueue_create_buffer[dtype](rows * row_size)
    inc.enqueue_fill(0)
    exc = ctx.enqueue_create_buffer[dtype](rows * row_size)
    exc.enqueue_fill(0)

    ctx.enqueue_function[inclusive_scan_rows[TPB], inclusive_scan_rows[TPB]](
        inc.unsafe_ptr(), a.unsafe_ptr(), row_size, grid_dim=rows, block_dim=TPB)
    ctx.enqueue_function[exclusive_scan_rows[TPB], exclusive_scan_rows[TPB]](
        exc.unsafe_ptr(), a.unsafe_ptr(), row_size, grid_dim=rows, block_dim=TPB)
    ctx.synchronize()

    with inc.map_to_host() as hi:
        with exc.map_to_host() as he:
            for r in range(rows):
                run = Scalar[dtype](0)
                for c in range(row_size):
                    # exclusivo: el acumulado ANTES de sumar este elemento
                    if abs(he[r * row_size + c] - run) > TOL:
                        raise Error("exclusive row=", r, " col=", c,
                                    " got=", he[r * row_size + c], " want=", run)
                    run += data[r * row_size + c]
                    # inclusivo: el acumulado DESPUES
                    if abs(hi[r * row_size + c] - run) > TOL:
                        raise Error("inclusive row=", r, " col=", c,
                                    " got=", hi[r * row_size + c], " want=", run)
                # el ultimo del inclusivo tiene que ser la suma de la fila
                if abs(hi[r * row_size + row_size - 1] - run) > TOL:
                    raise Error("inclusive: el ultimo no es la suma, fila ", r)
    print("PASS scans row_size", row_size)


def check_histogram_positions(ctx: DeviceContext) raises:
    """El uso clasico del exclusivo: convertir conteos en offsets de escritura.

    Con conteos [3,0,2,1] los offsets tienen que ser [0,3,3,5] -> cada bucket
    escribe en su tramo sin pisar al vecino.
    """
    counts = List[Scalar[dtype]]()
    counts.append(3.0)
    counts.append(0.0)
    counts.append(2.0)
    counts.append(1.0)
    n = len(counts)

    a = ctx.enqueue_create_buffer[dtype](n)
    a.enqueue_fill(0)
    with a.map_to_host() as h:
        for i in range(n):
            h[i] = counts[i]
    o = ctx.enqueue_create_buffer[dtype](n)
    o.enqueue_fill(0)

    ctx.enqueue_function[exclusive_scan_rows[TPB], exclusive_scan_rows[TPB]](
        o.unsafe_ptr(), a.unsafe_ptr(), n, grid_dim=1, block_dim=TPB)
    ctx.synchronize()

    want = List[Scalar[dtype]]()
    want.append(0.0)
    want.append(3.0)
    want.append(3.0)
    want.append(5.0)
    with o.map_to_host() as h:
        for i in range(n):
            if abs(h[i] - want[i]) > TOL:
                raise Error("offsets de histograma en ", i, ": got=", h[i],
                            " want=", want[i])
    print("PASS exclusive scan -> offsets de histograma")


def main() raises:
    with DeviceContext() as ctx:
        for row_size in [1, 4, 16, 17, 32]:
            check_scans(ctx, 4, row_size)
        check_histogram_positions(ctx)
