"""Prefix sum contra el acumulado en host.

Ademas del caso general, dos comprobaciones con intencion:
  - el ultimo del inclusivo tiene que ser la suma de la fila (si no, el scan se
    ha dejado elementos por el camino),
  - el exclusivo da posiciones de escritura sin huecos ni solapes, que es para
    lo que se usa de verdad.
"""

from std.gpu.host import DeviceContext

from ops.common import dtype
from ops.scan import inclusive_scan_rows, exclusive_scan_rows
from tests.helpers import upload, zeros, download, assert_close

comptime TPB = 32
comptime TOL = Scalar[dtype](1e-4)


def make_data(rows: Int, row_size: Int) -> List[Scalar[dtype]]:
    """Valores en 1..7. Todos positivos a proposito: asi el acumulado es
    estrictamente creciente y un error de un puesto salta a la vista."""
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
            # exclusivo = lo acumulado ANTES de este elemento
            assert_close(got_exc[i], running, TOL,
                         String("exclusive fila=", r, " col=", c))
            running += data[i]
            # inclusivo = lo acumulado DESPUES
            assert_close(got_inc[i], running, TOL,
                         String("inclusive fila=", r, " col=", c))
        assert_close(got_inc[r * row_size + row_size - 1], running, TOL,
                     String("el ultimo del inclusivo no es la suma, fila=", r))
    print("PASS scans row_size", row_size)


def check_histogram_positions(ctx: DeviceContext) raises:
    """El uso clasico del exclusivo: conteos -> offsets de escritura.

    Con conteos [3, 0, 2, 1] los offsets tienen que ser [0, 3, 3, 5]: cada bucket
    sabe donde empieza su tramo y no pisa al vecino. El 0 del segundo bucket es
    lo interesante, porque deja dos offsets repetidos (3 y 3) -- correcto, ese
    bucket no escribe nada.
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
        assert_close(got[i], want[i], TOL, String("offset de histograma ", i))
    print("PASS exclusive scan -> offsets de histograma")


def main() raises:
    with DeviceContext() as ctx:
        # row_size=1 comprueba que el scan no se cae con una fila trivial, y
        # 17 que los hilos sobrantes (que cargan 0) no aportan nada.
        for row_size in [1, 4, 16, 17, 32]:
            check_scans(ctx, 4, row_size)
        check_histogram_positions(ctx)
