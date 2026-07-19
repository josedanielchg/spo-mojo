"""Pruebas de ops/reductions.mojo contra bucles en host.

Los tamanos no son al azar:
  16   -> media fila de warp, el caso real de SPO (16 particulas)
  32   -> justo un warp / un bloque entero
  33   -> el caso "ragged": mas largo que el bloque y no multiplo. Aqui es donde
          se caen los guards mal escritos y donde el bucle a saltos de TPB tiene
          que dar la vuelta con hilos desparejados.
  1024 -> 32 elementos por hilo, para que el bucle de acumulacion se note.
"""

from std.gpu.host import DeviceContext
from std.gpu import WARP_SIZE

from ops.common import dtype, idx_dtype
from ops.reductions import sum_rows, warp_sum_rows, max_rows, argmax_rows
from tests.helpers import upload, zeros, download, assert_close, assert_eq_int

comptime TPB = 32

# Los datos son enteros pequenos, y float32 representa exacto todo entero hasta
# 2^24. O sea que las sumas de referencia y las de la GPU coinciden BIT a BIT
# aunque el orden de acumulacion sea distinto: la tolerancia sobra, pero la dejo
# por si algun dia cambio el patron a valores no enteros.
comptime TOL = Scalar[dtype](1e-4)


def fill_pattern(rows: Int, row_size: Int) -> List[Scalar[dtype]]:
    """Patron determinista y no monotono.

    El `% 31` hace que los valores se repitan cada 31 columnas, asi que con
    row_size=1024 hay un monton de empates en el maximo: el test de argmax pasa
    a ser tambien un test de la regla de desempate, gratis.
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
    """Referencia en host. Comparacion estricta -> se queda el indice menor,
    la misma regla que promete argmax_rows."""
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
                     String("sum_rows row_size=", row_size, " fila=", r))
    print("PASS sum_rows row_size", row_size)


def check_warp_sum(ctx: DeviceContext, rows: Int, row_size: Int) raises:
    """El butterfly de warp tiene que dar exactamente lo mismo que la version
    en arbol con shared memory. Son dos caminos distintos al mismo numero."""
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
                     String("warp_sum_rows row_size=", row_size, " fila=", r))
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
                     String("max_rows row_size=", row_size, " fila=", r))
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
                      String("argmax_rows row_size=", row_size, " fila=", r))
    print("PASS argmax_rows row_size", row_size)


def check_argmax_all_tied(ctx: DeviceContext) raises:
    """Caso extremo: la fila entera vale lo mismo. Tiene que salir el indice 0.

    Con 33 elementos y bloques de 32, el hilo 0 ve las columnas 0 y 32 (las dos
    empatadas), asi que ademas comprueba el desempate dentro del bucle a saltos,
    no solo el del arbol.
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
    assert_eq_int(Int(got[0]), 0, "argmax con toda la fila empatada")
    print("PASS argmax_rows empates -> indice menor")


def main() raises:
    with DeviceContext() as ctx:
        rows = 4
        for row_size in [16, 32, 33, 1024]:
            check_sum(ctx, rows, row_size)
            check_max(ctx, rows, row_size)
            check_argmax(ctx, rows, row_size)

        # El butterfly solo vale si la fila cabe en un warp.
        for row_size in [16, 32]:
            check_warp_sum(ctx, rows, row_size)

        check_argmax_all_tied(ctx)
