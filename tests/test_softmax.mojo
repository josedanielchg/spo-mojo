"""Softmax y logsumexp contra el golden de numpy.

El golden lo genera tests/golden/gen/gen_softmax.py. Las filas no son aleatorias
del todo: las cuatro primeras estan puestas a mano para reventar una
implementacion ingenua.
  fila 0: todo 0      -> softmax uniforme exacto, 1/COLS
  fila 1: todo +1000  -> exp(1000) = inf. Sin restar el max sale inf/inf = nan
  fila 2: un +1000    -> softmax casi one-hot
  fila 3: todo -1000  -> exp(-1000) = 0. Sin restar el max sale 0/0 = nan

Y ademas un bloque de filas ANCHAS (wide_*), con mas columnas que hilos por
bloque. Hasta E2.2 este fichero solo probaba COLS=16 con TPB=32, asi que el bucle
`i += TPB` con el que los kernels recorren la fila **daba siempre una sola vuelta
y la ruta de striding no se ejecutaba nunca**. WIDE_COLS=100 da tres vueltas
completas y una parcial, y la fila 1 pone su maximo en la ULTIMA columna: si el
striding no llegara al final, el maximo saldria mal y el softmax entero con el.
"""

from std.gpu.host import DeviceContext
from std.math import abs, log

from ops.common import dtype
from ops.softmax import log_softmax_rows, softmax_rows, logsumexp_rows
from tests.golden_io import read_f32
from tests.helpers import upload, zeros, download, assert_close

comptime TPB = 32
comptime WIDE_ROWS = 4
comptime WIDE_COLS = 100
comptime ROWS = 6
comptime COLS = 16
comptime TOL = Scalar[dtype](1e-6)


def main() raises:
    x = read_f32("tests/golden/softmax_in.bin")
    want_softmax = read_f32("tests/golden/softmax_out.bin")
    want_lse = read_f32("tests/golden/logsumexp_out.bin")

    if len(x) != ROWS * COLS:
        raise Error("golden softmax_in con tamano raro: ", len(x),
                    " (regenerar con gen_softmax.py?)")

    with DeviceContext() as ctx:
        a = upload[dtype](ctx, x)
        sm = zeros[dtype](ctx, ROWS * COLS)
        lse = zeros[dtype](ctx, ROWS)

        ctx.enqueue_function[softmax_rows[TPB], softmax_rows[TPB]](
            sm.unsafe_ptr(), a.unsafe_ptr(), COLS, grid_dim=ROWS, block_dim=TPB)
        ctx.enqueue_function[logsumexp_rows[TPB], logsumexp_rows[TPB]](
            lse.unsafe_ptr(), a.unsafe_ptr(), COLS, grid_dim=ROWS, block_dim=TPB)
        ctx.synchronize()

        got_sm = download[dtype](sm, ROWS * COLS)
        got_lse = download[dtype](lse, ROWS)

    # Elemento a elemento contra numpy.
    for r in range(ROWS):
        for c in range(COLS):
            assert_close(got_sm[r * COLS + c], want_softmax[r * COLS + c], TOL,
                         String("softmax fila=", r, " col=", c))
    print("PASS softmax_rows vs numpy (", ROWS, "filas x", COLS, ")")

    # La fila uniforme tiene que dar exactamente 1/COLS.
    uniform = Scalar[dtype](1.0) / Scalar[dtype](COLS)
    for c in range(COLS):
        assert_close(got_sm[c], uniform, TOL, String("fila uniforme col=", c))
    print("PASS softmax fila uniforme = 1/", COLS)

    # Cada fila tiene que sumar 1, incluidas las de +-1000. Esta comprobacion
    # es la que detecta el overflow: si exp desborda, la fila suma nan.
    for r in range(ROWS):
        total = Scalar[dtype](0)
        for c in range(COLS):
            total += got_sm[r * COLS + c]
        assert_close(total, 1.0, Scalar[dtype](1e-5),
                     String("la fila ", r, " no suma 1"))
    print("PASS softmax suma 1 por fila (sin overflow en +-1000)")

    # El logsumexp se compara con tolerancia relativa. La fila de +1000 da
    # ~1002.77, y a esa magnitud el ulp de float32 ya es ~6e-5: pedir 1e-6
    # absoluto seria pedir mas precision de la que existe.
    for r in range(ROWS):
        scale = abs(want_lse[r])
        if scale < 1.0:
            scale = Scalar[dtype](1.0)
        rel = abs(got_lse[r] - want_lse[r]) / scale
        if rel > TOL:
            raise Error("logsumexp fila=", r, ": got=", got_lse[r],
                        " want=", want_lse[r], " (error relativo=", rel, ")")
    print("PASS logsumexp_rows vs numpy")

    # Las pruebas nuevas abren su propio contexto: el `with` de arriba ya se
    # cerro y las comprobaciones de este bloque corren en host.
    with DeviceContext() as ctx2:
        test_log_softmax_matches_log_of_softmax(ctx2)
        test_wide_rows(ctx2)


def test_wide_rows(ctx: DeviceContext) raises:
    """Filas mas anchas que el bloque: la ruta de striding de los tres kernels."""
    x = read_f32("tests/golden/wide_in.bin")
    want_sm = read_f32("tests/golden/wide_softmax.bin")
    want_lse = read_f32("tests/golden/wide_logsumexp.bin")
    want_lsm = read_f32("tests/golden/wide_logsoftmax.bin")

    n = WIDE_ROWS * WIDE_COLS
    a = upload[dtype](ctx, x)
    sm = zeros[dtype](ctx, n)
    lse = zeros[dtype](ctx, WIDE_ROWS)
    lsm = zeros[dtype](ctx, n)

    ctx.enqueue_function[softmax_rows[TPB], softmax_rows[TPB]](
        sm.unsafe_ptr(), a.unsafe_ptr(), WIDE_COLS,
        grid_dim=WIDE_ROWS, block_dim=TPB)
    ctx.enqueue_function[logsumexp_rows[TPB], logsumexp_rows[TPB]](
        lse.unsafe_ptr(), a.unsafe_ptr(), WIDE_COLS,
        grid_dim=WIDE_ROWS, block_dim=TPB)
    ctx.enqueue_function[log_softmax_rows[TPB], log_softmax_rows[TPB]](
        lsm.unsafe_ptr(), a.unsafe_ptr(), WIDE_COLS,
        grid_dim=WIDE_ROWS, block_dim=TPB)
    ctx.synchronize()

    got_sm = download[dtype](sm, n)
    got_lse = download[dtype](lse, WIDE_ROWS)
    got_lsm = download[dtype](lsm, n)

    for i in range(n):
        assert_close(got_sm[i], want_sm[i], Scalar[dtype](1e-6),
                     String("wide softmax ", i))
        assert_close(got_lsm[i], want_lsm[i], Scalar[dtype](1e-4),
                     String("wide log softmax ", i))
    for r in range(WIDE_ROWS):
        scale = abs(want_lse[r])
        if scale < 1.0:
            scale = Scalar[dtype](1.0)
        rel = abs(got_lse[r] - want_lse[r]) / scale
        if rel > TOL:
            raise Error("wide logsumexp fila=", r, ": got=", got_lse[r],
                        " want=", want_lse[r])
        total = Scalar[dtype](0)
        for c in range(WIDE_COLS):
            total += got_sm[r * WIDE_COLS + c]
        assert_close(total, 1.0, Scalar[dtype](1e-5),
                     String("la fila ancha ", r, " no suma 1"))
    print("PASS filas de ", WIDE_COLS, " columnas (striding con TPB=", TPB, ")")


def test_log_softmax_matches_log_of_softmax(ctx: DeviceContext) raises:
    """El log_softmax coincide con el golden, y con log(softmax) donde este no
    desborda.

    La segunda parte es la que justifica que exista un kernel aparte: en las filas
    normales las dos rutas coinciden, pero en la fila de -1000 el softmax satura a
    valores diminutos y su log pierde precision, mientras que el log_softmax
    directo mantiene el valor exacto. Se comprueba donde el softmax es sano
    (> 1e-6) y se deja fuera donde no, que es justo el caso que motiva el kernel.
    """
    x = read_f32("tests/golden/softmax_in.bin")
    want_lsm = read_f32("tests/golden/logsoftmax_out.bin")
    n = ROWS * COLS

    a = upload[dtype](ctx, x)
    lsm = zeros[dtype](ctx, n)
    sm = zeros[dtype](ctx, n)
    ctx.enqueue_function[log_softmax_rows[TPB], log_softmax_rows[TPB]](
        lsm.unsafe_ptr(), a.unsafe_ptr(), COLS, grid_dim=ROWS, block_dim=TPB)
    ctx.enqueue_function[softmax_rows[TPB], softmax_rows[TPB]](
        sm.unsafe_ptr(), a.unsafe_ptr(), COLS, grid_dim=ROWS, block_dim=TPB)
    ctx.synchronize()

    got_lsm = download[dtype](lsm, n)
    got_sm = download[dtype](sm, n)

    for i in range(n):
        assert_close(got_lsm[i], want_lsm[i], Scalar[dtype](1e-4),
                     String("log softmax ", i))
        if got_sm[i] > Scalar[dtype](1e-6):
            assert_close(got_lsm[i], log(got_sm[i]), Scalar[dtype](1e-3),
                         String("log_softmax vs log(softmax) en ", i))
    print("PASS log_softmax_rows vs numpy, y coincide con log(softmax) donde "
          "este no pierde precision")
