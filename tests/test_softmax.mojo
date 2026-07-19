"""Softmax y logsumexp contra el golden de numpy.

El golden lo genera tests/golden/gen/gen_softmax.py. Las filas no son aleatorias
del todo: las cuatro primeras estan puestas a mano para reventar una
implementacion ingenua.
  fila 0: todo 0      -> softmax uniforme exacto, 1/COLS
  fila 1: todo +1000  -> exp(1000) = inf. Sin restar el max sale inf/inf = nan
  fila 2: un +1000    -> softmax casi one-hot
  fila 3: todo -1000  -> exp(-1000) = 0. Sin restar el max sale 0/0 = nan
"""

from std.gpu.host import DeviceContext
from std.math import abs

from ops.common import dtype
from ops.softmax import softmax_rows, logsumexp_rows
from tests.golden_io import read_f32
from tests.helpers import upload, zeros, download, assert_close

comptime TPB = 32
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

    # --- 1. elemento a elemento contra numpy ---
    for r in range(ROWS):
        for c in range(COLS):
            assert_close(got_sm[r * COLS + c], want_softmax[r * COLS + c], TOL,
                         String("softmax fila=", r, " col=", c))
    print("PASS softmax_rows vs numpy (", ROWS, "filas x", COLS, ")")

    # --- 2. la fila uniforme tiene que dar exactamente 1/COLS ---
    uniform = Scalar[dtype](1.0) / Scalar[dtype](COLS)
    for c in range(COLS):
        assert_close(got_sm[c], uniform, TOL, String("fila uniforme col=", c))
    print("PASS softmax fila uniforme = 1/", COLS)

    # --- 3. cada fila suma 1, incluidas las de +-1000 ---
    # Esta es la que detecta el overflow: si exp desborda, la fila suma nan.
    for r in range(ROWS):
        total = Scalar[dtype](0)
        for c in range(COLS):
            total += got_sm[r * COLS + c]
        assert_close(total, 1.0, Scalar[dtype](1e-5),
                     String("la fila ", r, " no suma 1"))
    print("PASS softmax suma 1 por fila (sin overflow en +-1000)")

    # --- 4. logsumexp, con tolerancia RELATIVA ---
    # La fila de +1000 da ~1002.77, y a esa magnitud el ulp de float32 ya es
    # ~6e-5: pedir 1e-6 absoluto seria pedir mas precision de la que existe.
    for r in range(ROWS):
        scale = abs(want_lse[r])
        if scale < 1.0:
            scale = Scalar[dtype](1.0)
        rel = abs(got_lse[r] - want_lse[r]) / scale
        if rel > TOL:
            raise Error("logsumexp fila=", r, ": got=", got_lse[r],
                        " want=", want_lse[r], " (error relativo=", rel, ")")
    print("PASS logsumexp_rows vs numpy")
