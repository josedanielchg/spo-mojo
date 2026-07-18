"""Softmax y logsumexp contra el golden de numpy (tests/golden/gen/gen_softmax.py).

Las filas 1 y 3 del golden son las importantes: +1000 y -1000 en todas las
columnas. Sin restar el maximo dan inf/inf y 0/0 respectivamente.
"""

from std.gpu.host import DeviceContext
from std.math import abs

from ops.softmax import softmax_rows, logsumexp_rows, dtype
from tests.golden_io import read_f32

comptime TPB = 32
comptime ROWS = 6
comptime COLS = 16
comptime TOL = Scalar[dtype](1e-6)


def main() raises:
    x = read_f32("tests/golden/softmax_in.bin")
    want_softmax = read_f32("tests/golden/softmax_out.bin")
    want_lse = read_f32("tests/golden/logsumexp_out.bin")

    if len(x) != ROWS * COLS:
        raise Error("golden softmax_in con tamano raro: ", len(x))

    with DeviceContext() as ctx:
        a = ctx.enqueue_create_buffer[dtype](ROWS * COLS)
        a.enqueue_fill(0)
        with a.map_to_host() as h:
            for i in range(ROWS * COLS):
                h[i] = x[i]

        sm = ctx.enqueue_create_buffer[dtype](ROWS * COLS)
        sm.enqueue_fill(0)
        lse = ctx.enqueue_create_buffer[dtype](ROWS)
        lse.enqueue_fill(0)

        ctx.enqueue_function[softmax_rows[TPB], softmax_rows[TPB]](
            sm.unsafe_ptr(), a.unsafe_ptr(), COLS, grid_dim=ROWS, block_dim=TPB)
        ctx.enqueue_function[logsumexp_rows[TPB], logsumexp_rows[TPB]](
            lse.unsafe_ptr(), a.unsafe_ptr(), COLS, grid_dim=ROWS, block_dim=TPB)
        ctx.synchronize()

        with sm.map_to_host() as h:
            for i in range(ROWS * COLS):
                if abs(h[i] - want_softmax[i]) > TOL:
                    raise Error("softmax en ", i, ": got=", h[i],
                                " want=", want_softmax[i])
        print("PASS softmax_rows vs numpy (", ROWS, "filas x", COLS, ")")

        # La fila uniforme tiene que dar exactamente 1/COLS.
        with sm.map_to_host() as h:
            uniform = Scalar[dtype](1.0) / Scalar[dtype](COLS)
            for c in range(COLS):
                if abs(h[c] - uniform) > TOL:
                    raise Error("fila uniforme col ", c, ": got=", h[c])
        print("PASS softmax fila uniforme = 1/", COLS)

        # Suma por fila = 1 en todas, incluidas las de +1000 y -1000.
        with sm.map_to_host() as h:
            for r in range(ROWS):
                total = Scalar[dtype](0)
                for c in range(COLS):
                    total += h[r * COLS + c]
                if abs(total - 1.0) > Scalar[dtype](1e-5):
                    raise Error("fila ", r, " no suma 1: ", total)
        print("PASS softmax suma 1 por fila (sin overflow en +-1000)")

        # logsumexp: tolerancia relativa, porque la fila de 1000 vale ~1002.77
        # y 1e-6 absoluto es menos que el ulp de float32 en esa magnitud.
        with lse.map_to_host() as h:
            for r in range(ROWS):
                err = abs(h[r] - want_lse[r])
                scale = abs(want_lse[r])
                if scale < 1.0:
                    scale = Scalar[dtype](1.0)
                if err / scale > Scalar[dtype](1e-6):
                    raise Error("logsumexp fila ", r, ": got=", h[r],
                                " want=", want_lse[r])
        print("PASS logsumexp_rows vs numpy")
