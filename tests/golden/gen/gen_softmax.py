"""Genera el golden de softmax/logsumexp por filas.

Correr desde la raiz de mojo_spo:
    ../.venv/bin/python tests/golden/gen/gen_softmax.py

Escribe:
    softmax_in.bin    (rows x cols, float32 crudo)
    softmax_out.bin   (rows x cols, softmax por filas)
    logsumexp_out.bin (rows, float32)
    logsoftmax_out.bin(rows x cols, log softmax por filas)
    softmax.txt       (shapes y que hay en cada fila)

Y lo mismo para una fila ANCHA (wide_*), con mas columnas que hilos por bloque.
Eso importa: los kernels recorren la fila con `i += TPB`, y con cols <= TPB ese
bucle da una sola vuelta. O sea que hasta ahora la ruta de striding no se habia
ejecutado NUNCA, ni en softmax_rows (que existe desde la fase 2) ni en el
log_softmax_rows nuevo. WIDE_COLS es ademas ragged (no multiplo de 32) para pisar
la ultima vuelta parcial.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

ROWS = 6
COLS = 16
WIDE_ROWS = 4
WIDE_COLS = 100      # > TPB=32 y no multiplo: 3 vueltas completas y una parcial

rng = np.random.default_rng(0)
x = rng.normal(0.0, 2.0, size=(ROWS, COLS)).astype(np.float32)

# Casos que quiero forzar, no dejar al azar:
x[0, :] = 0.0            # fila uniforme -> softmax = 1/COLS
x[1, :] = 1000.0         # todo enorme -> sin el truco del max esto es inf/inf
x[2, 0] = 1000.0         # un solo logit enorme -> softmax casi one-hot
x[3, :] = -1000.0        # todo muy negativo -> exp() satura a 0 sin el truco

# Softmax estable, igual que lo hara el kernel: restar el max de la fila.
m = x.max(axis=1, keepdims=True)
e = np.exp(x - m)
s = e.sum(axis=1, keepdims=True)
softmax = (e / s).astype(np.float32)
logsumexp = (m[:, 0] + np.log(s[:, 0])).astype(np.float32)
# log softmax calculado DIRECTO (no como log del softmax): un logit muy bajo
# desborda el softmax a 0 y su log seria -inf aunque el valor exacto fuera
# representable. La fila 3 (todo -1000) y la 2 (un 1000) lo pisan.
logsoftmax = ((x - m) - np.log(s)).astype(np.float32)

x.tofile(os.path.join(OUT, "softmax_in.bin"))
softmax.tofile(os.path.join(OUT, "softmax_out.bin"))
logsumexp.tofile(os.path.join(OUT, "logsumexp_out.bin"))
logsoftmax.tofile(os.path.join(OUT, "logsoftmax_out.bin"))

# --- la fila ancha: la ruta de striding ---
xw = rng.normal(0.0, 2.0, size=(WIDE_ROWS, WIDE_COLS)).astype(np.float32)
xw[0, :] = 0.0                    # uniforme sobre 100 columnas
xw[1, WIDE_COLS - 1] = 500.0      # el maximo en la ULTIMA columna: si el striding
                                  # no llegara al final, el max saldria mal
xw[2, :] = -800.0                 # underflow sin el truco del max
mw = xw.max(axis=1, keepdims=True)
ew = np.exp(xw - mw)
sw = ew.sum(axis=1, keepdims=True)
softmax_w = (ew / sw).astype(np.float32)
logsumexp_w = (mw[:, 0] + np.log(sw[:, 0])).astype(np.float32)
logsoftmax_w = ((xw - mw) - np.log(sw)).astype(np.float32)

xw.tofile(os.path.join(OUT, "wide_in.bin"))
softmax_w.tofile(os.path.join(OUT, "wide_softmax.bin"))
logsumexp_w.tofile(os.path.join(OUT, "wide_logsumexp.bin"))
logsoftmax_w.tofile(os.path.join(OUT, "wide_logsoftmax.bin"))

with open(os.path.join(OUT, "softmax.txt"), "w") as f:
    f.write(f"rows {ROWS}\n")
    f.write(f"cols {COLS}\n")
    f.write("softmax_in.bin float32 rows x cols\n")
    f.write("softmax_out.bin float32 rows x cols\n")
    f.write("logsumexp_out.bin float32 rows\n")
    f.write("row0 = todo 0 (uniforme)\n")
    f.write("row1 = todo 1000 (overflow si no se resta el max)\n")
    f.write("row2 = un 1000 y el resto normal (casi one-hot)\n")
    f.write("row3 = todo -1000 (underflow si no se resta el max)\n")
    f.write("row4,row5 = normal(0, 2) con seed 0\n")
    f.write("logsoftmax_out.bin float32 rows x cols\n")
    f.write(f"\nwide_rows {WIDE_ROWS}\nwide_cols {WIDE_COLS}\n")
    f.write("wide_*.bin: cols > TPB, para ejercitar el bucle `i += TPB`\n")
    f.write("wide row0 = uniforme; row1 = el maximo en la ULTIMA columna;\n")
    f.write("wide row2 = todo -800; row3 = normal(0,2)\n")

print("golden de softmax escrito en", OUT)
