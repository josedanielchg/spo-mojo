"""Genera el golden de softmax/logsumexp por filas.

Correr desde la raiz de mojo_spo:
    ../.venv/bin/python tests/golden/gen/gen_softmax.py

Escribe:
    softmax_in.bin    (rows x cols, float32 crudo)
    softmax_out.bin   (rows x cols, softmax por filas)
    logsumexp_out.bin (rows, float32)
    softmax.txt       (shapes y que hay en cada fila)
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

ROWS = 6
COLS = 16

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

x.tofile(os.path.join(OUT, "softmax_in.bin"))
softmax.tofile(os.path.join(OUT, "softmax_out.bin"))
logsumexp.tofile(os.path.join(OUT, "logsumexp_out.bin"))

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

print("golden de softmax escrito en", OUT)
