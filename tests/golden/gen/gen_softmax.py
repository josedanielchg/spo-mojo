"""Generates the golden for row-wise softmax/logsumexp.

Run from mojo_spo's root:
    ../.venv/bin/python tests/golden/gen/gen_softmax.py

It writes:
    softmax_in.bin    (rows x cols, raw float32)
    softmax_out.bin   (rows x cols, row-wise softmax)
    logsumexp_out.bin (rows, float32)
    logsoftmax_out.bin(rows x cols, row-wise log softmax)
    softmax.txt       (shapes and what is in each row)

And the same for a WIDE row (wide_*), with more columns than threads per block.
That matters: the kernels walk the row with `i += TPB`, and with cols <= TPB that
loop goes round exactly once. That is, until now the striding path had NEVER been
executed, neither in softmax_rows (which has existed since phase 2) nor in the new
log_softmax_rows. WIDE_COLS is also ragged (not a multiple of 32) so as to hit the
last partial turn.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

ROWS = 6
COLS = 16
WIDE_ROWS = 4
WIDE_COLS = 100      # > TPB=32 and not a multiple: 3 full turns and a partial one

rng = np.random.default_rng(0)
x = rng.normal(0.0, 2.0, size=(ROWS, COLS)).astype(np.float32)

# Cases I want to force, not leave to chance:
x[0, :] = 0.0            # uniform row -> softmax = 1/COLS
x[1, :] = 1000.0         # all huge -> without the max trick this is inf/inf
x[2, 0] = 1000.0         # a single huge logit -> nearly one-hot softmax
x[3, :] = -1000.0        # all very negative -> exp() saturates to 0 without the trick

# Stable softmax, exactly as the kernel will do it: subtract the row's max.
m = x.max(axis=1, keepdims=True)
e = np.exp(x - m)
s = e.sum(axis=1, keepdims=True)
softmax = (e / s).astype(np.float32)
logsumexp = (m[:, 0] + np.log(s[:, 0])).astype(np.float32)
# log softmax computed DIRECTLY (not as the log of the softmax): a very low logit
# underflows the softmax to 0 and its log would be -inf even though the exact value
# was representable. Row 3 (all -1000) and row 2 (one 1000) hit that.
logsoftmax = ((x - m) - np.log(s)).astype(np.float32)

x.tofile(os.path.join(OUT, "softmax_in.bin"))
softmax.tofile(os.path.join(OUT, "softmax_out.bin"))
logsumexp.tofile(os.path.join(OUT, "logsumexp_out.bin"))
logsoftmax.tofile(os.path.join(OUT, "logsoftmax_out.bin"))

# --- the wide row: the striding path ---
xw = rng.normal(0.0, 2.0, size=(WIDE_ROWS, WIDE_COLS)).astype(np.float32)
xw[0, :] = 0.0                    # uniform over 100 columns
xw[1, WIDE_COLS - 1] = 500.0      # the maximum in the LAST column: if the striding
                                  # did not reach the end, the max would be wrong
xw[2, :] = -800.0                 # underflow without the max trick
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
