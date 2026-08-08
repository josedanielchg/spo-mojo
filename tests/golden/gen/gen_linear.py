"""Generates the golden for the linear layer y = x @ W + b.

Run from mojo_spo's root:
    ../.venv/bin/python tests/golden/gen/gen_linear.py

Three cases, chosen on purpose and not at random:

  case 0   (4, 18) @ (18, 64)   the critic's FIRST real layer: 18 inputs (the
                                board's two planes) and 64 hidden
  case 1   (64, 64) @ (64, 64)  a hidden layer with a large batch: several tiles
                                across all three dimensions
  case 2   (7, 5) @ (5, 3)      deliberately ragged: no dimension is a multiple of
                                the tile, so it exercises every guard

It writes one .bin per tensor and a .txt with the shapes.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

CASES = [
    (4, 18, 64),    # M, K, N
    (64, 64, 64),
    (7, 5, 3),
]

rng = np.random.default_rng(7)
lines = []

for i, (M, K, N) in enumerate(CASES):
    # Network-initialisation-style scale: small, so that the sum of K terms does
    # not blow up and the relative error stays comparable across cases.
    x = rng.normal(0.0, 1.0, size=(M, K)).astype(np.float32)
    w = rng.normal(0.0, 1.0 / np.sqrt(K), size=(K, N)).astype(np.float32)
    b = rng.normal(0.0, 0.1, size=(N,)).astype(np.float32)

    # The reference. float32 all the way, to compare against the kernel under the
    # same conditions (numpy would accumulate in float64 if left to itself).
    y = (x @ w + b).astype(np.float32)

    x.tofile(os.path.join(OUT, f"linear{i}_x.bin"))
    w.tofile(os.path.join(OUT, f"linear{i}_w.bin"))
    b.tofile(os.path.join(OUT, f"linear{i}_b.bin"))
    y.tofile(os.path.join(OUT, f"linear{i}_y.bin"))
    lines.append(f"case{i} M {M} K {K} N {N}")

with open(os.path.join(OUT, "linear.txt"), "w") as f:
    f.write(f"cases {len(CASES)}\n")
    for line in lines:
        f.write(line + "\n")
    f.write("linear<i>_x.bin float32 M x K\n")
    f.write("linear<i>_w.bin float32 K x N\n")
    f.write("linear<i>_b.bin float32 N\n")
    f.write("linear<i>_y.bin float32 M x N   (= x @ w + b)\n")
    f.write("case0 = primera capa del critico (18 entradas -> 64 ocultas)\n")
    f.write("case1 = capa oculta 64x64 con batch 64 (varios tiles)\n")
    f.write("case2 = ragged 7x5x3 (ninguna dimension multiplo del tile)\n")

print("golden de la capa lineal escrito en", OUT)
for i, (M, K, N) in enumerate(CASES):
    print(f"  case{i}: ({M},{K}) @ ({K},{N}) + ({N},) -> ({M},{N})")
