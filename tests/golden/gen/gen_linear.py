"""Genera el golden de la capa lineal y = x @ W + b.

Correr desde la raiz de mojo_spo:
    ../.venv/bin/python tests/golden/gen/gen_linear.py

Tres casos, elegidos a proposito y no al azar:

  caso 0   (4, 18) @ (18, 64)   la PRIMERA capa real del critico: 18 entradas
                                (los dos planos del tablero) y 64 ocultas
  caso 1   (64, 64) @ (64, 64)  una capa oculta con batch grande: varios tiles
                                en las tres dimensiones
  caso 2   (7, 5) @ (5, 3)      ragged a proposito: ninguna dimension es multiplo
                                del tile, asi que ejercita todos los guards

Escribe un .bin por tensor y un .txt con las shapes.
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
    # Escala tipo inicializacion de red: pequena, para que la suma de K terminos
    # no se dispare y el error relativo sea comparable entre casos.
    x = rng.normal(0.0, 1.0, size=(M, K)).astype(np.float32)
    w = rng.normal(0.0, 1.0 / np.sqrt(K), size=(K, N)).astype(np.float32)
    b = rng.normal(0.0, 0.1, size=(N,)).astype(np.float32)

    # La referencia. float32 en todo el camino para comparar con el kernel en
    # las mismas condiciones (numpy acumularia en float64 si se le deja).
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
