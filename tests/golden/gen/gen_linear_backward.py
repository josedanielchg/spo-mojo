"""Golden de los gradientes de la capa lineal, calculados con JAX (autodiff).

Correr desde la raiz de mojo_spo:
    ../.venv/bin/python tests/golden/gen/gen_linear_backward.py

Por que JAX y no solo diferencias finitas: Stoix no escribe ningun backward a
mano, deriva con `jax.grad`, y eso da el gradiente EXACTO (autodiff, no una
aproximacion numerica). Comparar contra el es mas fuerte que comparar contra
diferencias finitas, que tienen error de truncamiento y de cancelacion.

OJO con la precision: hay que forzar `jax_default_matmul_precision=highest` (ver
abajo), o JAX calcula las matmuls en TF32 y el golden sale con solo ~3 digitos
buenos.

Cuatro casos, elegidos para tapar el punto ciego del primer test (que usaba
M=3, K=4, N=8: TODO menor que el tile de 16, asi que el bucle de tiles corria una
sola vez y la ruta multi-tile no se probaba):

    caso 0   M=3  K=4  N=8      la red mini de las diferencias finitas
    caso 1   M=20 K=18 N=64     la PRIMERA capa real del critico, batch ragged
    caso 2   M=64 K=64 N=64     varios tiles en las tres dimensiones
    caso 3   M=1  K=1  N=1      degenerado: una sola fila, un solo peso

Se guarda tambien `dy` (el gradiente que entra) porque el backward lo necesita
como input, y `y` para poder comprobar de paso el forward con las mismas shapes.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

# CRITICO: por defecto JAX usa TF32 en GPUs NVIDIA para las matmuls, que solo
# tiene 10 bits de mantisa (~3 digitos decimales). Un golden generado asi es
# MENOS preciso que el kernel que pretende verificar: medido, el error de TF32
# contra float64 es 1.3e-2, mientras que float32 de verdad da 2.3e-6. Sin esta
# linea, el test falla y parece un bug del backward cuando el problema esta en la
# referencia.
jax.config.update("jax_default_matmul_precision", "highest")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

CASES = [
    (3, 4, 8),
    (20, 18, 64),
    (64, 64, 64),
    (1, 1, 1),
]

rng = np.random.default_rng(23)
lines = []

for i, (M, K, N) in enumerate(CASES):
    x = rng.normal(0.0, 1.0, size=(M, K)).astype(np.float32)
    w = rng.normal(0.0, 1.0 / np.sqrt(K), size=(K, N)).astype(np.float32)
    b = rng.normal(0.0, 0.1, size=(N,)).astype(np.float32)
    # dy con valores TODOS distintos: con un dy uniforme, confundir una traspuesta
    # o reducir por el eje equivocado puede dar el mismo numero por casualidad.
    g = rng.normal(0.0, 1.0, size=(M, N)).astype(np.float32)

    def loss(x_, w_, b_):
        # L = suma(g * y) tiene exactamente dL/dy = g, que es lo que se le pasa
        # al backward de Mojo. Asi los dos calculan lo mismo.
        return jnp.sum(g * (x_ @ w_ + b_))

    # Autodiff exacto respecto a los tres argumentos.
    dx, dw, db = jax.grad(loss, argnums=(0, 1, 2))(
        jnp.asarray(x), jnp.asarray(w), jnp.asarray(b))
    y = np.asarray(jnp.asarray(x) @ jnp.asarray(w) + jnp.asarray(b), dtype=np.float32)

    x.tofile(os.path.join(OUT, f"linbwd{i}_x.bin"))
    w.tofile(os.path.join(OUT, f"linbwd{i}_w.bin"))
    b.tofile(os.path.join(OUT, f"linbwd{i}_b.bin"))
    g.tofile(os.path.join(OUT, f"linbwd{i}_dy.bin"))
    y.tofile(os.path.join(OUT, f"linbwd{i}_y.bin"))
    np.asarray(dw, dtype=np.float32).tofile(os.path.join(OUT, f"linbwd{i}_dw.bin"))
    np.asarray(db, dtype=np.float32).tofile(os.path.join(OUT, f"linbwd{i}_db.bin"))
    np.asarray(dx, dtype=np.float32).tofile(os.path.join(OUT, f"linbwd{i}_dx.bin"))

    lines.append(f"case{i} M {M} K {K} N {N}")
    print(f"  case{i}: M={M:3} K={K:3} N={N:3}  "
          f"|dW|max={float(jnp.abs(dw).max()):8.4f}  "
          f"|db|max={float(jnp.abs(db).max()):8.4f}  "
          f"|dx|max={float(jnp.abs(dx).max()):8.4f}")

with open(os.path.join(OUT, "linbwd.txt"), "w") as f:
    f.write(f"cases {len(CASES)}\n")
    for line in lines:
        f.write(line + "\n")
    f.write("gradientes calculados con jax.grad (autodiff exacto)\n")
    f.write("perdida L = sum(g * (x @ w + b)), asi que dL/dy = g\n")
    f.write("linbwd<i>_x.bin  float32 M x K\n")
    f.write("linbwd<i>_w.bin  float32 K x N\n")
    f.write("linbwd<i>_b.bin  float32 N\n")
    f.write("linbwd<i>_dy.bin float32 M x N   (el g de la perdida)\n")
    f.write("linbwd<i>_y.bin  float32 M x N   (el forward, de paso)\n")
    f.write("linbwd<i>_dw.bin float32 K x N\n")
    f.write("linbwd<i>_db.bin float32 N\n")
    f.write("linbwd<i>_dx.bin float32 M x K\n")

print("golden de gradientes (JAX) escrito en", OUT)
