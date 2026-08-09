"""Golden for the linear layer's gradients, computed with JAX (autodiff).

Run from mojo_spo's root:
    ../.venv/bin/python tests/golden/gen/gen_linear_backward.py

Why JAX and not just finite differences: Stoix writes no backward by hand, it
differentiates with `jax.grad`, and that gives the EXACT gradient (autodiff, not a
numerical approximation). Comparing against it is stronger than comparing against
finite differences, which carry truncation and cancellation error.

MIND the precision: `jax_default_matmul_precision=highest` has to be forced (see
below), or JAX computes the matmuls in TF32 and the golden comes out with only ~3
good digits.

Four cases, chosen to plug the first test's blind spot (it used M=3, K=4, N=8:
EVERYTHING smaller than the tile of 16, so the tile loop ran exactly once and the
multi-tile path was never tested):

    case 0   M=3  K=4  N=8      the mini network of the finite differences
    case 1   M=20 K=18 N=64     the critic's FIRST real layer, ragged batch
    case 2   M=64 K=64 N=64     several tiles across all three dimensions
    case 3   M=1  K=1  N=1      degenerate: a single row, a single weight

`dy` (the incoming gradient) is stored too because the backward needs it as input,
and `y` so that the forward can be checked along the way with the same shapes.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

# CRITICAL: by default JAX uses TF32 on NVIDIA GPUs for matmuls, which has only 10
# mantissa bits (~3 decimal digits). A golden generated that way is LESS precise
# than the kernel it is meant to verify: measured, TF32's error against float64 is
# 1.3e-2, whereas true float32 gives 2.3e-6. Without this line, the test fails and
# looks like a bug in the backward when the problem is in the reference.
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
    # dy with ALL different values: with a uniform dy, confusing a transpose or
    # reducing along the wrong axis can give the same number by coincidence.
    g = rng.normal(0.0, 1.0, size=(M, N)).astype(np.float32)

    def loss(x_, w_, b_):
        # L = sum(g * y) has exactly dL/dy = g, which is what gets passed to Mojo's
        # backward. That way both compute the same thing.
        return jnp.sum(g * (x_ @ w_ + b_))

    # Exact autodiff with respect to all three arguments.
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
    f.write("gradients computed with jax.grad (exact autodiff)\n")
    f.write("loss L = sum(g * (x @ w + b)), so dL/dy = g\n")
    f.write("linbwd<i>_x.bin  float32 M x K\n")
    f.write("linbwd<i>_w.bin  float32 K x N\n")
    f.write("linbwd<i>_b.bin  float32 N\n")
    f.write("linbwd<i>_dy.bin float32 M x N   (the loss's g)\n")
    f.write("linbwd<i>_y.bin  float32 M x N   (the forward, in passing)\n")
    f.write("linbwd<i>_dw.bin float32 K x N\n")
    f.write("linbwd<i>_db.bin float32 N\n")
    f.write("linbwd<i>_dx.bin float32 M x K\n")

print("gradient golden (JAX) written to", OUT)
