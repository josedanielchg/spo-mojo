"""Golden for Adam + global-norm clip, generated with OPTAX (Stoix's).

Run from mojo_spo's root:
    ../.venv/bin/python tests/golden/gen/gen_adam.py

The real library is used, not a reimplementation of the formulas: that way the
golden verifies that our kernel does the SAME thing Stoix will do, including the
details that get forgotten (the bias correction of both moments, and the exact
order of the operations).

The configuration is Stoix's, verified in ff_spo.py:

    optax.chain(
        optax.clip_by_global_norm(max_grad_norm),   # the clip FIRST
        optax.adam(lr, eps=1e-5),                   # and adam AFTERWARDS
    )

Mind eps: Stoix sets 1e-5 explicitly, not the 1e-8 optax ships by default. With
small gradients the difference shows.

The norm is GLOBAL: it is summed over ALL the tensors together, not tensor by
tensor. That is the typical mistake when reimplementing it.

Two cases, to cover the clip's two branches:
    case 0   small gradients -> the norm does not reach the limit, the clip does nothing
    case 1   large gradients -> the clip DOES bite

Three steps in each, because Adam's bias correction depends on the step number: a
single step would not tell an implementation with the correction from one without
it.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_default_matmul_precision", "highest")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

# A mini network with the same SHAPE as the critic (6 tensors), so that the test
# exercises the real case of a global norm over several tensors of different
# sizes.
IN_DIM, HIDDEN, OUT_DIM = 4, 5, 1
SHAPES = [
    ("w1", (IN_DIM, HIDDEN)),
    ("b1", (HIDDEN,)),
    ("w2", (HIDDEN, HIDDEN)),
    ("b2", (HIDDEN,)),
    ("w3", (HIDDEN, OUT_DIM)),
    ("b3", (OUT_DIM,)),
]

LR = 3e-4          # Stoix's config: actor_lr = critic_lr = 3e-4
MAX_NORM = 0.5     # Stoix's config: max_grad_norm = 0.5
EPS = 1e-5         # Stoix sets it explicitly
STEPS = 3

rng = np.random.default_rng(41)
lines = []

for case, grad_scale in enumerate([0.05, 5.0]):
    params = {name: rng.normal(0, 0.3, shape).astype(np.float32)
              for name, shape in SHAPES}
    # Gradients held FIXED across the three steps: that way the only thing that
    # changes between steps is Adam's internal state, which is exactly what we
    # want to verify.
    grads = {name: (rng.normal(0, grad_scale, shape)).astype(np.float32)
             for name, shape in SHAPES}

    optim = optax.chain(optax.clip_by_global_norm(MAX_NORM),
                        optax.adam(LR, eps=EPS))
    p = {k: jnp.asarray(v) for k, v in params.items()}
    g = {k: jnp.asarray(v) for k, v in grads.items()}
    state = optim.init(p)

    gnorm = float(optax.global_norm(g))
    clipped = gnorm > MAX_NORM
    lines.append(f"case{case} grad_scale {grad_scale} global_norm {gnorm:.6f} "
                 f"clip {'SI' if clipped else 'NO'}")
    print(f"  case{case}: norma global {gnorm:.4f}  "
          f"(limite {MAX_NORM}) -> el clip {'RECORTA' if clipped else 'no actua'}")

    for name, arr in params.items():
        arr.tofile(os.path.join(OUT, f"adam{case}_{name}_p0.bin"))
    for name, arr in grads.items():
        arr.tofile(os.path.join(OUT, f"adam{case}_{name}_g.bin"))

    for step in range(1, STEPS + 1):
        updates, state = optim.update(g, state, p)
        p = optax.apply_updates(p, updates)
        for name in p:
            np.asarray(p[name], dtype=np.float32).tofile(
                os.path.join(OUT, f"adam{case}_{name}_p{step}.bin"))
        total = sum(float(jnp.sum(jnp.abs(v))) for v in p.values())
        print(f"    paso {step}: suma |params| = {total:.6f}")

with open(os.path.join(OUT, "adam.txt"), "w") as f:
    f.write(f"in_dim {IN_DIM}\nhidden {HIDDEN}\nout_dim {OUT_DIM}\n")
    f.write(f"lr {LR}\nmax_norm {MAX_NORM}\neps {EPS}\nsteps {STEPS}\n")
    f.write("b1 0.9  b2 0.999  (defaults de optax.adam)\n")
    f.write("orden: clip_by_global_norm PRIMERO, adam DESPUES\n")
    f.write("adam<case>_<name>_p0.bin  params iniciales\n")
    f.write("adam<case>_<name>_g.bin   gradientes (fijos en los 3 pasos)\n")
    f.write("adam<case>_<name>_p<t>.bin params despues del paso t\n")
    for line in lines:
        f.write(line + "\n")

print("golden de adam escrito en", OUT)
