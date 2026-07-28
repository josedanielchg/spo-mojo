"""Golden de Adam + clip por norma global, generado con OPTAX (el de Stoix).

Correr desde la raiz de mojo_spo:
    ../.venv/bin/python tests/golden/gen/gen_adam.py

Se usa la libreria de verdad, no una reimplementacion de las formulas: asi el
golden verifica que nuestro kernel hace lo MISMO que hara Stoix, incluidos los
detalles que se olvidan (la correccion de sesgo de los dos momentos, y el orden
exacto de las operaciones).

La configuracion es la de Stoix, verificada en ff_spo.py:

    optax.chain(
        optax.clip_by_global_norm(max_grad_norm),   # PRIMERO el clip
        optax.adam(lr, eps=1e-5),                   # y DESPUES adam
    )

Ojo con eps: Stoix pone 1e-5 explicitamente, no el 1e-8 que trae optax por
defecto. Con gradientes pequenos la diferencia se nota.

La norma es GLOBAL: se suma sobre TODOS los tensores juntos, no tensor a tensor.
Ese es el error tipico al reimplementarlo.

Dos casos, para cubrir las dos ramas del clip:
    caso 0   gradientes pequenos -> la norma no llega al limite, el clip no actua
    caso 1   gradientes grandes  -> el clip SI recorta

Tres pasos en cada uno, porque la correccion de sesgo de Adam depende del numero
de paso: un solo paso no distinguiria una implementacion con correccion de una sin
ella.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_default_matmul_precision", "highest")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

# Una red mini con la misma FORMA que el critico (6 tensores), para que el test
# ejercite el caso real de una norma global sobre varios tensores de tamanos
# distintos.
IN_DIM, HIDDEN, OUT_DIM = 4, 5, 1
SHAPES = [
    ("w1", (IN_DIM, HIDDEN)),
    ("b1", (HIDDEN,)),
    ("w2", (HIDDEN, HIDDEN)),
    ("b2", (HIDDEN,)),
    ("w3", (HIDDEN, OUT_DIM)),
    ("b3", (OUT_DIM,)),
]

LR = 3e-4          # config de Stoix: actor_lr = critic_lr = 3e-4
MAX_NORM = 0.5     # config de Stoix: max_grad_norm = 0.5
EPS = 1e-5         # Stoix lo pone explicito
STEPS = 3

rng = np.random.default_rng(41)
lines = []

for case, grad_scale in enumerate([0.05, 5.0]):
    params = {name: rng.normal(0, 0.3, shape).astype(np.float32)
              for name, shape in SHAPES}
    # Gradientes FIJOS a lo largo de los tres pasos: asi lo unico que cambia
    # entre pasos es el estado interno de Adam, que es justo lo que se quiere
    # verificar.
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
