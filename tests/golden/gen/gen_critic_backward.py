"""Golden de los gradientes del MLP del critico, con JAX (autodiff).

Correr desde la raiz de mojo_spo:
    ../.venv/bin/python tests/golden/gen/gen_critic_backward.py

La red es 18 -> H -> H -> 1 con ReLU, y la perdida la MISMA que usa Stoix para su
critico: `rlax.l2_loss(pred, target).mean()`, o sea media de 0.5*(V - target)^2.

Aqui el gradiente ya no es trivial: tiene que atravesar dos ReLU. Eso es lo nuevo
respecto a E1.4 y lo que de verdad se quiere verificar — que la mascara del ReLU
se aplica en el sitio correcto y con la activacion correcta.

Se guardan tambien las activaciones (a1, a2) y V, para que el test pueda comprobar
que su forward coincide antes de comparar gradientes: si el forward ya difiere, el
gradiente va a diferir por razones que no son el backward.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

# Sin esto JAX usa TF32 en GPU (10 bits de mantisa) y el golden sale con ~3
# digitos buenos, menos preciso que el kernel que verifica. Medido en E1.4:
# 1.3e-2 de error con TF32 contra 2.3e-6 con float32 de verdad.
jax.config.update("jax_default_matmul_precision", "highest")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

IN_DIM = 18
OUT_DIM = 1
# Dos anchuras y dos batches. El batch 20 es ragged (no multiplo del tile de 16) y
# H=64 da varios tiles: entre los dos casos se pisan todas las rutas.
CASES = [(32, 20), (64, 64)]

rng = np.random.default_rng(31)
lines = []

for hidden, m in CASES:
    tag = f"cbwd_h{hidden}_m{m}"

    w1 = rng.normal(0, 1 / np.sqrt(IN_DIM), (IN_DIM, hidden)).astype(np.float32)
    b1 = rng.normal(0, 0.1, (hidden,)).astype(np.float32)
    w2 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, hidden)).astype(np.float32)
    b2 = rng.normal(0, 0.1, (hidden,)).astype(np.float32)
    w3 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, OUT_DIM)).astype(np.float32)
    b3 = rng.normal(0, 0.1, (OUT_DIM,)).astype(np.float32)

    # Tableros con la pinta de los de verdad: dos planos 0/1.
    x = np.zeros((m, IN_DIM), dtype=np.float32)
    for r in range(m):
        for c in range(9):
            who = rng.integers(0, 3)
            if who == 1:
                x[r, c] = 1.0
            elif who == 2:
                x[r, 9 + c] = 1.0
    target = rng.normal(0, 1, (m, OUT_DIM)).astype(np.float32)

    def forward(p, x_):
        a1 = jnp.maximum(x_ @ p[0] + p[1], 0.0)
        a2 = jnp.maximum(a1 @ p[2] + p[3], 0.0)
        return a1, a2, a2 @ p[4] + p[5]

    def loss(p, x_, t_):
        _, _, v = forward(p, x_)
        # rlax.l2_loss(pred, target) = 0.5*(pred-target)^2, y luego .mean()
        return jnp.mean(0.5 * (v - t_) ** 2)

    params = [jnp.asarray(a) for a in (w1, b1, w2, b2, w3, b3)]
    grads = jax.grad(loss)(params, jnp.asarray(x), jnp.asarray(target))
    a1, a2, v = forward(params, jnp.asarray(x))

    for name, arr in [("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2),
                      ("w3", w3), ("b3", b3), ("x", x), ("target", target)]:
        arr.tofile(os.path.join(OUT, f"{tag}_{name}.bin"))
    for name, arr in [("a1", a1), ("a2", a2), ("v", v)]:
        np.asarray(arr, dtype=np.float32).tofile(os.path.join(OUT, f"{tag}_{name}.bin"))
    for name, arr in zip(["dw1", "db1", "dw2", "db2", "dw3", "db3"], grads):
        np.asarray(arr, dtype=np.float32).tofile(os.path.join(OUT, f"{tag}_{name}.bin"))

    dead1 = float((np.asarray(a1) == 0).mean())
    dead2 = float((np.asarray(a2) == 0).mean())
    lines.append(f"hidden {hidden} batch {m} relu_apagadas {dead1:.3f}/{dead2:.3f}")
    print(f"  hidden {hidden:3} batch {m:2}: loss={float(loss(params, jnp.asarray(x), jnp.asarray(target))):.5f}  "
          f"|dW1|max={float(jnp.abs(grads[0]).max()):.5f}  "
          f"|dW3|max={float(jnp.abs(grads[4]).max()):.5f}  "
          f"ReLU apagadas {dead1:.0%}/{dead2:.0%}")

with open(os.path.join(OUT, "cbwd.txt"), "w") as f:
    f.write(f"in_dim {IN_DIM}\nout_dim {OUT_DIM}\n")
    f.write("casos (hidden, batch): " + " ".join(f"({h},{m})" for h, m in CASES) + "\n")
    f.write("perdida: mean(0.5*(V-target)^2), la misma que rlax.l2_loss().mean()\n")
    f.write("gradientes por jax.grad con precision HIGHEST\n")
    f.write("ficheros: cbwd_h<H>_m<M>_{w1,b1,w2,b2,w3,b3,x,target,a1,a2,v,"
            "dw1,db1,dw2,db2,dw3,db3}.bin\n")
    for line in lines:
        f.write(line + "\n")

print("golden del backward del critico escrito en", OUT)
