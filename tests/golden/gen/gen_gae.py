"""Golden de la GAE truncada, con la funcion REAL de Stoix.

Correr desde la raiz de mojo_spo:
    ../.venv/bin/python tests/golden/gen/gen_gae.py

Se importa `batch_truncated_generalized_advantage_estimation` de Stoix en vez de
reimplementar la formula: asi el golden verifica que calculamos lo MISMO que la
referencia, incluidos los detalles que se escapan al leer el paper.

Que hace, en corto:

    delta_t = r + gamma*v_t - v_tm1                (error TD de un paso)
    acc     = delta + gamma*lambda*acc*(1 - trunc) (acumulado hacia ATRAS)
    target  = v_tm1 + acc                          (lo que aprende el critico)

El detalle fino esta en el comentario del propio Stoix: en un punto de truncacion
el acumulador se resetea PERO el delta de ese paso si se usa. O sea que truncar no
borra la informacion del paso, corta la propagacion hacia atras.

Como lo llama SPO (ff_spo.py, _critic_loss_fn):
    discount_t = (1 - done) * gamma
    v_tm1      = critico TARGET sobre las observaciones
    v_t        = critico TARGET sobre las bootstrap_obs
    truncation = seq.truncated

Tres casos:
    caso 0   sin truncacion, con episodios que terminan (done) en medio
    caso 1   con truncacion en varios sitios: la rama que casi nunca se prueba
    caso 2   como el tres en raya de verdad: partidas cortas (3-5 pasos) que
             siempre terminan solas, nunca truncadas
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_default_matmul_precision", "highest")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))
# La raiz del repo de Stoix, para importar su implementacion de verdad.
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from stoix.utils.multistep import batch_truncated_generalized_advantage_estimation

GAMMA = 0.99        # config de Stoix
LAMBDA = 0.95       # config de Stoix
rng = np.random.default_rng(53)
lines = []


def write(case, name, arr):
    np.asarray(arr, dtype=np.float32).tofile(
        os.path.join(OUT, f"gae{case}_{name}.bin"))


def make_case(case, b, t, done_prob, trunc_prob, ttt_like=False):
    r = rng.normal(0, 1, (b, t)).astype(np.float32)
    v_tm1 = rng.normal(0, 1, (b, t)).astype(np.float32)
    v_t = rng.normal(0, 1, (b, t)).astype(np.float32)

    if ttt_like:
        # Tres en raya: la partida acaba cada 3-5 pasos y la recompensa solo llega
        # al final (1 / 0.5 / 0). Entre medias, cero.
        done = np.zeros((b, t), dtype=np.float32)
        r = np.zeros((b, t), dtype=np.float32)
        for i in range(b):
            k = 0
            while k < t:
                length = int(rng.integers(3, 6))
                k += length
                if k - 1 < t:
                    done[i, k - 1] = 1.0
                    r[i, k - 1] = float(rng.choice([0.0, 0.5, 1.0]))
        trunc = np.zeros((b, t), dtype=np.float32)
    else:
        done = (rng.random((b, t)) < done_prob).astype(np.float32)
        trunc = (rng.random((b, t)) < trunc_prob).astype(np.float32)
        # Un paso no puede ser terminal y truncado a la vez: son cosas distintas
        # (uno acabo de verdad, al otro lo cortaron).
        trunc = trunc * (1.0 - done)

    discount = ((1.0 - done) * GAMMA).astype(np.float32)

    adv, targets = batch_truncated_generalized_advantage_estimation(
        jnp.asarray(r), jnp.asarray(discount), LAMBDA,
        v_tm1=jnp.asarray(v_tm1), v_t=jnp.asarray(v_t),
        truncation_t=jnp.asarray(trunc))

    for name, arr in [("r", r), ("discount", discount), ("v_tm1", v_tm1),
                      ("v_t", v_t), ("trunc", trunc), ("adv", adv),
                      ("targets", targets)]:
        write(case, name, arr)

    lines.append(f"case{case} B {b} T {t} dones {int(done.sum())} "
                 f"truncs {int(trunc.sum())}")
    print(f"  case{case}: B={b} T={t}  {int(done.sum()):3} dones, "
          f"{int(trunc.sum()):3} truncaciones  "
          f"|adv|max={float(jnp.abs(adv).max()):.4f}")


make_case(0, b=4, t=16, done_prob=0.15, trunc_prob=0.0)
make_case(1, b=4, t=16, done_prob=0.10, trunc_prob=0.15)
make_case(2, b=8, t=32, done_prob=0.0, trunc_prob=0.0, ttt_like=True)

with open(os.path.join(OUT, "gae.txt"), "w") as f:
    f.write(f"gamma {GAMMA}\nlambda {LAMBDA}\n")
    f.write("generado con stoix.utils.multistep."
            "batch_truncated_generalized_advantage_estimation\n")
    f.write("discount = (1 - done) * gamma, como en ff_spo._critic_loss_fn\n")
    f.write("gae<case>_{r,discount,v_tm1,v_t,trunc,adv,targets}.bin float32 B x T\n")
    f.write("case0 = sin truncacion\ncase1 = con truncacion\n")
    f.write("case2 = como tres en raya (partidas de 3-5 pasos, recompensa al final)\n")
    for line in lines:
        f.write(line + "\n")

print("golden de la GAE escrito en", OUT)
