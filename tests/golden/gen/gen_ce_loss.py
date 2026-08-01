"""Golden de la entropia cruzada ponderada del M-step (ecuacion 11 del paper).

Correr desde la raiz de mojo_spo:
    ../.venv/bin/python tests/golden/gen/gen_ce_loss.py

La ecuacion 11 es

    max_theta  E_{s~mu} [ E_{a~q(.|s)} [ log pi(a|s,theta) ] ]

o sea, entropia cruzada contra q. El paper la escribe con q como DISTRIBUCION;
Stoix la implementa como estimador Monte Carlo sobre las N particulas:

    loss = -SUM_n  w_n * log pi(a_n)          (compute_cross_entropy_loss)

Las dos formas son la MISMA cantidad, no una aproximacion de la otra: como las
acciones raiz se repiten entre particulas, agrupando por accion

    SUM_n w_n log pi(a_n) = SUM_a ( SUM_{n: a_n = a} w_n ) log pi(a) = SUM_a q(a) log pi(a)

Este golden existe para DEMOSTRAR esa igualdad y no tener que argumentarla:
calcula la forma de particulas llamando a la funcion de Stoix DE VERDAD (importada,
no reescrita) y la forma densa por separado, comprueba que coinciden, y guarda las
dos entradas y el resultado. El test de Mojo implementa la densa y compara.

Por que nos interesa la densa: nuestro readout ya produce q como vector [B, 9], asi
que la suma exacta sobre 9 acciones sale mas barata y sin ruido de muestreo que
recolectar log-probs de 512 particulas con repeticiones.

Caso importante que se cubre a proposito: **acciones ilegales**. Ahi pi(a) = 0
exacto y log pi(a) = -inf, mientras que q(a) = 0. El producto 0 * (-inf) es NaN en
IEEE, asi que la implementacion TIENE que saltarse los terminos con q = 0 en vez de
multiplicar. El golden incluye tableros con casillas ocupadas para que el test lo
pise.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
import distrax

# La funcion de Stoix, importada tal cual: si cambia, el golden cambia con ella.
from stoix.systems.mpo.continuous_loss import compute_cross_entropy_loss

jax.config.update("jax_default_matmul_precision", "highest")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

NUM_ACTIONS = 9
NEG_INF = np.float32(np.finfo(np.float32).min)

# Tres montajes: pocos/muchos particulas y un batch ragged.
CASES = [
    ("small", 3, 16),      # B=3 estados, N=16 particulas
    ("mid", 7, 64),        # B ragged
    ("big", 32, 512),      # el N que usamos ahora tras subir TPB_PARTICLES
    # El kernel de la perdida se lanza con blocks_for(n_rows) y TPB=32, asi que
    # con B <= 32 corre en UN bloque y el guard `row >= n_rows` no se pisa nunca.
    # 70 son tres bloques y el ultimo a medias.
    ("multiblock", 70, 64),
]

rng = np.random.default_rng(101)


def make_case(batch, num_particles):
    """Logits del actor (ya enmascarados), acciones raiz y pesos normalizados."""
    # Mascara: cada estado tiene entre 2 y 9 casillas libres.
    mask = np.zeros((batch, NUM_ACTIONS), dtype=np.float32)
    for b in range(batch):
        n_free = rng.integers(2, NUM_ACTIONS + 1)
        free = rng.choice(NUM_ACTIONS, size=n_free, replace=False)
        mask[b, free] = 1.0

    raw = rng.normal(0.0, 1.0, size=(batch, NUM_ACTIONS)).astype(np.float32)
    logits = np.where(mask > 0, raw, NEG_INF).astype(np.float32)

    # Las particulas solo pueden tener acciones raiz LEGALES: es lo que produce la
    # busqueda, porque muestrea del prior enmascarado.
    actions = np.zeros((num_particles, batch), dtype=np.int32)
    for b in range(batch):
        legal = np.flatnonzero(mask[b])
        actions[:, b] = rng.choice(legal, size=num_particles, replace=True)

    # Pesos normalizados por estado, como los que salen del softmax del readout.
    w = rng.gamma(1.0, 1.0, size=(num_particles, batch)).astype(np.float32)
    w = (w / w.sum(axis=0, keepdims=True)).astype(np.float32)
    return mask, logits, actions, w


lines = []
for name, batch, num_particles in CASES:
    mask, logits, actions, w = make_case(batch, num_particles)

    # --- forma de particulas: la funcion de Stoix, sin tocar ---
    dist = distrax.Categorical(logits=jnp.asarray(logits))
    loss_particles = float(
        compute_cross_entropy_loss(jnp.asarray(actions), jnp.asarray(w), dist)
    )

    # --- forma densa: q agregada por accion, y luego la suma sobre 9 ---
    q = np.zeros((batch, NUM_ACTIONS), dtype=np.float64)
    for b in range(batch):
        for n in range(num_particles):
            q[b, actions[n, b]] += w[n, b]

    # log_softmax estable. En las ilegales da -inf, y ahi q vale 0: el termino se
    # SALTA en vez de multiplicarse, porque 0 * -inf = NaN.
    shifted = logits.astype(np.float64) - logits.astype(np.float64).max(
        axis=1, keepdims=True)
    log_pi = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    per_state = np.zeros(batch, dtype=np.float64)
    for b in range(batch):
        acc = 0.0
        for a in range(NUM_ACTIONS):
            if q[b, a] != 0.0:
                acc += q[b, a] * log_pi[b, a]
        per_state[b] = -acc
    loss_dense = float(per_state.mean())

    # LA comprobacion: las dos formas tienen que dar el mismo numero.
    diff = abs(loss_particles - loss_dense)
    assert diff < 2e-5, (
        f"{name}: la forma de particulas ({loss_particles}) y la densa "
        f"({loss_dense}) no coinciden, diff={diff}")
    # Y q tiene que ser una distribucion sobre las LEGALES.
    assert np.allclose(q.sum(axis=1), 1.0, atol=1e-5), "q no suma 1"
    assert (q[mask == 0] == 0.0).all(), "q pone masa en una casilla ilegal"

    logits.tofile(os.path.join(OUT, f"ce_{name}_logits.bin"))
    mask.tofile(os.path.join(OUT, f"ce_{name}_mask.bin"))
    q.astype(np.float32).tofile(os.path.join(OUT, f"ce_{name}_q.bin"))
    log_pi.astype(np.float32).tofile(os.path.join(OUT, f"ce_{name}_logpi.bin"))
    per_state.astype(np.float32).tofile(os.path.join(OUT, f"ce_{name}_per_state.bin"))
    np.float32(loss_dense).tofile(os.path.join(OUT, f"ce_{name}_loss.bin"))

    n_illegal = int((mask == 0).sum())
    lines.append(f"{name} batch {batch} particles {num_particles} "
                 f"loss {loss_dense:.8f} illegal_cells {n_illegal} "
                 f"diff_particles_vs_dense {diff:.3e}")
    print(f"  {name:6} B={batch:2} N={num_particles:3}  "
          f"loss={loss_dense:.6f}  |particulas-densa|={diff:.2e}  "
          f"ilegales={n_illegal}")

with open(os.path.join(OUT, "ce_loss.txt"), "w") as f:
    f.write(f"num_actions {NUM_ACTIONS}\n")
    f.write(f"neg_inf {NEG_INF!r}\n")
    f.write("casos: " + " ".join(n for n, _, _ in CASES) + "\n")
    f.write("ce_<caso>_logits.bin    float32 B x 9  (ya enmascarados con NEG_INF)\n")
    f.write("ce_<caso>_mask.bin      float32 B x 9  (1 legal, 0 ocupada)\n")
    f.write("ce_<caso>_q.bin         float32 B x 9  (la q agregada, suma 1)\n")
    f.write("ce_<caso>_logpi.bin     float32 B x 9  (log_softmax; -inf en ilegales)\n")
    f.write("ce_<caso>_per_state.bin float32 B     (la perdida de cada estado)\n")
    f.write("ce_<caso>_loss.bin      float32 1     (la media sobre el batch)\n")
    f.write("\n")
    f.write("La forma de particulas se calcula con compute_cross_entropy_loss de\n")
    f.write("Stoix (importada) y coincide con la densa; ver la columna diff.\n")
    f.write("\n".join(lines) + "\n")

print("golden de la entropia cruzada escrito en", OUT)
