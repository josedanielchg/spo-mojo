"""Golden for the M-step's weighted cross entropy (equation 11 of the paper).

Run from mojo_spo's root:
    ../.venv/bin/python tests/golden/gen/gen_ce_loss.py

Equation 11 is

    max_theta  E_{s~mu} [ E_{a~q(.|s)} [ log pi(a|s,theta) ] ]

that is, cross entropy against q. The paper writes it with q as a DISTRIBUTION;
Stoix implements it as a Monte Carlo estimator over the N particles:

    loss = -SUM_n  w_n * log pi(a_n)          (compute_cross_entropy_loss)

The two forms are the SAME quantity, not an approximation of one another: since the
root actions repeat across particles, grouping by action

    SUM_n w_n log pi(a_n) = SUM_a ( SUM_{n: a_n = a} w_n ) log pi(a) = SUM_a q(a) log pi(a)

This golden exists to PROVE that identity rather than argue for it: it computes the
particle form by calling Stoix's REAL function (imported, not rewritten) and the
dense form separately, checks that they agree, and stores both inputs and the
result. The Mojo test implements the dense one and compares.

Why the dense one interests us: our readout already produces q as a vector [B, 9],
so the exact sum over 9 actions comes out cheaper and free of sampling noise
compared with gathering log-probs from 512 particles with repetitions.

An important case covered on purpose: **illegal actions**. There pi(a) = 0 exactly
and log pi(a) = -inf, while q(a) = 0. The product 0 * (-inf) is NaN in IEEE, so the
implementation HAS to skip the terms with q = 0 rather than multiply. The golden
includes boards with occupied cells so that the test hits it.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
import distrax

# Stoix's function, imported as is: if it changes, the golden changes with it.
from stoix.systems.mpo.continuous_loss import compute_cross_entropy_loss

jax.config.update("jax_default_matmul_precision", "highest")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

NUM_ACTIONS = 9
NEG_INF = np.float32(np.finfo(np.float32).min)

# Three setups: few/many particles and a ragged batch.
CASES = [
    ("small", 3, 16),      # B=3 states, N=16 particles
    ("mid", 7, 64),        # ragged B
    ("big", 32, 512),      # the N we use now after raising TPB_PARTICLES
    # The loss kernel is launched with blocks_for(n_rows) and TPB=32, so with
    # B <= 32 it runs in ONE block and the `row >= n_rows` guard is never hit.
    # 70 is three blocks with the last one half full.
    ("multiblock", 70, 64),
]

rng = np.random.default_rng(101)


def make_case(batch, num_particles):
    """Actor logits (already masked), root actions and normalised weights."""
    # Mask: each state has between 2 and 9 free cells.
    mask = np.zeros((batch, NUM_ACTIONS), dtype=np.float32)
    for b in range(batch):
        n_free = rng.integers(2, NUM_ACTIONS + 1)
        free = rng.choice(NUM_ACTIONS, size=n_free, replace=False)
        mask[b, free] = 1.0

    raw = rng.normal(0.0, 1.0, size=(batch, NUM_ACTIONS)).astype(np.float32)
    logits = np.where(mask > 0, raw, NEG_INF).astype(np.float32)

    # The particles can only have LEGAL root actions: that is what the search
    # produces, because it samples from the masked prior.
    actions = np.zeros((num_particles, batch), dtype=np.int32)
    for b in range(batch):
        legal = np.flatnonzero(mask[b])
        actions[:, b] = rng.choice(legal, size=num_particles, replace=True)

    # Per-state normalised weights, like the ones coming out of the readout's
    # softmax.
    w = rng.gamma(1.0, 1.0, size=(num_particles, batch)).astype(np.float32)
    w = (w / w.sum(axis=0, keepdims=True)).astype(np.float32)
    return mask, logits, actions, w


lines = []
for name, batch, num_particles in CASES:
    mask, logits, actions, w = make_case(batch, num_particles)

    # --- particle form: Stoix's function, untouched ---
    dist = distrax.Categorical(logits=jnp.asarray(logits))
    loss_particles = float(
        compute_cross_entropy_loss(jnp.asarray(actions), jnp.asarray(w), dist)
    )

    # --- dense form: q aggregated per action, and then the sum over 9 ---
    q = np.zeros((batch, NUM_ACTIONS), dtype=np.float64)
    for b in range(batch):
        for n in range(num_particles):
            q[b, actions[n, b]] += w[n, b]

    # Stable log_softmax. On the illegal ones it gives -inf, and there q is 0: the
    # term is SKIPPED instead of multiplied, because 0 * -inf = NaN.
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

    # Entropy of q and KL divergence. The cross entropy decomposes into
    #     H(q, pi) = H(q) + KL(q || pi)
    # and H(q) does NOT depend on pi: it is the loss's floor. What training can
    # bring down is only the KL. Reporting the raw cross entropy makes the curve
    # unreadable, because its floor moves when q changes; the KL has a zero with a
    # meaning ("the actor reproduces what the search says").
    entropy = np.zeros(batch, dtype=np.float64)
    for b in range(batch):
        acc = 0.0
        for a in range(NUM_ACTIONS):
            if q[b, a] > 0.0:
                acc += q[b, a] * np.log(q[b, a])
        entropy[b] = -acc
    kl = per_state - entropy
    assert (kl > -1e-9).all(), "la KL no puede ser negativa"

    # THE check: both forms have to give the same number.
    diff = abs(loss_particles - loss_dense)
    assert diff < 2e-5, (
        f"{name}: la forma de particulas ({loss_particles}) y la densa "
        f"({loss_dense}) no coinciden, diff={diff}")
    # And q has to be a distribution over the LEGAL ones.
    assert np.allclose(q.sum(axis=1), 1.0, atol=1e-5), "q no suma 1"
    assert (q[mask == 0] == 0.0).all(), "q pone masa en una casilla ilegal"

    logits.tofile(os.path.join(OUT, f"ce_{name}_logits.bin"))
    mask.tofile(os.path.join(OUT, f"ce_{name}_mask.bin"))
    q.astype(np.float32).tofile(os.path.join(OUT, f"ce_{name}_q.bin"))
    log_pi.astype(np.float32).tofile(os.path.join(OUT, f"ce_{name}_logpi.bin"))
    per_state.astype(np.float32).tofile(os.path.join(OUT, f"ce_{name}_per_state.bin"))
    entropy.astype(np.float32).tofile(os.path.join(OUT, f"ce_{name}_entropy.bin"))
    kl.astype(np.float32).tofile(os.path.join(OUT, f"ce_{name}_kl.bin"))
    np.float32(loss_dense).tofile(os.path.join(OUT, f"ce_{name}_loss.bin"))

    n_illegal = int((mask == 0).sum())
    lines.append(f"{name} batch {batch} particles {num_particles} "
                 f"loss {loss_dense:.8f} illegal_cells {n_illegal} "
                 f"diff_particles_vs_dense {diff:.3e}")
    print(f"  {name:6} B={batch:2} N={num_particles:3}  "
          f"loss={loss_dense:.6f}  |particulas-densa|={diff:.2e}  "
          f"ilegales={n_illegal}  H(q)={entropy.mean():.4f}  KL={kl.mean():.4f}")

with open(os.path.join(OUT, "ce_loss.txt"), "w") as f:
    f.write(f"num_actions {NUM_ACTIONS}\n")
    f.write(f"neg_inf {NEG_INF!r}\n")
    f.write("casos: " + " ".join(n for n, _, _ in CASES) + "\n")
    f.write("ce_<caso>_logits.bin    float32 B x 9  (ya enmascarados con NEG_INF)\n")
    f.write("ce_<caso>_mask.bin      float32 B x 9  (1 legal, 0 ocupada)\n")
    f.write("ce_<caso>_q.bin         float32 B x 9  (la q agregada, suma 1)\n")
    f.write("ce_<caso>_logpi.bin     float32 B x 9  (log_softmax; -inf en ilegales)\n")
    f.write("ce_<caso>_per_state.bin float32 B     (la perdida de cada estado)\n")
    f.write("ce_<caso>_entropy.bin   float32 B     (H(q), el suelo de la perdida)\n")
    f.write("ce_<caso>_kl.bin        float32 B     (KL(q||pi) = perdida - H(q))\n")
    f.write("ce_<caso>_loss.bin      float32 1     (la media sobre el batch)\n")
    f.write("\n")
    f.write("La forma de particulas se calcula con compute_cross_entropy_loss de\n")
    f.write("Stoix (importada) y coincide con la densa; ver la columna diff.\n")
    f.write("\n".join(lines) + "\n")

print("golden de la entropia cruzada escrito en", OUT)
