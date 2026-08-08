"""Golden for the actor's backward: equation 11 differentiated with JAX's autodiff.

Run from mojo_spo's root:
    ../.venv/bin/python tests/golden/gen/gen_actor_backward.py

With no autodiff in Mojo, the gradient is written by hand, and by hand one makes
mistakes. Here the SAME function is differentiated with `jax.grad`, which is what
Stoix uses, and the six weight tensors plus the gradient with respect to the logits
are compared.

The full chain being differentiated is:

    x -> linear -> relu -> linear -> relu -> linear -> logits
      -> mask with NEG_INF -> log_softmax -> -SUM_a q(a) log pi(a) -> mean

The gradient with respect to the logits comes out analytically as

    dL/dz = (pi - q) / batch

and the golden stores it separately so as to localise a fault: if `dz` lines up but
`dW1` does not, the problem is in the network; if `dz` already fails, it is in the
loss.

Two details that have to be reproduced exactly or the comparison is worthless:

1. **The masking goes INSIDE what is differentiated.** The mask overwrites the
   logits with a constant, so the gradient with respect to those logits has to be 0
   and nothing must reach the weights that way. It is implemented with
   `jnp.where(mask, z, NEG_INF)`, which is what the kernel does.

2. **The mean is over the BATCH**, not over batch*actions. The 1/m factor has to be
   the same on both sides or the gradients differ by a constant and the test would
   blame it on a bug in the backward.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp

# Without this JAX would use TF32 for the matmuls and the golden would come out
# LESS precise than the kernel it is meant to verify. It happened to us in E1.4 and
# cost a session.
jax.config.update("jax_default_matmul_precision", "highest")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

IN_DIM = 18
NUM_ACTIONS = 9
NEG_INF = np.float32(np.finfo(np.float32).min)

CASES = [("bw_small", 5, 32), ("bw_big", 40, 64)]   # (name, batch, hidden)

rng = np.random.default_rng(7)


def he(fan_in, fan_out):
    return rng.normal(0.0, 1.0 / np.sqrt(fan_in),
                      size=(fan_in, fan_out)).astype(np.float32)


def make_inputs(batch):
    """Boards in network format, their legal mask, and a consistent q."""
    x = np.zeros((batch, IN_DIM), dtype=np.float32)
    mask = np.zeros((batch, NUM_ACTIONS), dtype=np.float32)
    q = np.zeros((batch, NUM_ACTIONS), dtype=np.float32)
    for b in range(batch):
        while True:
            x[b] = 0.0
            for c in range(9):
                who = rng.integers(0, 3)
                if who == 1:
                    x[b, c] = 1.0
                elif who == 2:
                    x[b, 9 + c] = 1.0
            free = 1.0 - (x[b, :9] + x[b, 9:])
            if free.sum() >= 2:
                mask[b] = free
                break
        # q only puts mass on the legal ones, like the one coming out of the
        # search. Some legal cell is deliberately left at zero: it is the q(a)=0
        # case with a finite log pi, different from the illegal one.
        legal = np.flatnonzero(mask[b])
        w = rng.gamma(1.0, 1.0, size=len(legal))
        if len(legal) > 2:
            w[rng.integers(0, len(legal))] = 0.0
        q[b, legal] = (w / w.sum()).astype(np.float32)
    return x, mask, q


def loss_fn(params, x, mask, q):
    """Equation 11, exactly as Mojo computes it."""
    w1, b1, w2, b2, w3, b3 = params
    a1 = jnp.maximum(x @ w1 + b1, 0.0)
    a2 = jnp.maximum(a1 @ w2 + b2, 0.0)
    z = a2 @ w3 + b3
    z = jnp.where(mask > 0, z, NEG_INF)
    log_pi = z - jax.scipy.special.logsumexp(z, axis=-1, keepdims=True)
    # The terms with q = 0 are skipped: 0 * (-inf) is NaN. A `jnp.where` over the
    # PRODUCT is not enough (the NaN sneaks in through the gradient), so log_pi is
    # cleaned before multiplying.
    safe_log_pi = jnp.where(q > 0, log_pi, 0.0)
    per_state = -jnp.sum(q * safe_log_pi, axis=-1)
    return jnp.mean(per_state)


lines = []
for name, batch, hidden in CASES:
    params = [he(IN_DIM, hidden),
              rng.normal(0.0, 0.1, size=(hidden,)).astype(np.float32),
              he(hidden, hidden),
              rng.normal(0.0, 0.1, size=(hidden,)).astype(np.float32),
              he(hidden, NUM_ACTIONS),
              rng.normal(0.0, 0.1, size=(NUM_ACTIONS,)).astype(np.float32)]
    x, mask, q = make_inputs(batch)

    jp = [jnp.asarray(p) for p in params]
    jx, jm, jq = jnp.asarray(x), jnp.asarray(mask), jnp.asarray(q)

    loss = float(loss_fn(jp, jx, jm, jq))
    grads = jax.grad(loss_fn)(jp, jx, jm, jq)

    # The gradient with respect to the logits, separately.
    def loss_from_z(z):
        z = jnp.where(jm > 0, z, NEG_INF)
        log_pi = z - jax.scipy.special.logsumexp(z, axis=-1, keepdims=True)
        safe = jnp.where(jq > 0, log_pi, 0.0)
        return jnp.mean(-jnp.sum(jq * safe, axis=-1))

    a1 = np.maximum(x @ params[0] + params[1], 0.0)
    a2 = np.maximum(a1 @ params[2] + params[3], 0.0)
    z_raw = (a2 @ params[4] + params[5]).astype(np.float32)
    dz = np.asarray(jax.grad(loss_from_z)(jnp.asarray(z_raw)), dtype=np.float32)

    # And the analytic form, (pi - q)/batch, to check they agree.
    zm = np.where(mask > 0, z_raw, NEG_INF)
    e = np.exp((zm - zm.max(axis=1, keepdims=True)).astype(np.float64))
    pi = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
    dz_analytic = np.where(mask > 0, (pi - q) / batch, 0.0).astype(np.float32)
    dz_diff = float(np.abs(dz - dz_analytic).max())
    assert dz_diff < 1e-7, f"{name}: (pi-q)/m no coincide con autodiff: {dz_diff}"
    # And on the illegal ones the autodiff has to give exactly 0.
    assert (np.abs(dz[mask == 0]) == 0.0).all(), \
        f"{name}: el gradiente se cuela por una casilla enmascarada"

    for arr, nm in [(x, "x"), (mask, "mask"), (q, "q"), (pi, "pi"),
                    (z_raw, "z"), (dz, "dz")]:
        arr.astype(np.float32).tofile(os.path.join(OUT, f"{name}_{nm}.bin"))
    for p, nm in zip(params, ["w1", "b1", "w2", "b2", "w3", "b3"]):
        p.tofile(os.path.join(OUT, f"{name}_{nm}.bin"))
    for g, nm in zip(grads, ["dw1", "db1", "dw2", "db2", "dw3", "db3"]):
        np.asarray(g, dtype=np.float32).tofile(
            os.path.join(OUT, f"{name}_{nm}.bin"))
    np.float32(loss).tofile(os.path.join(OUT, f"{name}_loss.bin"))

    gmax = max(float(np.abs(np.asarray(g)).max()) for g in grads)
    lines.append(f"{name} batch {batch} hidden {hidden} loss {loss:.8f} "
                 f"max|grad| {gmax:.6f} dz_vs_analytic {dz_diff:.2e}")
    print(f"  {name:9} B={batch:2} H={hidden:3}  loss={loss:.6f}  "
          f"max|grad|={gmax:.5f}  |dz - (pi-q)/m|={dz_diff:.1e}")

with open(os.path.join(OUT, "actor_backward.txt"), "w") as f:
    f.write(f"in_dim {IN_DIM}\nnum_actions {NUM_ACTIONS}\n")
    f.write("casos: " + " ".join(n for n, _, _ in CASES) + "\n")
    f.write("<caso>_{w1,b1,w2,b2,w3,b3}.bin  los pesos\n")
    f.write("<caso>_{dw1,db1,dw2,db2,dw3,db3}.bin  sus gradientes (jax.grad)\n")
    f.write("<caso>_x.bin float32 B x 18   observaciones\n")
    f.write("<caso>_mask.bin float32 B x 9 (1 legal, 0 ocupada)\n")
    f.write("<caso>_q.bin float32 B x 9    la politica objetivo\n")
    f.write("<caso>_pi.bin float32 B x 9   la politica de la red\n")
    f.write("<caso>_z.bin float32 B x 9    los logits SIN enmascarar\n")
    f.write("<caso>_dz.bin float32 B x 9   dL/dlogits = (pi-q)/B, 0 en ilegales\n")
    f.write("<caso>_loss.bin float32 1\n")
    f.write("\n".join(lines) + "\n")

print("golden del backward del actor escrito en", OUT)
