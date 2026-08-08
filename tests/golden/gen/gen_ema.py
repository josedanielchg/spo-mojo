"""Golden for the target networks' EMA, with OPTAX (Stoix's).

Run from mojo_spo's root:
    ../.venv/bin/python tests/golden/gen/gen_ema.py

Stoix uses `optax.incremental_update(online, target, tau)` with tau = 0.005
(verified in ff_spo.py, lines 1517-1522). It is generated with the real library
rather than reimplementing the formula, so that the test checks that we do the
same thing and not that we can copy an equation.

The size is deliberately LARGER than one block (1000 > 256 threads): the previous
test only used 4 and 5 elements, that is, a single block, and the multi-block path
was not exercised. It is the same blind spot that turned up in E1.4.

Ten steps, to see the slow convergence: after k steps the target has covered
1 - (1-tau)^k of the way towards the online one.
"""

import os

import jax.numpy as jnp
import numpy as np
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

N = 1000          # > TPB_OPT (256): several blocks
TAU = 0.005       # Stoix's config
STEPS = 10

rng = np.random.default_rng(67)
target = rng.normal(0, 1, (N,)).astype(np.float32)
online = rng.normal(0, 1, (N,)).astype(np.float32)

target.tofile(os.path.join(OUT, "ema_target0.bin"))
online.tofile(os.path.join(OUT, "ema_online.bin"))

t = jnp.asarray(target)
o = jnp.asarray(online)
for step in range(1, STEPS + 1):
    t = optax.incremental_update(o, t, TAU)
    np.asarray(t, dtype=np.float32).tofile(
        os.path.join(OUT, f"ema_target{step}.bin"))

frac = 1 - (1 - TAU) ** STEPS
print(f"  N={N} (>{256}, o sea varios bloques), tau={TAU}, {STEPS} pasos")
print(f"  tras {STEPS} pasos el target recorrio el {frac:.2%} del camino")

with open(os.path.join(OUT, "ema.txt"), "w") as f:
    f.write(f"n {N}\ntau {TAU}\nsteps {STEPS}\n")
    f.write("generado con optax.incremental_update(online, target, tau)\n")
    f.write("ema_target0.bin  target inicial\n")
    f.write("ema_online.bin   online (fijo)\n")
    f.write("ema_target<k>.bin target tras k pasos\n")
print("golden de la EMA escrito en", OUT)
