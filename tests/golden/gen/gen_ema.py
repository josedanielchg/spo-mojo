"""Golden de la EMA de los target networks, con OPTAX (el de Stoix).

Correr desde la raiz de mojo_spo:
    ../.venv/bin/python tests/golden/gen/gen_ema.py

Stoix usa `optax.incremental_update(online, target, tau)` con tau = 0.005
(verificado en ff_spo.py, lineas 1517-1522). Se genera con la libreria de verdad
en vez de reimplementar la formula, para que el test compruebe que hacemos lo
mismo y no que sabemos copiar una ecuacion.

Tamano a proposito MAYOR que un bloque (1000 > 256 hilos): el test anterior solo
usaba 4 y 5 elementos, o sea un unico bloque, y la ruta multi-bloque no se
ejercitaba. Es el mismo punto ciego que aparecio en E1.4.

Diez pasos, para ver la convergencia lenta: tras k pasos el target vale
1 - (1-tau)^k del camino hacia el online.
"""

import os

import jax.numpy as jnp
import numpy as np
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

N = 1000          # > TPB_OPT (256): varios bloques
TAU = 0.005       # config de Stoix
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
