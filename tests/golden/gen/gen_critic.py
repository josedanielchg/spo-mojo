"""Generates the golden for the critic's MLP: 18 -> H -> H -> 1 with ReLU.

Run from mojo_spo's root:
    ../.venv/bin/python tests/golden/gen/gen_critic.py

THREE network sizes, and that is a decision with a history: the size is fixed
neither by the paper nor by Stoix. Stoix uses `layer_sizes: [256, 256]` in its MLP
config, and the paper uses large ResNets (256-512 channels) for ITS environments,
which are far harder (Sokoban, Rubik, Brax). Tic-tac-toe has 18 inputs and ~5500
valid positions, so "how much network is needed?" gets answered by measuring:

    H = 32    1697 weights
    H = 64    5441 weights
    H = 256  70913 weights    (Stoix's size)

Two batches per architecture:
    M = 5    ragged (not a multiple of the tile of 16): it hits the guards
    M = 64   several tiles along the batch dimension

The intermediate activations (a1, a2) are stored TOO, not just the output: if the
test fails, comparing layer by layer says which one broke.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

IN_DIM = 18
OUT_DIM = 1
HIDDENS = [32, 64, 256]
BATCHES = [5, 64]

rng = np.random.default_rng(11)


def he_init(fan_in, fan_out):
    """He-style initialisation: scale 1/sqrt(fan_in) so that the activations
    neither grow nor die out when stacking ReLU layers."""
    return rng.normal(0.0, 1.0 / np.sqrt(fan_in),
                      size=(fan_in, fan_out)).astype(np.float32)


def make_boards(m):
    """m boards that look like the real ones: two 0/1 planes, and never the same
    cell in both (it cannot be mine and theirs at once)."""
    x = np.zeros((m, IN_DIM), dtype=np.float32)
    for r in range(m):
        for c in range(9):
            who = rng.integers(0, 3)      # 0 empty, 1 mine, 2 theirs
            if who == 1:
                x[r, c] = 1.0
            elif who == 2:
                x[r, 9 + c] = 1.0
    return x


lines = []
for hidden in HIDDENS:
    tag = f"h{hidden}"
    w1 = he_init(IN_DIM, hidden)
    b1 = rng.normal(0.0, 0.1, size=(hidden,)).astype(np.float32)
    w2 = he_init(hidden, hidden)
    b2 = rng.normal(0.0, 0.1, size=(hidden,)).astype(np.float32)
    w3 = he_init(hidden, OUT_DIM)
    b3 = rng.normal(0.0, 0.1, size=(OUT_DIM,)).astype(np.float32)

    for name, arr in [("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2),
                      ("w3", w3), ("b3", b3)]:
        arr.tofile(os.path.join(OUT, f"critic_{tag}_{name}.bin"))

    n_params = (IN_DIM * hidden + hidden * hidden + hidden * OUT_DIM
                + 2 * hidden + OUT_DIM)

    for m in BATCHES:
        x = make_boards(m)
        a1 = np.maximum(x @ w1 + b1, 0.0).astype(np.float32)
        a2 = np.maximum(a1 @ w2 + b2, 0.0).astype(np.float32)
        v = (a2 @ w3 + b3).astype(np.float32)

        x.tofile(os.path.join(OUT, f"critic_{tag}_x{m}.bin"))
        a1.tofile(os.path.join(OUT, f"critic_{tag}_a1_{m}.bin"))
        a2.tofile(os.path.join(OUT, f"critic_{tag}_a2_{m}.bin"))
        v.tofile(os.path.join(OUT, f"critic_{tag}_v{m}.bin"))

        # Fraction of neurons the ReLU leaves at zero. ~50% is healthy with this
        # initialisation; 0% would mean the network is effectively linear and
        # ~100% that it is dead (zero gradient).
        dead1 = float((a1 == 0).mean())
        dead2 = float((a2 == 0).mean())
        lines.append(f"hidden {hidden} batch {m}  pesos {n_params}  "
                     f"relu_apagadas {dead1:.3f} / {dead2:.3f}")
        print(f"  hidden {hidden:3} batch {m:2}: {n_params:6} pesos, "
              f"V en [{v.min():7.4f}, {v.max():7.4f}], "
              f"ReLU apagadas {dead1:.1%} / {dead2:.1%}")

with open(os.path.join(OUT, "critic.txt"), "w") as f:
    f.write(f"in_dim {IN_DIM}\nout_dim {OUT_DIM}\n")
    f.write(f"hiddens {' '.join(str(h) for h in HIDDENS)}\n")
    f.write(f"batches {' '.join(str(b) for b in BATCHES)}\n")
    f.write("los ficheros llevan el tag h<HIDDEN>:\n")
    f.write("critic_h<H>_w1.bin float32 in_dim x H\n")
    f.write("critic_h<H>_b1.bin float32 H\n")
    f.write("critic_h<H>_w2.bin float32 H x H\n")
    f.write("critic_h<H>_b2.bin float32 H\n")
    f.write("critic_h<H>_w3.bin float32 H x out_dim\n")
    f.write("critic_h<H>_b3.bin float32 out_dim\n")
    f.write("critic_h<H>_x<M>.bin   float32 M x in_dim  (tableros: dos planos 0/1)\n")
    f.write("critic_h<H>_a1_<M>.bin float32 M x H       (relu de la capa 1)\n")
    f.write("critic_h<H>_a2_<M>.bin float32 M x H       (relu de la capa 2)\n")
    f.write("critic_h<H>_v<M>.bin   float32 M x out_dim (la salida V)\n")
    for line in lines:
        f.write(line + "\n")

print("golden del critico escrito en", OUT)
