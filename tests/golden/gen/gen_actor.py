"""Generates the golden for the ACTOR's MLP: 18 -> H -> H -> 9, with masking.

Run from mojo_spo's root:
    ../.venv/bin/python tests/golden/gen/gen_actor.py

The network is the SAME as the critic's except for the output (9 logits instead of
1 value), so the forward reuses the code already verified in E1.3. What this golden
really adds is the new part: **the masking of illegal cells**.

Why the masking needs its own golden and "we already tested it in the prior" is not
enough: in the search's prior the legal logits were all 0, so masking and then
taking a softmax gave a uniform, and any indexing error would have looked just as
uniform. Here the logits are numbers different from one another coming out of a
network, so an indexing fault (masking the wrong cell) changes the distribution
detectably.

NEG_INF = the most negative finite float32 is used, not -inf, for the same reason
as `ttt_prior_logits_kernel`: a whole masked row (full board) would give nan with
-inf, and with MIN_FINITE it degenerates to uniform, which is harmless. The golden
reproduces that choice so that the test compares against what the kernel really
does.

Three widths, as in the critic: the network's size is fixed neither by the paper
nor by Stoix, so it gets measured.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

IN_DIM = 18
OUT_DIM = 9
HIDDENS = [32, 64, 256]
BATCHES = [5, 64]

# The same value envs/tictactoe.mojo uses: the most negative finite float32.
NEG_INF = np.float32(np.finfo(np.float32).min)

rng = np.random.default_rng(29)


def he_init(fan_in, fan_out):
    """He-style initialisation, same as in the critic."""
    return rng.normal(0.0, 1.0 / np.sqrt(fan_in),
                      size=(fan_in, fan_out)).astype(np.float32)


def make_boards(m):
    """m boards in network format (two 0/1 planes) and their legal mask.

    The mask comes from the SAME board: a cell is legal if it is in neither plane.
    That way the test also checks that the mask and the observation are consistent,
    which is exactly what would break if somebody touched the encoding.

    It is forced that no row is left without legal cells: a full board is a
    finished game and the actor does not get asked there. The degenerate case is
    tested separately, by hand, in the test.
    """
    x = np.zeros((m, IN_DIM), dtype=np.float32)
    mask = np.zeros((m, OUT_DIM), dtype=np.float32)
    for r in range(m):
        while True:
            x[r] = 0.0
            for c in range(9):
                who = rng.integers(0, 3)      # 0 empty, 1 mine, 2 theirs
                if who == 1:
                    x[r, c] = 1.0
                elif who == 2:
                    x[r, 9 + c] = 1.0
            free = 1.0 - (x[r, :9] + x[r, 9:])
            if free.sum() > 0:
                mask[r] = free
                break
    return x, mask


def masked_softmax(logits, mask):
    """softmax over the logits with the illegal cells set to NEG_INF.

    The row's maximum is subtracted before exponentiating (stable softmax), which
    is what `softmax_rows` does in Mojo. Reproducing it matters: with NEG_INF and
    without subtracting the maximum, exp() would underflow to 0 across the whole
    row.
    """
    masked = np.where(mask > 0, logits, NEG_INF).astype(np.float32)
    shifted = masked - masked.max(axis=1, keepdims=True)
    e = np.exp(shifted.astype(np.float64))
    return masked, (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


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
        arr.tofile(os.path.join(OUT, f"actor_{tag}_{name}.bin"))

    n_params = (IN_DIM * hidden + hidden * hidden + hidden * OUT_DIM
                + 2 * hidden + OUT_DIM)

    for m in BATCHES:
        x, mask = make_boards(m)
        a1 = np.maximum(x @ w1 + b1, 0.0).astype(np.float32)
        a2 = np.maximum(a1 @ w2 + b2, 0.0).astype(np.float32)
        raw = (a2 @ w3 + b3).astype(np.float32)
        masked, probs = masked_softmax(raw, mask)

        x.tofile(os.path.join(OUT, f"actor_{tag}_x{m}.bin"))
        mask.tofile(os.path.join(OUT, f"actor_{tag}_mask{m}.bin"))
        raw.tofile(os.path.join(OUT, f"actor_{tag}_raw{m}.bin"))
        masked.tofile(os.path.join(OUT, f"actor_{tag}_masked{m}.bin"))
        probs.tofile(os.path.join(OUT, f"actor_{tag}_probs{m}.bin"))

        # The generator's own checks: if the golden were wrong, the Mojo test would
        # pass it as good and we would never find out.
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), "las filas no suman 1"
        assert (probs[mask == 0] == 0.0).all(), "una casilla ilegal tiene masa"
        legal_per_row = mask.sum(axis=1)
        lines.append(f"hidden {hidden} batch {m}  pesos {n_params}  "
                     f"legales/fila min {legal_per_row.min():.0f} "
                     f"max {legal_per_row.max():.0f}")
        print(f"  hidden {hidden:3} batch {m:2}: {n_params:6} pesos, "
              f"logits en [{raw.min():7.4f}, {raw.max():7.4f}], "
              f"legales/fila {legal_per_row.min():.0f}-{legal_per_row.max():.0f}, "
              f"p_max {probs.max():.4f}")

with open(os.path.join(OUT, "actor.txt"), "w") as f:
    f.write(f"in_dim {IN_DIM}\nout_dim {OUT_DIM}\n")
    f.write(f"hiddens {' '.join(str(h) for h in HIDDENS)}\n")
    f.write(f"batches {' '.join(str(b) for b in BATCHES)}\n")
    f.write(f"neg_inf {NEG_INF!r}\n")
    f.write("los ficheros llevan el tag h<HIDDEN>:\n")
    f.write("actor_h<H>_w1.bin      float32 in_dim x H\n")
    f.write("actor_h<H>_b1.bin      float32 H\n")
    f.write("actor_h<H>_w2.bin      float32 H x H\n")
    f.write("actor_h<H>_b2.bin      float32 H\n")
    f.write("actor_h<H>_w3.bin      float32 H x out_dim\n")
    f.write("actor_h<H>_b3.bin      float32 out_dim\n")
    f.write("actor_h<H>_x<M>.bin      float32 M x in_dim  (dos planos 0/1)\n")
    f.write("actor_h<H>_mask<M>.bin   float32 M x out_dim (1 legal, 0 ocupada)\n")
    f.write("actor_h<H>_raw<M>.bin    float32 M x out_dim (logits SIN enmascarar)\n")
    f.write("actor_h<H>_masked<M>.bin float32 M x out_dim (logits enmascarados)\n")
    f.write("actor_h<H>_probs<M>.bin  float32 M x out_dim (softmax enmascarado)\n")
    f.write("\n".join(lines) + "\n")

print("golden del actor escrito en", OUT)
