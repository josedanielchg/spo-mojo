"""Golden for the truncated GAE, with Stoix's REAL function.

Run from mojo_spo's root:
    ../.venv/bin/python tests/golden/gen/gen_gae.py

`batch_truncated_generalized_advantage_estimation` is imported from Stoix rather
than reimplementing the formula: that way the golden verifies that we compute the
SAME thing as the reference, including the details that slip past when reading the
paper.

What it does, in short:

    delta_t = r + gamma*v_t - v_tm1                (one-step TD error)
    acc     = delta + gamma*lambda*acc*(1 - trunc) (accumulated BACKWARDS)
    target  = v_tm1 + acc                          (what the critic learns)

The fine detail is in Stoix's own comment: at a truncation point the accumulator
resets BUT that step's delta is still used. That is, truncating does not erase the
step's information, it cuts the backwards propagation.

How SPO calls it (ff_spo.py, _critic_loss_fn):
    discount_t = (1 - done) * gamma
    v_tm1      = TARGET critic over the observations
    v_t        = TARGET critic over the bootstrap_obs
    truncation = seq.truncated

Three cases:
    case 0   no truncation, with episodes that end (done) midway
    case 1   with truncation in several places: the branch that almost never gets
             tested
    case 2   like real tic-tac-toe: short games (3-5 steps) that always end on
             their own, never truncated
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_default_matmul_precision", "highest")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))
# The root of the Stoix repo, in order to import its real implementation.
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from stoix.utils.multistep import batch_truncated_generalized_advantage_estimation

GAMMA = 0.99        # Stoix's config
LAMBDA = 0.95       # Stoix's config
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
        # Tic-tac-toe: the game ends every 3-5 steps and the reward only arrives
        # at the end (1 / 0.5 / 0). In between, zero.
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
        # A step cannot be terminal and truncated at once: they are different
        # things (one really ended, the other was cut off).
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
          f"{int(trunc.sum()):3} truncations  "
          f"|adv|max={float(jnp.abs(adv).max()):.4f}")


make_case(0, b=4, t=16, done_prob=0.15, trunc_prob=0.0)
make_case(1, b=4, t=16, done_prob=0.10, trunc_prob=0.15)
make_case(2, b=8, t=32, done_prob=0.0, trunc_prob=0.0, ttt_like=True)

with open(os.path.join(OUT, "gae.txt"), "w") as f:
    f.write(f"gamma {GAMMA}\nlambda {LAMBDA}\n")
    f.write("generated with stoix.utils.multistep."
            "batch_truncated_generalized_advantage_estimation\n")
    f.write("discount = (1 - done) * gamma, as in ff_spo._critic_loss_fn\n")
    f.write("gae<case>_{r,discount,v_tm1,v_t,trunc,adv,targets}.bin float32 B x T\n")
    f.write("case0 = no truncation\ncase1 = with truncation\n")
    f.write("case2 = like tic-tac-toe (games of 3-5 steps, reward at the end)\n")
    for line in lines:
        f.write(line + "\n")

print("GAE golden written to", OUT)
