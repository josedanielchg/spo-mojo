"""How the search's kernels get launched: block sizes and checks.

This lives apart because every file of the E-step uses it, and because the two
block sizes are not interchangeable: picking the wrong one is a silent bug, not a
compile error.
"""

from systems.spo.spo_types import SPOConfig

comptime TPB = 32
"""Block size for the kernels that are a flat map: one thread per particle.

Here the row means nothing; all that matters is covering P elements, so the block
size is free and 32 (one warp) works fine."""

comptime TPB_PARTICLES = 512
"""Block size for the kernels whose ROW is the particle dimension (resampling,
ESS, the readout's softmax). There the whole block works over one env's N
particles, so N has to fit inside.

It is 128 and not 32 so that the demo can sweep N = 64. The extra threads cost
nothing: the guards switch them off and the GPU does not launch whole idle warps.
With N = 16 (the paper's) it is far more than enough."""


def blocks_for(n: Int) -> Int:
    """How many blocks of TPB threads are needed to cover n elements.

    It is `(n + TPB - 1) // TPB`, which was written out by hand some fifteen
    times. The rounding up is the reason EVERY map kernel needs its `if i < n`
    guard: almost always more threads are launched than there is data.
    """
    return (n + TPB - 1) // TPB


def check_search_config(cfg: SPOConfig) raises:
    """Checks on the HOST what the kernels cannot check for themselves.

    It exists because this error has already bitten me: with N = 64 and blocks of
    32 the search returned a WORSE policy than with N = 16, without warning about
    anything. The kernel's debug_assert catches it, but only if somebody runs with
    -D ASSERT=all; this check always fires and says exactly what to do."""
    if cfg.num_particles > TPB_PARTICLES:
        raise Error("num_particles=", cfg.num_particles, " no cabe en un bloque de ",
                    TPB_PARTICLES, ". Sube TPB_PARTICLES (potencia de dos) o baja N.")
    if cfg.num_actions > TPB:
        raise Error("num_actions=", cfg.num_actions, " no cabe en un bloque de ", TPB)
