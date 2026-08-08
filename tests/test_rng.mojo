"""RNG and categorical sampling.

The sampling is tested in two complementary ways:
  - with uniforms INJECTED by hand -> exact indices, zero tolerances. If this
    fails, the inverse CDF is wrong.
  - with the real RNG -> frequencies close to the probabilities. If this fails but
    the previous one passes, the problem is in the generator, not in the sampling.

Splitting it this way is what makes a failure say WHERE to look.
"""

from std.gpu.host import DeviceContext
from std.math import abs, log

from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform, categorical_from_logits, rand_uniform, RngKey
from tests.helpers import upload, zeros, filled, download, assert_eq_int

comptime TPB = 32


def log_probs_rows(probs: List[Scalar[dtype]], rows: Int) -> List[Scalar[dtype]]:
    """Repeats log(p) across `rows` rows. The kernel's softmax undoes the log and
    recovers p, so the probabilities come out exactly as requested."""
    out = List[Scalar[dtype]]()
    for _ in range(rows):
        for c in range(len(probs)):
            out.append(log(probs[c]))
    return out^


def make_probs() -> List[Scalar[dtype]]:
    """[0.1, 0.2, 0.3, 0.4]: unequal on purpose, so that a sampler that always
    returned the same thing (or uniform) would show up."""
    p = List[Scalar[dtype]]()
    p.append(0.1)
    p.append(0.2)
    p.append(0.3)
    p.append(0.4)
    return p^


def test_determinism() raises:
    """Same key -> same sequence. Different streams -> different sequences."""
    for i in range(64):
        if rand_uniform(42, 0, UInt32(i)) != rand_uniform(42, 0, UInt32(i)):
            raise Error("rand_uniform no es determinista en i=", i)

    # Two streams from the same seed have to decorrelate. I allow up to 2 out of 64
    # coincidences by pure chance; in practice 0 come out.
    same = 0
    for i in range(64):
        if rand_uniform(42, 0, UInt32(i)) == rand_uniform(42, 1, UInt32(i)):
            same += 1
    if same > 2:
        raise Error("los streams 0 y 1 se parecen demasiado: ", same, "/64 iguales")

    key = RngKey(42, 0)
    if key.split(1).stream == key.split(2).stream:
        raise Error("split(1) y split(2) dieron el mismo stream")
    print("PASS rng determinista y streams independientes")


def test_uniform_stats(ctx: DeviceContext) raises:
    """Mean ~ 0.5 and variance ~ 1/12 = 0.0833, which is what U(0,1) must give."""
    n = 100000
    o = zeros[dtype](ctx, n)

    blocks = (n + TPB - 1) // TPB
    ctx.enqueue_function[fill_uniform, fill_uniform](
        o.unsafe_ptr(), UInt32(7), UInt32(0), n, grid_dim=blocks, block_dim=TPB)
    ctx.synchronize()

    got = download[dtype](o, n)

    # I accumulate in float64: summing 100k floats in float32 loses enough
    # precision to move the mean in the third decimal.
    total = Float64(0)
    for i in range(n):
        v = Float64(got[i])
        if v < 0.0 or v >= 1.0:
            raise Error("uniforme fuera de [0,1) en ", i, ": ", got[i])
        total += v
    mean = total / Float64(n)

    variance = Float64(0)
    for i in range(n):
        d = Float64(got[i]) - mean
        variance += d * d
    variance /= Float64(n)

    if abs(mean - 0.5) > 0.01:
        raise Error("media fuera de rango: ", mean)
    if abs(variance - 1.0 / 12.0) > 0.005:
        raise Error("varianza fuera de rango: ", variance)
    print("PASS uniformes: media", mean, "varianza", variance)


def test_categorical_exact(ctx: DeviceContext) raises:
    """Injected uniforms -> exact indices, computed by hand.

    With probabilities [0.1, 0.2, 0.3, 0.4] the CDF is [0.1, 0.3, 0.6, 1.0]. I test
    the interior of each stretch and both edges.
    """
    probs = make_probs()
    row_size = len(probs)

    us = List[Scalar[dtype]]()
    want = List[Int]()
    us.append(0.0);   want.append(0)   # exact lower edge
    us.append(0.05);  want.append(0)   # inside [0.0, 0.1)
    us.append(0.15);  want.append(1)   # inside [0.1, 0.3)
    us.append(0.45);  want.append(2)   # inside [0.3, 0.6)
    us.append(0.75);  want.append(3)   # inside [0.6, 1.0)
    us.append(0.999); want.append(3)   # right up against the upper edge
    rows = len(us)

    logits = upload[dtype](ctx, log_probs_rows(probs, rows))
    u = upload[dtype](ctx, us)
    # Output at -1: if the kernel wrote nothing, it would show.
    o = filled[idx_dtype](ctx, rows, -1)

    ctx.enqueue_function[categorical_from_logits[TPB], categorical_from_logits[TPB]](
        o.unsafe_ptr(), logits.unsafe_ptr(), u.unsafe_ptr(), row_size,
        grid_dim=rows, block_dim=TPB)
    ctx.synchronize()

    got = download[idx_dtype](o, rows)
    for r in range(rows):
        assert_eq_int(Int(got[r]), want[r], String("u=", us[r]))
    print("PASS categorical con uniformes inyectados (indices exactos)")


def test_categorical_frequencies(ctx: DeviceContext) raises:
    """With the real RNG, the frequencies have to approach the probabilities."""
    probs = make_probs()
    row_size = len(probs)
    n = 100000

    logits = upload[dtype](ctx, log_probs_rows(probs, n))
    u = zeros[dtype](ctx, n)
    o = filled[idx_dtype](ctx, n, -1)

    blocks = (n + TPB - 1) // TPB
    ctx.enqueue_function[fill_uniform, fill_uniform](
        u.unsafe_ptr(), UInt32(1234), UInt32(5), n, grid_dim=blocks, block_dim=TPB)
    ctx.enqueue_function[categorical_from_logits[TPB], categorical_from_logits[TPB]](
        o.unsafe_ptr(), logits.unsafe_ptr(), u.unsafe_ptr(), row_size,
        grid_dim=n, block_dim=TPB)
    ctx.synchronize()

    got = download[idx_dtype](o, n)

    counts = List[Int]()
    for _ in range(row_size):
        counts.append(0)
    for i in range(n):
        k = Int(got[i])
        if k < 0 or k >= row_size:
            raise Error("indice invalido en ", i, ": ", k, " (el kernel no escribio?)")
        counts[k] += 1

    chi2 = Float64(0)
    for c in range(row_size):
        freq = Float64(counts[c]) / Float64(n)
        expected = Float64(probs[c])
        if abs(freq - expected) > 0.01:
            raise Error("categoria ", c, ": freq=", freq, " esperada=", expected)
        e = expected * Float64(n)
        d = Float64(counts[c]) - e
        chi2 += d * d / e

    # 3 degrees of freedom: 16.3 is p=0.001. With the seed fixed this is not flaky,
    # it always comes out the same; the threshold is there in case I touch the
    # generator.
    if chi2 > 16.3:
        raise Error("chi2 demasiado alto: ", chi2)
    print("PASS categorical: frecuencias ~ probabilidades (chi2 =", chi2, ")")


def main() raises:
    test_determinism()
    with DeviceContext() as ctx:
        test_uniform_stats(ctx)
        test_categorical_exact(ctx)
        test_categorical_frequencies(ctx)
