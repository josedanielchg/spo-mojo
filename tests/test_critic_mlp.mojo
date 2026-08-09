"""The critic's MLP against the numpy golden, layer by layer.

Golden: tests/golden/gen/gen_critic.py, architecture 18 -> H -> H -> 1 with ReLU
and three sizes: H = 32 (1697 weights), 64 (5441) and 256 (70913, Stoix's).

The INTERMEDIATE activations are compared, not just the output. If only V were
compared and it failed, there would be no telling which layer broke; comparing a1,
a2 and V localises the fault.

Each architecture is tested with two batches sharing weights, so it also checks
that the result does not depend on the batch size: 5 is not a multiple of the tile
of 16 and hits the guards, 64 spans several tiles.
"""

from std.gpu.host import DeviceContext
from std.math import abs, sqrt

from ops.common import dtype
from networks.mlp import (CriticParams, CriticCache, critic_forward,
                          zero_critic_params, relu)
from tests.golden_io import read_f32
from tests.helpers import upload, zeros, download, write_into

comptime GOLDEN = String("tests/golden/")
comptime IN_DIM = 18
comptime OUT_DIM = 1


def tag(hidden: Int) -> String:
    """Prefix of that architecture's golden files."""
    return GOLDEN + "critic_h" + String(hidden) + "_"


def load_params(ctx: DeviceContext, hidden: Int) raises -> CriticParams:
    """The golden weights of the given architecture, uploaded to the GPU."""
    t = tag(hidden)
    p = zero_critic_params(ctx, IN_DIM, hidden, OUT_DIM)
    write_into[dtype](p.w1, read_f32(t + "w1.bin"))
    write_into[dtype](p.b1, read_f32(t + "b1.bin"))
    write_into[dtype](p.w2, read_f32(t + "w2.bin"))
    write_into[dtype](p.b2, read_f32(t + "b2.bin"))
    write_into[dtype](p.w3, read_f32(t + "w3.bin"))
    write_into[dtype](p.b3, read_f32(t + "b3.bin"))
    return p^


def compare(got: List[Scalar[dtype]], want: List[Scalar[dtype]], n: Int,
            tol: Scalar[dtype], what: String) raises -> Scalar[dtype]:
    """Largest absolute difference; blows up if it exceeds the tolerance."""
    worst = Scalar[dtype](0)
    at = 0
    for i in range(n):
        d = abs(got[i] - want[i])
        if d > worst:
            worst = d
            at = i
    if worst > tol:
        raise Error(what, ": the largest difference is ", worst, " at index ", at,
                    " (got=", got[at], " want=", want[at], ", tol ", tol, ")")
    return worst


def check_batch(ctx: DeviceContext, hidden: Int, m: Int) raises:
    """One (architecture, batch) from the golden, comparing all three stages."""
    params = load_params(ctx, hidden)
    cache = CriticCache(ctx, m, hidden, OUT_DIM)
    t = tag(hidden)

    x = read_f32(t + "x" + String(m) + ".bin")
    want_a1 = read_f32(t + "a1_" + String(m) + ".bin")
    want_a2 = read_f32(t + "a2_" + String(m) + ".bin")
    want_v = read_f32(t + "v" + String(m) + ".bin")
    if len(x) != m * IN_DIM:
        raise Error("the input golden for batch ", m, " does not add up")

    critic_forward(ctx, params, cache, upload[dtype](ctx, x), m)
    ctx.synchronize()

    got_a1 = download[dtype](cache.a1, m * hidden)
    got_a2 = download[dtype](cache.a2, m * hidden)
    got_v = download[dtype](cache.value, m * OUT_DIM)

    # The tolerance grows with depth: each layer's error feeds into the next. It
    # scales with sqrt(fan_in) as in the linear layer's test.
    t1 = Scalar[dtype](1e-5) * sqrt(Scalar[dtype](IN_DIM))
    t2 = Scalar[dtype](2e-5) * sqrt(Scalar[dtype](hidden))
    t3 = Scalar[dtype](3e-5) * sqrt(Scalar[dtype](hidden))

    e1 = compare(got_a1, want_a1, m * hidden, t1,
                 String("h", hidden, " batch ", m, " layer 1"))
    e2 = compare(got_a2, want_a2, m * hidden, t2,
                 String("h", hidden, " batch ", m, " layer 2"))
    e3 = compare(got_v, want_v, m * OUT_DIM, t3,
                 String("h", hidden, " batch ", m, " V output"))

    print("      hidden", hidden, " batch", m,
          " error max: a1", e1, " a2", e2, " V", e3)


def test_against_numpy(ctx: DeviceContext) raises:
    """The three architectures x the two batches, layer by layer.

    All three run with the SAME code: the MLP is generic in its dimensions, so
    going from 32 to 256 does not touch a single line of Mojo. Several sizes are
    compared because the number of neurons is fixed neither by the paper nor by
    Stoix (Stoix uses [256,256]); for a game of 9 cells it has to be measured.
    """
    hiddens = List[Int]()
    hiddens.append(32); hiddens.append(64); hiddens.append(256)
    for i in range(len(hiddens)):
        check_batch(ctx, hiddens[i], 5)     # ragged: 5 is not a multiple of 16
        check_batch(ctx, hiddens[i], 64)    # several tiles across the batch
    print("PASS the MLP matches numpy on all 3 architectures (a1, a2 and V)")


def test_relu_actually_fires(ctx: DeviceContext) raises:
    """A check that the test above tests something.

    If the ReLU never clipped anything, the golden would pass all the same and we
    could not tell a network WITH a ReLU from one without. The generator reports
    ~50% of activations switched off; here it is verified on the real output: there
    have to be zeros in a1 and a2, and no negative value at all.
    """
    m = 64
    hidden = 64
    params = load_params(ctx, hidden)
    cache = CriticCache(ctx, m, hidden, OUT_DIM)
    x = read_f32(tag(hidden) + "x64.bin")
    critic_forward(ctx, params, cache, upload[dtype](ctx, x), m)
    ctx.synchronize()

    a1 = download[dtype](cache.a1, m * hidden)
    a2 = download[dtype](cache.a2, m * hidden)

    zeros1 = 0
    for i in range(m * hidden):
        if a1[i] < Scalar[dtype](0):
            raise Error("a1 has a negative value at ", i, ": ", a1[i])
        if a1[i] == Scalar[dtype](0):
            zeros1 += 1
    zeros2 = 0
    for i in range(m * hidden):
        if a2[i] < Scalar[dtype](0):
            raise Error("a2 has a negative value at ", i, ": ", a2[i])
        if a2[i] == Scalar[dtype](0):
            zeros2 += 1

    total = m * hidden
    if zeros1 == 0 or zeros2 == 0:
        raise Error("the ReLU clipped nothing: the golden could not tell a network "
                    "with a ReLU from one without")
    print("      switched off by the ReLU: layer1", Scalar[dtype](zeros1) / Scalar[dtype](total),
          " capa2", Scalar[dtype](zeros2) / Scalar[dtype](total))
    print("PASS the ReLU really clips (there are zeros and no negatives)")


def test_relu_is_exact(ncx: DeviceContext) raises:
    """The ReLU on its own, over chosen values: negatives to 0, the rest untouched.

    It includes 0 and very small values on both sides, which is where a badly
    written comparison (`<=` instead of `<`) would show up.
    """
    vals = List[Scalar[dtype]]()
    vals.append(-3.0); vals.append(-1e-7); vals.append(0.0)
    vals.append(1e-7); vals.append(0.5); vals.append(42.0)
    n = len(vals)

    buf = upload[dtype](ncx, vals)
    relu(ncx, buf, n)
    ncx.synchronize()
    got = download[dtype](buf, n)

    for i in range(n):
        want = vals[i] if vals[i] > Scalar[dtype](0) else Scalar[dtype](0)
        if got[i] != want:
            raise Error("relu(", vals[i], ") gave ", got[i], " and should be ", want)
    print("PASS relu exact on negatives, zero and tiny values")


def main() raises:
    with DeviceContext() as ctx:
        test_relu_is_exact(ctx)
        test_against_numpy(ctx)
        test_relu_actually_fires(ctx)
