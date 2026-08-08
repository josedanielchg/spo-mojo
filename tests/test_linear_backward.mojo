"""The linear layer's backward, verified by FINITE DIFFERENCES.

It is the M-step's most important test. Mojo has no autodiff, so the three
gradients are written by hand; if one is wrong, nothing fails: training simply does
not converge, weeks later and without saying why. The only defence is checking them
numerically.

The idea, without maths: a gradient says how much the loss changes if I move a
weight a little. That can be MEASURED directly -- move the weight, see how much it
changes -- and compared against what the formula says.

    measured gradient (central difference):  (L(w + e) - L(w - e)) / (2e)
    analytic gradient:                       what the kernel returns

The CENTRAL difference is used and not the one-sided one ((L(w+e) - L(w))/e)
because its error is proportional to e^2 rather than to e: with the same e it gives
two orders of magnitude more precision, and that is needed here.

On e = 1e-3: in float32 it cannot go much lower. With a very small e the two values
of L are nearly equal and subtracting them loses almost every significant figure
(cancellation), so the result is noise. With a very large e, the difference stops
approximating the derivative. 1e-3 is the compromise.

The test network is minimal (M=3, K=4, N=8) on purpose: that is 3*4 + 4*8 + 8 = 52
parameters, that is, 104 forwards. On the real network it would be tens of
thousands.
"""

from std.gpu.host import DeviceContext
from std.math import abs, sqrt

from ops.common import dtype
from networks.linear import linear_forward, linear_backward
from tests.golden_io import read_f32
from tests.helpers import upload, zeros, download, write_into

comptime GOLDEN = String("tests/golden/")

comptime M = 3
comptime K = 4
comptime N = 8
comptime EPS = Scalar[dtype](1e-3)
comptime TOL = Scalar[dtype](2e-2)
"""A RELATIVE tolerance. It is loose on purpose: the comparison does not seek
precision but catching a wrong gradient, and those usually fail by an integer factor
(2, -1, the batch) or by being transposed, not by 1%."""


def make_inputs() -> List[Scalar[dtype]]:
    """The input x [M, K], with deterministic values and mixed signs."""
    out = List[Scalar[dtype]]()
    for i in range(M * K):
        out.append(Scalar[dtype](((i * 37) % 13) - 6) * Scalar[dtype](0.25))
    return out^


def make_weights() -> List[Scalar[dtype]]:
    """Los pesos W [K, N], tambien deterministas."""
    out = List[Scalar[dtype]]()
    for i in range(K * N):
        out.append(Scalar[dtype](((i * 17) % 11) - 5) * Scalar[dtype](0.2))
    return out^


def make_bias() -> List[Scalar[dtype]]:
    out = List[Scalar[dtype]]()
    for i in range(N):
        out.append(Scalar[dtype](i % 3) * Scalar[dtype](0.3) - 0.3)
    return out^


def make_dy() -> List[Scalar[dtype]]:
    """The gradient dL/dy, that is, the weights of the scalar loss L = sum(g * y).

    That they all differ matters: with a uniform g, confusing a transpose or
    summing along the wrong axis can give the same number by coincidence.
    """
    out = List[Scalar[dtype]]()
    for i in range(M * N):
        out.append(Scalar[dtype](((i * 7) % 9) - 4) * Scalar[dtype](0.5))
    return out^


def loss(ctx: DeviceContext, x: List[Scalar[dtype]], w: List[Scalar[dtype]],
         b: List[Scalar[dtype]], g: List[Scalar[dtype]]) raises -> Scalar[dtype]:
    """L = sum_{m,n} g[m,n] * y[m,n], with y = x @ W + b.

    Any scalar loss will do; the only thing that matters is that its gradient with
    respect to y be exactly `g`, which is what will be passed to the backward.
    """
    yd = zeros[dtype](ctx, M * N)
    linear_forward(ctx, yd, upload[dtype](ctx, x), upload[dtype](ctx, w),
                   upload[dtype](ctx, b), M, K, N)
    ctx.synchronize()
    y = download[dtype](yd, M * N)

    total = Scalar[dtype](0)
    for i in range(M * N):
        total += g[i] * y[i]
    return total


def numeric_grad(ctx: DeviceContext, x: List[Scalar[dtype]],
                 w: List[Scalar[dtype]], b: List[Scalar[dtype]],
                 g: List[Scalar[dtype]], which: Int,
                 idx: Int) raises -> Scalar[dtype]:
    """Derivative of L with respect to parameter `idx` of `which` (0=x, 1=W, 2=b).

    Central difference: L is evaluated moving the parameter to one side and to the
    other.
    """
    xs = x.copy()
    ws = w.copy()
    bs = b.copy()

    if which == 0:
        base = xs[idx]
        xs[idx] = base + EPS
        up = loss(ctx, xs, ws, bs, g)
        xs[idx] = base - EPS
        down = loss(ctx, xs, ws, bs, g)
    elif which == 1:
        base = ws[idx]
        ws[idx] = base + EPS
        up = loss(ctx, xs, ws, bs, g)
        ws[idx] = base - EPS
        down = loss(ctx, xs, ws, bs, g)
    else:
        base = bs[idx]
        bs[idx] = base + EPS
        up = loss(ctx, xs, ws, bs, g)
        bs[idx] = base - EPS
        down = loss(ctx, xs, ws, bs, g)

    return (up - down) / (Scalar[dtype](2) * EPS)


def compare_grads(got: List[Scalar[dtype]], n: Int, ctx: DeviceContext,
                  x: List[Scalar[dtype]], w: List[Scalar[dtype]],
                  b: List[Scalar[dtype]], g: List[Scalar[dtype]],
                  which: Int, name: String) raises:
    """Compares the analytic gradient against the measured one, parameter by parameter."""
    worst = Scalar[dtype](0)
    worst_at = 0
    for i in range(n):
        num = numeric_grad(ctx, x, w, b, g, which, i)
        # Relative error, with a floor so as not to divide by near-zero when the
        # true gradient is 0.
        scale = abs(num)
        if scale < Scalar[dtype](1):
            scale = Scalar[dtype](1)
        rel = abs(got[i] - num) / scale
        if rel > worst:
            worst = rel
            worst_at = i
        if rel > TOL:
            raise Error(name, "[", i, "]: analitico ", got[i], " vs medido ", num,
                        " (error relativo ", rel, ")")
    print("      ", name, ": ", n, " parametros, peor error relativo ", worst,
          " (en el ", worst_at, ")")


def test_all_three_gradients(ctx: DeviceContext) raises:
    """The layer's three gradients against finite differences."""
    x = make_inputs()
    w = make_weights()
    b = make_bias()
    g = make_dy()

    dw = zeros[dtype](ctx, K * N)
    db = zeros[dtype](ctx, N)
    dx = zeros[dtype](ctx, M * K)
    linear_backward(ctx, dw, db, dx, upload[dtype](ctx, x), upload[dtype](ctx, w),
                    upload[dtype](ctx, g), M, K, N)
    ctx.synchronize()

    got_dw = download[dtype](dw, K * N)
    got_db = download[dtype](db, N)
    got_dx = download[dtype](dx, M * K)

    compare_grads(got_dw, K * N, ctx, x, w, b, g, 1, "dW")
    compare_grads(got_db, N, ctx, x, w, b, g, 2, "db")
    compare_grads(got_dx, M * K, ctx, x, w, b, g, 0, "dx")
    print("PASS los tres gradientes coinciden con las diferencias finitas")


def test_db_is_the_column_sum(ctx: DeviceContext) raises:
    """The db gradient, checked by hand: it is the column-wise sum of dy.

    It is the easiest gradient to verify without differentiating anything, and the
    one that gives away the bias having been added twice in the forward (it would
    come out double).
    """
    g = make_dy()
    dw = zeros[dtype](ctx, K * N)
    db = zeros[dtype](ctx, N)
    dx = zeros[dtype](ctx, M * K)
    linear_backward(ctx, dw, db, dx, upload[dtype](ctx, make_inputs()),
                    upload[dtype](ctx, make_weights()), upload[dtype](ctx, g),
                    M, K, N)
    ctx.synchronize()
    got = download[dtype](db, N)

    for n in range(N):
        want = Scalar[dtype](0)
        for m in range(M):
            want += g[m * N + n]
        if abs(got[n] - want) > Scalar[dtype](1e-5):
            raise Error("db[", n, "] dio ", got[n], " y la suma de columna es ",
                        want)
    print("PASS db es exactamente la suma de dy por columnas")


def grads_with(ctx: DeviceContext, x: List[Scalar[dtype]],
               ws: List[Scalar[dtype]],
               g: List[Scalar[dtype]]) raises -> List[Scalar[dtype]]:
    """Returns dW followed by dx, in a single list. At module level and not nested: in
    1.0.0b1 a nested function cannot capture the DeviceContext."""
    dw = zeros[dtype](ctx, K * N)
    db = zeros[dtype](ctx, N)
    dx = zeros[dtype](ctx, M * K)
    linear_backward(ctx, dw, db, dx, upload[dtype](ctx, x),
                    upload[dtype](ctx, ws), upload[dtype](ctx, g), M, K, N)
    ctx.synchronize()
    out = download[dtype](dw, K * N)
    for v in download[dtype](dx, M * K):
        out.append(v)
    return out^


def test_gradients_depend_on_the_right_things(ctx: DeviceContext) raises:
    """The dW gradient cannot depend on W, and dx must. It catches crossed arguments.

    It is a structural check: if in `dw = x^T @ dy` somebody wrote `w` instead of
    `x`, the finite differences would catch it, but this test says it more clearly.
    W is changed and it is checked that dW does not move (and that dx DOES).
    """
    x = make_inputs()
    w = make_weights()
    g = make_dy()

    base = grads_with(ctx, x, w, g)
    other = w.copy()
    for i in range(K * N):
        other[i] = other[i] + Scalar[dtype](1.5)     # W muy distinto
    moved = grads_with(ctx, x, other, g)

    for i in range(K * N):
        if abs(base[i] - moved[i]) > Scalar[dtype](1e-6):
            raise Error("dW cambio al cambiar W, y no deberia: indice ", i)
    changed = False
    for i in range(K * N, K * N + M * K):
        if abs(base[i] - moved[i]) > Scalar[dtype](1e-6):
            changed = True
    if not changed:
        raise Error("dx NO cambio al cambiar W, y deberia: dx = dy @ W^T")
    print("PASS dW no depende de W, y dx si (los argumentos no estan cruzados)")


def check_against_jax(ctx: DeviceContext, which: Int, m: Int, k: Int,
                      n: Int) raises:
    """One JAX golden case: the three gradients against exact autodiff."""
    tag = GOLDEN + "linbwd" + String(which) + "_"
    x = read_f32(tag + "x.bin")
    w = read_f32(tag + "w.bin")
    dy = read_f32(tag + "dy.bin")
    want_dw = read_f32(tag + "dw.bin")
    want_db = read_f32(tag + "db.bin")
    want_dx = read_f32(tag + "dx.bin")

    if (len(x) != m * k or len(w) != k * n or len(dy) != m * n
            or len(want_dw) != k * n or len(want_db) != n
            or len(want_dx) != m * k):
        raise Error("el golden del caso ", which, " no tiene las shapes esperadas")

    dw = zeros[dtype](ctx, k * n)
    db = zeros[dtype](ctx, n)
    dx = zeros[dtype](ctx, m * k)
    linear_backward(ctx, dw, db, dx, upload[dtype](ctx, x), upload[dtype](ctx, w),
                    upload[dtype](ctx, dy), m, k, n)
    ctx.synchronize()

    got_dw = download[dtype](dw, k * n)
    got_db = download[dtype](db, n)
    got_dx = download[dtype](dx, m * k)

    # A far harder tolerance than the finite differences': here the reference is
    # EXACT, so the only thing that can differ is float32 rounding from summing in
    # a different order. It scales with the dimension being reduced in each case.
    tw = Scalar[dtype](1e-5) * sqrt(Scalar[dtype](m))
    tb = Scalar[dtype](1e-5) * sqrt(Scalar[dtype](m))
    tx = Scalar[dtype](1e-5) * sqrt(Scalar[dtype](n))

    ew = worst_abs(got_dw, want_dw, k * n, tw, String("case", which, " dW"))
    eb = worst_abs(got_db, want_db, n, tb, String("case", which, " db"))
    ex = worst_abs(got_dx, want_dx, m * k, tx, String("case", which, " dx"))
    print("      case", which, " M=", m, " K=", k, " N=", n,
          "  error max: dW", ew, " db", eb, " dx", ex)


def worst_abs(got: List[Scalar[dtype]], want: List[Scalar[dtype]], n: Int,
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
        raise Error(what, ": diferencia ", worst, " en el indice ", at,
                    " (got=", got[at], " want=", want[at], ", tol ", tol, ")")
    return worst


def test_against_jax_autodiff(ctx: DeviceContext) raises:
    """The gradients against JAX, which is what Stoix uses (exact autodiff).

    It is the strong check, and it plugs the finite-difference test's blind spot:
    that one uses M=3, K=4, N=8, that is, EVERYTHING smaller than the tile of 16, so
    the forward's tile loop runs exactly once. Here there are cases with several
    tiles across all three dimensions and a degenerate 1x1x1 case.
    """
    check_against_jax(ctx, 0, 3, 4, 8)       # la red mini, para cruzar con las FD
    check_against_jax(ctx, 1, 20, 18, 64)    # la primera capa real, batch ragged
    check_against_jax(ctx, 2, 64, 64, 64)    # varios tiles en las tres dims
    check_against_jax(ctx, 3, 1, 1, 1)       # degenerado
    print("PASS los gradientes coinciden con el autodiff de JAX en los 4 casos")


def main() raises:
    with DeviceContext() as ctx:
        test_db_is_the_column_sum(ctx)
        test_gradients_depend_on_the_right_things(ctx)
        test_all_three_gradients(ctx)
        test_against_jax_autodiff(ctx)
