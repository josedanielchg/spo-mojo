"""El backward de la capa lineal, verificado por DIFERENCIAS FINITAS.

Es el test mas importante del M-step. Mojo no tiene autodiff, asi que los tres
gradientes estan escritos a mano; si uno esta mal, nada falla: el entrenamiento
simplemente no converge, semanas despues y sin decir por que. La unica defensa es
comprobarlos numericamente.

La idea, sin matematicas: un gradiente dice cuanto cambia la perdida si muevo un
peso un poquito. Eso se puede MEDIR directamente — mover el peso, ver cuanto
cambia — y comparar con lo que dice la formula.

    gradiente medido (diferencia central):   (L(w + e) - L(w - e)) / (2e)
    gradiente analitico:                     lo que devuelve el kernel

Se usa la diferencia CENTRAL y no la de un lado ((L(w+e) - L(w))/e) porque su
error es proporcional a e^2 en vez de a e: con el mismo e da dos ordenes de
magnitud mas de precision, y aqui hace falta.

Sobre e = 1e-3: en float32 no se puede bajar mucho mas. Con e muy pequeno los dos
valores de L son casi iguales y al restarlos se pierden casi todas las cifras
significativas (cancelacion), asi que el resultado es ruido. Con e muy grande, la
diferencia deja de aproximar la derivada. 1e-3 es el compromiso.

La red de prueba es minima (M=3, K=4, N=8) a proposito: son 3*4 + 4*8 + 8 = 52
parametros, o sea 104 forwards. Sobre la red real serian decenas de miles.
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
"""Tolerancia RELATIVA. Es floja a proposito: la comparacion no busca precision
sino cazar un gradiente equivocado, y esos suelen fallar por un factor entero (2,
-1, el batch) o por estar traspuestos, no por un 1%."""


def make_inputs() -> List[Scalar[dtype]]:
    """La entrada x [M, K], con valores deterministas y de signos mezclados."""
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
    """El gradiente dL/dy, o sea los pesos de la perdida escalar L = suma(g * y).

    Que sean todos distintos importa: con g uniforme, confundir una traspuesta o
    sumar por el eje equivocado puede dar el mismo numero por casualidad.
    """
    out = List[Scalar[dtype]]()
    for i in range(M * N):
        out.append(Scalar[dtype](((i * 7) % 9) - 4) * Scalar[dtype](0.5))
    return out^


def loss(ctx: DeviceContext, x: List[Scalar[dtype]], w: List[Scalar[dtype]],
         b: List[Scalar[dtype]], g: List[Scalar[dtype]]) raises -> Scalar[dtype]:
    """L = suma_{m,n} g[m,n] * y[m,n], con y = x @ W + b.

    Una perdida escalar cualquiera sirve; lo unico que importa es que su gradiente
    respecto a y sea exactamente `g`, que es lo que se le pasara al backward.
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
    """Derivada de L respecto al parametro `idx` de `which` (0=x, 1=W, 2=b).

    Diferencia central: se evalua L moviendo el parametro a un lado y al otro.
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
    """Compara el gradiente analitico con el medido, parametro a parametro."""
    worst = Scalar[dtype](0)
    worst_at = 0
    for i in range(n):
        num = numeric_grad(ctx, x, w, b, g, which, i)
        # Error relativo, con un suelo para no dividir por casi-cero cuando el
        # gradiente de verdad es 0.
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
    """Los tres gradientes de la capa contra diferencias finitas."""
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
    """El gradiente db, comprobado a mano: es la suma de dy por columnas.

    Es el gradiente mas facil de verificar sin derivar nada, y el que delata que
    el bias se haya sumado dos veces en el forward (saldria el doble).
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
    """Devuelve dW seguido de dx, en una sola lista. A nivel de modulo y no anidada: en
    1.0.0b1 una funcion anidada no puede capturar el DeviceContext."""
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
    """El gradiente dW no puede depender de W, y dx si. Caza argumentos cruzados.

    Es una comprobacion estructural: si en `dw = x^T @ dy` alguien escribiera `w`
    en vez de `x`, las diferencias finitas lo cazarian, pero este test lo dice mas
    claro. Se cambia W y se comprueba que dW no se mueve (y que dx SI).
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
    """Un caso del golden de JAX: los tres gradientes contra autodiff exacto."""
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

    # Tolerancia mucho mas dura que la de diferencias finitas: aqui la referencia
    # es EXACTA, asi que lo unico que puede diferir es el redondeo de float32 al
    # sumar en otro orden. Escala con la dimension que se reduce en cada caso.
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
    """Mayor diferencia absoluta; revienta si pasa de la tolerancia."""
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
    """Los gradientes contra JAX, que es lo que usa Stoix (autodiff exacto).

    Es la verificacion fuerte, y tapa el punto ciego del test de diferencias
    finitas: aquel usa M=3, K=4, N=8, o sea TODO menor que el tile de 16, asi que
    el bucle de tiles del forward corre una sola vez. Aqui hay casos con varios
    tiles en las tres dimensiones y un caso degenerado de 1x1x1.
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
