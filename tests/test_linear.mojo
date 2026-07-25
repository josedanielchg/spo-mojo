"""La capa lineal contra el golden de numpy.

Tres casos generados por tests/golden/gen/gen_linear.py, elegidos a proposito:

  case0  (4, 18) @ (18, 64)   la primera capa real del critico
  case1  (64, 64) @ (64, 64)  varios tiles en las tres dimensiones
  case2  (7, 5) @ (5, 3)      ragged: ninguna dimension es multiplo del tile,
                              asi que se pisan todos los guards

El caso ragged es el que importa: nuestras dimensiones reales (18 entradas, un
batch de particulas) casi nunca son multiplos de 16, asi que los bordes se
ejercitan en cada llamada de verdad.
"""

from std.gpu.host import DeviceContext
from std.math import abs, sqrt

from ops.common import dtype
from networks.linear import linear_forward, TILE
from tests.golden_io import read_f32
from tests.helpers import upload, zeros, download

comptime GOLDEN = String("tests/golden/")


def check_case(ctx: DeviceContext, which: Int, m: Int, k: Int, n: Int) raises:
    """Corre la capa sobre un caso del golden y compara con numpy."""
    x = read_f32(GOLDEN + "linear" + String(which) + "_x.bin")
    w = read_f32(GOLDEN + "linear" + String(which) + "_w.bin")
    b = read_f32(GOLDEN + "linear" + String(which) + "_b.bin")
    want = read_f32(GOLDEN + "linear" + String(which) + "_y.bin")

    if len(x) != m * k or len(w) != k * n or len(b) != n or len(want) != m * n:
        raise Error("el golden del caso ", which, " no tiene las shapes esperadas")

    xd = upload[dtype](ctx, x)
    wd = upload[dtype](ctx, w)
    bd = upload[dtype](ctx, b)
    yd = zeros[dtype](ctx, m * n)

    linear_forward(ctx, yd, xd, wd, bd, m, k, n)
    ctx.synchronize()
    got = download[dtype](yd, m * n)

    # Tolerancia RELATIVA a la escala del problema: sumar k terminos en float32
    # arrastra error proporcional a sqrt(k), y el orden de la suma no es el mismo
    # que el de numpy (aqui se acumula por tiles). Pedir 1e-6 absoluto seria
    # pedir mas precision de la que sobrevive a reordenar una suma.
    tol = Scalar[dtype](1e-5) * sqrt(Scalar[dtype](k))
    worst = Scalar[dtype](0)
    worst_at = 0
    for i in range(m * n):
        d = abs(got[i] - want[i])
        if d > worst:
            worst = d
            worst_at = i
    if worst > tol:
        raise Error("caso ", which, ": la mayor diferencia es ", worst,
                    " en el indice ", worst_at, " (got=", got[worst_at],
                    " want=", want[worst_at], ", tolerancia ", tol, ")")

    print("      case", which, " (", m, "x", k, ") @ (", k, "x", n,
          ")  error max", worst, " (tol", tol, ")")


def test_against_numpy(ctx: DeviceContext) raises:
    """Los tres casos del golden."""
    check_case(ctx, 0, 4, 18, 64)
    check_case(ctx, 1, 64, 64, 64)
    check_case(ctx, 2, 7, 5, 3)
    print("PASS la capa lineal coincide con numpy en los tres casos")


def test_bias_is_added_once(ctx: DeviceContext) raises:
    """Con W = 0, la salida tiene que ser exactamente el bias.

    Separa dos fallos que el golden no distingue: si el bias se sumara dos veces
    (una por tile, por ejemplo) el error se escondaria dentro del ruido de la
    matmul, pero aqui saldria justo el doble.
    """
    m = 5
    k = 20      # > TILE, o sea dos vueltas del bucle de tiles
    n = 3
    xs = List[Scalar[dtype]]()
    for i in range(m * k):
        xs.append(Scalar[dtype](i % 7) - 3.0)      # cualquier cosa, W=0 la anula
    ws = List[Scalar[dtype]]()
    for _ in range(k * n):
        ws.append(Scalar[dtype](0))
    bs = List[Scalar[dtype]]()
    for j in range(n):
        bs.append(Scalar[dtype](j) + 0.5)

    yd = zeros[dtype](ctx, m * n)
    linear_forward(ctx, yd, upload[dtype](ctx, xs), upload[dtype](ctx, ws),
                   upload[dtype](ctx, bs), m, k, n)
    ctx.synchronize()
    got = download[dtype](yd, m * n)

    for r in range(m):
        for c in range(n):
            v = got[r * n + c]
            if abs(v - bs[c]) > Scalar[dtype](1e-6):
                raise Error("con W=0 la salida deberia ser el bias: fila ", r,
                            " col ", c, " dio ", v, " y el bias es ", bs[c])
    print("PASS el bias se suma exactamente una vez (W=0 -> y = b)")


def test_identity_passes_input_through(ctx: DeviceContext) raises:
    """Con W = identidad y b = 0, la salida es la entrada.

    Comprueba la ORIENTACION de la matmul, que es lo que el golden no puede
    distinguir bien: si se confundieran filas con columnas al indexar W, el
    resultado seguiria pareciendo numeros plausibles. Con la identidad, un
    trasposicion se ve al instante.

    K = 20 > TILE a proposito: el 1 de cada fila cae en tiles distintos.
    """
    n = 20
    m = 3
    xs = List[Scalar[dtype]]()
    for r in range(m):
        for c in range(n):
            xs.append(Scalar[dtype](r * 100 + c))   # todos distintos
    ws = List[Scalar[dtype]]()
    for i in range(n):
        for j in range(n):
            ws.append(Scalar[dtype](1) if i == j else Scalar[dtype](0))
    bs = List[Scalar[dtype]]()
    for _ in range(n):
        bs.append(Scalar[dtype](0))

    yd = zeros[dtype](ctx, m * n)
    linear_forward(ctx, yd, upload[dtype](ctx, xs), upload[dtype](ctx, ws),
                   upload[dtype](ctx, bs), m, n, n)
    ctx.synchronize()
    got = download[dtype](yd, m * n)

    for i in range(m * n):
        if abs(got[i] - xs[i]) > Scalar[dtype](1e-6):
            raise Error("W=identidad deberia devolver la entrada: indice ", i,
                        " dio ", got[i], " y la entrada era ", xs[i])
    print("PASS W = identidad devuelve la entrada (la orientacion es correcta)")


def main() raises:
    with DeviceContext() as ctx:
        test_against_numpy(ctx)
        test_bias_is_added_once(ctx)
        test_identity_passes_input_through(ctx)
