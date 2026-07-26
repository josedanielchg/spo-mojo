"""El MLP del critico contra el golden de numpy, capa por capa.

Golden: tests/golden/gen/gen_critic.py, arquitectura 18 -> H -> H -> 1 con ReLU y
tres tamanos: H = 32 (1697 pesos), 64 (5441) y 256 (70913, el de Stoix).

Se comparan las activaciones INTERMEDIAS y no solo la salida. Si solo se comparara
V y fallara, no se sabria en que capa se rompio; comparando a1, a2 y V el fallo
queda localizado.

Cada arquitectura se prueba con dos batches que comparten pesos, asi que tambien
se comprueba que el resultado no depende del tamano del batch: 5 no es multiplo del
tile de 16 y pisa los guards, 64 ocupa varios tiles.
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
    """Prefijo de los ficheros del golden de esa arquitectura."""
    return GOLDEN + "critic_h" + String(hidden) + "_"


def load_params(ctx: DeviceContext, hidden: Int) raises -> CriticParams:
    """Los pesos del golden de la arquitectura dada, subidos a la GPU."""
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
    """Mayor diferencia absoluta; revienta si pasa de la tolerancia."""
    worst = Scalar[dtype](0)
    at = 0
    for i in range(n):
        d = abs(got[i] - want[i])
        if d > worst:
            worst = d
            at = i
    if worst > tol:
        raise Error(what, ": la mayor diferencia es ", worst, " en el indice ", at,
                    " (got=", got[at], " want=", want[at], ", tol ", tol, ")")
    return worst


def check_batch(ctx: DeviceContext, hidden: Int, m: Int) raises:
    """Un (arquitectura, batch) del golden, comparando las tres etapas."""
    params = load_params(ctx, hidden)
    cache = CriticCache(ctx, m, hidden, OUT_DIM)
    t = tag(hidden)

    x = read_f32(t + "x" + String(m) + ".bin")
    want_a1 = read_f32(t + "a1_" + String(m) + ".bin")
    want_a2 = read_f32(t + "a2_" + String(m) + ".bin")
    want_v = read_f32(t + "v" + String(m) + ".bin")
    if len(x) != m * IN_DIM:
        raise Error("el golden de entrada del batch ", m, " no cuadra")

    critic_forward(ctx, params, cache, upload[dtype](ctx, x), m)
    ctx.synchronize()

    got_a1 = download[dtype](cache.a1, m * hidden)
    got_a2 = download[dtype](cache.a2, m * hidden)
    got_v = download[dtype](cache.value, m * OUT_DIM)

    # La tolerancia crece con la profundidad: el error de cada capa entra en la
    # siguiente. Escala con sqrt(fan_in) como en el test de la lineal.
    t1 = Scalar[dtype](1e-5) * sqrt(Scalar[dtype](IN_DIM))
    t2 = Scalar[dtype](2e-5) * sqrt(Scalar[dtype](hidden))
    t3 = Scalar[dtype](3e-5) * sqrt(Scalar[dtype](hidden))

    e1 = compare(got_a1, want_a1, m * hidden, t1,
                 String("h", hidden, " batch ", m, " capa 1"))
    e2 = compare(got_a2, want_a2, m * hidden, t2,
                 String("h", hidden, " batch ", m, " capa 2"))
    e3 = compare(got_v, want_v, m * OUT_DIM, t3,
                 String("h", hidden, " batch ", m, " salida V"))

    print("      hidden", hidden, " batch", m,
          " error max: a1", e1, " a2", e2, " V", e3)


def test_against_numpy(ctx: DeviceContext) raises:
    """Las tres arquitecturas x los dos batches, capa por capa.

    Las tres corren con el MISMO codigo: el MLP es generico en las dimensiones,
    asi que pasar de 32 a 256 no toca ni una linea de Mojo. Se comparan varios
    tamanos porque el numero de neuronas no lo fija ni el paper ni Stoix (Stoix
    usa [256,256]); para un juego de 9 casillas hay que medirlo.
    """
    hiddens = List[Int]()
    hiddens.append(32); hiddens.append(64); hiddens.append(256)
    for i in range(len(hiddens)):
        check_batch(ctx, hiddens[i], 5)     # ragged: 5 no es multiplo de 16
        check_batch(ctx, hiddens[i], 64)    # varios tiles en el batch
    print("PASS el MLP coincide con numpy en las 3 arquitecturas (a1, a2 y V)")


def test_relu_actually_fires(ctx: DeviceContext) raises:
    """Comprobacion de que el test de arriba prueba algo.

    Si el ReLU nunca recortara nada, el golden pasaria igual y no distinguiriamos
    una red CON ReLU de una sin el. El generador reporta ~50% de activaciones
    apagadas; aqui se verifica en la salida real: tiene que haber ceros en a1 y a2,
    y ningun valor negativo.
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
            raise Error("a1 tiene un valor negativo en ", i, ": ", a1[i])
        if a1[i] == Scalar[dtype](0):
            zeros1 += 1
    zeros2 = 0
    for i in range(m * hidden):
        if a2[i] < Scalar[dtype](0):
            raise Error("a2 tiene un valor negativo en ", i, ": ", a2[i])
        if a2[i] == Scalar[dtype](0):
            zeros2 += 1

    total = m * hidden
    if zeros1 == 0 or zeros2 == 0:
        raise Error("el ReLU no recorto nada: el golden no distinguiria una red "
                    "con ReLU de una sin el")
    print("      apagadas por el ReLU: capa1", Scalar[dtype](zeros1) / Scalar[dtype](total),
          " capa2", Scalar[dtype](zeros2) / Scalar[dtype](total))
    print("PASS el ReLU recorta de verdad (hay ceros y ningun negativo)")


def test_relu_is_exact(ncx: DeviceContext) raises:
    """El ReLU suelto, sobre valores elegidos: negativos a 0, el resto intactos.

    Incluye el 0 y valores muy pequenos a los dos lados, que es donde una
    comparacion mal escrita (`<=` en vez de `<`) se notaria.
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
            raise Error("relu(", vals[i], ") dio ", got[i], " y deberia ser ", want)
    print("PASS relu exacto en negativos, cero y valores diminutos")


def main() raises:
    with DeviceContext() as ctx:
        test_relu_is_exact(ctx)
        test_against_numpy(ctx)
        test_relu_actually_fires(ctx)
