"""RNG y muestreo categorico.

El muestreo se prueba de dos formas que se complementan:
  - con uniformes INYECTADOS a mano -> indices exactos, cero tolerancias. Si
    esto falla, la CDF inversa esta mal.
  - con el RNG de verdad -> frecuencias cerca de las probabilidades. Si esto
    falla pero lo anterior pasa, el problema esta en el generador, no en el
    muestreo.

Separarlo asi es lo que hace que un fallo diga DONDE mirar.
"""

from std.gpu.host import DeviceContext
from std.math import abs, log

from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform, categorical_from_logits, rand_uniform, RngKey
from tests.helpers import upload, zeros, filled, download, assert_eq_int

comptime TPB = 32


def log_probs_rows(probs: List[Scalar[dtype]], rows: Int) -> List[Scalar[dtype]]:
    """Repite log(p) en `rows` filas. El softmax del kernel deshace el log y
    recupera p, asi que las probabilidades salen tal cual se piden."""
    out = List[Scalar[dtype]]()
    for _ in range(rows):
        for c in range(len(probs)):
            out.append(log(probs[c]))
    return out^


def make_probs() -> List[Scalar[dtype]]:
    """[0.1, 0.2, 0.3, 0.4]: desiguales a proposito, para que un muestreo que
    devolviera siempre lo mismo (o uniforme) se note."""
    p = List[Scalar[dtype]]()
    p.append(0.1)
    p.append(0.2)
    p.append(0.3)
    p.append(0.4)
    return p^


def test_determinism() raises:
    """Misma key -> misma secuencia. Streams distintos -> secuencias distintas."""
    for i in range(64):
        if rand_uniform(42, 0, UInt32(i)) != rand_uniform(42, 0, UInt32(i)):
            raise Error("rand_uniform no es determinista en i=", i)

    # Dos streams del mismo seed tienen que decorrelacionarse. Permito hasta 2
    # coincidencias de 64 por pura casualidad; en la practica salen 0.
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
    """Media ~ 0.5 y varianza ~ 1/12 = 0.0833, que es lo que debe dar U(0,1)."""
    n = 100000
    o = zeros[dtype](ctx, n)

    blocks = (n + TPB - 1) // TPB
    ctx.enqueue_function[fill_uniform, fill_uniform](
        o.unsafe_ptr(), UInt32(7), UInt32(0), n, grid_dim=blocks, block_dim=TPB)
    ctx.synchronize()

    got = download[dtype](o, n)

    # Acumulo en float64: sumar 100k floats en float32 pierde precision
    # suficiente como para mover la media en el tercer decimal.
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
    """Uniformes inyectados -> indices exactos, calculados a mano.

    Con probabilidades [0.1, 0.2, 0.3, 0.4] el CDF es [0.1, 0.3, 0.6, 1.0].
    Pruebo el interior de cada tramo y los dos bordes.
    """
    probs = make_probs()
    row_size = len(probs)

    us = List[Scalar[dtype]]()
    want = List[Int]()
    us.append(0.0);   want.append(0)   # borde inferior exacto
    us.append(0.05);  want.append(0)   # dentro de [0.0, 0.1)
    us.append(0.15);  want.append(1)   # dentro de [0.1, 0.3)
    us.append(0.45);  want.append(2)   # dentro de [0.3, 0.6)
    us.append(0.75);  want.append(3)   # dentro de [0.6, 1.0)
    us.append(0.999); want.append(3)   # pegado al borde superior
    rows = len(us)

    logits = upload[dtype](ctx, log_probs_rows(probs, rows))
    u = upload[dtype](ctx, us)
    # Salida a -1: si el kernel no escribiera nada, se veria.
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
    """Con el RNG real, las frecuencias tienen que acercarse a las probabilidades."""
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

    # 3 grados de libertad: 16.3 es el p=0.001. Con la seed fija esto no es
    # flaky, sale siempre lo mismo; el umbral es por si toco el generador.
    if chi2 > 16.3:
        raise Error("chi2 demasiado alto: ", chi2)
    print("PASS categorical: frecuencias ~ probabilidades (chi2 =", chi2, ")")


def main() raises:
    test_determinism()
    with DeviceContext() as ctx:
        test_uniform_stats(ctx)
        test_categorical_exact(ctx)
        test_categorical_frequencies(ctx)
