"""RNG y muestreo categorico.

Lo importante: el muestreo se prueba de dos formas complementarias.
 - Con uniformes inyectados a mano -> indices EXACTOS (determinista, sin
   tolerancias estadisticas).
 - Con el RNG de verdad -> frecuencias cerca de las probabilidades (chi2 suave).
"""

from std.gpu.host import DeviceContext
from std.math import abs, exp, log

from ops.rng import (fill_uniform, categorical_from_logits, rand_uniform,
                     RngKey, dtype, idx_dtype)

comptime TPB = 32


def test_determinism() raises:
    """Misma key -> misma secuencia. Distinto stream -> secuencia distinta."""
    for i in range(64):
        a = rand_uniform(42, 0, UInt32(i))
        b = rand_uniform(42, 0, UInt32(i))
        if a != b:
            raise Error("rand_uniform no es determinista en i=", i)

    same = 0
    for i in range(64):
        if rand_uniform(42, 0, UInt32(i)) == rand_uniform(42, 1, UInt32(i)):
            same += 1
    if same > 2:
        raise Error("los streams 0 y 1 se parecen demasiado: ", same, "/64 iguales")

    k = RngKey(42, 0)
    k1 = k.split(1)
    k2 = k.split(2)
    if k1.stream == k2.stream:
        raise Error("split(1) y split(2) dieron el mismo stream")
    print("PASS rng determinista y streams independientes")


def test_uniform_stats(ctx: DeviceContext) raises:
    """Media ~ 0.5 y varianza ~ 1/12 = 0.0833."""
    n = 100000
    o = ctx.enqueue_create_buffer[dtype](n)
    o.enqueue_fill(0)

    blocks = (n + TPB - 1) // TPB
    ctx.enqueue_function[fill_uniform, fill_uniform](
        o.unsafe_ptr(), UInt32(7), UInt32(0), n, grid_dim=blocks, block_dim=TPB)
    ctx.synchronize()

    with o.map_to_host() as h:
        total = Float64(0)
        for i in range(n):
            v = Float64(h[i])
            if v < 0.0 or v >= 1.0:
                raise Error("uniforme fuera de [0,1) en ", i, ": ", h[i])
            total += v
        mean = total / Float64(n)

        variance = Float64(0)
        for i in range(n):
            d = Float64(h[i]) - mean
            variance += d * d
        variance /= Float64(n)

    if abs(mean - 0.5) > 0.01:
        raise Error("media fuera de rango: ", mean)
    if abs(variance - 1.0 / 12.0) > 0.005:
        raise Error("varianza fuera de rango: ", variance)
    print("PASS uniformes: media", mean, "varianza", variance)


def test_categorical_exact(ctx: DeviceContext) raises:
    """Uniformes inyectados -> indices exactos, calculados a mano.

    Logits log([0.1, 0.2, 0.3, 0.4]) -> CDF [0.1, 0.3, 0.6, 1.0].
    Elijo uniformes que caen claramente dentro de cada tramo, y ademas los dos
    bordes (0.0 -> primer bucket, 0.999 -> ultimo).
    """
    row_size = 4
    probs = List[Scalar[dtype]]()
    probs.append(0.1)
    probs.append(0.2)
    probs.append(0.3)
    probs.append(0.4)

    us = List[Scalar[dtype]]()
    want = List[Int]()
    us.append(0.0);    want.append(0)   # borde inferior
    us.append(0.05);   want.append(0)   # dentro de [0.0, 0.1)
    us.append(0.15);   want.append(1)   # dentro de [0.1, 0.3)
    us.append(0.45);   want.append(2)   # dentro de [0.3, 0.6)
    us.append(0.75);   want.append(3)   # dentro de [0.6, 1.0)
    us.append(0.999);  want.append(3)   # borde superior
    rows = len(us)

    logits = ctx.enqueue_create_buffer[dtype](rows * row_size)
    logits.enqueue_fill(0)
    with logits.map_to_host() as h:
        for r in range(rows):
            for c in range(row_size):
                # log(p): el softmax lo devuelve a p
                h[r * row_size + c] = log(probs[c])

    u = ctx.enqueue_create_buffer[dtype](rows)
    u.enqueue_fill(0)
    with u.map_to_host() as h:
        for r in range(rows):
            h[r] = us[r]

    o = ctx.enqueue_create_buffer[idx_dtype](rows)
    o.enqueue_fill(-1)

    ctx.enqueue_function[categorical_from_logits[TPB], categorical_from_logits[TPB]](
        o.unsafe_ptr(), logits.unsafe_ptr(), u.unsafe_ptr(), row_size,
        grid_dim=rows, block_dim=TPB)
    ctx.synchronize()

    with o.map_to_host() as h:
        for r in range(rows):
            if Int(h[r]) != want[r]:
                raise Error("u=", us[r], " -> got=", Int(h[r]), " want=", want[r])
    print("PASS categorical con uniformes inyectados (indices exactos)")


def test_categorical_frequencies(ctx: DeviceContext) raises:
    """Con el RNG real, las frecuencias tienen que acercarse a las probabilidades."""
    row_size = 4
    n = 100000
    probs = List[Scalar[dtype]]()
    probs.append(0.1)
    probs.append(0.2)
    probs.append(0.3)
    probs.append(0.4)

    logits = ctx.enqueue_create_buffer[dtype](n * row_size)
    logits.enqueue_fill(0)
    with logits.map_to_host() as h:
        for r in range(n):
            for c in range(row_size):
                h[r * row_size + c] = log(probs[c])

    u = ctx.enqueue_create_buffer[dtype](n)
    u.enqueue_fill(0)
    blocks = (n + TPB - 1) // TPB
    ctx.enqueue_function[fill_uniform, fill_uniform](
        u.unsafe_ptr(), UInt32(1234), UInt32(5), n, grid_dim=blocks, block_dim=TPB)

    o = ctx.enqueue_create_buffer[idx_dtype](n)
    o.enqueue_fill(-1)
    ctx.enqueue_function[categorical_from_logits[TPB], categorical_from_logits[TPB]](
        o.unsafe_ptr(), logits.unsafe_ptr(), u.unsafe_ptr(), row_size,
        grid_dim=n, block_dim=TPB)
    ctx.synchronize()

    counts = List[Int]()
    for _ in range(row_size):
        counts.append(0)
    with o.map_to_host() as h:
        for i in range(n):
            k = Int(h[i])
            if k < 0 or k >= row_size:
                raise Error("indice invalido en ", i, ": ", k)
            counts[k] += 1

    # chi2 suave: cada frecuencia dentro de 0.01 absoluto de su probabilidad.
    chi2 = Float64(0)
    for c in range(row_size):
        freq = Float64(counts[c]) / Float64(n)
        expected = Float64(probs[c])
        if abs(freq - expected) > 0.01:
            raise Error("categoria ", c, ": freq=", freq, " esperada=", expected)
        e = expected * Float64(n)
        d = Float64(counts[c]) - e
        chi2 += d * d / e
    # 3 grados de libertad: 16.3 seria p<0.001. Con seed fija esto es estable.
    if chi2 > 16.3:
        raise Error("chi2 demasiado alto: ", chi2)
    print("PASS categorical: frecuencias ~ probabilidades (chi2 =", chi2, ")")


def main() raises:
    test_determinism()
    with DeviceContext() as ctx:
        test_uniform_stats(ctx)
        test_categorical_exact(ctx)
        test_categorical_frequencies(ctx)
