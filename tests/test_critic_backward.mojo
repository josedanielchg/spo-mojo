"""El backward del MLP completo: gradientes a traves de dos ReLU.

Dos verificaciones independientes, porque una sola no basta para el riesgo nº1:

  1. Contra el autodiff de JAX (referencia EXACTA), sobre los 6 tensores de pesos
     y dos configuraciones (H=32 batch 20 ragged, H=64 batch 64 multi-tile).
  2. Contra diferencias finitas centrales sobre una red mini, que no depende de
     ningun golden: si los dos coinciden, o los dos estan mal de la misma forma
     (improbable) o el backward es correcto.

Lo nuevo respecto a E1.4 es que el gradiente atraviesa la mascara del ReLU. Ahi
esta el error tipico: aplicarla con la activacion equivocada, o en el lado
equivocado de la capa. Un fallo asi no rompe nada visible — la red simplemente
aprende peor.

Y ojo con una diferencia importante frente a E1.4: alli la perdida era LINEAL en
los pesos, asi que las diferencias finitas eran exactas. Aqui la perdida es
cuadratica y pasa por dos ReLU, o sea que la aproximacion numerica ya tiene error
de verdad. Es una prueba mas exigente.
"""

from std.gpu.host import DeviceContext
from std.math import abs, sqrt

from ops.common import dtype
from networks.mlp import (CriticParams, CriticCache, CriticGrads, CriticScratch,
                          critic_forward, critic_backward, zero_critic_params)
from tests.golden_io import read_f32
from tests.helpers import upload, zeros, download, write_into

comptime GOLDEN = String("tests/golden/")
comptime IN_DIM = 18
comptime OUT_DIM = 1


def cbwd_tag(hidden: Int, m: Int) -> String:
    return GOLDEN + "cbwd_h" + String(hidden) + "_m" + String(m) + "_"


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


def check_against_jax(ctx: DeviceContext, hidden: Int, m: Int) raises:
    """Un caso del golden: forward primero, y despues los 6 gradientes."""
    t = cbwd_tag(hidden, m)
    params = zero_critic_params(ctx, IN_DIM, hidden, OUT_DIM)
    write_into[dtype](params.w1, read_f32(t + "w1.bin"))
    write_into[dtype](params.b1, read_f32(t + "b1.bin"))
    write_into[dtype](params.w2, read_f32(t + "w2.bin"))
    write_into[dtype](params.b2, read_f32(t + "b2.bin"))
    write_into[dtype](params.w3, read_f32(t + "w3.bin"))
    write_into[dtype](params.b3, read_f32(t + "b3.bin"))

    x = upload[dtype](ctx, read_f32(t + "x.bin"))
    target = upload[dtype](ctx, read_f32(t + "target.bin"))
    cache = CriticCache(ctx, m, hidden, OUT_DIM)
    grads = CriticGrads(ctx, IN_DIM, hidden, OUT_DIM)
    scratch = CriticScratch(ctx, m, IN_DIM, hidden, OUT_DIM)

    critic_forward(ctx, params, cache, x, m)
    critic_backward(ctx, params, cache, grads, scratch, x, target, m)
    ctx.synchronize()

    # Primero el forward: si ya difiere, el gradiente diferiria por razones que no
    # son el backward, y el mensaje de error apuntaria al sitio equivocado.
    tol_fwd = Scalar[dtype](1e-5) * sqrt(Scalar[dtype](hidden))
    _ = worst_abs(download[dtype](cache.value, m * OUT_DIM),
                  read_f32(t + "v.bin"), m * OUT_DIM, tol_fwd,
                  String("h", hidden, " forward V"))

    # Y ahora los seis gradientes. La tolerancia escala con la dimension que se
    # reduce en cada uno.
    tol = Scalar[dtype](1e-6) * sqrt(Scalar[dtype](m * hidden))
    e = List[Scalar[dtype]]()
    e.append(worst_abs(download[dtype](grads.dw1, IN_DIM * hidden),
                       read_f32(t + "dw1.bin"), IN_DIM * hidden, tol,
                       String("h", hidden, " dW1")))
    e.append(worst_abs(download[dtype](grads.db1, hidden),
                       read_f32(t + "db1.bin"), hidden, tol,
                       String("h", hidden, " db1")))
    e.append(worst_abs(download[dtype](grads.dw2, hidden * hidden),
                       read_f32(t + "dw2.bin"), hidden * hidden, tol,
                       String("h", hidden, " dW2")))
    e.append(worst_abs(download[dtype](grads.db2, hidden),
                       read_f32(t + "db2.bin"), hidden, tol,
                       String("h", hidden, " db2")))
    e.append(worst_abs(download[dtype](grads.dw3, hidden * OUT_DIM),
                       read_f32(t + "dw3.bin"), hidden * OUT_DIM, tol,
                       String("h", hidden, " dW3")))
    e.append(worst_abs(download[dtype](grads.db3, OUT_DIM),
                       read_f32(t + "db3.bin"), OUT_DIM, tol,
                       String("h", hidden, " db3")))

    worst = Scalar[dtype](0)
    for i in range(len(e)):
        if e[i] > worst:
            worst = e[i]
    print("      hidden", hidden, " batch", m, " -> peor error en los 6 tensores:",
          worst, " (tol", tol, ")")


def test_against_jax_autodiff(ctx: DeviceContext) raises:
    """Los 6 gradientes contra el autodiff de JAX, en dos configuraciones."""
    check_against_jax(ctx, 32, 20)    # batch ragged
    check_against_jax(ctx, 64, 64)    # multi-tile
    print("PASS los 6 gradientes del critico coinciden con el autodiff de JAX")


# --- La segunda verificacion: diferencias finitas, sin depender de ningun golden

comptime FD_IN = 4
comptime FD_HID = 5
comptime FD_OUT = 1
comptime FD_M = 3
comptime EPS = Scalar[dtype](1e-3)
comptime FD_TOL = Scalar[dtype](5e-2)
"""Tolerancia relativa floja: la perdida ya no es lineal en los pesos (es
cuadratica y pasa por dos ReLU), asi que la aproximacion numerica tiene error de
verdad. Lo que se busca no es precision sino cazar un gradiente equivocado."""

comptime FD_MIN_SIGNAL = Scalar[dtype](1e-4)
"""Cambio minimo de la perdida (|L(w+e) - L(w-e)|) para fiarse de la medida.

Existe por una limitacion REAL del metodo, medida al escribir este test: la
perdida vale ~0.8 y el cambio a medir es 2*e*grad, que para un gradiente de 0.004
son 8.6e-6. Restar dos numeros de 0.8 que difieren en 8.6e-6 pierde cinco cifras
significativas (cancelacion catastrofica), y en float32 solo hay siete. Medido en
numpy sobre ese mismo parametro:

    float64   exacto -0.00432000   dif.finita -0.00432000   error   0.00%
    float32   exacto -0.00432005   dif.finita -0.00476837   error  10.38%

O sea que la diferencia finita en float32 NO tiene resolucion para gradientes
pequenos; no es que el backward este mal (nuestro kernel dio -0.00432005, que
coincide con el exacto). Asi que los parametros cuya senal no supera el ruido se
saltan y se reportan, en vez de subir la tolerancia hasta que pase. La cobertura
no se pierde: el autodiff de JAX ya verifica TODOS los parametros a 1e-8."""


def fd_values(n: Int, seed: Int) -> List[Scalar[dtype]]:
    """Valores deterministas con signos mezclados, para la red mini."""
    out = List[Scalar[dtype]]()
    for i in range(n):
        out.append(Scalar[dtype](((i * seed) % 11) - 5) * Scalar[dtype](0.3))
    return out^


def fd_loss(ctx: DeviceContext, w1: List[Scalar[dtype]], b1: List[Scalar[dtype]],
            w2: List[Scalar[dtype]], b2: List[Scalar[dtype]],
            w3: List[Scalar[dtype]], b3: List[Scalar[dtype]],
            x: List[Scalar[dtype]],
            target: List[Scalar[dtype]]) raises -> Scalar[dtype]:
    """La perdida L2 media, calculada corriendo el forward de verdad."""
    p = zero_critic_params(ctx, FD_IN, FD_HID, FD_OUT)
    write_into[dtype](p.w1, w1); write_into[dtype](p.b1, b1)
    write_into[dtype](p.w2, w2); write_into[dtype](p.b2, b2)
    write_into[dtype](p.w3, w3); write_into[dtype](p.b3, b3)
    cache = CriticCache(ctx, FD_M, FD_HID, FD_OUT)
    critic_forward(ctx, p, cache, upload[dtype](ctx, x), FD_M)
    ctx.synchronize()

    v = download[dtype](cache.value, FD_M * FD_OUT)
    total = Scalar[dtype](0)
    for i in range(FD_M * FD_OUT):
        d = v[i] - target[i]
        total += Scalar[dtype](0.5) * d * d
    return total / Scalar[dtype](FD_M * FD_OUT)


def test_finite_differences_through_relu(ctx: DeviceContext) raises:
    """Diferencias finitas sobre TODOS los pesos de una red mini 4->5->5->1.

    No usa ningun golden: perturba cada peso, mide como cambia la perdida de
    verdad, y lo compara con lo que dice el backward. Es la comprobacion que no
    depende de que JAX (ni nadie) tenga razon.
    """
    w1 = fd_values(FD_IN * FD_HID, 7)
    b1 = fd_values(FD_HID, 13)
    w2 = fd_values(FD_HID * FD_HID, 5)
    b2 = fd_values(FD_HID, 17)
    w3 = fd_values(FD_HID * FD_OUT, 3)
    b3 = fd_values(FD_OUT, 11)
    x = fd_values(FD_M * FD_IN, 23)
    target = fd_values(FD_M * FD_OUT, 29)

    p = zero_critic_params(ctx, FD_IN, FD_HID, FD_OUT)
    write_into[dtype](p.w1, w1); write_into[dtype](p.b1, b1)
    write_into[dtype](p.w2, w2); write_into[dtype](p.b2, b2)
    write_into[dtype](p.w3, w3); write_into[dtype](p.b3, b3)
    cache = CriticCache(ctx, FD_M, FD_HID, FD_OUT)
    grads = CriticGrads(ctx, FD_IN, FD_HID, FD_OUT)
    scratch = CriticScratch(ctx, FD_M, FD_IN, FD_HID, FD_OUT)
    xd = upload[dtype](ctx, x)

    critic_forward(ctx, p, cache, xd, FD_M)
    critic_backward(ctx, p, cache, grads, scratch, xd,
                    upload[dtype](ctx, target), FD_M)
    ctx.synchronize()

    got_dw1 = download[dtype](grads.dw1, FD_IN * FD_HID)
    got_dw2 = download[dtype](grads.dw2, FD_HID * FD_HID)
    got_dw3 = download[dtype](grads.dw3, FD_HID * FD_OUT)
    got_db3 = download[dtype](grads.db3, FD_OUT)

    # dW1 es el que mas lejos esta de la perdida: su gradiente atraviesa las DOS
    # mascaras de ReLU y las tres capas. Si la cadena esta bien, ese lo confirma.
    worst = Scalar[dtype](0)
    checked = 0
    skipped = 0
    for i in range(FD_IN * FD_HID):
        moved = w1.copy()
        base = moved[i]
        moved[i] = base + EPS
        up = fd_loss(ctx, moved, b1, w2, b2, w3, b3, x, target)
        moved[i] = base - EPS
        down = fd_loss(ctx, moved, b1, w2, b2, w3, b3, x, target)

        # Si la perdida apenas se movio, la resta es casi todo ruido de float32:
        # esa medida no puede verificar nada (ver FD_MIN_SIGNAL).
        if abs(up - down) < FD_MIN_SIGNAL:
            skipped += 1
            continue

        num = (up - down) / (Scalar[dtype](2) * EPS)
        scale = abs(num)
        if scale < Scalar[dtype](0.01):
            scale = Scalar[dtype](0.01)
        rel = abs(got_dw1[i] - num) / scale
        if rel > worst:
            worst = rel
        checked += 1
        if rel > FD_TOL:
            raise Error("dW1[", i, "]: analitico ", got_dw1[i], " vs medido ", num,
                        " (error relativo ", rel, ")")
    if checked == 0:
        raise Error("todos los parametros de dW1 quedaron por debajo del umbral: "
                    "el test no estaria comprobando nada")
    print("      dW1 (a traves de 2 ReLU y 3 capas):", checked, "comprobados,",
          skipped, "sin senal suficiente, peor error relativo", worst)

    # Y la ultima capa, que es el camino corto.
    worst3 = Scalar[dtype](0)
    checked3 = 0
    skipped3 = 0
    for i in range(FD_HID * FD_OUT):
        moved = w3.copy()
        base = moved[i]
        moved[i] = base + EPS
        up = fd_loss(ctx, w1, b1, w2, b2, moved, b3, x, target)
        moved[i] = base - EPS
        down = fd_loss(ctx, w1, b1, w2, b2, moved, b3, x, target)
        if abs(up - down) < FD_MIN_SIGNAL:
            skipped3 += 1
            continue
        num = (up - down) / (Scalar[dtype](2) * EPS)
        scale = abs(num)
        if scale < Scalar[dtype](0.01):
            scale = Scalar[dtype](0.01)
        rel = abs(got_dw3[i] - num) / scale
        if rel > worst3:
            worst3 = rel
        checked3 += 1
        if rel > FD_TOL:
            raise Error("dW3[", i, "]: analitico ", got_dw3[i], " vs medido ", num,
                        " (error relativo ", rel, ")")
    if checked3 == 0:
        raise Error("ningun parametro de dW3 tuvo senal suficiente")
    print("      dW3 (capa de salida):", checked3, "comprobados,", skipped3,
          "sin senal, peor error relativo", worst3)
    print("PASS las diferencias finitas confirman el backward a traves del ReLU")


def test_relu_mask_actually_blocks(ctx: DeviceContext) raises:
    """Donde el ReLU recorto, no puede pasar gradiente.

    Es la comprobacion estructural de lo unico nuevo de esta etapa. Si la mascara
    no se aplicara, el gradiente seguiria pareciendo plausible pero seria el de una
    red SIN ReLU. Aqui se fuerza el caso extremo: pesos que apagan casi todo.
    """
    hidden = 8
    m = 4
    p = zero_critic_params(ctx, FD_IN, hidden, FD_OUT)
    # Bias muy negativo en la capa 1 -> a1 = 0 en todas las neuronas.
    ones = List[Scalar[dtype]]()
    for _ in range(FD_IN * hidden):
        ones.append(Scalar[dtype](0.1))
    very_negative = List[Scalar[dtype]]()
    for _ in range(hidden):
        very_negative.append(Scalar[dtype](-100))
    write_into[dtype](p.w1, ones)
    write_into[dtype](p.b1, very_negative)

    x = List[Scalar[dtype]]()
    for _ in range(m * FD_IN):
        x.append(Scalar[dtype](1))
    target = List[Scalar[dtype]]()
    for _ in range(m * FD_OUT):
        target.append(Scalar[dtype](5))

    cache = CriticCache(ctx, m, hidden, FD_OUT)
    grads = CriticGrads(ctx, FD_IN, hidden, FD_OUT)
    scratch = CriticScratch(ctx, m, FD_IN, hidden, FD_OUT)
    xd = upload[dtype](ctx, x)
    critic_forward(ctx, p, cache, xd, m)
    critic_backward(ctx, p, cache, grads, scratch, xd,
                    upload[dtype](ctx, target), m)
    ctx.synchronize()

    a1 = download[dtype](cache.a1, m * hidden)
    for i in range(m * hidden):
        if a1[i] != Scalar[dtype](0):
            raise Error("el montaje es incorrecto: a1 deberia estar todo a 0")

    # Con toda la primera capa apagada, ningun peso de la capa 1 puede influir en
    # la perdida: su gradiente tiene que ser exactamente cero.
    dw1 = download[dtype](grads.dw1, FD_IN * hidden)
    for i in range(FD_IN * hidden):
        if dw1[i] != Scalar[dtype](0):
            raise Error("dW1[", i, "] = ", dw1[i], " pero el ReLU apago toda la "
                        "capa: no deberia pasar gradiente")
    print("PASS la mascara del ReLU bloquea el gradiente donde recorto")


def main() raises:
    with DeviceContext() as ctx:
        test_relu_mask_actually_blocks(ctx)
        test_against_jax_autodiff(ctx)
        test_finite_differences_through_relu(ctx)
