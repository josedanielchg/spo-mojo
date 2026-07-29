"""La EMA de los target networks: target <- tau*online + (1-tau)*target.

Es el `optax.incremental_update(online, target, tau)` de Stoix, con tau = 0.005.

Se rompe en silencio de una forma concreta: cambiando el orden de los argumentos.
Con tau tan pequeno, el target apenas deberia moverse; si se movieran casi del
todo, el entrenamiento se veria inestable sin que nada fallara. Por eso los tests
usan numeros exactos y los dos casos limite (tau=1 copia, tau=0 no toca).
"""

from std.gpu.host import DeviceContext
from std.math import abs

from ops.common import dtype
from networks.optim import ema_update
from tests.golden_io import read_f32
from tests.helpers import upload, download

comptime GOLDEN = String("tests/golden/")

comptime TAU = Scalar[dtype](0.005)


def test_ema_moves_slowly(ctx: DeviceContext) raises:
    """La formula: target <- tau*online + (1-tau)*target, con numeros exactos.

    Con target=0, online=1 y tau=0.005, tras un paso el target vale exactamente
    0.005. Que sea TAN pequeno es el punto: si alguien cambiara el orden de los
    argumentos, el target saltaria a 0.995 y se veria al instante.
    """
    n = 5
    tgt = List[Scalar[dtype]]()
    onl = List[Scalar[dtype]]()
    for _ in range(n):
        tgt.append(Scalar[dtype](0))
        onl.append(Scalar[dtype](1))

    target = upload[dtype](ctx, tgt)
    online = upload[dtype](ctx, onl)
    ema_update(ctx, target, online, n, TAU)
    ctx.synchronize()
    got = download[dtype](target, n)
    for i in range(n):
        if abs(got[i] - TAU) > Scalar[dtype](1e-7):
            raise Error("tras un paso el target deberia valer tau (", TAU,
                        "), dio ", got[i])

    # Y tras muchos pasos se acerca a online, sin pasarse.
    for _ in range(200):
        ema_update(ctx, target, online, n, TAU)
    ctx.synchronize()
    got2 = download[dtype](target, n)
    for i in range(n):
        if got2[i] <= got[i] or got2[i] >= Scalar[dtype](1):
            raise Error("tras 200 pasos el target deberia estar entre ", got[i],
                        " y 1, dio ", got2[i])
    print("PASS la EMA mueve el target un tau por paso (tras 201:", got2[0], ")")


def test_ema_with_tau_one_copies(ctx: DeviceContext) raises:
    """Con tau=1 el target se convierte en una copia exacta del online.

    Es el caso limite que confirma la orientacion de la formula sin depender de
    ninguna tolerancia.
    """
    n = 4
    tgt = List[Scalar[dtype]](); onl = List[Scalar[dtype]]()
    for i in range(n):
        tgt.append(Scalar[dtype](-7))
        onl.append(Scalar[dtype](i) * 3.5)

    target = upload[dtype](ctx, tgt)
    ema_update(ctx, target, upload[dtype](ctx, onl), n, Scalar[dtype](1))
    ctx.synchronize()
    got = download[dtype](target, n)
    for i in range(n):
        if got[i] != onl[i]:
            raise Error("con tau=1 el target deberia ser el online: ", got[i],
                        " vs ", onl[i])

    # Y con tau=0 no se mueve nada.
    target2 = upload[dtype](ctx, tgt)
    ema_update(ctx, target2, upload[dtype](ctx, onl), n, Scalar[dtype](0))
    ctx.synchronize()
    got2 = download[dtype](target2, n)
    for i in range(n):
        if got2[i] != Scalar[dtype](-7):
            raise Error("con tau=0 el target no deberia moverse, dio ", got2[i])
    print("PASS tau=1 copia el online y tau=0 no toca nada")


def test_against_optax_multiblock(ctx: DeviceContext) raises:
    """Contra el `optax.incremental_update` de verdad, y con VARIOS BLOQUES.

    Tapa dos huecos de la primera version de este test:

      1. solo se comparaba contra mi propia lectura de la formula, no contra la
         libreria que usa Stoix;
      2. se usaban 4 y 5 elementos, o sea un unico bloque de 256 hilos, asi que
         la ruta multi-bloque nunca se ejecutaba. Es el mismo punto ciego que
         aparecio verificando el backward en E1.4.

    El golden usa n = 1000 (cuatro bloques) y diez pasos encadenados, que ademas
    comprueba que el estado se arrastra bien de un paso al siguiente.
    """
    n = 1000
    target = upload[dtype](ctx, read_f32(GOLDEN + "ema_target0.bin"))
    online = upload[dtype](ctx, read_f32(GOLDEN + "ema_online.bin"))

    worst = Scalar[dtype](0)
    for step in range(1, 11):
        ema_update(ctx, target, online, n, TAU)
        ctx.synchronize()
        got = download[dtype](target, n)
        want = read_f32(GOLDEN + "ema_target" + String(step) + ".bin")
        if len(want) != n:
            raise Error("el golden del paso ", step, " no tiene ", n, " valores")
        for i in range(n):
            d = abs(got[i] - want[i])
            if d > worst:
                worst = d
            if d > Scalar[dtype](1e-6):
                raise Error("paso ", step, " valor ", i, ": ", got[i],
                            " vs optax ", want[i], " (diff ", d, ")")
    print("      n =", n, "(4 bloques), 10 pasos: peor diferencia", worst)
    print("PASS la EMA coincide con optax.incremental_update en varios bloques")


def test_ema_ragged_size(ctx: DeviceContext) raises:
    """Un tamano que no es multiplo del bloque: los hilos de mas no escriben.

    Con n = 300 y bloques de 256 hay un segundo bloque con 212 hilos sobrantes.
    Si el guard fallara, escribirian fuera del buffer.
    """
    n = 300
    tgt = List[Scalar[dtype]]()
    onl = List[Scalar[dtype]]()
    for i in range(n):
        tgt.append(Scalar[dtype](0))
        onl.append(Scalar[dtype](i))

    target = upload[dtype](ctx, tgt)
    ema_update(ctx, target, upload[dtype](ctx, onl), n, Scalar[dtype](1))
    ctx.synchronize()
    got = download[dtype](target, n)
    for i in range(n):
        if got[i] != Scalar[dtype](i):
            raise Error("con tau=1 y n=", n, " el valor ", i, " deberia ser ", i,
                        " y dio ", got[i])
    print("PASS tamano ragged (n =", n, ", 2 bloques) sin desbordes")


def main() raises:
    with DeviceContext() as ctx:
        test_ema_moves_slowly(ctx)
        test_ema_with_tau_one_copies(ctx)
        test_ema_ragged_size(ctx)
        test_against_optax_multiblock(ctx)
