"""The target networks' EMA: target <- tau*online + (1-tau)*target.

It is Stoix's `optax.incremental_update(online, target, tau)`, with tau = 0.005.

It breaks silently in one specific way: by swapping the argument order. With tau
this small, the target should barely move; if it moved almost all the way,
training would look unstable without anything failing. That is why the tests use
exact numbers and both limiting cases (tau=1 copies, tau=0 touches nothing).
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
    """The formula: target <- tau*online + (1-tau)*target, with exact numbers.

    With target=0, online=1 and tau=0.005, after one step the target is exactly
    0.005. Its being THAT small is the point: if somebody swapped the argument
    order, the target would jump to 0.995 and it would show up instantly.
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

    # And after many steps it approaches online, without overshooting.
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
    """With tau=1 the target becomes an exact copy of the online one.

    It is the limiting case that confirms the formula's orientation without
    depending on any tolerance.
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

    # And with tau=0 nothing moves.
    target2 = upload[dtype](ctx, tgt)
    ema_update(ctx, target2, upload[dtype](ctx, onl), n, Scalar[dtype](0))
    ctx.synchronize()
    got2 = download[dtype](target2, n)
    for i in range(n):
        if got2[i] != Scalar[dtype](-7):
            raise Error("con tau=0 el target no deberia moverse, dio ", got2[i])
    print("PASS tau=1 copia el online y tau=0 no toca nada")


def test_against_optax_multiblock(ctx: DeviceContext) raises:
    """Against the real `optax.incremental_update`, and with SEVERAL BLOCKS.

    It plugs two holes in this test's first version:

      1. it only compared against my own reading of the formula, not against the
         library Stoix uses;
      2. it used 4 and 5 elements, that is, a single block of 256 threads, so the
         multi-block path was never executed. It is the same blind spot that turned
         up while verifying the backward in E1.4.

    The golden uses n = 1000 (four blocks) and ten chained steps, which also checks
    that the state carries correctly from one step to the next.
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
    """A size that is not a multiple of the block: the extra threads do not write.

    With n = 300 and blocks of 256 there is a second block with 212 leftover
    threads. If the guard failed, they would write outside the buffer.
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
