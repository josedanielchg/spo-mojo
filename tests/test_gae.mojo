"""La GAE truncada contra la funcion REAL de Stoix.

Golden: tests/golden/gen/gen_gae.py, que importa
`batch_truncated_generalized_advantage_estimation` del propio repo de Stoix.

Tres casos:
    case0  sin truncacion, con episodios que terminan en medio
    case1  CON truncacion: la rama que casi nunca se ejercita y donde vive el bug
           silencioso clasico de RL
    case2  como el tres en raya de verdad: partidas de 3-5 pasos, recompensa solo
           al final

Ademas del golden hay dos comprobaciones a mano, que no dependen de que Stoix tenga
razon: el caso de un solo paso (donde la GAE se reduce al error TD, calculable a
mano) y el de la truncacion (donde se comprueba que corta el acumulado pero
conserva su propio delta, y que sin truncar SI arrastra).
"""

from std.gpu.host import DeviceContext
from std.math import abs

from ops.common import dtype
from rl_utils.multistep import truncated_gae
from tests.golden_io import read_f32
from tests.helpers import upload, zeros, download

comptime GOLDEN = String("tests/golden/")
comptime GAMMA = Scalar[dtype](0.99)
comptime LAMBDA = Scalar[dtype](0.95)
comptime TOL = Scalar[dtype](1e-5)


def check_case(ctx: DeviceContext, which: Int, b: Int, t: Int) raises:
    """Un caso del golden: ventajas y objetivos."""
    p = GOLDEN + "gae" + String(which) + "_"
    n = b * t

    reward = upload[dtype](ctx, read_f32(p + "r.bin"))
    discount = upload[dtype](ctx, read_f32(p + "discount.bin"))
    v_tm1 = upload[dtype](ctx, read_f32(p + "v_tm1.bin"))
    v_t = upload[dtype](ctx, read_f32(p + "v_t.bin"))
    trunc = upload[dtype](ctx, read_f32(p + "trunc.bin"))
    want_adv = read_f32(p + "adv.bin")
    want_tgt = read_f32(p + "targets.bin")
    if len(want_adv) != n:
        raise Error("el golden del caso ", which, " no tiene la shape esperada")

    adv = zeros[dtype](ctx, n)
    targets = zeros[dtype](ctx, n)
    truncated_gae(ctx, adv, targets, reward, discount, v_tm1, v_t, trunc,
                  b, t, LAMBDA)
    ctx.synchronize()

    got_adv = download[dtype](adv, n)
    got_tgt = download[dtype](targets, n)

    worst_a = Scalar[dtype](0)
    worst_t = Scalar[dtype](0)
    for i in range(n):
        da = abs(got_adv[i] - want_adv[i])
        dt = abs(got_tgt[i] - want_tgt[i])
        if da > worst_a:
            worst_a = da
        if dt > worst_t:
            worst_t = dt
        if da > TOL:
            raise Error("case", which, " ventaja[", i, "]: ", got_adv[i],
                        " vs stoix ", want_adv[i], " (diff ", da, ")")
        if dt > TOL:
            raise Error("case", which, " objetivo[", i, "]: ", got_tgt[i],
                        " vs stoix ", want_tgt[i], " (diff ", dt, ")")
    print("      case", which, " B=", b, " T=", t,
          " error max: ventajas", worst_a, " objetivos", worst_t)


def test_against_stoix(ctx: DeviceContext) raises:
    """Los tres casos contra la implementacion de Stoix."""
    check_case(ctx, 0, 4, 16)     # sin truncacion
    check_case(ctx, 1, 4, 16)     # CON truncacion
    check_case(ctx, 2, 8, 32)     # como tres en raya
    print("PASS la GAE truncada coincide con la de Stoix en los tres casos")


def test_single_step_is_td_error(ctx: DeviceContext) raises:
    """Con T=1 la GAE se reduce al error TD, calculable a mano.

    No depende de ningun golden: con un solo paso no hay nada que acumular, asi
    que la ventaja tiene que ser exactamente r + gamma*v_t - v_tm1.
    """
    r = List[Scalar[dtype]](); r.append(2.0)
    d = List[Scalar[dtype]](); d.append(GAMMA)
    vtm1 = List[Scalar[dtype]](); vtm1.append(0.5)
    vt = List[Scalar[dtype]](); vt.append(3.0)
    tr = List[Scalar[dtype]](); tr.append(0.0)

    adv = zeros[dtype](ctx, 1)
    targets = zeros[dtype](ctx, 1)
    truncated_gae(ctx, adv, targets, upload[dtype](ctx, r), upload[dtype](ctx, d),
                  upload[dtype](ctx, vtm1), upload[dtype](ctx, vt),
                  upload[dtype](ctx, tr), 1, 1, LAMBDA)
    ctx.synchronize()

    want_adv = Scalar[dtype](2.0) + GAMMA * Scalar[dtype](3.0) - Scalar[dtype](0.5)
    got_adv = download[dtype](adv, 1)[0]
    got_tgt = download[dtype](targets, 1)[0]
    if abs(got_adv - want_adv) > TOL:
        raise Error("con T=1 la ventaja deberia ser el error TD ", want_adv,
                    ", dio ", got_adv)
    if abs(got_tgt - (Scalar[dtype](0.5) + want_adv)) > TOL:
        raise Error("el objetivo deberia ser v_tm1 + ventaja")
    print("PASS con un solo paso la GAE es exactamente el error TD")


def test_truncation_cuts_the_accumulator(ctx: DeviceContext) raises:
    """En un paso truncado, la ventaja es SOLO su delta: no arrastra lo de despues.

    Es el detalle fino de la implementacion de Stoix ("reset accumulator at
    truncation points while still using the current delta") y el que separa una
    GAE correcta de una que trata la truncacion como si fuera terminal.

    Montaje: dos pasos, el primero truncado. Su ventaja tiene que ser su propio
    delta, sin rastro del segundo paso.
    """
    t_len = 2
    r = List[Scalar[dtype]](); r.append(1.0); r.append(10.0)
    d = List[Scalar[dtype]](); d.append(GAMMA); d.append(GAMMA)
    vtm1 = List[Scalar[dtype]](); vtm1.append(0.0); vtm1.append(0.0)
    vt = List[Scalar[dtype]](); vt.append(0.0); vt.append(0.0)
    tr = List[Scalar[dtype]](); tr.append(1.0); tr.append(0.0)   # el paso 0 truncado

    adv = zeros[dtype](ctx, t_len)
    targets = zeros[dtype](ctx, t_len)
    truncated_gae(ctx, adv, targets, upload[dtype](ctx, r), upload[dtype](ctx, d),
                  upload[dtype](ctx, vtm1), upload[dtype](ctx, vt),
                  upload[dtype](ctx, tr), 1, t_len, LAMBDA)
    ctx.synchronize()
    got = download[dtype](adv, t_len)

    # El paso 1 (el ultimo) no tiene nada detras: su ventaja es su delta = 10.
    if abs(got[1] - Scalar[dtype](10)) > TOL:
        raise Error("el ultimo paso deberia valer su delta (10), dio ", got[1])
    # El paso 0 esta truncado: su delta es 1, y NO debe arrastrar el 10 de detras.
    if abs(got[0] - Scalar[dtype](1)) > TOL:
        raise Error("el paso truncado deberia valer solo su delta (1) y no "
                    "arrastrar lo de despues; dio ", got[0])

    # Y para que se vea que la comprobacion distingue: sin truncar, el mismo
    # montaje SI arrastra.
    tr2 = List[Scalar[dtype]](); tr2.append(0.0); tr2.append(0.0)
    adv2 = zeros[dtype](ctx, t_len)
    targets2 = zeros[dtype](ctx, t_len)
    truncated_gae(ctx, adv2, targets2, upload[dtype](ctx, r),
                  upload[dtype](ctx, d), upload[dtype](ctx, vtm1),
                  upload[dtype](ctx, vt), upload[dtype](ctx, tr2), 1, t_len,
                  LAMBDA)
    ctx.synchronize()
    got2 = download[dtype](adv2, t_len)
    want2 = Scalar[dtype](1) + GAMMA * LAMBDA * Scalar[dtype](10)
    if abs(got2[0] - want2) > TOL:
        raise Error("sin truncar, el paso 0 deberia arrastrar: esperaba ", want2,
                    " y dio ", got2[0])
    print("PASS la truncacion corta el acumulado pero conserva su propio delta")


def main() raises:
    with DeviceContext() as ctx:
        test_single_step_is_td_error(ctx)
        test_truncation_cuts_the_accumulator(ctx)
        test_against_stoix(ctx)
