"""Adam + clip por norma global contra el golden de OPTAX (la libreria de Stoix).

Golden: tests/golden/gen/gen_adam.py, con `optax.chain(clip_by_global_norm(0.5),
adam(3e-4, eps=1e-5))`, que es literalmente la configuracion de ff_spo.py.

Dos casos y tres pasos cada uno:
    case0  gradientes pequenos, la norma global (0.354) no llega al limite (0.5)
    case1  gradientes grandes, norma 43.3 -> el clip recorta

Tres pasos y no uno porque la correccion de sesgo de Adam depende del numero de
paso: con un solo paso, una implementacion SIN correccion daria casi lo mismo.

Se comparan los seis tensores a la vez, que es como se usa de verdad: la norma es
GLOBAL, o sea que el factor del clip depende de todos los gradientes juntos. Un
test tensor a tensor no detectaria el error de recortar cada uno por separado.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs, sqrt

from ops.common import dtype
from networks.optim import (AdamState, adam_step, sum_squares,
                            global_clip_scale)
from tests.golden_io import read_f32
from tests.helpers import upload, zeros, download, write_into

comptime GOLDEN = String("tests/golden/")
comptime IN_DIM = 4
comptime HIDDEN = 5
comptime OUT_DIM = 1
comptime LR = Scalar[dtype](3e-4)
comptime MAX_NORM = Scalar[dtype](0.5)
comptime STEPS = 3
comptime TOL = Scalar[dtype](2e-6)
"""Tolerancia dura: la referencia es exacta y las operaciones son elementwise, asi
que solo puede diferir el redondeo de float32."""


def check_tensor(ctx: DeviceContext, buf: DeviceBuffer[dtype], prefix: String,
                 name: String, step: Int, size: Int,
                 which: Int) raises -> Scalar[dtype]:
    """Un tensor contra el golden de optax tras el paso `step`."""
    got = download[dtype](buf, size)
    want = read_f32(prefix + name + "_p" + String(step) + ".bin")
    worst = Scalar[dtype](0)
    for j in range(size):
        d = abs(got[j] - want[j])
        if d > worst:
            worst = d
        if d > TOL:
            raise Error("case", which, " paso ", step, " ", name, "[", j, "]: ",
                        got[j], " vs optax ", want[j], " (diff ", d, ")")
    return worst


def run_case(ctx: DeviceContext, which: Int) raises:
    """Los tres pasos de un caso, comparando los 6 tensores tras cada uno.

    Los seis van explicitos y no en una lista porque `AdamState` posee buffers de
    device y no es copiable, asi que no cabe en un `List`. Y de paso queda como lo
    tendra el learner de verdad, que trabajara con campos nombrados.
    """
    prefix = GOLDEN + "adam" + String(which) + "_"
    n_w1 = IN_DIM * HIDDEN
    n_b1 = HIDDEN
    n_w2 = HIDDEN * HIDDEN
    n_b2 = HIDDEN
    n_w3 = HIDDEN * OUT_DIM
    n_b3 = OUT_DIM

    p_w1 = upload[dtype](ctx, read_f32(prefix + "w1_p0.bin"))
    p_b1 = upload[dtype](ctx, read_f32(prefix + "b1_p0.bin"))
    p_w2 = upload[dtype](ctx, read_f32(prefix + "w2_p0.bin"))
    p_b2 = upload[dtype](ctx, read_f32(prefix + "b2_p0.bin"))
    p_w3 = upload[dtype](ctx, read_f32(prefix + "w3_p0.bin"))
    p_b3 = upload[dtype](ctx, read_f32(prefix + "b3_p0.bin"))

    g_w1 = upload[dtype](ctx, read_f32(prefix + "w1_g.bin"))
    g_b1 = upload[dtype](ctx, read_f32(prefix + "b1_g.bin"))
    g_w2 = upload[dtype](ctx, read_f32(prefix + "w2_g.bin"))
    g_b2 = upload[dtype](ctx, read_f32(prefix + "b2_g.bin"))
    g_w3 = upload[dtype](ctx, read_f32(prefix + "w3_g.bin"))
    g_b3 = upload[dtype](ctx, read_f32(prefix + "b3_g.bin"))

    s_w1 = AdamState(ctx, n_w1)
    s_b1 = AdamState(ctx, n_b1)
    s_w2 = AdamState(ctx, n_w2)
    s_b2 = AdamState(ctx, n_b2)
    s_w3 = AdamState(ctx, n_w3)
    s_b3 = AdamState(ctx, n_b3)

    for step in range(1, STEPS + 1):
        # 1. La norma GLOBAL: suma de cuadrados sobre los SEIS tensores juntos.
        total_sq = (sum_squares(ctx, g_w1, n_w1) + sum_squares(ctx, g_b1, n_b1)
                    + sum_squares(ctx, g_w2, n_w2) + sum_squares(ctx, g_b2, n_b2)
                    + sum_squares(ctx, g_w3, n_w3) + sum_squares(ctx, g_b3, n_b3))
        scale = global_clip_scale(total_sq, MAX_NORM)

        # 2. Y el paso de Adam en cada tensor, con ESE MISMO factor.
        adam_step(ctx, p_w1, g_w1, s_w1, n_w1, LR, scale, step)
        adam_step(ctx, p_b1, g_b1, s_b1, n_b1, LR, scale, step)
        adam_step(ctx, p_w2, g_w2, s_w2, n_w2, LR, scale, step)
        adam_step(ctx, p_b2, g_b2, s_b2, n_b2, LR, scale, step)
        adam_step(ctx, p_w3, g_w3, s_w3, n_w3, LR, scale, step)
        adam_step(ctx, p_b3, g_b3, s_b3, n_b3, LR, scale, step)
        ctx.synchronize()

        # 3. Contra optax, tensor a tensor.
        worst = check_tensor(ctx, p_w1, prefix, "w1", step, n_w1, which)
        for e in [check_tensor(ctx, p_b1, prefix, "b1", step, n_b1, which),
                  check_tensor(ctx, p_w2, prefix, "w2", step, n_w2, which),
                  check_tensor(ctx, p_b2, prefix, "b2", step, n_b2, which),
                  check_tensor(ctx, p_w3, prefix, "w3", step, n_w3, which),
                  check_tensor(ctx, p_b3, prefix, "b3", step, n_b3, which)]:
            if e > worst:
                worst = e
        if step == 1:
            print("      case", which, " norma global", sqrt(total_sq),
                  " factor del clip", scale)
        print("        paso", step, ": peor diferencia", worst)


def test_against_optax(ctx: DeviceContext) raises:
    """Los dos casos: sin clip y con clip."""
    run_case(ctx, 0)
    run_case(ctx, 1)
    print("PASS adam + clip global coinciden con optax en los dos casos")


def test_global_norm_is_global(ctx: DeviceContext) raises:
    """La norma suma TODOS los tensores, no cada uno por su lado.

    Es el error clasico al reimplementar el clip: recortar tensor a tensor cambia
    la DIRECCION del gradiente conjunto; el clip global solo cambia su longitud.
    Con tres tensores de norma 3, 4 y 12, la global es 13 (no 12, ni 3+4+12).
    """
    a = List[Scalar[dtype]](); a.append(3.0)
    b = List[Scalar[dtype]](); b.append(4.0)
    c = List[Scalar[dtype]](); c.append(12.0)

    total = (sum_squares(ctx, upload[dtype](ctx, a), 1)
             + sum_squares(ctx, upload[dtype](ctx, b), 1)
             + sum_squares(ctx, upload[dtype](ctx, c), 1))
    norm = sqrt(total)
    if abs(norm - Scalar[dtype](13)) > Scalar[dtype](1e-5):
        raise Error("la norma global de (3,4,12) deberia ser 13, dio ", norm)

    # Y el factor recorta hasta el limite exacto, no mas.
    scale = global_clip_scale(total, Scalar[dtype](6.5))
    if abs(scale - Scalar[dtype](0.5)) > Scalar[dtype](1e-6):
        raise Error("con norma 13 y limite 6.5 el factor deberia ser 0.5, dio ",
                    scale)
    # Por debajo del limite no toca nada.
    if global_clip_scale(total, Scalar[dtype](20)) != Scalar[dtype](1):
        raise Error("si la norma cabe, el factor tiene que ser exactamente 1")
    print("PASS la norma es global (3,4,12 -> 13) y el clip escala al limite")


def test_sum_squares_multi_block(ctx: DeviceContext) raises:
    """La suma de cuadrados con mas elementos que un bloque.

    Con 1000 elementos hacen falta 4 bloques de 256, asi que se ejercita la
    reduccion parcial y la suma final en el host. Todos a 1, para que el resultado
    sea exactamente el numero de elementos.
    """
    n = 1000
    ones = List[Scalar[dtype]]()
    for _ in range(n):
        ones.append(Scalar[dtype](1))
    got = sum_squares(ctx, upload[dtype](ctx, ones), n)
    if abs(got - Scalar[dtype](n)) > Scalar[dtype](1e-3):
        raise Error("la suma de 1000 unos al cuadrado deberia ser 1000, dio ", got)
    print("PASS suma de cuadrados correcta con varios bloques (n =", n, ")")


def main() raises:
    with DeviceContext() as ctx:
        test_sum_squares_multi_block(ctx)
        test_global_norm_is_global(ctx)
        test_against_optax(ctx)
