"""La entropia cruzada ponderada del M-step, contra el golden.

El golden no es "lo que yo creo que da la ecuacion 11": lo genera llamando a
`compute_cross_entropy_loss` de Stoix DE VERDAD (importada, no reescrita) en su
forma de particulas, y comprueba ahi mismo que coincide con la forma densa que
implementamos aqui. Las diferencias medidas fueron de 1e-9 a 5e-8, o sea ruido de
float32.

Eso importa porque la eleccion de la forma densa podria parecer una desviacion, y
no lo es: como las acciones raiz se repiten entre particulas,

    SUM_n w_n log pi(a_n)  =  SUM_a q(a) log pi(a)

es la misma suma reagrupada. El golden convierte ese argumento en un numero.
"""

from std.gpu.host import DeviceContext
from std.math import log

from ops.buffers import zero_buffer
from ops.common import dtype, NEG_INF
from ops.softmax import log_softmax_rows
from networks.actor_loss import cross_entropy_rows, entropy_rows, kl_rows
from tests.golden_io import read_f32
from tests.helpers import upload, download, filled, assert_close

comptime TOL = Scalar[dtype](2e-5)
comptime GOLDEN = String("tests/golden/")
comptime NUM_ACTIONS = 9
comptime TPB_ROW = 32


def mean_of(values: List[Scalar[dtype]], n: Int) -> Scalar[dtype]:
    total = Scalar[dtype](0)
    for i in range(n):
        total += values[i]
    return total / Scalar[dtype](n)


def check_case(ctx: DeviceContext, name: String, batch: Int) raises:
    """Un caso del golden: del log-softmax a la perdida por estado y a la media."""
    tag = GOLDEN + "ce_" + name + "_"
    logits = upload[dtype](ctx, read_f32(tag + "logits.bin"))
    q = upload[dtype](ctx, read_f32(tag + "q.bin"))
    mask = read_f32(tag + "mask.bin")

    # 1. log pi, calculado directo y no como log(softmax).
    log_pi = zero_buffer[dtype](ctx, batch * NUM_ACTIONS)
    ctx.enqueue_function[log_softmax_rows[TPB_ROW], log_softmax_rows[TPB_ROW]](
        log_pi.unsafe_ptr(), logits.unsafe_ptr(), NUM_ACTIONS,
        grid_dim=batch, block_dim=TPB_ROW)
    ctx.synchronize()

    want_logpi = read_f32(tag + "logpi.bin")
    got_logpi = download[dtype](log_pi, batch * NUM_ACTIONS)
    for i in range(batch * NUM_ACTIONS):
        if mask[i] == Scalar[dtype](0):
            # En las ilegales solo se exige que sea "efectivamente menos
            # infinito". El patron de bits exacto de un valor saturado no es el
            # contrato; que no contribuya al log, si.
            if got_logpi[i] > Scalar[dtype](-1e30):
                raise Error(name, ": el log pi de la casilla ilegal ", i,
                            " deberia ser muy negativo y vale ", got_logpi[i])
        else:
            assert_close(got_logpi[i], want_logpi[i], TOL,
                         String(name, " log pi ", i))

    # 2. La perdida de cada estado y su media.
    per_state = zero_buffer[dtype](ctx, batch)
    cross_entropy_rows(ctx, per_state, q, log_pi, batch, NUM_ACTIONS)
    ctx.synchronize()

    want_per = read_f32(tag + "per_state.bin")
    got_per = download[dtype](per_state, batch)
    for b in range(batch):
        assert_close(got_per[b], want_per[b], TOL,
                     String(name, " perdida del estado ", b))

    # 3. La descomposicion H(q,pi) = H(q) + KL(q||pi).
    h = zero_buffer[dtype](ctx, batch)
    kl = zero_buffer[dtype](ctx, batch)
    entropy_rows(ctx, h, q, batch, NUM_ACTIONS)
    kl_rows(ctx, kl, per_state, h, batch)
    ctx.synchronize()

    want_h = read_f32(tag + "entropy.bin")
    want_kl = read_f32(tag + "kl.bin")
    got_h = download[dtype](h, batch)
    got_kl = download[dtype](kl, batch)
    for b in range(batch):
        assert_close(got_h[b], want_h[b], TOL, String(name, " H(q) del estado ", b))
        assert_close(got_kl[b], want_kl[b], TOL, String(name, " KL del estado ", b))
        # La KL es una divergencia: nunca negativa.
        if got_kl[b] < Scalar[dtype](-1e-5):
            raise Error(name, ": KL negativa en el estado ", b, ": ", got_kl[b])
        # Y la descomposicion tiene que cerrar exactamente.
        assert_close(got_h[b] + got_kl[b], got_per[b], TOL,
                     String(name, " H(q)+KL deberia dar la perdida en ", b))

    want_loss = read_f32(tag + "loss.bin")
    got_loss = mean_of(got_per, batch)
    assert_close(got_loss, want_loss[0], TOL, String(name, " perdida media"))
    print("PASS ce ", name, " (B=", batch, "): perdida ", got_loss,
          " = H(q) ", mean_of(got_h, batch), " + KL ", mean_of(got_kl, batch))


def test_matches_stoix_golden(ctx: DeviceContext) raises:
    """Los tres casos: batch pequeno, batch ragged, y el N grande que usamos.

    El golden de cada uno se genero comparando la forma densa contra
    `compute_cross_entropy_loss` de Stoix, asi que pasar esto es pasar las dos.
    """
    check_case(ctx, "small", 3)
    check_case(ctx, "mid", 7)
    check_case(ctx, "big", 32)
    # 70 filas son tres bloques con TPB=32 y el ultimo a medias. Sin este caso el
    # guard `row >= n_rows` del kernel no se ejecutaria nunca: todos los demas
    # batches caben en un bloque. Es el mismo punto ciego que ya me comi en E1.7 y
    # E1.9, asi que ahora va en la lista fija.
    check_case(ctx, "multiblock", 70)


def test_zero_weight_terms_do_not_produce_nan(ctx: DeviceContext) raises:
    """Un termino con q = 0 y log pi = -inf DE VERDAD no da NaN.

    En el golden las casillas ilegales quedan en el float32 finito mas negativo,
    asi que 0 * eso da 0 y el guard del kernel no llega a demostrarse. Aqui se
    fuerza el caso peor: un -inf autentico. Sin el `if w != 0`, IEEE da
    0 * (-inf) = NaN, y ese NaN no falla ruidosamente -- se propaga a la perdida,
    al gradiente y a los pesos, y el entrenamiento se rompe sin que nadie sepa
    donde.

    Es exactamente el tipo de fallo que este proyecto ya se comio una vez (el bug
    del workspace en A6), asi que va con prueba.
    """
    batch = 2
    inf_neg = Scalar[dtype](-1) / Scalar[dtype](0)     # -inf autentico

    lp = List[Scalar[dtype]]()
    qs = List[Scalar[dtype]]()
    for _ in range(batch):
        for a in range(NUM_ACTIONS):
            legal = a < 3                      # solo las tres primeras
            lp.append(Scalar[dtype](-1.0986123) if legal else inf_neg)
            qs.append(Scalar[dtype](1) / Scalar[dtype](3) if legal
                      else Scalar[dtype](0))
    log_pi = upload[dtype](ctx, lp)
    q = upload[dtype](ctx, qs)
    per_state = filled[dtype](ctx, batch, Scalar[dtype](-99))

    cross_entropy_rows(ctx, per_state, q, log_pi, batch, NUM_ACTIONS)
    ctx.synchronize()
    got = download[dtype](per_state, batch)

    for b in range(batch):
        if got[b] != got[b]:                   # nan != nan
            raise Error("el estado ", b, " dio NaN: falta el guard de q == 0")
        # Tres acciones equiprobables: -SUM (1/3)*log(1/3) = log(3) = 1.0986123
        assert_close(got[b], Scalar[dtype](1.0986123), TOL,
                     String("perdida del estado ", b))
    print("PASS los terminos con q=0 se saltan y un -inf de verdad no da NaN")


def loss_of(ctx: DeviceContext, log_pi_vals: List[Scalar[dtype]],
            q_vals: List[Scalar[dtype]]) raises -> Scalar[dtype]:
    """La perdida de un solo estado con 3 acciones legales y 6 tapadas.

    A nivel de modulo y no anidada porque en Mojo 1.0.0b1 una funcion anidada no
    puede capturar `ctx` ni una `List` del ambito de fuera.
    """
    lp = List[Scalar[dtype]]()
    qs = List[Scalar[dtype]]()
    for a in range(NUM_ACTIONS):
        if a < 3:
            lp.append(log_pi_vals[a]); qs.append(q_vals[a])
        else:
            lp.append(NEG_INF); qs.append(Scalar[dtype](0))
    log_pi = upload[dtype](ctx, lp)
    q = upload[dtype](ctx, qs)
    out = zero_buffer[dtype](ctx, 1)
    cross_entropy_rows(ctx, out, q, log_pi, 1, NUM_ACTIONS)
    ctx.synchronize()
    return download[dtype](out, 1)[0]


def test_loss_is_minimised_when_pi_equals_q(ctx: DeviceContext) raises:
    """La perdida es minima cuando pi coincide con q, y crece al alejarse.

    Es la propiedad que hace que optimizar esto tenga sentido: la ecuacion 11
    proyecta q sobre la red, asi que el minimo tiene que estar en pi = q. Si el
    signo estuviera invertido, o si se hubiera colado un log de mas, la perdida
    seria minima en otro sitio y el entrenamiento empujaria al reves -- y eso NO
    lo detecta un golden, porque un golden solo comprueba un punto.

    Se compara la perdida en pi = q contra la entropia de q (su valor exacto en el
    minimo) y contra dos politicas desviadas.
    """
    q_vals = List[Scalar[dtype]]()
    q_vals.append(0.5); q_vals.append(0.3); q_vals.append(0.2)

    # pi = q: la perdida vale exactamente la entropia de q.
    at_min = List[Scalar[dtype]]()
    entropy = Scalar[dtype](0)
    for a in range(3):
        at_min.append(log(q_vals[a]))
        entropy += -q_vals[a] * log(q_vals[a])
    got_min = loss_of(ctx, at_min, q_vals)
    assert_close(got_min, entropy, Scalar[dtype](1e-5),
                 "en pi = q la perdida tiene que valer la entropia de q")

    # Dos desviaciones: la uniforme, y otra que invierte el orden de preferencia.
    uniform = List[Scalar[dtype]]()
    for _ in range(3):
        uniform.append(log(Scalar[dtype](1) / Scalar[dtype](3)))
    got_uniform = loss_of(ctx, uniform, q_vals)

    flipped = List[Scalar[dtype]]()
    flipped.append(log(Scalar[dtype](0.2)))
    flipped.append(log(Scalar[dtype](0.3)))
    flipped.append(log(Scalar[dtype](0.5)))
    got_flipped = loss_of(ctx, flipped, q_vals)

    if got_uniform <= got_min:
        raise Error("la uniforme deberia perder mas que pi=q: ", got_uniform,
                    " vs ", got_min)
    if got_flipped <= got_uniform:
        raise Error("invertir el orden deberia ser peor que la uniforme: ",
                    got_flipped, " vs ", got_uniform)
    print("PASS la perdida es minima en pi=q (=", got_min,
          "), uniforme ", got_uniform, ", invertida ", got_flipped)


def test_kl_is_zero_exactly_when_pi_equals_q(ctx: DeviceContext) raises:
    """La KL vale 0 en pi=q y es positiva en cuanto se aparta.

    Es lo que hace util el diagnostico: la ENTROPIA CRUZADA en pi=q vale H(q), que
    puede ser cualquier numero y ademas cambia entre iteraciones porque q la
    produce la busqueda. La KL tiene un cero con significado -- "el actor
    reproduce exactamente lo que dice la busqueda" -- y ese cero es el mismo en
    todas las configuraciones, asi que si se puede comparar entre brazos.
    """
    q_vals = List[Scalar[dtype]]()
    q_vals.append(0.5); q_vals.append(0.3); q_vals.append(0.2)

    at_min = List[Scalar[dtype]]()
    for a in range(3):
        at_min.append(log(q_vals[a]))
    uniform = List[Scalar[dtype]]()
    for _ in range(3):
        uniform.append(log(Scalar[dtype](1) / Scalar[dtype](3)))

    ce_min = loss_of(ctx, at_min, q_vals)
    ce_uni = loss_of(ctx, uniform, q_vals)

    # H(q) con el kernel, sobre el mismo vector relleno a 9 acciones.
    qs = List[Scalar[dtype]]()
    for a in range(NUM_ACTIONS):
        qs.append(q_vals[a] if a < 3 else Scalar[dtype](0))
    q = upload[dtype](ctx, qs)
    h = zero_buffer[dtype](ctx, 1)
    entropy_rows(ctx, h, q, 1, NUM_ACTIONS)
    ctx.synchronize()
    got_h = download[dtype](h, 1)[0]

    # H(q) a mano: 1.0296531 para (0.5, 0.3, 0.2).
    assert_close(got_h, Scalar[dtype](1.0296531), TOL, "H(q) de (0.5,0.3,0.2)")
    assert_close(ce_min - got_h, Scalar[dtype](0), Scalar[dtype](1e-5),
                 "la KL en pi=q tiene que ser 0")
    if ce_uni - got_h <= Scalar[dtype](0):
        raise Error("la KL de la uniforme deberia ser positiva: ", ce_uni - got_h)
    print("PASS KL = 0 en pi=q (perdida ", ce_min, " = H(q) ", got_h,
          ") y ", ce_uni - got_h, " con la uniforme")


def test_entropy_edge_cases(ctx: DeviceContext) raises:
    """H de una one-hot es 0 y H de una uniforme sobre k es log(k).

    Los dos extremos, que son los que fijan la escala. El one-hot es ademas el
    caso peligroso: q vale 1 en una casilla y 0 en las otras ocho, asi que si el
    guard `w > 0` faltara habria ocho terminos 0*log(0) = NaN y H saldria NaN en
    vez de 0. Y es un caso REAL: la q de nuestro readout corregido sale casi
    one-hot (medimos 0.9999999).
    """
    # Fila 0: one-hot. Fila 1: uniforme sobre 3. Fila 2: uniforme sobre las 9.
    qs = List[Scalar[dtype]]()
    for a in range(NUM_ACTIONS):
        qs.append(Scalar[dtype](1) if a == 4 else Scalar[dtype](0))
    for a in range(NUM_ACTIONS):
        qs.append(Scalar[dtype](1) / Scalar[dtype](3) if a < 3
                  else Scalar[dtype](0))
    for _ in range(NUM_ACTIONS):
        qs.append(Scalar[dtype](1) / Scalar[dtype](NUM_ACTIONS))

    q = upload[dtype](ctx, qs)
    h = zero_buffer[dtype](ctx, 3)
    entropy_rows(ctx, h, q, 3, NUM_ACTIONS)
    ctx.synchronize()
    got = download[dtype](h, 3)

    if got[0] != got[0]:
        raise Error("H de una one-hot salio NaN: falta el guard de q == 0")
    assert_close(got[0], Scalar[dtype](0), TOL, "H de una one-hot")
    assert_close(got[1], log(Scalar[dtype](3)), TOL, "H de la uniforme sobre 3")
    assert_close(got[2], log(Scalar[dtype](NUM_ACTIONS)), TOL,
                 "H de la uniforme sobre 9")
    print("PASS H(one-hot)=0 sin NaN, H(uniforme sobre k)=log k")


def main() raises:
    with DeviceContext() as ctx:
        test_matches_stoix_golden(ctx)
        test_zero_weight_terms_do_not_produce_nan(ctx)
        test_loss_is_minimised_when_pi_equals_q(ctx)
        test_kl_is_zero_exactly_when_pi_equals_q(ctx)
        test_entropy_edge_cases(ctx)
