"""El replay buffer de trayectorias: FIFO, campos y muestreo.

No tiene matematica interesante, pero si una forma tipica de romperse en silencio:
perder o mezclar secuencias al dar la vuelta. Asi que se comprueba con valores
marcados y numeros exactos, no con tolerancias.
"""

from ops.common import dtype
from rl_utils.buffer import TrajectoryBuffer

comptime T_LEN = 4
comptime OBS_DIM = 3
comptime N_ACT = 4


def seq_obs(value: Int) -> List[Scalar[dtype]]:
    """Una secuencia de observaciones marcadas con `value`, para reconocerlas."""
    out = List[Scalar[dtype]]()
    for i in range(T_LEN * OBS_DIM):
        out.append(Scalar[dtype](value * 100 + i))
    return out^


def q_for(value: Int) -> List[Scalar[dtype]]:
    """Una q marcada con `value`, para reconocerla al salir. A nivel de modulo:
    una funcion anidada no puede capturar variables del ambito de fuera."""
    out = List[Scalar[dtype]]()
    for i in range(T_LEN * N_ACT):
        out.append(Scalar[dtype](value * 1000 + i))
    return out^


def seq_steps(value: Int) -> List[Scalar[dtype]]:
    out = List[Scalar[dtype]]()
    for i in range(T_LEN):
        out.append(Scalar[dtype](value * 10 + i))
    return out^


def add_marked(mut buf: TrajectoryBuffer, value: Int) raises:
    """Mete una secuencia entera marcada con `value`."""
    buf.add(seq_obs(value), seq_steps(value), seq_steps(value),
            seq_steps(value), seq_obs(value + 1000))


def test_buffer_fifo_and_wraparound() raises:
    """Al llenarse, las secuencias nuevas pisan las mas viejas, en orden.

    Con capacidad 3 y cinco secuencias metidas (1..5), tienen que quedar las tres
    ultimas (3, 4, 5) y en los huecos correctos. Es donde un ring buffer mal
    escrito pierde datos o los deja desordenados.
    """
    buf = TrajectoryBuffer(3, T_LEN, OBS_DIM)
    if buf.size() != 0:
        raise Error("un buffer recien creado deberia estar vacio")

    for v in range(1, 4):
        add_marked(buf, v)
    if buf.size() != 3 or not buf.is_full():
        raise Error("tras 3 secuencias con capacidad 3 deberia estar lleno")

    # Dos mas: pisan a la 1 y a la 2.
    add_marked(buf, 4)
    add_marked(buf, 5)
    if buf.size() != 3:
        raise Error("la capacidad no puede crecer: ", buf.size())

    # El hueco 0 tiene ahora la 4, el 1 la 5, y el 2 sigue con la 3.
    want = List[Int](); want.append(4); want.append(5); want.append(3)
    for slot in range(3):
        idx = List[Int](); idx.append(slot)
        got = buf.gather(idx)
        expected = seq_obs(want[slot])
        for i in range(T_LEN * OBS_DIM):
            if got[i] != expected[i]:
                raise Error("el hueco ", slot, " deberia tener la secuencia ",
                            want[slot], " pero el valor ", i, " es ", got[i])
    print("PASS el buffer es FIFO y da la vuelta sin mezclar secuencias")


def test_buffer_fields_dont_cross() raises:
    """Cada campo se guarda en el suyo: reward, done y truncated no se mezclan.

    Se meten con valores distintos a proposito para que un cruce se vea.
    """
    buf = TrajectoryBuffer(2, T_LEN, OBS_DIM)
    r = List[Scalar[dtype]](); d = List[Scalar[dtype]](); tr = List[Scalar[dtype]]()
    for i in range(T_LEN):
        r.append(Scalar[dtype](i) + 0.5)      # 0.5, 1.5, 2.5, 3.5
        d.append(Scalar[dtype](1) if i == T_LEN - 1 else Scalar[dtype](0))
        tr.append(Scalar[dtype](0))
    buf.add(seq_obs(7), r, d, tr, seq_obs(8))

    idx = List[Int](); idx.append(0)
    got_r = buf.gather_steps(idx, 0)
    got_d = buf.gather_steps(idx, 1)
    got_t = buf.gather_steps(idx, 2)
    for i in range(T_LEN):
        if got_r[i] != r[i]:
            raise Error("reward[", i, "] salio ", got_r[i], " y era ", r[i])
        if got_d[i] != d[i]:
            raise Error("done[", i, "] salio ", got_d[i], " y era ", d[i])
        if got_t[i] != tr[i]:
            raise Error("truncated[", i, "] salio ", got_t[i])

    # Y las bootstrap_obs no son las obs.
    got_obs = buf.gather(idx)
    got_boot = buf.gather_bootstrap(idx)
    if got_obs[0] == got_boot[0]:
        raise Error("obs y bootstrap_obs no deberian coincidir en este montaje")
    print("PASS los campos del buffer no se cruzan entre si")


def test_buffer_sampling_is_deterministic() raises:
    """Misma semilla, mismos indices; y siempre dentro de lo que hay guardado."""
    buf = TrajectoryBuffer(8, T_LEN, OBS_DIM)
    for v in range(1, 6):        # 5 secuencias, capacidad 8: no da la vuelta
        add_marked(buf, v)

    a = buf.sample_indices(20, UInt32(123), UInt32(0))
    b = buf.sample_indices(20, UInt32(123), UInt32(0))
    c = buf.sample_indices(20, UInt32(999), UInt32(0))

    for i in range(20):
        if a[i] != b[i]:
            raise Error("la misma semilla deberia dar los mismos indices")
        if a[i] < 0 or a[i] >= buf.size():
            raise Error("indice fuera de las secuencias validas: ", a[i])

    same = True
    for i in range(20):
        if a[i] != c[i]:
            same = False
    if same:
        raise Error("dos semillas distintas dieron exactamente los mismos "
                    "indices: la semilla no se esta usando")
    print("PASS el muestreo es determinista, depende de la semilla y no se sale")


def test_gather_respects_order_and_repeats() raises:
    """`gather` con VARIOS indices, en el orden pedido y con repetidos.

    Los tests anteriores solo pedian un indice cada vez, asi que no comprobaban ni
    el orden ni que un indice repetido salga dos veces. Y repetidos los va a haber:
    el muestreo es CON REEMPLAZO, asi que en un batch real se repiten secuencias.
    """
    buf = TrajectoryBuffer(4, T_LEN, OBS_DIM)
    for v in range(1, 5):
        add_marked(buf, v)          # secuencias 1..4 en los huecos 0..3

    # Orden deliberadamente desordenado, con el 2 repetido.
    idx = List[Int]()
    idx.append(3); idx.append(0); idx.append(2); idx.append(2)

    got = buf.gather(idx)
    span = T_LEN * OBS_DIM
    if len(got) != len(idx) * span:
        raise Error("gather deberia devolver ", len(idx), " secuencias")

    want_values = List[Int]()
    want_values.append(4); want_values.append(1); want_values.append(3)
    want_values.append(3)
    for k in range(len(idx)):
        expected = seq_obs(want_values[k])
        for i in range(span):
            if got[k * span + i] != expected[i]:
                raise Error("la posicion ", k, " del batch deberia ser la "
                            "secuencia ", want_values[k], ", pero el valor ", i,
                            " es ", got[k * span + i])

    # Lo mismo para los campos por paso y para las bootstrap_obs.
    got_r = buf.gather_steps(idx, 0)
    if len(got_r) != len(idx) * T_LEN:
        raise Error("gather_steps deberia devolver ", len(idx), "x", T_LEN)
    for k in range(len(idx)):
        expected = seq_steps(want_values[k])
        for i in range(T_LEN):
            if got_r[k * T_LEN + i] != expected[i]:
                raise Error("gather_steps posicion ", k, " valor ", i)

    got_b = buf.gather_bootstrap(idx)
    for k in range(len(idx)):
        expected = seq_obs(want_values[k] + 1000)
        for i in range(span):
            if got_b[k * span + i] != expected[i]:
                raise Error("gather_bootstrap posicion ", k, " valor ", i)
    print("PASS gather respeta el orden pedido y admite indices repetidos")


def test_sampling_more_than_stored() raises:
    """Pedir mas muestras que secuencias guardadas funciona (hay reemplazo).

    En el entrenamiento real el batch (32) puede ser mayor que lo que hay al
    principio, asi que esto tiene que estar soportado y no salirse de rango.
    """
    buf = TrajectoryBuffer(16, T_LEN, OBS_DIM)
    add_marked(buf, 1)
    add_marked(buf, 2)          # solo 2 secuencias guardadas

    idx = buf.sample_indices(32, UInt32(5), UInt32(0))
    if len(idx) != 32:
        raise Error("deberia devolver 32 indices")
    for i in range(32):
        if idx[i] < 0 or idx[i] >= 2:
            raise Error("indice ", idx[i], " fuera de las 2 secuencias validas")

    # Y el gather de ese batch grande no revienta.
    got = buf.gather(idx)
    if len(got) != 32 * T_LEN * OBS_DIM:
        raise Error("el gather del batch grande no tiene el tamano esperado")
    print("PASS se pueden pedir mas muestras que secuencias (con reemplazo)")


def test_buffer_rejects_bad_input() raises:
    """Una secuencia con el tamano equivocado se rechaza en vez de corromper.

    Sin esta comprobacion, meter una secuencia corta escribiria basura en los
    pasos que faltan y nadie se enteraria hasta ver el entrenamiento raro.
    """
    buf = TrajectoryBuffer(2, T_LEN, OBS_DIM)
    short = List[Scalar[dtype]]()
    for _ in range(T_LEN - 1):
        short.append(Scalar[dtype](0))

    failed = False
    try:
        buf.add(seq_obs(1), short, seq_steps(1), seq_steps(1), seq_obs(2))
    except:
        failed = True
    if not failed:
        raise Error("deberia rechazar una secuencia con menos pasos de la cuenta")

    # Y un buffer vacio no deja muestrear.
    empty = TrajectoryBuffer(2, T_LEN, OBS_DIM)
    failed2 = False
    try:
        _ = empty.sample_indices(1, UInt32(0), UInt32(0))
    except:
        failed2 = True
    if not failed2:
        raise Error("muestrear de un buffer vacio deberia dar error")
    print("PASS el buffer rechaza secuencias mal formadas y el muestreo en vacio")


def test_q_roundtrip_and_validation() raises:
    """La q entra y sale intacta, y una q mal formada se rechaza.

    La q es el objetivo del actor (ecuacion 11). Si se guardara mal, el actor
    aprenderia de basura y NADA fallaria: la perdida bajaria igual, contra un
    objetivo equivocado. De ahi que se compruebe el valor exacto y no una
    tolerancia.
    """
    buf = TrajectoryBuffer(3, T_LEN, OBS_DIM, N_ACT)

    for v in range(1, 4):
        buf.add(seq_obs(v), seq_steps(v), seq_steps(v), seq_steps(v),
                seq_obs(v + 1000), q_for(v))

    idx = List[Int](); idx.append(2); idx.append(0)
    got = buf.gather_q(idx)
    want_v = List[Int](); want_v.append(3); want_v.append(1)
    span = T_LEN * N_ACT
    for k in range(2):
        expected = q_for(want_v[k])
        for i in range(span):
            if got[k * span + i] != expected[i]:
                raise Error("q de la secuencia ", want_v[k], " valor ", i,
                            ": ", got[k * span + i], " != ", expected[i])

    # Una q con el tamano equivocado se rechaza.
    short = List[Scalar[dtype]]()
    for _ in range(span - 1):
        short.append(Scalar[dtype](0))
    failed = False
    try:
        buf.add(seq_obs(9), seq_steps(9), seq_steps(9), seq_steps(9),
                seq_obs(9), short)
    except:
        failed = True
    if not failed:
        raise Error("deberia rechazar una q con menos valores de la cuenta")

    # Y un buffer sin q no deja pedirla.
    plain = TrajectoryBuffer(2, T_LEN, OBS_DIM)
    add_marked(plain, 1)
    failed2 = False
    try:
        _ = plain.gather_q(idx)
    except:
        failed2 = True
    if not failed2:
        raise Error("un buffer creado sin q no deberia dejar pedir gather_q")
    print("PASS la q del buffer va y vuelve intacta, y valida su tamano")


def main() raises:
    test_buffer_fifo_and_wraparound()
    test_buffer_fields_dont_cross()
    test_buffer_sampling_is_deterministic()
    test_gather_respects_order_and_repeats()
    test_sampling_more_than_stored()
    test_buffer_rejects_bad_input()
    test_q_roundtrip_and_validation()
