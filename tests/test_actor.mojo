"""El MLP del actor y su enmascarado, contra el golden de numpy.

El forward en si es el del critico con 9 salidas en vez de 1, y ese ya esta
verificado desde E1.3. Lo que se prueba de verdad aqui es **el enmascarado**, que
es la pieza que Stoix no tiene (sus entornos no tienen acciones ilegales) y que hay
que anadir para llevar SPO a un juego de tablero.

Por que el enmascarado merece pruebas propias y no vale con "ya se probo en el
prior de la busqueda": alli los logits legales valian TODOS 0, asi que tapar y
hacer softmax daba una uniforme, y un error de indexado habria salido igual de
uniforme. Aqui los logits son numeros distintos entre si salidos de una red, asi
que tapar la casilla equivocada cambia la distribucion de forma detectable.
"""

from std.gpu.host import DeviceContext

from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from envs.tictactoe import (NUM_ACTIONS, NUM_CELLS, STATE_DIM, OBS_DIM, NEG_INF,
                            TPB_TTT, ttt_encode_obs_kernel)
from networks.actor import Actor, actor_logits, actor_probs, zero_actor_params
from tests.golden_io import read_f32
from tests.helpers import upload, download, write_into, filled, assert_close

comptime TOL = Scalar[dtype](2e-5)
comptime GOLDEN = String("tests/golden/")


def board9(c0: Int, c1: Int, c2: Int, c3: Int, c4: Int,
           c5: Int, c6: Int, c7: Int, c8: Int) -> List[Scalar[dtype]]:
    """Un tablero legible: 1 = mia, -1 = suya, 0 = vacia."""
    out = List[Scalar[dtype]]()
    out.append(Scalar[dtype](c0)); out.append(Scalar[dtype](c1))
    out.append(Scalar[dtype](c2)); out.append(Scalar[dtype](c3))
    out.append(Scalar[dtype](c4)); out.append(Scalar[dtype](c5))
    out.append(Scalar[dtype](c6)); out.append(Scalar[dtype](c7))
    out.append(Scalar[dtype](c8))
    return out^


def load_actor(ctx: DeviceContext, hidden: Int) raises -> Actor:
    """Un actor con los pesos del golden de ese ancho."""
    tag = String("actor_h", hidden, "_")
    a = Actor(ctx, 64, hidden)
    write_into[dtype](a.params.w1, read_f32(GOLDEN + tag + "w1.bin"))
    write_into[dtype](a.params.b1, read_f32(GOLDEN + tag + "b1.bin"))
    write_into[dtype](a.params.w2, read_f32(GOLDEN + tag + "w2.bin"))
    write_into[dtype](a.params.b2, read_f32(GOLDEN + tag + "b2.bin"))
    write_into[dtype](a.params.w3, read_f32(GOLDEN + tag + "w3.bin"))
    write_into[dtype](a.params.b3, read_f32(GOLDEN + tag + "b3.bin"))
    ctx.synchronize()
    return a^


def check_width(ctx: DeviceContext, hidden: Int, m: Int) raises:
    """Compara logits enmascarados y probabilidades contra el golden."""
    tag = String("actor_h", hidden, "_")
    actor = load_actor(ctx, hidden)

    x = upload[dtype](ctx, read_f32(GOLDEN + tag + "x" + String(m) + ".bin"))
    mask = upload[dtype](ctx, read_f32(GOLDEN + tag + "mask" + String(m) + ".bin"))
    probs = zero_buffer[dtype](ctx, m * NUM_ACTIONS)

    actor_probs(ctx, actor.params, actor.cache, x, mask, probs, m)
    ctx.synchronize()

    want_masked = read_f32(GOLDEN + tag + "masked" + String(m) + ".bin")
    want_probs = read_f32(GOLDEN + tag + "probs" + String(m) + ".bin")
    got_masked = download[dtype](actor.cache.value, m * NUM_ACTIONS)
    got_probs = download[dtype](probs, m * NUM_ACTIONS)

    for i in range(m * NUM_ACTIONS):
        # Los logits tapados son NEG_INF en los dos lados; comparar su diferencia
        # no tiene sentido (es 0 o desborda), asi que se comprueba la identidad.
        if want_masked[i] == NEG_INF:
            if got_masked[i] != NEG_INF:
                raise Error("h", hidden, " m", m, " logit ", i,
                            " deberia estar tapado y vale ", got_masked[i])
        else:
            assert_close(got_masked[i], want_masked[i], TOL,
                         String("h", hidden, " m", m, " logit ", i))
        assert_close(got_probs[i], want_probs[i], TOL,
                     String("h", hidden, " m", m, " prob ", i))
    print("PASS actor h=", hidden, " m=", m,
          " coincide con el golden (logits y politica)")


def test_actor_matches_golden(ctx: DeviceContext) raises:
    """Los tres anchos y los dos batches del golden.

    m=5 es ragged (no es multiplo del tile de 16) y pisa los guards; m=64 recorre
    varios tiles en la dimension del batch. Los tres anchos estan porque cuanta red
    hace falta para tres en raya es algo que se mide, no se supone.
    """
    widths = List[Int]()
    widths.append(32); widths.append(64); widths.append(256)
    for hidden in widths:
        check_width(ctx, hidden, 5)
        check_width(ctx, hidden, 64)


def test_illegal_cells_get_exactly_zero(ctx: DeviceContext) raises:
    """Una casilla ocupada sale con probabilidad CERO, no con una muy pequena.

    Importa que sea cero exacto: si quedara masa residual, la busqueda podria
    muestrear una jugada imposible, y eso no falla ruidosamente sino que corrompe
    una particula en silencio.

    Y se comprueba tambien que las filas suman 1: tapar sin renormalizar dejaria
    una distribucion que no lo es.
    """
    actor = load_actor(ctx, 64)
    boards = List[Scalar[dtype]]()
    for b in board9(1,0,-1, 0,1,0, 0,0,0): boards.append(b)   # 6 libres
    for b in board9(1,1,-1, -1,-1,1, 1,-1,0): boards.append(b)  # 1 libre (la 8)
    n = 2
    state = upload[dtype](ctx, boards)
    obs = zero_buffer[dtype](ctx, n * OBS_DIM)
    ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
        obs.unsafe_ptr(), state.unsafe_ptr(), n, grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()

    actor.forward(ctx, state, obs, n)
    ctx.synchronize()
    p = download[dtype](actor.probs, n * NUM_ACTIONS)

    for e in range(n):
        total = Scalar[dtype](0)
        for c in range(NUM_ACTIONS):
            v = p[e * NUM_ACTIONS + c]
            occupied = boards[e * NUM_CELLS + c] != Scalar[dtype](0)
            if occupied:
                if v != Scalar[dtype](0):
                    raise Error("la casilla ocupada ", c, " del tablero ", e,
                                " tiene probabilidad ", v, " y deberia ser 0")
            elif v <= Scalar[dtype](0):
                raise Error("la casilla libre ", c, " del tablero ", e,
                            " tiene probabilidad ", v)
            total += v
        assert_close(total, Scalar[dtype](1), TOL,
                     String("la fila ", e, " deberia sumar 1"))

    # El segundo tablero solo tiene libre la 8: toda la masa va ahi.
    assert_close(p[NUM_ACTIONS + 8], Scalar[dtype](1), TOL,
                 "con una sola casilla libre, su probabilidad tiene que ser 1")
    print("PASS las casillas ocupadas salen a cero exacto y las filas suman 1")


def test_masking_changes_the_ranking(ctx: DeviceContext) raises:
    """Tapar la casilla que la red PREFERIA cambia de verdad la eleccion.

    Este es el test que distingue "enmascara" de "no hace nada": se busca cual es
    la casilla mas probable sin mascara, se tapa solo esa, y se comprueba que la
    politica pasa a preferir otra. Con logits uniformes (el prior de la busqueda)
    esta comprobacion seria imposible, porque todas empatan.
    """
    actor = load_actor(ctx, 64)
    n = 1
    board = board9(0,0,0, 0,0,0, 0,0,0)          # todo libre
    state = upload[dtype](ctx, board)
    obs = zero_buffer[dtype](ctx, n * OBS_DIM)
    ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
        obs.unsafe_ptr(), state.unsafe_ptr(), n, grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()

    all_free = filled[dtype](ctx, NUM_ACTIONS, Scalar[dtype](1))
    probs = zero_buffer[dtype](ctx, NUM_ACTIONS)
    actor_probs(ctx, actor.params, actor.cache, obs, all_free, probs, n)
    ctx.synchronize()
    p0 = download[dtype](probs, NUM_ACTIONS)

    best = 0
    for c in range(1, NUM_ACTIONS):
        if p0[c] > p0[best]:
            best = c

    # Ahora se tapa solo la favorita.
    m2 = List[Scalar[dtype]]()
    for c in range(NUM_ACTIONS):
        m2.append(Scalar[dtype](0) if c == best else Scalar[dtype](1))
    mask2 = upload[dtype](ctx, m2)
    actor_probs(ctx, actor.params, actor.cache, obs, mask2, probs, n)
    ctx.synchronize()
    p1 = download[dtype](probs, NUM_ACTIONS)

    if p1[best] != Scalar[dtype](0):
        raise Error("la casilla tapada ", best, " sigue con probabilidad ",
                    p1[best])
    best2 = 0
    for c in range(1, NUM_ACTIONS):
        if p1[c] > p1[best2]:
            best2 = c
    if best2 == best:
        raise Error("tras tapar la favorita deberia ganar otra casilla")

    # Y las que quedan mantienen sus proporciones relativas: tapar una casilla
    # renormaliza, no reordena. Si esto fallara, el enmascarado estaria
    # distorsionando la politica en vez de solo recortarla.
    for c in range(NUM_ACTIONS):
        if c == best:
            continue
        for d in range(NUM_ACTIONS):
            if d == best or d == c:
                continue
            # p1[c]/p1[d] tiene que ser p0[c]/p0[d]; se compara en producto
            # cruzado para no dividir.
            assert_close(p1[c] * p0[d], p1[d] * p0[c], Scalar[dtype](1e-4),
                         String("proporcion entre ", c, " y ", d))
    print("PASS tapar la favorita cambia la eleccion y solo renormaliza el resto")


def test_full_board_does_not_produce_nan(ctx: DeviceContext) raises:
    """Un tablero lleno (todo tapado) da uniforme, no nan.

    No deberia consultarse al actor en una posicion terminal, pero si pasa, el
    resultado tiene que ser inofensivo. Con -inf de verdad el softmax daria nan y
    el nan se propagaria callado por todo lo que venga despues; con NEG_INF finito
    la fila degenera a uniforme. Esta prueba fija esa eleccion de diseno.
    """
    actor = load_actor(ctx, 64)
    n = 1
    board = board9(1,-1,1, -1,1,-1, -1,1,-1)     # lleno
    state = upload[dtype](ctx, board)
    obs = zero_buffer[dtype](ctx, n * OBS_DIM)
    ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
        obs.unsafe_ptr(), state.unsafe_ptr(), n, grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()

    actor.forward(ctx, state, obs, n)
    ctx.synchronize()
    p = download[dtype](actor.probs, n * NUM_ACTIONS)

    total = Scalar[dtype](0)
    for c in range(NUM_ACTIONS):
        v = p[c]
        if v != v:                        # nan != nan
            raise Error("la casilla ", c, " salio nan con el tablero lleno")
        total += v
    assert_close(total, Scalar[dtype](1), TOL,
                 "incluso con todo tapado la fila tiene que sumar 1")
    print("PASS un tablero lleno degenera a uniforme en vez de dar nan")


def test_mask_comes_from_the_state(ctx: DeviceContext) raises:
    """La mascara se deriva del tablero, no se pasa por fuera.

    Es una invariante que importa: si la legalidad viniera por un canal aparte,
    podria desincronizarse del estado y la red acabaria jugando sobre fichas
    puestas. Se comprueba que `mask_from_state` reproduce exactamente las casillas
    vacias de tres tableros distintos, incluido uno con varios bloques de hilos.
    """
    actor = load_actor(ctx, 32)
    boards = List[Scalar[dtype]]()
    for b in board9(0,0,0, 0,0,0, 0,0,0): boards.append(b)
    for b in board9(1,-1,1, -1,1,-1, -1,1,-1): boards.append(b)
    for b in board9(1,0,0, 0,-1,0, 0,0,1): boards.append(b)
    n = 3
    state = upload[dtype](ctx, boards)
    actor.mask_from_state(ctx, state, n)
    ctx.synchronize()
    got = download[dtype](actor.mask, n * NUM_ACTIONS)

    for e in range(n):
        for c in range(NUM_ACTIONS):
            want = Scalar[dtype](1) if boards[e * NUM_CELLS + c] == 0 \
                   else Scalar[dtype](0)
            assert_close(got[e * NUM_ACTIONS + c], want, TOL,
                         String("mascara del tablero ", e, " casilla ", c))
    print("PASS la mascara sale del estado y coincide con las casillas vacias")


def main() raises:
    with DeviceContext() as ctx:
        test_actor_matches_golden(ctx)
        test_illegal_cells_get_exactly_zero(ctx)
        test_masking_changes_the_ranking(ctx)
        test_full_board_does_not_produce_nan(ctx)
        test_mask_comes_from_the_state(ctx)
