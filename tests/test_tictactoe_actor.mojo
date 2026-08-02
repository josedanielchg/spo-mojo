"""El modelo de busqueda con el prior del ACTOR: que la red dirija la busqueda.

Igual que con el critico, aqui no se comprueba si el prior es BUENO -- eso lo mide
E2.6 jugando partidas. Se comprueba el cableado, y este en particular puede
romperse de una forma especialmente traicionera: si los logits del actor no
llegaran, el modelo se quedaria con los ceros del constructor, que tras el
enmascarado dan **exactamente el prior uniforme** de siempre. O sea que la busqueda
seguiria funcionando igual de bien y el bucle EM no se cerraria, sin que nada
fallara ni cambiara un numero.

De ahi que la prueba central no sea "el prior tiene tal valor" sino **"el prior NO
es uniforme"**, con pesos que producen una preferencia clara.
"""

from std.gpu.host import DeviceContext

from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from envs.tictactoe import (NUM_ACTIONS, NUM_CELLS, STATE_DIM, OBS_DIM, NEG_INF,
                            CELL_EMPTY)
from envs.tictactoe_actor import TicTacToeActor
from networks.actor import zero_actor_params
from systems.spo.particles import Particles, StepOutputs
from systems.spo.spo_types import SPOConfig
from tests.helpers import (upload, download, write_into, zeros, filled,
                           assert_close)

comptime TOL = Scalar[dtype](1e-5)
comptime HIDDEN = 8


def board9(c0: Int, c1: Int, c2: Int, c3: Int, c4: Int,
           c5: Int, c6: Int, c7: Int, c8: Int) -> List[Scalar[dtype]]:
    out = List[Scalar[dtype]]()
    out.append(Scalar[dtype](c0)); out.append(Scalar[dtype](c1))
    out.append(Scalar[dtype](c2)); out.append(Scalar[dtype](c3))
    out.append(Scalar[dtype](c4)); out.append(Scalar[dtype](c5))
    out.append(Scalar[dtype](c6)); out.append(Scalar[dtype](c7))
    out.append(Scalar[dtype](c8))
    return out^


def cfg_for(num_envs: Int, num_particles: Int) -> SPOConfig:
    return SPOConfig(num_envs=num_envs, num_particles=num_particles,
                     num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                     search_depth=4, resample_period=99, temperature=0.5,
                     search_gamma=1.0, search_gae_lambda=1.0)


def biased_model(ctx: DeviceContext, max_batch: Int) raises -> TicTacToeActor:
    """Un modelo cuyo actor prefiere claramente la casilla 4 (el centro).

    Con w1 = w2 = 0 la red ignora el tablero y saca b3, asi que poniendo b3 con un
    pico en la 4 se obtiene un prior fijo y predecible. Sirve para comprobar que
    los logits LLEGAN, que es lo que se puede romper en silencio.
    """
    m = TicTacToeActor(ctx, max_batch, HIDDEN, Scalar[dtype](0.9))
    b3 = List[Scalar[dtype]]()
    for a in range(NUM_ACTIONS):
        b3.append(Scalar[dtype](3) if a == 4 else Scalar[dtype](0))
    write_into[dtype](m.params.b3, b3)
    ctx.synchronize()
    return m^


def test_root_prior_comes_from_the_network(ctx: DeviceContext) raises:
    """El prior de la raiz sale de la red y NO es uniforme.

    Es la prueba que separa "el actor dirige la busqueda" de "el actor esta
    conectado pero no influye". Con los pesos a cero el prior enmascarado seria
    uniforme, indistinguible del de siempre; aqui la red prefiere el centro y eso
    tiene que verse.
    """
    num_envs = 2
    cfg = cfg_for(num_envs, 4)
    model = biased_model(ctx, cfg.num_search_particles())

    boards = List[Scalar[dtype]]()
    for b in board9(0,0,0, 0,0,0, 0,0,0): boards.append(b)      # todo libre
    for b in board9(1,0,-1, 0,0,0, 0,0,0): boards.append(b)     # 0 y 2 ocupadas
    root_state = upload[dtype](ctx, boards)

    logits = filled[dtype](ctx, num_envs * NUM_ACTIONS, Scalar[dtype](99))
    value = filled[dtype](ctx, num_envs, Scalar[dtype](77))
    model.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()

    got = download[dtype](logits, num_envs * NUM_ACTIONS)
    got_v = download[dtype](value, num_envs)

    for e in range(num_envs):
        for c in range(NUM_ACTIONS):
            v = got[e * NUM_ACTIONS + c]
            if boards[e * NUM_CELLS + c] != CELL_EMPTY:
                if v != NEG_INF:
                    raise Error("la casilla ocupada ", c, " del env ", e,
                                " deberia estar tapada y vale ", v)
            elif c == 4:
                assert_close(v, Scalar[dtype](3), TOL,
                             String("el centro del env ", e))
            else:
                assert_close(v, Scalar[dtype](0), TOL,
                             String("la casilla ", c, " del env ", e))
        # Y el valor sigue siendo 0, como el planificador.
        assert_close(got_v[e], Scalar[dtype](0), TOL,
                     String("V del env ", e, " deberia ser 0"))

    # La comprobacion que de verdad importa: NO es uniforme.
    all_equal = True
    for c in range(1, NUM_ACTIONS):
        if got[c] != got[0]:
            all_equal = False
    if all_equal:
        raise Error("el prior salio uniforme: los logits del actor no estan "
                    "llegando, y el bucle EM no se cerraria")
    print("PASS el prior de la raiz sale de la red y no es uniforme")


def test_step_prior_uses_the_new_state(ctx: DeviceContext) raises:
    """Tras avanzar, el prior se evalua en el tablero NUEVO.

    Es donde se muestrea la accion siguiente, asi que evaluarlo en el estado viejo
    dejaria a la busqueda proponiendo jugadas sobre casillas que acaban de
    ocuparse. El caso esta elegido para que la 4 (la favorita de la red) quede
    OCUPADA tras el paso: si el prior mirara el estado viejo, la 4 seguiria con su
    logit 3 en vez de estar tapada.
    """
    cfg = cfg_for(1, 1)
    model = biased_model(ctx, cfg.num_search_particles())
    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    # X juega la 4 (el centro). El rival respondera en alguna libre.
    write_into[dtype](particles.state, board9(1,-1,1, -1,0,0, 0,0,0))
    acts = List[Scalar[idx_dtype]](); acts.append(Scalar[idx_dtype](4))
    write_into[idx_dtype](outputs.next_action, acts)
    us = List[Scalar[dtype]](); us.append(Scalar[dtype](0.1))
    step_us = upload[dtype](ctx, us)

    model.step(ctx, cfg, particles, outputs, step_us)
    ctx.synchronize()

    new_board = download[dtype](particles.state, NUM_CELLS)
    got = download[dtype](outputs.action_logits, NUM_ACTIONS)

    if new_board[4] == CELL_EMPTY:
        raise Error("la casilla 4 deberia estar ocupada tras el paso")
    if got[4] != NEG_INF:
        raise Error("la casilla 4 esta ocupada en el estado NUEVO, asi que su "
                    "logit deberia estar tapado; vale ", got[4],
                    " (el prior esta mirando el estado viejo)")
    for c in range(NUM_ACTIONS):
        want_masked = new_board[c] != CELL_EMPTY
        if want_masked and got[c] != NEG_INF:
            raise Error("la casilla ", c, " esta ocupada y su logit vale ", got[c])
        if not want_masked:
            assert_close(got[c], Scalar[dtype](0), TOL,
                         String("logit de la casilla libre ", c))
    print("PASS el prior del step se evalua en el estado nuevo")


def test_sync_from_brings_the_trained_actor(ctx: DeviceContext) raises:
    """`sync_from` trae los pesos y cambia el prior; sin el, seria uniforme.

    Si no se llamara, el modelo se quedaria con los ceros del constructor y el
    prior enmascarado seria EXACTAMENTE el uniforme de siempre. O sea que el bucle
    EM parecería cerrado y no lo estaria, sin que ningun test de forma lo notara.
    Esta prueba compara los dos casos explicitamente.
    """
    num_envs = 1
    cfg = cfg_for(num_envs, 2)
    fresh = TicTacToeActor(ctx, cfg.num_search_particles(), HIDDEN,
                           Scalar[dtype](0.9))
    boards = board9(0,0,0, 0,0,0, 0,0,0)
    root_state = upload[dtype](ctx, boards)
    logits = zeros[dtype](ctx, NUM_ACTIONS)
    value = zeros[dtype](ctx, num_envs)

    # Sin sincronizar: uniforme, y por eso hace falta el test.
    fresh.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()
    before = download[dtype](logits, NUM_ACTIONS)
    for c in range(1, NUM_ACTIONS):
        assert_close(before[c], before[0], TOL,
                     "sin sincronizar, el prior deberia ser uniforme")

    # Un actor entrenado de mentira: prefiere la esquina 8.
    src = zero_actor_params(ctx, HIDDEN)
    b3 = List[Scalar[dtype]]()
    for a in range(NUM_ACTIONS):
        b3.append(Scalar[dtype](2.5) if a == 8 else Scalar[dtype](0))
    write_into[dtype](src.b3, b3)
    ctx.synchronize()

    fresh.sync_from(ctx, src)
    fresh.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()
    after = download[dtype](logits, NUM_ACTIONS)
    assert_close(after[8], Scalar[dtype](2.5), TOL,
                 "tras sync_from el prior deberia venir de los pesos copiados")
    assert_close(after[0], Scalar[dtype](0), TOL, "y el resto a 0")

    # La copia es independiente: tocar el origen despues no mueve al modelo.
    b3b = List[Scalar[dtype]]()
    for a in range(NUM_ACTIONS):
        b3b.append(Scalar[dtype](-9) if a == 8 else Scalar[dtype](0))
    write_into[dtype](src.b3, b3b)
    ctx.synchronize()
    fresh.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()
    again = download[dtype](logits, NUM_ACTIONS)
    assert_close(again[8], Scalar[dtype](2.5), TOL,
                 "el modelo tiene su propia copia; el origen ya no le afecta")

    # Y una forma incompatible se rechaza.
    bad = zero_actor_params(ctx, HIDDEN + 1)
    failed = False
    try:
        fresh.sync_from(ctx, bad)
    except:
        failed = True
    if not failed:
        raise Error("sync_from deberia rechazar un actor con otra forma")
    print("PASS sync_from trae el actor entrenado, es independiente y valida")


def test_many_particles_multi_block(ctx: DeviceContext) raises:
    """5 envs x 13 particulas = 65: mas de un bloque y tamano no redondo.

    Los kernels llevan guard, pero un guard solo se prueba si alguna vez se lanza
    con un tamano que no cuadra. Es el punto ciego que ya me comi cuatro veces en
    este proyecto, asi que va desde el primer dia.
    """
    cfg = cfg_for(5, 13)
    p_total = cfg.num_search_particles()
    model = biased_model(ctx, p_total)
    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    boards = List[Scalar[dtype]]()
    acts = List[Scalar[idx_dtype]]()
    us = List[Scalar[dtype]]()
    for p in range(p_total):
        for b in board9(0,0,0, 0,0,0, 0,0,0): boards.append(b)
        acts.append(Scalar[idx_dtype](p % 9))
        us.append(Scalar[dtype](0.1))
    write_into[dtype](particles.state, boards)
    write_into[idx_dtype](outputs.next_action, acts)
    step_us = upload[dtype](ctx, us)

    model.step(ctx, cfg, particles, outputs, step_us)
    ctx.synchronize()

    new_boards = download[dtype](particles.state, p_total * NUM_CELLS)
    got = download[dtype](outputs.action_logits, p_total * NUM_ACTIONS)
    for p in range(p_total):
        for c in range(NUM_ACTIONS):
            v = got[p * NUM_ACTIONS + c]
            if new_boards[p * NUM_CELLS + c] != CELL_EMPTY:
                if v != NEG_INF:
                    raise Error("particula ", p, " casilla ", c,
                                " ocupada pero su logit vale ", v)
            elif c == 4:
                assert_close(v, Scalar[dtype](3), TOL,
                             String("particula ", p, " centro"))
            else:
                assert_close(v, Scalar[dtype](0), TOL,
                             String("particula ", p, " casilla ", c))
    print("PASS con 65 particulas (varios bloques, tamano no redondo) sale igual")


def test_rejects_more_boards_than_reserved(ctx: DeviceContext) raises:
    """Pedir mas tableros de los reservados da error, no corrupcion silenciosa."""
    cfg = cfg_for(4, 8)                        # 32 particulas
    small = TicTacToeActor(ctx, 4, HIDDEN, Scalar[dtype](0.9))
    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)
    us = zero_buffer[dtype](ctx, cfg.num_search_particles())

    failed = False
    try:
        small.step(ctx, cfg, particles, outputs, us)
    except:
        failed = True
    if not failed:
        raise Error("deberia rechazar 32 particulas con sitio para 4")
    print("PASS el modelo rechaza mas tableros de los reservados")


def main() raises:
    with DeviceContext() as ctx:
        test_root_prior_comes_from_the_network(ctx)
        test_step_prior_uses_the_new_state(ctx)
        test_sync_from_brings_the_trained_actor(ctx)
        test_many_particles_multi_block(ctx)
        test_rejects_more_boards_than_reserved(ctx)
