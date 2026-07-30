"""El modelo de busqueda con critico: que V salga de la red y llegue donde debe.

Aqui no se comprueba si el critico es BUENO — eso lo mide el experimento de E1.11
jugando partidas. Se comprueba el cableado, que es donde este tipo de codigo se
rompe en silencio: si `next_value` se quedara a 0, o si el valor se leyera del
buffer equivocado, la busqueda seguiria funcionando y dando resultados
razonables, solo que sin usar el critico. Nada fallaria.

El truco de todas las pruebas es el mismo: pesos elegidos para que V(s) sea un
numero CONOCIDO, y comparar contra ese numero. Con w1 = w2 = w3 = 0 y b3 = c:

    a1 = x@0 + b1 = b1  ->  relu
    a2 = a1@0 + b2 = b2 ->  relu
    V  = a2@0 + b3 = c              para cualquier tablero

asi que V vale c y no depende de la entrada. Eso permite predecir a mano lo que
tiene que aparecer en cada buffer.
"""

from std.gpu.host import DeviceContext

from ops.common import dtype, idx_dtype
from envs.tictactoe import (NUM_ACTIONS, STATE_DIM, NUM_CELLS, OBS_DIM,
                            CELL_EMPTY, NEG_INF)
from envs.tictactoe_critic import TicTacToeCritic
from networks.mlp import zero_critic_params
from systems.spo.particles import Particles, StepOutputs
from systems.spo.spo_types import SPOConfig
from tests.helpers import (upload, download, write_into, zeros, filled,
                           assert_close)

comptime TOL = Scalar[dtype](1e-5)
comptime HIDDEN = 8
comptime GAMMA = Scalar[dtype](0.9)


def board9(c0: Int, c1: Int, c2: Int, c3: Int, c4: Int,
           c5: Int, c6: Int, c7: Int, c8: Int) -> List[Scalar[dtype]]:
    """Un tablero legible: 9 codigos de casilla en el orden 0..8."""
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
                     search_depth=4, resample_period=4, temperature=0.5,
                     search_gamma=GAMMA, search_gae_lambda=1.0)


def constant_model(ctx: DeviceContext, max_batch: Int,
                   c: Scalar[dtype]) raises -> TicTacToeCritic:
    """Un modelo cuyo critico devuelve siempre `c`, sea cual sea el tablero."""
    model = TicTacToeCritic(ctx, max_batch, HIDDEN, Scalar[dtype](1))
    b3 = List[Scalar[dtype]](); b3.append(c)
    write_into[dtype](model.params.b3, b3)      # el resto ya esta a cero
    ctx.synchronize()
    return model^


def test_eval_root_uses_the_network(ctx: DeviceContext) raises:
    """`eval_root` escribe V(s) de la red, no 0 y no lo que hubiera antes.

    El buffer de salida se rellena a 99 aposta: si el modelo no escribiera nada,
    el test veria 99 y fallaria. Con `TicTacToe` (sin critico) aqui saldria 0.
    """
    num_envs = 3
    cfg = cfg_for(num_envs, 4)
    model = constant_model(ctx, cfg.num_search_particles(), Scalar[dtype](0.375))

    boards = List[Scalar[dtype]]()
    for b in board9(1,0,-1, 0,1,0, 0,0,0): boards.append(b)
    for b in board9(0,0,0, 0,0,0, 0,0,0): boards.append(b)
    for b in board9(1,1,-1, -1,-1,1, 1,-1,0): boards.append(b)
    root_state = upload[dtype](ctx, boards)

    logits = zeros[dtype](ctx, num_envs * NUM_ACTIONS)
    value = filled[dtype](ctx, num_envs, Scalar[dtype](99))
    model.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()

    got_value = download[dtype](value, num_envs)
    got_logits = download[dtype](logits, num_envs * NUM_ACTIONS)
    for e in range(num_envs):
        assert_close(got_value[e], Scalar[dtype](0.375), TOL,
                     String("V de la raiz del env ", e))
        # Y el prior sigue enmascarado igual que sin critico: anadir el valor no
        # puede haber roto la parte que ya funcionaba.
        for c in range(NUM_ACTIONS):
            want = Scalar[dtype](0) if boards[e * NUM_CELLS + c] == CELL_EMPTY \
                   else NEG_INF
            assert_close(got_logits[e * NUM_ACTIONS + c], want, TOL,
                         String("prior env ", e, " casilla ", c))
    print("PASS eval_root toma V de la red y mantiene el prior enmascarado")


def run_step(ctx: DeviceContext, model: TicTacToeCritic, cfg: SPOConfig,
             boards: List[Scalar[dtype]], actions: List[Int],
             uniforms: List[Scalar[dtype]],
             depths: List[Int] = List[Int]()) raises -> List[Scalar[dtype]]:
    """Un step del modelo; devuelve [next_value..., discount...] concatenados."""
    p_total = cfg.num_search_particles()
    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    if len(depths) > 0:
        ds = List[Scalar[idx_dtype]]()
        for d in depths: ds.append(Scalar[idx_dtype](d))
        write_into[idx_dtype](particles.depth, ds)

    write_into[dtype](particles.state, boards)
    acts = List[Scalar[idx_dtype]]()
    for a in actions: acts.append(Scalar[idx_dtype](a))
    write_into[idx_dtype](outputs.next_action, acts)
    # next_value marcado: si el modelo no lo escribiera, se veria el -7.
    outputs.next_value.enqueue_fill(Scalar[dtype](-7))
    step_us = upload[dtype](ctx, uniforms)

    model.step(ctx, cfg, particles, outputs, step_us)
    ctx.synchronize()

    out = download[dtype](outputs.next_value, p_total)
    for d in download[dtype](outputs.discount, p_total): out.append(d)
    return out^


def test_bootstrap_formula(ctx: DeviceContext) raises:
    """El bootstrap vale discount * search_gamma * V(s'), acabe o no la partida.

    Dos particulas a proposito:
      - la 0 sigue viva  -> discount 1 -> next_value = gamma * c
      - la 1 gana y acaba -> discount 0 -> next_value = 0, aunque V valga c

    El segundo caso es el que importa de verdad: si el discount no se plegara, una
    particula terminal arrastraria valor futuro que no existe y la busqueda
    sobrevaloraria las lineas que acaban.
    """
    cfg = cfg_for(1, 2)
    c = Scalar[dtype](0.5)
    model = constant_model(ctx, cfg.num_search_particles(), c)

    boards = List[Scalar[dtype]]()
    # p0: tablero vacio, juega la 0 -> no gana, el rival responde, sigue.
    for b in board9(0,0,0, 0,0,0, 0,0,0): boards.append(b)
    # p1: X en 0 y 1, juega la 2 -> linea 0-1-2 -> victoria, terminal.
    for b in board9(1,1,0, -1,-1,0, 0,0,0): boards.append(b)

    actions = List[Int](); actions.append(0); actions.append(2)
    us = List[Scalar[dtype]]()
    us.append(Scalar[dtype](0.1)); us.append(Scalar[dtype](0.1))

    got = run_step(ctx, model, cfg, boards, actions, us)
    nv0 = got[0]; nv1 = got[1]; d0 = got[2]; d1 = got[3]

    assert_close(d0, Scalar[dtype](1), TOL, "la particula 0 deberia seguir viva")
    assert_close(d1, Scalar[dtype](0), TOL, "la particula 1 deberia haber acabado")
    assert_close(nv0, GAMMA * c, TOL, "next_value de la particula viva")
    assert_close(nv1, Scalar[dtype](0), TOL,
                 "next_value de una particula terminal tiene que ser 0")
    print("PASS el bootstrap es discount * gamma * V(s') y respeta el terminal")


def test_depth_discounted_bootstrap(ctx: DeviceContext) raises:
    """En el modo coherente el bootstrap lleva gamma_r^(d+1), no `search_gamma`.

    Es el modo que arregla el desajuste de escalas: la recompensa que produce la
    dinamica ya viene con gamma_r^d, asi que el valor tiene que traer gamma_r^(d+1)
    para que sumarlos signifique algo. Se comprueba en varias profundidades a la
    vez porque el fallo tipico es equivocarse en uno: usar d en vez de d+1.

    Ojo con lo que NO tiene que pasar: `search_gamma` (0.9 en estas pruebas) no
    puede aparecer en el resultado. Si apareciera, el modelo estaria usando el
    kernel de siempre y el modo nuevo seria decorativo.
    """
    n_p = 4
    cfg = cfg_for(1, n_p)
    c = Scalar[dtype](0.5)
    gamma_r = Scalar[dtype](0.7)

    model = TicTacToeCritic(ctx, cfg.num_search_particles(), HIDDEN, gamma_r,
                            depth_discounted=True)
    b3 = List[Scalar[dtype]](); b3.append(c)
    write_into[dtype](model.params.b3, b3)
    ctx.synchronize()

    boards = List[Scalar[dtype]]()
    actions = List[Int]()
    us = List[Scalar[dtype]]()
    depths = List[Int]()
    for p in range(n_p):
        for b in board9(0,0,0, 0,0,0, 0,0,0): boards.append(b)
        actions.append(p)                 # tablero vacio: ninguna jugada acaba
        us.append(Scalar[dtype](0.1))
        depths.append(p)                  # profundidades 0, 1, 2, 3

    got = run_step(ctx, model, cfg, boards, actions, us, depths)
    for p in range(n_p):
        want = c
        for _ in range(p + 1):            # gamma^(d+1), a mano
            want *= gamma_r
        assert_close(got[p], want, TOL,
                     String("bootstrap en la profundidad ", p))
    # 0.7^1*0.5 = 0.35, y no 0.9*0.5 = 0.45: el search_gamma no se ha colado.
    assert_close(got[0], Scalar[dtype](0.35), TOL,
                 "en la profundidad 0 el factor deberia ser gamma_r, no search_gamma")
    print("PASS el bootstrap coherente descuenta por profundidad con gamma_r^(d+1)")


def test_value_depends_on_the_board(ctx: DeviceContext) raises:
    """Con pesos de verdad, dos tableros distintos dan V distintos.

    Es lo que la prueba del valor constante no puede ver: si `eval_root` codificara
    siempre el mismo tablero (por leer mal el buffer, o por no reescribir `obs`),
    V saldria identico para todos los envs y con b3 constante nadie lo notaria.
    Aqui la red SI mira la entrada, asi que un V repetido delata el fallo.

    Los pesos son a mano y sencillos: w1 lleva cada casilla del plano "mias" a una
    neurona distinta, y w3 las suma. Con eso V(s) = numero de fichas mias, que se
    puede contar mirando el tablero.
    """
    num_envs = 2
    cfg = cfg_for(num_envs, 2)
    model = TicTacToeCritic(ctx, cfg.num_search_particles(), HIDDEN,
                            Scalar[dtype](1))

    # w1 es [OBS_DIM, HIDDEN]: la casilla i del plano "mias" activa la neurona i.
    # Solo hay HIDDEN=8 neuronas para 9 casillas, asi que la 8 se queda fuera; da
    # igual, los tableros de la prueba no la usan.
    w1 = List[Scalar[dtype]]()
    for i in range(OBS_DIM):
        for h in range(HIDDEN):
            w1.append(Scalar[dtype](1) if (i < HIDDEN and i == h)
                      else Scalar[dtype](0))
    write_into[dtype](model.params.w1, w1)

    # w2 = identidad, para que las activaciones pasen intactas a la segunda capa.
    w2 = List[Scalar[dtype]]()
    for i in range(HIDDEN):
        for h in range(HIDDEN):
            w2.append(Scalar[dtype](1) if i == h else Scalar[dtype](0))
    write_into[dtype](model.params.w2, w2)

    # w3 = todo unos: V = suma de las activaciones = numero de fichas mias.
    w3 = List[Scalar[dtype]]()
    for _ in range(HIDDEN): w3.append(Scalar[dtype](1))
    write_into[dtype](model.params.w3, w3)
    ctx.synchronize()

    boards = List[Scalar[dtype]]()
    for b in board9(1,1,1, 0,0,0, 0,0,0): boards.append(b)   # 3 fichas mias
    for b in board9(1,-1,0, -1,0,0, 0,0,0): boards.append(b) # 1 ficha mia
    root_state = upload[dtype](ctx, boards)

    logits = zeros[dtype](ctx, num_envs * NUM_ACTIONS)
    value = filled[dtype](ctx, num_envs, Scalar[dtype](99))
    model.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()

    got = download[dtype](value, num_envs)
    assert_close(got[0], Scalar[dtype](3), TOL, "V del tablero con 3 fichas mias")
    assert_close(got[1], Scalar[dtype](1), TOL, "V del tablero con 1 ficha mia")
    print("PASS V depende del tablero: la observacion llega bien a la red")


def test_sync_from_copies_the_weights(ctx: DeviceContext) raises:
    """`sync_from` trae los pesos de otro critico y cambia lo que predice el modelo.

    Sin esto no habria forma de meter el critico ENTRENADO en la busqueda: el
    modelo se quedaria con los ceros del constructor y V seria 0, o sea el
    planificador de siempre disfrazado.
    """
    cfg = cfg_for(1, 2)
    model = constant_model(ctx, cfg.num_search_particles(), Scalar[dtype](0.1))

    src = zero_critic_params(ctx, OBS_DIM, HIDDEN, 1)
    b3 = List[Scalar[dtype]](); b3.append(Scalar[dtype](0.8))
    write_into[dtype](src.b3, b3)
    ctx.synchronize()

    model.sync_from(ctx, src)
    ctx.synchronize()

    boards = board9(1,0,-1, 0,1,0, 0,0,0)
    root_state = upload[dtype](ctx, boards)
    logits = zeros[dtype](ctx, NUM_ACTIONS)
    value = filled[dtype](ctx, 1, Scalar[dtype](99))
    model.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()

    got = download[dtype](value, 1)
    assert_close(got[0], Scalar[dtype](0.8), TOL,
                 "tras sync_from, V deberia venir de los pesos copiados")

    # Y la copia es independiente: tocar el origen despues no mueve al modelo.
    b3b = List[Scalar[dtype]](); b3b.append(Scalar[dtype](-0.4))
    write_into[dtype](src.b3, b3b)
    ctx.synchronize()
    model.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()
    got2 = download[dtype](value, 1)
    assert_close(got2[0], Scalar[dtype](0.8), TOL,
                 "el modelo tiene su propia copia; el origen ya no le afecta")

    # Y una forma incompatible se rechaza en vez de copiar basura.
    bad = zero_critic_params(ctx, OBS_DIM, HIDDEN + 1, 1)
    failed = False
    try:
        model.sync_from(ctx, bad)
    except:
        failed = True
    if not failed:
        raise Error("sync_from deberia rechazar un critico con otra forma")
    print("PASS sync_from copia los pesos, es independiente y valida la forma")


def test_many_particles_multi_block(ctx: DeviceContext) raises:
    """Mas de un bloque y un tamano no redondo: 5 envs x 13 particulas = 65.

    Los kernels de este fichero llevan guard, pero el guard solo se ejerce si
    alguna vez se lanza con un tamano que no cuadra con el bloque. Todas las
    pruebas de arriba usan tamanos minusculos que caben en un bloque, asi que sin
    este caso el `if p < n` nunca se probaria de verdad.
    """
    cfg = cfg_for(5, 13)
    p_total = cfg.num_search_particles()          # 65
    c = Scalar[dtype](0.25)
    model = constant_model(ctx, p_total, c)

    boards = List[Scalar[dtype]]()
    actions = List[Int]()
    us = List[Scalar[dtype]]()
    for p in range(p_total):
        # Tableros vacios: ninguno acaba en un turno, asi que discount = 1 en
        # todos y next_value tiene que ser gamma*c en los 65.
        for b in board9(0,0,0, 0,0,0, 0,0,0): boards.append(b)
        actions.append(p % 9)
        us.append(Scalar[dtype](0.1))

    got = run_step(ctx, model, cfg, boards, actions, us)
    for p in range(p_total):
        assert_close(got[p_total + p], Scalar[dtype](1), TOL,
                     String("discount de la particula ", p))
        assert_close(got[p], GAMMA * c, TOL,
                     String("next_value de la particula ", p))
    print("PASS con 65 particulas (varios bloques, tamano no redondo) sale igual")


def main() raises:
    with DeviceContext() as ctx:
        test_eval_root_uses_the_network(ctx)
        test_bootstrap_formula(ctx)
        test_depth_discounted_bootstrap(ctx)
        test_value_depends_on_the_board(ctx)
        test_sync_from_copies_the_weights(ctx)
        test_many_particles_multi_block(ctx)
