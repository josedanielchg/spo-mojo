"""Dinamica del MDP de juguete, calculada a mano.

Con L=8 y horizonte 4, V(pos) = 8 - pos. Los tres casos que hay que distinguir:

  paso normal    GOOD desde 0: r=1, rec_disc=1, bootstrap = V(1) = 7
  TRUNCACION     GOOD desde 3: r=1, rec_disc=0 (deja de simular)
                               PERO bootstrap = V(4) = 4, porque habia futuro
  TERMINAL       BAD  desde 0: r=0, rec_disc=0 y bootstrap = 0, no hay futuro

La diferencia entre las dos ultimas filas es todo el sentido de este fichero: en
las dos el rec_discount vale 0, y si uno se equivoca y pone tambien el bootstrap
a 0 en la truncacion, el bug no se nota hasta que el entrenamiento no converge.
"""

from std.gpu.host import DeviceContext

from ops.common import dtype, idx_dtype
from envs.toy_chain import (toy_recurrent_kernel, toy_value_kernel,
                            toy_policy_logits_kernel, toy_value,
                            default_toy_config, ToyChainConfig,
                            ACTION_BAD, ACTION_GOOD, NUM_ACTIONS, STATE_DIM)
from systems.spo.spo_types import (SPOConfig, Particles, StepOutputs,
                                   default_config)
from tests.helpers import upload, zeros, download, assert_close

comptime TPB = 32
comptime TOL = Scalar[dtype](1e-6)


@fieldwise_init
struct StepResult(Movable):
    """Lo que devuelve un paso, ya bajado al host. Es un struct y no una tupla
    porque en 1.0.0b1 una tupla de List no se deja construir."""
    var next_pos: List[Scalar[dtype]]
    var reward: List[Scalar[dtype]]
    var discount: List[Scalar[dtype]]
    var bootstrap: List[Scalar[dtype]]


def run_step(ctx: DeviceContext, positions: List[Scalar[dtype]],
             actions: List[Scalar[idx_dtype]], cfg: ToyChainConfig,
             search_gamma: Scalar[dtype]) raises -> StepResult:
    """Un paso del modelo sobre las posiciones dadas."""
    n = len(positions)
    state = upload[dtype](ctx, positions)
    action = upload[idx_dtype](ctx, actions)
    reward = zeros[dtype](ctx, n)
    discount = zeros[dtype](ctx, n)
    next_value = zeros[dtype](ctx, n)

    blocks = (n + TPB - 1) // TPB
    ctx.enqueue_function[toy_recurrent_kernel, toy_recurrent_kernel](
        state.unsafe_ptr(), action.unsafe_ptr(), reward.unsafe_ptr(),
        discount.unsafe_ptr(), next_value.unsafe_ptr(), n,
        cfg.chain_length, cfg.horizon, cfg.value_scale, search_gamma,
        grid_dim=blocks, block_dim=TPB)
    ctx.synchronize()

    return StepResult(download[dtype](state, n), download[dtype](reward, n),
                      download[dtype](discount, n), download[dtype](next_value, n))


def test_value_function(ctx: DeviceContext) raises:
    """V(pos) = 8 - pos, y nunca negativo pasada la ultima casilla."""
    cfg = default_toy_config()
    positions = List[Scalar[dtype]]()
    for p in range(10):
        positions.append(Scalar[dtype](p))
    n = len(positions)

    state = upload[dtype](ctx, positions)
    value = zeros[dtype](ctx, n)
    ctx.enqueue_function[toy_value_kernel, toy_value_kernel](
        value.unsafe_ptr(), state.unsafe_ptr(), n, cfg.chain_length, cfg.value_scale,
        grid_dim=1, block_dim=TPB)
    ctx.synchronize()

    got = download[dtype](value, n)
    for p in range(n):
        want = Scalar[dtype](cfg.chain_length - p)
        if want < 0.0:
            want = 0.0
        assert_close(got[p], want, TOL, String("V(", p, ")"))
    print("PASS toy value = casillas restantes (y >= 0)")


def test_uniform_prior(ctx: DeviceContext) raises:
    """El prior es uniforme: todos los logits iguales."""
    n = 5
    positions = List[Scalar[dtype]]()
    for p in range(n):
        positions.append(Scalar[dtype](p))

    state = upload[dtype](ctx, positions)
    logits = zeros[dtype](ctx, n * NUM_ACTIONS)
    ctx.enqueue_function[toy_policy_logits_kernel, toy_policy_logits_kernel](
        logits.unsafe_ptr(), state.unsafe_ptr(), n, grid_dim=1, block_dim=TPB)
    ctx.synchronize()

    got = download[dtype](logits, n * NUM_ACTIONS)
    for i in range(n):
        assert_close(got[i * NUM_ACTIONS + ACTION_BAD],
                     got[i * NUM_ACTIONS + ACTION_GOOD], TOL,
                     String("prior no uniforme en la particula ", i))
    print("PASS prior uniforme")


def test_normal_step(ctx: DeviceContext) raises:
    """GOOD desde 0, 1 y 2: avanza, da 1 de recompensa y sigue viva."""
    cfg = default_toy_config()
    positions = List[Scalar[dtype]]()
    actions = List[Scalar[idx_dtype]]()
    for p in range(3):          # 0, 1, 2 -> llegan a 1, 2, 3 < horizonte 4
        positions.append(Scalar[dtype](p))
        actions.append(Scalar[idx_dtype](ACTION_GOOD))

    r = run_step(ctx, positions, actions, cfg, 1.0)

    for p in range(3):
        assert_close(r.next_pos[p], Scalar[dtype](p + 1), TOL, String("next_pos desde ", p))
        assert_close(r.reward[p], 1.0, TOL, String("reward desde ", p))
        # Sigue viva: el rec_discount es 1.
        assert_close(r.discount[p], 1.0, TOL, String("rec_discount desde ", p))
        # bootstrap = 1 * gamma * V(p+1)
        assert_close(r.bootstrap[p], toy_value(Scalar[dtype](p + 1), cfg.chain_length, cfg.value_scale), TOL,
                     String("bootstrap desde ", p))
    print("PASS paso normal: avanza, r=1, sigue viva")


def test_terminal_vs_truncation(ctx: DeviceContext) raises:
    """El test que justifica todo el fichero: los dos finales no son iguales."""
    cfg = default_toy_config()

    positions = List[Scalar[dtype]]()
    actions = List[Scalar[idx_dtype]]()
    # [0] TERMINAL: BAD desde la casilla 0
    positions.append(0.0); actions.append(Scalar[idx_dtype](ACTION_BAD))
    # [1] TRUNCACION: GOOD desde 3 -> llega a 4 == horizonte
    positions.append(3.0); actions.append(Scalar[idx_dtype](ACTION_GOOD))

    r = run_step(ctx, positions, actions, cfg, 1.0)

    # El caso terminal.
    assert_close(r.reward[0], 0.0, TOL, "terminal: recompensa")
    assert_close(r.discount[0], 0.0, TOL, "terminal: rec_discount tiene que ser 0")
    assert_close(r.bootstrap[0], 0.0, TOL,
                 "terminal: el bootstrap tiene que ser 0, no hay futuro")

    # Y el de truncacion, que es donde esta la diferencia.
    assert_close(r.next_pos[1], 4.0, TOL, "truncacion: posicion final")
    assert_close(r.reward[1], 1.0, TOL, "truncacion: el paso si dio recompensa")
    assert_close(r.discount[1], 0.0, TOL,
                 "truncacion: rec_discount 0, la particula deja de simular")
    # Y aqui la diferencia: V(4) = 8 - 4 = 4, NO cero.
    assert_close(r.bootstrap[1], 4.0, TOL,
                 "truncacion: el bootstrap conserva el valor del futuro")

    print("PASS terminal (bootstrap 0) vs truncacion (bootstrap", r.bootstrap[1], ")")


def test_zero_value_variant(ctx: DeviceContext) raises:
    """Con value_scale=0 el modelo no tiene critico: V==0 en todas partes.

    Es el modo que usara la demo de CartPole en la fase 4, donde los pesos SMC
    degeneran al retorno acumulado.
    """
    cfg = ToyChainConfig(chain_length=8, horizon=4, value_scale=0.0)

    positions = List[Scalar[dtype]]()
    actions = List[Scalar[idx_dtype]]()
    positions.append(0.0); actions.append(Scalar[idx_dtype](ACTION_GOOD))
    positions.append(3.0); actions.append(Scalar[idx_dtype](ACTION_GOOD))

    r = run_step(ctx, positions, actions, cfg, 1.0)

    for i in range(2):
        assert_close(r.bootstrap[i], 0.0, TOL, String("V==0: bootstrap en ", i))
        assert_close(r.reward[i], 1.0, TOL, String("V==0: la recompensa no cambia, ", i))
    print("PASS variante sin critico (value_scale=0)")


def test_search_gamma_applies(ctx: DeviceContext) raises:
    """El bootstrap escala con search_gamma; la recompensa del paso no."""
    cfg = default_toy_config()
    positions = List[Scalar[dtype]]()
    actions = List[Scalar[idx_dtype]]()
    positions.append(0.0); actions.append(Scalar[idx_dtype](ACTION_GOOD))

    r = run_step(ctx, positions, actions, cfg, 0.5)

    assert_close(r.reward[0], 1.0, TOL, "gamma no debe tocar la recompensa")
    # 0.5 * V(1) = 0.5 * 7
    assert_close(r.bootstrap[0], 0.5 * toy_value(1.0, cfg.chain_length, cfg.value_scale), TOL,
                 "el bootstrap tiene que llevar search_gamma")
    print("PASS search_gamma se aplica solo al bootstrap")


def test_types_fit_the_toy_model(ctx: DeviceContext) raises:
    """Los tipos de la busqueda se instancian con el juguete.

    Es el "la interfaz encaja" de la fase: comprueba que SPOConfig/Particles/
    StepOutputs compilan y reservan lo que toca para este modelo, antes de que
    exista el nucleo SMC que los va a llenar.
    """
    num_envs = 3
    cfg = default_config(num_envs, STATE_DIM, NUM_ACTIONS)

    p = cfg.num_search_particles()
    if p != num_envs * 16:
        raise Error("num_search_particles deberia ser envs*particulas, salio ", p)

    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)
    ctx.synchronize()

    # Todo tiene que arrancar a cero: el peso SMC, la gae y la marca de terminal.
    weights = download[dtype](particles.resample_td_weights, p)
    gae = download[dtype](particles.gae, p)
    terminal = download[idx_dtype](particles.terminal, p)
    for i in range(p):
        assert_close(weights[i], 0.0, TOL, String("peso inicial ", i))
        assert_close(gae[i], 0.0, TOL, String("gae inicial ", i))
        if Int(terminal[i]) != 0:
            raise Error("la particula ", i, " nace marcada como terminal")

    # El buffer de logits es [P, num_actions], no [P].
    logits = download[dtype](outputs.action_logits, p * cfg.num_actions)
    if len(logits) != p * NUM_ACTIONS:
        raise Error("action_logits deberia ser P*num_actions")

    print("PASS los tipos de la busqueda encajan con el juguete (P =", p, ")")


def main() raises:
    with DeviceContext() as ctx:
        test_types_fit_the_toy_model(ctx)
        test_value_function(ctx)
        test_uniform_prior(ctx)
        test_normal_step(ctx)
        test_terminal_vs_truncation(ctx)
        test_zero_value_variant(ctx)
        test_search_gamma_applies(ctx)
