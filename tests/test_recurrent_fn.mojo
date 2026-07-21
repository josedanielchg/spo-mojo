"""Una profundidad completa de la busqueda, con valores dictados a mano.

No uso root_fn aqui a proposito: coloco las particulas donde quiero y les dicto
la accion, para poder comprobar los tres finales por separado y con numeros
exactos. Es el equivalente de "una particula con valores puestos a mano" del plan.

Con L=8 y horizonte 4, V(pos) = 8 - pos.
"""

from std.gpu.host import DeviceContext
from std.math import abs, log

from ops.common import dtype, idx_dtype
from envs.toy_chain import (toy_value, default_toy_chain,
                            ACTION_BAD, ACTION_GOOD, NUM_ACTIONS, STATE_DIM)
from systems.spo.particles import Particles, StepOutputs
from systems.spo.spo_types import SPOConfig
from systems.spo.root import sample_next_actions
from tests.helpers import upload, download, write_into, assert_close, assert_eq_int

comptime TOL = Scalar[dtype](1e-6)


def make_config(num_particles: Int) -> SPOConfig:
    """Un solo env con `num_particles` particulas: asi el indice p es la particula."""
    return SPOConfig(
        num_envs=1, num_particles=num_particles, num_actions=NUM_ACTIONS,
        state_dim=STATE_DIM, search_depth=4, resample_period=4,
        temperature=0.5, search_gamma=1.0, search_gae_lambda=1.0)


def test_one_depth_all_three_endings(ctx: DeviceContext) raises:
    """Las tres particulas cubren los tres caminos posibles de un paso."""
    toy = default_toy_chain()
    cfg = make_config(3)
    p_total = cfg.num_search_particles()

    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    # [0] paso normal : pos 0, GOOD -> 1
    # [1] truncacion  : pos 3, GOOD -> 4 == horizonte
    # [2] terminal    : pos 0, BAD
    positions = List[Scalar[dtype]]()
    positions.append(0.0); positions.append(3.0); positions.append(0.0)
    actions = List[Scalar[idx_dtype]]()
    actions.append(Scalar[idx_dtype](ACTION_GOOD))
    actions.append(Scalar[idx_dtype](ACTION_GOOD))
    actions.append(Scalar[idx_dtype](ACTION_BAD))

    # El valor viejo de cada particula: V(0)=8, V(3)=5, V(0)=8.
    values = List[Scalar[dtype]]()
    values.append(8.0); values.append(5.0); values.append(8.0)

    write_into[dtype](particles.state, positions)
    write_into[idx_dtype](outputs.next_action, actions)
    write_into[dtype](particles.value, values)

    # Uniformes fijos: 0.1 cae en la accion 0 con prior uniforme (CDF [0.5, 1.0]).
    us = List[Scalar[dtype]]()
    for _ in range(p_total):
        us.append(0.1)
    uniforms = upload[dtype](ctx, us)

    toy.step(ctx, cfg, particles, outputs)
    sample_next_actions(ctx, outputs, cfg, uniforms)
    ctx.synchronize()

    state = download[dtype](particles.state, p_total)
    reward = download[dtype](outputs.reward, p_total)
    discount = download[dtype](outputs.discount, p_total)
    next_value = download[dtype](outputs.next_value, p_total)
    old_value = download[dtype](particles.value, p_total)

    # La primera hace un paso normal: sigue viva y arrastra V(1) = 7.
    assert_close(state[0], 1.0, TOL, "normal: posicion")
    assert_close(reward[0], 1.0, TOL, "normal: recompensa")
    assert_close(discount[0], 1.0, TOL, "normal: sigue viva")
    assert_close(next_value[0], 7.0, TOL, "normal: bootstrap = V(1)")

    # La segunda se trunca: deja de simular pero conserva V(4) = 4.
    assert_close(state[1], 4.0, TOL, "truncada: posicion")
    assert_close(reward[1], 1.0, TOL, "truncada: el paso dio recompensa")
    assert_close(discount[1], 0.0, TOL, "truncada: rec_discount 0")
    assert_close(next_value[1], 4.0, TOL,
                 "truncada: el bootstrap no es 0, habia futuro")

    # La tercera muere de verdad: deja de simular y ademas pierde el futuro.
    assert_close(reward[2], 0.0, TOL, "terminal: sin recompensa")
    assert_close(discount[2], 0.0, TOL, "terminal: rec_discount 0")
    assert_close(next_value[2], 0.0, TOL, "terminal: bootstrap 0")

    # Las dos ultimas comparten rec_discount pero no bootstrap: ese contraste es
    # lo que separa "se acabo el tiempo" de "te has muerto".
    if discount[1] != discount[2]:
        raise Error("truncada y terminal deberian compartir rec_discount")
    if next_value[1] == next_value[2]:
        raise Error("truncada y terminal NO deberian compartir bootstrap")

    # Y el valor viejo tiene que seguir intacto, porque lo necesita el error TD.
    assert_close(old_value[0], 8.0, TOL, "recurrent_fn no debe tocar particles.value")
    assert_close(old_value[1], 5.0, TOL, "recurrent_fn no debe tocar particles.value")

    print("PASS una profundidad: normal / truncada / terminal")


def test_next_action_is_sampled_and_scored(ctx: DeviceContext) raises:
    """Tras el paso, cada particula tiene accion siguiente valida y su log-prob.

    El prior del juguete es uniforme sobre 2 acciones, asi que la log-prob tiene
    que ser log(0.5) sea cual sea la accion que salga.
    """
    toy = default_toy_chain()
    cfg = make_config(8)
    p_total = cfg.num_search_particles()

    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    positions = List[Scalar[dtype]]()
    actions = List[Scalar[idx_dtype]]()
    for _ in range(p_total):
        positions.append(0.0)
        actions.append(Scalar[idx_dtype](ACTION_GOOD))
    write_into[dtype](particles.state, positions)
    write_into[idx_dtype](outputs.next_action, actions)

    # Uniformes repartidos: unos por debajo de 0.5 (accion 0) y otros por encima.
    us = List[Scalar[dtype]]()
    for p in range(p_total):
        us.append(Scalar[dtype](p) / Scalar[dtype](p_total))
    uniforms = upload[dtype](ctx, us)

    toy.step(ctx, cfg, particles, outputs)
    sample_next_actions(ctx, outputs, cfg, uniforms)
    ctx.synchronize()

    next_actions = download[idx_dtype](outputs.next_action, p_total)
    logps = download[dtype](outputs.next_prior_logits, p_total)

    want_logp = log(Scalar[dtype](0.5))
    seen_zero = False
    seen_one = False
    for p in range(p_total):
        a = Int(next_actions[p])
        if a < 0 or a >= NUM_ACTIONS:
            raise Error("accion siguiente invalida en ", p, ": ", a)
        if a == 0:
            seen_zero = True
        else:
            seen_one = True
        assert_close(logps[p], want_logp, Scalar[dtype](1e-5),
                     String("log-prob de la particula ", p))

    # Con uniformes repartidos a los dos lados del corte tienen que salir las dos
    # acciones; si solo saliera una, el muestreo estaria roto aunque las
    # log-probs cuadraran.
    if not seen_zero or not seen_one:
        raise Error("el muestreo devolvio siempre la misma accion")
    print("PASS accion siguiente muestreada y puntuada con el prior")


def test_dead_particle_keeps_walking_is_not_a_problem(ctx: DeviceContext) raises:
    """Una particula ya muerta se vuelve a pisar, y da igual.

    El modelo no hace auto-reset ni comprueba `terminal`: sigue aplicando la
    dinamica. Lo documento con un test porque parece un bug y no lo es: la
    mascara terminal del nucleo SMC congela su peso, asi que lo que le pase
    despues de morir no entra en la busqueda. Aqui solo compruebo que el paso no
    explota ni produce valores raros.
    """
    toy = default_toy_chain()
    cfg = make_config(2)
    p_total = cfg.num_search_particles()

    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    # Las dos ya pasadas del horizonte, avanzando igualmente.
    positions = List[Scalar[dtype]]()
    positions.append(6.0); positions.append(9.0)
    actions = List[Scalar[idx_dtype]]()
    actions.append(Scalar[idx_dtype](ACTION_GOOD))
    actions.append(Scalar[idx_dtype](ACTION_GOOD))
    write_into[dtype](particles.state, positions)
    write_into[idx_dtype](outputs.next_action, actions)

    us = List[Scalar[dtype]]()
    us.append(0.1); us.append(0.9)
    uniforms = upload[dtype](ctx, us)

    toy.step(ctx, cfg, particles, outputs)
    sample_next_actions(ctx, outputs, cfg, uniforms)
    ctx.synchronize()

    next_value = download[dtype](outputs.next_value, p_total)
    # V esta recortado a 0, asi que pasarse del final no da valores negativos.
    assert_close(next_value[0], toy_value(7.0, toy.chain_length,
                                          toy.value_scale), TOL,
                 "V(7) tras pasar el horizonte")
    assert_close(next_value[1], 0.0, TOL, "V no puede ser negativo pasado el final")
    print("PASS pasarse del final no produce valores negativos")


def main() raises:
    with DeviceContext() as ctx:
        test_one_depth_all_three_endings(ctx)
        test_next_action_is_sampled_and_scored(ctx)
        test_dead_particle_keeps_walking_is_not_a_problem(ctx)
