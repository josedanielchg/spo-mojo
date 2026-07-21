"""Peso SMC y GAE hacia adelante, con dos profundidades hechas a mano.

Todo lo de aqui se calcula sin GPU en un papel, asi que los valores esperados son
constantes literales y no una re-implementacion del kernel (que es la trampa
clasica: si el test repite la formula del codigo, comprueba que el codigo hace lo
que hace, no que hace lo correcto).

Con search_gamma = search_gae_lambda = 1, el factor de la GAE es discount^depth.
"""

from std.gpu.host import DeviceContext

from ops.common import dtype, idx_dtype
from systems.spo.spo_types import SPOConfig, Particles, StepOutputs
from systems.spo.smc_search import update_particles
from tests.helpers import download, write_into, assert_close, assert_eq_int

comptime TOL = Scalar[dtype](1e-6)


def make_config(num_particles: Int) -> SPOConfig:
    return SPOConfig(
        num_envs=1, num_particles=num_particles, num_actions=2,
        state_dim=1, search_depth=4, resample_period=4,
        temperature=0.5, search_gamma=1.0, search_gae_lambda=1.0)


def set_step(outputs: StepOutputs, rewards: List[Scalar[dtype]],
             discounts: List[Scalar[dtype]],
             next_values: List[Scalar[dtype]]) raises:
    """Dicta el resultado de un paso del modelo, sin ejecutar ningun modelo."""
    write_into[dtype](outputs.reward, rewards)
    write_into[dtype](outputs.discount, discounts)
    write_into[dtype](outputs.next_value, next_values)
    logits = List[Scalar[dtype]]()
    for _ in range(len(rewards)):
        logits.append(0.0)
    write_into[dtype](outputs.next_prior_logits, logits)


def test_two_depths_by_hand(ctx: DeviceContext) raises:
    """Dos profundidades seguidas de una particula viva.

    Profundidad 0:  V=10, r=1, V'=12  ->  td = 1 + 12 - 10 = 3
                    peso = 0 + 3 = 3
                    gae  = 0 + 3 * (1*1*1)^0 = 3 * 1 = 3
    Profundidad 1:  V=12, r=2, V'=11  ->  td = 2 + 11 - 12 = 1
                    peso = 3 + 1 = 4
                    gae  = 3 + 1 * 1^1 = 4
    """
    cfg = make_config(1)
    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    v0 = List[Scalar[dtype]]()
    v0.append(10.0)
    write_into[dtype](particles.value, v0)

    # Profundidad 0.
    r = List[Scalar[dtype]]();  r.append(1.0)
    d = List[Scalar[dtype]]();  d.append(1.0)     # sigue viva
    nv = List[Scalar[dtype]](); nv.append(12.0)
    set_step(outputs, r, d, nv)
    update_particles(ctx, particles, outputs, cfg)
    ctx.synchronize()

    assert_close(download[dtype](particles.resample_td_weights, 1)[0], 3.0, TOL,
                 "profundidad 0: peso")
    assert_close(download[dtype](particles.gae, 1)[0], 3.0, TOL,
                 "profundidad 0: gae")
    assert_close(download[dtype](particles.value, 1)[0], 12.0, TOL,
                 "profundidad 0: el valor nuevo releva al viejo")
    assert_eq_int(Int(download[idx_dtype](particles.depth, 1)[0]), 1,
                  "profundidad 0: depth++")

    # Profundidad 1.
    r2 = List[Scalar[dtype]]();  r2.append(2.0)
    d2 = List[Scalar[dtype]]();  d2.append(1.0)
    nv2 = List[Scalar[dtype]](); nv2.append(11.0)
    set_step(outputs, r2, d2, nv2)
    update_particles(ctx, particles, outputs, cfg)
    ctx.synchronize()

    assert_close(download[dtype](particles.resample_td_weights, 1)[0], 4.0, TOL,
                 "profundidad 1: peso acumulado")
    assert_close(download[dtype](particles.gae, 1)[0], 4.0, TOL,
                 "profundidad 1: gae acumulada")
    assert_eq_int(Int(download[idx_dtype](particles.depth, 1)[0]), 2,
                  "profundidad 1: depth++")
    print("PASS dos profundidades a mano: peso 4, gae 4")


def test_death_marks_terminal_and_then_freezes(ctx: DeviceContext) raises:
    """Una particula que muere deja de acumular a partir de la profundidad siguiente.

    El paso que la mata si cuenta, porque todavia no estaba muerta al entrar, y
    a partir de ahi su peso queda congelado.

    Profundidad 0:  V=10, r=1, V'=0, discount=0 (muere)
                    td = 1 + 0 - 10 = -9 ; peso = -9 ; terminal <- 1
    Profundidad 1:  lo que sea; la mascara la anula -> peso sigue en -9
    """
    cfg = make_config(1)
    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    v0 = List[Scalar[dtype]]()
    v0.append(10.0)
    write_into[dtype](particles.value, v0)

    r = List[Scalar[dtype]]();  r.append(1.0)
    d = List[Scalar[dtype]]();  d.append(0.0)     # discount 0 -> muere
    nv = List[Scalar[dtype]](); nv.append(0.0)
    set_step(outputs, r, d, nv)
    update_particles(ctx, particles, outputs, cfg)
    ctx.synchronize()

    assert_close(download[dtype](particles.resample_td_weights, 1)[0], -9.0, TOL,
                 "el paso que mata si cuenta")
    assert_eq_int(Int(download[idx_dtype](particles.terminal, 1)[0]), 1,
                  "discount 0 tiene que marcar terminal")

    # Profundidad 1: ya esta muerta, asi que todo lo que venga se ignora.
    r2 = List[Scalar[dtype]]();  r2.append(100.0)   # recompensa enorme a proposito
    d2 = List[Scalar[dtype]]();  d2.append(0.0)
    nv2 = List[Scalar[dtype]](); nv2.append(50.0)
    set_step(outputs, r2, d2, nv2)
    update_particles(ctx, particles, outputs, cfg)
    ctx.synchronize()

    assert_close(download[dtype](particles.resample_td_weights, 1)[0], -9.0, TOL,
                 "una particula muerta no puede seguir sumando peso")
    print("PASS la muerte cuenta una vez y luego congela el peso")


def test_dead_particle_gae_is_frozen_by_the_decay(ctx: DeviceContext) raises:
    """La GAE de una particula muerta se congela sola, sin mascara.

    A diferencia del peso, la GAE no lleva mascara terminal (igual que en Stoix).
    Lo que la congela es que su discount es 0 y el factor es discount^depth: en
    la profundidad 1 o mayor, 0^depth = 0 y el delta no entra.

    Este test existe porque la ausencia de mascara parece un olvido y no lo es.
    """
    cfg = make_config(1)
    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    # Arranco ya en profundidad 1 y muerta, con una gae acumulada de 7.
    v0 = List[Scalar[dtype]]();  v0.append(10.0)
    g0 = List[Scalar[dtype]]();  g0.append(7.0)
    dep = List[Scalar[idx_dtype]](); dep.append(Scalar[idx_dtype](1))
    ter = List[Scalar[idx_dtype]](); ter.append(Scalar[idx_dtype](1))
    write_into[dtype](particles.value, v0)
    write_into[dtype](particles.gae, g0)
    write_into[idx_dtype](particles.depth, dep)
    write_into[idx_dtype](particles.terminal, ter)

    r = List[Scalar[dtype]]();  r.append(100.0)
    d = List[Scalar[dtype]]();  d.append(0.0)      # muerta -> discount 0
    nv = List[Scalar[dtype]](); nv.append(50.0)
    set_step(outputs, r, d, nv)
    update_particles(ctx, particles, outputs, cfg)
    ctx.synchronize()

    assert_close(download[dtype](particles.gae, 1)[0], 7.0, TOL,
                 "0^depth deberia anular el delta de una particula muerta")
    print("PASS la gae de una particula muerta se congela por el decay")


def test_depth_zero_always_counts(ctx: DeviceContext) raises:
    """En la profundidad 0 el factor es discount^0 = 1, incluso con discount 0.

    Es el caso limite de 0^0 = 1. Importa porque si se implementara con una
    exponenciacion que devuelva 0 para 0^0, el primer paso de toda particula que
    muere de inmediato desapareceria de la GAE.

    V=10, r=1, V'=0, discount=0 -> td = -9, y la gae tiene que ser -9, no 0.
    """
    cfg = make_config(1)
    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    v0 = List[Scalar[dtype]]()
    v0.append(10.0)
    write_into[dtype](particles.value, v0)

    r = List[Scalar[dtype]]();  r.append(1.0)
    d = List[Scalar[dtype]]();  d.append(0.0)
    nv = List[Scalar[dtype]](); nv.append(0.0)
    set_step(outputs, r, d, nv)
    update_particles(ctx, particles, outputs, cfg)
    ctx.synchronize()

    assert_close(download[dtype](particles.gae, 1)[0], -9.0, TOL,
                 "el paso de profundidad 0 tiene que contar entero (0^0 = 1)")
    print("PASS la profundidad 0 cuenta entera aunque el discount sea 0")


def test_many_particles_are_independent(ctx: DeviceContext) raises:
    """Cada particula lleva su cuenta: el kernel no mezcla vecinas."""
    n = 64
    cfg = make_config(n)
    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    values = List[Scalar[dtype]]()
    rewards = List[Scalar[dtype]]()
    discounts = List[Scalar[dtype]]()
    next_values = List[Scalar[dtype]]()
    for p in range(n):
        values.append(Scalar[dtype](p))
        rewards.append(Scalar[dtype](2 * p))
        discounts.append(1.0)
        next_values.append(Scalar[dtype](p) + 1.0)
    write_into[dtype](particles.value, values)
    set_step(outputs, rewards, discounts, next_values)

    update_particles(ctx, particles, outputs, cfg)
    ctx.synchronize()

    weights = download[dtype](particles.resample_td_weights, n)
    for p in range(n):
        # td = 2p + (p+1) - p = 2p + 1
        assert_close(weights[p], Scalar[dtype](2 * p + 1), TOL,
                     String("peso de la particula ", p))
    print("PASS", n, "particulas independientes")


def main() raises:
    with DeviceContext() as ctx:
        test_two_depths_by_hand(ctx)
        test_death_marks_terminal_and_then_freezes(ctx)
        test_dead_particle_gae_is_frozen_by_the_decay(ctx)
        test_depth_zero_always_counts(ctx)
        test_many_particles_are_independent(ctx)
