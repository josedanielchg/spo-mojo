"""El ruido de Dirichlet en la raiz, contra la formula de rlax.

Se espeja `apply_exploration_noise` de Stoix (`ff_spo.py:119`), que llama a
`rlax.add_dirichlet_noise`:

    noisy = (1 - fraction) * prior + fraction * noise,     noise ~ Dir(alpha)

Con alpha = 1 (el valor por defecto de Stoix) la Dirichlet simetrica es la uniforme
sobre el simplex, y se muestrea normalizando exponenciales: e_i = -ln(u_i),
noise = e / SUM(e). Aqui se comprueba esa cuenta con uniformes conocidas, asi que el
valor esperado se calcula a mano y no depende del generador.

Lo que se prueba, y por que cada cosa:
  - con fraction = 0 los logits quedan **bit a bit iguales** (es el defecto de
    Stoix, asi que toda la suite anterior depende de que sea inerte);
  - con fraction > 0 sale exactamente la formula;
  - una casilla tapada (NEG_INF) **sigue tapada**, o el ruido dejaria a la busqueda
    muestreando jugadas ilegales;
  - el ruido suma 1 sobre las acciones (es una Dirichlet, no un vector cualquiera);
  - y con mas de un bloque de hilos.
"""

from std.gpu.host import DeviceContext
from std.math import log, abs

from ops.common import dtype, NEG_INF
from systems.spo.root import add_dirichlet_noise_kernel
from systems.spo.launch import TPB, blocks_for
from tests.helpers import upload, download, assert_close

comptime TOL = Scalar[dtype](1e-5)
comptime N_ACT = 9


def run_noise(ctx: DeviceContext, logits: List[Scalar[dtype]],
              us: List[Scalar[dtype]], n_envs: Int,
              fraction: Scalar[dtype]) raises -> List[Scalar[dtype]]:
    lg = upload[dtype](ctx, logits)
    u = upload[dtype](ctx, us)
    ctx.enqueue_function[add_dirichlet_noise_kernel, add_dirichlet_noise_kernel](
        lg.unsafe_ptr(), u.unsafe_ptr(), n_envs, N_ACT, fraction,
        grid_dim=blocks_for(n_envs), block_dim=TPB)
    ctx.synchronize()
    return download[dtype](lg, n_envs * N_ACT)


def make_uniforms(n: Int, seed: Int) -> List[Scalar[dtype]]:
    """Uniformes deterministas en (0,1), sin depender del generador de la GPU."""
    out = List[Scalar[dtype]]()
    x = seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) % 2147483648
        out.append(Scalar[dtype](x % 100000 + 1) / Scalar[dtype](100001))
    return out^


def test_fraction_zero_is_a_no_op(ctx: DeviceContext) raises:
    """Con fraction = 0 los logits no se mueven NADA.

    Es el valor por defecto de Stoix, asi que toda la suite anterior (28 ficheros)
    depende de que esto sea inerte. Si el kernel tocara los logits con fraction=0,
    habria cambiado silenciosamente todos los resultados medidos hasta ahora.
    """
    n_envs = 3
    logits = List[Scalar[dtype]]()
    for e in range(n_envs):
        for a in range(N_ACT):
            logits.append(Scalar[dtype](e) - Scalar[dtype](a) * Scalar[dtype](0.37))
    us = make_uniforms(n_envs * N_ACT, 11)

    got = run_noise(ctx, logits, us, n_envs, Scalar[dtype](0))
    for i in range(n_envs * N_ACT):
        if got[i] != logits[i]:
            raise Error("con fraction=0 el logit ", i, " cambio: ", logits[i],
                        " -> ", got[i])
    print("PASS con fraction = 0 el ruido es inerte (bit a bit)")


def test_matches_the_rlax_formula(ctx: DeviceContext) raises:
    """(1-f)*prior + f*noise, con la Dirichlet(1) calculada a mano.

    El valor esperado se computa en host con los MISMOS uniformes, asi que si el
    kernel normalizara mal (por ejemplo dividiendo por n_actions en vez de por la
    suma) se veria: el ruido dejaria de sumar 1.
    """
    n_envs = 2
    fraction = Scalar[dtype](0.25)
    logits = List[Scalar[dtype]]()
    for e in range(n_envs):
        for a in range(N_ACT):
            logits.append(Scalar[dtype](0.5) * Scalar[dtype](a)
                          - Scalar[dtype](e))
    us = make_uniforms(n_envs * N_ACT, 29)

    got = run_noise(ctx, logits, us, n_envs, fraction)

    for e in range(n_envs):
        total = Scalar[dtype](0)
        for a in range(N_ACT):
            total += -log(us[e * N_ACT + a])
        noise_sum = Scalar[dtype](0)
        for a in range(N_ACT):
            noise = (-log(us[e * N_ACT + a])) / total
            noise_sum += noise
            want = (Scalar[dtype](1) - fraction) * logits[e * N_ACT + a] \
                   + fraction * noise
            assert_close(got[e * N_ACT + a], want, TOL,
                         String("env ", e, " accion ", a))
        # El ruido es una Dirichlet: sus componentes suman 1.
        assert_close(noise_sum, Scalar[dtype](1), TOL,
                     String("el ruido del env ", e, " deberia sumar 1"))
    print("PASS el ruido coincide con la formula de rlax (Dirichlet alpha=1)")


def test_masked_actions_stay_masked(ctx: DeviceContext) raises:
    """Una casilla tapada sigue efectivamente tapada tras el ruido.

    Si no, la busqueda muestrearia jugadas ilegales, y `ttt_apply` no comprueba
    legalidad: sobrescribiria la ficha del rival. El argumento es que
    (1-f)*NEG_INF domina cualquier ruido acotado en [0,1] mientras f < 1, pero eso
    hay que verlo, no suponerlo.
    """
    n_envs = 1
    fraction = Scalar[dtype](0.5)
    logits = List[Scalar[dtype]]()
    for a in range(N_ACT):
        logits.append(NEG_INF if a % 2 == 0 else Scalar[dtype](1.0))
    us = make_uniforms(N_ACT, 7)

    got = run_noise(ctx, logits, us, n_envs, fraction)
    for a in range(N_ACT):
        if a % 2 == 0:
            if got[a] > Scalar[dtype](-1e30):
                raise Error("la accion tapada ", a, " dejo de estarlo: ", got[a])
        else:
            if got[a] <= Scalar[dtype](-1e30):
                raise Error("la accion legal ", a, " se tapo: ", got[a])
    print("PASS las acciones tapadas siguen tapadas con fraction = 0.5")


def test_multi_block(ctx: DeviceContext) raises:
    """70 envs: mas de un bloque con TPB=32, y tamano no redondo.

    El guard `if e >= n_envs` solo se prueba si alguna vez se lanza con un tamano
    que no cuadra. Es el punto ciego que ya me comi cinco veces en este proyecto.
    """
    n_envs = 70
    fraction = Scalar[dtype](0.3)
    logits = List[Scalar[dtype]]()
    for e in range(n_envs):
        for a in range(N_ACT):
            logits.append(Scalar[dtype](e % 5) - Scalar[dtype](a))
    us = make_uniforms(n_envs * N_ACT, 101)

    got = run_noise(ctx, logits, us, n_envs, fraction)
    # Se comprueba el ultimo env, que es el que cae en el bloque parcial.
    e = n_envs - 1
    total = Scalar[dtype](0)
    for a in range(N_ACT):
        total += -log(us[e * N_ACT + a])
    for a in range(N_ACT):
        noise = (-log(us[e * N_ACT + a])) / total
        want = (Scalar[dtype](1) - fraction) * logits[e * N_ACT + a] \
               + fraction * noise
        assert_close(got[e * N_ACT + a], want, TOL,
                     String("env ", e, " accion ", a))
    print("PASS con 70 envs (varios bloques, tamano no redondo) sale igual")


def test_noise_flattens_a_peaked_prior(ctx: DeviceContext) raises:
    """El ruido acerca un prior muy picudo a la uniforme.

    Es el efecto por el que se activa: en E2.6 medimos que un prior aprendido muy
    seguro deja acciones legales con muy pocas particulas. Se comprueba que la
    diferencia entre el logit mayor y el menor BAJA al aplicar el ruido, que es la
    definicion operativa de "aplanar".
    """
    n_envs = 1
    logits = List[Scalar[dtype]]()
    for a in range(N_ACT):
        logits.append(Scalar[dtype](5) if a == 3 else Scalar[dtype](0))
    us = make_uniforms(N_ACT, 55)

    spread_before = Scalar[dtype](5)
    got = run_noise(ctx, logits, us, n_envs, Scalar[dtype](0.5))
    lo = got[0]; hi = got[0]
    for a in range(N_ACT):
        if got[a] < lo: lo = got[a]
        if got[a] > hi: hi = got[a]
    spread_after = hi - lo
    if spread_after >= spread_before:
        raise Error("el ruido deberia aplanar: rango ", spread_before, " -> ",
                    spread_after)
    print("PASS el ruido aplana un prior picudo: rango ", spread_before, " -> ",
          spread_after)


def main() raises:
    with DeviceContext() as ctx:
        test_fraction_zero_is_a_no_op(ctx)
        test_matches_the_rlax_formula(ctx)
        test_masked_actions_stay_masked(ctx)
        test_multi_block(ctx)
        test_noise_flattens_a_peaked_prior(ctx)
