"""El bucle de entorno REAL de CartPole (etapa 4.1).

A diferencia de la busqueda (imaginacion), aqui las acciones se ejecutan de
verdad, se acumula el retorno y el episodio se reinicia al terminar. Los tests
usan escenarios donde el retorno es EXACTO y conocido de antemano, no una
re-implementacion del kernel.
"""

from std.gpu.host import DeviceContext
from std.math import abs

from ops.common import dtype, idx_dtype
from envs.cartpole import cartpole_step_envs, STATE_DIM
from tests.helpers import upload, zeros, download, assert_close, assert_eq_int

comptime SEED = UInt32(555)


@fieldwise_init
struct EpisodeResult(Movable):
    """Lo que devuelve un episodio. Struct y no tupla porque en 1.0.0b1 una tupla
    con una List no se deja construir como valor de retorno."""
    var ret: Scalar[dtype]
    var steps: Int
    var final_state: List[Scalar[dtype]]     # tras el auto-reset, si termino


def run_fixed_action_episode(ctx: DeviceContext, state0: List[Scalar[dtype]],
                             action: Int, max_steps: Int) raises -> EpisodeResult:
    """Un env con una accion FIJA hasta que termina (o max_steps)."""
    state = upload[dtype](ctx, state0)
    acts = List[Scalar[idx_dtype]]()
    acts.append(Scalar[idx_dtype](action))
    actions = upload[idx_dtype](ctx, acts)
    reward = zeros[dtype](ctx, 1)
    done = zeros[idx_dtype](ctx, 1)
    reset_u = zeros[dtype](ctx, 4)

    ret = Scalar[dtype](0)
    steps = 0
    for step in range(max_steps):
        cartpole_step_envs(ctx, state, actions, reward, done, reset_u, 1,
                           SEED, UInt32(step))
        ctx.synchronize()
        ret += download[dtype](reward, 1)[0]
        steps += 1
        if Int(download[idx_dtype](done, 1)[0]) != 0:
            break

    return EpisodeResult(ret, steps, download[dtype](state, STATE_DIM))


def make_state(x: Scalar[dtype], x_dot: Scalar[dtype], theta: Scalar[dtype],
               theta_dot: Scalar[dtype], time: Scalar[dtype]) -> List[Scalar[dtype]]:
    s = List[Scalar[dtype]]()
    s.append(x); s.append(x_dot); s.append(theta); s.append(theta_dot); s.append(time)
    return s^


def test_return_is_step_count(ctx: DeviceContext) raises:
    """El retorno de un episodio = pasos aguantados (reward 1 por paso).

    Desde time=498, casi vertical: aguanta la fisica y termina por LIMITE de pasos
    en el paso 2 (time 498->499->500). Retorno esperado = 2."""
    res = run_fixed_action_episode(ctx, make_state(0, 0, 0, 0, 498), 0, 20)
    assert_eq_int(res.steps, 2, "deberia terminar en 2 pasos (limite de tiempo)")
    assert_close(res.ret, 2.0, 1e-6, "el retorno deberia ser 2 (reward 1 por paso)")
    print("PASS retorno = pasos aguantados (limite de tiempo: 2)")


def test_pole_fall_ends_episode(ctx: DeviceContext) raises:
    """Desde theta=0.20 (al borde) con velocidad hacia el umbral, el palo cae en
    el primer paso: theta 0.20 + 0.02*1.0 = 0.22 > 0.2094. Retorno = 1."""
    res = run_fixed_action_episode(ctx, make_state(0, 0, 0.20, 1.0, 0), 1, 20)
    assert_eq_int(res.steps, 1, "el palo deberia caer en 1 paso")
    assert_close(res.ret, 1.0, 1e-6, "el retorno deberia ser 1")
    print("PASS caida del palo termina el episodio (retorno 1)")


def test_auto_reset_after_episode(ctx: DeviceContext) raises:
    """Tras terminar, el env se auto-resetea: time=0 y estado en [-0.05, 0.05)."""
    res = run_fixed_action_episode(ctx, make_state(0, 0, 0.20, 1.0, 0), 1, 20)
    assert_close(res.final_state[4], 0.0, 1e-6, "time deberia resetearse a 0")
    for d in range(4):
        if res.final_state[d] < -0.05 or res.final_state[d] > 0.05:
            raise Error("componente ", d, " fuera del rango de reset: ", res.final_state[d])
    print("PASS el env se auto-resetea tras terminar (time=0, estado en rango)")


def test_alive_env_keeps_going(ctx: DeviceContext) raises:
    """Un env que NO termina no se resetea: sigue acumulando. Desde vertical con
    time=0, en 5 pasos no cae ni llega al limite, asi que retorno = 5 y sigue."""
    res = run_fixed_action_episode(ctx, make_state(0, 0, 0, 0, 0), 0, 5)
    assert_eq_int(res.steps, 5, "no deberia haber terminado en 5 pasos")
    assert_close(res.ret, 5.0, 1e-6, "retorno = 5 pasos sin terminar")
    # time avanzo a 5, no se reseteo
    assert_close(res.final_state[4], 5.0, 1e-6, "time deberia ser 5 (sin reset)")
    print("PASS un env vivo sigue acumulando sin resetearse")


def main() raises:
    with DeviceContext() as ctx:
        test_return_is_step_count(ctx)
        test_pole_fall_ends_episode(ctx)
        test_auto_reset_after_episode(ctx)
        test_alive_env_keeps_going(ctx)
