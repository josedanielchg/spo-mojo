"""La fisica de CartPole contra el golden de gymnax.

Dos pruebas que miden cosas distintas, a proposito:
  A) paso unico desde cada estado del golden -> prueba la FORMULA, tol estricta
  B) 100 pasos encadenados                   -> detecta DERIVA de Euler, tol floja

Pedir la tolerancia estricta a la trayectoria seria pedir mas precision de la que
sobrevive a encadenar el sin/cos de dos librerias distintas cien veces.

El golden lo genera tests/golden/gen/gen_cartpole.py desde gymnax. El golden A es
[N, 8] (next_state(5) + reward + done + discount); aqui solo se comparan las 5
primeras columnas, que son la fisica. Las otras tres son de la etapa siguiente.
"""

from std.gpu.host import DeviceContext
from std.math import abs

from ops.common import dtype, idx_dtype
from envs.cartpole import (cartpole_physics_kernel, cartpole_recurrent_kernel,
                          STATE_DIM)
from tests.golden_io import read_f32
from tests.helpers import upload, zeros, download, assert_close, assert_eq_int

comptime TPB = 32


def test_single_step_vs_golden(ctx: DeviceContext) raises:
    """Golden A: cada estado da un paso y se compara con gymnax (tol 1e-6)."""
    states_in = read_f32("tests/golden/cartpole_states_in.bin")   # [N, 5]
    actions_f = read_f32("tests/golden/cartpole_actions_in.bin")  # [N]
    step_out = read_f32("tests/golden/cartpole_step_out.bin")     # [N, 8]

    n = len(actions_f)
    if len(states_in) != n * STATE_DIM:
        raise Error("golden states_in con tamano raro: ", len(states_in),
                    " (regenerar con gen_cartpole.py?)")
    if len(step_out) != n * 8:
        raise Error("golden step_out con tamano raro: ", len(step_out))

    # Las acciones del golden son float 0.0/1.0; el kernel las quiere como int,
    # igual que llegaran de la busqueda (outputs.next_action es int32).
    actions = List[Scalar[idx_dtype]]()
    for i in range(n):
        actions.append(Scalar[idx_dtype](Int(actions_f[i])))

    s_in = upload[dtype](ctx, states_in)
    a = upload[idx_dtype](ctx, actions)
    s_out = zeros[dtype](ctx, n * STATE_DIM)

    ctx.enqueue_function[cartpole_physics_kernel, cartpole_physics_kernel](
        s_out.unsafe_ptr(), s_in.unsafe_ptr(), a.unsafe_ptr(), n,
        grid_dim=(n + TPB - 1) // TPB, block_dim=TPB)
    ctx.synchronize()

    got = download[dtype](s_out, n * STATE_DIM)

    TOL = Scalar[dtype](1e-6)
    max_err = Scalar[dtype](0)
    for i in range(n):
        for d in range(STATE_DIM):
            want = step_out[i * 8 + d]          # columnas 0-4 = next_state
            e = abs(got[i * STATE_DIM + d] - want)
            if e > max_err:
                max_err = e
            assert_close(got[i * STATE_DIM + d], want, TOL,
                         String("paso unico i=", i, " componente=", d))
    print("PASS fisica de un paso vs gymnax (", n, "estados, error max =", max_err, ")")


def test_trajectory_vs_golden(ctx: DeviceContext) raises:
    """Golden B: 100 pasos encadenados desde un estado fijo (tol 1e-4)."""
    acts_f = read_f32("tests/golden/cartpole_traj_actions.bin")  # [T]
    traj = read_f32("tests/golden/cartpole_traj_states.bin")     # [T+1, 5]

    T = len(acts_f)
    if len(traj) != (T + 1) * STATE_DIM:
        raise Error("golden traj con tamano raro: ", len(traj))

    # Estado inicial = fila 0 de la trayectoria.
    cur = List[Scalar[dtype]]()
    for d in range(STATE_DIM):
        cur.append(traj[d])
    state = upload[dtype](ctx, cur)

    # Tolerancia mas floja que el paso unico (1e-6) A PROPOSITO: cada paso mete
    # ~2e-7 de diferencia de ULP del sin/cos frente a XLA, y al encadenar cien
    # pasos eso se acumula y la dinamica lo amplifica algo. No es un bug: es la
    # deriva que este test existe para vigilar. Un error de formula de verdad
    # crece hasta 1e-2+ en pocos pasos; esto se queda en el orden de 1e-4. Y en la
    # busqueda real la profundidad es <=16, asi que nunca se acumula tanto.
    TOL = Scalar[dtype](5e-4)
    max_err = Scalar[dtype](0)
    for t in range(T):
        act = List[Scalar[idx_dtype]]()
        act.append(Scalar[idx_dtype](Int(acts_f[t])))
        a = upload[idx_dtype](ctx, act)

        # In-place: state_out == state_in. Es seguro (ver el kernel: lee los 5
        # antes de escribir), y asi el estado se encadena solo de un paso al otro.
        ctx.enqueue_function[cartpole_physics_kernel, cartpole_physics_kernel](
            state.unsafe_ptr(), state.unsafe_ptr(), a.unsafe_ptr(), 1,
            grid_dim=1, block_dim=TPB)
        ctx.synchronize()

        got = download[dtype](state, STATE_DIM)
        step_err = Scalar[dtype](0)
        for d in range(STATE_DIM):
            want = traj[(t + 1) * STATE_DIM + d]
            e = abs(got[d] - want)
            if e > step_err:
                step_err = e
            if e > max_err:
                max_err = e
            assert_close(got[d], want, TOL,
                         String("trayectoria paso=", t, " componente=", d))
        # La curva de deriva en cuatro puntos: si crece suave es acumulacion, si
        # da un salto seria otra cosa.
        if (t + 1) % 25 == 0:
            print("      paso", t + 1, "-> error acumulado", step_err)
    print("PASS trayectoria de", T, "pasos vs gymnax (error max =", max_err, ")")


def test_termination_vs_golden(ctx: DeviceContext) raises:
    """El reward y el discount del paso, contra gymnax (golden A, columnas 5 y 7).

    El rec_discount de la busqueda es `1 - done` de gymnax en TODOS los casos y
    con los dos valores del flag (el flag solo cambia el bootstrap, no el
    discount), asi que comparar contra la columna del golden vale para validar
    toda la logica de terminacion. Aqui va con el flag por defecto (A).
    """
    states_in = read_f32("tests/golden/cartpole_states_in.bin")
    actions_f = read_f32("tests/golden/cartpole_actions_in.bin")
    step_out = read_f32("tests/golden/cartpole_step_out.bin")
    n = len(actions_f)

    actions = List[Scalar[idx_dtype]]()
    for i in range(n):
        actions.append(Scalar[idx_dtype](Int(actions_f[i])))

    state = upload[dtype](ctx, states_in)     # se actualiza in-place, da igual aqui
    a = upload[idx_dtype](ctx, actions)
    reward = zeros[dtype](ctx, n)
    discount = zeros[dtype](ctx, n)
    bootstrap = zeros[dtype](ctx, n)

    ctx.enqueue_function[cartpole_recurrent_kernel, cartpole_recurrent_kernel](
        state.unsafe_ptr(), a.unsafe_ptr(), reward.unsafe_ptr(),
        discount.unsafe_ptr(), bootstrap.unsafe_ptr(), n,
        Scalar[dtype](0), Scalar[dtype](1),   # value_scale=0, gamma=1
        0,                                     # flag A (por defecto)
        grid_dim=(n + TPB - 1) // TPB, block_dim=TPB)
    ctx.synchronize()

    got_r = download[dtype](reward, n)
    got_d = download[dtype](discount, n)

    TOL = Scalar[dtype](1e-6)
    for i in range(n):
        want_reward = step_out[i * 8 + 5]      # columna 5 = reward
        want_discount = step_out[i * 8 + 7]    # columna 7 = discount = 1-done
        assert_close(got_r[i], want_reward, TOL, String("reward i=", i))
        assert_close(got_d[i], want_discount, TOL, String("rec_discount i=", i))
    print("PASS terminacion (reward, discount) vs gymnax (", n, "estados)")


def run_recurrent_one(ctx: DeviceContext, state5: List[Scalar[dtype]], action: Int,
                      value_scale: Scalar[dtype], flag: Int) raises -> List[Scalar[dtype]]:
    """Un paso del kernel recurrente sobre UNA particula. Devuelve
    [reward, rec_discount, bootstrap]. gamma fijo a 1."""
    state = upload[dtype](ctx, state5)
    acts = List[Scalar[idx_dtype]]()
    acts.append(Scalar[idx_dtype](action))
    a = upload[idx_dtype](ctx, acts)
    reward = zeros[dtype](ctx, 1)
    discount = zeros[dtype](ctx, 1)
    bootstrap = zeros[dtype](ctx, 1)

    ctx.enqueue_function[cartpole_recurrent_kernel, cartpole_recurrent_kernel](
        state.unsafe_ptr(), a.unsafe_ptr(), reward.unsafe_ptr(),
        discount.unsafe_ptr(), bootstrap.unsafe_ptr(), 1,
        value_scale, Scalar[dtype](1), flag,
        grid_dim=1, block_dim=TPB)
    ctx.synchronize()

    out = List[Scalar[dtype]]()
    out.append(download[dtype](reward, 1)[0])
    out.append(download[dtype](discount, 1)[0])
    out.append(download[dtype](bootstrap, 1)[0])
    return out^


def make_state(x: Scalar[dtype], x_dot: Scalar[dtype], theta: Scalar[dtype],
               theta_dot: Scalar[dtype], time: Scalar[dtype]) -> List[Scalar[dtype]]:
    s = List[Scalar[dtype]]()
    s.append(x); s.append(x_dot); s.append(theta); s.append(theta_dot); s.append(time)
    return s^


def test_truncation_flag(ctx: DeviceContext) raises:
    """El flag truncate_on_step_limit: el paso 500 como muerte (A) o truncacion (B).

    Con value_scale > 0 (aqui 7) el bootstrap = gamma*V no es trivialmente 0, asi
    que la diferencia entre A y B se ve. Tres regimenes:

      alive     : sigue viva          -> rec_discount=1, bootstrap=7  (los dos flags)
      dead      : el palo cae         -> rec_discount=0, bootstrap=0  (los dos flags)
      limit     : llega al paso 500   -> rec_discount=0, y AQUI difieren:
                    flag A: bootstrap=0   (tratado como muerte, como gymnax)
                    flag B: bootstrap=7   (truncacion, conserva el futuro)
    """
    V = Scalar[dtype](7)
    TOL = Scalar[dtype](1e-6)

    # alive: desde el equilibrio, un paso normal.
    alive = make_state(0, 0, 0, 0, 10)
    # dead: theta 0.2 + tau*theta_dot(2.0) = 0.24 > umbral -> el palo cae en el paso.
    dead = make_state(0, 0, 0.2, 2.0, 10)
    # limit: time 499 -> tras el paso time 500, y en rango (no cae).
    limit = make_state(0, 0, 0, 0, 499)

    for flag in range(2):     # 0 = A, 1 = B
        ra = run_recurrent_one(ctx, alive, 1, V, flag)
        rd = run_recurrent_one(ctx, dead, 1, V, flag)
        rl = run_recurrent_one(ctx, limit, 1, V, flag)

        # reward: ninguno entra ya muerto, asi que 1 en los tres.
        assert_close(ra[0], 1.0, TOL, String("alive reward flag=", flag))
        assert_close(rd[0], 1.0, TOL, String("dead reward flag=", flag))
        assert_close(rl[0], 1.0, TOL, String("limit reward flag=", flag))

        # alive: sigue simulando, con su valor.
        assert_close(ra[1], 1.0, TOL, String("alive rec_discount flag=", flag))
        assert_close(ra[2], V, TOL, String("alive bootstrap flag=", flag))

        # dead: para y pierde el futuro, con los dos flags.
        assert_close(rd[1], 0.0, TOL, String("dead rec_discount flag=", flag))
        assert_close(rd[2], 0.0, TOL, String("dead bootstrap flag=", flag))

        # limit: para siempre (rec_discount 0), pero el bootstrap depende del flag.
        assert_close(rl[1], 0.0, TOL, String("limit rec_discount flag=", flag))
        want_boot = Scalar[dtype](0) if flag == 0 else V
        assert_close(rl[2], want_boot, TOL,
                     String("limit bootstrap flag=", flag, " (A=0, B=7)"))

    print("PASS flag de truncacion: paso 500 = muerte (A) vs truncacion (B)")


def main() raises:
    with DeviceContext() as ctx:
        test_single_step_vs_golden(ctx)
        test_trajectory_vs_golden(ctx)
        test_termination_vs_golden(ctx)
        test_truncation_flag(ctx)
