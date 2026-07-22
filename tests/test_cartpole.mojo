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
from ops.rng import fill_uniform
from envs.cartpole import (cartpole_physics_kernel, cartpole_recurrent_kernel,
                          cartpole_reset_kernel, cartpole_auto_reset_kernel,
                          STATE_DIM)
from tests.golden_io import read_f32
from tests.helpers import (upload, zeros, download, write_into, assert_close,
                          assert_eq_int)

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


def test_reset_range_and_determinism(ctx: DeviceContext) raises:
    """10k resets: todos en [-0.05, 0.05), time=0, media ~0, y deterministas."""
    n = 10000
    u = zeros[dtype](ctx, n * 4)      # 4 uniformes por env
    ctx.enqueue_function[fill_uniform, fill_uniform](
        u.unsafe_ptr(), UInt32(123), UInt32(0), n * 4,
        grid_dim=(n * 4 + TPB - 1) // TPB, block_dim=TPB)

    state = zeros[dtype](ctx, n * STATE_DIM)
    ctx.enqueue_function[cartpole_reset_kernel, cartpole_reset_kernel](
        state.unsafe_ptr(), u.unsafe_ptr(), n,
        grid_dim=(n + TPB - 1) // TPB, block_dim=TPB)
    ctx.synchronize()

    got = download[dtype](state, n * STATE_DIM)

    # Rango: las 4 dinamicas en [-0.05, 0.05), time exactamente 0.
    # Media en float64: sumar 40k floats en float32 arrastra error.
    sums = List[Float64]()
    for _ in range(4):
        sums.append(Float64(0))
    for e in range(n):
        base = e * STATE_DIM
        for d in range(4):
            v = got[base + d]
            if v < -0.05 or v > 0.05:
                raise Error("componente ", d, " del env ", e, " fuera de rango: ", v)
            sums[d] += Float64(v)
        if got[base + 4] != 0.0:
            raise Error("time del env ", e, " no es 0: ", got[base + 4])

    # U(-0.05, 0.05) tiene media 0. Con 10k muestras el error tipico es ~3e-4, asi
    # que |media| > 0.002 delataria un sesgo (p. ej. mapear a U(0, 0.1) por error).
    for d in range(4):
        mean = sums[d] / Float64(n)
        if abs(mean) > 0.002:
            raise Error("la media de la componente ", d, " esta sesgada: ", mean)

    # Determinismo: misma seed -> mismos estados, bit a bit.
    u2 = zeros[dtype](ctx, n * 4)
    ctx.enqueue_function[fill_uniform, fill_uniform](
        u2.unsafe_ptr(), UInt32(123), UInt32(0), n * 4,
        grid_dim=(n * 4 + TPB - 1) // TPB, block_dim=TPB)
    state2 = zeros[dtype](ctx, n * STATE_DIM)
    ctx.enqueue_function[cartpole_reset_kernel, cartpole_reset_kernel](
        state2.unsafe_ptr(), u2.unsafe_ptr(), n,
        grid_dim=(n + TPB - 1) // TPB, block_dim=TPB)
    ctx.synchronize()
    got2 = download[dtype](state2, n * STATE_DIM)
    for i in range(n * STATE_DIM):
        if got[i] != got2[i]:
            raise Error("reset no determinista en ", i, ": ", got[i], " vs ", got2[i])

    print("PASS reset:", n, "estados en rango, time=0, media ~0, deterministas")


def test_auto_reset_conditional(ctx: DeviceContext) raises:
    """El auto-reset toca SOLO los envs con done=1; los vivos quedan intactos."""
    n = 4
    # Todos empiezan en un estado claramente NO reseteado, para notar quien cambia.
    marked = List[Scalar[dtype]]()
    for _ in range(n):
        marked.append(1.0); marked.append(1.0); marked.append(1.0)
        marked.append(1.0); marked.append(50.0)     # time=50, imposible tras reset
    state = upload[dtype](ctx, marked)

    # done = [1, 0, 1, 0] -> resetean el 0 y el 2, siguen el 1 y el 3.
    done_list = List[Scalar[idx_dtype]]()
    done_list.append(1); done_list.append(0); done_list.append(1); done_list.append(0)
    done = upload[idx_dtype](ctx, done_list)

    u = zeros[dtype](ctx, n * 4)
    ctx.enqueue_function[fill_uniform, fill_uniform](
        u.unsafe_ptr(), UInt32(7), UInt32(0), n * 4,
        grid_dim=(n * 4 + TPB - 1) // TPB, block_dim=TPB)

    ctx.enqueue_function[cartpole_auto_reset_kernel, cartpole_auto_reset_kernel](
        state.unsafe_ptr(), done.unsafe_ptr(), u.unsafe_ptr(), n,
        grid_dim=(n + TPB - 1) // TPB, block_dim=TPB)
    ctx.synchronize()

    got = download[dtype](state, n * STATE_DIM)
    for e in range(n):
        base = e * STATE_DIM
        if Int(done_list[e]) != 0:
            # reseteado: en rango y time=0
            for d in range(4):
                v = got[base + d]
                if v < -0.05 or v > 0.05:
                    raise Error("env reseteado ", e, " componente ", d, " fuera de rango: ", v)
            assert_close(got[base + 4], 0.0, 1e-6, String("env reseteado ", e, " time"))
        else:
            # intacto: sigue con las marcas (1,1,1,1,50)
            for d in range(4):
                assert_close(got[base + d], 1.0, 1e-6,
                             String("env vivo ", e, " no deberia cambiar, componente ", d))
            assert_close(got[base + 4], 50.0, 1e-6, String("env vivo ", e, " time"))
    print("PASS auto-reset: solo resetea los envs terminados")


def main() raises:
    with DeviceContext() as ctx:
        test_single_step_vs_golden(ctx)
        test_trajectory_vs_golden(ctx)
        test_termination_vs_golden(ctx)
        test_truncation_flag(ctx)
        test_reset_range_and_determinism(ctx)
        test_auto_reset_conditional(ctx)
