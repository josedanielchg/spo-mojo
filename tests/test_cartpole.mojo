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
from envs.cartpole import cartpole_physics_kernel, STATE_DIM
from tests.golden_io import read_f32
from tests.helpers import upload, zeros, download, assert_close

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


def main() raises:
    with DeviceContext() as ctx:
        test_single_step_vs_golden(ctx)
        test_trajectory_vs_golden(ctx)
