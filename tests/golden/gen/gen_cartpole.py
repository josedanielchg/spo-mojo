"""Genera el golden de CartPole a partir de gymnax.

Correr desde la raiz de mojo_spo:
    ../.venv/bin/python tests/golden/gen/gen_cartpole.py

Importa el CartPole de gymnax de verdad y llama a su `step_env` (el paso crudo,
SIN auto-reset), asi el golden es exactamente lo que produce gymnax y no mi
transcripcion de las formulas. `envs/cartpole.mojo` tiene que reproducirlo.

Escribe dos goldens:

  A) PASO UNICO desde N estados variados (para probar la FORMULA sin acumular
     error). Incluye angulos grandes, velocidades altas, los bordes de
     terminacion, el paso 500 y estados YA terminales (para el reward = 1 - prev).
       cartpole_states_in.bin   [N, 5]  estados de entrada (x, x_dot, theta, theta_dot, time)
       cartpole_actions_in.bin  [N]     accion de cada uno (0.0 / 1.0)
       cartpole_step_out.bin    [N, 8]  next_state(5) + reward + done + discount

  B) TRAYECTORIA libre de T pasos desde un estado fijo (para detectar DERIVA:
     el error de Euler que se acumula paso a paso). Un controlador simple la
     mantiene cerca del equilibrio para que no se dispare.
       cartpole_traj_actions.bin [T]      la secuencia de acciones
       cartpole_traj_states.bin  [T+1, 5] la trayectoria (fila 0 = estado inicial)

  cartpole.txt  las shapes y que hay en cada sitio.

Nota sobre truncacion: gymnax NO distingue truncacion de terminacion -- su
discount es 1 - done, y `done` incluye el limite de 500 pasos. O sea que este
golden captura el comportamiento de la OPCION A (paso 500 = terminal). La opcion
B (el flag truncate_on_step_limit) es una desviacion de gymnax y se prueba aparte
en Mojo, no contra este golden.
"""

import os

import jax
import numpy as np
from gymnax.environments.classic_control.cartpole import CartPole, EnvParams, EnvState

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

env = CartPole()
params = env.default_params
KEY = jax.random.PRNGKey(0)  # step_env de cartpole no lo usa, pero la firma lo pide

TH = float(params.theta_threshold_radians)  # 0.2094395...
XT = float(params.x_threshold)              # 2.4
MAX_STEPS = int(params.max_steps_in_episode)  # 500


def gymnax_step(state5, action):
    """Un paso crudo de gymnax. Devuelve (next_state5, reward, done, discount)."""
    x, x_dot, theta, theta_dot, time = state5
    state = EnvState(
        x=jax.numpy.float32(x),
        x_dot=jax.numpy.float32(x_dot),
        theta=jax.numpy.float32(theta),
        theta_dot=jax.numpy.float32(theta_dot),
        time=int(time),
    )
    _, ns, reward, done, info = env.step_env(KEY, state, int(action), params)
    next5 = [float(ns.x), float(ns.x_dot), float(ns.theta), float(ns.theta_dot), float(ns.time)]
    return next5, float(reward), float(done), float(info["discount"])


# ---------------------------------------------------------------------------
# GOLDEN A -- paso unico desde estados escogidos a mano
# ---------------------------------------------------------------------------

# Cada fila: (x, x_dot, theta, theta_dot, time, action). Agrupadas por intencion.
cases = []

# 1. Cerca del equilibrio, perturbaciones pequenas, las dos acciones.
for th in (-0.05, 0.0, 0.05):
    for a in (0, 1):
        cases.append((0.0, 0.0, th, 0.0, 0, a))

# 2. Angulos mayores (aun validos) -> cos/sin lejos de 1/0.
for th in (-0.15, 0.15):
    for a in (0, 1):
        cases.append((0.3, -0.2, th, 0.4, 10, a))

# 3. Velocidades altas -> el termino theta_dot**2 pesa.
cases.append((1.0, 1.5, 0.1, 2.0, 5, 1))
cases.append((-1.0, -1.5, -0.1, -2.0, 5, 0))
cases.append((0.0, 0.0, 0.0, 3.0, 5, 1))

# 4. Borde del angulo: justo por debajo y por encima del umbral.
cases.append((0.0, 0.0, 0.20, 0.0, 3, 1))    # < 0.2094 -> sigue vivo
cases.append((0.0, 0.0, 0.21, 0.0, 3, 1))    # > 0.2094 -> muere (done)
cases.append((0.0, 0.0, -0.21, 0.0, 3, 0))   # simetrico

# 5. Borde de la posicion: cerca de +-2.4.
cases.append((2.35, 0.5, 0.0, 0.0, 3, 1))    # el paso lo empuja fuera
cases.append((-2.35, -0.5, 0.0, 0.0, 3, 0))

# 6. Limite de pasos: time 499 -> tras el paso time 500 -> done.
cases.append((0.0, 0.0, 0.0, 0.0, 499, 1))
cases.append((0.1, 0.1, 0.05, 0.1, 499, 0))

# 7. Estados YA terminales al entrar -> reward tiene que ser 0 (1 - prev_terminal).
cases.append((3.0, 0.0, 0.0, 0.0, 10, 1))    # x fuera de pista
cases.append((0.0, 0.0, 0.5, 0.0, 10, 0))    # angulo pasado
cases.append((0.0, 0.0, 0.0, 0.0, 500, 1))   # ya en el limite de pasos
cases.append((0.0, 0.0, 0.0, 0.0, 600, 0))   # pasado el limite

# 8. Unos cuantos aleatorios pero fijos, por si me deje algun regimen.
rng = np.random.default_rng(0)
for _ in range(6):
    x = float(rng.uniform(-2.0, 2.0))
    x_dot = float(rng.uniform(-2.0, 2.0))
    theta = float(rng.uniform(-0.18, 0.18))
    theta_dot = float(rng.uniform(-2.0, 2.0))
    time = int(rng.integers(0, 400))
    a = int(rng.integers(0, 2))
    cases.append((x, x_dot, theta, theta_dot, time, a))

N = len(cases)
states_in = np.array([c[:5] for c in cases], dtype=np.float32)
actions_in = np.array([c[5] for c in cases], dtype=np.float32)

step_out = np.zeros((N, 8), dtype=np.float32)
for i, c in enumerate(cases):
    next5, reward, done, discount = gymnax_step(c[:5], c[5])
    step_out[i, :5] = next5
    step_out[i, 5] = reward
    step_out[i, 6] = done
    step_out[i, 7] = discount

states_in.tofile(os.path.join(OUT, "cartpole_states_in.bin"))
actions_in.tofile(os.path.join(OUT, "cartpole_actions_in.bin"))
step_out.tofile(os.path.join(OUT, "cartpole_step_out.bin"))

# ---------------------------------------------------------------------------
# GOLDEN B -- trayectoria libre de T pasos
# ---------------------------------------------------------------------------

T = 100
init = [0.0, 0.0, 0.05, 0.0, 0]     # casi vertical, ligera inclinacion
traj = np.zeros((T + 1, 5), dtype=np.float32)
acts = np.zeros(T, dtype=np.float32)
traj[0] = init

state5 = list(init)
for t in range(T):
    # Controlador PD simple sobre el angulo: empuja hacia donde cae el palo.
    # No es optimo, pero mantiene la trayectoria acotada y cerca del regimen
    # que de verdad importa (casi vertical), donde tol 1e-4 tiene sentido.
    theta, theta_dot = state5[2], state5[3]
    action = 1 if (theta + 0.5 * theta_dot) > 0.0 else 0
    acts[t] = float(action)
    next5, _, _, _ = gymnax_step(state5, action)
    traj[t + 1] = next5
    state5 = next5

acts.tofile(os.path.join(OUT, "cartpole_traj_actions.bin"))
traj.tofile(os.path.join(OUT, "cartpole_traj_states.bin"))

max_abs = float(np.max(np.abs(traj[:, :4])))  # ignoro time para el rango

with open(os.path.join(OUT, "cartpole.txt"), "w") as f:
    f.write(f"N {N}\n")
    f.write(f"T {T}\n")
    f.write(f"theta_threshold {TH}\n")
    f.write(f"x_threshold {XT}\n")
    f.write(f"max_steps {MAX_STEPS}\n")
    f.write("state = (x, x_dot, theta, theta_dot, time)\n")
    f.write("--- golden A: paso unico ---\n")
    f.write("cartpole_states_in.bin  float32 [N, 5]  estados de entrada\n")
    f.write("cartpole_actions_in.bin float32 [N]     accion de cada uno (0/1)\n")
    f.write("cartpole_step_out.bin   float32 [N, 8]  next_state(5)+reward+done+discount\n")
    f.write("--- golden B: trayectoria ---\n")
    f.write("cartpole_traj_actions.bin float32 [T]      secuencia de acciones\n")
    f.write("cartpole_traj_states.bin  float32 [T+1, 5] trayectoria (fila 0 = inicial)\n")
    f.write(f"traj_max_abs_state {max_abs}\n")

print(f"golden de cartpole escrito en {OUT}  (N={N}, T={T}, max|traj|={max_abs:.4f})")
