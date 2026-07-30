"""E1.11: entrenar el critico, enchufarlo a la busqueda y medir si sirve.

La pregunta de la etapa 1 entera es una sola: **¿bajan las derrotas?** El
planificador sin critico pierde el 2.02% de las partidas medido en A7; el juego
optimo pierde el 0.00%. Ese 2% son amenazas del rival que la busqueda no vio.

Que hace exactamente el critico en el peso, sacado del codigo y no de la intuicion.
`update_particles_kernel` acumula `weights += r_d + next_value_d - value_d` y
justo despues hace `value_{d+1} = next_value_d`, asi que la suma TELESCOPA y el
peso final de una particula es

    peso = SUMA_d r_d  +  (ultimo bootstrap)  -  V(s_raiz)

Y `root.mojo:49` reparte el MISMO V(s_raiz) a todas las particulas de un env, asi
que ese termino es una constante por env y el softmax lo cancela entero. O sea que
el critico NO entra como linea base: su unico efecto es el bootstrap de las
particulas que siguen VIVAS al final de la busqueda. Escrito por casos, con V ~ c:

                        sin critico          con critico
    ganar en d          gamma_r^d            gamma_r^d
    perder en d         0                    0
    seguir viva         0                    c ~ 0.93

Ahi esta el problema. Con `reward_gamma = 0.7`, ganar en la profundidad 1 vale 0.7
y seguir vivo vale 0.93: **sobrevivir puntua mas que ganar**, salvo que se gane en
el acto. La busqueda con critico prefiere no resolver la partida. Y no es un fallo
del critico, es un desajuste de escalas: la recompensa lleva gamma_r^d plegado
(decision de A6) y el valor no.

De ahi salen los seis montajes que se miden abajo:

  1-4. el cruce {sin critico, con critico} x {gamma_r 0.7, 1.0}, que es el barrido
       que el plan pedia re-medir ahora que el bootstrap tiene efecto;
  5.   el bootstrap descontado por profundidad (`depth_discounted`), que pone las
       dos mitades de la suma en la misma escala y deja el orden que se quiere:

           ganar pronto  >  ganar tarde  >  sobrevivir  >  perder

       Es la configuracion que la teoria dice que deberia funcionar, y esta aqui
       para no declarar un resultado negativo sin haber probado el caso favorable.
  6.   el mismo montaje 5 pero con un "critico" que devuelve siempre la constante
       c. Separa NIVEL de INFORMACION: si empata con el critico entrenado, es que
       la red no aporta nada mas que su media, y la separacion de 0.0072 que se
       midio en E1.10 no llega a cambiar ninguna decision.

Todos los brazos juegan las MISMAS partidas: mismo numero de envs, mismos pasos y
la misma semilla para el rival. Lo unico que cambia entre ellos es el modelo.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns

from bench.metrics import fmt_fixed, wilson_lo, wilson_hi
from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform
from envs.tictactoe import (TicTacToe, ttt_reset_kernel, ttt_env_step_kernel,
                            ttt_auto_reset_kernel, ttt_encode_obs_kernel,
                            NUM_ACTIONS, STATE_DIM, OBS_DIM, TPB_TTT)
from envs.tictactoe_critic import TicTacToeCritic
from envs.tictactoe_runner import RNG_RIVAL
from networks.mlp import CriticCache, critic_forward
from rl_utils.buffer import TrajectoryBuffer
from systems.spo.particles import SearchWorkspace
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from systems.spo.search_model import SearchModel
from tests.helpers import download, write_into
from demos.train_critic import (Critic, init_critic_weights, collect, update,
                                SEED, NUM_ENVS, HIDDEN, OUT_DIM, BATCH, ROLLOUT,
                                BUFFER_CAP, NUM_PARTICLES, SEARCH_DEPTH,
                                RESAMPLE_PERIOD, TEMPERATURE, REWARD_GAMMA)

# La evaluacion es aparte del entrenamiento: mas partidas y mas envs, porque lo
# que se quiere es un intervalo de confianza estrecho sobre una tasa del ~2%.
comptime EVAL_ENVS = 128
comptime EVAL_STEPS = 400
comptime EVAL_SEED = UInt32(20260730)

comptime TRAIN_ROUNDS = 25
comptime UPDATES_PER_ROUND = 20


@fieldwise_init
struct Arm(Copyable, Movable):
    """El resultado de un brazo del experimento."""
    var name: String
    var wins: Int
    var draws: Int
    var losses: Int
    var seconds: Float64

    def games(self) -> Int:
        return self.wins + self.draws + self.losses

    def score(self) -> Float64:
        n = self.games()
        return (Float64(self.wins) + 0.5 * Float64(self.draws)) / Float64(n)


def eval_config(reward_gamma: Scalar[dtype]) -> SPOConfig:
    """La config de busqueda de A7, con EVAL_ENVS partidas a la vez."""
    return SPOConfig(num_envs=EVAL_ENVS, num_particles=NUM_PARTICLES,
                     num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                     search_depth=SEARCH_DEPTH, resample_period=RESAMPLE_PERIOD,
                     temperature=TEMPERATURE, search_gamma=1.0,
                     search_gae_lambda=1.0)


def play[M: SearchModel](ctx: DeviceContext, name: String, model: M,
                         cfg: SPOConfig, steps: Int) raises -> Arm:
    """Juega `steps` turnos en EVAL_ENVS partidas y cuenta el marcador.

    Es el bucle de A7 tal cual: buscar, jugar, dejar responder al rival, anotar
    las que acaben y resetear las acabadas. La semilla del rival depende solo del
    paso, asi que todos los brazos se enfrentan a la misma secuencia de rivales.
    """
    ws = SearchWorkspace(ctx, cfg)
    blocks = (EVAL_ENVS + TPB_TTT - 1) // TPB_TTT

    state = zero_buffer[dtype](ctx, EVAL_ENVS * STATE_DIM)
    reward = zero_buffer[dtype](ctx, EVAL_ENVS)
    done = zero_buffer[idx_dtype](ctx, EVAL_ENVS)
    u_rival = zero_buffer[dtype](ctx, EVAL_ENVS)

    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()

    wins = 0
    draws = 0
    losses = 0
    start = perf_counter_ns()
    for step in range(steps):
        search[M](ctx, ws, cfg, model, state,
                  EVAL_SEED ^ (UInt32(step) * 2654435761))
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), EVAL_SEED, RNG_RIVAL + UInt32(step), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), ws.output.action.unsafe_ptr(),
            u_rival.unsafe_ptr(), reward.unsafe_ptr(), done.unsafe_ptr(),
            EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

        with reward.map_to_host() as rh:
            with done.map_to_host() as dh:
                for e in range(EVAL_ENVS):
                    if Int(dh[e]) != 0:
                        r = rh[e]
                        if r > Scalar[dtype](0.75): wins += 1
                        elif r > Scalar[dtype](0.25): draws += 1
                        else: losses += 1

        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()
    seconds = Float64(perf_counter_ns() - start) / 1e9
    return Arm(name, wins, draws, losses, seconds)


def pct(x: Float64) -> String:
    return fmt_fixed(x * 100.0, 2) + "%"


def show(arm: Arm) raises:
    """Una linea de la tabla, con el IC de Wilson sobre las DERROTAS.

    El intervalo va en las derrotas y no en las victorias porque es la columna que
    responde a la pregunta: el juego optimo pierde 0.00%, asi que las derrotas
    miden directamente lo que la busqueda no vio.
    """
    n = arm.games()
    lo = wilson_lo(arm.losses, n)
    hi = wilson_hi(arm.losses, n)
    print("  ", arm.name,
          "  n=", n,
          "  gana ", pct(Float64(arm.wins) / Float64(n)),
          "  empata ", pct(Float64(arm.draws) / Float64(n)),
          "  PIERDE ", pct(Float64(arm.losses) / Float64(n)),
          " IC[", fmt_fixed(lo * 100.0, 2), ", ", fmt_fixed(hi * 100.0, 2), "]",
          "  score ", fmt_fixed(arm.score(), 4))


def verdict(base: Arm, other: Arm) raises:
    """Compara dos brazos por la tasa de derrotas y dice si se puede afirmar algo.

    La regla que nos pusimos: si los intervalos de Wilson se SOLAPAN, no se afirma
    mejora. No solaparse es una condicion mas exigente que el test formal de dos
    proporciones, asi que quedarse con ella es conservador y facil de defender.
    """
    n1 = base.games(); n2 = other.games()
    lo1 = wilson_lo(base.losses, n1); hi1 = wilson_hi(base.losses, n1)
    lo2 = wilson_lo(other.losses, n2); hi2 = wilson_hi(other.losses, n2)
    p1 = Float64(base.losses) / Float64(n1)
    p2 = Float64(other.losses) / Float64(n2)

    print("   ", other.name, " vs ", base.name, ":")
    print("      derrotas ", pct(p1), " -> ", pct(p2))
    if hi2 < lo1:
        print("      los intervalos NO se solapan y el nuevo esta por debajo:",
              " se puede afirmar que pierde menos.")
    elif lo2 > hi1:
        print("      los intervalos NO se solapan y el nuevo esta por ENCIMA:",
              " pierde mas, la diferencia es real.")
    else:
        print("      los intervalos se SOLAPAN: no se afirma diferencia.")


def mean_value(ctx: DeviceContext, mut critic: Critic, cfg: SPOConfig,
               model: TicTacToe) raises -> Scalar[dtype]:
    """V medio del critico sobre tableros de partidas reales.

    Es el numero `c` del razonamiento de la cabecera: el nivel de referencia con
    el que se van a comparar las recompensas. Se imprime para poder seguir la
    cuenta a mano en el informe.
    """
    blocks = (EVAL_ENVS + TPB_TTT - 1) // TPB_TTT
    cache = CriticCache(ctx, EVAL_ENVS, HIDDEN, OUT_DIM)
    ws = SearchWorkspace(ctx, cfg)
    state = zero_buffer[dtype](ctx, EVAL_ENVS * STATE_DIM)
    obs = zero_buffer[dtype](ctx, EVAL_ENVS * OBS_DIM)
    reward = zero_buffer[dtype](ctx, EVAL_ENVS)
    done = zero_buffer[idx_dtype](ctx, EVAL_ENVS)
    u_rival = zero_buffer[dtype](ctx, EVAL_ENVS)

    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)

    total = Scalar[dtype](0)
    count = 0
    for step in range(30):
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            obs.unsafe_ptr(), state.unsafe_ptr(), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        critic_forward(ctx, critic.online, cache, obs, EVAL_ENVS)
        ctx.synchronize()
        vs = download[dtype](cache.value, EVAL_ENVS)
        for e in range(EVAL_ENVS):
            total += vs[e]
            count += 1

        search[TicTacToe](ctx, ws, cfg, model, state,
                          EVAL_SEED ^ (UInt32(step) * 40503))
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), EVAL_SEED, RNG_RIVAL + UInt32(step), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), ws.output.action.unsafe_ptr(),
            u_rival.unsafe_ptr(), reward.unsafe_ptr(), done.unsafe_ptr(),
            EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()
    return total / Scalar[dtype](count)


def train(ctx: DeviceContext, mut critic: Critic) raises:
    """El entrenamiento de E1.10, tal cual: el critico aprende de la busqueda
    SIN critico, que es la unica politica que hay todavia."""
    cfg = SPOConfig(num_envs=NUM_ENVS, num_particles=NUM_PARTICLES,
                    num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                    search_depth=SEARCH_DEPTH, resample_period=RESAMPLE_PERIOD,
                    temperature=TEMPERATURE, search_gamma=1.0,
                    search_gae_lambda=1.0)
    model = TicTacToe(reward_gamma=REWARD_GAMMA)
    ws = SearchWorkspace(ctx, cfg)
    buf = TrajectoryBuffer(BUFFER_CAP, ROLLOUT, OBS_DIM)

    blocks = (NUM_ENVS + TPB_TTT - 1) // TPB_TTT
    state = zero_buffer[dtype](ctx, NUM_ENVS * STATE_DIM)
    obs_buf = zero_buffer[dtype](ctx, NUM_ENVS * OBS_DIM)
    next_obs_buf = zero_buffer[dtype](ctx, NUM_ENVS * OBS_DIM)
    reward = zero_buffer[dtype](ctx, NUM_ENVS)
    done = zero_buffer[idx_dtype](ctx, NUM_ENVS)
    u_rival = zero_buffer[dtype](ctx, NUM_ENVS)
    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), NUM_ENVS, grid_dim=blocks, block_dim=TPB_TTT)

    step = 0
    last = Scalar[dtype](0)
    for round_idx in range(TRAIN_ROUNDS):
        _ = collect(ctx, buf, cfg, model, ws, state, obs_buf, next_obs_buf,
                    reward, done, u_rival, SEED, round_idx)
        for _ in range(UPDATES_PER_ROUND):
            step += 1
            last = update(ctx, critic, buf, step, SEED)
    print("   entrenado:", TRAIN_ROUNDS * UPDATES_PER_ROUND,
          "pasos de gradiente, perdida final", last)


def main() raises:
    with DeviceContext() as ctx:
        print("=== E1.11: el critico entra en la busqueda ===")
        print("   evaluacion:", EVAL_ENVS, "partidas a la vez x", EVAL_STEPS,
              "turnos, misma semilla en todos los brazos")
        print()

        # 1. Entrenar el critico (E1.10).
        print("--- 1. entrenamiento del critico ---")
        critic = Critic(ctx, BATCH * ROLLOUT)
        init_critic_weights(ctx, critic, SEED)
        train(ctx, critic)

        cfg_train = SPOConfig(num_envs=EVAL_ENVS, num_particles=NUM_PARTICLES,
                              num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                              search_depth=SEARCH_DEPTH,
                              resample_period=RESAMPLE_PERIOD,
                              temperature=TEMPERATURE, search_gamma=1.0,
                              search_gae_lambda=1.0)
        c = mean_value(ctx, critic, cfg_train, TicTacToe(REWARD_GAMMA))
        print("   V medio sobre posiciones reales: c =", c)
        print("   peso de una particula segun como acabe (V(raiz) se cancela):")
        print("      ganar en d=0   ", Scalar[dtype](1))
        print("      ganar en d=1   ", Scalar[dtype](0.7),
              "   <- con reward_gamma=0.7")
        print("      ganar en d=3   ", Scalar[dtype](0.343))
        print("      seguir viva    ", c, "  <- esto es lo que anade el critico")
        print("      perder         ", Scalar[dtype](0))
        print()

        # 2. Los brazos.
        max_batch = EVAL_ENVS * NUM_PARTICLES
        m_c07 = TicTacToeCritic(ctx, max_batch, HIDDEN, Scalar[dtype](0.7))
        m_c10 = TicTacToeCritic(ctx, max_batch, HIDDEN, Scalar[dtype](1.0))
        m_dd = TicTacToeCritic(ctx, max_batch, HIDDEN, Scalar[dtype](0.7),
                               depth_discounted=True)
        m_c07.sync_from(ctx, critic.online)
        m_c10.sync_from(ctx, critic.online)
        m_dd.sync_from(ctx, critic.online)

        # El control que separa NIVEL de INFORMACION: un "critico" que devuelve la
        # constante c y nada mas. Con w3 = 0 la red ignora la entrada y saca b3.
        # Si este brazo empata con el critico entrenado, entonces lo que aporta el
        # entrenado es solo su media, y la separacion de 0.0072 que se midio en
        # E1.10 no llega a mover ninguna decision.
        m_const = TicTacToeCritic(ctx, max_batch, HIDDEN, Scalar[dtype](0.7),
                                  depth_discounted=True)
        b3 = List[Scalar[dtype]](); b3.append(c)
        write_into[dtype](m_const.params.b3, b3)   # w1,w2,w3 se quedan a cero
        ctx.synchronize()

        # Calentamiento: la primera busqueda de cada tipo paga la compilacion.
        _ = play[TicTacToe](ctx, "warmup", TicTacToe(REWARD_GAMMA),
                            eval_config(REWARD_GAMMA), 3)
        _ = play[TicTacToeCritic](ctx, "warmup", m_c07, eval_config(0.7), 3)

        print("--- 2. seis montajes, las mismas partidas ---")
        a_base = play[TicTacToe](ctx, "sin critico  gamma_r=0.7",
                                 TicTacToe(Scalar[dtype](0.7)),
                                 eval_config(0.7), EVAL_STEPS)
        show(a_base)
        a_g1 = play[TicTacToe](ctx, "sin critico  gamma_r=1.0",
                               TicTacToe(Scalar[dtype](1.0)),
                               eval_config(1.0), EVAL_STEPS)
        show(a_g1)
        a_c07 = play[TicTacToeCritic](ctx, "CON critico  gamma_r=0.7", m_c07,
                                      eval_config(0.7), EVAL_STEPS)
        show(a_c07)
        a_c10 = play[TicTacToeCritic](ctx, "CON critico  gamma_r=1.0", m_c10,
                                      eval_config(1.0), EVAL_STEPS)
        show(a_c10)
        a_dd = play[TicTacToeCritic](ctx, "CON critico  escala coherente", m_dd,
                                     eval_config(0.7), EVAL_STEPS)
        show(a_dd)
        a_const = play[TicTacToeCritic](ctx, "V constante  escala coherente",
                                        m_const, eval_config(0.7), EVAL_STEPS)
        show(a_const)
        print()

        print("--- 3. veredicto (IC de Wilson al 95% sobre las derrotas) ---")
        verdict(a_base, a_c07)
        verdict(a_base, a_c10)
        verdict(a_base, a_g1)
        verdict(a_base, a_dd)
        print()
        print("--- 4. ¿aporta el critico algo mas que su media? ---")
        verdict(a_dd, a_const)
        print()
        print("   referencia exacta: el juego optimo pierde 0.00%;")
        print("   jugar al azar pierde 28.81%.")
