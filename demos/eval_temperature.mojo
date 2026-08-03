"""La temperatura: nuestro 0.02 contra el 0.5 de Stoix, con el sistema fiel.

Ultimo apano sin re-medir. `temperature = 0.02` lo elegimos en el barrido de M1
(0.5 -> 0.784 ... 0.02 -> 0.9739), o sea **cuando no habia ni actor ni critico**.
Stoix usa `temperature.adaptive: True` por defecto y su valor fijo es **0.5**.

Es el mismo patron que `reward_gamma`: un hiperparametro afinado para compensar
componentes que faltaban. Y ya sabemos como acaba ese patron -- gamma_r = 0.7 daba
2.65% de derrotas con el actor conectado y gamma_r = 1.0 (el fiel) da 0.62%. Asi que
hay que comprobar si con el actor y el critico puestos el 0.5 de la referencia va
igual o mejor: seria una desviacion menos que justificar.

La temperatura entra en DOS sitios, asi que hay que entrenar con cada valor y no
solo medir: el remuestreo (`resample_logits_kernel`) y el readout. O sea que cambia
la q que el actor aprende, no solo como se juega.

Montaje: SPO literal (gamma_r = 1.0, sin castigo, con remuestreo, readout de SPO),
prior del actor, critico conectado, ruido de Dirichlet a 0.25. Cero desviaciones.
"""

from std.gpu.host import DeviceContext

from bench.metrics import fmt_fixed, wilson_lo, wilson_hi
from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform
from envs.tictactoe import (TicTacToe, ttt_reset_kernel, ttt_env_step_kernel,
                            ttt_auto_reset_kernel, NUM_ACTIONS, STATE_DIM,
                            TPB_TTT)
from envs.tictactoe_actor import TicTacToeActor
from envs.tictactoe_runner import RNG_RIVAL
from systems.spo.particles import SearchWorkspace
from systems.spo.readout import readout_greedy
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from demos.train_spo import train_run, ActorLearner, HIDDEN, SEARCH_DEPTH
from demos.train_critic import Critic

comptime EVAL_ENVS = 128
comptime EVAL_STEPS = 300
comptime EVAL_SEED = UInt32(20260805)
comptime PARTICLES = 128
comptime PERIOD = 3
comptime GAMMA_R = Scalar[dtype](1.0)
comptime PENALTY = Scalar[dtype](0.0)
comptime NOISE = Scalar[dtype](0.25)


def pct(x: Float64) -> String:
    return fmt_fixed(x * 100.0, 2) + "%"


def play(ctx: DeviceContext, actor: ActorLearner, critic: Critic,
         temp: Scalar[dtype], steps: Int,
         greedy: Bool = True) raises -> List[Int]:
    """Juega y devuelve [gana, empata, pierde].

    `greedy = False` reproduce el protocolo de evaluacion de Stoix: su evaluador
    usa el MISMO root_fn y juega `search_output.action`, que es la accion
    MUESTREADA de q (`evaluator.py:55-57`). Jugar la moda es una desviacion
    nuestra del protocolo de medida, y con la temperatura importa mucho: al
    muestrear, tau controla directamente cuan aleatoria sale la jugada."""
    cfg = SPOConfig(num_envs=EVAL_ENVS, num_particles=PARTICLES,
                    num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                    search_depth=SEARCH_DEPTH, resample_period=PERIOD,
                    temperature=temp, search_gamma=1.0, search_gae_lambda=1.0,
                    dirichlet_alpha=1.0, dirichlet_fraction=NOISE)
    amodel = TicTacToeActor(ctx, cfg.num_search_particles(), HIDDEN, GAMMA_R,
                            PENALTY, True, False)
    amodel.sync_from(ctx, actor.net.params)
    amodel.sync_critic_from(ctx, critic.online)
    ws = SearchWorkspace(ctx, cfg)
    q_buf = zero_buffer[dtype](ctx, EVAL_ENVS * NUM_ACTIONS)

    blocks = (EVAL_ENVS + TPB_TTT - 1) // TPB_TTT
    state = zero_buffer[dtype](ctx, EVAL_ENVS * STATE_DIM)
    reward = zero_buffer[dtype](ctx, EVAL_ENVS)
    done = zero_buffer[idx_dtype](ctx, EVAL_ENVS)
    u_rival = zero_buffer[dtype](ctx, EVAL_ENVS)
    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()

    w = 0; d = 0; l = 0
    for step in range(steps):
        search[TicTacToeActor](ctx, ws, cfg, amodel, state,
                               EVAL_SEED ^ (UInt32(step) * 2654435761))
        # Sin greedy no se toca `ws.output.action`: `search` ya la dejo
        # muestreada de q, que es lo que juega el evaluador de Stoix.
        if greedy:
            readout_greedy(ctx, ws.particles, ws.output, cfg, q_buf)
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), EVAL_SEED, RNG_RIVAL + UInt32(step),
            EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
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
                        if r > Scalar[dtype](0.75): w += 1
                        elif r > Scalar[dtype](0.25): d += 1
                        else: l += 1
        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()
    out = List[Int](); out.append(w); out.append(d); out.append(l)
    return out^


def play_g(ctx: DeviceContext, actor: ActorLearner, critic: Critic,
           temp: Scalar[dtype], gamma_r: Scalar[dtype], steps: Int,
           greedy: Bool) raises -> List[Int]:
    """Como `play` pero con gamma_r variable, para el barrido de abajo."""
    depth_disc = gamma_r < Scalar[dtype](1)
    cfg = SPOConfig(num_envs=EVAL_ENVS, num_particles=PARTICLES,
                    num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                    search_depth=SEARCH_DEPTH, resample_period=PERIOD,
                    temperature=temp, search_gamma=1.0, search_gae_lambda=1.0,
                    dirichlet_alpha=1.0, dirichlet_fraction=NOISE)
    amodel = TicTacToeActor(ctx, cfg.num_search_particles(), HIDDEN, gamma_r,
                            PENALTY, True, depth_disc)
    amodel.sync_from(ctx, actor.net.params)
    amodel.sync_critic_from(ctx, critic.online)
    ws = SearchWorkspace(ctx, cfg)
    q_buf = zero_buffer[dtype](ctx, EVAL_ENVS * NUM_ACTIONS)
    blocks = (EVAL_ENVS + TPB_TTT - 1) // TPB_TTT
    state = zero_buffer[dtype](ctx, EVAL_ENVS * STATE_DIM)
    reward = zero_buffer[dtype](ctx, EVAL_ENVS)
    done = zero_buffer[idx_dtype](ctx, EVAL_ENVS)
    u_rival = zero_buffer[dtype](ctx, EVAL_ENVS)
    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()
    w = 0; d = 0; l = 0
    for step in range(steps):
        search[TicTacToeActor](ctx, ws, cfg, amodel, state,
                               EVAL_SEED ^ (UInt32(step) * 2654435761))
        if greedy:
            readout_greedy(ctx, ws.particles, ws.output, cfg, q_buf)
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), EVAL_SEED, RNG_RIVAL + UInt32(step),
            EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
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
                        if r > Scalar[dtype](0.75): w += 1
                        elif r > Scalar[dtype](0.25): d += 1
                        else: l += 1
        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()
    out = List[Int](); out.append(w); out.append(d); out.append(l)
    return out^


def main() raises:
    with DeviceContext() as ctx:
        print("=== La temperatura: nuestro 0.02 contra el 0.5 de Stoix ===")
        print("   SPO literal + actor + critico + ruido de Dirichlet 0.25")
        print("   Se ENTRENA con cada temperatura, porque cambia la q que el")
        print("   actor aprende, no solo como se juega.")
        print("   referencias exactas: azar 0.6484 | optimo 0.9974 (pierde 0.00%)")
        print()

        temps = List[Float64]()
        temps.append(0.02)      # el nuestro, afinado en M1 sin redes
        temps.append(0.1)
        temps.append(0.5)       # el `fixed_temperature` de Stoix
        names = List[String]()
        names.append(String("0.02  (nuestro, M1)"))
        names.append(String("0.10               "))
        names.append(String("0.50  (Stoix)      "))

        rows = List[String]()
        for i in range(len(temps)):
            t = Scalar[dtype](temps[i])
            print("--- entrenando con temperatura", t, "---")
            r = train_run(ctx, names[i], True, True, PERIOD, GAMMA_R, PENALTY,
                          PARTICLES, True, False, NOISE, t)
            for g in range(2):
                greedy = g == 0
                res = play(ctx, r.actor, r.critic, t, EVAL_STEPS, greedy)
                n = res[0] + res[1] + res[2]
                score = (Float64(res[0]) + 0.5 * Float64(res[1])) / Float64(n)
                tag = String("moda      ") if greedy \
                      else String("MUESTREADA")
                line = String("   tau=", names[i], " ", tag, " n=", n,
                              "  gana ", pct(Float64(res[0]) / Float64(n)),
                              "  empata ", pct(Float64(res[1]) / Float64(n)),
                              "  PIERDE ", pct(Float64(res[2]) / Float64(n)),
                              " IC[", fmt_fixed(wilson_lo(res[2], n) * 100.0, 2),
                              ",", fmt_fixed(wilson_hi(res[2], n) * 100.0, 2),
                              "]  score ", fmt_fixed(score, 4),
                              "  KL ", r.result.kl_last)
                rows.append(line)
                print(line)
            print()

        print("=== resumen ===")
        for i in range(len(rows)):
            print(rows[i])
        print()
        print("=== gamma_r bajo LOS DOS protocolos ===")
        print("   El veredicto de la auditoria (gamma_r=1.0 mejor que 0.7) se")
        print("   midio SOLO con la moda. Con juego muestreado la cosa puede")
        print("   cambiar: gamma_r rompe empates entre acciones que ganan, y esos")
        print("   empates solo importan si se sortea.")
        print()
        gs = List[Float64]()
        gs.append(1.0); gs.append(0.7)
        for gi in range(len(gs)):
            g = Scalar[dtype](gs[gi])
            rr = train_run(ctx, String("gamma_r=", g), True, True, PERIOD, g,
                           PENALTY, PARTICLES, True, g < Scalar[dtype](1),
                           NOISE, Scalar[dtype](0.02))
            for gg in range(2):
                greedy = gg == 0
                res = play_g(ctx, rr.actor, rr.critic, Scalar[dtype](0.02), g,
                             EVAL_STEPS, greedy)
                n = res[0] + res[1] + res[2]
                sc = (Float64(res[0]) + 0.5 * Float64(res[1])) / Float64(n)
                print("   gamma_r=", g, " ",
                      "moda      " if greedy else "MUESTREADA",
                      " pierde ", pct(Float64(res[2]) / Float64(n)),
                      " IC[", fmt_fixed(wilson_lo(res[2], n) * 100.0, 2), ",",
                      fmt_fixed(wilson_hi(res[2], n) * 100.0, 2), "]",
                      " score ", fmt_fixed(sc, 4))
        print()
        print("   La fila MUESTREADA es el protocolo de Stoix (evaluator.py:57).")
        print("   La fila moda es una desviacion NUESTRA del protocolo de medida,")
        print("   que hasta ahora estabamos usando sin haberlo verificado.")
