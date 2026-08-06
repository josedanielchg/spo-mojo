"""J1/J2: la pata SPO-Mojo de la comparacion de tres, con el agente ENTRENADO.

    ./run.sh bench/bench_agent_tictactoe.mojo [ruta_csv]

Sustituye a `bench/bench_tictactoe.mojo` como pata de la comparacion. Aquel mide el
**planificador desnudo** (prior uniforme, sin red), que era lo unico que existia en
el Milestone 1; comparar eso contra dos agentes entrenados seria comparar cosas
distintas. Aquel sigue valiendo como el brazo "sin red" del eje de asimetria, y por
eso este banco lo vuelve a medir aqui dentro, con los MISMOS hiperparametros, en vez
de citar su CSV viejo (que ademas se midio con gamma_r = 0.7 y 64 particulas).

**Entrena y mide en la misma corrida.** La implementacion en Mojo no persiste pesos,
asi que no hay checkpoint que cargar: `train_run` devuelve el actor entrenado
precisamente para poder medirlo sin reentrenar. La contrapartida es que el agente
medido es el de ESTA corrida; la semilla esta fijada para que sea reproducible.

**Cuatro filas, no una.** Se cruzan dos ejes que el plan pide separados:

  * protocolo de lectura: `moda` (argmax de q, comparable con el argmax de visitas
    del MCTS y con la fila `moda` de SPO-Stoix) y `muestreada` (el sorteo de q, que
    es lo que hace el evaluador de Stoix);
  * regimen de tiempo: `lote` (64 partidas a la vez: throughput) y `latencia` (una
    sola partida, que es lo unico comparable con el MCTS, que juega en serie).

El esquema de 17 columnas solo tiene un hueco de tiempo, asi que el regimen va en la
etiqueta `mode` en vez de en una columna nueva -- cambiar el esquema romperia la
concatenacion con el CSV del MCTS, que ya esta escrito.
"""

from std.gpu.host import DeviceContext
from std.sys import argv
from std.time import perf_counter_ns

from bench.metrics import PlannerMetrics, write_csv_rows, fmt_fixed
from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform
from envs.tictactoe import (TicTacToe, ttt_reset_kernel, ttt_env_step_kernel,
                            ttt_auto_reset_kernel, NUM_ACTIONS, STATE_DIM,
                            TPB_TTT)
from envs.tictactoe_actor import TicTacToeActor
from envs.tictactoe_runner import RNG_RIVAL
from systems.spo.particles import SearchWorkspace
from systems.spo.readout import readout_expected, readout_greedy, q_histogram
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from demos.train_spo import (train_run, ActorLearner, HIDDEN, SEARCH_DEPTH,
                             TEMPERATURE, NO_RESAMPLE, REWARD_GAMMA,
                             LOSS_PENALTY, NUM_PARTICLES)
from demos.train_critic import Critic

# Pasos de entorno por ronda de entrenamiento: ROLLOUT (16) x NUM_ENVS (32).
comptime STEPS_PER_ROUND = 16 * 32

comptime BENCH_SEED = UInt32(20260805)
comptime BATCH_ENVS = 64
# 360 pasos x 64 envs ~ 7000 partidas: la misma muestra que la pata SPO-Stoix, para
# que los intervalos de Wilson sean comparables entre patas.
comptime BATCH_STEPS = 360
# La fila de latencia existe para el TIEMPO, pero su score se escribe igual en el
# CSV, asi que necesita una muestra que no sea absurda: con 40 pasos salian 13
# partidas y un score de 1.0000, por encima del techo exacto. Con 2000 salen ~650.
comptime LATENCY_STEPS = 2000
comptime RNG_BENCH_READOUT = UInt32(51000)

# Referencias exactas por recursion sobre todos los estados.
comptime RANDOM_SCORE = 0.6484
comptime OPTIMAL_SCORE = 0.9974


@fieldwise_init
struct Timed(Movable):
    """Marcador y tiempo de una tanda."""
    var wins: Int
    var draws: Int
    var losses: Int
    var moves: Int
    var decisions: Int
    var seconds: Float64

    def games(self) -> Int:
        return self.wins + self.draws + self.losses

    def score(self) -> Float64:
        n = self.games()
        d = n if n > 0 else 1
        return (Float64(self.wins) + 0.5 * Float64(self.draws)) / Float64(d)


def play_timed(ctx: DeviceContext, actor: ActorLearner, critic: Critic,
               num_envs: Int, num_steps: Int, particles: Int, use_actor: Bool,
               greedy: Bool, spo_readout: Bool = False) raises -> Timed:
    """Juega contra el rival aleatorio midiendo el tiempo de juego.

    `use_actor` elige el prior de la busqueda: la red entrenada o el uniforme.
    `greedy` elige el protocolo: la moda de q o un sorteo de q.
    `spo_readout` elige QUE q: la del paper (histograma ponderado, ecuacion 6) o
    nuestra variante `readout_expected`, que promedia por accion antes de exponenciar.

    El critico va SIEMPRE conectado. No es una opcion: el peso del E-step de SPO es
    `SUMA_d r_d + gamma^(d+1) V(s_ultimo) - V(raiz)`, asi que sin V las particulas
    que acaban en un tablero no terminal no aportan ninguna estimacion de valor. El
    defecto de `TicTacToeActor` y de `train_run` es `use_critic=False`, que es una
    VARIANTE nuestra y no SPO; heredarlo aqui en silencio dejo toda la comparacion
    del Milestone 4 midiendo un algoritmo distinto del que dice medir.
    """
    cfg = SPOConfig(num_envs=num_envs, num_particles=particles,
                    num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                    search_depth=SEARCH_DEPTH, resample_period=NO_RESAMPLE,
                    temperature=TEMPERATURE, search_gamma=1.0,
                    search_gae_lambda=1.0)
    model = TicTacToe(REWARD_GAMMA, LOSS_PENALTY)
    # `depth_disc` sigue la misma regla que la medicion del critico de E2: con
    # gamma_r < 1 el bootstrap se descuenta por profundidad, gamma_r^(d+1) V.
    depth_disc = REWARD_GAMMA < Scalar[dtype](1)
    amodel = TicTacToeActor(ctx, cfg.num_search_particles(), HIDDEN,
                            REWARD_GAMMA, LOSS_PENALTY, True, depth_disc)
    amodel.sync_from(ctx, actor.net.params)
    amodel.sync_critic_from(ctx, critic.online)
    ws = SearchWorkspace(ctx, cfg)
    blocks = (num_envs + TPB_TTT - 1) // TPB_TTT

    state = zero_buffer[dtype](ctx, num_envs * STATE_DIM)
    reward = zero_buffer[dtype](ctx, num_envs)
    done = zero_buffer[idx_dtype](ctx, num_envs)
    u_rival = zero_buffer[dtype](ctx, num_envs)
    u_readout = zero_buffer[dtype](ctx, num_envs)
    q_buf = zero_buffer[dtype](ctx, num_envs * NUM_ACTIONS)
    logits_buf = zero_buffer[dtype](ctx, num_envs * NUM_ACTIONS)

    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), num_envs, grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()          # el reset no cuenta como tiempo de juego

    wins = 0
    draws = 0
    losses = 0
    moves = 0

    start = perf_counter_ns()
    for step in range(num_steps):
        sd = BENCH_SEED ^ (UInt32(step) * 2654435761)
        if use_actor:
            search[TicTacToeActor](ctx, ws, cfg, amodel, state, sd)
        else:
            search[TicTacToe](ctx, ws, cfg, model, state, sd)

        # El sorteo necesita numeros frescos; la moda los ignora, pero se rellenan
        # igualmente para que los dos brazos hagan el MISMO trabajo y el tiempo sea
        # comparable entre protocolos.
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_readout.unsafe_ptr(), BENCH_SEED,
            RNG_BENCH_READOUT + UInt32(step), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
        if spo_readout:
            if greedy:
                # La moda del histograma ponderado. Solo pisa `output.action`.
                readout_greedy(ctx, ws.particles, ws.output, cfg, q_buf)
            else:
                # El sorteo de q ya lo hizo `search` (readout_weighted); aqui solo
                # se materializa q. Pisar la accion seria sortear dos veces.
                q_histogram(ctx, ws.particles, ws.output, cfg, q_buf)
        else:
            readout_expected(ctx, ws.particles, ws.output, cfg, logits_buf, q_buf,
                             u_readout, greedy)

        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), BENCH_SEED, RNG_RIVAL + UInt32(step),
            num_envs, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), ws.output.action.unsafe_ptr(),
            u_rival.unsafe_ptr(), reward.unsafe_ptr(), done.unsafe_ptr(),
            num_envs, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

        with reward.map_to_host() as rh:
            with done.map_to_host() as dh:
                for e in range(num_envs):
                    moves += 1
                    if Int(dh[e]) != 0:
                        r = rh[e]
                        if r > Scalar[dtype](0.75): wins += 1
                        elif r > Scalar[dtype](0.25): draws += 1
                        else: losses += 1

        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()
    seconds = Float64(perf_counter_ns() - start) / 1e9

    # Una decision por env y por turno: la busqueda decide para todos a la vez.
    return Timed(wins, draws, losses, moves, num_envs * num_steps, seconds)


def metrics_of(mode: String, t: Timed, particles: Int) -> PlannerMetrics:
    return PlannerMetrics(
        mode=mode, games=t.games(), iterations=particles,
        exploration=Float64(TEMPERATURE), seed=Int(BENCH_SEED),
        total_runtime_s=t.seconds, total_moves=t.moves,
        decisions=t.decisions,
        total_simulations=t.decisions * particles * SEARCH_DEPTH,
        x_wins=t.wins, o_wins=t.losses, draws=t.draws)


def check_ceiling(mode: String, t: Timed) raises:
    """Un score por encima del techo exacto no es un buen resultado: es un bug.

    Es la guarda que descubrio la fuga de RNG en SPO-Stoix (0.9997 contra un techo
    de 0.9974). La sigma es la EMPIRICA y no la cota del peor caso: cerca del techo
    casi todo son victorias, la varianza real cae unas mil veces, y con la cota la
    guarda se traga cualquier anomalia sin protestar.
    """
    n = t.games()
    if n == 0:
        raise Error("la tanda `" + mode + "` no completo ninguna partida")
    s = t.score()
    e_sq = (Float64(t.wins) + 0.25 * Float64(t.draws)) / Float64(n)
    variance = e_sq - s * s
    if variance < 0.0:
        variance = 0.0
    se = (variance / Float64(n)) ** 0.5
    if s > OPTIMAL_SCORE + 4.0 * (se if se > 1e-12 else 1e-12):
        raise Error("`" + mode + "` puntua " + fmt_fixed(s, 4)
                    + ", por encima del techo exacto "
                    + fmt_fixed(OPTIMAL_SCORE, 4)
                    + ". Es IMPOSIBLE jugando limpio: hay una fuga de informacion.")


def pad(s: String, width: Int, left: Bool = False) -> String:
    """Rellena a `width`. Mojo no trae ljust/rjust en String."""
    out = s
    while len(out) < width:
        out = (String(" ") + out) if left else (out + String(" "))
    return out


def show(mode: String, t: Timed) raises:
    n = t.games()
    d = Float64(n if n > 0 else 1)
    print("  " + pad(mode, 30)
          + " n=" + pad(String(n), 5, True)
          + "  gana " + fmt_fixed(100.0 * Float64(t.wins) / d, 2) + "%"
          + "  empata " + fmt_fixed(100.0 * Float64(t.draws) / d, 2) + "%"
          + "  pierde " + fmt_fixed(100.0 * Float64(t.losses) / d, 2) + "%"
          + "  score " + fmt_fixed(t.score(), 4))


def main() raises:
    args = argv()
    path = String(args[1]) if len(args) > 1 else String("results/bench_spo_mojo.csv")
    # Presupuesto de entrenamiento en RONDAS. Por defecto el del entrenador; se pasa
    # por linea de ordenes para poder igualar los 600.000 pasos de entorno con los
    # que se entreno la pata SPO-Stoix (1172 rondas x 512 pasos).
    rounds = Int(args[2]) if len(args) > 2 else 30
    # "fiel" = el readout del paper (histograma ponderado). Cualquier otra cosa =
    # nuestra variante. Se entrena UN ACTOR POR READOUT: un actor destilado de la q
    # de un readout no es el prior adecuado del otro.
    spo_readout = (String(args[3]) == "fiel") if len(args) > 3 else False

    with DeviceContext() as ctx:
        print("SPO-Mojo: agente ENTRENADO sobre tres en raya (rival aleatorio)")
        print("  particulas", NUM_PARTICLES, " profundidad", SEARCH_DEPTH,
              " temperatura", TEMPERATURE, " gamma_r", REWARD_GAMMA)
        print("  referencias exactas: azar", RANDOM_SCORE,
              " optimo", OPTIMAL_SCORE)
        print()

        print("=== entrenando (la implementacion en Mojo no persiste pesos) ===")
        print("  rondas", rounds, " = ", rounds * STEPS_PER_ROUND,
              "pasos de entorno")
        train_start = perf_counter_ns()
        print("  readout:", "fiel (histograma ponderado)" if spo_readout
              else "variante (media por accion)")
        outcome = train_run(ctx, "spo-mojo", use_actor=True,
                            spo_readout=spo_readout, use_critic=True,
                            depth_disc=REWARD_GAMMA < Scalar[dtype](1),
                            rounds=rounds)
        train_seconds = Float64(perf_counter_ns() - train_start) / 1e9
        print("  tiempo de entrenamiento: " + fmt_fixed(train_seconds, 1)
              + " s  (se paga UNA vez; el MCTS no lo paga, pero paga busqueda en"
              + " cada jugada)")
        print()

        # Calentamiento: la primera llamada paga la compilacion de los kernels y ese
        # coste no es parte de lo que se quiere medir.
        _ = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS, 3, NUM_PARTICLES,
                       True, True)

        rows = List[PlannerMetrics]()

        print("=== lote de", BATCH_ENVS, "partidas (throughput) ===")
        moda = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS, BATCH_STEPS,
                          NUM_PARTICLES, True, True, spo_readout)
        show("smc_agent_moda_lote", moda)
        check_ceiling("smc_agent_moda_lote", moda)
        rows.append(metrics_of("smc_agent_moda_lote", moda, NUM_PARTICLES))

        muestreada = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS, BATCH_STEPS,
                                NUM_PARTICLES, True, False, spo_readout)
        show("smc_agent_muestreada_lote", muestreada)
        check_ceiling("smc_agent_muestreada_lote", muestreada)
        rows.append(metrics_of("smc_agent_muestreada_lote", muestreada,
                               NUM_PARTICLES))

        # El brazo SIN red, con los mismos hiperparametros: es el termino de
        # comparacion del eje de asimetria (pagar entrenamiento una vez frente a
        # pagar busqueda en cada jugada).
        sin_red = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS, BATCH_STEPS,
                             NUM_PARTICLES, False, True, spo_readout)
        show("smc_planificador_moda_lote", sin_red)
        check_ceiling("smc_planificador_moda_lote", sin_red)
        rows.append(metrics_of("smc_planificador_moda_lote", sin_red,
                               NUM_PARTICLES))

        print()
        print("=== una partida a la vez (latencia, comparable con el MCTS) ===")
        lat = play_timed(ctx, outcome.actor, outcome.critic, 1, LATENCY_STEPS, NUM_PARTICLES,
                         True, True)
        latency = lat.seconds / Float64(lat.decisions)
        show("smc_agent_moda_latencia", lat)
        check_ceiling("smc_agent_moda_latencia", lat)
        rows.append(metrics_of("smc_agent_moda_latencia", lat, NUM_PARTICLES))

        batched_cost = moda.seconds / Float64(moda.decisions)
        print("  latencia        : " + fmt_fixed(latency, 6) + " s/decision")
        print("  coste repartido : " + fmt_fixed(batched_cost, 6) + " s/decision")
        print("  ganancia de lote: x" + fmt_fixed(latency / batched_cost, 1)
              + "  (" + String(BATCH_ENVS) + " partidas a la vez)")

        # --- El eje que de verdad discrimina: presupuesto de busqueda ---
        #
        # A 128 particulas el score SATURA (el agente y el planificador sin red dan
        # lo mismo), asi que comparar fuerza a presupuesto maximo no distingue nada.
        # Lo que el prior aprendido deberia comprar es llegar al mismo nivel con
        # MENOS busqueda, que es el argumento practico de SPO frente al MCTS: el
        # entrenamiento se paga una vez y la busqueda se paga en cada jugada.
        print()
        print("=== curva de presupuesto (misma red, distinto numero de particulas) ===")
        print("  particulas   con red                        sin red")
        budgets = List[Int]()
        budgets.append(4)
        budgets.append(8)
        budgets.append(16)
        budgets.append(32)
        budgets.append(64)
        budgets.append(128)
        for b in budgets:
            con = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS, BATCH_STEPS, b,
                             True, True, spo_readout)
            sin = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS, BATCH_STEPS, b,
                             False, True, spo_readout)
            check_ceiling("presupuesto_con_red_" + String(b), con)
            check_ceiling("presupuesto_sin_red_" + String(b), sin)
            print("  " + pad(String(b), 10, True)
                  + "   score " + fmt_fixed(con.score(), 4)
                  + " pierde " + fmt_fixed(100.0 * Float64(con.losses)
                                           / Float64(con.games()), 2) + "%"
                  + "      score " + fmt_fixed(sin.score(), 4)
                  + " pierde " + fmt_fixed(100.0 * Float64(sin.losses)
                                           / Float64(sin.games()), 2) + "%")
            rows.append(metrics_of("presupuesto_con_red_" + String(b), con, b))
            rows.append(metrics_of("presupuesto_sin_red_" + String(b), sin, b))

        write_csv_rows(rows, path)
        print()
        print("csv escrito en", path)
