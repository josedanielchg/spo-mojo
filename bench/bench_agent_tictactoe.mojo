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
from envs.tictactoe import (TicTacToe, ttt_reset_alt_kernel, ttt_env_step_kernel,
                            ttt_auto_reset_alt_kernel, ttt_seat_opens_first,
                            NUM_ACTIONS, STATE_DIM, TPB_TTT)
from envs.tictactoe_actor import TicTacToeActor
from envs.tictactoe_runner import RNG_RIVAL
from systems.spo.particles import SearchWorkspace
from systems.spo.readout import readout_expected, readout_greedy, q_histogram
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from demos.train_spo import (train_run, ActorLearner, HIDDEN, SEARCH_DEPTH,
                             TEMPERATURE, NO_RESAMPLE, REWARD_GAMMA,
                             LOSS_PENALTY, NUM_PARTICLES, SEED as TRAIN_SEED)
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
comptime RNG_BENCH_READOUT = UInt32(4_000_000)
comptime RNG_OPEN = UInt32(5_000_000)

# Plafonds exacts, par siege, calcules par recursion sur toutes les positions.
# Ils different, et c'est le point : en ouvrant, maximiser le score et ne jamais
# perdre sont la MEME politique ; en repondant, ce sont deux politiques
# differentes, et la premiere accepte 0,42 % de defaites.
comptime OPT_PREMIER = 0.9974
comptime OPT_SECOND = 0.9624
comptime OPT_MOYENNE = 0.9799
comptime HASARD_MOYENNE = 0.5000

# Referencias exactas por recursion sobre todos los estados, jugando primero.
comptime RANDOM_SCORE = 0.6484
comptime OPTIMAL_SCORE = 0.9974


@fieldwise_init
struct Timed(Movable):
    """Marcador et temps d'une campagne, SEPARES PAR SIEGE.

    L'indice 0 designe les parties ou l'agent ouvre, l'indice 1 celles ou
    l'adversaire ouvre. Les deux ne se melangent jamais dans un compteur commun,
    pour une raison qui n'a rien d'esthetique : a duree de campagne egale, le
    siege ou l'adversaire ouvre produit PLUS de parties, puisqu'il consomme une
    case et raccourcit le jeu. Mesure : 12 233 parties contre 14 801. Agreger les
    deux reviendrait donc a ponderer la moyenne vers ce siege --- ce qui donne
    0,4845 la ou la valeur exacte du jeu au hasard vaut 0,5000. Le biais n'est pas
    du bruit, aucune barre d'erreur ne le signalerait.

    La moyenne se calcule donc comme la moyenne NON PONDEREE des deux scores.
    """
    var wins: InlineArray[Int, 2]
    var draws: InlineArray[Int, 2]
    var losses: InlineArray[Int, 2]
    var envs: InlineArray[Int, 2]
    var moves: Int
    var decisions: Int
    var seconds: Float64

    def part_envs(self, k: Int) -> Float64:
        """Part des environnements tenant le siege k.

        C'est ce ratio, et non celui des parties, qui repartit le temps, les
        decisions et les coups. Chaque environnement decide une fois par pas, quel
        que soit son siege : le partage y est donc exactement celui des
        environnements. Le partage des PARTIES, lui, est different --- 3 251 contre
        3 600 sur une meme campagne --- parce que les parties ou l'adversaire ouvre
        sont plus courtes et s'en termine davantage.
        """
        total = self.envs[0] + self.envs[1]
        return Float64(self.envs[k]) / Float64(total if total > 0 else 1)

    def games(self, k: Int) -> Int:
        return self.wins[k] + self.draws[k] + self.losses[k]

    def games_total(self) -> Int:
        return self.games(0) + self.games(1)

    def score(self, k: Int) -> Float64:
        n = self.games(k)
        d = n if n > 0 else 1
        return (Float64(self.wins[k]) + 0.5 * Float64(self.draws[k])) / Float64(d)

    def score_moyen(self) -> Float64:
        """Moyenne NON PONDEREE des deux sieges. Voir la note de la structure."""
        return 0.5 * (self.score(0) + self.score(1))

    def perte(self, k: Int) -> Float64:
        n = self.games(k)
        return 100.0 * Float64(self.losses[k]) / Float64(n if n > 0 else 1)

    def perte_moyenne(self) -> Float64:
        return 0.5 * (self.perte(0) + self.perte(1))


def play_timed(ctx: DeviceContext, actor: ActorLearner, critic: Critic,
               num_envs: Int, num_steps: Int, particles: Int, use_actor: Bool,
               greedy: Bool, spo_readout: Bool = False,
               gamma_r: Scalar[dtype] = REWARD_GAMMA,
               penalty: Scalar[dtype] = LOSS_PENALTY,
               depth: Int = SEARCH_DEPTH,
               temp: Scalar[dtype] = TEMPERATURE,
               period: Int = NO_RESAMPLE) raises -> Timed:
    """Juega contra el rival aleatorio midiendo el tiempo de juego.

    `use_actor` elige el prior de la busqueda: la red entrenada o el uniforme.
    `greedy` elige el protocolo: la moda de q o un sorteo de q.
    `spo_readout` elige QUE q: la del paper (histograma ponderado, ecuacion 6) o
    nuestra variante `readout_expected`, que promedia por accion antes de exponenciar.
    `gamma_r` es el descuento por profundidad de la recompensa, que Stoix NO tiene
    (pasa la recompensa cruda). Es parametro para poder falsar la hipotesis de la
    escala de pesos: con gamma_r < 1 las particulas ganadoras dejan de empatar entre
    si, y como `exp(w/tau)` con tau pequeno se comporta como un maximo, la q del
    readout fiel pasa a medir "alguna gano MUY pronto" en vez de "cuantas ganan".
    `penalty` pone la derrota en -penalty en vez de en 0. Stoix no lo tiene: alli lo
    que distingue a una particula muerta de una viva es que su bootstrap vale 0.
    `depth`, `temp` y `period` son los tres mandos de busqueda que Stoix tambien
    expone (`search_depth`, `temperature.fixed_temperature`, `resampling.period`).
    Estan parametrizados para poder barrer los MISMOS ejes en las dos
    implementaciones y comparar la FORMA de la respuesta, que es mejor evidencia de
    equivalencia que coincidir en un solo punto.

    El critico va SIEMPRE conectado. No es una opcion: el peso del E-step de SPO es
    `SUMA_d r_d + gamma^(d+1) V(s_ultimo) - V(raiz)`, asi que sin V las particulas
    que acaban en un tablero no terminal no aportan ninguna estimacion de valor. El
    defecto de `TicTacToeActor` y de `train_run` es `use_critic=False`, que es una
    VARIANTE nuestra y no SPO; heredarlo aqui en silencio dejo toda la comparacion
    del Milestone 4 midiendo un algoritmo distinto del que dice medir.
    """
    cfg = SPOConfig(num_envs=num_envs, num_particles=particles,
                    num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                    search_depth=depth, resample_period=period,
                    temperature=temp, search_gamma=1.0,
                    search_gae_lambda=1.0)
    model = TicTacToe(gamma_r, penalty)
    # `depth_disc` sigue la misma regla que la medicion del critico de E2: con
    # gamma_r < 1 el bootstrap se descuenta por profundidad, gamma_r^(d+1) V.
    depth_disc = gamma_r < Scalar[dtype](1)
    amodel = TicTacToeActor(ctx, cfg.num_search_particles(), HIDDEN,
                            gamma_r, penalty, True, depth_disc)
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

    u_open = zero_buffer[dtype](ctx, num_envs)
    ctx.enqueue_function[fill_uniform, fill_uniform](
        u_open.unsafe_ptr(), BENCH_SEED, RNG_OPEN, num_envs,
        grid_dim=blocks, block_dim=TPB_TTT)
    ctx.enqueue_function[ttt_reset_alt_kernel, ttt_reset_alt_kernel](
        state.unsafe_ptr(), u_open.unsafe_ptr(), num_envs,
        grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()          # el reset no cuenta como tiempo de juego

    var wins = InlineArray[Int, 2](fill=0)
    var draws = InlineArray[Int, 2](fill=0)
    var losses = InlineArray[Int, 2](fill=0)
    var envs = InlineArray[Int, 2](fill=0)
    for e in range(num_envs):
        envs[0 if ttt_seat_opens_first(e) else 1] += 1
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
                        # Le siege se lit dans la parite de l'indice : c'est tout
                        # l'interet du partage stratifie.
                        k = 0 if ttt_seat_opens_first(e) else 1
                        r = rh[e]
                        if r > Scalar[dtype](0.75): wins[k] += 1
                        elif r > Scalar[dtype](0.25): draws[k] += 1
                        else: losses[k] += 1

        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_open.unsafe_ptr(), BENCH_SEED, RNG_OPEN + UInt32(step) + 1,
            num_envs, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_auto_reset_alt_kernel, ttt_auto_reset_alt_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), u_open.unsafe_ptr(), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()
    seconds = Float64(perf_counter_ns() - start) / 1e9

    # Una decision por env y por turno: la busqueda decide para todos a la vez.
    return Timed(wins, draws, losses, envs, moves, num_envs * num_steps, seconds)


def metrics_of(mode: String, t: Timed, particles: Int, k: Int) -> PlannerMetrics:
    """Une ligne de CSV pour UN siege. Jamais les deux agreges.

    Le temps, les decisions et les coups sont repartis au prorata des
    ENVIRONNEMENTS du siege, pas de ses parties : chaque environnement decide une
    fois par pas quel que soit son siege. Les deux sieges tournent dans la meme
    campagne et ne peuvent pas etre chronometres separement, mais ce partage-la
    est exact, et les grandeurs derivees (temps par decision, simulations par
    seconde) n'en dependent meme pas puisque le facteur s'y simplifie.
    """
    part = t.part_envs(k)
    return PlannerMetrics(
        mode=mode, games=t.games(k), iterations=particles,
        exploration=Float64(TEMPERATURE), seed=Int(BENCH_SEED),
        total_runtime_s=t.seconds * part,
        total_moves=Int(Float64(t.moves) * part),
        decisions=Int(Float64(t.decisions) * part),
        total_simulations=Int(Float64(t.decisions) * part) * particles * SEARCH_DEPTH,
        x_wins=t.wins[k], o_wins=t.losses[k], draws=t.draws[k])


def deux_lignes(mut rows: List[PlannerMetrics], mode: String, t: Timed,
                particles: Int) -> None:
    """Ajoute les lignes d'une campagne, une par siege.

    Un siege sans partie est saute plutot qu'ecrit a zero. Le cas se produit pour
    la mesure de latence, qui tourne sur UN environnement : le partage etant fait
    sur la parite de l'indice, un environnement unique ne couvre qu'un siege. Cette
    ligne-la existe pour le TEMPS, pas pour le score, donc la limitation est sans
    consequence --- mais elle doit etre visible dans le CSV plutot que masquee par
    une ligne de zeros qui ressemblerait a une defaite totale.
    """
    if t.games(0) > 0:
        rows.append(metrics_of(mode + "_1er", t, particles, 0))
    if t.games(1) > 0:
        rows.append(metrics_of(mode + "_2e", t, particles, 1))


def check_ceiling(mode: String, t: Timed) raises:
    """Un score au-dessus du plafond exact n'est pas un bon resultat : c'est un bug.

    C'est la garde qui a revele la fuite de generateur dans SPO-Stoix (0,9997
    contre un plafond de 0,9974). Le sigma est EMPIRIQUE et non une borne du pire
    cas : pres du plafond presque tout est victoire, la variance reelle chute d'un
    facteur mille, et la borne prudente rendrait la garde aveugle.

    Chaque siege a SON plafond. En ouvrant 0,9974, en repondant 0,9624 : les
    confondre laisserait passer un score impossible du cote du second.
    """
    for k in range(2):
        n = t.games(k)
        if n == 0:
            # Siege non couvert : c'est le cas normal a un seul environnement.
            # Rien a verifier, et surtout rien a signaler comme anomalie.
            continue
        s = t.score(k)
        e_sq = (Float64(t.wins[k]) + 0.25 * Float64(t.draws[k])) / Float64(n)
        variance = e_sq - s * s
        if variance < 0.0:
            variance = 0.0
        se = (variance / Float64(n)) ** 0.5
        plafond = OPT_PREMIER if k == 0 else OPT_SECOND
        if s > plafond + 4.0 * (se if se > 1e-12 else 1e-12):
            raise Error("`" + mode + "` obtient " + fmt_fixed(s, 4)
                        + " au siege " + String(k) + ", au-dessus du plafond exact "
                        + fmt_fixed(plafond, 4)
                        + ". Impossible en jouant proprement : il y a une fuite "
                        + "d'information.")


def pad(s: String, width: Int, left: Bool = False) -> String:
    """Remplit a `width`. Mojo n'a ni ljust ni rjust sur String."""
    out = s
    while len(out) < width:
        out = (String(" ") + out) if left else (out + String(" "))
    return out


def show(mode: String, t: Timed) raises:
    """Affiche les deux sieges et leur moyenne non ponderee."""
    if t.games(0) == 0 or t.games(1) == 0:
        k = 0 if t.games(0) > 0 else 1
        nom = String("1er") if k == 0 else String("2e")
        print("  " + pad(mode, 28) + " " + nom + " seulement  "
              + fmt_fixed(t.score(k), 4) + "/" + fmt_fixed(t.perte(k), 2) + "%"
              + "   n=" + String(t.games(k)))
        return
    print("  " + pad(mode, 28)
          + " 1er " + fmt_fixed(t.score(0), 4) + "/" + fmt_fixed(t.perte(0), 2) + "%"
          + "   2e " + fmt_fixed(t.score(1), 4) + "/" + fmt_fixed(t.perte(1), 2) + "%"
          + "   moy " + fmt_fixed(t.score_moyen(), 4)
          + "/" + fmt_fixed(t.perte_moyenne(), 2) + "%"
          + "   n=" + String(t.games_total()))


def two(con: Timed, sin_red: Timed) raises -> String:
    """Les deux colonnes d'un balayage : avec et sans reseau, moyenne des sieges."""
    return ("   avec reseau  " + fmt_fixed(con.score_moyen(), 4)
            + " / " + fmt_fixed(con.perte_moyenne(), 2) + "%"
            + "     sans reseau  " + fmt_fixed(sin_red.score_moyen(), 4)
            + " / " + fmt_fixed(sin_red.perte_moyenne(), 2) + "%")


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
    # Particulas con las que se MIDE. El entrenamiento sigue con NUM_PARTICLES: subir
    # la busqueda solo en evaluacion es legitimo y es lo que se compara con el MCTS,
    # que tambien elige sus simulaciones por jugada en el momento de jugar.
    n_part = Int(args[4]) if len(args) > 4 else NUM_PARTICLES
    # Descuento por profundidad de la recompensa. 1.0 = lo que hace Stoix.
    gamma_r = Scalar[dtype](Float64(String(args[5]))) if len(args) > 5 \
              else REWARD_GAMMA
    # Castigo por derrota. 0.0 = lo que hace Stoix.
    penalty = Scalar[dtype](Float64(String(args[6]))) if len(args) > 6 \
              else LOSS_PENALTY
    # Semilla de ENTRENAMIENTO. Se pasa por linea de ordenes para poder repetir la
    # misma configuracion con varias semillas y reportar la desviacion ENTRE ellas,
    # que es la incertidumbre que el intervalo de Wilson no captura.
    train_seed = UInt32(Int(String(args[7]))) if len(args) > 7 else TRAIN_SEED
    # "cabecera" mide solo las filas de la comparacion y se salta los barridos.
    # Es lo que hace falta para repetir la MISMA configuracion con varias semillas:
    # los barridos no cambian la conclusion y multiplicarian el coste por veinte.
    solo_cabecera = (String(args[8]) == "cabecera") if len(args) > 8 else False

    with DeviceContext() as ctx:
        print("SPO-Mojo: agente ENTRENADO sobre tres en raya (rival aleatorio)")
        print("  particulas: entrena con", NUM_PARTICLES, " mide con", n_part)
        print("  profundidad", SEARCH_DEPTH, " temperatura", TEMPERATURE)
        print("  gamma_r", gamma_r, " (Stoix pasa la recompensa cruda: 1.0)")
        print("  loss_penalty", penalty, " (Stoix no lo tiene: 0.0)")
        print("  semilla de entrenamiento", train_seed)
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
                            spo_readout=spo_readout, gamma_r=gamma_r,
                            penalty=penalty, use_critic=True,
                            depth_disc=gamma_r < Scalar[dtype](1),
                            rounds=rounds, seed=train_seed)
        train_seconds = Float64(perf_counter_ns() - train_start) / 1e9
        print("  tiempo de entrenamiento: " + fmt_fixed(train_seconds, 1)
              + " s  (se paga UNA vez; el MCTS no lo paga, pero paga busqueda en"
              + " cada jugada)")
        print()

        # Calentamiento: la primera llamada paga la compilacion de los kernels y ese
        # coste no es parte de lo que se quiere medir.
        _ = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS, 3, n_part,
                       True, True, spo_readout, gamma_r, penalty)

        rows = List[PlannerMetrics]()

        print("=== lote de", BATCH_ENVS, "partidas (throughput) ===")
        moda = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS, BATCH_STEPS,
                          n_part, True, True, spo_readout, gamma_r, penalty)
        show("smc_agent_moda_lote", moda)
        check_ceiling("smc_agent_moda_lote", moda)
        deux_lignes(rows, "smc_agent_moda_lote", moda, n_part)

        muestreada = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS, BATCH_STEPS,
                                n_part, True, False, spo_readout, gamma_r, penalty)
        show("smc_agent_muestreada_lote", muestreada)
        check_ceiling("smc_agent_muestreada_lote", muestreada)
        deux_lignes(rows, "smc_agent_muestreada_lote", muestreada, n_part)

        # El brazo SIN red, con los mismos hiperparametros: es el termino de
        # comparacion del eje de asimetria (pagar entrenamiento una vez frente a
        # pagar busqueda en cada jugada).
        sin_red = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS, BATCH_STEPS,
                             n_part, False, True, spo_readout, gamma_r, penalty)
        show("smc_planificador_moda_lote", sin_red)
        check_ceiling("smc_planificador_moda_lote", sin_red)
        deux_lignes(rows, "smc_planificador_moda_lote", sin_red, n_part)

        print()
        print("=== una partida a la vez (latencia, comparable con el MCTS) ===")
        lat = play_timed(ctx, outcome.actor, outcome.critic, 1, LATENCY_STEPS, n_part,
                         True, True)
        latency = lat.seconds / Float64(lat.decisions)
        show("smc_agent_moda_latencia", lat)
        check_ceiling("smc_agent_moda_latencia", lat)
        deux_lignes(rows, "smc_agent_moda_latencia", lat, n_part)

        batched_cost = moda.seconds / Float64(moda.decisions)
        print("  latencia        : " + fmt_fixed(latency, 6) + " s/decision")
        print("  coste repartido : " + fmt_fixed(batched_cost, 6) + " s/decision")
        print("  ganancia de lote: x" + fmt_fixed(latency / batched_cost, 1)
              + "  (" + String(BATCH_ENVS) + " partidas a la vez)")

        if solo_cabecera:
            write_csv_rows(rows, path)
            print()
            print("csv escrito en", path, " (solo cabecera)")
            return

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
        budgets.append(256)
        budgets.append(512)
        for b in budgets:
            con = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS, BATCH_STEPS, b,
                             True, True, spo_readout, gamma_r, penalty)
            sin = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS, BATCH_STEPS, b,
                             False, True, spo_readout, gamma_r, penalty)
            check_ceiling("presupuesto_con_red_" + String(b), con)
            check_ceiling("presupuesto_sin_red_" + String(b), sin)
            print("  " + pad(String(b), 10, True) + two(con, sin))
            deux_lignes(rows, "presupuesto_con_red_" + String(b), con, b)
            deux_lignes(rows, "presupuesto_sin_red_" + String(b), sin, b)

        # --- Barrido de los ejes que Stoix tambien expone ---
        #
        # El agente queda FIJO y se mueve solo la busqueda, que es lo que se puede
        # hacer identico en las dos implementaciones. Lo que se compara no son los
        # valores absolutos (son dos agentes distintos) sino la FORMA: donde satura,
        # donde cruza, en que direccion responde. Dos implementaciones del mismo
        # algoritmo tienen que responder igual a los mismos mandos.
        print()
        print("=== BARRIDO: profundidad (particulas fijas en", n_part, ") ===")
        depths = List[Int]()
        depths.append(1); depths.append(2); depths.append(3)
        depths.append(4); depths.append(6); depths.append(8)
        for d in depths:
            c = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS,
                           BATCH_STEPS, n_part, True, True, spo_readout,
                           gamma_r, penalty, d, TEMPERATURE, NO_RESAMPLE)
            sn = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS,
                            BATCH_STEPS, n_part, False, True, spo_readout,
                            gamma_r, penalty, d, TEMPERATURE, NO_RESAMPLE)
            check_ceiling("profundidad_" + String(d), c)
            check_ceiling("profundidad_sinred_" + String(d), sn)
            print("  profundidad " + pad(String(d), 3, True) + two(c, sn))
            deux_lignes(rows, "barrido_profundidad_" + String(d), c, n_part)
            deux_lignes(rows, "barrido_profundidad_sinred_" + String(d),
                        sn, n_part)

        print()
        print("=== BARRIDO: temperatura ===")
        temps = List[Scalar[dtype]]()
        temps.append(0.005); temps.append(0.02); temps.append(0.1)
        temps.append(0.5); temps.append(1.0)
        for tv in temps:
            c = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS,
                           BATCH_STEPS, n_part, True, True, spo_readout,
                           gamma_r, penalty, SEARCH_DEPTH, tv, NO_RESAMPLE)
            sn = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS,
                            BATCH_STEPS, n_part, False, True, spo_readout,
                            gamma_r, penalty, SEARCH_DEPTH, tv, NO_RESAMPLE)
            check_ceiling("temperatura", c)
            check_ceiling("temperatura_sinred", sn)
            print("  tau " + fmt_fixed(Float64(tv), 3) + two(c, sn))
            lbl = fmt_fixed(Float64(tv), 3)
            deux_lignes(rows, "barrido_tau_" + lbl, c, n_part)
            deux_lignes(rows, "barrido_tau_sinred_" + lbl, sn, n_part)

        print()
        print("=== BARRIDO: periodo de remuestreo (99 = apagado) ===")
        periods = List[Int]()
        periods.append(1); periods.append(2); periods.append(4)
        periods.append(8); periods.append(99)
        for pr in periods:
            c = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS,
                           BATCH_STEPS, n_part, True, True, spo_readout,
                           gamma_r, penalty, SEARCH_DEPTH, TEMPERATURE, pr)
            sn = play_timed(ctx, outcome.actor, outcome.critic, BATCH_ENVS,
                            BATCH_STEPS, n_part, False, True, spo_readout,
                            gamma_r, penalty, SEARCH_DEPTH, TEMPERATURE, pr)
            check_ceiling("remuestreo", c)
            check_ceiling("remuestreo_sinred", sn)
            print("  periodo " + pad(String(pr), 3, True) + two(c, sn))
            deux_lignes(rows, "barrido_remuestreo_" + String(pr), c, n_part)
            deux_lignes(rows, "barrido_remuestreo_sinred_" + String(pr), sn, n_part)

        write_csv_rows(rows, path)
        print()
        print("csv escrito en", path)
