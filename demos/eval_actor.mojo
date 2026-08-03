"""E2.6: las tres mediciones que cierran la etapa del actor.

  1. **La red SOLA**, jugando sin buscar. Es la prueba de destilacion: ¿se metio en
     los pesos algo del conocimiento de la busqueda? Se compara contra el azar
     exacto (0.6484), calculado por recursion.
  2. **El 2x2**: {SPO fiel, corregido} x {sin actor, con actor}. La celda que
     demuestra el bucle EM es la segunda columna: la busqueda CON red superando a
     la busqueda SIN red con el mismo presupuesto.
  3. **La curva de presupuesto**: score contra numero de particulas, con y sin red.
     Con red deberia alcanzarse antes el mismo nivel -- que es el argumento
     practico de SPO frente a MCTS (pagas el entrenamiento una vez y luego buscas
     menos).

**Por que el 2x2 y no una comparacion directa.** Comparar "actor + readout
corregido" contra "sin actor + readout de SPO" mezclaria dos cambios y no se podria
atribuir la mejora a ninguno. Con las cuatro celdas se aisla la contribucion del
actor (dentro de cada fila) y la del readout (entre filas).

**Cada montaje conserva SUS hiperparametros**, que es lo honesto: el de SPO va con
remuestreo, gamma_r = 0.7 y sin castigo (el de M1); el corregido va sin remuestreo,
gamma_r = 0.9 y castigo 1 (el de E1.11c). No es una ablacion pura del readout, es
"los dos sistemas que tenemos, cada uno en su mejor ajuste, con y sin prior
aprendido". Y cada uno entrena SU PROPIO actor con SU PROPIA q: un actor entrenado
con la q de un readout no es el prior adecuado para el otro.

Todas las partidas se juegan con la **moda** de q, no con una muestra: se esta
midiendo fuerza, no explorando.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from bench.metrics import fmt_fixed, wilson_lo, wilson_hi
from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform
from envs.tictactoe import (TicTacToe, ttt_reset_kernel, ttt_env_step_kernel,
                            ttt_auto_reset_kernel, ttt_encode_obs_kernel,
                            NUM_ACTIONS, STATE_DIM, OBS_DIM, TPB_TTT)
from envs.tictactoe_actor import TicTacToeActor
from envs.tictactoe_runner import RNG_RIVAL
from networks.actor import Actor
from systems.spo.launch import TPB, blocks_for
from systems.spo.particles import SearchWorkspace
from systems.spo.readout import (readout_greedy, readout_expected,
                                 argmax_action_kernel)
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from tests.helpers import download
from demos.train_spo import (train_run, ActorLearner, HIDDEN, SEARCH_DEPTH,
                             TEMPERATURE, NO_RESAMPLE)

comptime EVAL_ENVS = 128
comptime EVAL_STEPS = 300
comptime SWEEP_STEPS = 100
comptime EVAL_SEED = UInt32(20260803)

# Referencias exactas, por recursion sobre todos los estados.
comptime RANDOM_SCORE = 0.6484
comptime OPTIMAL_SCORE = 0.9974

# Los dos montajes, cada uno con sus hiperparametros.
comptime SPO_PERIOD = 3
comptime SPO_GAMMA = Scalar[dtype](0.7)
comptime SPO_PENALTY = Scalar[dtype](0.0)
comptime FIX_GAMMA = Scalar[dtype](0.9)
comptime FIX_PENALTY = Scalar[dtype](1.0)
comptime EVAL_PARTICLES = 128


@fieldwise_init
struct Arm(Copyable, Movable):
    var name: String
    var wins: Int
    var draws: Int
    var losses: Int

    def games(self) -> Int:
        return self.wins + self.draws + self.losses

    def score(self) -> Float64:
        n = self.games()
        return (Float64(self.wins) + 0.5 * Float64(self.draws)) / Float64(n)


def pct(x: Float64) -> String:
    return fmt_fixed(x * 100.0, 2) + "%"


def show(a: Arm) raises:
    n = a.games()
    s = a.score()
    # El IC va sobre el SCORE via las victorias y las derrotas por separado: el
    # score es una media de 1/0.5/0 y su IC binomial no aplica tal cual, asi que se
    # informa el IC de las derrotas (la columna que el juego optimo clava a 0).
    print("   ", a.name, "  n=", n,
          "  gana ", pct(Float64(a.wins) / Float64(n)),
          "  empata ", pct(Float64(a.draws) / Float64(n)),
          "  PIERDE ", pct(Float64(a.losses) / Float64(n)),
          " IC[", fmt_fixed(wilson_lo(a.losses, n) * 100.0, 2), ",",
          fmt_fixed(wilson_hi(a.losses, n) * 100.0, 2), "]",
          "  score ", fmt_fixed(s, 4))


@fieldwise_init
struct NetArm(Copyable, Movable):
    var arm: Arm
    var illegal: Int
    """Cuantas veces la red eligio una casilla ocupada. TIENE que ser 0."""


def play_network_only(ctx: DeviceContext, name: String, actor: Actor,
                      steps: Int) raises -> NetArm:
    """La red decide sola: argmax de pi, sin busqueda ninguna.

    Es la prueba de destilacion. Si la red no hubiera aprendido nada de la
    busqueda, jugaria como el prior inicial (casi uniforme sobre las legales) y
    sacaria el score del azar, 0.6484.
    """
    blocks = (EVAL_ENVS + TPB_TTT - 1) // TPB_TTT
    state = zero_buffer[dtype](ctx, EVAL_ENVS * STATE_DIM)
    obs = zero_buffer[dtype](ctx, EVAL_ENVS * OBS_DIM)
    action = zero_buffer[idx_dtype](ctx, EVAL_ENVS)
    reward = zero_buffer[dtype](ctx, EVAL_ENVS)
    done = zero_buffer[idx_dtype](ctx, EVAL_ENVS)
    u_rival = zero_buffer[dtype](ctx, EVAL_ENVS)

    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()

    wins = 0; draws = 0; losses = 0
    illegal = 0
    for step in range(steps):
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            obs.unsafe_ptr(), state.unsafe_ptr(), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        actor.forward(ctx, state, obs, EVAL_ENVS)
        # La jugada es el argmax de pi. Las ocupadas salen a 0 exacto, asi que el
        # argmax nunca puede caer en una casilla ilegal.
        ctx.enqueue_function[argmax_action_kernel, argmax_action_kernel](
            action.unsafe_ptr(), actor.probs.unsafe_ptr(), EVAL_ENVS,
            NUM_ACTIONS, grid_dim=blocks_for(EVAL_ENVS), block_dim=TPB)
        # ANTES de aplicarla: ¿es legal? `ttt_apply` no lo comprueba (lo dice su
        # docstring), asi que una jugada ilegal sobrescribiria la ficha del rival
        # y el score saldria inflado. Se CUENTA en vez de suponer que el
        # enmascarado basta.
        ctx.synchronize()
        with state.map_to_host() as sh:
            with action.map_to_host() as ah:
                for e in range(EVAL_ENVS):
                    if sh[e * STATE_DIM + Int(ah[e])] != Scalar[dtype](0):
                        illegal += 1

        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), EVAL_SEED, RNG_RIVAL + UInt32(step),
            EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), action.unsafe_ptr(), u_rival.unsafe_ptr(),
            reward.unsafe_ptr(), done.unsafe_ptr(), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
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
    return NetArm(Arm(name, wins, draws, losses), illegal)


def diagnose_prior(ctx: DeviceContext, actor: ActorLearner, spo_readout: Bool,
                   particles: Int, steps: Int) raises:
    """¿Colapsa el prior aprendido la diversidad de la busqueda?

    Es la explicacion candidata de por que el criterio 2 falla. Si el prior es muy
    picudo, `categorical_from_logits` sortea casi siempre la misma accion raiz: las
    particulas acaban explorando UNA jugada y la busqueda no tiene nada que
    comparar. Se mide contando cuantas acciones raiz DISTINTAS muestrean las
    particulas, con prior de actor y con prior uniforme, sobre EL MISMO estado y
    con LA MISMA semilla.
    """
    period = SPO_PERIOD if spo_readout else NO_RESAMPLE
    gamma_r = SPO_GAMMA if spo_readout else FIX_GAMMA
    penalty = SPO_PENALTY if spo_readout else FIX_PENALTY
    cfg = SPOConfig(num_envs=EVAL_ENVS, num_particles=particles,
                    num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                    search_depth=SEARCH_DEPTH, resample_period=period,
                    temperature=TEMPERATURE, search_gamma=1.0,
                    search_gae_lambda=1.0)
    model = TicTacToe(gamma_r, penalty)
    amodel = TicTacToeActor(ctx, cfg.num_search_particles(), HIDDEN, gamma_r,
                            penalty)
    amodel.sync_from(ctx, actor.net.params)
    ws = SearchWorkspace(ctx, cfg)
    q_buf = zero_buffer[dtype](ctx, EVAL_ENVS * NUM_ACTIONS)
    logits_buf = zero_buffer[dtype](ctx, EVAL_ENVS * NUM_ACTIONS)
    u_dummy = zero_buffer[dtype](ctx, EVAL_ENVS)

    blocks = (EVAL_ENVS + TPB_TTT - 1) // TPB_TTT
    state = zero_buffer[dtype](ctx, EVAL_ENVS * STATE_DIM)
    reward = zero_buffer[dtype](ctx, EVAL_ENVS)
    done = zero_buffer[idx_dtype](ctx, EVAL_ENVS)
    u_rival = zero_buffer[dtype](ctx, EVAL_ENVS)
    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()

    sum_a = Scalar[dtype](0); sum_u = Scalar[dtype](0)
    sum_legal = Scalar[dtype](0); count = 0
    # Y la otra hipotesis, especifica del readout de MEDIA: la media por accion
    # tiene varianza inversa al numero de particulas que la muestrean. Un prior
    # picudo deja acciones legales con 1 o 2 particulas, y su media es ruido -- que
    # luego compite de tu a tu con la de una accion muestreada 40 veces. El readout
    # de SPO no sufre esto porque su suma de exponenciales pesa implicitamente por
    # cuantas particulas hay.
    thin_a = 0; thin_u = 0; min_a = Scalar[dtype](0); min_u = Scalar[dtype](0)

    for step in range(steps):
        sd = EVAL_SEED ^ (UInt32(step) * 2654435761)
        search[TicTacToeActor](ctx, ws, cfg, amodel, state, sd)
        ctx.synchronize()
        roots_a = download[idx_dtype](ws.particles.root_actions,
                                     cfg.num_search_particles())
        search[TicTacToe](ctx, ws, cfg, model, state, sd)
        ctx.synchronize()
        roots_u = download[idx_dtype](ws.particles.root_actions,
                                      cfg.num_search_particles())

        with state.map_to_host() as sh:
            for e in range(EVAL_ENVS):
                seen_a = List[Int](); seen_u = List[Int]()
                for _ in range(NUM_ACTIONS):
                    seen_a.append(0); seen_u.append(0)
                for n in range(particles):
                    seen_a[Int(roots_a[e * particles + n])] = 1
                    seen_u[Int(roots_u[e * particles + n])] = 1
                da = 0; du = 0; legal = 0
                for a in range(NUM_ACTIONS):
                    da += seen_a[a]; du += seen_u[a]
                    if sh[e * STATE_DIM + a] == Scalar[dtype](0):
                        legal += 1
                sum_a += Scalar[dtype](da); sum_u += Scalar[dtype](du)
                sum_legal += Scalar[dtype](legal); count += 1

                # Cuenta de particulas por accion, y cuantas acciones quedan
                # "flacas" (menos de 4 particulas: su media es poco fiable).
                cnt_a = List[Int](); cnt_u = List[Int]()
                for _ in range(NUM_ACTIONS):
                    cnt_a.append(0); cnt_u.append(0)
                for n in range(particles):
                    cnt_a[Int(roots_a[e * particles + n])] += 1
                    cnt_u[Int(roots_u[e * particles + n])] += 1
                lo_a = particles; lo_u = particles
                for a in range(NUM_ACTIONS):
                    if cnt_a[a] > 0:
                        if cnt_a[a] < 4: thin_a += 1
                        if cnt_a[a] < lo_a: lo_a = cnt_a[a]
                    if cnt_u[a] > 0:
                        if cnt_u[a] < 4: thin_u += 1
                        if cnt_u[a] < lo_u: lo_u = cnt_u[a]
                min_a += Scalar[dtype](lo_a); min_u += Scalar[dtype](lo_u)

        # Avanzar con la busqueda CON actor, para recorrer las posiciones que ese
        # agente visita de verdad.
        search[TicTacToeActor](ctx, ws, cfg, amodel, state, sd)
        if spo_readout:
            readout_greedy(ctx, ws.particles, ws.output, cfg, q_buf)
        else:
            readout_expected(ctx, ws.particles, ws.output, cfg, logits_buf,
                             q_buf, u_dummy, True)
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), EVAL_SEED, RNG_RIVAL + UInt32(step),
            EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), ws.output.action.unsafe_ptr(),
            u_rival.unsafe_ptr(), reward.unsafe_ptr(), done.unsafe_ptr(),
            EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

    c = Scalar[dtype](count)
    print("      acciones raiz DISTINTAS de", particles, "particulas:",
          " con prior ", sum_a / c, " | sin prior ", sum_u / c,
          " | legales ", sum_legal / c)
    print("      particulas de la accion MENOS muestreada:  con prior ",
          min_a / c, " | sin prior ", min_u / c)
    print("      acciones 'flacas' (<4 particulas) por posicion:  con prior ",
          Scalar[dtype](thin_a) / c, " | sin prior ",
          Scalar[dtype](thin_u) / c)


def play_search(ctx: DeviceContext, name: String, actor: ActorLearner,
                use_actor: Bool, spo_readout: Bool, particles: Int,
                steps: Int) raises -> Arm:
    """Juega con la busqueda, con o sin prior aprendido, con uno u otro readout.

    Cada montaje trae sus hiperparametros: el de SPO con remuestreo y gamma_r 0.7,
    el corregido sin remuestreo, gamma_r 0.9 y castigo 1.
    """
    period = SPO_PERIOD if spo_readout else NO_RESAMPLE
    gamma_r = SPO_GAMMA if spo_readout else FIX_GAMMA
    penalty = SPO_PENALTY if spo_readout else FIX_PENALTY

    cfg = SPOConfig(num_envs=EVAL_ENVS, num_particles=particles,
                    num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                    search_depth=SEARCH_DEPTH, resample_period=period,
                    temperature=TEMPERATURE, search_gamma=1.0,
                    search_gae_lambda=1.0)
    model = TicTacToe(gamma_r, penalty)
    amodel = TicTacToeActor(ctx, cfg.num_search_particles(), HIDDEN, gamma_r,
                            penalty)
    amodel.sync_from(ctx, actor.net.params)
    ws = SearchWorkspace(ctx, cfg)
    q_buf = zero_buffer[dtype](ctx, EVAL_ENVS * NUM_ACTIONS)
    logits_buf = zero_buffer[dtype](ctx, EVAL_ENVS * NUM_ACTIONS)
    u_dummy = zero_buffer[dtype](ctx, EVAL_ENVS)

    blocks = (EVAL_ENVS + TPB_TTT - 1) // TPB_TTT
    state = zero_buffer[dtype](ctx, EVAL_ENVS * STATE_DIM)
    reward = zero_buffer[dtype](ctx, EVAL_ENVS)
    done = zero_buffer[idx_dtype](ctx, EVAL_ENVS)
    u_rival = zero_buffer[dtype](ctx, EVAL_ENVS)
    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), EVAL_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()

    wins = 0; draws = 0; losses = 0
    for step in range(steps):
        sd = EVAL_SEED ^ (UInt32(step) * 2654435761)
        if use_actor:
            search[TicTacToeActor](ctx, ws, cfg, amodel, state, sd)
        else:
            search[TicTacToe](ctx, ws, cfg, model, state, sd)
        # La moda de q en los dos casos: se mide fuerza, no exploracion.
        if spo_readout:
            readout_greedy(ctx, ws.particles, ws.output, cfg, q_buf)
        else:
            readout_expected(ctx, ws.particles, ws.output, cfg, logits_buf,
                             q_buf, u_dummy, True)

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
                        if r > Scalar[dtype](0.75): wins += 1
                        elif r > Scalar[dtype](0.25): draws += 1
                        else: losses += 1
        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), EVAL_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()
    return Arm(name, wins, draws, losses)


def verdict(base: Arm, other: Arm) raises:
    """Compara dos brazos por las derrotas, con la regla de siempre: si los IC de
    Wilson se solapan, no se afirma diferencia."""
    n1 = base.games(); n2 = other.games()
    hi2 = wilson_hi(other.losses, n2); lo1 = wilson_lo(base.losses, n1)
    lo2 = wilson_lo(other.losses, n2); hi1 = wilson_hi(base.losses, n1)
    p1 = Float64(base.losses) / Float64(n1)
    p2 = Float64(other.losses) / Float64(n2)
    print("      derrotas ", pct(p1), " -> ", pct(p2),
          "   score ", fmt_fixed(base.score(), 4), " -> ",
          fmt_fixed(other.score(), 4))
    if hi2 < lo1:
        print("      los IC NO se solapan y el nuevo esta por debajo: pierde menos.")
    elif lo2 > hi1:
        print("      los IC NO se solapan y el nuevo esta por ENCIMA: pierde mas.")
    else:
        print("      los IC se SOLAPAN: no se afirma diferencia en derrotas.")


def main() raises:
    with DeviceContext() as ctx:
        print("=== E2.6: la red sola, el 2x2, y la curva de presupuesto ===")
        print("   evaluacion:", EVAL_ENVS, "partidas a la vez x", EVAL_STEPS,
              "turnos, jugando la MODA de q")
        print("   referencias exactas: azar", RANDOM_SCORE, " optimo",
              OPTIMAL_SCORE, "(este ultimo pierde 0.00%)")
        print()

        # --- entrenar un actor por montaje, cada uno con SU q ---
        print("--- 1. entrenamiento (un actor por montaje) ---")
        fixed = train_run(ctx, String("corregido, bucle EM"), True, False,
                          NO_RESAMPLE, FIX_GAMMA, FIX_PENALTY, EVAL_PARTICLES)
        spo = train_run(ctx, String("SPO fiel, bucle EM"), True, True,
                        SPO_PERIOD, SPO_GAMMA, SPO_PENALTY, EVAL_PARTICLES)

        print("--- 2. la red SOLA (destilacion: sin buscar nada) ---")
        net_fix = play_network_only(ctx, String("red del montaje corregido"),
                                    fixed.actor.net, EVAL_STEPS)
        net_spo = play_network_only(ctx, String("red del montaje SPO      "),
                                    spo.actor.net, EVAL_STEPS)
        show(net_fix.arm)
        show(net_spo.arm)
        print("    jugadas ILEGALES:  corregido ", net_fix.illegal, "  SPO ",
              net_spo.illegal, "   <- tienen que ser 0, o el score esta inflado")
        print("    referencia: jugar al azar da score", RANDOM_SCORE,
              "y pierde 28.81%")
        print()

        print("--- 3. el 2x2 (mismo presupuesto: N =", EVAL_PARTICLES, ") ---")
        a_spo_no = play_search(ctx, String("SPO fiel   sin actor"), spo.actor,
                               False, True, EVAL_PARTICLES, EVAL_STEPS)
        a_spo_yes = play_search(ctx, String("SPO fiel   CON actor"), spo.actor,
                                True, True, EVAL_PARTICLES, EVAL_STEPS)
        a_fix_no = play_search(ctx, String("corregido  sin actor"), fixed.actor,
                               False, False, EVAL_PARTICLES, EVAL_STEPS)
        a_fix_yes = play_search(ctx, String("corregido  CON actor"), fixed.actor,
                                True, False, EVAL_PARTICLES, EVAL_STEPS)
        show(a_spo_no); show(a_spo_yes); show(a_fix_no); show(a_fix_yes)
        print()
        print("--- 4. ¿la red mejora la busqueda que la entreno? ---")
        print("    (es la celda que demuestra el bucle EM)")
        print("    SPO fiel:")
        verdict(a_spo_no, a_spo_yes)
        print("    corregido:")
        verdict(a_fix_no, a_fix_yes)
        print()
        print("--- 4b. ¿por que el prior no ayuda a presupuesto alto? ---")
        print("    Candidata: un prior picudo colapsa la diversidad de acciones")
        print("    raiz, y la busqueda se queda sin nada que comparar.")
        print("    corregido:")
        diagnose_prior(ctx, fixed.actor, False, EVAL_PARTICLES, 40)
        print("    SPO fiel:")
        diagnose_prior(ctx, spo.actor, True, EVAL_PARTICLES, 40)
        print()

        print("--- 5. curva de presupuesto (montaje corregido) ---")
        print("    nuestro barrido sin red ya medido: 4->0.865  16->0.953  64->0.969")
        ns = List[Int]()
        ns.append(4); ns.append(16); ns.append(64); ns.append(128)
        for i in range(len(ns)):
            no = play_search(ctx, String("N=", ns[i], " sin actor"),
                             fixed.actor, False, False, ns[i], SWEEP_STEPS)
            yes = play_search(ctx, String("N=", ns[i], " CON actor"),
                              fixed.actor, True, False, ns[i], SWEEP_STEPS)
            print("    N=", ns[i], "  sin red score ",
                  fmt_fixed(no.score(), 4), " (pierde ",
                  pct(Float64(no.losses) / Float64(no.games())), ")",
                  "   con red score ", fmt_fixed(yes.score(), 4),
                  " (pierde ", pct(Float64(yes.losses) / Float64(yes.games())),
                  ")")
        print()
        print("    Si con red se alcanza antes el mismo nivel, ese es el argumento")
        print("    practico de SPO frente a MCTS: el entrenamiento se paga una vez")
        print("    y luego cada decision cuesta menos busqueda.")
