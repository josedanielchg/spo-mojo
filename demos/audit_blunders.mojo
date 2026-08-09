"""How can it lose? A move-by-move audit of the planner.

Optimal play against a random rival loses 0.00% of its games. Ours loses ~2%. This
demo does not measure: it DIAGNOSES. On every agent turn it looks at the board from
the outside, with tic-tac-toe rules written by hand on the host, and classifies the
position before letting the search play:

    can win now            there is a cell that makes it three in a row
    has to block           the rival has two in a row and one free cell
    double threat          the rival has TWO winning cells -> no block is possible
                           any more, the mistake happened earlier
    quiet                  neither one nor the other

And then it compares against what the search played. That separates two completely
different faults:

  - **tactical fault**: there was a threat one move away and it did not block it.
    That means the search failed to see something at depth 1.
  - **strategic fault**: it reached a position with a double threat, where nothing
    can be done any more. The real mistake was several moves earlier, when the fork
    was allowed.

The difference matters a great deal for knowing what to fix. If they are tactical
faults, the problem is the search's resolution (particles, temperature, readout).
If they are forks, the problem is that the search maximises the EXPECTED result
against a random rival and a fork is only punished in the few particles where the
random rival happens to exploit it.

On top of that, in the positions where it fails, how much q mass the search gave to
the correct move gets printed. That tells "it did not see it" from "it saw it and
still chose otherwise".
"""

from std.gpu.host import DeviceContext

from bench.metrics import fmt_fixed, wilson_lo, wilson_hi
from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform
from envs.tictactoe import (TicTacToe, ttt_reset_kernel, ttt_env_step_kernel,
                            ttt_auto_reset_kernel, NUM_ACTIONS, NUM_CELLS,
                            STATE_DIM, CELL_AGENT, CELL_RIVAL, CELL_EMPTY,
                            TPB_TTT)
from envs.tictactoe_runner import RNG_RIVAL
from systems.spo.particles import SearchWorkspace
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from systems.spo.readout import readout_greedy, readout_expected

comptime SEED = UInt32(20260731)
comptime ENVS = 128
comptime STEPS = 400
comptime SWEEP_STEPS = 150
comptime NUM_PARTICLES = 64
comptime SWEEP_STEPS_BIG = 60
comptime SEARCH_DEPTH = 6
comptime RESAMPLE_PERIOD = 3
comptime TEMPERATURE = Scalar[dtype](0.02)
comptime REWARD_GAMMA = Scalar[dtype](0.7)

# The 8 lines, as in the environment's kernel but here on the host.
comptime LINES = 8


def line_cell(line: Int, k: Int) -> Int:
    """Cell k (0..2) of line `line` (0..7). Rows, columns, diagonals."""
    if line == 0: return k            # fila 0: 0,1,2
    if line == 1: return 3 + k
    if line == 2: return 6 + k
    if line == 3: return 3 * k        # columna 0: 0,3,6
    if line == 4: return 1 + 3 * k
    if line == 5: return 2 + 3 * k
    if line == 6: return 4 * k        # diagonal: 0,4,8
    return 2 + 2 * k                  # antidiagonal: 2,4,6


def winning_cells(board: List[Scalar[dtype]], base: Int,
                  player: Scalar[dtype]) -> List[Int]:
    """The cells that give `player` an IMMEDIATE win.

    A cell counts if there is a line with two of `player`'s marks and that cell
    empty. All of them are returned, without repeats, because the number of
    distinct cells is precisely what separates a "simple threat" (blockable) from a
    "fork" (already lost).
    """
    found = List[Int]()
    for line in range(LINES):
        mine = 0
        empty_cell = -1
        for k in range(3):
            c = line_cell(line, k)
            v = board[base + c]
            if v == player:
                mine += 1
            elif v == CELL_EMPTY:
                empty_cell = c
        if mine == 2 and empty_cell >= 0:
            already = False
            for f in found:
                if f == empty_cell:
                    already = True
            if not already:
                found.append(empty_cell)
    return found^


def pct(x: Float64) -> String:
    return fmt_fixed(x * 100.0, 2) + "%"


def ratio(a: Int, b: Int) -> String:
    if b == 0:
        return String("  -   (0 casos)")
    return (fmt_fixed(100.0 * Float64(a) / Float64(b), 2) + "%  ("
            + String(a) + " de " + String(b) + ")")


@fieldwise_init
struct Audit(Copyable, Movable):
    """What comes out of an audit."""
    var temperature: Scalar[dtype]
    var reward_gamma: Scalar[dtype]
    var loss_penalty: Scalar[dtype]
    var particles: Int
    var expected: Bool
    var period: Int
    var wins: Int
    var draws: Int
    var losses: Int
    var can_win: Int
    var took_win: Int
    var must_block: Int
    var blocked: Int
    var forked: Int
    var created_fork: Int
    var q_missed: Scalar[dtype]
    var missed: Int

    def games(self) -> Int:
        return self.wins + self.draws + self.losses


def audit(ctx: DeviceContext, temperature: Scalar[dtype],
          reward_gamma: Scalar[dtype], steps: Int,
          loss_penalty: Scalar[dtype] = 0,
          num_particles: Int = NUM_PARTICLES,
          expected: Bool = False,
          period: Int = RESAMPLE_PERIOD) raises -> Audit:
    """Plays and classifies every agent turn. See the file header."""
    cfg = SPOConfig(num_envs=ENVS, num_particles=num_particles,
                    num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                    search_depth=SEARCH_DEPTH, resample_period=period,
                    temperature=temperature, search_gamma=1.0,
                    search_gae_lambda=1.0)
    model = TicTacToe(reward_gamma, loss_penalty)
    ws = SearchWorkspace(ctx, cfg)
    q_buf = zero_buffer[dtype](ctx, ENVS * NUM_ACTIONS)
    logits_buf = zero_buffer[dtype](ctx, ENVS * NUM_ACTIONS)
    us_dummy = zero_buffer[dtype](ctx, ENVS)
    blocks = (ENVS + TPB_TTT - 1) // TPB_TTT

    state = zero_buffer[dtype](ctx, ENVS * STATE_DIM)
    reward = zero_buffer[dtype](ctx, ENVS)
    done = zero_buffer[idx_dtype](ctx, ENVS)
    u_rival = zero_buffer[dtype](ctx, ENVS)
    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), ENVS, grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()

    wins = 0; draws = 0; losses = 0
    can_win = 0; took_win = 0
    must_block = 0; blocked = 0
    forked = 0; created_fork = 0
    q_missed = Scalar[dtype](0); missed = 0

    for step in range(steps):
        search[TicTacToe](ctx, ws, cfg, model, state,
                          SEED ^ (UInt32(step) * 2654435761))
        if expected:
            readout_expected(ctx, ws.particles, ws.output, cfg, logits_buf,
                             q_buf, us_dummy, True)
        else:
            readout_greedy(ctx, ws.particles, ws.output, cfg, q_buf)
        ctx.synchronize()

        board = List[Scalar[dtype]]()
        with state.map_to_host() as sh:
            for i in range(ENVS * STATE_DIM):
                board.append(sh[i])
        acts = List[Int]()
        with ws.output.action.map_to_host() as ah:
            for e in range(ENVS):
                acts.append(Int(ah[e]))
        qs = List[Scalar[dtype]]()
        with q_buf.map_to_host() as qh:
            for i in range(ENVS * NUM_ACTIONS):
                qs.append(qh[i])

        for e in range(ENVS):
            base = e * STATE_DIM
            a = acts[e]
            mine = winning_cells(board, base, CELL_AGENT)
            theirs = winning_cells(board, base, CELL_RIVAL)

            if len(mine) > 0:
                can_win += 1
                hit = False
                for c in mine:
                    if c == a: hit = True
                if hit: took_win += 1
            elif len(theirs) >= 2:
                forked += 1
            elif len(theirs) == 1:
                must_block += 1
                if a == theirs[0]:
                    blocked += 1
                else:
                    q_missed += qs[e * NUM_ACTIONS + theirs[0]]
                    missed += 1

            if board[base + a] == CELL_EMPTY:
                board[base + a] = CELL_AGENT
                after = winning_cells(board, base, CELL_RIVAL)
                board[base + a] = CELL_EMPTY
                if len(after) >= 2 and len(theirs) < 2:
                    created_fork += 1

        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), SEED, RNG_RIVAL + UInt32(step), ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), ws.output.action.unsafe_ptr(),
            u_rival.unsafe_ptr(), reward.unsafe_ptr(), done.unsafe_ptr(),
            ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

        with reward.map_to_host() as rh:
            with done.map_to_host() as dh:
                for e in range(ENVS):
                    if Int(dh[e]) != 0:
                        r = rh[e]
                        if r > Scalar[dtype](0.75): wins += 1
                        elif r > Scalar[dtype](0.25): draws += 1
                        else: losses += 1

        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()
    return Audit(temperature, reward_gamma, loss_penalty, num_particles,
                 expected, period, wins, draws, losses, can_win,
                 took_win, must_block, blocked, forked, created_fork,
                 q_missed, missed)


def row(a: Audit) raises:
    n = a.games()
    blk = 100.0 * Float64(a.blocked) / Float64(a.must_block) \
          if a.must_block > 0 else 0.0
    fin = 100.0 * Float64(a.took_win) / Float64(a.can_win) \
          if a.can_win > 0 else 0.0
    print("   tau=", fmt_fixed(Float64(a.temperature), 2),
          " gamma_r=", fmt_fixed(Float64(a.reward_gamma), 2),
          " castigo=", fmt_fixed(Float64(a.loss_penalty), 2),
          " N=", a.particles,
          "  " + ("MEDIA" if a.expected else " SPO "),
          " resamp=" + ("no " if a.period > SEARCH_DEPTH else "si "),
          "  bloquea ", fmt_fixed(blk, 1), "%",
          "  remata ", fmt_fixed(fin, 1), "%",
          "  pierde ", pct(Float64(a.losses) / Float64(n)),
          " IC[", fmt_fixed(wilson_lo(a.losses, n) * 100.0, 2), ",",
          fmt_fixed(wilson_hi(a.losses, n) * 100.0, 2), "]",
          "  score ", fmt_fixed((Float64(a.wins) + 0.5 * Float64(a.draws))
                                / Float64(n), 4))


def main() raises:
    with DeviceContext() as ctx:
        print("=== auditoria: por que pierde el planificador ===")
        print("   ", ENVS, "partidas en paralelo, jugando la moda de q")
        print()

        base = audit(ctx, TEMPERATURE, REWARD_GAMMA, STEPS)
        n = base.games()
        print("--- 1. el montaje actual (tau=0.02, gamma_r=0.7) ---")
        print("   partidas", n,
              "  gana", pct(Float64(base.wins) / Float64(n)),
              "  empata", pct(Float64(base.draws) / Float64(n)),
              "  pierde", pct(Float64(base.losses) / Float64(n)))
        print()
        print("   remata cuando puede ganar ya:   ",
              ratio(base.took_win, base.can_win))
        print("   bloquea cuando hay UNA amenaza: ",
              ratio(base.blocked, base.must_block))
        print()
        print("   turnos que ya llegaban con DOBLE amenaza (irremediables):",
              base.forked)
        print("   turnos en los que el agente REGALA una doble amenaza:  ",
              base.created_fork)
        if base.missed > 0:
            print("   masa media de q sobre la casilla que no bloqueo:",
                  base.q_missed / Scalar[dtype](base.missed))
            print("   (no es ~0: la ve y aun asi la descarta)")
        print()

        print("--- 2. la hipotesis, y su barrido ---")
        print("   Bloquear cuesta un turno. Con el descuento por profundidad")
        print("   gamma_r, ganar un turno mas tarde vale gamma_r veces menos, y")
        print("   el softmax a temperatura tau convierte esa diferencia en un")
        print("   factor e^(delta/tau). Con gamma_r=0.7 y tau=0.02 el bloqueo")
        print("   sale ~1500 veces peor que rematar. Perder, en cambio, vale 0:")
        print("   lo MISMO que una partida sin resolver.")
        print()
        print("   Si eso es cierto, subir tau (menos max, mas media) o subir")
        print("   gamma_r (menos prisa) tiene que subir el bloqueo.")
        print()

        taus = List[Float64]()
        taus.append(0.02); taus.append(0.1); taus.append(0.3)
        gammas = List[Float64]()
        gammas.append(0.7); gammas.append(0.9); gammas.append(1.0)

        for gi in range(len(gammas)):
            for ti in range(len(taus)):
                a = audit(ctx, Scalar[dtype](taus[ti]),
                          Scalar[dtype](gammas[gi]), SWEEP_STEPS)
                row(a)
            print()

        print("--- 3. el arreglo directo: que perder CUESTE ---")
        print("   El barrido de arriba solo mueve el compromiso entre prisa y")
        print("   prudencia, porque no toca la causa: perder vale 0, lo mismo")
        print("   que no resolver. Con loss_penalty una derrota vale -p, que es")
        print("   el convenio +1/0/-1 de los juegos. Es modelado de recompensa")
        print("   DENTRO de la busqueda; el marcador sigue contando igual.")
        print()
        pens = List[Float64]()
        pens.append(0.0); pens.append(0.5); pens.append(1.0); pens.append(3.0)
        for gi in range(len(gammas)):
            for pi in range(len(pens)):
                a = audit(ctx, TEMPERATURE, Scalar[dtype](gammas[gi]),
                          SWEEP_STEPS, Scalar[dtype](pens[pi]))
                row(a)
            print()

        print("--- 4. ¿es falta de particulas o es el estimador? ---")
        print("   Con 64 particulas y hasta 9 acciones raiz, cada accion recibe")
        print("   ~7 muestras: poco. Si el bloqueo fuera un problema de RUIDO,")
        print("   subir N tiene que arreglarlo. Si es SESGO del estimador, no.")
        print()
        ns = List[Int]()
        # The cap is TPB_PARTICLES=128: the per-env kernels use one block.
        ns.append(16); ns.append(32); ns.append(64); ns.append(128)
        for ni in range(len(ns)):
            a = audit(ctx, TEMPERATURE, REWARD_GAMMA, SWEEP_STEPS_BIG,
                      Scalar[dtype](0), ns[ni])
            row(a)

        print()
        print("=== 5. la causa real, y el arreglo ===")
        print("   La sonda destapo algo que el razonamiento no habia visto: tras")
        print("   la busqueda los pesos finales son TODOS CERO. Es correcto -")
        print("   resampling.mojo:151 los pone a cero al remuestrear, porque en")
        print("   SMC la informacion del peso pasa a la MULTIPLICIDAD de las")
        print("   particulas. Con profundidad 6 y periodo 3 el ultimo remuestreo")
        print("   cae casi al final, asi que el readout acaba leyendo el")
        print("   histograma de cuentas y nada mas.")
        print()
        print("   Y ahi esta el fallo de verdad: al remuestrear, una particula que")
        print("   PIERDE se mata, pero su hueco lo rellena una COPIA de otra")
        print("   particula con peso alto, que puede tener la misma accion raiz.")
        print("   Matar perdedoras no le baja la cuenta a su accion. Por eso el")
        print("   castigo por derrota no hacia nada: la particula ya iba a morir.")
        print()
        print("   Dos cambios, y hacen falta LOS DOS:")
        print("     1. sin remuestreo, para que los pesos lleguen enteros al final")
        print("     2. promediar por accion antes de exponenciar, para que una")
        print("        derrota arrastre a su accion en vez de solo desaparecer")
        print()
        gs = List[Float64](); ps = List[Float64]()
        es = List[Float64](); pe = List[Int]()
        gs.append(0.7); ps.append(0.0); es.append(0.0); pe.append(3)    # la base
        gs.append(0.7); ps.append(0.0); es.append(0.0); pe.append(99)
        gs.append(0.7); ps.append(0.0); es.append(1.0); pe.append(99)
        gs.append(1.0); ps.append(0.0); es.append(1.0); pe.append(99)
        gs.append(1.0); ps.append(1.0); es.append(1.0); pe.append(99)
        gs.append(0.9); ps.append(1.0); es.append(1.0); pe.append(99)
        for i in range(len(gs)):
            a = audit(ctx, TEMPERATURE, Scalar[dtype](gs[i]), SWEEP_STEPS,
                      Scalar[dtype](ps[i]), NUM_PARTICLES, es[i] > 0.5, pe[i])
            row(a)

        print()
        print("=== 6. ¿hasta donde llega? ===")
        print("   Con el estimador de SUMA de exponenciales, subir N saturaba")
        print("   (28.8 -> 40.0 -> 36.4): era SESGO, no ruido. Con la MEDIA el")
        print("   sesgo desaparece y lo que queda es varianza, asi que ahora N SI")
        print("   tiene que ayudar. Es la prueba definitiva de que el diagnostico")
        print("   era correcto.")
        print()
        ns2 = List[Int]()
        ns2.append(16); ns2.append(32); ns2.append(64); ns2.append(128)
        ns2.append(256); ns2.append(512)
        for i in range(len(ns2)):
            a = audit(ctx, TEMPERATURE, Scalar[dtype](0.9), SWEEP_STEPS,
                      Scalar[dtype](1.0), ns2[i], True, 99)
            row(a)
        print()
        print()
        print("   La temperatura NO se barre aqui a proposito: con lectura")
        print("   codiciosa q = softmax(media/tau) y argmax(softmax(x/tau)) =")
        print("   argmax(x) para cualquier tau > 0, asi que no puede cambiar la")
        print("   jugada. Si importaba con el readout de SPO, porque una SUMA de")
        print("   exponenciales no es invariante de escala. Volvera a importar")
        print("   cuando la lectura vuelva a muestrear, en el M-step.")

        print()
        print("=== 7. confirmacion, tanda larga ===")
        print("   El montaje ganador con", STEPS, "turnos, contra la base en las")
        print("   mismas condiciones. Referencias exactas por recursion:")
        print("   el juego optimo pierde 0.00% y saca score 0.9974; al azar,")
        print("   28.81% y 0.6484.")
        print()
        final = audit(ctx, TEMPERATURE, Scalar[dtype](0.9), STEPS,
                      Scalar[dtype](1.0), 512, True, 99)
        row(base)
        row(final)
        nf = final.games()
        print()
        print("   derrotas", pct(Float64(base.losses) / Float64(n)), "->",
              pct(Float64(final.losses) / Float64(nf)),
              "   (", base.losses, "->", final.losses, "partidas perdidas )")
        if wilson_hi(final.losses, nf) < wilson_lo(base.losses, n):
            print("   los intervalos NO se solapan: la mejora es real.")
        else:
            print("   los intervalos se solapan: no se afirma mejora.")
