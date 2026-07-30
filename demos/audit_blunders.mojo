"""¿Como puede perder? Auditoria jugada a jugada del planificador.

El juego optimo contra un rival aleatorio pierde el 0.00% de las partidas. El
nuestro pierde ~2%. Este demo no mide: DIAGNOSTICA. En cada turno del agente mira
el tablero desde fuera, con reglas de tres en raya escritas a mano en el host, y
clasifica la posicion antes de dejar jugar a la busqueda:

    puede ganar ya          hay una casilla que le hace tres en raya
    tiene que bloquear      el rival tiene dos en raya y una casilla libre
    doble amenaza          el rival tiene DOS casillas ganadoras -> ya no hay
                            bloqueo posible, el error fue antes
    tranquila               ni una cosa ni la otra

Y despues compara con lo que jugo la busqueda. Asi se separa lo que son dos fallos
completamente distintos:

  - **fallo tactico**: habia una amenaza a un solo movimiento y no la bloqueo.
    Eso significa que la busqueda no vio algo que estaba a profundidad 1.
  - **fallo estrategico**: llego a una posicion con doble amenaza, donde ya no se
    puede hacer nada. El error real fue varias jugadas antes, al permitir la
    horquilla.

La diferencia importa mucho para saber que arreglar. Si son fallos tacticos, el
problema es de resolucion de la busqueda (particulas, temperatura, readout). Si
son horquillas, el problema es que la busqueda maximiza el resultado ESPERADO
contra un rival aleatorio y una horquilla solo se castiga en las pocas particulas
donde el rival aleatorio acierta a explotarla.

Ademas, en las posiciones donde falla, se imprime cuanta masa de q le dio la
busqueda a la jugada correcta. Eso distingue "no lo vio" de "lo vio y aun asi
eligio otra".
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
from systems.spo.readout import readout_greedy

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

# Las 8 lineas, como en el kernel del entorno pero aqui en el host.
comptime LINES = 8


def line_cell(line: Int, k: Int) -> Int:
    """La casilla k (0..2) de la linea `line` (0..7). Filas, columnas, diagonales."""
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
    """Las casillas que le dan la victoria INMEDIATA a `player`.

    Una casilla vale si hay una linea con dos fichas de `player` y esa casilla
    vacia. Se devuelven todas y sin repetir, porque el numero de casillas
    distintas es justo lo que separa "amenaza simple" (bloqueable) de "horquilla"
    (ya perdida).
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
    """Lo que sale de una auditoria."""
    var temperature: Scalar[dtype]
    var reward_gamma: Scalar[dtype]
    var loss_penalty: Scalar[dtype]
    var particles: Int
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
          num_particles: Int = NUM_PARTICLES) raises -> Audit:
    """Juega y clasifica cada turno del agente. Ver la cabecera del fichero."""
    cfg = SPOConfig(num_envs=ENVS, num_particles=num_particles,
                    num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                    search_depth=SEARCH_DEPTH, resample_period=RESAMPLE_PERIOD,
                    temperature=temperature, search_gamma=1.0,
                    search_gae_lambda=1.0)
    model = TicTacToe(reward_gamma, loss_penalty)
    ws = SearchWorkspace(ctx, cfg)
    q_buf = zero_buffer[dtype](ctx, ENVS * NUM_ACTIONS)
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
    return Audit(temperature, reward_gamma, loss_penalty, num_particles, wins,
                 draws, losses, can_win,
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
        # El tope es TPB_PARTICLES=128: los kernels por env usan un bloque.
        ns.append(16); ns.append(32); ns.append(64); ns.append(128)
        for ni in range(len(ns)):
            a = audit(ctx, TEMPERATURE, REWARD_GAMMA, SWEEP_STEPS_BIG,
                      Scalar[dtype](0), ns[ni])
            row(a)
