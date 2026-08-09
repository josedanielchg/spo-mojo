"""Compares the two planners' CSVs and prints the table.

    ./run.sh bench/compare.mojo <csv_mcts> <csv_smc>

It reads the two rows (same 17-column schema) and puts them side by side against
tic-tac-toe's two EXACT references, computed by recursion over all states and not
measured:

    random X vs random O  : wins 58.49%  draws 12.70%  loses 28.81%  score 0.6484
    optimal X vs random O : wins 99.48%  draws  0.52%  loses  0.00%  score 0.9974

Having the ceiling changes how the comparison reads: without it, "96.8% wins" and
"99.0%" look almost the same; with it, one can see that one leaves 2.4% of the
possible improvement uncollected and the other next to nothing.

On timing, the table is deliberately cautious. The CSV's column is
`avg_decision_time_s`, and it does NOT mean the same thing in the two rows: the
MCTS plays the games serially on the CPU, so it is its real latency; the SMC search
plans for 64 games at once on the GPU, so it is amortised cost (throughput). The
search's latency is measured separately, with a single env, in
bench_tictactoe.mojo.
"""

from std.sys import argv

from bench.metrics import fmt_fixed, wilson_lo, wilson_hi

comptime RANDOM_SCORE = 0.6484
comptime OPTIMAL_SCORE = 0.9974


@fieldwise_init
struct Row(Copyable, Movable):
    """One row of the common CSV, with what the table needs."""
    var language: String
    var mode: String
    var games: Int
    var iterations: Int
    var runtime_s: Float64
    var decisions: Int
    var simulations: Int
    var nodes: Int
    var x_wins: Int
    var o_wins: Int
    var draws: Int
    var time_per_decision: Float64

    def score(self) -> Float64:
        n = self.games if self.games > 0 else 1
        return (Float64(self.x_wins) + 0.5 * Float64(self.draws)) / Float64(n)

    def rate(self, k: Int) -> Float64:
        n = self.games if self.games > 0 else 1
        return Float64(k) / Float64(n)


def read_row(path: String) raises -> Row:
    """The CSV's second line (the first is the header)."""
    text: String
    with open(path, "r") as f:
        text = f.read()

    lines = text.split("\n")
    if len(lines) < 2:
        raise Error("el csv ", path, " no tiene fila de datos")
    c = lines[1].split(",")
    if len(c) < 17:
        raise Error("el csv ", path, " tiene ", len(c), " columnas, esperaba 17")

    # `split` returns StringSlice, so it has to be wrapped in String/Int/Float64.
    return Row(language=String(c[0]), mode=String(c[1]),
               games=Int(String(c[2])), iterations=Int(String(c[3])),
               runtime_s=Float64(String(c[6])), decisions=Int(String(c[8])),
               simulations=Int(String(c[9])), nodes=Int(String(c[10])),
               x_wins=Int(String(c[11])), o_wins=Int(String(c[12])),
               draws=Int(String(c[13])),
               time_per_decision=Float64(String(c[16])))


def pct(v: Float64) -> String:
    return fmt_fixed(v * 100.0, 2) + "%"


def show_strength(label: String, r: Row) raises:
    """One line of the strength table, with the wins' Wilson interval."""
    frac = (r.score() - RANDOM_SCORE) / (OPTIMAL_SCORE - RANDOM_SCORE)
    print("  ", label,
          "  gana ", pct(r.rate(r.x_wins)),
          " [", fmt_fixed(wilson_lo(r.x_wins, r.games), 3), ",",
          fmt_fixed(wilson_hi(r.x_wins, r.games), 3), "]",
          "  empata ", pct(r.rate(r.draws)),
          "  pierde ", pct(r.rate(r.o_wins)),
          "  score ", fmt_fixed(r.score(), 4),
          "  (", pct(frac), " del camino azar->optimo )")


def main() raises:
    args = argv()
    if len(args) < 3:
        raise Error("uso: compare.mojo <csv_mcts> <csv_smc>")
    mcts = read_row(String(args[1]))
    smc = read_row(String(args[2]))

    print("=== TRES EN RAYA: dos planificadores, mismo juego, mismo rival ===")
    print()
    print("--- 1. fuerza (contra rival aleatorio) ---")
    print("   referencia exacta, jugar al azar     gana 58.49%  empata 12.70%  pierde 28.81%  score 0.6484")
    show_strength("MCTS  (" + mcts.language + ")", mcts)
    show_strength("SMC   (" + smc.language + ")", smc)
    print("   referencia exacta, juego optimo      gana 99.48%  empata  0.52%  pierde  0.00%  score 0.9974")
    print()
    print("   Las dos referencias se calculan por recursion sobre todos los estados,")
    print("   no se estiman. El juego optimo NUNCA pierde, asi que la columna de")
    print("   derrotas mide directamente las amenazas que cada planificador no vio.")

    print()
    print("--- 2. cuanto trabajo hace cada uno por decision ---")
    print("   MCTS :", mcts.iterations, "simulaciones/decision, arbol de",
          mcts.nodes, "nodos en total")
    print("   SMC  :", smc.iterations, "particulas/decision, sin arbol (",
          smc.nodes, "nodos )")
    print()
    print("   No es la misma unidad y por eso no se dividen: el MCTS gasta su")
    print("   presupuesto construyendo un arbol que refina las jugadas siguientes;")
    print("   la busqueda SMC lo gasta en trayectorias que solo puntuan la primera.")
    print("   Ahi esta la diferencia de fuerza, y tambien por que uno necesita")
    print("   empaquetar el tablero en bitboards y el otro no.")

    print()
    print("--- 3. rendimiento ---")
    print("   MCTS   tiempo/decision", fmt_fixed(mcts.time_per_decision * 1e6, 1),
          "us   (en serie en CPU: es latencia)")
    print("   SMC    tiempo/decision", fmt_fixed(smc.time_per_decision * 1e6, 1),
          "us   (por lotes en GPU: es throughput)")
    print("   x", fmt_fixed(mcts.time_per_decision / smc.time_per_decision, 1),
          "a favor de la busqueda en decisiones por segundo.")
    print()
    print("   Pero medido a LATENCIA (una partida a la vez) la busqueda esta en")
    print("   ~330 us, o sea en el mismo orden que el MCTS. Su ventaja no es")
    print("   decidir mas rapido, es decidir para 64 partidas de una vez.")
