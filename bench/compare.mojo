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
        raise Error("csv ", path, " has no data row")
    c = lines[1].split(",")
    if len(c) < 17:
        raise Error("csv ", path, " has ", len(c), " columns, expected 17")

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
          "  wins ", pct(r.rate(r.x_wins)),
          " [", fmt_fixed(wilson_lo(r.x_wins, r.games), 3), ",",
          fmt_fixed(wilson_hi(r.x_wins, r.games), 3), "]",
          "  draws ", pct(r.rate(r.draws)),
          "  loses ", pct(r.rate(r.o_wins)),
          "  score ", fmt_fixed(r.score(), 4),
          "  (", pct(frac), " of the random->optimal path )")


def main() raises:
    args = argv()
    if len(args) < 3:
        raise Error("usage: compare.mojo <csv_mcts> <csv_smc>")
    mcts = read_row(String(args[1]))
    smc = read_row(String(args[2]))

    print("=== TIC-TAC-TOE: two planners, same game, same opponent ===")
    print()
    print("--- 1. strength (against a random opponent) ---")
    print("   exact reference, random play         wins 58.49%  draws 12.70%  loses 28.81%  score 0.6484")
    show_strength("MCTS  (" + mcts.language + ")", mcts)
    show_strength("SMC   (" + smc.language + ")", smc)
    print("   exact reference, optimal play        wins 99.48%  draws  0.52%  loses  0.00%  score 0.9974")
    print()
    print("   Both references are computed by recursion over every state, not")
    print("   estimated. Optimal play NEVER loses, so the losses column measures")
    print("   directly the threats each planner failed to see.")

    print()
    print("--- 2. how much work each one does per decision ---")
    print("   MCTS :", mcts.iterations, "simulations/decision, tree of",
          mcts.nodes, "nodes in total")
    print("   SMC  :", smc.iterations, "particles/decision, no tree (",
          smc.nodes, "nodes )")
    print()
    print("   It is not the same unit, which is why they are not divided: MCTS")
    print("   spends its budget building a tree that refines the moves that")
    print("   follow; the SMC search spends it on trajectories that score only")
    print("   the first. That is where the strength gap is, and also why one")
    print("   needs to pack the board into bitboards and the other does not.")

    print()
    print("--- 3. performance ---")
    print("   MCTS   time/decision", fmt_fixed(mcts.time_per_decision * 1e6, 1),
          "us   (serial on CPU: this is latency)")
    print("   SMC    time/decision", fmt_fixed(smc.time_per_decision * 1e6, 1),
          "us   (batched on GPU: this is throughput)")
    print("   x", fmt_fixed(mcts.time_per_decision / smc.time_per_decision, 1),
          "in favour of the search in decisions per second.")
    print()
    print("   But measured at LATENCY (one game at a time) the search sits at")
    print("   ~330 us, the same order as MCTS. Its advantage is not deciding")
    print("   faster, it is deciding for 64 games at once.")
