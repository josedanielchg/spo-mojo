"""Benchmark metrics: the same CSV schema as the MCTS implementation.

The columns are EXACTLY those of `MCTS-mojo-tictactoe/src/metrics.mojo`, in the
same order, so that the two CSVs can be concatenated and compared row by row. The
schema was designed for MCTS, so what each column corresponds to in an SMC search
has to be spelled out:

    column               MCTS                        SMC search
    -------------------  --------------------------  --------------------------
    mode                 mcts_vs_random              smc_vs_random
    iterations           simulations per decision    particles per decision
    exploration          UCT constant                temperature
    total_simulations    rollouts                    particle steps
                                                     (particles x depth)
    total_nodes          tree nodes                  0: the SMC search builds no
                                                     tree
    mcts_decisions       decisions taken             idem (agent turns)

`total_nodes = 0` is not a blank to be filled in: it is the datum. The structural
difference between the two planners is exactly that -- one accumulates a tree and
the other overwrites a set of particles -- and that is why one needs to pack the
board into bitboards and the other does not.

On timing, an honest comparison needs TWO numbers and not one:

  * LATENCY: how long ONE isolated decision takes. It is what the MCTS measures,
    since it plays games serially on the CPU.
  * THROUGHPUT: decisions per second with the whole batch. The search plans for 64
    games at once on the GPU, so its amortised cost per decision is far lower than
    its latency.

Comparing the GPU's throughput against the CPU's latency would be cheating, and
comparing only latencies would hide precisely the batched approach's advantage.
"""

from std.math import sqrt

# The canonical column order, copied verbatim from the MCTS so as to be able to
# concatenate.
comptime CSV_HEADER: StaticString = (
    "language,mode,games,iterations,exploration,seed,"
    "total_runtime_s,total_moves,mcts_decisions,total_simulations,"
    "total_nodes,x_wins,o_wins,draws,"
    "simulations_per_second,nodes_per_second,avg_decision_time_s"
)


def fmt_fixed(value: Float64, decimals: Int) -> String:
    """Formats with exactly `decimals` decimals (printf's "%.Nf").

    With integer arithmetic on the scaled value, just as in the MCTS, so that both
    implementations write the numbers the same way.
    """
    scale = Int64(1)
    for _ in range(decimals):
        scale *= 10
    negative = value < 0
    magnitude = -value if negative else value
    scaled = Int64(magnitude * Float64(scale) + 0.5)
    integer_part = scaled // scale
    fraction = scaled % scale

    digits = String(fraction)
    while digits.byte_length() < decimals:
        digits = "0" + digits
    sign = "-" if negative else ""
    return sign + String(integer_part) + "." + digits


def wilson_lo(successes: Int, n: Int) -> Float64:
    """Lower end of the 95% Wilson interval."""
    if n <= 0:
        return 0.0
    z = 1.959963984540054
    nf = Float64(n)
    p = Float64(successes) / nf
    z2 = z * z
    denom = 1.0 + z2 / nf
    center = (p + z2 / (2.0 * nf)) / denom
    margin = z / denom * sqrt(p * (1.0 - p) / nf + z2 / (4.0 * nf * nf))
    return center - margin


def wilson_hi(successes: Int, n: Int) -> Float64:
    """Upper end of the 95% Wilson interval.

    Wilson and not the textbook Wald (p +- z*sqrt(p(1-p)/n)), which at p=0 or p=1
    gives an interval of zero width -- precisely the regime of a planner that
    almost always beats random play. Wilson stays inside [0,1] and keeps its
    coverage. It is returned as two functions because a tuple of Float64 as a
    return value causes trouble in 1.0.0b1 (see docs/api_notes.md).
    """
    if n <= 0:
        return 0.0
    z = 1.959963984540054
    nf = Float64(n)
    p = Float64(successes) / nf
    z2 = z * z
    denom = 1.0 + z2 / nf
    center = (p + z2 / (2.0 * nf)) / denom
    margin = z / denom * sqrt(p * (1.0 - p) / nf + z2 / (4.0 * nf * nf))
    return center + margin


def rate_with_ci(successes: Int, n: Int) -> String:
    """`(rate, 95% CI [lo, hi])`, the same format as the MCTS's summary."""
    safe = n if n > 0 else 1
    rate = Float64(successes) / Float64(safe)
    return ("(" + fmt_fixed(rate, 3) + ", 95% CI ["
            + fmt_fixed(wilson_lo(successes, n), 3) + ", "
            + fmt_fixed(wilson_hi(successes, n), 3) + "])")


@fieldwise_init
struct PlannerMetrics(Copyable, Movable):
    """One batch of games from a planner, in the common schema."""

    var mode: String
    """smc_vs_random, to tell it apart from the MCTS's mcts_vs_random."""

    var games: Int
    var iterations: Int
    """Particles per decision: the analogue of the MCTS's iterations."""

    var exploration: Float64
    """The search's temperature, in the UCT constant's column."""

    var seed: Int
    var total_runtime_s: Float64
    var total_moves: Int
    var decisions: Int
    var total_simulations: Int
    """Particle steps: particles x depth x decisions."""

    var x_wins: Int
    var o_wins: Int
    var draws: Int

    def games_or_one(self) -> Int:
        return self.games if self.games > 0 else 1

    def runtime_or_eps(self) -> Float64:
        return self.total_runtime_s if self.total_runtime_s > 0 else 1e-9

    def simulations_per_second(self) -> Float64:
        return Float64(self.total_simulations) / self.runtime_or_eps()

    def avg_decision_time_s(self) -> Float64:
        """AMORTISED cost per decision: with batching, many decisions come out at
        once, so this is throughput and not latency. See the header."""
        d = self.decisions if self.decisions > 0 else 1
        return self.total_runtime_s / Float64(d)

    def score(self) -> Float64:
        """1 win, 0.5 draw, 0 loss. Comparable with random play's 0.6484 and
        optimal play's 0.9974, both computed exactly."""
        return (Float64(self.x_wins) + 0.5 * Float64(self.draws)) \
               / Float64(self.games_or_one())

    def to_csv_row(self) -> String:
        cols = List[String]()
        cols.append("mojo-gpu")          # language: tells it apart from the MCTS's "mojo" (CPU)
        cols.append(self.mode)
        cols.append(String(self.games))
        cols.append(String(self.iterations))
        cols.append(fmt_fixed(self.exploration, 6))
        cols.append(String(self.seed))
        cols.append(fmt_fixed(self.total_runtime_s, 6))
        cols.append(String(self.total_moves))
        cols.append(String(self.decisions))
        cols.append(String(self.total_simulations))
        cols.append("0")                 # total_nodes: the SMC search has no tree
        cols.append(String(self.x_wins))
        cols.append(String(self.o_wins))
        cols.append(String(self.draws))
        cols.append(fmt_fixed(self.simulations_per_second(), 6))
        cols.append(fmt_fixed(0.0, 6))   # nodes_per_second: no tree, not applicable
        cols.append(fmt_fixed(self.avg_decision_time_s(), 6))
        return String(",").join(cols)

    def summary(self) -> String:
        lines = List[String]()
        lines.append("=== SMC search Tic-Tac-Toe Benchmark ===")
        lines.append("language        : mojo-gpu")
        lines.append("mode            : " + self.mode)
        lines.append("games           : " + String(self.games))
        lines.append("particles       : " + String(self.iterations))
        lines.append("temperature     : " + fmt_fixed(self.exploration, 6))
        lines.append("seed            : " + String(self.seed))
        lines.append("--- results (95% Wilson CI over " + String(self.games)
                     + " games) ---")
        lines.append("X wins          : " + String(self.x_wins) + " "
                     + rate_with_ci(self.x_wins, self.games))
        lines.append("O wins          : " + String(self.o_wins) + " "
                     + rate_with_ci(self.o_wins, self.games))
        lines.append("draws           : " + String(self.draws) + " "
                     + rate_with_ci(self.draws, self.games))
        lines.append("score           : " + fmt_fixed(self.score(), 4)
                     + "   (random 0.6484, optimal 0.9974, both exact)")
        lines.append("total moves     : " + String(self.total_moves))
        lines.append("--- performance ---")
        lines.append("total runtime   : " + fmt_fixed(self.total_runtime_s, 6) + " s")
        lines.append("decisions       : " + String(self.decisions))
        lines.append("particle steps  : " + String(self.total_simulations))
        lines.append("steps/second    : " + fmt_fixed(self.simulations_per_second(), 2))
        lines.append("time/decision   : " + fmt_fixed(self.avg_decision_time_s(), 6)
                     + " s  (batched: throughput, not latency)")
        return String("\n").join(lines)


def write_csv(metrics: PlannerMetrics, path: String) raises:
    """Writes ONE row, starting the file from scratch."""
    rows = List[PlannerMetrics]()
    rows.append(metrics)
    write_csv_rows(rows, path)


def write_csv_rows(rows: List[PlannerMetrics], path: String) raises:
    """Writes SEVERAL rows from one batch, with the header in front.

    A batch may need more than one row because the 17-column schema has only one
    slot for time (`avg_decision_time_s`) and Milestone 4's comparison needs two
    different numbers: the latency of a single decision and a batch's amortised
    cost. Putting them in the same row would force a schema change and break the
    concatenation with the MCTS's CSV, which is already written. They go as
    separate rows, told apart by the `mode` label.
    """
    text = String(CSV_HEADER) + "\n"
    for m in rows:
        text += m.to_csv_row() + "\n"
    with open(path, "w") as out:
        out.write(text)
