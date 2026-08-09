"""The benchmark's metrics: formatting, Wilson intervals and the CSV row.

No GPU: it is all host arithmetic. It matters that it be exact because these
numbers are the ones compared against the MCTS, and a different formatting would
make the two CSVs unreadable together.

The Wilson intervals are checked against values computed separately with the
reference formula, not against what this same implementation returns.
"""

from std.math import abs

from bench.metrics import (PlannerMetrics, CSV_HEADER, fmt_fixed, wilson_lo,
                           wilson_hi, rate_with_ci)


def check(got: Float64, want: Float64, tol: Float64, what: String) raises:
    if abs(got - want) > tol:
        raise Error(what, ": got=", got, " want=", want)


def test_fmt_fixed() raises:
    """The formatting has to match "%.Nf", rounding and negatives included."""
    if fmt_fixed(1.0, 3) != String("1.000"):
        raise Error("1.0 to 3 decimals: ", fmt_fixed(1.0, 3))
    if fmt_fixed(0.5, 2) != String("0.50"):
        raise Error("0.5 to 2 decimals: ", fmt_fixed(0.5, 2))
    # Leading zeros in the fractional part: the classic bug.
    if fmt_fixed(1.0005, 3) != String("1.001"):
        raise Error("1.0005 to 3 decimals: ", fmt_fixed(1.0005, 3))
    if fmt_fixed(2.03, 2) != String("2.03"):
        raise Error("2.03 to 2 decimals: ", fmt_fixed(2.03, 2))
    if fmt_fixed(-1.5, 1) != String("-1.5"):
        raise Error("negative: ", fmt_fixed(-1.5, 1))
    if fmt_fixed(0.0, 4) != String("0.0000"):
        raise Error("zero: ", fmt_fixed(0.0, 4))
    print("PASS fmt_fixed matches %.Nf")


def test_wilson_reference_values() raises:
    """95% Wilson against values computed with the reference formula.

    The two cases that matter are the extremes: at 50/50 the Wald interval would
    give zero width, and Wilson gives [0.9287, 1.0]. That is precisely the regime
    of a planner that almost always wins, that is, ours.
    """
    tol = 1e-6
    check(wilson_lo(50, 50), 0.928652, tol, "wilson_lo 50/50")
    check(wilson_hi(50, 50), 1.000000, tol, "wilson_hi 50/50")
    check(wilson_lo(0, 50), 0.000000, tol, "wilson_lo 0/50")
    check(wilson_hi(0, 50), 0.071348, tol, "wilson_hi 0/50")
    check(wilson_lo(25, 50), 0.366445, tol, "wilson_lo 25/50")
    check(wilson_hi(25, 50), 0.633555, tol, "wilson_hi 25/50")
    check(wilson_lo(968, 1000), 0.955175, tol, "wilson_lo 968/1000")
    check(wilson_hi(968, 1000), 0.977243, tol, "wilson_hi 968/1000")
    # And the degenerate case: with no games, there is no interval.
    check(wilson_lo(0, 0), 0.0, tol, "wilson_lo with no samples")
    check(wilson_hi(0, 0), 0.0, tol, "wilson_hi with no samples")
    print("PASS Wilson intervals against reference values")


def test_score_matches_the_exact_scale() raises:
    """The score (1 / 0.5 / 0) on the same scale as the exact references."""
    # Random play's exact proportions give 0.6484.
    m = PlannerMetrics(mode="test", games=10000, iterations=64, exploration=0.02,
                       seed=1, total_runtime_s=1.0, total_moves=0, decisions=0,
                       total_simulations=0, x_wins=5849, o_wins=2881, draws=1270)
    check(m.score(), 0.6484, 1e-9, "the exact random score")

    # All wins -> 1.0; all draws -> 0.5; all losses -> 0.
    allw = PlannerMetrics(mode="t", games=10, iterations=1, exploration=0.0, seed=0,
                          total_runtime_s=1.0, total_moves=0, decisions=0,
                          total_simulations=0, x_wins=10, o_wins=0, draws=0)
    alld = PlannerMetrics(mode="t", games=10, iterations=1, exploration=0.0, seed=0,
                          total_runtime_s=1.0, total_moves=0, decisions=0,
                          total_simulations=0, x_wins=0, o_wins=0, draws=10)
    alll = PlannerMetrics(mode="t", games=10, iterations=1, exploration=0.0, seed=0,
                          total_runtime_s=1.0, total_moves=0, decisions=0,
                          total_simulations=0, x_wins=0, o_wins=10, draws=0)
    check(allw.score(), 1.0, 1e-9, "all wins")
    check(alld.score(), 0.5, 1e-9, "all draws")
    check(alll.score(), 0.0, 1e-9, "all losses")
    print("PASS the score uses the 1 / 0.5 / 0 scale")


def test_csv_row_matches_the_shared_schema() raises:
    """The row has as many columns as the header, and in the same order.

    If this drifts, the MCTS's and the search's CSVs stop being readable together,
    which is the only reason for copying the schema.
    """
    m = PlannerMetrics(mode="smc_vs_random", games=1189, iterations=64,
                       exploration=0.02, seed=20260724, total_runtime_s=2.5,
                       total_moves=4000, decisions=3840,
                       total_simulations=1474560, x_wins=1151, o_wins=24, draws=14)
    row = m.to_csv_row()

    want_cols = len(String(CSV_HEADER).split(","))
    got_cols = len(row.split(","))
    if got_cols != want_cols:
        raise Error("the row has ", got_cols, " columns and the header ",
                    want_cols)

    # The first column distinguishes the platform: the MCTS writes "mojo" (CPU).
    if not row.startswith("mojo-gpu,smc_vs_random,1189,64,"):
        raise Error("the start of the row is not the expected one: ", row)
    print("PASS the CSV row matches the shared schema (", want_cols, "columns )")


def test_derived_rates() raises:
    """The derived columns are computed from the raw ones, and without dividing by zero."""
    m = PlannerMetrics(mode="t", games=100, iterations=64, exploration=0.02, seed=1,
                       total_runtime_s=2.0, total_moves=350, decisions=200,
                       total_simulations=76800, x_wins=96, o_wins=2, draws=2)
    check(m.simulations_per_second(), 38400.0, 1e-6, "steps per second")
    check(m.avg_decision_time_s(), 0.01, 1e-9, "time per decision")

    # With no decisions and no time, it must not blow up.
    z = PlannerMetrics(mode="t", games=0, iterations=1, exploration=0.0, seed=0,
                       total_runtime_s=0.0, total_moves=0, decisions=0,
                       total_simulations=0, x_wins=0, o_wins=0, draws=0)
    _ = z.simulations_per_second()
    _ = z.avg_decision_time_s()
    check(z.score(), 0.0, 1e-9, "score with no games")
    print("PASS derived rates and degenerate cases")


def main() raises:
    test_fmt_fixed()
    test_wilson_reference_values()
    test_score_matches_the_exact_scale()
    test_csv_row_matches_the_shared_schema()
    test_derived_rates()
