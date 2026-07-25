"""Las metricas del benchmark: formato, intervalos de Wilson y fila CSV.

Sin GPU: es todo aritmetica de host. Importa que sea exacto porque estos numeros
son los que se comparan contra el MCTS, y un formateo distinto haria que los dos
CSV no se pudieran leer juntos.

Los intervalos de Wilson estan contra valores calculados aparte con la formula de
referencia, no contra lo que devuelve esta misma implementacion.
"""

from std.math import abs

from bench.metrics import (PlannerMetrics, CSV_HEADER, fmt_fixed, wilson_lo,
                           wilson_hi, rate_with_ci)


def check(got: Float64, want: Float64, tol: Float64, what: String) raises:
    if abs(got - want) > tol:
        raise Error(what, ": got=", got, " want=", want)


def test_fmt_fixed() raises:
    """El formateo tiene que coincidir con "%.Nf", incluidos redondeo y negativos."""
    if fmt_fixed(1.0, 3) != String("1.000"):
        raise Error("1.0 con 3 decimales: ", fmt_fixed(1.0, 3))
    if fmt_fixed(0.5, 2) != String("0.50"):
        raise Error("0.5 con 2 decimales: ", fmt_fixed(0.5, 2))
    # Ceros a la izquierda en la parte fraccionaria: el fallo clasico.
    if fmt_fixed(1.0005, 3) != String("1.001"):
        raise Error("1.0005 con 3 decimales: ", fmt_fixed(1.0005, 3))
    if fmt_fixed(2.03, 2) != String("2.03"):
        raise Error("2.03 con 2 decimales: ", fmt_fixed(2.03, 2))
    if fmt_fixed(-1.5, 1) != String("-1.5"):
        raise Error("negativo: ", fmt_fixed(-1.5, 1))
    if fmt_fixed(0.0, 4) != String("0.0000"):
        raise Error("cero: ", fmt_fixed(0.0, 4))
    print("PASS fmt_fixed coincide con %.Nf")


def test_wilson_reference_values() raises:
    """Wilson al 95% contra valores calculados con la formula de referencia.

    Los dos casos que importan son los extremos: con 50/50 el intervalo de Wald
    daria ancho cero, y Wilson da [0.9287, 1.0]. Ese es justo el regimen de un
    planificador que gana casi siempre, o sea el nuestro.
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
    # Y el caso degenerado: sin partidas, no hay intervalo.
    check(wilson_lo(0, 0), 0.0, tol, "wilson_lo sin muestras")
    check(wilson_hi(0, 0), 0.0, tol, "wilson_hi sin muestras")
    print("PASS intervalos de Wilson contra valores de referencia")


def test_score_matches_the_exact_scale() raises:
    """La puntuacion (1 / 0.5 / 0) en la misma escala que las referencias exactas."""
    # Las proporciones exactas del juego al azar dan 0.6484.
    m = PlannerMetrics(mode="test", games=10000, iterations=64, exploration=0.02,
                       seed=1, total_runtime_s=1.0, total_moves=0, decisions=0,
                       total_simulations=0, x_wins=5849, o_wins=2881, draws=1270)
    check(m.score(), 0.6484, 1e-9, "la puntuacion del azar exacto")

    # Todo victorias -> 1.0; todo empates -> 0.5; todo derrotas -> 0.
    allw = PlannerMetrics(mode="t", games=10, iterations=1, exploration=0.0, seed=0,
                          total_runtime_s=1.0, total_moves=0, decisions=0,
                          total_simulations=0, x_wins=10, o_wins=0, draws=0)
    alld = PlannerMetrics(mode="t", games=10, iterations=1, exploration=0.0, seed=0,
                          total_runtime_s=1.0, total_moves=0, decisions=0,
                          total_simulations=0, x_wins=0, o_wins=0, draws=10)
    alll = PlannerMetrics(mode="t", games=10, iterations=1, exploration=0.0, seed=0,
                          total_runtime_s=1.0, total_moves=0, decisions=0,
                          total_simulations=0, x_wins=0, o_wins=10, draws=0)
    check(allw.score(), 1.0, 1e-9, "todo victorias")
    check(alld.score(), 0.5, 1e-9, "todo empates")
    check(alll.score(), 0.0, 1e-9, "todo derrotas")
    print("PASS la puntuacion usa la escala 1 / 0.5 / 0")


def test_csv_row_matches_the_shared_schema() raises:
    """La fila tiene tantas columnas como la cabecera, y en el mismo orden.

    Si esto se descuadra, los CSV del MCTS y de la busqueda dejan de poder leerse
    juntos, que es el unico motivo de copiar el esquema.
    """
    m = PlannerMetrics(mode="smc_vs_random", games=1189, iterations=64,
                       exploration=0.02, seed=20260724, total_runtime_s=2.5,
                       total_moves=4000, decisions=3840,
                       total_simulations=1474560, x_wins=1151, o_wins=24, draws=14)
    row = m.to_csv_row()

    want_cols = len(String(CSV_HEADER).split(","))
    got_cols = len(row.split(","))
    if got_cols != want_cols:
        raise Error("la fila tiene ", got_cols, " columnas y la cabecera ",
                    want_cols)

    # La primera columna distingue plataforma: el MCTS escribe "mojo" (CPU).
    if not row.startswith("mojo-gpu,smc_vs_random,1189,64,"):
        raise Error("el comienzo de la fila no es el esperado: ", row)
    print("PASS la fila CSV cuadra con el esquema comun (", want_cols, "columnas )")


def test_derived_rates() raises:
    """Las columnas derivadas se calculan de las crudas, y sin dividir por cero."""
    m = PlannerMetrics(mode="t", games=100, iterations=64, exploration=0.02, seed=1,
                       total_runtime_s=2.0, total_moves=350, decisions=200,
                       total_simulations=76800, x_wins=96, o_wins=2, draws=2)
    check(m.simulations_per_second(), 38400.0, 1e-6, "pasos por segundo")
    check(m.avg_decision_time_s(), 0.01, 1e-9, "tiempo por decision")

    # Sin decisiones ni tiempo, no puede explotar.
    z = PlannerMetrics(mode="t", games=0, iterations=1, exploration=0.0, seed=0,
                       total_runtime_s=0.0, total_moves=0, decisions=0,
                       total_simulations=0, x_wins=0, o_wins=0, draws=0)
    _ = z.simulations_per_second()
    _ = z.avg_decision_time_s()
    check(z.score(), 0.0, 1e-9, "puntuacion sin partidas")
    print("PASS tasas derivadas y casos degenerados")


def main() raises:
    test_fmt_fixed()
    test_wilson_reference_values()
    test_score_matches_the_exact_scale()
    test_csv_row_matches_the_shared_schema()
    test_derived_rates()
