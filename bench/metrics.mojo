"""Metricas del benchmark: mismo esquema CSV que la implementacion MCTS.

Las columnas son EXACTAMENTE las de `MCTS-mojo-tictactoe/src/metrics.mojo`, en el
mismo orden, para que los dos CSV se puedan concatenar y comparar fila a fila. El
esquema esta pensado para MCTS, asi que hay que decir con que se corresponde cada
columna en una busqueda SMC:

    columna              MCTS                        busqueda SMC
    -------------------  --------------------------  --------------------------
    mode                 mcts_vs_random              smc_vs_random
    iterations           simulaciones por decision   particulas por decision
    exploration          constante UCT               temperatura
    total_simulations    rollouts                    pasos de particula
                                                     (particulas x profundidad)
    total_nodes          nodos del arbol             0: la busqueda SMC no
                                                     construye arbol
    mcts_decisions       decisiones tomadas          idem (turnos del agente)

`total_nodes = 0` no es un hueco por rellenar: es el dato. La diferencia estructural
entre los dos planificadores es justo esa -- uno acumula un arbol y el otro pisa un
conjunto de particulas -- y por eso uno necesita empaquetar el tablero en bitboards
y el otro no.

Sobre el tiempo, la comparacion honesta necesita DOS numeros y no uno:

  * LATENCIA: lo que tarda UNA decision aislada. Es lo que mide el MCTS, que juega
    partidas en serie en la CPU.
  * THROUGHPUT: decisiones por segundo con el lote entero. La busqueda planifica
    para 64 partidas a la vez en la GPU, asi que su coste por decision repartido es
    mucho menor que su latencia.

Comparar el throughput de la GPU contra la latencia de la CPU seria hacer trampa, y
comparar solo latencias esconderia justo la ventaja del enfoque por lotes.
"""

from std.math import sqrt

# El orden canonico de columnas, copiado literal del MCTS para poder concatenar.
comptime CSV_HEADER: StaticString = (
    "language,mode,games,iterations,exploration,seed,"
    "total_runtime_s,total_moves,mcts_decisions,total_simulations,"
    "total_nodes,x_wins,o_wins,draws,"
    "simulations_per_second,nodes_per_second,avg_decision_time_s"
)


def fmt_fixed(value: Float64, decimals: Int) -> String:
    """Formatea con exactamente `decimals` decimales (el "%.Nf" de printf).

    Con aritmetica entera sobre el valor escalado, igual que en el MCTS, para que
    las dos implementaciones escriban los numeros de la misma forma.
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
    """Extremo inferior del intervalo de Wilson al 95%."""
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
    """Extremo superior del intervalo de Wilson al 95%.

    Wilson y no el Wald de manual (p +- z*sqrt(p(1-p)/n)), que en p=0 o p=1 da un
    intervalo de ancho cero -- justo el regimen de un planificador que gana casi
    siempre al azar. Wilson se queda dentro de [0,1] y mantiene la cobertura.
    Se devuelve en dos funciones porque una tupla de Float64 como valor de retorno
    da problemas en 1.0.0b1 (ver docs/api_notes.md).
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
    """`(tasa, 95% CI [lo, hi])`, el mismo formato que el resumen del MCTS."""
    safe = n if n > 0 else 1
    rate = Float64(successes) / Float64(safe)
    return ("(" + fmt_fixed(rate, 3) + ", 95% CI ["
            + fmt_fixed(wilson_lo(successes, n), 3) + ", "
            + fmt_fixed(wilson_hi(successes, n), 3) + "])")


@fieldwise_init
struct PlannerMetrics(Copyable, Movable):
    """Una tanda de partidas de un planificador, en el esquema comun."""

    var mode: String
    """smc_vs_random, para distinguirlo del mcts_vs_random del MCTS."""

    var games: Int
    var iterations: Int
    """Particulas por decision: el analogo de las iteraciones del MCTS."""

    var exploration: Float64
    """La temperatura de la busqueda, en la columna de la constante UCT."""

    var seed: Int
    var total_runtime_s: Float64
    var total_moves: Int
    var decisions: Int
    var total_simulations: Int
    """Pasos de particula: particulas x profundidad x decisiones."""

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
        """Coste por decision REPARTIDO: con lotes, muchas decisiones salen a la
        vez, asi que esto es throughput y no latencia. Ver la cabecera."""
        d = self.decisions if self.decisions > 0 else 1
        return self.total_runtime_s / Float64(d)

    def score(self) -> Float64:
        """1 victoria, 0.5 empate, 0 derrota. Comparable con el 0.6484 del azar y
        el 0.9974 del juego optimo, los dos calculados exactamente."""
        return (Float64(self.x_wins) + 0.5 * Float64(self.draws)) \
               / Float64(self.games_or_one())

    def to_csv_row(self) -> String:
        cols = List[String]()
        cols.append("mojo-gpu")          # language: distingue del "mojo" (CPU) del MCTS
        cols.append(self.mode)
        cols.append(String(self.games))
        cols.append(String(self.iterations))
        cols.append(fmt_fixed(self.exploration, 6))
        cols.append(String(self.seed))
        cols.append(fmt_fixed(self.total_runtime_s, 6))
        cols.append(String(self.total_moves))
        cols.append(String(self.decisions))
        cols.append(String(self.total_simulations))
        cols.append("0")                 # total_nodes: la busqueda SMC no tiene arbol
        cols.append(String(self.x_wins))
        cols.append(String(self.o_wins))
        cols.append(String(self.draws))
        cols.append(fmt_fixed(self.simulations_per_second(), 6))
        cols.append(fmt_fixed(0.0, 6))   # nodes_per_second: sin arbol, no aplica
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
    """Escribe UNA fila, empezando el fichero de cero."""
    rows = List[PlannerMetrics]()
    rows.append(metrics)
    write_csv_rows(rows, path)


def write_csv_rows(rows: List[PlannerMetrics], path: String) raises:
    """Escribe VARIAS filas de una tanda, con la cabecera delante.

    Una tanda puede necesitar mas de una fila porque el esquema de 17 columnas solo
    tiene un hueco para el tiempo (`avg_decision_time_s`) y la comparacion del
    Milestone 4 necesita dos numeros distintos: la latencia de una decision suelta y
    el coste repartido de un lote. Meterlos en la misma fila obligaria a cambiar el
    esquema y romperia la concatenacion con el CSV del MCTS, que ya esta escrito.
    Van como filas separadas, distinguidas por la etiqueta `mode`.
    """
    text = String(CSV_HEADER) + "\n"
    for m in rows:
        text += m.to_csv_row() + "\n"
    with open(path, "w") as out:
        out.write(text)
