"""Demo 1: el E-step en vivo.

Ensena lo que hace SPO antes de que exista ninguna red neuronal: partiendo de una
politica que no sabe nada (prior uniforme), la busqueda SMC produce una politica
mejorada q solo por simular. Es la Figura 1 del paper convertida en demo.

    ./run.sh demos/demo_estep.mojo

Imprime tres cosas y deja dos CSV en results/ con los numeros crudos:

  1. prior vs q          la mejora, que es el resultado principal
  2. ESS y entropia por profundidad   como se degrada la busqueda y como la
                                      recupera el resampling
  3. barridos de temperatura y de numero de particulas
                         eta controla lo agresiva que es la mejora,
                         N controla lo fiable que es la estimacion
"""

from std.gpu.host import DeviceContext
from std.math import log

from ops.common import dtype, idx_dtype
from envs.toy_chain import (default_toy_chain, ToyChain,
                            ACTION_BAD, ACTION_GOOD, NUM_ACTIONS, STATE_DIM)
from systems.spo.particles import SearchWorkspace
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from tests.helpers import upload, download

comptime SEED = UInt32(20260719)
comptime NUM_ENVS = 64
"""Bastantes envs: cada uno es una busqueda independiente desde el mismo estado,
asi que promediar sobre ellos da una medida estable sin repetir el experimento."""


@fieldwise_init
struct SearchStats(Movable):
    """Lo que se mide de una configuracion de busqueda."""
    var q_good: Scalar[dtype]
    """Masa que la politica mejorada pone en la accion buena, promediada."""
    var ess: List[Scalar[dtype]]
    """ESS medio por profundidad."""
    var entropy: List[Scalar[dtype]]
    """Entropia media de los pesos por profundidad."""


def run_search(ctx: DeviceContext, num_particles: Int,
               temperature: Scalar[dtype], depth: Int, period: Int,
               toy: ToyChain) raises -> SearchStats:
    """Corre una busqueda completa y resume el resultado."""
    cfg = SPOConfig(
        num_envs=NUM_ENVS, num_particles=num_particles, num_actions=NUM_ACTIONS,
        state_dim=STATE_DIM, search_depth=depth, resample_period=period,
        temperature=temperature, search_gamma=1.0, search_gae_lambda=1.0)

    ws = SearchWorkspace(ctx, cfg)

    root_state = List[Scalar[dtype]]()
    for _ in range(NUM_ENVS):
        root_state.append(0.0)      # todos en la casilla de salida

    search[ToyChain](ctx, ws, cfg, toy, upload[dtype](ctx, root_state), SEED)
    ctx.synchronize()

    p_total = cfg.num_search_particles()
    actions = download[idx_dtype](ws.output.sampled_actions, p_total)
    weights = download[dtype](ws.output.sampled_action_weights, p_total)

    # q(GOOD) = histograma de las acciones raiz ponderado por sus pesos.
    q_total = Scalar[dtype](0)
    for e in range(NUM_ENVS):
        for n in range(num_particles):
            p = e * num_particles + n
            if Int(actions[p]) == ACTION_GOOD:
                q_total += weights[p]
    q_good = q_total / Scalar[dtype](NUM_ENVS)

    ess_raw = download[dtype](ws.output.ess, depth * NUM_ENVS)
    ent_raw = download[dtype](ws.output.entropy, depth * NUM_ENVS)
    ess = List[Scalar[dtype]]()
    entropy = List[Scalar[dtype]]()
    for d in range(depth):
        se = Scalar[dtype](0)
        sh = Scalar[dtype](0)
        for e in range(NUM_ENVS):
            se += ess_raw[d * NUM_ENVS + e]
            sh += ent_raw[d * NUM_ENVS + e]
        ess.append(se / Scalar[dtype](NUM_ENVS))
        entropy.append(sh / Scalar[dtype](NUM_ENVS))

    return SearchStats(q_good, ess^, entropy^)


def bar(value: Scalar[dtype], width: Int) -> String:
    """Barra de texto, para poder ensenar el resultado sin salir de la terminal."""
    filled = Int(value * Scalar[dtype](width))
    out = String("")
    for i in range(width):
        out += "#" if i < filled else "."
    return out


def main() raises:
    with DeviceContext() as ctx:
        print("=" * 66)
        print(" Demo 1 - El E-step en vivo (MDP de juguete, sin redes)")
        print("=" * 66)
        print(" El prior es uniforme y no se ha entrenado NADA.")
        print(" Todo lo que mejore la politica viene de simular.")
        print()

        # La configuracion del paper: prior contra politica mejorada.
        short = default_toy_chain()
        # Para el panel del ESS hace falta un pasillo largo: con el corto todas
        # las particulas se truncan en la profundidad 4 y a partir de ahi sus
        # pesos quedan congelados, o sea ESS = N artificialmente.
        long_chain = ToyChain(chain_length=30, horizon=30, value_scale=1.0)

        base = run_search(ctx, 16, 0.5, 4, 4, short)
        prior_good = Scalar[dtype](1.0) / Scalar[dtype](NUM_ACTIONS)

        print(" 1. La politica antes y despues de la busqueda")
        print("    (16 particulas, profundidad 4, temperatura 0.5)")
        print()
        print("      accion   prior              q mejorada")
        print("      BAD      ", bar(prior_good, 20), " ", prior_good,
              "   ", bar(1.0 - base.q_good, 20), " ", 1.0 - base.q_good)
        print("      GOOD     ", bar(prior_good, 20), " ", prior_good,
              "   ", bar(base.q_good, 20), " ", base.q_good)
        print()

        # Como evoluciona la salud de la busqueda con la profundidad.
        deep = run_search(ctx, 16, 0.5, 8, 4, long_chain)
        print(" 2. Salud de la busqueda por profundidad (ESS de 16, periodo 4)")
        print("    (pasillo largo, para que las particulas no mueran antes de tiempo)")
        print("    El ESS baja segun las particulas se separan y el resampling")
        print("    lo recupera. La flecha marca donde se resamplea.")
        print()
        print("      depth   ESS                          entropia")
        for d in range(len(deep.ess)):
            mark = "  <- resampling" if (d + 1) % 4 == 0 else ""
            print("      ", d, "     ", bar(deep.ess[d] / 16.0, 20), " ",
                  deep.ess[d], "   ", deep.entropy[d], mark)
        print()

        # Barrido de temperatura.
        print(" 3. Que hace la temperatura eta")
        print("    Baja = solo sobreviven las mejores (mas mejora, menos ESS).")
        print("    Alta = todas parecidas (menos mejora, mas ESS).")
        print()
        print("      eta     q(GOOD)                      ESS final")
        temps = List[Scalar[dtype]]()
        temps.append(0.1); temps.append(0.5); temps.append(2.0)
        for i in range(len(temps)):
            st = run_search(ctx, 16, temps[i], 4, 4, short)
            print("      ", temps[i], "  ", bar(st.q_good, 20), " ", st.q_good,
                  "   ", st.ess[len(st.ess) - 1])
        print()

        # Barrido del numero de particulas.
        print(" 4. Que hace el numero de particulas N")
        print("    Mas particulas = mejor estimacion de la politica mejorada.")
        print()
        print("      N       q(GOOD)")
        counts = List[Int]()
        counts.append(4); counts.append(16); counts.append(64)
        for i in range(len(counts)):
            st = run_search(ctx, counts[i], 0.5, 4, 4, short)
            print("      ", counts[i], "    ", bar(st.q_good, 20), " ", st.q_good)
        print()

        # Los numeros crudos a disco, por si quiero mirarlos luego.
        with open("results/estep_policy.csv", "w") as f:
            f.write(String("setting,temperature,num_particles,action,probability\n"))
            f.write(String("prior,0.5,16,BAD,", prior_good, "\n"))
            f.write(String("prior,0.5,16,GOOD,", prior_good, "\n"))
            for i in range(len(temps)):
                st = run_search(ctx, 16, temps[i], 4, 4, short)
                f.write(String("search,", temps[i], ",16,BAD,", 1.0 - st.q_good, "\n"))
                f.write(String("search,", temps[i], ",16,GOOD,", st.q_good, "\n"))
            for i in range(len(counts)):
                st = run_search(ctx, counts[i], 0.5, 4, 4, short)
                f.write(String("search,0.5,", counts[i], ",BAD,", 1.0 - st.q_good, "\n"))
                f.write(String("search,0.5,", counts[i], ",GOOD,", st.q_good, "\n"))

        with open("results/estep_ess.csv", "w") as f:
            f.write(String("depth,ess,entropy,num_particles,resample_period\n"))
            for d in range(len(deep.ess)):
                f.write(String(d, ",", deep.ess[d], ",", deep.entropy[d], ",16,4\n"))

        print(" Numeros crudos en results/estep_policy.csv y results/estep_ess.csv")
        print("=" * 66)
