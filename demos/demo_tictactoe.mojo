"""Demo 2: la busqueda SMC juega al tres en raya.

La pregunta que decide la fase A: ¿planificar juega mejor que tirar al azar? No hay
ninguna red entrenada -- el prior es uniforme sobre las casillas legales y V=0, asi
que toda la mejora viene de simular partidas hacia delante.

    ./run.sh demos/demo_tictactoe.mojo

Las dos referencias son EXACTAS, no estimadas: en un juego tan pequeno se calculan
por recursion sobre todos los estados (`ttt_exact.py` / `ttt_optimal.py`).

    suelo   X al azar    vs O al azar : gana 58.49%  empata 12.70%  pierde 28.81%  score 0.6484
    techo   X optimo     vs O al azar : gana 99.48%  empata  0.52%  pierde  0.00%  score 0.9974
    aqui    X = busqueda vs O al azar : gana 96.80%  empata  1.18%  pierde  2.02%  score 0.9739

O sea 1.50x el azar, y el 97.6% del maximo alcanzable. Tener el techo importa: dice
cuanto margen queda de verdad (poco) y da la referencia con la que comparar el MCTS.
La cifra que mas informa es la de DERROTAS, porque el juego optimo no pierde NUNCA:
ese 2% son amenazas del rival que la busqueda no bloqueo.

Imprime cuatro cosas:
  1. linea base vs busqueda      el resultado principal
  2. barrido del descuento       por que la recompensa tiene que descontarse
  3. barrido de profundidad      cuanto ayuda simular mas turnos
  4. barrido de particulas       cuanto ayuda simular mas variantes

Los dos hallazgos que hicieron falta para llegar aqui, los dos encontrados con esta
demo (primero daba 0.997x, o sea igual que el azar):

  * Un bug del nucleo: `root_fn` no ponia a cero los acumuladores, asi que al
    reutilizar el workspace la segunda busqueda heredaba `terminal = 1` y la
    mascara congelaba TODOS los pesos. La busqueda degeneraba en elegir al azar sin
    fallar en nada. Tiene test de regresion en test_search.mojo.
  * El descuento por profundidad (`reward_gamma`): sin el, ganar en el primer turno
    vale lo mismo que ganar en el cuarto, y el softmax no puede distinguir "gane
    seguro" de "gane con suerte". Se ve en el barrido 2: gamma=1 da 0.81 y
    gamma=0.7 da 0.97.

Los parametros salen de barrer cada uno midiendo partidas: la temperatura es lo que
mas manda (0.5 -> 0.78, 0.1 -> 0.94, 0.02 -> 0.974, y ahi satura), el descuento
tiene un valle plano entre 0.5 y 0.7, y el periodo de resampling apenas se nota.

Nota de escala para leer el barrido de profundidad: cada paso del modelo es un turno
completo (agente + rival) y una partida dura como mucho 5 jugadas del agente, asi
que la mejora satura en cuanto la profundidad cubre el final de la partida.
"""

from std.gpu.host import DeviceContext

from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform
from envs.tictactoe import (TicTacToe, ttt_reset_kernel,
                            ttt_env_step_kernel, ttt_auto_reset_kernel,
                            NUM_ACTIONS, STATE_DIM, TPB_TTT)
from envs.tictactoe_runner import play_random_games, MatchStats, RNG_RIVAL
from systems.spo.particles import SearchWorkspace
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search

comptime SEED = UInt32(20260724)
comptime NUM_ENVS = 64
"""Partidas en paralelo. Cada una es independiente, asi que mas envs = mas partidas
por turno y una medida mas estable."""

comptime EXACT_RANDOM_SCORE = Scalar[dtype](0.6484)
"""Puntuacion del agente aleatorio, calculada exactamente (no medida)."""


def search_config(num_particles: Int, depth: Int, period: Int,
                  temperature: Scalar[dtype]) -> SPOConfig:
    """Config de busqueda para TTT.

    `search_gamma` se queda en 1 porque solo multiplica al bootstrap `V(s')`, y aqui
    V es 0: no tiene ningun efecto. El descuento que SI importa es el del modelo
    (`reward_gamma`), que se aplica a la recompensa por profundidad.
    """
    return SPOConfig(
        num_envs=NUM_ENVS, num_particles=num_particles, num_actions=NUM_ACTIONS,
        state_dim=STATE_DIM, search_depth=depth, resample_period=period,
        temperature=temperature, search_gamma=1.0, search_gae_lambda=1.0)


def play_search_games(ctx: DeviceContext, cfg: SPOConfig, model: TicTacToe,
                      num_steps: Int, seed: UInt32) raises -> MatchStats:
    """Juega partidas de verdad usando la BUSQUEDA como politica.

    En cada turno: se planifica desde el estado real (una busqueda completa por
    env), se ejecuta la accion elegida, se anota la partida si acabo y se reinicia.

    Vive en demos/ y no en envs/ a proposito: esto importa el algoritmo de busqueda,
    y la regla del proyecto es que un entorno nunca dependa del algoritmo. El nivel
    de aplicacion (una demo) si puede depender de los dos.

    El `SearchWorkspace` se reserva UNA vez fuera del bucle: aqui la busqueda corre
    en cada turno, asi que reservar sus buffers cada vez seria trabajo puro por nada.
    """
    ws = SearchWorkspace(ctx, cfg)
    num_envs = cfg.num_envs
    blocks = (num_envs + TPB_TTT - 1) // TPB_TTT

    state = zero_buffer[dtype](ctx, num_envs * STATE_DIM)
    reward = zero_buffer[dtype](ctx, num_envs)
    done = zero_buffer[idx_dtype](ctx, num_envs)
    u_rival = zero_buffer[dtype](ctx, num_envs)

    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), num_envs, grid_dim=blocks, block_dim=TPB_TTT)

    wins = 0
    draws = 0
    losses = 0

    for step in range(num_steps):
        # Planifica desde el estado real. La busqueda copia el tablero a sus
        # particulas, asi que no toca el estado del entorno.
        # La seed cambia por turno (con un multiplicador impar grande) para que dos
        # turnos desde el mismo tablero no sorteen exactamente las mismas variantes.
        search[TicTacToe](ctx, ws, cfg, model, state,
                          seed ^ (UInt32(step) * 2654435761))

        # Y ejecuta la accion elegida en el entorno real.
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), seed, RNG_RIVAL + UInt32(step), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), ws.output.action.unsafe_ptr(),
            u_rival.unsafe_ptr(), reward.unsafe_ptr(), done.unsafe_ptr(),
            num_envs, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

        with reward.map_to_host() as rh:
            with done.map_to_host() as dh:
                for e in range(num_envs):
                    if Int(dh[e]) != 0:
                        r = rh[e]
                        if r > Scalar[dtype](0.75):
                            wins += 1
                        elif r > Scalar[dtype](0.25):
                            draws += 1
                        else:
                            losses += 1

        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)

    return MatchStats(wins, draws, losses)


def show(label: String, s: MatchStats) raises:
    """Una linea de marcador, con la mejora sobre el azar exacto."""
    print("  ", label,
          "  partidas", s.games(),
          " gana", s.win_rate(),
          " empata", Scalar[dtype](s.draws) / Scalar[dtype](s.games()),
          " pierde", Scalar[dtype](s.losses) / Scalar[dtype](s.games()),
          " score", s.score(),
          " (x", s.score() / EXACT_RANDOM_SCORE, "vs azar )")


def main() raises:
    with DeviceContext() as ctx:
        print("=== 1. linea base vs busqueda ===")
        base = play_random_games(ctx, NUM_ENVS, 200, SEED)
        show("azar     ", base)
        print("            (exacto: gana 0.5849  empata 0.1270  pierde 0.2881  score 0.6484 )")

        cfg = search_config(num_particles=64, depth=6, period=3, temperature=0.02)
        srch = play_search_games(ctx, cfg, TicTacToe(reward_gamma=0.7), 60, SEED)
        show("busqueda ", srch)

        print()
        print("=== 2. barrido del descuento de recompensa (N=64, depth=6, temp=0.02) ===")
        print("    gamma=1 deja el peso SMC casi ciego: ganar ya vale igual que ganar despues")
        gammas = List[Scalar[dtype]]()
        gammas.append(1.0); gammas.append(0.9); gammas.append(0.7); gammas.append(0.5)
        for i in range(len(gammas)):
            c = search_config(num_particles=64, depth=6, period=3, temperature=0.02)
            s = play_search_games(ctx, c, TicTacToe(reward_gamma=gammas[i]), 60, SEED)
            show(String("gamma ", gammas[i], "  "), s)

        print()
        print("=== 3. barrido de profundidad (N=64, gamma=0.7) ===")
        depths = List[Int]()
        depths.append(1); depths.append(2); depths.append(3)
        depths.append(5); depths.append(8)
        for i in range(len(depths)):
            d = depths[i]
            c = search_config(num_particles=64, depth=d, period=3, temperature=0.02)
            s = play_search_games(ctx, c, TicTacToe(reward_gamma=0.7), 60, SEED)
            show(String("depth ", d, "  "), s)

        print()
        print("=== 4. barrido de particulas (depth=6, gamma=0.7) ===")
        counts = List[Int]()
        counts.append(4); counts.append(16); counts.append(64); counts.append(128)
        for i in range(len(counts)):
            n = counts[i]
            c = search_config(num_particles=n, depth=6, period=3, temperature=0.02)
            s = play_search_games(ctx, c, TicTacToe(reward_gamma=0.7), 60, SEED)
            show(String("N = ", n, "  "), s)
