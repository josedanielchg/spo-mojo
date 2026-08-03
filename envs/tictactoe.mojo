"""Tic-Tac-Toe como entorno para la busqueda SMC.

El tablero 3x3 se guarda como 9 floats, uno por casilla (STATE_DIM = 9):

    indices          codigo de cada casilla:
    0 1 2              0.0  = vacia
    3 4 5              1.0  = ficha del AGENTE (X, juega primero)
    6 7 8             -1.0  = ficha del RIVAL  (O, jugara al azar)

Por que 9 floats sueltos y no un bitboard empaquetado como en el MCTS: aqui no
hay un arbol que acumule miles de tableros -- la busqueda pisa las particulas en
el sitio en cada profundidad -- asi que empaquetar no ahorra nada que importe, y
ademas la red de la fase M querra el tablero justo asi, como numeros sueltos.

El agente es SIEMPRE X, y (fase A3) un "step" del modelo sera jugada del agente +
jugada aleatoria del rival, de modo que cada vez que la busqueda mira el estado le
toca a X: no hace falta guardar de quien es el turno.

Este fichero, como toy_chain, no importa la busqueda: solo los tipos y (mas
adelante) el contrato SearchModel. El E-step no sabe que hay TTT detras.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from ops.common import dtype, idx_dtype, GlobalF32, GlobalI32, NEG_INF
from systems.spo.particles import Particles, StepOutputs
from systems.spo.search_model import SearchModel
from systems.spo.spo_types import SPOConfig

# El tablero son 9 casillas: 9 floats de estado, 9 acciones (una por casilla).
comptime NUM_CELLS = 9
comptime STATE_DIM = 9
comptime NUM_ACTIONS = 9

# Codigo de cada casilla. El agente (X) es +1 y el rival (O) es -1 a proposito:
# quedan simetricos respecto al 0 (la casilla vacia), lo que le viene bien a la
# red de la fase M.
comptime CELL_EMPTY = Scalar[dtype](0)
comptime CELL_AGENT = Scalar[dtype](1)
comptime CELL_RIVAL = Scalar[dtype](-1)

comptime TPB_TTT = 32

# Lo que ve la RED (no la busqueda): dos planos de 9, primero las fichas propias y
# despues las del rival. Es la codificacion que usa pgx para tres en raya
# (`observation.shape == (3, 3, 2)`), y la usamos igual aqui para que la red de
# Mojo y la de Stoix reciban exactamente lo mismo y la comparacion sea limpia.
comptime NUM_PLANES = 2
comptime OBS_DIM = NUM_PLANES * NUM_CELLS   # 18


def ttt_cell(state: GlobalF32, p: Int, c: Int) -> Scalar[dtype]:
    """La casilla c (0..8) de la particula p.

    Fija en un solo sitio la convencion de layout: las 9 casillas de una
    particula van seguidas, con paso STATE_DIM, para que un kernel por-particula
    lea las suyas y no las de la vecina.
    """
    return state[p * STATE_DIM + c]


def ttt_three(state: GlobalF32, p: Int, a: Int, b: Int, c: Int,
              player: Scalar[dtype]) -> Bool:
    """True si `player` ocupa las tres casillas a, b, c de la particula p.

    Comparo casilla != player y salgo en cuanto falla, en vez de encadenar tres
    `and` sobre comparaciones de SIMD-bool, que es lo que se traga sin dudas el
    compilador de kernels."""
    if ttt_cell(state, p, a) != player:
        return False
    if ttt_cell(state, p, b) != player:
        return False
    if ttt_cell(state, p, c) != player:
        return False
    return True


def ttt_has_won(state: GlobalF32, p: Int, player: Scalar[dtype]) -> Bool:
    """True si `player` (CELL_AGENT o CELL_RIVAL) completa alguna de las 8 lineas.

    Son las mismas 8 lineas que WIN_MASKS en el MCTS (3 filas, 3 columnas, 2
    diagonales), aqui como ternas de indices porque el tablero son 9 floats y no
    un bitboard. El paralelismo esta en los hilos (una particula por hilo), asi
    que dentro del hilo un simple OR de las 8 lineas es lo natural -- no hace
    falta el truco SIMD-sobre-lineas que usa el MCTS en CPU.
    """
    return (ttt_three(state, p, 0, 1, 2, player)      # filas
            or ttt_three(state, p, 3, 4, 5, player)
            or ttt_three(state, p, 6, 7, 8, player)
            or ttt_three(state, p, 0, 3, 6, player)    # columnas
            or ttt_three(state, p, 1, 4, 7, player)
            or ttt_three(state, p, 2, 5, 8, player)
            or ttt_three(state, p, 0, 4, 8, player)    # diagonales
            or ttt_three(state, p, 2, 4, 6, player))


def ttt_has_won_kernel(won_out: GlobalF32, state: GlobalF32, n_particles: Int,
                       player: Scalar[dtype]):
    """1.0 si `player` gano en el tablero de su particula, 0.0 si no. Un hilo por
    particula."""
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p < n_particles:
        won_out[p] = Scalar[dtype](1) if ttt_has_won(state, p, player) \
                     else Scalar[dtype](0)


def ttt_is_legal(state: GlobalF32, p: Int, c: Int) -> Bool:
    """True si la casilla c de la particula p esta vacia (jugada legal)."""
    if ttt_cell(state, p, c) == CELL_EMPTY:
        return True
    return False


def ttt_legal_mask_kernel(mask_out: GlobalF32, state: GlobalF32,
                          n_particles: Int):
    """Mascara de acciones legales [n_particles, NUM_ACTIONS]: 1.0 si la casilla
    esta libre, 0.0 si esta ocupada. Un hilo por particula.

    La usara la busqueda para tapar las acciones ilegales del prior (metiendo
    -inf en las ocupadas) y el rival aleatorio para sortear solo entre libres.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p < n_particles:
        for c in range(NUM_ACTIONS):
            mask_out[p * NUM_ACTIONS + c] = Scalar[dtype](1) \
                if ttt_is_legal(state, p, c) else Scalar[dtype](0)


def ttt_apply(state: GlobalF32, p: Int, cell: Int, player: Scalar[dtype]):
    """Pone la ficha de `player` en la casilla `cell` de la particula p, in-place.

    No comprueba legalidad: da por hecho que quien llama ya eligio una casilla
    libre (via la mascara legal). Es la primitiva que usara el step de la fase A3
    para la jugada del agente y la del rival.
    """
    state[p * STATE_DIM + cell] = player


def ttt_apply_kernel(state: GlobalF32, action: GlobalI32, n_particles: Int,
                     player: Scalar[dtype]):
    """Cada particula pone una ficha de `player` en la casilla que dice su accion.
    Un hilo por particula, modifica el estado in-place."""
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p < n_particles:
        ttt_apply(state, p, Int(action[p]), player)


def ttt_is_full(state: GlobalF32, p: Int) -> Bool:
    """True si no queda ninguna casilla vacia (no hay jugadas legales)."""
    for c in range(NUM_CELLS):
        if ttt_cell(state, p, c) == CELL_EMPTY:
            return False
    return True


def ttt_is_terminal(state: GlobalF32, p: Int) -> Bool:
    """True si la partida acabo: gano alguien o el tablero esta lleno (empate)."""
    if ttt_has_won(state, p, CELL_AGENT):
        return True
    if ttt_has_won(state, p, CELL_RIVAL):
        return True
    if ttt_is_full(state, p):
        return True
    return False


def ttt_reward(state: GlobalF32, p: Int) -> Scalar[dtype]:
    """Recompensa del tablero desde la vista del AGENTE (X):

        gana X       -> +1.0
        empate       -> +0.5   (tablero lleno sin linea)
        gana O       ->  0.0   (derrota)
        no terminal  ->  0.0   (recompensa de paso: el juego sigue)

    Gana-X se comprueba antes que lleno: una jugada ganadora que ademas llena el
    tablero es victoria, no empate. `terminal` y `reward` son salidas separadas:
    una derrota y un paso intermedio dan 0 los dos, y se distinguen por terminal.
    """
    if ttt_has_won(state, p, CELL_AGENT):
        return Scalar[dtype](1)
    if ttt_has_won(state, p, CELL_RIVAL):
        return Scalar[dtype](0)
    if ttt_is_full(state, p):
        return Scalar[dtype](0.5)
    return Scalar[dtype](0)


def ttt_outcome_kernel(terminal_out: GlobalF32, reward_out: GlobalF32,
                       state: GlobalF32, n_particles: Int):
    """Por particula: terminal (1.0/0.0) y recompensa del agente. Un hilo por
    particula."""
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p < n_particles:
        terminal_out[p] = Scalar[dtype](1) if ttt_is_terminal(state, p) \
                          else Scalar[dtype](0)
        reward_out[p] = ttt_reward(state, p)


def ttt_prior_logits_kernel(logits_out: GlobalF32, state: GlobalF32,
                            n_envs: Int):
    """Prior de la busqueda en los estados raiz: [n_envs, NUM_ACTIONS].

    Uniforme sobre las casillas LEGALES (logit 0) y tapado en las ocupadas
    (NEG_INF), de modo que el softmax da probabilidad 0 a jugar sobre una ficha ya
    puesta. Es el analogo del prior uniforme del juguete, pero enmascarado porque
    en TTT no todas las acciones son legales.

    Uso NEG_INF (MIN_FINITE) y no -inf de verdad: si un tablero no tuviera ninguna
    casilla legal (no deberia pasar en una raiz, pero por si acaso), una fila
    entera de -inf daria nan en el softmax; con MIN_FINITE degenera a uniforme,
    que es inofensivo. Un hilo por env.
    """
    e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e < n_envs:
        for c in range(NUM_ACTIONS):
            logits_out[e * NUM_ACTIONS + c] = Scalar[dtype](0) \
                if ttt_is_legal(state, e, c) else NEG_INF


def ttt_random_legal_cell(state: GlobalF32, p: Int, u: Scalar[dtype]) -> Int:
    """La casilla vacia numero floor(u*m) (m = nº de vacias), contando desde 0.

    Uniforme sobre las casillas libres, sin listas: cuenta cuantas hay y devuelve
    la k-esima. Es el analogo del random_set_bit del MCTS, aqui sobre floats. u en
    [0,1). Devuelve -1 si no hay ninguna, cosa que no ocurre en el step (solo se
    llama cuando el tablero no esta lleno).
    """
    m = 0
    for c in range(NUM_CELLS):
        if ttt_is_legal(state, p, c):
            m += 1
    if m == 0:
        return -1
    k = Int(u * Scalar[dtype](m))
    if k >= m:            # guarda por si u redondeara justo a m
        k = m - 1
    j = 0
    for c in range(NUM_CELLS):
        if ttt_is_legal(state, p, c):
            if j == k:
                return c
            j += 1
    return -1             # inalcanzable con m > 0


@fieldwise_init
struct TTTOutcome(Copyable, Movable):
    """Lo que sale de avanzar un tablero un turno completo. Struct de escalares,
    que una funcion de device si puede devolver (ver docs/api_notes.md)."""
    var reward: Scalar[dtype]
    """Vista del AGENTE: 1 gana, 0.5 empate, 0 pierde o sigue."""
    var terminal: Scalar[dtype]
    """1.0 si la partida acabo, 0.0 si sigue."""


def ttt_advance(state: GlobalF32, p: Int, action: Int,
                u: Scalar[dtype]) -> TTTOutcome:
    """Un turno completo: juega el agente (X) y responde el rival al azar.

    Modifica el tablero de p in-place y devuelve recompensa y terminal:
      1. El agente juega `action`.
      2. Si X gana o el tablero se llena -> terminal (el rival no llega a jugar).
      3. Si no, el rival (O) juega una casilla legal al azar, elegida con `u`.
      4. Si O gana o se llena -> terminal.
    Cada has_won/is_full se calcula UNA vez, en el orden del juego.

    Es la dinamica compartida entre la busqueda (ttt_dynamics_kernel) y el entorno
    real (ttt_env_step_kernel), igual que cartpole_advance compartia la fisica: asi
    las reglas viven en un solo sitio y no pueden desincronizarse.

    Memory-safe por construccion: `action` viene acotada a [0,9) y el rival solo
    juega cuando el tablero NO esta lleno, asi que random_legal_cell siempre
    encuentra hueco y nunca devuelve -1.
    """
    ttt_apply(state, p, action, CELL_AGENT)
    if ttt_has_won(state, p, CELL_AGENT):
        return TTTOutcome(Scalar[dtype](1), Scalar[dtype](1))     # gana el agente
    if ttt_is_full(state, p):
        return TTTOutcome(Scalar[dtype](0.5), Scalar[dtype](1))   # empate

    cell = ttt_random_legal_cell(state, p, u)
    ttt_apply(state, p, cell, CELL_RIVAL)
    if ttt_has_won(state, p, CELL_RIVAL):
        return TTTOutcome(Scalar[dtype](0), Scalar[dtype](1))     # derrota
    if ttt_is_full(state, p):
        return TTTOutcome(Scalar[dtype](0.5), Scalar[dtype](1))   # empate
    return TTTOutcome(Scalar[dtype](0), Scalar[dtype](0))         # sigue


def ttt_dynamics_kernel(state: GlobalF32, action: GlobalI32,
                        step_uniforms: GlobalF32, depth: GlobalI32,
                        reward_out: GlobalF32, discount_out: GlobalF32,
                        next_value_out: GlobalF32, n_particles: Int,
                        reward_gamma: Scalar[dtype],
                        loss_penalty: Scalar[dtype]):
    """El step estocastico de TTT para la BUSQUEDA: un turno + gamma plegada.

    Avanza con `ttt_advance` y traduce el resultado al contrato del nucleo SMC:
    discount (0 si terminal, 1 si sigue) y next_value (0: sin critico todavia).

    Como el juguete, NO comprueba si la particula ya estaba muerta: una particula
    terminal se vuelve a pisar y da igual, porque su peso ya lo congelo el nucleo
    SMC. Un hilo por particula.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p >= n_particles:
        return

    out = ttt_advance(state, p, Int(action[p]), step_uniforms[p])
    # Descuento por profundidad: ganar YA vale mas que ganar despues. Sin esto, y
    # con V=0, el peso SMC es la suma de recompensas sin descontar, asi que una
    # particula que gana en el paso 0 empata con otra que gana en el paso 3 -- y el
    # softmax no puede distinguir "gane seguro" de "gane con suerte".
    g = Scalar[dtype](1)
    for _ in range(Int(depth[p])):
        g *= reward_gamma

    # Castigo por derrota. Sin el, `ttt_advance` devuelve 0 tanto si la partida se
    # PIERDE como si simplemente SIGUE, asi que el peso SMC no puede distinguir
    # las dos cosas y la busqueda no tiene ningun motivo para bloquear una
    # amenaza. Con loss_penalty > 0 una derrota pasa a valer -loss_penalty, que es
    # el convenio habitual de los juegos (+1 / 0 / -1). Con 0 se recupera el
    # comportamiento original.
    r = out.reward
    if loss_penalty != 0 and out.terminal != 0 and out.reward == 0:
        r = -loss_penalty

    reward_out[p] = r * g
    discount_out[p] = Scalar[dtype](1) - out.terminal
    next_value_out[p] = Scalar[dtype](0)


def ttt_reset_kernel(state: GlobalF32, n_envs: Int):
    """Empieza partida nueva: tablero vacio. El agente (X) siempre mueve primero,
    asi que no hay nada mas que inicializar. Un hilo por env."""
    e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e < n_envs:
        for c in range(NUM_CELLS):
            state[e * STATE_DIM + c] = CELL_EMPTY


def ttt_env_step_kernel(state: GlobalF32, action: GlobalI32,
                        u_rival: GlobalF32, reward_out: GlobalF32,
                        done_out: GlobalI32, n_envs: Int):
    """Un turno en el entorno REAL: como el de la busqueda pero sin gamma.

    Comparte `ttt_advance` con el kernel de la busqueda, asi que las reglas son
    literalmente las mismas; lo unico que cambia es la salida: aqui interesa
    `done` (para reiniciar la partida y contar el resultado) en vez del discount
    y el bootstrap. Un hilo por env.
    """
    e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e < n_envs:
        out = ttt_advance(state, e, Int(action[e]), u_rival[e])
        reward_out[e] = out.reward
        done_out[e] = Scalar[idx_dtype](1) if out.terminal != Scalar[dtype](0) \
                      else Scalar[idx_dtype](0)


def ttt_auto_reset_kernel(state: GlobalF32, done: GlobalI32, n_envs: Int):
    """Los envs que acaban de terminar empiezan partida nueva; los demas siguen.

    Va DESPUES de que el host haya leido reward y done: el tablero se limpia, pero
    el resultado de la partida ya esta anotado en sus buffers. Es el auto-reset del
    entorno REAL; dentro de la busqueda no existe (una particula terminal se queda
    quieta y su peso lo congela la mascara terminal). Un hilo por env.
    """
    e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e < n_envs and Int(done[e]) != 0:
        for c in range(NUM_CELLS):
            state[e * STATE_DIM + c] = CELL_EMPTY


def ttt_random_policy_kernel(action_out: GlobalI32, state: GlobalF32,
                             uniforms: GlobalF32, n_envs: Int):
    """La politica ALEATORIA del agente: una casilla legal al azar por env.

    Es la linea base contra la que se compara la busqueda: si planificar no gana
    mas partidas que esto, la busqueda no esta aportando nada. Un hilo por env.
    """
    e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e < n_envs:
        action_out[e] = Scalar[idx_dtype](
            ttt_random_legal_cell(state, e, uniforms[e]))


def ttt_encode_obs_kernel(obs_out: GlobalF32, state: GlobalF32, n: Int):
    """Traduce el tablero al formato que come la red: [n, 9] -> [n, OBS_DIM].

    De 9 casillas con codigo (+1 mia / -1 suya / 0 vacia) a dos planos binarios
    puestos uno detras del otro:

        obs[0 .. 8]   1 si la casilla es MIA,    0 si no
        obs[9 .. 17]  1 si la casilla es SUYA,   0 si no

    Una casilla vacia queda a 0 en LOS DOS planos, asi que "vacia" se codifica por
    ausencia y no hace falta un tercer plano.

    Por que dos planos y no los 9 floats tal cual: con un solo numero la red
    tendria que deducir sola que -1, 0 y +1 son tres CATEGORIAS y no una escala
    donde el 0 esta en medio. Y sobre todo, es la codificacion que usa pgx, que es
    lo que vera la implementacion de Stoix: las dos redes reciben lo mismo.

    Sirve igual para particulas (durante la busqueda) que para envs (en el bucle
    real): solo cambia `n`. Un hilo por tablero.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p >= n:
        return

    base = p * OBS_DIM
    for c in range(NUM_CELLS):
        v = ttt_cell(state, p, c)
        obs_out[base + c] = Scalar[dtype](1) if v == CELL_AGENT \
                            else Scalar[dtype](0)
        obs_out[base + NUM_CELLS + c] = Scalar[dtype](1) if v == CELL_RIVAL \
                                        else Scalar[dtype](0)


def ttt_read_cells_kernel(cells_out: GlobalF32, state: GlobalF32,
                          n_particles: Int):
    """Copia las 9 casillas de cada particula a la salida usando ttt_cell.

    Todavia no hace logica de juego: existe para comprobar que el accesor indexa
    la casilla correcta de la particula correcta (que el paso STATE_DIM es el que
    toca y las particulas no se solapan). Un hilo por particula.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p < n_particles:
        for c in range(NUM_CELLS):
            cells_out[p * NUM_CELLS + c] = ttt_cell(state, p, c)


struct TicTacToe(SearchModel, Copyable, Movable):
    """Tic-Tac-Toe como modelo de busqueda: el agente (X) contra un rival aleatorio.

    La regla es fija y el agente siempre es X; no hay critico todavia (V=0), asi que
    la busqueda mejora un prior uniforme enmascarado partiendo de valor cero -- el
    modo "planificador" del Milestone 1.

    Como toy_chain, no importa la busqueda: solo el contrato SearchModel y los
    tipos. Los kernels que usa (prior de A3a, dinamica de A3b) estan probados por
    separado; aqui solo se cablean en los dos metodos del contrato.
    """

    var reward_gamma: Scalar[dtype]
    """Descuento de la recompensa por profundidad. 1.0 = sin descuento (ganar ya
    vale igual que ganar despues, que en un juego con premio terminal deja el peso
    SMC casi ciego); <1 premia ganar rapido."""

    var loss_penalty: Scalar[dtype]
    """Lo que vale perder, en negativo. 0 = el convenio original (perder vale 0,
    lo mismo que no haber resuelto la partida); 1 = el convenio de juegos
    +1/0/-1. Es MODELADO de recompensa dentro de la busqueda: la recompensa del
    entorno real, la que cuenta el marcador, no se toca."""

    def __init__(out self, reward_gamma: Scalar[dtype],
                 loss_penalty: Scalar[dtype] = 0):
        self.reward_gamma = reward_gamma
        self.loss_penalty = loss_penalty

    def eval_root(self, ctx: DeviceContext, cfg: SPOConfig,
                  root_state: DeviceBuffer[dtype],
                  logits_out: DeviceBuffer[dtype],
                  value_out: DeviceBuffer[dtype]) raises:
        """El prior enmascarado sobre los estados raiz, y V=0 (sin critico)."""
        blocks = (cfg.num_envs + TPB_TTT - 1) // TPB_TTT
        ctx.enqueue_function[ttt_prior_logits_kernel, ttt_prior_logits_kernel](
            logits_out.unsafe_ptr(), root_state.unsafe_ptr(), cfg.num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
        # V = 0: modo planificador. Se pisa explicitamente porque el workspace se
        # reutiliza entre busquedas y podria traer valores viejos.
        value_out.enqueue_fill(0)

    def step(self, ctx: DeviceContext, cfg: SPOConfig, particles: Particles,
             outputs: StepOutputs, step_uniforms: DeviceBuffer[dtype]) raises:
        """Avanza las P particulas (agente + rival al azar) y evalua el prior en el
        estado NUEVO, que es de donde se muestreara la accion siguiente."""
        p_total = cfg.num_search_particles()
        blocks = (p_total + TPB_TTT - 1) // TPB_TTT

        ctx.enqueue_function[ttt_dynamics_kernel, ttt_dynamics_kernel](
            particles.state.unsafe_ptr(), outputs.next_action.unsafe_ptr(),
            step_uniforms.unsafe_ptr(), particles.depth.unsafe_ptr(),
            outputs.reward.unsafe_ptr(), outputs.discount.unsafe_ptr(),
            outputs.next_value.unsafe_ptr(), p_total, self.reward_gamma,
            self.loss_penalty, grid_dim=blocks, block_dim=TPB_TTT)

        ctx.enqueue_function[ttt_prior_logits_kernel, ttt_prior_logits_kernel](
            outputs.action_logits.unsafe_ptr(), particles.state.unsafe_ptr(),
            p_total, grid_dim=blocks, block_dim=TPB_TTT)


def default_tictactoe() -> TicTacToe:
    """El modelo de TTT **fiel a SPO**: sin descuento por profundidad.

    Antes esto devolvia `reward_gamma = 0.7`, y eso estaba mal como DEFECTO. El
    0.7 es un apano nuestro de A6 (plegar gamma^d dentro de la recompensa) que
    Stoix no hace: su `recurrent_fn` pasa la recompensa cruda del entorno
    (`reward=next_timestep.reward`, ff_spo.py:341). Tenerlo como valor por defecto
    hacia que nuestra desviacion PARECIERA el metodo.

    Y ademas resulto ser danina: con el actor conectado, gamma_r = 1.0 da 0.62% de
    derrotas y gamma_r = 0.7 da 2.65%. El apano existia para compensar la falta de
    politica; cuando la politica llega, estorba.

    Se conserva el parametro para poder reproducir las medidas de A6 y E1.11, pero
    hay que pasarlo a mano y a la vista.
    """
    return TicTacToe(reward_gamma=1.0)


def ttt_legal_mask_from_obs_kernel(mask_out: GlobalF32, obs: GlobalF32,
                                   n: Int):
    """Mascara de legales leyendo la OBSERVACION en vez del tablero.

    Existe porque el buffer de entrenamiento guarda observaciones (18 floats), no
    estados (9). Se podria guardar tambien el estado, pero seria duplicar
    informacion que ya esta ahi: en la codificacion de dos planos una casilla esta
    libre si y solo si vale 0 en LOS DOS.

    Que la mascara salga siempre del mismo sitio que ve la red importa: si viniera
    por un canal aparte podria desincronizarse y el actor acabaria poniendo
    probabilidad sobre fichas ya puestas. Un hilo por fila.
    """
    r = Int(block_dim.x * block_idx.x + thread_idx.x)
    if r >= n:
        return
    for c in range(NUM_CELLS):
        mine = obs[r * OBS_DIM + c]
        theirs = obs[r * OBS_DIM + NUM_CELLS + c]
        mask_out[r * NUM_ACTIONS + c] = Scalar[dtype](1) \
            if (mine == 0 and theirs == 0) else Scalar[dtype](0)
