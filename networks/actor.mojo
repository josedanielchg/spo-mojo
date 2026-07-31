"""El actor: la red que elige jugada. 18 -> H -> H -> 9, con enmascarado.

Es la pieza que le faltaba al sistema. Hasta ahora la busqueda partia de un prior
UNIFORME sobre las casillas legales; el actor lo sustituye por una distribucion
aprendida, y eso cierra el bucle EM del paper:

    E-step   la busqueda planifica y produce q, la politica mejorada
    M-step   el actor aprende a imitar q
    y en la vuelta siguiente la busqueda arranca del actor, no de uniforme

La RED es exactamente la misma que la del critico salvo en la salida: 9 logits en
vez de 1 valor. Por eso aqui no se reimplementa el MLP — `critic_forward` ya es
generico en `out_dim` y esta verificado contra goldens y contra autodiff desde
E1.3/E1.5. Reutilizarlo no es pereza: es no volver a arriesgar codigo que ya pasa
por tres verificaciones independientes.

Lo genuinamente nuevo esta abajo, y es el **enmascarado**. En tres en raya no todas
las acciones son legales, y una red no tiene forma de saberlo sola: si no se le
tapa, aprendera a poner probabilidad sobre casillas ocupadas y la busqueda gastara
particulas en jugadas imposibles. Stoix no trae nada de esto -- sus entornos no
tienen acciones ilegales -- asi que es una de las piezas que hay que anadir para
llevar SPO a un juego de tablero.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from ops.buffers import zero_buffer
from ops.common import dtype, GlobalF32
from ops.softmax import softmax_rows
from envs.tictactoe import NUM_ACTIONS, NEG_INF, OBS_DIM, ttt_legal_mask_kernel
from networks.mlp import (CriticParams, CriticCache, critic_forward,
                          zero_critic_params)
from systems.spo.launch import TPB, blocks_for

# Los pesos y las activaciones son los mismos tipos que los del critico: mismo
# MLP, distinta salida. Los alias existen para que el codigo del M-step se lea
# como lo que es ("los pesos del actor") sin duplicar structs.
comptime ActorParams = CriticParams
comptime ActorCache = CriticCache

comptime TPB_ACTOR = 32


def zero_actor_params(ctx: DeviceContext, hidden: Int) raises -> ActorParams:
    """Pesos del actor a cero: OBS_DIM -> hidden -> hidden -> NUM_ACTIONS."""
    return zero_critic_params(ctx, OBS_DIM, hidden, NUM_ACTIONS)


def mask_logits_kernel(logits: GlobalF32, mask: GlobalF32, n: Int,
                       num_actions: Int):
    """Tapa in-place los logits de las acciones ilegales. Un hilo por (fila, accion).

    `mask` vale 1.0 en las legales y 0.0 en las ocupadas, que es lo que produce
    `ttt_legal_mask_kernel`.

    Se usa NEG_INF (el float32 finito mas negativo) y no -inf de verdad, por la
    misma razon que `ttt_prior_logits_kernel`: si una fila entera estuviera tapada
    (tablero lleno), con -inf el softmax daria nan y el nan se propagaria a todo lo
    que toque despues, callado. Con MIN_FINITE la fila degenera a uniforme, que en
    una posicion terminal da igual porque nadie va a jugar esa accion.

    In-place a proposito: el buffer de logits que sale del MLP ya tiene el tamano
    justo y no hace falta otro.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= n * num_actions:
        return
    if mask[i] == Scalar[dtype](0):
        logits[i] = NEG_INF


def actor_logits(ctx: DeviceContext, params: ActorParams, cache: ActorCache,
                 obs: DeviceBuffer[dtype], mask: DeviceBuffer[dtype],
                 m: Int) raises:
    """Logits enmascarados de m tableros. Quedan en `cache.value` [m, NUM_ACTIONS].

    Dos pasos: el MLP (ya verificado) y el tapado. No hace softmax -- quien lo
    necesite lo pide aparte, porque el M-step trabaja en log-espacio y aplicarlo
    aqui obligaria a deshacerlo.
    """
    critic_forward(ctx, params, cache, obs, m)
    n_cells = m * params.out_dim
    ctx.enqueue_function[mask_logits_kernel, mask_logits_kernel](
        cache.value.unsafe_ptr(), mask.unsafe_ptr(), m, params.out_dim,
        grid_dim=blocks_for(n_cells), block_dim=TPB)


def actor_probs(ctx: DeviceContext, params: ActorParams, cache: ActorCache,
                obs: DeviceBuffer[dtype], mask: DeviceBuffer[dtype],
                probs_out: DeviceBuffer[dtype], m: Int) raises:
    """La politica del actor: pi(a|s) para m tableros, ya enmascarada.

    Las casillas ocupadas salen con probabilidad exactamente 0, porque
    exp(NEG_INF - max) desborda a 0 en float32. No es una aproximacion que dependa
    de la escala: NEG_INF esta a ~1e38 del maximo tipico de un logit, y exp de eso
    es cero exacto.
    """
    actor_logits(ctx, params, cache, obs, mask, m)
    ctx.enqueue_function[softmax_rows[TPB_ACTOR], softmax_rows[TPB_ACTOR]](
        probs_out.unsafe_ptr(), cache.value.unsafe_ptr(), params.out_dim,
        grid_dim=m, block_dim=TPB_ACTOR)


struct Actor(Movable):
    """El actor con sus buffers, listo para usarse desde la busqueda o el learner.

    Se queda con la mascara y las probabilidades reservadas para `max_batch`
    tableros, que es lo que hace falta tanto en la raiz (num_envs) como durante el
    entrenamiento (batch * rollout).
    """

    var params: ActorParams
    var cache: ActorCache
    var mask: DeviceBuffer[dtype]
    """[max_batch, NUM_ACTIONS] 1 legal, 0 ocupada."""
    var probs: DeviceBuffer[dtype]
    """[max_batch, NUM_ACTIONS] la politica, tras el softmax enmascarado."""
    var hidden: Int

    def __init__(out self, ctx: DeviceContext, max_batch: Int,
                 hidden: Int) raises:
        self.params = zero_actor_params(ctx, hidden)
        self.cache = ActorCache(ctx, max_batch, hidden, NUM_ACTIONS)
        self.mask = zero_buffer[dtype](ctx, max_batch * NUM_ACTIONS)
        self.probs = zero_buffer[dtype](ctx, max_batch * NUM_ACTIONS)
        self.hidden = hidden

    def mask_from_state(self, ctx: DeviceContext, state: DeviceBuffer[dtype],
                        m: Int) raises:
        """Rellena la mascara leyendo el tablero. La legalidad NO se pasa por
        fuera: se deriva del estado, que es la unica fuente de verdad."""
        ctx.enqueue_function[ttt_legal_mask_kernel, ttt_legal_mask_kernel](
            self.mask.unsafe_ptr(), state.unsafe_ptr(), m,
            grid_dim=(m + TPB_ACTOR - 1) // TPB_ACTOR, block_dim=TPB_ACTOR)

    def forward(self, ctx: DeviceContext, state: DeviceBuffer[dtype],
                obs: DeviceBuffer[dtype], m: Int) raises:
        """Del tablero a la politica: mascara, MLP, tapado y softmax.

        `obs` tiene que traer ya la observacion de dos planos (la produce
        `ttt_encode_obs_kernel`); el actor no la calcula para no duplicar esa
        conversion, que ya vive en el entorno.
        """
        self.mask_from_state(ctx, state, m)
        actor_probs(ctx, self.params, self.cache, obs, self.mask, self.probs, m)
