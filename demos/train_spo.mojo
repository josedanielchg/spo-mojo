"""E2.4: el bucle de aprendizaje completo. Actor y critico, los dos entrenando.

Hasta aqui el critico aprendia solo (E1.10) y el actor existia pero no lo llamaba
nadie. Esto los junta y cierra la mitad del bucle EM:

    E-step   la busqueda planifica y produce q, la politica mejorada
    M-step   el critico aprende a predecir retornos  (perdida L2, E1.10)
             el actor aprende a imitar q             (ecuacion 11, E2.2/E2.3)

Falta la vuelta: que el prior de la busqueda salga del actor en vez de ser
uniforme. Eso es E2.5, y hasta entonces la busqueda no se entera de que hay un
actor -- aqui el actor solo mira. Que sea asi a proposito importa: permite medir
si el actor APRENDE antes de meterlo a decidir, y si algo sale raro en E2.5 ya se
sabe que no viene de aqui.

**Optimizador propio para cada red**, como en Stoix, que tiene `actor_lr` y
`critic_lr` por separado (los dos a 3e-4 en su config) y encadena el clip por
norma dentro de cada optax. O sea: dos estados de Adam, dos clips globales
independientes. Mezclarlos haria que la norma de uno recortara los gradientes del
otro.

**Que se reporta y por que.** La entropia cruzada cruda no sirve de curva: se
descompone en H(q) + KL(q||pi) y H(q) NO depende del actor, asi que el suelo se
mueve cuando cambia q. Se reporta la **KL**, cuyo cero significa "el actor
reproduce lo que dice la busqueda" y es comparable entre configuraciones. Ver el
comentario largo en networks/actor_loss.mojo.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from std.math import log

from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.copy import copy_kernel
from ops.rng import fill_uniform
from ops.softmax import log_softmax_rows
from envs.tictactoe import (TicTacToe, ttt_reset_kernel, ttt_env_step_kernel,
                            ttt_auto_reset_kernel, ttt_encode_obs_kernel,
                            ttt_legal_mask_from_obs_kernel, NUM_ACTIONS,
                            NUM_CELLS, STATE_DIM, OBS_DIM, TPB_TTT)
from envs.tictactoe_actor import TicTacToeActor
from envs.tictactoe_runner import RNG_RIVAL
from networks.actor import Actor, actor_probs
from networks.actor_loss import (actor_backward, cross_entropy_rows,
                                 entropy_rows, kl_rows)
from networks.mlp import (CriticCache, CriticGrads, CriticScratch,
                          critic_forward, critic_backward)
from networks.optim import (AdamState, adam_step, ema_update, sum_squares,
                            global_clip_scale)
from rl_utils.buffer import TrajectoryBuffer
from rl_utils.multistep import truncated_gae
from systems.spo.particles import SearchWorkspace
from systems.spo.readout import readout_expected
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from tests.helpers import upload, download, write_into
from demos.train_critic import Critic, init_critic_weights

comptime SEED = UInt32(20260802)
comptime NUM_ENVS = 32
comptime ROLLOUT = 16
comptime HIDDEN = 64
comptime GAMMA = Scalar[dtype](0.99)
comptime GAE_LAMBDA = Scalar[dtype](0.95)
comptime CRITIC_LR = Scalar[dtype](3e-4)
comptime ACTOR_LR = Scalar[dtype](3e-4)
comptime MAX_GRAD_NORM = Scalar[dtype](0.5)
comptime TAU = Scalar[dtype](0.005)
comptime BATCH = 16
comptime BUFFER_CAP = 256

# La busqueda, con el montaje que quedo tras E1.11c: sin remuestreo, readout de
# media, y el castigo por derrota. Es el que juega a 0.00% de derrotas.
comptime NUM_PARTICLES = 128
comptime SEARCH_DEPTH = 6
comptime NO_RESAMPLE = 99
comptime TEMPERATURE = Scalar[dtype](0.02)
comptime REWARD_GAMMA = Scalar[dtype](0.9)
comptime LOSS_PENALTY = Scalar[dtype](1.0)

# Stream propio para el sorteo de la accion en el readout. RNG_POLICY (20000) y
# RNG_RIVAL (30000) ya estan tomados; 40000 no colisiona con ninguno.
comptime RNG_READOUT = UInt32(40000)

comptime TRAIN_ROUNDS = 30
comptime UPDATES_PER_ROUND = 80


struct ActorLearner(Movable):
    """El actor con todo lo que hace falta para entrenarlo."""

    var net: Actor
    var grads: CriticGrads
    var scratch: CriticScratch
    var a_w1: AdamState
    var a_b1: AdamState
    var a_w2: AdamState
    var a_b2: AdamState
    var a_w3: AdamState
    var a_b3: AdamState

    def __init__(out self, ctx: DeviceContext, max_batch: Int) raises:
        self.net = Actor(ctx, max_batch, HIDDEN)
        self.grads = CriticGrads(ctx, OBS_DIM, HIDDEN, NUM_ACTIONS)
        self.scratch = CriticScratch(ctx, max_batch, OBS_DIM, HIDDEN,
                                     NUM_ACTIONS)
        self.a_w1 = AdamState(ctx, OBS_DIM * HIDDEN)
        self.a_b1 = AdamState(ctx, HIDDEN)
        self.a_w2 = AdamState(ctx, HIDDEN * HIDDEN)
        self.a_b2 = AdamState(ctx, HIDDEN)
        self.a_w3 = AdamState(ctx, HIDDEN * NUM_ACTIONS)
        self.a_b3 = AdamState(ctx, NUM_ACTIONS)


def init_actor_weights(ctx: DeviceContext, mut actor: ActorLearner,
                       seed: UInt32) raises:
    """Pesos tipo He y biases a cero, igual que el critico.

    Con biases a cero y pesos centrados, la politica inicial es practicamente
    uniforme sobre las legales -- que es exactamente el prior con el que la
    busqueda ha estado trabajando hasta ahora. O sea que en E2.5 el actor entrara
    sin dar un salto brusco.
    """
    fan_ins = List[Int]()
    fan_ins.append(OBS_DIM); fan_ins.append(HIDDEN); fan_ins.append(HIDDEN)
    sizes = List[Int]()
    sizes.append(OBS_DIM * HIDDEN); sizes.append(HIDDEN * HIDDEN)
    sizes.append(HIDDEN * NUM_ACTIONS)

    for layer in range(3):
        n = sizes[layer]
        scale = Scalar[dtype](1) / Scalar[dtype](fan_ins[layer]) ** 0.5
        buf = zero_buffer[dtype](ctx, n)
        ctx.enqueue_function[fill_uniform, fill_uniform](
            buf.unsafe_ptr(), seed, UInt32(900 + layer), n,
            grid_dim=(n + 255) // 256, block_dim=256)
        ctx.synchronize()
        u = download[dtype](buf, n)
        vals = List[Scalar[dtype]]()
        for i in range(n):
            vals.append((u[i] * Scalar[dtype](2) - Scalar[dtype](1)) * scale)
        if layer == 0:
            write_into[dtype](actor.net.params.w1, vals)
        elif layer == 1:
            write_into[dtype](actor.net.params.w2, vals)
        else:
            write_into[dtype](actor.net.params.w3, vals)
    ctx.synchronize()


def collect(ctx: DeviceContext, mut buf: TrajectoryBuffer, cfg: SPOConfig,
            model: TicTacToe, amodel: TicTacToeActor, use_actor: Bool,
            ws: SearchWorkspace, state: DeviceBuffer[dtype],
            obs_buf: DeviceBuffer[dtype], next_obs_buf: DeviceBuffer[dtype],
            q_buf: DeviceBuffer[dtype], logits_buf: DeviceBuffer[dtype],
            reward: DeviceBuffer[dtype], done: DeviceBuffer[idx_dtype],
            u_rival: DeviceBuffer[dtype], u_readout: DeviceBuffer[dtype],
            seed: UInt32,
            round_idx: Int) raises -> Scalar[dtype]:
    """Juega ROLLOUT turnos y guarda observaciones, recompensas Y la q.

    La diferencia con el `collect` de E1.10 es esa q: es el objetivo del actor, y
    hay que capturarla EN EL MOMENTO, porque el workspace de la busqueda se
    reutiliza y en el turno siguiente ya no esta.

    Como alli, `next_obs` se toma DESPUES del paso pero ANTES del auto-reset: si se
    cogiera despues, el bootstrap miraria a un tablero vacio de otra partida.
    """
    blocks = (NUM_ENVS + TPB_TTT - 1) // TPB_TTT

    obs_hist = List[Scalar[dtype]]()
    next_hist = List[Scalar[dtype]]()
    q_hist = List[Scalar[dtype]]()
    rew_hist = List[Scalar[dtype]]()
    done_hist = List[Scalar[dtype]]()
    for _ in range(NUM_ENVS * ROLLOUT * OBS_DIM):
        obs_hist.append(Scalar[dtype](0))
        next_hist.append(Scalar[dtype](0))
    for _ in range(NUM_ENVS * ROLLOUT * NUM_ACTIONS):
        q_hist.append(Scalar[dtype](0))
    for _ in range(NUM_ENVS * ROLLOUT):
        rew_hist.append(Scalar[dtype](0))
        done_hist.append(Scalar[dtype](0))

    finished = 0
    score_sum = Scalar[dtype](0)

    for t in range(ROLLOUT):
        stream = UInt32(round_idx * ROLLOUT + t)

        # La observacion ANTES de mover: es la que va con esta q.
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            obs_buf.unsafe_ptr(), state.unsafe_ptr(), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)

        # Con `use_actor`, la busqueda parte del prior de la red en vez de
        # uniforme: es lo que cierra el bucle EM.
        if use_actor:
            search[TicTacToeActor](ctx, ws, cfg, amodel, state,
                                   seed ^ (stream * 2654435761))
        else:
            search[TicTacToe](ctx, ws, cfg, model, state,
                              seed ^ (stream * 2654435761))
        # El readout de la variante deja q densa en q_buf Y elige la accion.
        # Se sortea de q (greedy=False) en vez de coger la moda: ese sorteo ES la
        # exploracion de SPO, y sin el el buffer solo veria una linea de juego.
        # Uniformes FRESCOS: reutilizar los que dejo la busqueda haria que el
        # sorteo estuviera correlacionado con el muestreo de particulas.
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_readout.unsafe_ptr(), seed, RNG_READOUT + stream, NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        readout_expected(ctx, ws.particles, ws.output, cfg, logits_buf, q_buf,
                         u_readout, False)
        ctx.synchronize()

        obs_now = download[dtype](obs_buf, NUM_ENVS * OBS_DIM)
        q_now = download[dtype](q_buf, NUM_ENVS * NUM_ACTIONS)

        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), seed, RNG_RIVAL + stream, NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), ws.output.action.unsafe_ptr(),
            u_rival.unsafe_ptr(), reward.unsafe_ptr(), done.unsafe_ptr(),
            NUM_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            next_obs_buf.unsafe_ptr(), state.unsafe_ptr(), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

        next_now = download[dtype](next_obs_buf, NUM_ENVS * OBS_DIM)
        rew_now = download[dtype](reward, NUM_ENVS)
        done_now = download[idx_dtype](done, NUM_ENVS)

        for e in range(NUM_ENVS):
            for k in range(OBS_DIM):
                obs_hist[(e * ROLLOUT + t) * OBS_DIM + k] = obs_now[e * OBS_DIM + k]
                next_hist[(e * ROLLOUT + t) * OBS_DIM + k] = next_now[e * OBS_DIM + k]
            for k in range(NUM_ACTIONS):
                q_hist[(e * ROLLOUT + t) * NUM_ACTIONS + k] = \
                    q_now[e * NUM_ACTIONS + k]
            rew_hist[e * ROLLOUT + t] = rew_now[e]
            d = Scalar[dtype](1) if Int(done_now[e]) != 0 else Scalar[dtype](0)
            done_hist[e * ROLLOUT + t] = d
            if d != 0:
                finished += 1
                score_sum += rew_now[e]

        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)

    # Una secuencia por env.
    for e in range(NUM_ENVS):
        seq_obs = List[Scalar[dtype]]()
        seq_next = List[Scalar[dtype]]()
        seq_q = List[Scalar[dtype]]()
        seq_r = List[Scalar[dtype]]()
        seq_d = List[Scalar[dtype]]()
        seq_tr = List[Scalar[dtype]]()
        for t in range(ROLLOUT):
            for k in range(OBS_DIM):
                seq_obs.append(obs_hist[(e * ROLLOUT + t) * OBS_DIM + k])
                seq_next.append(next_hist[(e * ROLLOUT + t) * OBS_DIM + k])
            for k in range(NUM_ACTIONS):
                seq_q.append(q_hist[(e * ROLLOUT + t) * NUM_ACTIONS + k])
            seq_r.append(rew_hist[e * ROLLOUT + t])
            seq_d.append(done_hist[e * ROLLOUT + t])
            seq_tr.append(Scalar[dtype](0))
        buf.add(seq_obs, seq_r, seq_d, seq_tr, seq_next, seq_q)

    return score_sum / Scalar[dtype](finished) if finished > 0 \
        else Scalar[dtype](0)


def states_from_buffer(buf: TrajectoryBuffer, n_want: Int, seed: UInt32,
                       stream: UInt32) raises -> List[Scalar[dtype]]:
    """`n_want` tableros sacados del buffer, reconstruidos desde la observacion.

    Hace falta porque el suelo de la KL tiene que medirse sobre la MISMA
    distribucion de estados con la que se entrena. Medirlo sobre posiciones
    frescas da un numero que no es comparable -- me paso: sali con un suelo de
    1.21 y un actor a 0.94, o sea el actor "por debajo del suelo", que es
    imposible y solo significaba que estaba comparando dos cosas distintas.
    """
    idx = buf.sample_indices(n_want, seed, stream)
    obs = buf.gather(idx)
    span = buf.t_len * buf.obs_dim
    out = List[Scalar[dtype]]()
    for k in range(n_want):
        # Un paso al azar dentro de la secuencia, para no coger siempre el t=0
        # (que es casi siempre el tablero vacio).
        t = Int((seed + UInt32(k) * 2654435761) % UInt32(buf.t_len))
        base = k * span + t * buf.obs_dim
        for c in range(NUM_CELLS):
            mine = obs[base + c]
            theirs = obs[base + NUM_CELLS + c]
            if mine != 0:
                out.append(Scalar[dtype](1))
            elif theirs != 0:
                out.append(Scalar[dtype](-1))
            else:
                out.append(Scalar[dtype](0))
    return out^


def measure_q_noise(ctx: DeviceContext, cfg: SPOConfig, model: TicTacToe,
                    amodel: TicTacToeActor, use_actor: Bool,
                    ws: SearchWorkspace, state: DeviceBuffer[dtype],
                    q_buf: DeviceBuffer[dtype], logits_buf: DeviceBuffer[dtype],
                    u_readout: DeviceBuffer[dtype], reps: Int,
                    seed: UInt32) raises -> Scalar[dtype]:
    """El SUELO de la KL: cuanto ruido tiene q como objetivo.

    Hace falta por la misma razon que hizo falta separar H(q) de la entropia
    cruzada. La q que el actor imita sale de una busqueda con particulas
    ALEATORIAS: la misma posicion buscada dos veces da q distintas. Un actor es una
    funcion determinista del estado, asi que lo mejor que puede aprender es la
    MEDIA de esas q, y su KL no bajara del ruido que las separa.

    **La forma de calcularlo importa, y me equivoque dos veces antes de dar con
    ella.** El suelo es E_k[KL(q_k || q_media)], y con pi = q_media eso vale

        suelo = H(q_media) - E_k[H(q_k)]

    Esa segunda forma es la que se usa aqui, y es la unica estable. Calcularlo como
    KL directa exige estimar bien q_media, y con q casi one-hot sobre 5-9 casillas
    eso pide muchisimas repeticiones: si la casilla de q_k aparece en pocas de las
    demas, el log de una probabilidad diminuta infla el resultado sin limite (con
    leave-one-out y 12 repeticiones me salio un suelo de 1.33, por encima de la KL
    del actor, o sea imposible). Con la diferencia de entropias no hay logaritmos
    de valores diminutos y el numero es estable.

    Queda un sesgo: H(q_media) se estima con `reps` muestras y sale algo bajo, asi
    que el suelo tambien. Se mitiga subiendo reps, y main() lo mide con dos valores
    distintos para que se vea si el numero se mueve.
    """
    all_q = List[Scalar[dtype]]()
    n = cfg.num_envs * NUM_ACTIONS
    blocks = (cfg.num_envs + TPB_TTT - 1) // TPB_TTT

    for k in range(reps):
        # Con EL MISMO modelo que genero los datos de entrenamiento. Medirlo con
        # otro da un suelo de otra distribucion de q y el numero no es comparable:
        # me volvio a pasar aqui -- salio un "138% de lo reducible", imposible, y
        # era esto.
        if use_actor:
            search[TicTacToeActor](ctx, ws, cfg, amodel, state,
                                   seed ^ (UInt32(7919 + k) * 2654435761))
        else:
            search[TicTacToe](ctx, ws, cfg, model, state,
                              seed ^ (UInt32(7919 + k) * 2654435761))
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_readout.unsafe_ptr(), seed, UInt32(50000 + k), cfg.num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
        readout_expected(ctx, ws.particles, ws.output, cfg, logits_buf, q_buf,
                         u_readout, False)
        ctx.synchronize()
        qk = download[dtype](q_buf, n)
        for i in range(n):
            all_q.append(qk[i])

    # E_k[H(q_k)]: la entropia media de las q individuales.
    mean_h = Scalar[dtype](0)
    for k in range(reps):
        for e in range(cfg.num_envs):
            acc = Scalar[dtype](0)
            for a in range(NUM_ACTIONS):
                w = all_q[k * n + e * NUM_ACTIONS + a]
                if w > Scalar[dtype](0):
                    acc += w * log(w)
            mean_h += -acc
    mean_h /= Scalar[dtype](reps * cfg.num_envs)

    # H(q_media): la entropia de la media.
    h_mean = Scalar[dtype](0)
    for e in range(cfg.num_envs):
        acc = Scalar[dtype](0)
        for a in range(NUM_ACTIONS):
            m = Scalar[dtype](0)
            for k in range(reps):
                m += all_q[k * n + e * NUM_ACTIONS + a]
            m /= Scalar[dtype](reps)
            if m > Scalar[dtype](0):
                acc += m * log(m)
        h_mean += -acc
    h_mean /= Scalar[dtype](cfg.num_envs)

    # Por Jensen H(media) >= media(H), asi que esto no puede salir negativo.
    return h_mean - mean_h


@fieldwise_init
struct Report(Copyable, Movable):
    """Lo que sale de un paso de aprendizaje."""
    var critic_loss: Scalar[dtype]
    var cross_entropy: Scalar[dtype]
    var entropy_q: Scalar[dtype]
    var kl: Scalar[dtype]
    var actor_gnorm: Scalar[dtype]
    var actor_clip: Scalar[dtype]


def update(ctx: DeviceContext, mut critic: Critic, mut actor: ActorLearner,
           buf: TrajectoryBuffer, step: Int, seed: UInt32) raises -> Report:
    """Un paso de gradiente sobre las DOS redes, cada una con su optimizador."""
    idx = buf.sample_indices(BATCH, seed, UInt32(step))
    n = BATCH * ROLLOUT

    obs = upload[dtype](ctx, buf.gather(idx))
    boot = upload[dtype](ctx, buf.gather_bootstrap(idx))
    q = upload[dtype](ctx, buf.gather_q(idx))
    reward = upload[dtype](ctx, buf.gather_steps(idx, 0))
    done_host = buf.gather_steps(idx, 1)
    trunc = upload[dtype](ctx, buf.gather_steps(idx, 2))

    disc_host = List[Scalar[dtype]]()
    for i in range(n):
        disc_host.append((Scalar[dtype](1) - done_host[i]) * GAMMA)
    discount = upload[dtype](ctx, disc_host)

    # ---------------- el critico, igual que en E1.10 ----------------
    v_tm1 = zero_buffer[dtype](ctx, n)
    v_t = zero_buffer[dtype](ctx, n)
    critic_forward(ctx, critic.target, critic.tcache, obs, n)
    ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
        v_tm1.unsafe_ptr(), critic.tcache.value.unsafe_ptr(), n,
        grid_dim=(n + 255) // 256, block_dim=256)
    critic_forward(ctx, critic.target, critic.tcache, boot, n)
    ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
        v_t.unsafe_ptr(), critic.tcache.value.unsafe_ptr(), n,
        grid_dim=(n + 255) // 256, block_dim=256)

    adv = zero_buffer[dtype](ctx, n)
    targets = zero_buffer[dtype](ctx, n)
    truncated_gae(ctx, adv, targets, reward, discount, v_tm1, v_t, trunc,
                  BATCH, ROLLOUT, GAE_LAMBDA)

    critic_forward(ctx, critic.online, critic.cache, obs, n)
    ctx.synchronize()
    pred = download[dtype](critic.cache.value, n)
    tgt = download[dtype](targets, n)
    c_loss = Scalar[dtype](0)
    for i in range(n):
        d = pred[i] - tgt[i]
        c_loss += Scalar[dtype](0.5) * d * d
    c_loss /= Scalar[dtype](n)

    critic_backward(ctx, critic.online, critic.cache, critic.grads,
                    critic.scratch, obs, targets, n)
    ctx.synchronize()

    c_sq = (sum_squares(ctx, critic.grads.dw1, OBS_DIM * HIDDEN)
            + sum_squares(ctx, critic.grads.db1, HIDDEN)
            + sum_squares(ctx, critic.grads.dw2, HIDDEN * HIDDEN)
            + sum_squares(ctx, critic.grads.db2, HIDDEN)
            + sum_squares(ctx, critic.grads.dw3, HIDDEN)
            + sum_squares(ctx, critic.grads.db3, 1))
    c_scale = global_clip_scale(c_sq, MAX_GRAD_NORM)

    adam_step(ctx, critic.online.w1, critic.grads.dw1, critic.a_w1,
              OBS_DIM * HIDDEN, CRITIC_LR, c_scale, step)
    adam_step(ctx, critic.online.b1, critic.grads.db1, critic.a_b1,
              HIDDEN, CRITIC_LR, c_scale, step)
    adam_step(ctx, critic.online.w2, critic.grads.dw2, critic.a_w2,
              HIDDEN * HIDDEN, CRITIC_LR, c_scale, step)
    adam_step(ctx, critic.online.b2, critic.grads.db2, critic.a_b2,
              HIDDEN, CRITIC_LR, c_scale, step)
    adam_step(ctx, critic.online.w3, critic.grads.dw3, critic.a_w3,
              HIDDEN, CRITIC_LR, c_scale, step)
    adam_step(ctx, critic.online.b3, critic.grads.db3, critic.a_b3,
              1, CRITIC_LR, c_scale, step)

    ema_update(ctx, critic.target.w1, critic.online.w1, OBS_DIM * HIDDEN, TAU)
    ema_update(ctx, critic.target.b1, critic.online.b1, HIDDEN, TAU)
    ema_update(ctx, critic.target.w2, critic.online.w2, HIDDEN * HIDDEN, TAU)
    ema_update(ctx, critic.target.b2, critic.online.b2, HIDDEN, TAU)
    ema_update(ctx, critic.target.w3, critic.online.w3, HIDDEN, TAU)
    ema_update(ctx, critic.target.b3, critic.online.b3, 1, TAU)

    # ---------------- el actor ----------------
    # La mascara sale de la observacion, que es lo unico que el buffer guarda.
    ctx.enqueue_function[ttt_legal_mask_from_obs_kernel,
                         ttt_legal_mask_from_obs_kernel](
        actor.net.mask.unsafe_ptr(), obs.unsafe_ptr(), n,
        grid_dim=(n + 255) // 256, block_dim=256)
    actor_probs(ctx, actor.net.params, actor.net.cache, obs, actor.net.mask,
                actor.net.probs, n)

    # Perdida y diagnostico. La entropia cruzada se descompone en H(q) + KL, y
    # solo la KL depende del actor.
    log_pi = zero_buffer[dtype](ctx, n * NUM_ACTIONS)
    ce = zero_buffer[dtype](ctx, n)
    hq = zero_buffer[dtype](ctx, n)
    kl = zero_buffer[dtype](ctx, n)
    ctx.enqueue_function[log_softmax_rows[32], log_softmax_rows[32]](
        log_pi.unsafe_ptr(), actor.net.cache.value.unsafe_ptr(), NUM_ACTIONS,
        grid_dim=n, block_dim=32)
    cross_entropy_rows(ctx, ce, q, log_pi, n, NUM_ACTIONS)
    entropy_rows(ctx, hq, q, n, NUM_ACTIONS)
    kl_rows(ctx, kl, ce, hq, n)
    ctx.synchronize()

    ce_h = download[dtype](ce, n)
    hq_h = download[dtype](hq, n)
    kl_h = download[dtype](kl, n)
    ce_m = Scalar[dtype](0); hq_m = Scalar[dtype](0); kl_m = Scalar[dtype](0)
    for i in range(n):
        ce_m += ce_h[i]; hq_m += hq_h[i]; kl_m += kl_h[i]
    ce_m /= Scalar[dtype](n); hq_m /= Scalar[dtype](n); kl_m /= Scalar[dtype](n)

    actor_backward(ctx, actor.net.params, actor.net.cache, actor.grads,
                   actor.scratch, obs, actor.net.probs, q, actor.net.mask, n)
    ctx.synchronize()

    # Clip GLOBAL propio: la norma del actor no debe recortar al critico ni al
    # reves. Es lo que hace Stoix al encadenar el clip dentro de cada optax.
    a_sq = (sum_squares(ctx, actor.grads.dw1, OBS_DIM * HIDDEN)
            + sum_squares(ctx, actor.grads.db1, HIDDEN)
            + sum_squares(ctx, actor.grads.dw2, HIDDEN * HIDDEN)
            + sum_squares(ctx, actor.grads.db2, HIDDEN)
            + sum_squares(ctx, actor.grads.dw3, HIDDEN * NUM_ACTIONS)
            + sum_squares(ctx, actor.grads.db3, NUM_ACTIONS))
    a_scale = global_clip_scale(a_sq, MAX_GRAD_NORM)
    a_norm = a_sq ** 0.5

    adam_step(ctx, actor.net.params.w1, actor.grads.dw1, actor.a_w1,
              OBS_DIM * HIDDEN, ACTOR_LR, a_scale, step)
    adam_step(ctx, actor.net.params.b1, actor.grads.db1, actor.a_b1,
              HIDDEN, ACTOR_LR, a_scale, step)
    adam_step(ctx, actor.net.params.w2, actor.grads.dw2, actor.a_w2,
              HIDDEN * HIDDEN, ACTOR_LR, a_scale, step)
    adam_step(ctx, actor.net.params.b2, actor.grads.db2, actor.a_b2,
              HIDDEN, ACTOR_LR, a_scale, step)
    adam_step(ctx, actor.net.params.w3, actor.grads.dw3, actor.a_w3,
              HIDDEN * NUM_ACTIONS, ACTOR_LR, a_scale, step)
    adam_step(ctx, actor.net.params.b3, actor.grads.db3, actor.a_b3,
              NUM_ACTIONS, ACTOR_LR, a_scale, step)
    ctx.synchronize()

    return Report(c_loss, ce_m, hq_m, kl_m, a_norm, a_scale)


@fieldwise_init
struct ArmResult(Copyable, Movable):
    var name: String
    var kl_first: Scalar[dtype]
    var kl_last: Scalar[dtype]
    var floor: Scalar[dtype]
    var critic_first: Scalar[dtype]
    var critic_last: Scalar[dtype]
    var score: Scalar[dtype]
    var hq_first: Scalar[dtype]
    var hq_last: Scalar[dtype]


def train_run(ctx: DeviceContext, name: String,
              use_actor: Bool) raises -> ArmResult:
    """Un brazo completo: entrena actor y critico, con o sin prior aprendido.

    Los dos brazos comparten semilla, config y numero de pasos. Lo unico que
    cambia es de donde sale el prior de la busqueda, asi que la diferencia se
    puede atribuir a eso y no a otra cosa.
    """
    cfg = SPOConfig(num_envs=NUM_ENVS, num_particles=NUM_PARTICLES,
                    num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                    search_depth=SEARCH_DEPTH, resample_period=NO_RESAMPLE,
                    temperature=TEMPERATURE, search_gamma=1.0,
                    search_gae_lambda=1.0)
    model = TicTacToe(REWARD_GAMMA, LOSS_PENALTY)
    ws = SearchWorkspace(ctx, cfg)

    n_rows = BATCH * ROLLOUT
    critic = Critic(ctx, n_rows)
    init_critic_weights(ctx, critic, SEED)
    actor = ActorLearner(ctx, n_rows)
    init_actor_weights(ctx, actor, SEED)
    amodel = TicTacToeActor(ctx, cfg.num_search_particles(), HIDDEN,
                            REWARD_GAMMA, LOSS_PENALTY)
    amodel.sync_from(ctx, actor.net.params)
    buf = TrajectoryBuffer(BUFFER_CAP, ROLLOUT, OBS_DIM, NUM_ACTIONS)

    blocks = (NUM_ENVS + TPB_TTT - 1) // TPB_TTT
    state = zero_buffer[dtype](ctx, NUM_ENVS * STATE_DIM)
    obs_buf = zero_buffer[dtype](ctx, NUM_ENVS * OBS_DIM)
    next_obs_buf = zero_buffer[dtype](ctx, NUM_ENVS * OBS_DIM)
    q_buf = zero_buffer[dtype](ctx, NUM_ENVS * NUM_ACTIONS)
    logits_buf = zero_buffer[dtype](ctx, NUM_ENVS * NUM_ACTIONS)
    reward = zero_buffer[dtype](ctx, NUM_ENVS)
    done = zero_buffer[idx_dtype](ctx, NUM_ENVS)
    u_rival = zero_buffer[dtype](ctx, NUM_ENVS)
    u_readout = zero_buffer[dtype](ctx, NUM_ENVS)
    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), NUM_ENVS, grid_dim=blocks, block_dim=TPB_TTT)

    print("--- brazo:", name, "---")
    print("  ronda   score    critico      H(q)       KL      |g_actor|")
    step = 0
    first = Report(0, 0, 0, 0, 0, 0)
    last = Report(0, 0, 0, 0, 0, 0)
    last_score = Scalar[dtype](0)
    for round_idx in range(TRAIN_ROUNDS):
        score = collect(ctx, buf, cfg, model, amodel, use_actor, ws, state,
                        obs_buf, next_obs_buf, q_buf, logits_buf, reward, done,
                        u_rival, u_readout, SEED, round_idx)
        last_score = score
        for e in range(UPDATES_PER_ROUND):
            step += 1
            r = update(ctx, critic, actor, buf, step, SEED)
            if round_idx == 0 and e == 0:
                first = r.copy()
            last = r.copy()
        # AQUI se cierra el bucle: la busqueda de la ronda siguiente vera al actor
        # que se acaba de entrenar. Sin esta linea el M-step entrenaria una red que
        # nadie usa.
        if use_actor:
            amodel.sync_from(ctx, actor.net.params)
            ctx.synchronize()
        if round_idx % 6 == 0 or round_idx == TRAIN_ROUNDS - 1:
            print("   ", round_idx, "  ", score, "  ", last.critic_loss,
                  "  ", last.entropy_q, "  ", last.kl, "  ", last.actor_gnorm)

    st_vals = states_from_buffer(buf, NUM_ENVS, SEED, UInt32(999))
    write_into[dtype](state, st_vals)
    ctx.synchronize()
    floor = measure_q_noise(ctx, cfg, model, amodel, use_actor, ws, state,
                            q_buf, logits_buf, u_readout, 48, SEED)
    print()
    return ArmResult(name, first.kl, last.kl, floor, first.critic_loss,
                     last.critic_loss, last_score, first.entropy_q,
                     last.entropy_q)


def show(r: ArmResult) raises:
    reducible = r.kl_first - r.floor
    frac = (r.kl_first - r.kl_last) / reducible if reducible > 0 \
           else Scalar[dtype](0)
    print("  ", r.name)
    print("      critico ", r.critic_first, " -> ", r.critic_last)
    print("      KL ", r.kl_first, " -> ", r.kl_last, "   suelo ", r.floor,
          "   recorrido ", frac * Scalar[dtype](100), "% de lo reducible")
    print("      score de la busqueda al final: ", r.score)
    print("      H(q) ", r.hq_first, " -> ", r.hq_last,
          "   <- si esto cambia, las KL de los dos brazos miden objetivos "
          "distintos")


def main() raises:
    with DeviceContext() as ctx:
        print("=== E2.5: el prior del actor entra en la busqueda ===")
        print("   envs", NUM_ENVS, " rollout", ROLLOUT, " red", HIDDEN,
              " batch", BATCH, " lr", ACTOR_LR)
        print("   busqueda: N", NUM_PARTICLES, " profundidad", SEARCH_DEPTH,
              " sin remuestreo, readout de media")
        print("   Los dos brazos comparten semilla y pasos; lo unico que cambia")
        print("   es de donde sale el prior.")
        print()

        uniform = train_run(ctx, String("prior UNIFORME (E2.4)"), False)
        learned = train_run(ctx, String("prior del ACTOR (bucle EM cerrado)"),
                            True)

        print("=== comparacion ===")
        show(uniform)
        show(learned)
        print()
        print("   El score es contra rival aleatorio, con la accion sorteada de q")
        print("   (no la moda), asi que no es comparable con el 0.9936 de E1.11c,")
        print("   que se midio jugando la moda. La medicion buena, con partidas")
        print("   suficientes y las cuatro celdas del 2x2, es E2.6.")
