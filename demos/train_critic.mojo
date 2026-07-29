"""El bucle de entrenamiento del critico: jugar, guardar, aprender.

    ./run.sh demos/train_critic.mojo

Junta todo lo de la etapa 1. Dos fases que se alternan, como en Stoix:

    ACTUAR   jugar partidas con la busqueda y guardar las transiciones
    APRENDER muestrear del buffer y dar pasos de gradiente sobre el critico

El orden de la fase de aprender es el de `_critic_loss_fn` de ff_spo.py, y hay un
detalle facil de confundir: **se usan DOS redes**.

    pred   = critico ONLINE (obs)             <- lo que se entrena
    v_tm1  = critico TARGET (obs)             |
    v_t    = critico TARGET (bootstrap_obs)   |-- para calcular los objetivos
    targets = GAE(reward, (1-done)*gamma, lambda, v_tm1, v_t, truncated)
    loss   = media de 0.5*(pred - targets)^2

Si se usara la online para los objetivos, el critico perseguiria un blanco que se
mueve a la vez que el: por eso existe la copia lenta (la EMA de E1.7).

En esta etapa la busqueda sigue con V = 0. El critico aprende MIRANDO las partidas
que genera esa busqueda, pero todavia no la alimenta; enchufarlo es E1.11, y ahi
es donde se medira si sirve para algo.

Se miden dos cosas, y la segunda importa mas:

  1. **¿baja la perdida?** Un critico que no reduce su error no ha aprendido nada.
  2. **¿su valor SEPARA las partidas ganadas de las perdidas?** Porque la perdida
     tambien bajaria si el critico se limitara a predecir siempre la media.

La segunda medida no usa correlacion sino la diferencia de medias por resultado, y
es a proposito: la busqueda gana el ~97% de las partidas, asi que el resultado es
casi constante y correlacionar contra algo casi constante mide ruido. Con clases
tan desbalanceadas, comparar medias por grupo aguanta mejor.

Una expectativa que conviene tener antes de mirar los numeros: el rival juega AL
AZAR, asi que el resultado de una partida concreta tiene mucho de suerte. Una
posicion buena puede acabar en derrota si el rival acierta. El critico deberia
predecir el valor ESPERADO, no el resultado concreto, asi que la separacion entre
ganadas y perdidas sera pequena por construccion.

Por eso la separacion se reporta con su ERROR ESTANDAR y en sigmas: con clases tan
desbalanceadas (~97% de victorias) una diferencia de medias pequena puede ser puro
azar. Midiendo con pocas muestras salia 1.7 sigmas (no afirmable) y con 6.7 veces
mas, 2.5 sigmas (afirmable) — el efecto estaba, faltaban datos para verlo.

Resultados con la configuracion actual (25 rondas, 500 updates):

    perdida            0.41  ->  0.017          (25 veces menor)
    V medio            0.00  ->  0.9346         (coincide con el valor esperado)
    separacion         0.0009 -> 0.0072 +/- 0.0029  (2.47 sigmas: real)

La separacion es REAL pero pequena: V queda casi constante entre posiciones. Si eso
basta para que la busqueda vea amenazas es otra pregunta, y se responde en E1.11
midiendo el 2.02% de derrotas, no suponiendo.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.copy import copy_kernel
from ops.rng import fill_uniform
from envs.tictactoe import (TicTacToe, ttt_reset_kernel, ttt_env_step_kernel,
                            ttt_auto_reset_kernel, ttt_encode_obs_kernel,
                            NUM_ACTIONS, STATE_DIM, OBS_DIM, TPB_TTT)
from envs.tictactoe_runner import RNG_RIVAL
from networks.mlp import (CriticParams, CriticCache, CriticGrads, CriticScratch,
                          critic_forward, critic_backward, zero_critic_params)
from networks.optim import (AdamState, adam_step, ema_update, sum_squares,
                            global_clip_scale)
from rl_utils.buffer import TrajectoryBuffer
from rl_utils.multistep import truncated_gae
from systems.spo.particles import SearchWorkspace
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from tests.helpers import upload, download, write_into

# --- configuracion (los valores de Stoix donde aplica) ---
comptime SEED = UInt32(20260729)
comptime NUM_ENVS = 32
comptime ROLLOUT = 16
"""Pasos por env en cada fase de actuacion. Con partidas de ~4 turnos, cada
secuencia cubre unas 4 partidas completas."""

comptime HIDDEN = 64
comptime OUT_DIM = 1
comptime GAMMA = Scalar[dtype](0.99)
comptime GAE_LAMBDA = Scalar[dtype](0.95)
comptime CRITIC_LR = Scalar[dtype](3e-4)
comptime MAX_GRAD_NORM = Scalar[dtype](0.5)
comptime TAU = Scalar[dtype](0.005)
comptime BATCH = 16
comptime BUFFER_CAP = 256

# La busqueda que genera las partidas: la config afinada en A6/A7.
comptime NUM_PARTICLES = 64
comptime SEARCH_DEPTH = 6
comptime RESAMPLE_PERIOD = 3
comptime TEMPERATURE = Scalar[dtype](0.02)
comptime REWARD_GAMMA = Scalar[dtype](0.7)


struct Critic(Movable):
    """El critico entero: las dos redes, los buffers de trabajo y Adam."""

    var online: CriticParams
    var target: CriticParams
    var cache: CriticCache
    """Para el forward de la red online (el que se entrena)."""
    var tcache: CriticCache
    """Para los dos forwards de la red target."""
    var grads: CriticGrads
    var scratch: CriticScratch

    var a_w1: AdamState
    var a_b1: AdamState
    var a_w2: AdamState
    var a_b2: AdamState
    var a_w3: AdamState
    var a_b3: AdamState

    def __init__(out self, ctx: DeviceContext, max_batch: Int) raises:
        self.online = zero_critic_params(ctx, OBS_DIM, HIDDEN, OUT_DIM)
        self.target = zero_critic_params(ctx, OBS_DIM, HIDDEN, OUT_DIM)
        self.cache = CriticCache(ctx, max_batch, HIDDEN, OUT_DIM)
        self.tcache = CriticCache(ctx, max_batch, HIDDEN, OUT_DIM)
        self.grads = CriticGrads(ctx, OBS_DIM, HIDDEN, OUT_DIM)
        self.scratch = CriticScratch(ctx, max_batch, OBS_DIM, HIDDEN, OUT_DIM)
        self.a_w1 = AdamState(ctx, OBS_DIM * HIDDEN)
        self.a_b1 = AdamState(ctx, HIDDEN)
        self.a_w2 = AdamState(ctx, HIDDEN * HIDDEN)
        self.a_b2 = AdamState(ctx, HIDDEN)
        self.a_w3 = AdamState(ctx, HIDDEN * OUT_DIM)
        self.a_b3 = AdamState(ctx, OUT_DIM)


def init_critic_weights(ctx: DeviceContext, mut critic: Critic,
                        seed: UInt32) raises:
    """Pesos iniciales tipo He y biases a cero; el target arranca igual.

    Que las dos redes empiecen IDENTICAS importa: si difirieran, los objetivos
    estarian sesgados desde el primer update y costaria ver por que.
    """
    fan_ins = List[Int]()
    fan_ins.append(OBS_DIM); fan_ins.append(HIDDEN); fan_ins.append(HIDDEN)
    sizes = List[Int]()
    sizes.append(OBS_DIM * HIDDEN); sizes.append(HIDDEN * HIDDEN)
    sizes.append(HIDDEN * OUT_DIM)

    for layer in range(3):
        n = sizes[layer]
        scale = Scalar[dtype](1) / Scalar[dtype](fan_ins[layer]) ** 0.5
        vals = List[Scalar[dtype]]()
        buf = zero_buffer[dtype](ctx, n)
        ctx.enqueue_function[fill_uniform, fill_uniform](
            buf.unsafe_ptr(), seed, UInt32(500 + layer), n,
            grid_dim=(n + 255) // 256, block_dim=256)
        ctx.synchronize()
        u = download[dtype](buf, n)
        for i in range(n):
            # De U(0,1) a U(-scale, scale): centrado en cero y con la escala de He.
            vals.append((u[i] * Scalar[dtype](2) - Scalar[dtype](1)) * scale)

        if layer == 0:
            write_into[dtype](critic.online.w1, vals)
            write_into[dtype](critic.target.w1, vals)
        elif layer == 1:
            write_into[dtype](critic.online.w2, vals)
            write_into[dtype](critic.target.w2, vals)
        else:
            write_into[dtype](critic.online.w3, vals)
            write_into[dtype](critic.target.w3, vals)
    ctx.synchronize()


def collect(ctx: DeviceContext, mut buf: TrajectoryBuffer, cfg: SPOConfig,
            model: TicTacToe, ws: SearchWorkspace, state: DeviceBuffer[dtype],
            obs_buf: DeviceBuffer[dtype], next_obs_buf: DeviceBuffer[dtype],
            reward: DeviceBuffer[dtype], done: DeviceBuffer[idx_dtype],
            u_rival: DeviceBuffer[dtype], seed: UInt32,
            round_idx: Int) raises -> Scalar[dtype]:
    """Juega ROLLOUT turnos en NUM_ENVS partidas y guarda las secuencias.

    Devuelve la puntuacion media de las partidas terminadas, para poder ver que
    la busqueda sigue jugando igual de bien mientras el critico aprende.

    Detalle importante: `next_obs` se captura DESPUES del paso pero ANTES del
    auto-reset. Es el `bootstrap_obs` de Stoix: si se cogiera despues del reset,
    el bootstrap miraria a un tablero vacio de una partida nueva.
    """
    blocks = (NUM_ENVS + TPB_TTT - 1) // TPB_TTT

    # Historia por env, en host: [T, ...] por cada uno.
    obs_hist = List[Scalar[dtype]]()
    next_hist = List[Scalar[dtype]]()
    rew_hist = List[Scalar[dtype]]()
    done_hist = List[Scalar[dtype]]()
    for _ in range(NUM_ENVS * ROLLOUT * OBS_DIM):
        obs_hist.append(Scalar[dtype](0))
        next_hist.append(Scalar[dtype](0))
    for _ in range(NUM_ENVS * ROLLOUT):
        rew_hist.append(Scalar[dtype](0))
        done_hist.append(Scalar[dtype](0))

    wins = 0
    draws = 0
    losses = 0

    for t in range(ROLLOUT):
        # La observacion ANTES de decidir.
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            obs_buf.unsafe_ptr(), state.unsafe_ptr(), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)

        step_seed = seed ^ (UInt32(round_idx * 1000 + t) * 2654435761)
        search[TicTacToe](ctx, ws, cfg, model, state, step_seed)

        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), seed, RNG_RIVAL + UInt32(round_idx * 64 + t),
            NUM_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), ws.output.action.unsafe_ptr(),
            u_rival.unsafe_ptr(), reward.unsafe_ptr(), done.unsafe_ptr(),
            NUM_ENVS, grid_dim=blocks, block_dim=TPB_TTT)

        # La observacion siguiente, ANTES del auto-reset.
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            next_obs_buf.unsafe_ptr(), state.unsafe_ptr(), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

        o = download[dtype](obs_buf, NUM_ENVS * OBS_DIM)
        no = download[dtype](next_obs_buf, NUM_ENVS * OBS_DIM)
        r = download[dtype](reward, NUM_ENVS)
        d = download[idx_dtype](done, NUM_ENVS)
        for e in range(NUM_ENVS):
            for k in range(OBS_DIM):
                obs_hist[(e * ROLLOUT + t) * OBS_DIM + k] = o[e * OBS_DIM + k]
                next_hist[(e * ROLLOUT + t) * OBS_DIM + k] = no[e * OBS_DIM + k]
            rew_hist[e * ROLLOUT + t] = r[e]
            done_hist[e * ROLLOUT + t] = Scalar[dtype](1) if Int(d[e]) != 0 \
                                         else Scalar[dtype](0)
            if Int(d[e]) != 0:
                if r[e] > Scalar[dtype](0.75): wins += 1
                elif r[e] > Scalar[dtype](0.25): draws += 1
                else: losses += 1

        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)

    # Cada env aporta una secuencia entera.
    zeros_t = List[Scalar[dtype]]()
    for _ in range(ROLLOUT):
        zeros_t.append(Scalar[dtype](0))      # truncated: en TTT nunca ocurre
    for e in range(NUM_ENVS):
        seq_obs = List[Scalar[dtype]]()
        seq_next = List[Scalar[dtype]]()
        seq_r = List[Scalar[dtype]]()
        seq_d = List[Scalar[dtype]]()
        for t in range(ROLLOUT):
            for k in range(OBS_DIM):
                seq_obs.append(obs_hist[(e * ROLLOUT + t) * OBS_DIM + k])
                seq_next.append(next_hist[(e * ROLLOUT + t) * OBS_DIM + k])
            seq_r.append(rew_hist[e * ROLLOUT + t])
            seq_d.append(done_hist[e * ROLLOUT + t])
        buf.add(seq_obs, seq_r, seq_d, zeros_t, seq_next)

    games = wins + draws + losses
    if games == 0:
        return Scalar[dtype](0)
    return (Scalar[dtype](wins) + Scalar[dtype](0.5) * Scalar[dtype](draws)) \
           / Scalar[dtype](games)


def update(ctx: DeviceContext, mut critic: Critic, buf: TrajectoryBuffer,
           step: Int, seed: UInt32) raises -> Scalar[dtype]:
    """Un paso de gradiente sobre el critico. Devuelve la perdida ANTES del paso."""
    idx = buf.sample_indices(BATCH, seed, UInt32(step))
    n = BATCH * ROLLOUT          # filas de red: cada paso de cada secuencia

    obs = upload[dtype](ctx, buf.gather(idx))
    boot = upload[dtype](ctx, buf.gather_bootstrap(idx))
    reward = upload[dtype](ctx, buf.gather_steps(idx, 0))
    done_host = buf.gather_steps(idx, 1)
    trunc = upload[dtype](ctx, buf.gather_steps(idx, 2))

    # discount = (1 - done) * gamma, tal cual lo monta ff_spo._critic_loss_fn.
    disc_host = List[Scalar[dtype]]()
    for i in range(n):
        disc_host.append((Scalar[dtype](1) - done_host[i]) * GAMMA)
    discount = upload[dtype](ctx, disc_host)

    # 1. Los valores del TARGET, que son los que arman los objetivos.
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

    # 2. Los objetivos, con la GAE truncada.
    adv = zero_buffer[dtype](ctx, n)
    targets = zero_buffer[dtype](ctx, n)
    truncated_gae(ctx, adv, targets, reward, discount, v_tm1, v_t, trunc,
                  BATCH, ROLLOUT, GAE_LAMBDA)

    # 3. La prediccion de la red ONLINE, que es la que se entrena.
    critic_forward(ctx, critic.online, critic.cache, obs, n)
    ctx.synchronize()

    pred = download[dtype](critic.cache.value, n)
    tgt = download[dtype](targets, n)
    loss = Scalar[dtype](0)
    for i in range(n):
        d = pred[i] - tgt[i]
        loss += Scalar[dtype](0.5) * d * d
    loss /= Scalar[dtype](n)

    # 4. Gradientes, clip GLOBAL sobre los seis tensores, y Adam.
    critic_backward(ctx, critic.online, critic.cache, critic.grads,
                    critic.scratch, obs, targets, n)
    ctx.synchronize()

    total_sq = (sum_squares(ctx, critic.grads.dw1, OBS_DIM * HIDDEN)
                + sum_squares(ctx, critic.grads.db1, HIDDEN)
                + sum_squares(ctx, critic.grads.dw2, HIDDEN * HIDDEN)
                + sum_squares(ctx, critic.grads.db2, HIDDEN)
                + sum_squares(ctx, critic.grads.dw3, HIDDEN * OUT_DIM)
                + sum_squares(ctx, critic.grads.db3, OUT_DIM))
    scale = global_clip_scale(total_sq, MAX_GRAD_NORM)

    adam_step(ctx, critic.online.w1, critic.grads.dw1, critic.a_w1,
              OBS_DIM * HIDDEN, CRITIC_LR, scale, step)
    adam_step(ctx, critic.online.b1, critic.grads.db1, critic.a_b1,
              HIDDEN, CRITIC_LR, scale, step)
    adam_step(ctx, critic.online.w2, critic.grads.dw2, critic.a_w2,
              HIDDEN * HIDDEN, CRITIC_LR, scale, step)
    adam_step(ctx, critic.online.b2, critic.grads.db2, critic.a_b2,
              HIDDEN, CRITIC_LR, scale, step)
    adam_step(ctx, critic.online.w3, critic.grads.dw3, critic.a_w3,
              HIDDEN * OUT_DIM, CRITIC_LR, scale, step)
    adam_step(ctx, critic.online.b3, critic.grads.db3, critic.a_b3,
              OUT_DIM, CRITIC_LR, scale, step)

    # 5. Y el target se acerca un poquito al online.
    ema_update(ctx, critic.target.w1, critic.online.w1, OBS_DIM * HIDDEN, TAU)
    ema_update(ctx, critic.target.b1, critic.online.b1, HIDDEN, TAU)
    ema_update(ctx, critic.target.w2, critic.online.w2, HIDDEN * HIDDEN, TAU)
    ema_update(ctx, critic.target.b2, critic.online.b2, HIDDEN, TAU)
    ema_update(ctx, critic.target.w3, critic.online.w3, HIDDEN * OUT_DIM, TAU)
    ema_update(ctx, critic.target.b3, critic.online.b3, OUT_DIM, TAU)
    ctx.synchronize()

    return loss


def evaluate(ctx: DeviceContext, mut critic: Critic, cfg: SPOConfig,
             model: TicTacToe, ws: SearchWorkspace, state: DeviceBuffer[dtype],
             obs_buf: DeviceBuffer[dtype], reward: DeviceBuffer[dtype],
             done: DeviceBuffer[idx_dtype], u_rival: DeviceBuffer[dtype],
             eval_cache: CriticCache, steps: Int,
             seed: UInt32) raises -> Scalar[dtype]:
    """¿Predice V el resultado de la partida? Devuelve la correlacion.

    Que la perdida baje NO demuestra que el critico haya aprendido algo util:
    podria estar prediciendo siempre la media y bajaria igual. La prueba de verdad
    es si su valor SEPARA las partidas que se ganan de las que no.

    Se juega, se anota V(s) en cada paso, y cuando la partida acaba se le asigna a
    todos sus pasos el resultado real (1 / 0.5 / 0). Al final se calcula la
    correlacion de Pearson entre las dos series. Cerca de 0 = el critico no
    distingue nada; positiva y alta = predice.
    """
    blocks = (NUM_ENVS + TPB_TTT - 1) // TPB_TTT

    # Por env: los valores predichos de la partida en curso.
    pending = List[Scalar[dtype]]()
    pending_count = List[Int]()
    for _ in range(NUM_ENVS * 16):
        pending.append(Scalar[dtype](0))
    for _ in range(NUM_ENVS):
        pending_count.append(0)

    vs = List[Scalar[dtype]]()      # V(s) de pasos de partidas ya terminadas
    outs = List[Scalar[dtype]]()    # y el resultado real de su partida

    for t in range(steps):
        ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
            obs_buf.unsafe_ptr(), state.unsafe_ptr(), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        # El valor que el critico le da al tablero ACTUAL.
        critic_forward(ctx, critic.online, eval_cache, obs_buf, NUM_ENVS)
        search[TicTacToe](ctx, ws, cfg, model, state,
                          seed ^ (UInt32(9000 + t) * 2654435761))
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), seed, RNG_RIVAL + UInt32(3000 + t), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), ws.output.action.unsafe_ptr(),
            u_rival.unsafe_ptr(), reward.unsafe_ptr(), done.unsafe_ptr(),
            NUM_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

        v = download[dtype](eval_cache.value, NUM_ENVS)
        r = download[dtype](reward, NUM_ENVS)
        d = download[idx_dtype](done, NUM_ENVS)
        for e in range(NUM_ENVS):
            c = pending_count[e]
            if c < 16:
                pending[e * 16 + c] = v[e]
                pending_count[e] = c + 1
            if Int(d[e]) != 0:
                # La partida acabo: todos sus pasos comparten el resultado.
                for k in range(pending_count[e]):
                    vs.append(pending[e * 16 + k])
                    outs.append(r[e])
                pending_count[e] = 0

        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)

    n = len(vs)
    if n < 10:
        print("      (muy pocas partidas para evaluar)")
        return Scalar[dtype](0)

    # Media de V separada por resultado. Se usa esto y NO la correlacion porque
    # las clases estan MUY desbalanceadas: la busqueda gana el ~96%, asi que el
    # resultado es casi constante y correlacionar contra algo casi constante mide
    # ruido, no capacidad predictiva. Comparar medias por grupo si aguanta el
    # desbalanceo.
    sum_win = Scalar[dtype](0); n_win = 0
    sum_draw = Scalar[dtype](0); n_draw = 0
    sum_loss = Scalar[dtype](0); n_loss = 0
    for i in range(n):
        if outs[i] > Scalar[dtype](0.75):
            sum_win += vs[i]; n_win += 1
        elif outs[i] > Scalar[dtype](0.25):
            sum_draw += vs[i]; n_draw += 1
        else:
            sum_loss += vs[i]; n_loss += 1

    mw = sum_win / Scalar[dtype](n_win) if n_win > 0 else Scalar[dtype](0)
    md = sum_draw / Scalar[dtype](n_draw) if n_draw > 0 else Scalar[dtype](0)
    ml = sum_loss / Scalar[dtype](n_loss) if n_loss > 0 else Scalar[dtype](0)

    # La desviacion de cada grupo, para saber si la diferencia de medias es real o
    # ruido. Con solo ~50 posiciones perdidas de ~1900, una separacion pequena
    # puede ser puro azar, y afirmarla sin comprobarlo seria inventarse un
    # resultado.
    var_w = Scalar[dtype](0)
    var_l = Scalar[dtype](0)
    for i in range(n):
        if outs[i] > Scalar[dtype](0.75):
            dw = vs[i] - mw
            var_w += dw * dw
        elif outs[i] <= Scalar[dtype](0.25):
            dl = vs[i] - ml
            var_l += dl * dl
    if n_win > 1:
        var_w /= Scalar[dtype](n_win - 1)
    if n_loss > 1:
        var_l /= Scalar[dtype](n_loss - 1)

    print("      posiciones:", n, " -> ganadas", n_win, "(V medio", mw, ")",
          " empatadas", n_draw, "(", md, ")", " perdidas", n_loss, "(", ml, ")")

    if n_loss < 2 or n_win < 2:
        return Scalar[dtype](0)

    # Error estandar de la diferencia de medias, y cuantas sigmas es.
    se = (var_w / Scalar[dtype](n_win) + var_l / Scalar[dtype](n_loss)) ** 0.5
    diff = mw - ml
    sigmas = diff / se if se > Scalar[dtype](0) else Scalar[dtype](0)
    print("      separacion", diff, " +/-", se, " -> ", sigmas, "sigmas")
    if sigmas > Scalar[dtype](2):
        print("      (>2 sigmas: la separacion es real, no ruido)")
    else:
        print("      (<2 sigmas: NO se puede afirmar que separe)")
    return diff


def main() raises:
    with DeviceContext() as ctx:
        print("=== entrenamiento del critico sobre tres en raya ===")
        print("  envs", NUM_ENVS, " rollout", ROLLOUT, " red", HIDDEN,
              " batch", BATCH, " lr", CRITIC_LR)
        print()

        cfg = SPOConfig(num_envs=NUM_ENVS, num_particles=NUM_PARTICLES,
                        num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                        search_depth=SEARCH_DEPTH,
                        resample_period=RESAMPLE_PERIOD,
                        temperature=TEMPERATURE, search_gamma=1.0,
                        search_gae_lambda=1.0)
        model = TicTacToe(reward_gamma=REWARD_GAMMA)
        ws = SearchWorkspace(ctx, cfg)

        critic = Critic(ctx, BATCH * ROLLOUT)
        init_critic_weights(ctx, critic, SEED)
        eval_cache = CriticCache(ctx, NUM_ENVS, HIDDEN, OUT_DIM)
        buf = TrajectoryBuffer(BUFFER_CAP, ROLLOUT, OBS_DIM)

        blocks = (NUM_ENVS + TPB_TTT - 1) // TPB_TTT
        state = zero_buffer[dtype](ctx, NUM_ENVS * STATE_DIM)
        obs_buf = zero_buffer[dtype](ctx, NUM_ENVS * OBS_DIM)
        next_obs_buf = zero_buffer[dtype](ctx, NUM_ENVS * OBS_DIM)
        reward = zero_buffer[dtype](ctx, NUM_ENVS)
        done = zero_buffer[idx_dtype](ctx, NUM_ENVS)
        u_rival = zero_buffer[dtype](ctx, NUM_ENVS)
        ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
            state.unsafe_ptr(), NUM_ENVS, grid_dim=blocks, block_dim=TPB_TTT)

        # La correlacion ANTES de entrenar: con pesos al azar deberia ser ~0.
        print("  ANTES de entrenar:")
        corr0 = evaluate(ctx, critic, cfg, model, ws, state, obs_buf, reward,
                         done, u_rival, eval_cache, 400, SEED)
        print("      separacion V(ganadas) - V(perdidas):", corr0)
        print()

        step = 0
        for round_idx in range(25):
            score = collect(ctx, buf, cfg, model, ws, state, obs_buf,
                            next_obs_buf, reward, done, u_rival, SEED, round_idx)

            first = Scalar[dtype](0)
            last = Scalar[dtype](0)
            for e in range(20):
                step += 1
                l = update(ctx, critic, buf, step, SEED)
                if e == 0:
                    first = l
                last = l
            print("  ronda", round_idx, " secuencias", buf.size(),
                  " score de la busqueda", score,
                  " perdida", first, "->", last)

        print()
        print("  DESPUES de entrenar:")
        corr1 = evaluate(ctx, critic, cfg, model, ws, state, obs_buf, reward,
                         done, u_rival, eval_cache, 400, SEED)
        print("      separacion V(ganadas) - V(perdidas):", corr1)
        print()
        print("  separacion antes", corr0, " -> despues", corr1)
        print("  Que la perdida baje no basta: un critico que prediga siempre la")
        print("  media tambien la bajaria. Lo que dice si aprendio algo util es si")
        print("  su valor SEPARA las partidas que se ganan de las que se pierden.")
