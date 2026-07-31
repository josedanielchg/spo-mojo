"""Fase 5 de la busqueda: leer. Aqui sale la politica mejorada q.

Es el puente al M-step, y lo que hay que entender es que `q` no es un array de
probabilidades: son DOS cosas juntas.

    sampled_actions         las N acciones raiz que sobrevivieron, con
                            repeticiones si el resampling copio alguna varias veces
    sampled_action_weights  softmax(peso/temperatura) de cada una

Su histograma ponderado ES q, la ecuacion 6 del paper. La accion que se ejecuta
en el entorno real se sortea de ahi -- se muestrea, no se coge el argmax, y de
ese muestreo sale la exploracion.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from ops.common import dtype, idx_dtype, GlobalF32, GlobalI32
from ops.copy import copy_kernel
from ops.rng import categorical_from_logits
from ops.softmax import softmax_rows
from systems.spo.launch import TPB, TPB_PARTICLES, blocks_for
from systems.spo.particles import Particles, SearchScratch, SPOOutput
from systems.spo.resampling import resample_logits_kernel
from systems.spo.spo_types import SPOConfig


def select_action_kernel(action_out: GlobalI32, root_actions: GlobalI32,
                         chosen: GlobalI32, num_envs: Int, num_particles: Int):
    """La accion final del env es la accion RAIZ de la particula sorteada."""
    env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env < num_envs:
        action_out[env] = root_actions[env * num_particles + Int(chosen[env])]


def mean_over_particles_kernel(out_mean: GlobalF32, values: GlobalF32,
                               num_envs: Int, num_particles: Int):
    """Media por env. num_particles es 16, asi que un hilo por env va sobrado."""
    env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= num_envs:
        return
    total = Scalar[dtype](0)
    for n in range(num_particles):
        total += values[env * num_particles + n]
    out_mean[env] = total / Scalar[dtype](num_particles)


def q_histogram_kernel(q_out: GlobalF32, root_actions: GlobalI32,
                       weights: GlobalF32, num_envs: Int, num_particles: Int,
                       num_actions: Int):
    """q[env, a] = suma de los pesos de las particulas cuya accion raiz es `a`.

    Es la ecuacion 6 del paper escrita como array de verdad: `readout_weighted`
    deja q implicita (acciones + pesos, con repeticiones) porque para muestrear
    basta con eso, pero para coger el maximo hay que agregar por accion primero.

    Un hilo por (env, accion), y cada uno recorre las particulas de su env. Sin
    atomicos: cada hilo escribe en su propia casilla.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= num_envs * num_actions:
        return
    env = i // num_actions
    a = i % num_actions
    total = Scalar[dtype](0)
    for n in range(num_particles):
        if Int(root_actions[env * num_particles + n]) == a:
            total += weights[env * num_particles + n]
    q_out[i] = total


def argmax_action_kernel(action_out: GlobalI32, q: GlobalF32, num_envs: Int,
                         num_actions: Int):
    """La accion con mas masa de q. Un hilo por env.

    Empate resuelto por el indice mas bajo (`>` estricto), que es determinista y
    es lo que hace `jnp.argmax`.
    """
    env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= num_envs:
        return
    best = 0
    best_q = q[env * num_actions]
    for a in range(1, num_actions):
        v = q[env * num_actions + a]
        if v > best_q:
            best_q = v
            best = a
    action_out[env] = Scalar[idx_dtype](best)


def readout_greedy(ctx: DeviceContext, particles: Particles, output: SPOOutput,
                   cfg: SPOConfig, q_buf: DeviceBuffer[dtype]) raises:
    """La MODA de q en vez de una muestra de q. Para EVALUAR, no para entrenar.

    Se llama DESPUES de `search`, y lo unico que hace es pisar `output.action`;
    todo lo demas que produjo la busqueda (los pesos, las ventajas, el valor)
    queda intacto, asi que el M-step seguiria viendo exactamente lo mismo.

    Por que hace falta: `readout_weighted` sortea la accion de q, y ese sorteo ES
    la exploracion del algoritmo. Perfecto mientras se aprende, pero al medir
    fuerza mete a proposito jugadas subobtimas y la medida sale peor de lo que el
    agente sabe jugar. Separar las dos cosas es lo normal en RL (muestrear para
    entrenar, moda para evaluar) y aqui ademas hace falta para comparar de forma
    justa contra un MCTS, que elige su jugada por el maximo de visitas.

    `q_buf` es [num_envs, num_actions].
    """
    n_cells = cfg.num_envs * cfg.num_actions
    ctx.enqueue_function[q_histogram_kernel, q_histogram_kernel](
        q_buf.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        output.sampled_action_weights.unsafe_ptr(), cfg.num_envs,
        cfg.num_particles, cfg.num_actions,
        grid_dim=blocks_for(n_cells), block_dim=TPB)

    ctx.enqueue_function[argmax_action_kernel, argmax_action_kernel](
        output.action.unsafe_ptr(), q_buf.unsafe_ptr(), cfg.num_envs,
        cfg.num_actions, grid_dim=blocks_for(cfg.num_envs), block_dim=TPB)


def action_mean_logits_kernel(logits_out: GlobalF32, root_actions: GlobalI32,
                              raw_weights: GlobalF32, num_envs: Int,
                              num_particles: Int, num_actions: Int,
                              temperature: Scalar[dtype]):
    """logits[env, a] = (media de los pesos de las particulas de `a`) / temperatura.

    La diferencia con `q_histogram_kernel` es dónde entra la exponencial, y esa
    diferencia lo cambia todo. La ecuacion 6 hace

        q(a)  =  SUMA_{p en a}  exp(peso_p / tau)          <- exponencial primero

    y aqui se hace

        q(a)  ∝  exp( MEDIA_{p en a}(peso_p) / tau )       <- media primero

    Con la suma de exponenciales una particula mala aporta ~0, pero es que ya
    aportaba ~0 comparada con una buena: nunca RESTA. Por eso la accion se juzga
    por sus mejores particulas y el riesgo es invisible. Con la media, una
    particula que pierde arrastra a su accion hacia abajo en proporcion a lo
    frecuente que sea.

    Formalmente: es la diferencia entre estimar E[exp(A/tau)] y exp(E[A]/tau).
    Coinciden si el entorno es DETERMINISTA (los del paper); con un rival
    aleatorio no, y la brecha de Jensen es un sesgo optimista.

    Una accion que ninguna particula probo se marca con -inf para que su q sea 0.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= num_envs * num_actions:
        return
    env = i // num_actions
    a = i % num_actions
    total = Scalar[dtype](0)
    count = 0
    for n in range(num_particles):
        if Int(root_actions[env * num_particles + n]) == a:
            total += raw_weights[env * num_particles + n]
            count += 1
    if count == 0:
        logits_out[i] = Scalar[dtype](-1e30)
    else:
        logits_out[i] = (total / Scalar[dtype](count)) / temperature


def readout_expected(ctx: DeviceContext, particles: Particles,
                     output: SPOOutput, cfg: SPOConfig,
                     logits_buf: DeviceBuffer[dtype],
                     q_buf: DeviceBuffer[dtype],
                     uniforms: DeviceBuffer[dtype], greedy: Bool) raises:
    """Readout que promedia por accion ANTES de exponenciar. VARIANTE, no SPO.

    Se aparta a proposito de la ecuacion 6 del paper. Esta aqui porque la
    auditoria de `demos/audit_blunders.mojo` mostro que el readout original no
    puede castigar el riesgo (ver `action_mean_logits_kernel`), y la unica forma
    de DEMOSTRAR que esa es la causa es cambiarlo y ver si el bloqueo sube.

    Deja `output.action` y `q_buf` [num_envs, num_actions]. No toca
    `sampled_action_weights` ni `sampled_advantages`, asi que lo que vería el
    M-step sigue siendo lo de SPO.

    `greedy` elige entre la moda (evaluar) y una muestra (entrenar/explorar).
    """
    n_cells = cfg.num_envs * cfg.num_actions

    ctx.enqueue_function[action_mean_logits_kernel, action_mean_logits_kernel](
        logits_buf.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        particles.resample_td_weights.unsafe_ptr(), cfg.num_envs,
        cfg.num_particles, cfg.num_actions, cfg.temperature,
        grid_dim=blocks_for(n_cells), block_dim=TPB)

    ctx.enqueue_function[softmax_rows[TPB_PARTICLES], softmax_rows[TPB_PARTICLES]](
        q_buf.unsafe_ptr(), logits_buf.unsafe_ptr(), cfg.num_actions,
        grid_dim=cfg.num_envs, block_dim=TPB_PARTICLES)

    if greedy:
        ctx.enqueue_function[argmax_action_kernel, argmax_action_kernel](
            output.action.unsafe_ptr(), q_buf.unsafe_ptr(), cfg.num_envs,
            cfg.num_actions, grid_dim=blocks_for(cfg.num_envs), block_dim=TPB)
    else:
        # Aqui la categorica es sobre ACCIONES, no sobre particulas: el indice que
        # sale ya ES la jugada, sin pasar por root_actions.
        ctx.enqueue_function[categorical_from_logits[TPB_PARTICLES],
                             categorical_from_logits[TPB_PARTICLES]](
            output.action.unsafe_ptr(), logits_buf.unsafe_ptr(),
            uniforms.unsafe_ptr(), cfg.num_actions,
            grid_dim=cfg.num_envs, block_dim=TPB_PARTICLES)


def readout_weighted(ctx: DeviceContext, particles: Particles,
                     scratch: SearchScratch, output: SPOOutput, cfg: SPOConfig,
                     uniforms: DeviceBuffer[dtype]) raises:
    """Lee el resultado de la busqueda. Espeja `readout_weighted` de Stoix.

    `uniforms` solo necesita num_envs valores (uno por env), pero se pasa un
    buffer de P por comodidad: se leen los primeros num_envs.
    """
    p_total = cfg.num_search_particles()
    blocks_p = blocks_for(p_total)

    # logits = peso / temperatura, los mismos que usa el resampling
    ctx.enqueue_function[resample_logits_kernel, resample_logits_kernel](
        scratch.resample_logits.unsafe_ptr(),
        particles.resample_td_weights.unsafe_ptr(),
        p_total, cfg.temperature, grid_dim=blocks_p, block_dim=TPB)

    # pesos normalizados de cada accion raiz
    ctx.enqueue_function[softmax_rows[TPB_PARTICLES], softmax_rows[TPB_PARTICLES]](
        output.sampled_action_weights.unsafe_ptr(),
        scratch.resample_logits.unsafe_ptr(), cfg.num_particles,
        grid_dim=cfg.num_envs, block_dim=TPB_PARTICLES)

    # una particula por env, sorteada con esos pesos
    ctx.enqueue_function[categorical_from_logits[TPB_PARTICLES],
                         categorical_from_logits[TPB_PARTICLES]](
        output.action.unsafe_ptr(), scratch.resample_logits.unsafe_ptr(),
        uniforms.unsafe_ptr(), cfg.num_particles,
        grid_dim=cfg.num_envs, block_dim=TPB_PARTICLES)

    # ...y su accion raiz es la que se ejecuta. Se reusa output.action como
    # destino: entra el indice de particula y sale la accion.
    ctx.enqueue_function[select_action_kernel, select_action_kernel](
        output.action.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        output.action.unsafe_ptr(), cfg.num_envs, cfg.num_particles,
        grid_dim=blocks_for(cfg.num_envs), block_dim=TPB)

    ctx.enqueue_function[copy_kernel[idx_dtype], copy_kernel[idx_dtype]](
        output.sampled_actions.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        p_total, grid_dim=blocks_p, block_dim=TPB)

    ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
        output.sampled_advantages.unsafe_ptr(), particles.gae.unsafe_ptr(),
        p_total, grid_dim=blocks_p, block_dim=TPB)

    # El valor de la raiz, no el del final del rollout: es lo que Stoix mete en
    # SPOOutput.value (jnp.mean(root.particle_values, axis=-1)).
    ctx.enqueue_function[mean_over_particles_kernel, mean_over_particles_kernel](
        output.value.unsafe_ptr(), output.root_values.unsafe_ptr(),
        cfg.num_envs, cfg.num_particles,
        grid_dim=blocks_for(cfg.num_envs), block_dim=TPB)
