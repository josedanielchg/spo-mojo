"""El nucleo de la busqueda SMC. Espeja la clase SPO de stoix/systems/spo/ff_spo.py.

Todo lo de aqui es independiente del modelo: no sabe si detras hay un MDP de
juguete, CartPole o una red neuronal. Lo unico especifico del modelo son los tres
kernels que describe spo_types.mojo, y los llama quien orquesta, no este fichero.

Orden de la busqueda, igual que en Stoix:

    root_fn        siembra N particulas por env desde el prior
    recurrent_fn   avanza las P particulas un paso  (por profundidad)
    weight update  acumula el error TD en el log-peso
    resample       cada `resample_period` pasos
    readout        muestrea la accion final de softmax(w/eta)

Convencion de indices: la particula p = env * num_particles + n. Es plana a
proposito, asi la mayoria de kernels son un map de un hilo por particula y
`env = p // num_particles` sale con una division.
"""

from std.builtin.debug_assert import debug_assert
from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import exp, log
from std.memory import stack_allocation, AddressSpace

from ops.common import dtype, idx_dtype, NEG_INF, GlobalF32, GlobalI32
from ops.reductions import block_reduce_max, block_reduce_sum
from ops.rng import categorical_from_logits
from ops.scan import block_scan_inclusive
from ops.softmax import softmax_rows
from systems.spo.spo_types import (SPOConfig, Particles, StepOutputs,
                                   SearchScratch, SPOOutput)

comptime TPB = 32
"""Bloque de los kernels que son un map plano: un hilo por particula."""

comptime TPB_PARTICLES = 128
"""Bloque de los kernels cuya FILA es la dimension de particulas (resampling,
ESS, softmax del readout). Ahi el bloque entero trabaja sobre las N particulas de
un env, asi que N tiene que caber dentro.

Es 128 y no 32 para que la demo pueda barrer N = 64. Los hilos de mas no cuestan
nada: los guards los desactivan y la GPU no lanza warps enteros ociosos.
Con N = 16 (lo del paper) sobra de largo."""


def check_search_config(cfg: SPOConfig) raises:
    """Comprueba en HOST lo que los kernels no pueden comprobar solos.

    Existe porque este error ya me mordio: con N = 64 y bloques de 32 la busqueda
    devolvia una politica PEOR que con N = 16, sin avisar de nada. El
    debug_assert del kernel lo caza, pero solo si alguien corre con -D ASSERT=all;
    esta comprobacion salta siempre y dice exactamente que hacer."""
    if cfg.num_particles > TPB_PARTICLES:
        raise Error("num_particles=", cfg.num_particles, " no cabe en un bloque de ",
                    TPB_PARTICLES, ". Sube TPB_PARTICLES (potencia de dos) o baja N.")
    if cfg.num_actions > TPB:
        raise Error("num_actions=", cfg.num_actions, " no cabe en un bloque de ", TPB)


def broadcast_state_kernel(particle_state: GlobalF32, root_state: GlobalF32,
                           n_particles: Int, num_particles: Int, state_dim: Int):
    """Copia el estado del env a cada una de sus N particulas.
    Es el `broadcast_tree` de Stoix. Un hilo por (particula, componente del estado).
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    total = n_particles * state_dim
    if i >= total:
        return

    p = i // state_dim          # que particula
    d = i % state_dim           # que componente de su estado
    env = p // num_particles    # de que env viene

    particle_state[p * state_dim + d] = root_state[env * state_dim + d]


def broadcast_value_kernel(particle_value: GlobalF32, root_value: GlobalF32,
                           n_particles: Int, num_particles: Int):
    """V(s_raiz) replicado a las N particulas del env. Todas arrancan con el
    mismo valor porque todas estan en el mismo estado."""
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p < n_particles:
        particle_value[p] = root_value[p // num_particles]


def broadcast_logits_kernel(particle_logits: GlobalF32, root_logits: GlobalF32,
                            n_particles: Int, num_particles: Int, num_actions: Int):
    """Replica los logits del prior del env a sus N particulas.

    Se hace para poder reutilizar `categorical_from_logits` de la fase 2, que
    muestrea una accion por FILA. Replicando, cada particula tiene su fila y saca
    su propia accion; sin replicar habria que escribir un kernel que saque N
    muestras de la misma fila, o sea codigo nuevo que ya esta probado.

    A partir de la profundidad 1 esta replicacion ya no hace falta: cada particula
    esta en un estado distinto y tiene sus propios logits.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    total = n_particles * num_actions
    if i >= total:
        return

    p = i // num_actions
    a = i % num_actions
    env = p // num_particles

    particle_logits[p * num_actions + a] = root_logits[env * num_actions + a]


def log_prob_of_action_kernel(log_prob_out: GlobalF32, logits: GlobalF32,
                              actions: GlobalI32, n_particles: Int,
                              num_actions: Int):
    """log pi(a|s) de la accion que le toco a cada particula.

    Esta función calcula qué probabilidad daba la política a la acción que 
    eligió cada partícula. No elige la acción. La acción ya fue elegida por 
    categorical_from_logits.
    
    log_softmax(logits)[a] = logits[a] - logsumexp(logits), con el max restado
    para no desbordar (misma receta que ops/softmax.mojo, pero aqui el bucle es
    de un solo hilo sobre num_actions, que son 2 o 4; montar un bloque entero
    para eso seria desperdiciarlo).

    En Stoix esto es `pi.log_prob(sampled_actions)`.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p >= n_particles:
        return

    base = p * num_actions

    biggest = NEG_INF
    for a in range(num_actions):
        v = logits[base + a]
        if v > biggest:
            biggest = v

    total = Scalar[dtype](0)
    for a in range(num_actions):
        total += exp(logits[base + a] - biggest)

    chosen = Int(actions[p])
    log_prob_out[p] = logits[base + chosen] - (biggest + log(total))


def copy_actions_kernel(dst: GlobalI32, src: GlobalI32, n: Int):
    """Copia plana de acciones. Hace falta porque la accion vive en dos sitios:
    `root_actions` la guarda para siempre (es la que se acabara ejecutando en el
    entorno real) y `next_action` es el carry que se pisa en cada profundidad."""
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n:
        dst[i] = src[i]


def root_fn(ctx: DeviceContext, particles: Particles, outputs: StepOutputs,
            cfg: SPOConfig, root_state: DeviceBuffer[dtype],
            root_logits: DeviceBuffer[dtype], root_value: DeviceBuffer[dtype],
            uniforms: DeviceBuffer[dtype]) raises:
    """Siembra la busqueda: N particulas por env, una accion cada una.

    Equivale al `root_fn` de Stoix

    Entradas (las calcula el modelo sobre los estados RAIZ, uno por env):
        root_state  [num_envs, state_dim]
        root_logits [num_envs, num_actions]  puntuaciones de la política para cada acción
        root_value  [num_envs]               V(s_raiz)
        uniforms    [P]                      numeros aleatorios para sortear acciones

    Salida: `particles` queda listo para la profundidad 0. Los campos que se
    acumulan (peso, gae, terminal, depth) arrancan a cero, como en
    `init_particles` de Stoix.
    """
    check_search_config(cfg)
    p_total = cfg.num_search_particles()

    # 1. Copiamos el estado raiz de cada entorno a todas sus particulas.
    # Cada particula necesita su propia copia para poder simular un futuro distinto.
    state_elems = p_total * cfg.state_dim   # Total de componentes que copiamos.
    ctx.enqueue_function[broadcast_state_kernel, broadcast_state_kernel](
        particles.state.unsafe_ptr(), root_state.unsafe_ptr(),
        p_total, cfg.num_particles, cfg.state_dim,
        grid_dim=(state_elems + TPB - 1) // TPB, block_dim=TPB)

    # 2. Copiamos tambien el valor V(estado raiz). Las particulas de un mismo
    # entorno empiezan con el mismo valor porque todavia comparten estado.
    ctx.enqueue_function[broadcast_value_kernel, broadcast_value_kernel](
        particles.value.unsafe_ptr(), root_value.unsafe_ptr(),
        p_total, cfg.num_particles,
        grid_dim=(p_total + TPB - 1) // TPB, block_dim=TPB)

    # 3. Copiamos a cada particula las puntuaciones de todas las acciones que
    # da la politica actual. Son logits, no probabilidades ni politicas antiguas.
    logit_elems = p_total * cfg.num_actions
    ctx.enqueue_function[broadcast_logits_kernel, broadcast_logits_kernel](
        outputs.action_logits.unsafe_ptr(), root_logits.unsafe_ptr(),
        p_total, cfg.num_particles, cfg.num_actions,
        grid_dim=(logit_elems + TPB - 1) // TPB, block_dim=TPB)

    # 4. Sorteamos la primera accion de cada particula. Todas parten de la misma
    # distribucion, pero usan numeros aleatorios distintos y pueden elegir
    # acciones diferentes. El resultado se guarda en root_actions.
    ctx.enqueue_function[categorical_from_logits[TPB], categorical_from_logits[TPB]](
        particles.root_actions.unsafe_ptr(), outputs.action_logits.unsafe_ptr(),
        uniforms.unsafe_ptr(), cfg.num_actions,
        grid_dim=p_total, block_dim=TPB)

    # 5. Para cada particula guardamos log(probabilidad) de la accion que acaba
    # de elegir segun la politica original. Esto no vuelve a elegir la accion.
    ctx.enqueue_function[log_prob_of_action_kernel, log_prob_of_action_kernel](
        particles.prior_logits.unsafe_ptr(), outputs.action_logits.unsafe_ptr(),
        particles.root_actions.unsafe_ptr(), p_total, cfg.num_actions,
        grid_dim=(p_total + TPB - 1) // TPB, block_dim=TPB)

    # 6. Preparamos el primer paso: la accion raiz es la que se ejecuta en la
    # profundidad 0. root_actions se conserva hasta el final; next_action ira
    # cambiando para indicar la accion que toca ejecutar en cada profundidad.
    ctx.enqueue_function[copy_actions_kernel, copy_actions_kernel](
        outputs.next_action.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        p_total, grid_dim=(p_total + TPB - 1) // TPB, block_dim=TPB)


def sample_next_actions(ctx: DeviceContext, outputs: StepOutputs, cfg: SPOConfig,
                        uniforms: DeviceBuffer[dtype]) raises:
    """La mitad generica del recurrent_fn: elegir la accion de la profundidad siguiente.

    En Stoix el `recurrent_fn` hace cuatro cosas: avanzar el entorno, evaluar
    actor y critico en el estado nuevo, muestrear la siguiente accion, y plegar
    gamma/truncacion. Las que dependen del modelo (avanzar, evaluar, plegar) van
    en el kernel del modelo; estas dos, que son iguales para todos, viven aqui.

    Entra `outputs.action_logits` [P, num_actions] (ya escrito por el
    policy_logits_kernel del modelo sobre el estado NUEVO) y salen
    `next_action` y `next_prior_logits`.

    `uniforms` tiene que traer P valores frescos: uno por particula. Reusar los
    de la profundidad anterior correlacionaria las trayectorias.
    """
    p_total = cfg.num_search_particles()

    ctx.enqueue_function[categorical_from_logits[TPB], categorical_from_logits[TPB]](
        outputs.next_action.unsafe_ptr(), outputs.action_logits.unsafe_ptr(),
        uniforms.unsafe_ptr(), cfg.num_actions,
        grid_dim=p_total, block_dim=TPB)

    ctx.enqueue_function[log_prob_of_action_kernel, log_prob_of_action_kernel](
        outputs.next_prior_logits.unsafe_ptr(), outputs.action_logits.unsafe_ptr(),
        outputs.next_action.unsafe_ptr(), p_total, cfg.num_actions,
        grid_dim=(p_total + TPB - 1) // TPB, block_dim=TPB)


def update_particles_kernel(
        # estado de la particula, se actualiza in-place
        weights: GlobalF32, gae: GlobalF32, value: GlobalF32,
        terminal: GlobalI32, depth: GlobalI32, prior_logits: GlobalF32,
        # lo que produjo el paso del modelo
        reward: GlobalF32, discount: GlobalF32, next_value: GlobalF32,
        next_prior_logits: GlobalF32,
        # config
        n_particles: Int, search_gamma: Scalar[dtype],
        search_gae_lambda: Scalar[dtype]):
    """Cierra una profundidad: peso SMC, GAE, y el relevo de estado.

    Funde tres cosas que en Stoix son funciones separadas (`smc_weight_update_fn`,
    `calculate_gae` y `update_particles`) porque con un hilo por particula las
    tres son la misma pasada: se leen los valores viejos y se escriben los nuevos.

    El orden dentro del hilo es lo unico delicado: el error TD y la GAE necesitan
    el V(s) y la profundidad VIEJOS, asi que se calculan antes de pisarlos.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p >= n_particles:
        return

    old_value = value[p]
    old_depth = Int(depth[p])
    was_terminal = Int(terminal[p]) != 0
    step_discount = discount[p]

    # El error TD no se multiplica por gamma porque ya viene plegado dentro de
    # next_value, que es el bootstrap_value que devolvio el modelo. Stoix lo
    # comenta igual: "We do not multiply by discount as we do it in the
    # recurrent_fn".
    td_error = reward[p] + next_value[p] - old_value

    # La mascara terminal congela el peso de las particulas ya muertas, porque
    # lo que les pase despues de morir no deberia cambiar su evidencia.
    mask = Scalar[dtype](0) if was_terminal else Scalar[dtype](1)
    weights[p] = weights[p] + td_error * mask

    # La GAE va al reves de lo habitual. Normalmente se calcula hacia atras en
    # el tiempo, pero aqui la busqueda avanza y no se puede mirar al futuro, asi
    # que cada profundidad anade su delta descontado por
    # (gamma*lambda*discount)^profundidad.
    #
    # Fijate en que no hay mascara terminal, igual que en Stoix. Lo que congela a
    # una particula muerta es que su discount vale 0 y entonces el factor se
    # anula solo. La excepcion es la profundidad 0, donde el exponente es 0 y el
    # factor vale 1 pase lo que pase, y eso es correcto: el primer paso siempre
    # cuenta entero aunque la particula muera en el.
    decay_base = search_gamma * search_gae_lambda * step_discount
    decay = Scalar[dtype](1)
    for _ in range(old_depth):
        decay *= decay_base
    gae[p] = gae[p] + td_error * decay

    # Y el relevo: lo nuevo pasa a ser lo actual.
    value[p] = next_value[p]
    prior_logits[p] = next_prior_logits[p]
    depth[p] = Scalar[idx_dtype](old_depth + 1)
    # `terminal` es pegajoso: una vez muerta, muerta. Un discount de 0 la mata,
    # tanto si fue muerte real como truncacion.
    if was_terminal or step_discount == 0.0:
        terminal[p] = Scalar[idx_dtype](1)


def update_particles(ctx: DeviceContext, particles: Particles,
                     outputs: StepOutputs, cfg: SPOConfig) raises:
    """Encola el cierre de una profundidad."""
    p_total = cfg.num_search_particles()
    ctx.enqueue_function[update_particles_kernel, update_particles_kernel](
        particles.resample_td_weights.unsafe_ptr(), particles.gae.unsafe_ptr(),
        particles.value.unsafe_ptr(), particles.terminal.unsafe_ptr(),
        particles.depth.unsafe_ptr(), particles.prior_logits.unsafe_ptr(),
        outputs.reward.unsafe_ptr(), outputs.discount.unsafe_ptr(),
        outputs.next_value.unsafe_ptr(), outputs.next_prior_logits.unsafe_ptr(),
        p_total, cfg.search_gamma, cfg.search_gae_lambda,
        grid_dim=(p_total + TPB - 1) // TPB, block_dim=TPB)


# El resampling es lo que convierte peso en multiplicidad: las particulas
# prometedoras se copian varias veces y las malas desaparecen. Despues todas
# vuelven a tener el mismo peso, y la informacion de cual era mejor ya no vive en
# los pesos sino en cuantas copias hay de cada una.

def resample_logits_kernel(logits_out: GlobalF32, weights: GlobalF32,
                           n_particles: Int, temperature: Scalar[dtype]):
    """logits = pesos / temperatura, el `get_resample_logits` de Stoix.

    La temperatura decide lo agresiva que es la busqueda. Con una temperatura
    baja solo sobreviven las mejores particulas y el ESS se hunde; con una alta
    se reparte casi por igual y la busqueda apenas mejora al prior. Es el mismo
    eta que el M-step acabara aprendiendo.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p < n_particles:
        logits_out[p] = weights[p] / temperature


def resample_indices_kernel[TPB_P: Int](
        indices_out: GlobalI32, logits: GlobalF32, uniforms: GlobalF32,
        num_particles: Int):
    """Sortea N indices por env, con probabilidad softmax(logits) (CDF inversa).

    Un bloque por env y un hilo por particula. El CDF se construye una sola vez
    en shared memory y luego cada uno de los N hilos hace su busqueda con su
    propio uniforme, o sea N muestras del mismo CDF. Esa es la diferencia con
    `categorical_from_logits`, que saca una sola muestra por fila.

    Como en el muestreo de la fase 2, el CDF se deja sin normalizar y lo que se
    escala es el uniforme.
    """
    debug_assert(num_particles <= TPB_P,
                 "resample: las particulas de un env tienen que caber en el bloque")

    shared = stack_allocation[TPB_P, Scalar[dtype],
                              address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    env = Int(block_idx.x)
    base = env * num_particles

    # softmax estable -> CDF
    shared[tid] = logits[base + tid] if tid < num_particles else NEG_INF
    barrier()
    row_max = block_reduce_max[TPB_P](shared, tid)

    shared[tid] = exp(logits[base + tid] - row_max) if tid < num_particles else Scalar[dtype](0)
    barrier()
    block_scan_inclusive[TPB_P](shared, tid)

    if tid >= num_particles:
        return

    total = shared[num_particles - 1]
    target = uniforms[base + tid] * total

    # Barrido lineal. Con 16 particulas no compensa una busqueda binaria y asi
    # el codigo dice exactamente lo que hace.
    chosen = num_particles - 1     # por defecto el ultimo, por si el redondeo
                                   # deja el target justo en el borde
    for n in range(num_particles):
        cdf_prev = shared[n - 1] if n > 0 else Scalar[dtype](0)
        if cdf_prev <= target and target < shared[n]:
            chosen = n
            break

    indices_out[base + tid] = Scalar[idx_dtype](chosen)


def gather_particles_kernel(
        # destino (los buffers de scratch)
        dst_state: GlobalF32, dst_root_actions: GlobalI32,
        dst_prior_logits: GlobalF32, dst_value: GlobalF32,
        dst_terminal: GlobalI32, dst_depth: GlobalI32,
        # origen (las particulas de verdad)
        src_state: GlobalF32, src_root_actions: GlobalI32,
        src_prior_logits: GlobalF32, src_value: GlobalF32,
        src_terminal: GlobalI32, src_depth: GlobalI32,
        indices: GlobalI32, n_particles: Int, num_particles: Int,
        state_dim: Int):
    """dst[i] = src[indices[i]], con los indices dentro del mismo env.

    Va a scratch y no in-place: si escribiera encima, un hilo podria pisar un
    hueco que otro todavia no ha leido. Es la carrera clasica del gather.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p >= n_particles:
        return

    env = p // num_particles
    src = env * num_particles + Int(indices[p])

    for d in range(state_dim):
        dst_state[p * state_dim + d] = src_state[src * state_dim + d]
    dst_root_actions[p] = src_root_actions[src]
    dst_prior_logits[p] = src_prior_logits[src]
    dst_value[p] = src_value[src]
    dst_terminal[p] = src_terminal[src]
    dst_depth[p] = src_depth[src]


def copy_back_kernel(
        dst_state: GlobalF32, dst_root_actions: GlobalI32,
        dst_prior_logits: GlobalF32, dst_value: GlobalF32,
        dst_terminal: GlobalI32, dst_depth: GlobalI32, dst_weights: GlobalF32,
        src_state: GlobalF32, src_root_actions: GlobalI32,
        src_prior_logits: GlobalF32, src_value: GlobalF32,
        src_terminal: GlobalI32, src_depth: GlobalI32,
        n_particles: Int, state_dim: Int):
    """Devuelve el scratch a las particulas y resetea el peso a cero.

    El reset es parte del resampling: tras repartir las copias, todas las
    particulas vuelven a estar igual de bien consideradas.
    """
    p = Int(block_dim.x * block_idx.x + thread_idx.x)
    if p >= n_particles:
        return

    for d in range(state_dim):
        dst_state[p * state_dim + d] = src_state[p * state_dim + d]
    dst_root_actions[p] = src_root_actions[p]
    dst_prior_logits[p] = src_prior_logits[p]
    dst_value[p] = src_value[p]
    dst_terminal[p] = src_terminal[p]
    dst_depth[p] = src_depth[p]
    dst_weights[p] = Scalar[dtype](0)


def resample(ctx: DeviceContext, particles: Particles, scratch: SearchScratch,
             cfg: SPOConfig, uniforms: DeviceBuffer[dtype]) raises:
    """Resampling completo. Espeja `resample` de Stoix (ff_spo.py:846).

    La gae no se toca, y conviene fijarse en eso. Stoix hace el gather de todo
    y luego reescribe la gae con la de antes del resampling (ff_spo.py:914), asi
    que en la practica se queda tal cual estaba. El motivo, segun su comentario,
    es que el loss de la temperatura necesita las ventajas de antes de resamplear
    para apuntar bien al KL; el precio es que la gae solo cubre hasta el ultimo
    resampling.

    Aqui sale mas simple: basta con no copiarla, que es lo mismo y ahorra un
    buffer.
    """
    p_total = cfg.num_search_particles()
    blocks = (p_total + TPB - 1) // TPB

    ctx.enqueue_function[resample_logits_kernel, resample_logits_kernel](
        scratch.resample_logits.unsafe_ptr(),
        particles.resample_td_weights.unsafe_ptr(),
        p_total, cfg.temperature, grid_dim=blocks, block_dim=TPB)

    ctx.enqueue_function[resample_indices_kernel[TPB_PARTICLES],
                         resample_indices_kernel[TPB_PARTICLES]](
        scratch.indices.unsafe_ptr(), scratch.resample_logits.unsafe_ptr(),
        uniforms.unsafe_ptr(), cfg.num_particles,
        grid_dim=cfg.num_envs, block_dim=TPB_PARTICLES)

    ctx.enqueue_function[gather_particles_kernel, gather_particles_kernel](
        scratch.state.unsafe_ptr(), scratch.root_actions.unsafe_ptr(),
        scratch.prior_logits.unsafe_ptr(), scratch.value.unsafe_ptr(),
        scratch.terminal.unsafe_ptr(), scratch.depth.unsafe_ptr(),
        particles.state.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        particles.prior_logits.unsafe_ptr(), particles.value.unsafe_ptr(),
        particles.terminal.unsafe_ptr(), particles.depth.unsafe_ptr(),
        scratch.indices.unsafe_ptr(), p_total, cfg.num_particles, cfg.state_dim,
        grid_dim=blocks, block_dim=TPB)

    ctx.enqueue_function[copy_back_kernel, copy_back_kernel](
        particles.state.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        particles.prior_logits.unsafe_ptr(), particles.value.unsafe_ptr(),
        particles.terminal.unsafe_ptr(), particles.depth.unsafe_ptr(),
        particles.resample_td_weights.unsafe_ptr(),
        scratch.state.unsafe_ptr(), scratch.root_actions.unsafe_ptr(),
        scratch.prior_logits.unsafe_ptr(), scratch.value.unsafe_ptr(),
        scratch.terminal.unsafe_ptr(), scratch.depth.unsafe_ptr(),
        p_total, cfg.state_dim, grid_dim=blocks, block_dim=TPB)


def ess_entropy_kernel[TPB_P: Int](ess_out: GlobalF32, entropy_out: GlobalF32,
                                   logits: GlobalF32, num_particles: Int):
    """ESS y entropia de los pesos normalizados, por env.

    El ESS (effective sample size) es 1 / sum(w_i^2) con los pesos normalizados,
    y dice cuantas particulas estan aportando de verdad. Con pesos uniformes vale
    N, porque todas cuentan igual; cuando un peso se lo lleva todo baja a 1 y la
    busqueda ha colapsado. Es la senal de cuando toca resamplear, y en la demo se
    ve caer entre resamplings y recuperarse justo despues.

    Un bloque por env, un hilo por particula.
    """
    shared = stack_allocation[TPB_P, Scalar[dtype],
                              address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    env = Int(block_idx.x)
    base = env * num_particles

    active = tid < num_particles

    # softmax estable de los logits
    shared[tid] = logits[base + tid] if active else NEG_INF
    barrier()
    row_max = block_reduce_max[TPB_P](shared, tid)

    e = exp(logits[base + tid] - row_max) if active else Scalar[dtype](0)
    shared[tid] = e
    barrier()
    total = block_reduce_sum[TPB_P](shared, tid)

    w = e / total if active else Scalar[dtype](0)

    # sum(w^2) -> ESS
    shared[tid] = w * w
    barrier()
    sum_sq = block_reduce_sum[TPB_P](shared, tid)

    # -sum(w log w) -> entropia. El TINY evita log(0) en las particulas que se
    # quedaron sin masa; Stoix usa el mismo truco con finfo.tiny.
    TINY = Scalar[dtype](1e-30)
    shared[tid] = -w * log(w + TINY) if active else Scalar[dtype](0)
    barrier()
    ent = block_reduce_sum[TPB_P](shared, tid)

    if tid == 0:
        ess_out[env] = Scalar[dtype](1) / sum_sq
        entropy_out[env] = ent


def select_action_kernel(action_out: GlobalI32, root_actions: GlobalI32,
                         chosen: GlobalI32, num_envs: Int, num_particles: Int):
    """La accion final del env es la accion RAIZ de la particula sorteada."""
    env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env < num_envs:
        action_out[env] = root_actions[env * num_particles + Int(chosen[env])]


def copy_f32_kernel(dst: GlobalF32, src: GlobalF32, n: Int):
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n:
        dst[i] = src[i]


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


def compute_ess_entropy(ctx: DeviceContext, particles: Particles,
                        scratch: SearchScratch, output: SPOOutput,
                        cfg: SPOConfig, depth: Int) raises:
    """Mide ESS y entropia con los pesos de ahora y los guarda en la fila `depth`."""
    p_total = cfg.num_search_particles()
    ctx.enqueue_function[resample_logits_kernel, resample_logits_kernel](
        scratch.resample_logits.unsafe_ptr(),
        particles.resample_td_weights.unsafe_ptr(),
        p_total, cfg.temperature,
        grid_dim=(p_total + TPB - 1) // TPB, block_dim=TPB)

    row = depth * cfg.num_envs
    ctx.enqueue_function[ess_entropy_kernel[TPB_PARTICLES],
                         ess_entropy_kernel[TPB_PARTICLES]](
        output.ess.unsafe_ptr() + row, output.entropy.unsafe_ptr() + row,
        scratch.resample_logits.unsafe_ptr(), cfg.num_particles,
        grid_dim=cfg.num_envs, block_dim=TPB_PARTICLES)


def readout_weighted(ctx: DeviceContext, particles: Particles,
                     scratch: SearchScratch, output: SPOOutput, cfg: SPOConfig,
                     uniforms: DeviceBuffer[dtype]) raises:
    """Lee el resultado de la busqueda. Espeja `readout_weighted` de Stoix (ff_spo.py:465).

    La politica mejorada q queda representada por dos cosas juntas:
      - `sampled_actions`: las N acciones raiz que sobrevivieron (con repeticiones
        si el resampling copio alguna varias veces),
      - `sampled_action_weights`: softmax(peso/temperatura) de cada una.
    Su histograma ponderado es q, y la accion que se ejecuta se sortea de ahi.

    `uniforms` solo necesita num_envs valores (uno por env), pero se pasa un
    buffer de P por comodidad: se leen los primeros num_envs.
    """
    p_total = cfg.num_search_particles()
    blocks_p = (p_total + TPB - 1) // TPB

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
        grid_dim=(cfg.num_envs + TPB - 1) // TPB, block_dim=TPB)

    ctx.enqueue_function[copy_actions_kernel, copy_actions_kernel](
        output.sampled_actions.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        p_total, grid_dim=blocks_p, block_dim=TPB)

    ctx.enqueue_function[copy_f32_kernel, copy_f32_kernel](
        output.sampled_advantages.unsafe_ptr(), particles.gae.unsafe_ptr(),
        p_total, grid_dim=blocks_p, block_dim=TPB)

    # El valor de la raiz, no el del final del rollout: es lo que Stoix mete en
    # SPOOutput.value (jnp.mean(root.particle_values, axis=-1)).
    ctx.enqueue_function[mean_over_particles_kernel, mean_over_particles_kernel](
        output.value.unsafe_ptr(), output.root_values.unsafe_ptr(),
        cfg.num_envs, cfg.num_particles,
        grid_dim=(cfg.num_envs + TPB - 1) // TPB, block_dim=TPB)


def snapshot_root_values(ctx: DeviceContext, particles: Particles,
                         output: SPOOutput, cfg: SPOConfig) raises:
    """Guarda V(s_raiz) justo despues de sembrar, antes de que el rollout lo pise."""
    p_total = cfg.num_search_particles()
    ctx.enqueue_function[copy_f32_kernel, copy_f32_kernel](
        output.root_values.unsafe_ptr(), particles.value.unsafe_ptr(), p_total,
        grid_dim=(p_total + TPB - 1) // TPB, block_dim=TPB)


def smc_depth_close(ctx: DeviceContext, particles: Particles,
                    outputs: StepOutputs, scratch: SearchScratch,
                    output: SPOOutput, cfg: SPOConfig, depth: Int,
                    resample_uniforms: DeviceBuffer[dtype]) raises:
    """Todo lo que va despues del paso del modelo en una profundidad.

    Es la parte generica de `one_step_rollout` de Stoix (ff_spo.py:568): sirve
    igual para el juguete, CartPole o el MLP. Un modelo nuevo solo tiene que
    llamar a su recurrent_fn y luego a esto.

    El orden importa: el ESS se mide con los pesos ya actualizados pero antes de
    resamplear, que es donde se ve el colapso que justifica el resampling.
    """
    update_particles(ctx, particles, outputs, cfg)
    compute_ess_entropy(ctx, particles, scratch, output, cfg, depth)

    # Modo 'period' de Stoix: resamplea cuando (depth+1) es multiplo del periodo.
    if (depth + 1) % cfg.resample_period == 0:
        resample(ctx, particles, scratch, cfg, resample_uniforms)
