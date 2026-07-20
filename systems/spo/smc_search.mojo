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

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import exp, log

from ops.common import dtype, idx_dtype, NEG_INF, GlobalF32, GlobalI32
from ops.rng import categorical_from_logits
from systems.spo.spo_types import SPOConfig, Particles, StepOutputs

comptime TPB = 32


def broadcast_state_kernel(particle_state: GlobalF32, root_state: GlobalF32,
                           n_particles: Int, num_particles: Int, state_dim: Int):
    """Copia el estado del env a cada una de sus N particulas.

    Es el `broadcast_tree` de Stoix (ff_spo.py:215). Alli es una linea porque JAX
    hace el broadcast solo; aqui hay que copiar de verdad, y esa copia ES la
    esencia de la busqueda: cada particula necesita SU propio mundo para poder
    diverger del de las demas.

    Un hilo por (particula, componente del estado).
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

    log_softmax(logits)[a] = logits[a] - logsumexp(logits), con el max restado
    para no desbordar (misma receta que ops/softmax.mojo, pero aqui el bucle es
    de un solo hilo sobre num_actions, que son 2 o 4; montar un bloque entero
    para eso seria desperdiciarlo).

    En Stoix esto es `pi.log_prob(sampled_actions)`. No hace falta para la
    busqueda en si (el comentario de Stoix dice "not strictly necessary"), pero
    se guarda porque es el prior contra el que el M-step mide el KL.
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

    Equivale al `root_fn` de Stoix (ff_spo.py:163) menos el ruido Dirichlet, que
    en ff_spo.yaml viene con fraccion 0.0, o sea desactivado.

    Entradas (las calcula el modelo sobre los estados RAIZ, uno por env):
        root_state  [num_envs, state_dim]
        root_logits [num_envs, num_actions]  el prior en la raiz
        root_value  [num_envs]               V(s_raiz)
        uniforms    [P]                      uno por particula, para muestrear

    Salida: `particles` queda listo para la profundidad 0. Los campos que se
    acumulan (peso, gae, terminal, depth) arrancan a cero, como en
    `init_particles` de Stoix.
    """
    p_total = cfg.num_search_particles()

    # 1. Cada particula se lleva una copia del mundo y del valor de la raiz.
    state_elems = p_total * cfg.state_dim
    ctx.enqueue_function[broadcast_state_kernel, broadcast_state_kernel](
        particles.state.unsafe_ptr(), root_state.unsafe_ptr(),
        p_total, cfg.num_particles, cfg.state_dim,
        grid_dim=(state_elems + TPB - 1) // TPB, block_dim=TPB)

    ctx.enqueue_function[broadcast_value_kernel, broadcast_value_kernel](
        particles.value.unsafe_ptr(), root_value.unsafe_ptr(),
        p_total, cfg.num_particles,
        grid_dim=(p_total + TPB - 1) // TPB, block_dim=TPB)

    # 2. Los logits del prior, replicados para poder muestrear por filas.
    logit_elems = p_total * cfg.num_actions
    ctx.enqueue_function[broadcast_logits_kernel, broadcast_logits_kernel](
        outputs.action_logits.unsafe_ptr(), root_logits.unsafe_ptr(),
        p_total, cfg.num_particles, cfg.num_actions,
        grid_dim=(logit_elems + TPB - 1) // TPB, block_dim=TPB)

    # 3. Una accion por particula. Aqui es donde N particulas del mismo env se
    #    separan: mismo estado, misma distribucion, distinto uniforme.
    ctx.enqueue_function[categorical_from_logits[TPB], categorical_from_logits[TPB]](
        particles.root_actions.unsafe_ptr(), outputs.action_logits.unsafe_ptr(),
        uniforms.unsafe_ptr(), cfg.num_actions,
        grid_dim=p_total, block_dim=TPB)

    # 4. Y su log-prob bajo el prior.
    ctx.enqueue_function[log_prob_of_action_kernel, log_prob_of_action_kernel](
        particles.prior_logits.unsafe_ptr(), outputs.action_logits.unsafe_ptr(),
        particles.root_actions.unsafe_ptr(), p_total, cfg.num_actions,
        grid_dim=(p_total + TPB - 1) // TPB, block_dim=TPB)

    # 5. La accion de la raiz es tambien la que se ejecuta en la profundidad 0.
    #    Stoix arranca su scan con carry = (particles, particles.root_actions);
    #    aqui el carry es outputs.next_action, asi que hay que sembrarlo.
    ctx.enqueue_function[copy_actions_kernel, copy_actions_kernel](
        outputs.next_action.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        p_total, grid_dim=(p_total + TPB - 1) // TPB, block_dim=TPB)

    # Nada de ctx.synchronize() aqui: los cuatro kernels van al mismo stream y se
    # ejecutan en orden (leccion del Puzzle 14). Solo sincroniza quien lea.


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
