"""Fase 1 de la busqueda: sembrar. El `root_fn` de Stoix.

Partiendo de un estado por entorno, deja N particulas por entorno listas para la
profundidad 0: todas en el mismo estado y con el mismo valor, pero cada una con
SU accion sorteada del prior. Ese sorteo es lo unico que las separa al principio;
todo lo demas (que diverjan, que unas resulten mejores) viene despues.

Aqui vive tambien `sample_next_actions`, que no es de la raiz pero es la otra
mitad generica del paso: el modelo deja los logits del estado nuevo y esto
sortea la accion de la profundidad siguiente. Estan juntas porque son el mismo
gesto -- muestrear una accion y anotar su log-prob-- en dos momentos distintos.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import exp, log

from ops.common import dtype, idx_dtype, NEG_INF, GlobalF32, GlobalI32
from ops.copy import copy_kernel
from ops.rng import categorical_from_logits
from systems.spo.launch import TPB, blocks_for, check_search_config
from systems.spo.particles import Particles, StepOutputs, SPOOutput
from systems.spo.spo_types import SPOConfig


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


def add_dirichlet_noise_kernel(logits: GlobalF32, u: GlobalF32, n_envs: Int,
                               n_actions: Int, fraction: Scalar[dtype]):
    """Ruido de exploracion de Dirichlet en la raiz. Un hilo por env.

    Espeja `apply_exploration_noise` de Stoix (`ff_spo.py:119`), que llama a
    `rlax.add_dirichlet_noise` y hace

        noisy = (1 - fraction) * prior + fraction * noise,   noise ~ Dir(alpha)

    **Un detalle que hay que decir en voz alta:** Stoix le pasa `pi.logits`, o sea
    que mezcla un vector de LOGITS con un vector de PROBABILIDADES (la Dirichlet
    suma 1). El docstring de rlax dice "prior policy vector", asi que la semantica
    no cuadra del todo. Con `fraction = 0` (su valor por defecto) da igual porque
    el termino desaparece; con fraction > 0 el ruido aporta como mucho `fraction`
    a un logit. Se reproduce igual porque es lo que hace la referencia, y se anota
    la rareza en vez de "arreglarla" por nuestra cuenta.

    Con alpha = 1 la Dirichlet simetrica es la uniforme sobre el simplex, y se
    muestrea normalizando exponenciales: e_i = -ln(u_i), noise = e / SUM(e). Es
    exacto, sin muestreador Gamma. alpha = 1 es el valor por defecto de Stoix.

    Las casillas tapadas (NEG_INF) siguen tapadas: (1-f)*NEG_INF domina cualquier
    ruido acotado en [0,1] mientras f < 1.
    """
    e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= n_envs:
        return
    base = e * n_actions
    total = Scalar[dtype](0)
    for a in range(n_actions):
        # -ln(u) con u en (0,1]. Se acota por abajo para no pedir log(0).
        uu = u[base + a]
        if uu < Scalar[dtype](1e-7):
            uu = Scalar[dtype](1e-7)
        total += -log(uu)
    if total <= Scalar[dtype](0):
        return
    for a in range(n_actions):
        uu = u[base + a]
        if uu < Scalar[dtype](1e-7):
            uu = Scalar[dtype](1e-7)
        noise = (-log(uu)) / total
        logits[base + a] = (Scalar[dtype](1) - fraction) * logits[base + a] \
                           + fraction * noise


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
    acumulan (peso, gae, terminal, depth) se ponen a cero AQUI, como en
    `init_particles` de Stoix.

    Ponerlos a cero explicitamente no es adorno: el `SearchWorkspace` se reserva
    una vez y se reutiliza en cada busqueda, asi que sin este reset la segunda
    busqueda heredaria el estado de la primera. Y el campo que mas duele es
    `terminal`: si llega con 1, la mascara de `update_particles` congela el peso
    desde la profundidad 0 y NINGUNA particula acumula nada. Los pesos se quedan
    todos a cero, el softmax del readout sale uniforme y la busqueda degenera en
    elegir al azar entre las acciones raiz -- silenciosamente, sin fallar.
    """
    check_search_config(cfg)
    p_total = cfg.num_search_particles()

    # 0. Los acumuladores a cero. Ver el porque en el docstring.
    particles.resample_td_weights.enqueue_fill(0)
    particles.gae.enqueue_fill(0)
    particles.terminal.enqueue_fill(0)
    particles.depth.enqueue_fill(0)

    # 1. Copiamos el estado raiz de cada entorno a todas sus particulas.
    # Cada particula necesita su propia copia para poder simular un futuro distinto.
    state_elems = p_total * cfg.state_dim   # Total de componentes que copiamos.
    ctx.enqueue_function[broadcast_state_kernel, broadcast_state_kernel](
        particles.state.unsafe_ptr(), root_state.unsafe_ptr(),
        p_total, cfg.num_particles, cfg.state_dim,
        grid_dim=blocks_for(state_elems), block_dim=TPB)

    # 2. Copiamos tambien el valor V(estado raiz). Las particulas de un mismo
    # entorno empiezan con el mismo valor porque todavia comparten estado.
    ctx.enqueue_function[broadcast_value_kernel, broadcast_value_kernel](
        particles.value.unsafe_ptr(), root_value.unsafe_ptr(),
        p_total, cfg.num_particles,
        grid_dim=blocks_for(p_total), block_dim=TPB)

    # 3. Copiamos a cada particula las puntuaciones de todas las acciones que
    # da la politica actual. Son logits, no probabilidades ni politicas antiguas.
    logit_elems = p_total * cfg.num_actions
    ctx.enqueue_function[broadcast_logits_kernel, broadcast_logits_kernel](
        outputs.action_logits.unsafe_ptr(), root_logits.unsafe_ptr(),
        p_total, cfg.num_particles, cfg.num_actions,
        grid_dim=blocks_for(logit_elems), block_dim=TPB)

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
        grid_dim=blocks_for(p_total), block_dim=TPB)

    # 6. Preparamos el primer paso: la accion raiz es la que se ejecuta en la
    # profundidad 0. root_actions se conserva hasta el final; next_action ira
    # cambiando para indicar la accion que toca ejecutar en cada profundidad.
    ctx.enqueue_function[copy_kernel[idx_dtype], copy_kernel[idx_dtype]](
        outputs.next_action.unsafe_ptr(), particles.root_actions.unsafe_ptr(),
        p_total, grid_dim=blocks_for(p_total), block_dim=TPB)


def sample_next_actions(ctx: DeviceContext, outputs: StepOutputs, cfg: SPOConfig,
                        uniforms: DeviceBuffer[dtype]) raises:
    """La mitad generica del recurrent_fn: elegir la accion de la profundidad siguiente.

    En Stoix el `recurrent_fn` hace cuatro cosas: avanzar el entorno, evaluar
    actor y critico en el estado nuevo, muestrear la siguiente accion, y plegar
    gamma/truncacion. Las que dependen del modelo (avanzar, evaluar, plegar) van
    en el kernel del modelo; estas dos, que son iguales para todos, viven aqui.

    Entra `outputs.action_logits` [P, num_actions] (ya escrito por el modelo sobre
    el estado NUEVO) y salen `next_action` y `next_prior_logits`.

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
        grid_dim=blocks_for(p_total), block_dim=TPB)


def snapshot_root_values(ctx: DeviceContext, particles: Particles,
                         output: SPOOutput, cfg: SPOConfig) raises:
    """Guarda V(s_raiz) justo despues de sembrar, antes de que el rollout lo pise.

    Hace falta porque `particles.value` va avanzando con las particulas, pero la
    salida publica reporta el valor de la RAIZ."""
    p_total = cfg.num_search_particles()
    ctx.enqueue_function[copy_kernel[dtype], copy_kernel[dtype]](
        output.root_values.unsafe_ptr(), particles.value.unsafe_ptr(), p_total,
        grid_dim=blocks_for(p_total), block_dim=TPB)
