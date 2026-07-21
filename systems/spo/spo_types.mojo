"""Tipos de la busqueda SPO. Espeja stoix/systems/spo/spo_types.py.

Mantengo los nombres de Stoix (`Particles`, `resample_td_weights`, `root_actions`...)
para poder poner los dos ficheros uno al lado del otro y comparar.

Sobre por que la interfaz del modelo no es un trait: en Stoix la clase SPO recibe
un `recurrent_fn` abstracto y no sabe si detras hay una red neuronal, un entorno o
un MDP de juguete. Queria lo mismo aqui, pero en Mojo 1.0.0b1 no se puede. Probe
pasar el modelo como parametro comptime de tipo funcion y el compilador no acepta
el tipo, ni con `capturing` ni con `escaping` (esta ultima ya ni existe). Pasar un
kernel entero como parametro tampoco vale, porque `enqueue_function` no consigue
inferir su `signature_func`. Los intentos estan en docs/api_notes.md.

Asi que un modelo son tres kernels con firma fija, y el resto del nucleo SMC
(pesos, GAE, resampling, ESS, readout) es codigo compartido que no se duplica.

El contrato de los tres kernels, con P = num_envs * num_particles:

  1. policy_logits_kernel(logits_out, state, n_particles)
        logits_out: [P, num_actions]  logits del prior en el estado de cada particula
        state:      [P, state_dim]
     Un hilo por particula.

  2. value_kernel(value_out, state, n_particles)
        value_out:  [P]   V(s) de cada particula
     Un hilo por particula.

  3. recurrent_kernel(state, action, reward_out, discount_out, next_value_out,
                      n_particles, ...config del modelo...)
        Avanza cada particula un paso y devuelve lo que la busqueda necesita:
          reward_out:     [P]  recompensa del paso
          discount_out:   [P]  el `rec_discount` de Stoix, ya con la truncacion
                               plegada: discount * (1 - truncated)
          next_value_out: [P]  el `bootstrap_value` de Stoix:
                               discount_real * search_gamma * V(s')
        `state` se actualiza in-place.
     Un hilo por particula.

El plegado de gamma y de la truncacion vive dentro del kernel del modelo, igual
que en el `recurrent_fn` de Stoix, para que el nucleo SMC no tenga que saber nada
del entorno.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from ops.common import dtype, idx_dtype


@fieldwise_init
struct SPOConfig(Copyable, Movable):
    """Los hiperparametros de la busqueda. Valores por defecto = ff_spo.yaml."""

    var num_envs: Int
    var num_particles: Int
    var num_actions: Int
    var state_dim: Int
    var search_depth: Int
    var resample_period: Int
    """Cada cuantos pasos se hace resampling (modo 'period' de Stoix)."""
    var temperature: Scalar[dtype]
    """Temperatura fija. En la fase 8 pasa a ser el softplus del dual aprendido."""
    var search_gamma: Scalar[dtype]
    var search_gae_lambda: Scalar[dtype]

    def num_search_particles(self) -> Int:
        """P = envs * particulas. Es el tamano de casi todos los buffers."""
        return self.num_envs * self.num_particles


def default_config(num_envs: Int, state_dim: Int, num_actions: Int) -> SPOConfig:
    """Los valores del paper / ff_spo.yaml para la busqueda.

    La temperatura es fija (0.5, el `fixed_temperature` de Stoix) hasta que en la
    fase 8 existan los duales y se pueda aprender.
    """
    return SPOConfig(
        num_envs=num_envs,
        num_particles=16,
        num_actions=num_actions,
        state_dim=state_dim,
        search_depth=4,
        resample_period=4,
        temperature=0.5,
        search_gamma=1.0,
        search_gae_lambda=1.0,
    )


struct Particles(Movable):
    """Las P = envs*particulas trayectorias hipoteticas, como struct-of-arrays.

    Stoix usa un NamedTuple de arrays [NumEnvs, NumParticles, ...] y deja que JAX
    haga el tree_map. Aqui cada campo es un DeviceBuffer plano de P elementos y el
    indice es `env * num_particles + particula`; con eso la mayoria de kernels son
    un simple map de un hilo por particula.

    Los nombres son los de Stoix a proposito (ver Particles en ff_spo.py:354).
    """

    var state: DeviceBuffer[dtype]
    """[P, state_dim] estado completo del simulador, no solo la observacion."""

    var root_actions: DeviceBuffer[idx_dtype]
    """[P] la accion de profundidad 0. Es lo unico que al final se ejecuta."""

    var resample_td_weights: DeviceBuffer[dtype]
    """[P] suma de errores TD desde el ultimo resampling. Es el log-peso SMC."""

    var prior_logits: DeviceBuffer[dtype]
    """[P] log-prob de la accion de la particula bajo el prior. Informativo."""

    var value: DeviceBuffer[dtype]
    """[P] V(s) del estado actual de la particula."""

    var terminal: DeviceBuffer[idx_dtype]
    """[P] 1 si la particula ya murio. Es pegajoso: una vez a 1 no vuelve a 0."""

    var depth: DeviceBuffer[idx_dtype]
    """[P] cuantos pasos ha avanzado."""

    var gae: DeviceBuffer[dtype]
    """[P] ventaja acumulada hacia adelante. Alimenta el loss de la temperatura."""

    def __init__(out self, ctx: DeviceContext, config: SPOConfig) raises:
        p = config.num_search_particles()
        self.state = ctx.enqueue_create_buffer[dtype](p * config.state_dim)
        self.state.enqueue_fill(0)
        self.root_actions = ctx.enqueue_create_buffer[idx_dtype](p)
        self.root_actions.enqueue_fill(0)
        self.resample_td_weights = ctx.enqueue_create_buffer[dtype](p)
        self.resample_td_weights.enqueue_fill(0)
        self.prior_logits = ctx.enqueue_create_buffer[dtype](p)
        self.prior_logits.enqueue_fill(0)
        self.value = ctx.enqueue_create_buffer[dtype](p)
        self.value.enqueue_fill(0)
        self.terminal = ctx.enqueue_create_buffer[idx_dtype](p)
        self.terminal.enqueue_fill(0)
        self.depth = ctx.enqueue_create_buffer[idx_dtype](p)
        self.depth.enqueue_fill(0)
        self.gae = ctx.enqueue_create_buffer[dtype](p)
        self.gae.enqueue_fill(0)


struct StepOutputs(Movable):
    """Lo que devuelve un paso del modelo: el `SPORecurrentFnOutput` de Stoix.

    Son buffers de trabajo que se reescriben en cada profundidad; los saco de
    Particles porque no forman parte del estado de la particula, solo del paso.
    """

    var reward: DeviceBuffer[dtype]
    """[P] recompensa del paso."""

    var discount: DeviceBuffer[dtype]
    """[P] el rec_discount de Stoix: discount*(1-truncated). A 0 marca "esta
    particula dejo de simular", tanto si murio de verdad como si la truncaron."""

    var next_value: DeviceBuffer[dtype]
    """[P] el bootstrap_value de Stoix: discount_real * search_gamma * V(s').
    Cuidado con la diferencia respecto al campo de arriba: en una truncacion el
    discount vale 0 pero esto no, porque el estado truncado si tiene futuro y hay
    que arrastrar su valor."""

    var next_action: DeviceBuffer[idx_dtype]
    """[P] la accion que la particula ejecutara en la SIGUIENTE profundidad."""

    var next_prior_logits: DeviceBuffer[dtype]
    """[P] log-prob de esa accion bajo el prior."""

    var action_logits: DeviceBuffer[dtype]
    """[P, num_actions] logits del prior en el nuevo estado. Es de donde se
    muestrea next_action, y se guarda porque el muestreo va en otro kernel."""

    def __init__(out self, ctx: DeviceContext, config: SPOConfig) raises:
        p = config.num_search_particles()
        self.reward = ctx.enqueue_create_buffer[dtype](p)
        self.reward.enqueue_fill(0)
        self.discount = ctx.enqueue_create_buffer[dtype](p)
        self.discount.enqueue_fill(0)
        self.next_value = ctx.enqueue_create_buffer[dtype](p)
        self.next_value.enqueue_fill(0)
        self.next_action = ctx.enqueue_create_buffer[idx_dtype](p)
        self.next_action.enqueue_fill(0)
        self.next_prior_logits = ctx.enqueue_create_buffer[dtype](p)
        self.next_prior_logits.enqueue_fill(0)
        self.action_logits = ctx.enqueue_create_buffer[dtype](p * config.num_actions)
        self.action_logits.enqueue_fill(0)


struct SearchScratch(Movable):
    """Buffers auxiliares del resampling.

    El resampling es un gather: la particula i pasa a ser una copia de la
    particula idx[i]. Hacerlo in-place seria una carrera de libro, porque un hilo
    puede escribir su destino antes de que otro haya leido ese mismo hueco, asi
    que hace falta un buffer intermedio por campo.

    Solo estan los seis campos que se copian. `resample_td_weights` no hace falta
    porque se resetea a cero, y `gae` tampoco porque se preserva sin reordenar
    (ver la nota en resample()).
    """

    var state: DeviceBuffer[dtype]
    var root_actions: DeviceBuffer[idx_dtype]
    var prior_logits: DeviceBuffer[dtype]
    var value: DeviceBuffer[dtype]
    var terminal: DeviceBuffer[idx_dtype]
    var depth: DeviceBuffer[idx_dtype]

    var indices: DeviceBuffer[idx_dtype]
    """[P] a que particula copia cada hueco."""

    var resample_logits: DeviceBuffer[dtype]
    """[P] los pesos SMC divididos por la temperatura."""

    def __init__(out self, ctx: DeviceContext, config: SPOConfig) raises:
        p = config.num_search_particles()
        self.state = ctx.enqueue_create_buffer[dtype](p * config.state_dim)
        self.state.enqueue_fill(0)
        self.root_actions = ctx.enqueue_create_buffer[idx_dtype](p)
        self.root_actions.enqueue_fill(0)
        self.prior_logits = ctx.enqueue_create_buffer[dtype](p)
        self.prior_logits.enqueue_fill(0)
        self.value = ctx.enqueue_create_buffer[dtype](p)
        self.value.enqueue_fill(0)
        self.terminal = ctx.enqueue_create_buffer[idx_dtype](p)
        self.terminal.enqueue_fill(0)
        self.depth = ctx.enqueue_create_buffer[idx_dtype](p)
        self.depth.enqueue_fill(0)
        self.indices = ctx.enqueue_create_buffer[idx_dtype](p)
        self.indices.enqueue_fill(0)
        self.resample_logits = ctx.enqueue_create_buffer[dtype](p)
        self.resample_logits.enqueue_fill(0)


struct SPOOutput(Movable):
    """El resultado publico de una busqueda. Espeja SPOOutput de spo_types.py."""

    var action: DeviceBuffer[idx_dtype]
    """[num_envs] la accion que se ejecuta de verdad en el entorno."""

    var sampled_actions: DeviceBuffer[idx_dtype]
    """[P] las N acciones raiz que sobrevivieron. Su histograma es la politica
    mejorada q, que es justo lo que el M-step intenta imitar."""

    var sampled_action_weights: DeviceBuffer[dtype]
    """[P] el peso de cada una: softmax(w/temperatura) por env."""

    var value: DeviceBuffer[dtype]
    """[num_envs] la media de V(s_raiz) sobre las particulas."""

    var sampled_advantages: DeviceBuffer[dtype]
    """[P] la gae de cada particula. Alimenta el loss de la temperatura."""

    var root_values: DeviceBuffer[dtype]
    """[P] copia de V(s_raiz) por particula, guardada antes de que el rollout
    pise particles.value."""

    var ess: DeviceBuffer[dtype]
    """[search_depth, num_envs] tamano de muestra efectivo en cada profundidad."""

    var entropy: DeviceBuffer[dtype]
    """[search_depth, num_envs] entropia de los pesos en cada profundidad."""

    def __init__(out self, ctx: DeviceContext, config: SPOConfig) raises:
        p = config.num_search_particles()
        self.action = ctx.enqueue_create_buffer[idx_dtype](config.num_envs)
        self.action.enqueue_fill(0)
        self.sampled_actions = ctx.enqueue_create_buffer[idx_dtype](p)
        self.sampled_actions.enqueue_fill(0)
        self.sampled_action_weights = ctx.enqueue_create_buffer[dtype](p)
        self.sampled_action_weights.enqueue_fill(0)
        self.value = ctx.enqueue_create_buffer[dtype](config.num_envs)
        self.value.enqueue_fill(0)
        self.sampled_advantages = ctx.enqueue_create_buffer[dtype](p)
        self.sampled_advantages.enqueue_fill(0)
        self.root_values = ctx.enqueue_create_buffer[dtype](p)
        self.root_values.enqueue_fill(0)
        metrics = config.search_depth * config.num_envs
        self.ess = ctx.enqueue_create_buffer[dtype](metrics)
        self.ess.enqueue_fill(0)
        self.entropy = ctx.enqueue_create_buffer[dtype](metrics)
        self.entropy.enqueue_fill(0)
