"""Tipos de la busqueda SPO. Similar a stoix/systems/spo/spo_types.py.

Mantenemos los nombres de Stoix (`Particles`, `resample_td_weights`, `root_actions`...)
para poder poner los dos ficheros uno al lado del otro y comparar.

Aqui vive tambien `SearchModel`, el contrato que la busqueda le pide a un modelo.
En Stoix la clase SPO recibe un `recurrent_fn` abstracto y no sabe si detras hay
una red neuronal, un entorno o un MDP de juguete; esto es lo mismo en Mojo.

Que el contrato viva en este fichero y no en smc_search.mojo es a proposito: asi
un entorno implementa `SearchModel` importando solo los tipos de datos, sin
depender del algoritmo de busqueda. La flecha va entorno -> tipos <- busqueda, y
`envs/cartpole.mojo` podra usarse como entorno real sin arrastrar el E-step.
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
    """Es la cantidad de números necesarios para representar el estado de un entorno."""

    var search_depth: Int
    
    var resample_period: Int
    """Cada cuantos pasos se hace resampling (modo 'period' de Stoix)."""
    
    var temperature: Scalar[dtype]
    """Temperatura fija. Luego la haremos dinamica de acuerdo a lo aprendido por la politica."""

    var search_gamma: Scalar[dtype]
    """Indica cuánto importan las recompensas futuras:
        gamma = 1.0 implica que futuro y presente pesan igual
        gamma = 0.9 entonces cada paso futuro pesa un 90% del anterior
    """

    var search_gae_lambda: Scalar[dtype]    
    """Controla cuánto se acumulan los errores TD de varios pasos para calcular la ventaja. 
            lambda bajo = ventaja más basada en pasos cercanos
            lambda alto = considera más profundamente la trayectoria completa
    """


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
    """Las P = envs*particulas trayectorias hipoteticas, como struct-of-arrays (SoA).

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
    """[P] suma de errores TD desde el ultimo resampling. Peso bajo trayectoria prometedora"""

    var prior_logits: DeviceBuffer[dtype]
    """[P] Se guarda el logaritmo de la probabilidad de la acción elegida. log π(acción elegida | estado)
    Servira para comparar posteriormente la politica original con la politica mejorada durante el M-step."""

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
    """[num_envs] la accion que se ejecuta de verdad en el entorno. Una sola por env"""

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


struct SearchWorkspace(Movable):
    """Toda la memoria que una busqueda necesita, reservada UNA vez.

    Antes cada busqueda se reservaba sus propios buffers de uniformes y de raiz al
    entrar y los soltaba al salir. En el juguete daba igual, pero en la fase 8 la
    busqueda corre en cada paso de entorno, y reservar memoria de device miles de
    veces es trabajo puro por nada. Aqui se construye el workspace una vez y se
    reutiliza en todas las llamadas a `search`.
    """

    var particles: Particles
    var outputs: StepOutputs
    var scratch: SearchScratch
    var output: SPOOutput

    var root_logits: DeviceBuffer[dtype]
    """[num_envs, num_actions] el prior evaluado en los estados raiz."""

    var root_value: DeviceBuffer[dtype]
    """[num_envs] V(s_raiz)."""

    var u_action: DeviceBuffer[dtype]
    """[P] uniformes para sortear acciones. Se rellena de nuevo en cada
    profundidad; se puede reutilizar sin miedo porque el stream es unico y ejecuta
    en orden: el relleno siguiente no puede adelantar al kernel que leyo el anterior."""

    var u_resample: DeviceBuffer[dtype]
    """[P] uniformes del resampling, en un buffer aparte para que el muestreo de
    acciones y el de resampling no compartan secuencia."""

    def __init__(out self, ctx: DeviceContext, cfg: SPOConfig) raises:
        p = cfg.num_search_particles()
        self.particles = Particles(ctx, cfg)
        self.outputs = StepOutputs(ctx, cfg)
        self.scratch = SearchScratch(ctx, cfg)
        self.output = SPOOutput(ctx, cfg)

        self.root_logits = ctx.enqueue_create_buffer[dtype](
            cfg.num_envs * cfg.num_actions)
        self.root_logits.enqueue_fill(0)
        self.root_value = ctx.enqueue_create_buffer[dtype](cfg.num_envs)
        self.root_value.enqueue_fill(0)
        self.u_action = ctx.enqueue_create_buffer[dtype](p)
        self.u_action.enqueue_fill(0)
        self.u_resample = ctx.enqueue_create_buffer[dtype](p)
        self.u_resample.enqueue_fill(0)


trait SearchModel:
    """Lo que la busqueda necesita saber hacer a un modelo. Son dos cosas.

    Es el equivalente del `recurrent_fn` abstracto que recibe la clase SPO de
    Stoix: el nucleo SMC (pesos, GAE, resampling, ESS, readout) no sabe si detras
    hay un MDP de juguete, CartPole o un MLP.

    Como funciona esto en Mojo 1.0.0b1, que es lo que costo encontrar: el modelo es
    una INSTANCIA que se queda siempre en el host, y sus `enqueue_function` viven
    DENTRO de sus propios metodos, donde el simbolo del kernel es concreto. El
    kernel nunca cruza la frontera generica; solo la cruza el tipo. Intentar lo
    contrario (pasar el kernel como parametro comptime, o como funcion de device)
    no compila -- ver docs/api_notes.md.

    Que el modelo sea una instancia y no un tipo suelto tambien importa: asi puede
    llevar estado propio. El juguete lleva tres numeros; el MLP de la fase 5
    llevara los DeviceBuffer de los pesos del actor y del critico.
    """

    def eval_root(self, ctx: DeviceContext, cfg: SPOConfig,
                  root_state: DeviceBuffer[dtype],
                  logits_out: DeviceBuffer[dtype],
                  value_out: DeviceBuffer[dtype]) raises:
        """El prior y el valor en los estados RAIZ, uno por entorno.

            root_state  [num_envs, state_dim]   entrada
            logits_out  [num_envs, num_actions] salida: los logits del prior
            value_out   [num_envs]              salida: V(s_raiz)
        """
        ...

    def step(self, ctx: DeviceContext, cfg: SPOConfig, particles: Particles,
             outputs: StepOutputs) raises:
        """Avanza las P particulas una profundidad. Es el `recurrent_fn` del modelo.

        Lee  `particles.state` y `outputs.next_action` (la accion que toca ejecutar)
        y escribe:
            particles.state         el estado nuevo, in-place
            outputs.reward          [P] recompensa del paso
            outputs.discount        [P] rec_discount: discount * (1 - truncated)
            outputs.next_value      [P] bootstrap: discount_real * search_gamma * V(s')
            outputs.action_logits   [P, num_actions] el prior en el estado NUEVO

        El plegado de gamma y de la truncacion es responsabilidad del modelo, igual
        que en el `recurrent_fn` de Stoix, para que el nucleo SMC no tenga que saber
        nada del entorno.

        Lo que NO hace: tocar `particles.value` (el error TD necesita el V viejo) ni
        sortear la accion siguiente -- de eso se encarga `sample_next_actions`, que
        es generico y lo llama la busqueda.
        """
        ...
