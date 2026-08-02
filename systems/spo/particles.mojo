"""Los contenedores de la busqueda: el enjambre, el paso, el scratch y la salida.

Todos siguen el mismo patron: struct-of-arrays con `DeviceBuffer` planos de P =
envs*particulas elementos, indexados por `p = env * num_particles + n`. Stoix usa
arrays [NumEnvs, NumParticles, ...] y deja que JAX haga el tree_map; aqui el
indice plano hace que la mayoria de kernels sean un map de un hilo por particula.

Cuatro contenedores y un bundle:

    Particles       el estado del enjambre, lo que sobrevive de una profundidad
                    a la siguiente
    StepOutputs     lo que produce UN paso del modelo, se reescribe cada vez
    SearchScratch   buffers intermedios del resampling
    SPOOutput       el resultado publico de la busqueda
    SearchWorkspace los cuatro anteriores mas los uniformes, reservado una vez
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from systems.spo.spo_types import SPOConfig


struct Particles(Movable):
    """Las P trayectorias hipoteticas: el estado del enjambre.

    Los nombres son los de Stoix a proposito (ver `Particles` en ff_spo.py).
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
        self.state = zero_buffer[dtype](ctx, p * config.state_dim)
        self.root_actions = zero_buffer[idx_dtype](ctx, p)
        self.resample_td_weights = zero_buffer[dtype](ctx, p)
        self.prior_logits = zero_buffer[dtype](ctx, p)
        self.value = zero_buffer[dtype](ctx, p)
        self.terminal = zero_buffer[idx_dtype](ctx, p)
        self.depth = zero_buffer[idx_dtype](ctx, p)
        self.gae = zero_buffer[dtype](ctx, p)


struct StepOutputs(Movable):
    """Lo que devuelve un paso del modelo: el `SPORecurrentFnOutput` de Stoix.

    Son buffers de trabajo que se reescriben en cada profundidad; estan fuera de
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
        self.reward = zero_buffer[dtype](ctx, p)
        self.discount = zero_buffer[dtype](ctx, p)
        self.next_value = zero_buffer[dtype](ctx, p)
        self.next_action = zero_buffer[idx_dtype](ctx, p)
        self.next_prior_logits = zero_buffer[dtype](ctx, p)
        self.action_logits = zero_buffer[dtype](ctx, p * config.num_actions)


struct SearchScratch(Movable):
    """Buffers auxiliares del resampling.

    El resampling es un gather: la particula i pasa a ser una copia de la
    particula idx[i]. Hacerlo in-place seria una carrera de libro, porque un hilo
    puede escribir su destino antes de que otro haya leido ese mismo hueco, asi
    que hace falta un buffer intermedio por campo.

    Solo estan los seis campos que se copian. `resample_td_weights` no hace falta
    porque se resetea a cero, y `gae` tampoco porque se preserva sin reordenar
    (ver la nota en resampling.mojo).
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
        self.state = zero_buffer[dtype](ctx, p * config.state_dim)
        self.root_actions = zero_buffer[idx_dtype](ctx, p)
        self.prior_logits = zero_buffer[dtype](ctx, p)
        self.value = zero_buffer[dtype](ctx, p)
        self.terminal = zero_buffer[idx_dtype](ctx, p)
        self.depth = zero_buffer[idx_dtype](ctx, p)
        self.indices = zero_buffer[idx_dtype](ctx, p)
        self.resample_logits = zero_buffer[dtype](ctx, p)


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
        metrics = config.search_depth * config.num_envs
        self.action = zero_buffer[idx_dtype](ctx, config.num_envs)
        self.sampled_actions = zero_buffer[idx_dtype](ctx, p)
        self.sampled_action_weights = zero_buffer[dtype](ctx, p)
        self.value = zero_buffer[dtype](ctx, config.num_envs)
        self.sampled_advantages = zero_buffer[dtype](ctx, p)
        self.root_values = zero_buffer[dtype](ctx, p)
        self.ess = zero_buffer[dtype](ctx, metrics)
        self.entropy = zero_buffer[dtype](ctx, metrics)


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

    var u_step: DeviceBuffer[dtype]
    """[P] uniformes para la transicion estocastica del modelo (p. ej. la jugada
    del rival aleatorio en TTT). Uno por particula, se rellena de nuevo en cada
    profundidad desde su propio stream. Un modelo determinista los ignora."""

    var u_noise: DeviceBuffer[dtype]
    """[num_envs, num_actions] uniformes del ruido de Dirichlet en la raiz. Va en
    su propio buffer para que el ruido no comparta secuencia con el muestreo de
    acciones: si la compartieran, la exploracion que el ruido intenta anadir
    estaria correlacionada con lo que ya se muestreo."""

    var u_resample: DeviceBuffer[dtype]
    """[P] uniformes del resampling, en un buffer aparte para que el muestreo de
    acciones y el de resampling no compartan secuencia."""

    def __init__(out self, ctx: DeviceContext, cfg: SPOConfig) raises:
        p = cfg.num_search_particles()
        self.particles = Particles(ctx, cfg)
        self.outputs = StepOutputs(ctx, cfg)
        self.scratch = SearchScratch(ctx, cfg)
        self.output = SPOOutput(ctx, cfg)
        self.root_logits = zero_buffer[dtype](ctx, cfg.num_envs * cfg.num_actions)
        self.root_value = zero_buffer[dtype](ctx, cfg.num_envs)
        self.u_action = zero_buffer[dtype](ctx, p)
        self.u_step = zero_buffer[dtype](ctx, p)
        self.u_noise = zero_buffer[dtype](ctx, cfg.num_envs * cfg.num_actions)
        self.u_resample = zero_buffer[dtype](ctx, p)
