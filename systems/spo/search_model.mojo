"""El contrato que la busqueda le pide a un modelo. Solo esto.

Tiene fichero propio a proposito: es la frontera del sistema. Lo que este de este
lado (dos metodos) es todo lo que hay que escribir para enchufar un entorno nuevo
a la busqueda; el resto del E-step no se toca.

Que viva aqui y no en el fichero de la busqueda importa para la direccion de las
dependencias: un entorno implementa `SearchModel` importando solo el contrato y
los tipos de datos, sin depender del algoritmo. Asi `envs/cartpole.mojo` podra
usarse como entorno real, en el bucle de aprendizaje, sin arrastrar el E-step.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from ops.common import dtype
from systems.spo.particles import Particles, StepOutputs
from systems.spo.spo_types import SPOConfig


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
             outputs: StepOutputs, step_uniforms: DeviceBuffer[dtype]) raises:
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

        `step_uniforms` [P] son uniformes en [0,1), uno por particula, para los
        modelos con transicion ESTOCASTICA (p. ej. TTT contra un rival aleatorio:
        con ellos se elige la casilla del rival). Un modelo determinista los ignora;
        vienen de un stream RNG propio de la busqueda, distinto por profundidad.

        Lo que NO hace: tocar `particles.value` (el error TD necesita el V viejo) ni
        sortear la accion siguiente -- de eso se encarga `sample_next_actions`, que
        es generico y lo llama la busqueda.
        """
        ...
