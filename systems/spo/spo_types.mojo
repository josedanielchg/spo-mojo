"""Los hiperparametros de la busqueda SPO.

Este fichero es solo la configuracion. Lo demas del vocabulario esta al lado:

    particles.mojo      los contenedores (Particles, StepOutputs, SPOOutput...)
    search_model.mojo   el contrato que la busqueda le pide a un modelo

Los nombres son los de Stoix a proposito (`stoix/configs/system/spo/ff_spo.yaml`)
para poder poner los dos ficheros uno al lado del otro y comparar.
"""

from ops.common import dtype


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
