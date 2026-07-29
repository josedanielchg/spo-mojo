"""Como se lanzan los kernels de la busqueda: tamanos de bloque y comprobaciones.

Esta aparte porque lo usan todos los ficheros del E-step, y porque los dos
tamanos de bloque no son intercambiables: elegir el que no toca es un bug
silencioso, no un error de compilacion.
"""

from systems.spo.spo_types import SPOConfig

comptime TPB = 32
"""Bloque de los kernels que son un map plano: un hilo por particula.

Aqui la fila no significa nada; solo importa cubrir P elementos, asi que el
tamano del bloque es libre y 32 (un warp) va bien."""

comptime TPB_PARTICLES = 512
"""Bloque de los kernels cuya FILA es la dimension de particulas (resampling,
ESS, softmax del readout). Ahi el bloque entero trabaja sobre las N particulas de
un env, asi que N tiene que caber dentro.

Es 128 y no 32 para que la demo pueda barrer N = 64. Los hilos de mas no cuestan
nada: los guards los desactivan y la GPU no lanza warps enteros ociosos.
Con N = 16 (lo del paper) sobra de largo."""


def blocks_for(n: Int) -> Int:
    """Cuantos bloques de TPB hilos hacen falta para cubrir n elementos.

    Es `(n + TPB - 1) // TPB`, que estaba escrito a mano unas quince veces. El
    redondeo hacia arriba es la razon de que TODO kernel de map necesite su guard
    `if i < n`: casi siempre se lanzan mas hilos que datos.
    """
    return (n + TPB - 1) // TPB


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
