"""El arnes que hace jugar partidas de verdad y lleva la cuenta.

Separado de `tictactoe.mojo` a proposito: alli estan las REGLAS (tablero, turno,
reset) y aqui quien las hace correr en un bucle y cuenta los resultados. Mismo
reparto que tenia el runner de CartPole.

De momento solo la LINEA BASE: el agente juega al azar. Es el numero contra el que
se compara la busqueda en la fase A6 -- si planificar no gana mas partidas que
tirar una casilla al azar, la busqueda no esta aportando nada.

El seguimiento va en el HOST: cada turno se bajan reward y done a la CPU y se
anotan las partidas que acaban. Es trafico host<->device por paso, pero esto no es
el bucle caliente (el learner del M-step si lo sera), asi que aqui prima la
claridad. Moverlo a device queda como mejora futura.
"""

from std.gpu.host import DeviceContext

from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform
from envs.tictactoe import (ttt_reset_kernel, ttt_env_step_kernel,
                            ttt_auto_reset_kernel, ttt_random_policy_kernel,
                            STATE_DIM, TPB_TTT)

# Streams del RNG del entorno real. Van en rangos altos para no cruzarse con los
# de la busqueda (raiz 0, acciones 100+d, paso 500+d, resampling 900+d, readout
# 7777), asi las partidas y la planificacion no comparten secuencia ni cuando
# comparten semilla.
# Flux de nombres aleatoires au niveau de l'ENVIRONNEMENT. Chacun est decale par
# un compteur de pas qui, sur un entrainement long, atteint plusieurs dizaines de
# milliers : avec 16 pas par ronde et 1172 rondes, il monte a 18 752. Les flux
# doivent donc etre espaces de bien plus que cela, sinon deux d'entre eux finissent
# par tirer EXACTEMENT les memes nombres.
#
# Ce n'est pas hypothetique : espaces de 10 000, `RNG_RIVAL` et `RNG_READOUT` se
# recouvraient a partir de la ronde 625, et l'adversaire rejouait alors les tirages
# que la lecture avait deja utilises. C'est la meme faute que celle relevee dans
# l'implementation de reference -- deux usages independants partageant une source.
# Un espacement de 10^6 laisse de la marge pour environ cinquante fois plus de
# rondes que la campagne la plus longue menee ici.
comptime RNG_POLICY = UInt32(1_000_000)
comptime RNG_RIVAL = UInt32(2_000_000)


@fieldwise_init
struct MatchStats(Copyable, Movable):
    """El marcador de una tanda de partidas, desde la vista del agente (X)."""

    var wins: Int
    var draws: Int
    var losses: Int

    def games(self) -> Int:
        return self.wins + self.draws + self.losses

    def win_rate(self) -> Scalar[dtype]:
        """Fraccion de partidas ganadas. 0 si no se completo ninguna."""
        n = self.games()
        if n == 0:
            return Scalar[dtype](0)
        return Scalar[dtype](self.wins) / Scalar[dtype](n)

    def score(self) -> Scalar[dtype]:
        """Puntuacion media por partida: 1 victoria, 0.5 empate, 0 derrota.

        Resume las tres cifras en un numero, y es exactamente la recompensa media
        que ve el agente, asi que es lo que compara con la busqueda.
        """
        n = self.games()
        if n == 0:
            return Scalar[dtype](0)
        return (Scalar[dtype](self.wins) + Scalar[dtype](0.5) * Scalar[dtype](self.draws)) \
               / Scalar[dtype](n)


def play_random_games(ctx: DeviceContext, num_envs: Int, num_steps: Int,
                      seed: UInt32) raises -> MatchStats:
    """Juega `num_steps` turnos en `num_envs` partidas paralelas, al azar.

    El agente elige una casilla libre al azar y el rival tambien. Cada partida que
    acaba se anota y el env empieza otra (auto-reset), asi que en `num_steps`
    turnos salen del orden de num_envs*num_steps/3.5 partidas completas.

    Devuelve el marcador. Las partidas que quedan a medias al agotarse los turnos
    no se cuentan; eso sub-representa ligeramente las partidas largas (una queda
    sin terminar por env), asi que conviene usar bastantes turnos para que el sesgo
    sea despreciable.
    """
    blocks = (num_envs + TPB_TTT - 1) // TPB_TTT

    state = zero_buffer[dtype](ctx, num_envs * STATE_DIM)
    action = zero_buffer[idx_dtype](ctx, num_envs)
    reward = zero_buffer[dtype](ctx, num_envs)
    done = zero_buffer[idx_dtype](ctx, num_envs)
    u_policy = zero_buffer[dtype](ctx, num_envs)
    u_rival = zero_buffer[dtype](ctx, num_envs)

    ctx.enqueue_function[ttt_reset_kernel, ttt_reset_kernel](
        state.unsafe_ptr(), num_envs, grid_dim=blocks, block_dim=TPB_TTT)

    wins = 0
    draws = 0
    losses = 0

    for step in range(num_steps):
        # La jugada del agente: una casilla libre al azar.
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_policy.unsafe_ptr(), seed, RNG_POLICY + UInt32(step), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_random_policy_kernel, ttt_random_policy_kernel](
            action.unsafe_ptr(), state.unsafe_ptr(), u_policy.unsafe_ptr(),
            num_envs, grid_dim=blocks, block_dim=TPB_TTT)

        # El turno completo (el rival responde dentro del kernel).
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_rival.unsafe_ptr(), seed, RNG_RIVAL + UInt32(step), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
            state.unsafe_ptr(), action.unsafe_ptr(), u_rival.unsafe_ptr(),
            reward.unsafe_ptr(), done.unsafe_ptr(), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

        # Anota las que acabaron. La recompensa dice el resultado: 1 / 0.5 / 0.
        with reward.map_to_host() as rh:
            with done.map_to_host() as dh:
                for e in range(num_envs):
                    if Int(dh[e]) != 0:
                        r = rh[e]
                        if r > Scalar[dtype](0.75):
                            wins += 1
                        elif r > Scalar[dtype](0.25):
                            draws += 1
                        else:
                            losses += 1

        # Y las partidas terminadas empiezan otra. Va DESPUES de leer el resultado.
        ctx.enqueue_function[ttt_auto_reset_kernel, ttt_auto_reset_kernel](
            state.unsafe_ptr(), done.unsafe_ptr(), num_envs,
            grid_dim=blocks, block_dim=TPB_TTT)

    return MatchStats(wins, draws, losses)


def main() raises:
    """Corre la linea base y la imprime, para poder mirarla a ojo.

    Referencia exacta (calculada por recursion sobre todos los estados, con las dos
    partes jugando uniformemente al azar): X gana 58.49%, empata 12.70% y pierde
    28.81%, o sea una puntuacion media de 0.6484. Si estas cifras salen, el bucle
    entero -- turno, deteccion de final, conteo y auto-reset -- esta bien.
    """
    with DeviceContext() as ctx:
        stats = play_random_games(ctx, 64, 200, UInt32(12345))
        print("linea base (agente al azar vs rival al azar):")
        print("  partidas :", stats.games())
        print("  gana X   :", stats.wins, "(", stats.win_rate(), ")")
        print("  empate   :", stats.draws)
        print("  gana O   :", stats.losses)
        print("  score    :", stats.score(), " (exacto: 0.6484 )")
