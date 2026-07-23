"""Runners de CartPole: jugar episodios REALES y medir el retorno medio.

Separa el 'jugar' del 'modelo': `cartpole.mojo` es el entorno; esto es el arnes
que lo hace correr episodios y lleva la cuenta de los retornos. Lo usan la demo 2
y su test.

De momento solo la politica ALEATORIA (la linea base). El runner de la busqueda
se anade en la demo, que ya trae las piezas de systems.spo.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext

from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform
from envs.cartpole import (cartpole_reset_kernel, cartpole_step_envs,
                          random_actions_kernel, STATE_DIM, TPB_CART)

# Streams del RNG del runner, separados por uso.
comptime RNG_RESET_INIT = UInt32(0)
comptime RNG_ACTION = UInt32(1000)
comptime RNG_AUTORESET = UInt32(5000)


@fieldwise_init
struct RolloutStats(Movable):
    """El resumen de una tanda de episodios."""
    var mean_return: Scalar[dtype]
    var num_episodes: Int


def random_policy_return(ctx: DeviceContext, num_envs: Int, num_steps: Int,
                         seed: UInt32) raises -> RolloutStats:
    """Corre `num_steps` pasos con accion aleatoria en `num_envs` entornos y
    devuelve el retorno medio por episodio.

    El seguimiento del retorno va en el HOST: cada paso se bajan reward y done, se
    acumula el retorno por env, y cuando un env termina se anota su retorno y su
    contador vuelve a cero. Es tráfico host<->device por paso (consciente: la demo
    no es el bucle caliente), a cambio de un bucle simple y claro."""
    blocks = (num_envs + TPB_CART - 1) // TPB_CART

    state = zero_buffer[dtype](ctx, num_envs * STATE_DIM)
    actions = zero_buffer[idx_dtype](ctx, num_envs)
    reward = zero_buffer[dtype](ctx, num_envs)
    done = zero_buffer[idx_dtype](ctx, num_envs)
    u_action = zero_buffer[dtype](ctx, num_envs)
    u_reset = zero_buffer[dtype](ctx, num_envs * 4)

    # Reset inicial de todos los envs.
    ctx.enqueue_function[fill_uniform, fill_uniform](
        u_reset.unsafe_ptr(), seed, RNG_RESET_INIT, num_envs * 4,
        grid_dim=(num_envs * 4 + TPB_CART - 1) // TPB_CART, block_dim=TPB_CART)
    ctx.enqueue_function[cartpole_reset_kernel, cartpole_reset_kernel](
        state.unsafe_ptr(), u_reset.unsafe_ptr(), num_envs,
        grid_dim=blocks, block_dim=TPB_CART)

    # Retorno acumulado del episodio en curso por env (host), y los completados.
    returns = List[Scalar[dtype]]()
    for _ in range(num_envs):
        returns.append(Scalar[dtype](0))
    episodes = List[Scalar[dtype]]()

    for step in range(num_steps):
        # Accion aleatoria fresca para cada env.
        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_action.unsafe_ptr(), seed, RNG_ACTION + UInt32(step), num_envs,
            grid_dim=blocks, block_dim=TPB_CART)
        ctx.enqueue_function[random_actions_kernel, random_actions_kernel](
            actions.unsafe_ptr(), u_action.unsafe_ptr(), num_envs,
            grid_dim=blocks, block_dim=TPB_CART)

        # Un paso del entorno real + auto-reset de los que terminen.
        cartpole_step_envs(ctx, state, actions, reward, done, u_reset, num_envs,
                           seed, RNG_AUTORESET + UInt32(step))
        ctx.synchronize()

        # Bookkeeping en host.
        with reward.map_to_host() as rh:
            with done.map_to_host() as dh:
                for e in range(num_envs):
                    returns[e] += rh[e]
                    if Int(dh[e]) != 0:
                        episodes.append(returns[e])
                        returns[e] = Scalar[dtype](0)

    total = Scalar[dtype](0)
    for i in range(len(episodes)):
        total += episodes[i]
    n = len(episodes)
    mean = total / Scalar[dtype](n) if n > 0 else Scalar[dtype](0)
    return RolloutStats(mean, n)
