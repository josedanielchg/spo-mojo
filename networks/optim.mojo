"""Adam and the global-norm clip. Learning's "correct" step.

The backward says IN WHICH DIRECTION to move each weight; this decides HOW MUCH.
Without an optimiser, a large gradient would take an enormous jump and a small one
would move nothing, and training would be unstable.

Adam keeps two moving averages per weight:

    m  the recent direction         (mean of the gradient)
    v  how much that weight moves   (mean of the squared gradient)

and advances by `lr * m / (sqrt(v) + eps)`. The division is what makes it robust:
a weight whose gradient oscillates a lot (large v) moves little, and one with a
small but steady gradient advances just the same. Both averages start at zero, so
at the beginning they underestimate; that is why they are corrected by dividing by
(1 - beta^t), and that correction depends on the STEP NUMBER.

Before Adam comes the global-norm clip, exactly as in Stoix:

    optax.chain(clip_by_global_norm(max_norm), adam(lr, eps=1e-5))

"Global" means the norm is computed over ALL the tensors together, not tensor by
tensor. It is the typical mistake when reimplementing it: clipping each tensor
separately changes the DIRECTION of the joint gradient, and this only changes its
length.

Mind eps: Stoix sets 1e-5 explicitly, not optax's default 1e-8.
"""

from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt
from std.memory import stack_allocation, AddressSpace

from ops.buffers import zero_buffer
from ops.common import dtype, GlobalF32

comptime TPB_OPT = 256
"""Threads per block for the optimiser's kernels."""

comptime BETA1 = Scalar[dtype](0.9)
comptime BETA2 = Scalar[dtype](0.999)
comptime ADAM_EPS = Scalar[dtype](1e-5)
"""Stoix's values. b1/b2 are optax's defaults; eps is NOT (optax ships 1e-8,
Stoix raises it to 1e-5)."""


def sum_squares_kernel[TPB: Int](partials: GlobalF32, x: GlobalF32, n: Int):
    """Sum of squares per block: `partials[block] = sum of x[i]^2`.

    Tree reduction in shared memory (Puzzle 12). Each block leaves a partial and
    the host sums them: it is a few floats of traffic and it avoids having to
    coordinate blocks, which on a GPU calls for another kernel (Puzzle 14).
    """
    shared = stack_allocation[TPB, Scalar[dtype],
                              address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    i = Int(block_dim.x * block_idx.x + tid)

    v = x[i] if i < n else Scalar[dtype](0)
    shared[tid] = v * v
    barrier()

    stride = TPB // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        barrier()
        stride //= 2

    if tid == 0:
        partials[Int(block_idx.x)] = shared[0]


def adam_step_kernel(param: GlobalF32, grad: GlobalF32, m: GlobalF32,
                     v: GlobalF32, n: Int, lr: Scalar[dtype],
                     grad_scale: Scalar[dtype], bc1: Scalar[dtype],
                     bc2: Scalar[dtype]):
    """One Adam step, elementwise.

    `grad_scale` is the global clip's factor (1.0 if it does not clip), applied
    here so as not to have to rewrite the gradients in another pass.

    `bc1` and `bc2` are 1/(1 - beta^t), the bias corrections, computed on the host
    because they depend on the step and not on the element.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= n:
        return

    g = grad[i] * grad_scale
    mi = BETA1 * m[i] + (Scalar[dtype](1) - BETA1) * g
    vi = BETA2 * v[i] + (Scalar[dtype](1) - BETA2) * g * g
    m[i] = mi
    v[i] = vi

    m_hat = mi * bc1
    v_hat = vi * bc2
    param[i] = param[i] - lr * m_hat / (sqrt(v_hat) + ADAM_EPS)


struct AdamState(Movable):
    """One weight tensor's two moments. They start at zero, as in optax."""

    var m: DeviceBuffer[dtype]
    var v: DeviceBuffer[dtype]
    var size: Int

    def __init__(out self, ctx: DeviceContext, size: Int) raises:
        self.m = zero_buffer[dtype](ctx, size)
        self.v = zero_buffer[dtype](ctx, size)
        self.size = size


def sum_squares(ctx: DeviceContext, x: DeviceBuffer[dtype],
                n: Int) raises -> Scalar[dtype]:
    """Sum of squares of a buffer, closing the reduction on the host."""
    blocks = (n + TPB_OPT - 1) // TPB_OPT
    partials = zero_buffer[dtype](ctx, blocks)
    ctx.enqueue_function[sum_squares_kernel[TPB_OPT],
                         sum_squares_kernel[TPB_OPT]](
        partials.unsafe_ptr(), x.unsafe_ptr(), n,
        grid_dim=blocks, block_dim=TPB_OPT)
    ctx.synchronize()

    total = Scalar[dtype](0)
    with partials.map_to_host() as h:
        for i in range(blocks):
            total += h[i]
    return total


def global_clip_scale(total_sq: Scalar[dtype],
                      max_norm: Scalar[dtype]) -> Scalar[dtype]:
    """The clip's factor: 1 if the norm fits, max_norm/norm if it overshoots.

    It is computed from the sum of squares of ALL the tensors, not of one.
    """
    norm = sqrt(total_sq)
    if norm <= max_norm:
        return Scalar[dtype](1)
    return max_norm / norm


def ema_kernel(target: GlobalF32, online: GlobalF32, n: Int,
               tau: Scalar[dtype]):
    """target <- tau*online + (1-tau)*target, elementwise.

    It is Stoix's `optax.incremental_update(online, target, tau)`, verified
    against their code. With tau = 0.005 the target moves 0.5% per step: next to
    nothing.

    What it is for: the critic learns to predict a target that is computed... with
    the critic itself. If the online network were used for both, the target would
    move at the same time as the prediction and training would chase its own tail.
    The slow copy gives a nearly still target.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n:
        target[i] = tau * online[i] + (Scalar[dtype](1) - tau) * target[i]


def ema_update(ctx: DeviceContext, target: DeviceBuffer[dtype],
               online: DeviceBuffer[dtype], n: Int,
               tau: Scalar[dtype]) raises:
    """Moves a target tensor towards its online one."""
    ctx.enqueue_function[ema_kernel, ema_kernel](
        target.unsafe_ptr(), online.unsafe_ptr(), n, tau,
        grid_dim=(n + TPB_OPT - 1) // TPB_OPT, block_dim=TPB_OPT)


def adam_step(ctx: DeviceContext, param: DeviceBuffer[dtype],
              grad: DeviceBuffer[dtype], state: AdamState, n: Int,
              lr: Scalar[dtype], grad_scale: Scalar[dtype], step: Int) raises:
    """Updates one tensor. `step` starts at 1 (as in optax)."""
    b1t = Scalar[dtype](1)
    b2t = Scalar[dtype](1)
    for _ in range(step):
        b1t *= BETA1
        b2t *= BETA2
    bc1 = Scalar[dtype](1) / (Scalar[dtype](1) - b1t)
    bc2 = Scalar[dtype](1) / (Scalar[dtype](1) - b2t)

    ctx.enqueue_function[adam_step_kernel, adam_step_kernel](
        param.unsafe_ptr(), grad.unsafe_ptr(), state.m.unsafe_ptr(),
        state.v.unsafe_ptr(), n, lr, grad_scale, bc1, bc2,
        grid_dim=(n + TPB_OPT - 1) // TPB_OPT, block_dim=TPB_OPT)
