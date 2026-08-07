"""The critic's MLP: a stack of linear layers with ReLU in between.

    x  [M, in_dim]  ──linear──►  ──relu──►  ──linear──►  ──relu──►  ──linear──►  V  [M, out_dim]
                        h1                      h2                     (no relu:
                                                                        V may be
                                                                        negative)

It is generic in the dimensions on purpose: `networks/` imports nothing from
`envs/`, just as the search does not import the environment. Whoever builds it
(the learner) decides what size it has; for tic-tac-toe it will be
18 -> 64 -> 64 -> 1.

The activations are stored in a `CriticCache` instead of being thrown away. It is
not needed for the forward yet, but the manual backward needs them (Puzzle 22's
technique: cache the forward so as not to recompute it when differentiating), and
allocating them here avoids having to change the signature later.

A detail that saves a buffer: the ReLU's mask can be derived from the ALREADY
applied activation (post-relu > 0 if and only if pre-relu > 0), so the relu is
done in place and the previous value need not be stored.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from ops.buffers import zero_buffer
from ops.common import dtype, GlobalF32
from networks.linear import linear_forward, linear_backward

comptime TPB_NET = 256
"""Threads per block for the networks' elementwise kernels."""


def relu_kernel(a: GlobalF32, n: Int):
    """In-place ReLU: a[i] = max(a[i], 0). One thread per element."""
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n:
        if a[i] < Scalar[dtype](0):
            a[i] = Scalar[dtype](0)


def relu(ctx: DeviceContext, a: DeviceBuffer[dtype], n: Int) raises:
    """Enqueues the ReLU over the buffer's first n elements."""
    ctx.enqueue_function[relu_kernel, relu_kernel](
        a.unsafe_ptr(), n,
        grid_dim=(n + TPB_NET - 1) // TPB_NET, block_dim=TPB_NET)


@fieldwise_init
struct CriticParams(Movable):
    """The critic's weights. The struct lives on the host; the buffers, on the GPU."""

    var w1: DeviceBuffer[dtype]
    """[in_dim, hidden]"""
    var b1: DeviceBuffer[dtype]
    """[hidden]"""
    var w2: DeviceBuffer[dtype]
    """[hidden, hidden]"""
    var b2: DeviceBuffer[dtype]
    """[hidden]"""
    var w3: DeviceBuffer[dtype]
    """[hidden, out_dim]"""
    var b3: DeviceBuffer[dtype]
    """[out_dim]"""

    var in_dim: Int
    var hidden: Int
    var out_dim: Int


def zero_critic_params(ctx: DeviceContext, in_dim: Int, hidden: Int,
                       out_dim: Int) raises -> CriticParams:
    """Weights at zero, to be filled in later (from a golden or from an init)."""
    return CriticParams(
        w1=zero_buffer[dtype](ctx, in_dim * hidden),
        b1=zero_buffer[dtype](ctx, hidden),
        w2=zero_buffer[dtype](ctx, hidden * hidden),
        b2=zero_buffer[dtype](ctx, hidden),
        w3=zero_buffer[dtype](ctx, hidden * out_dim),
        b3=zero_buffer[dtype](ctx, out_dim),
        in_dim=in_dim, hidden=hidden, out_dim=out_dim)


struct CriticCache(Movable):
    """One forward's activations, which the backward will read again.

    They are allocated for the largest batch that will be used; a forward with
    fewer rows simply uses the beginning of each buffer.
    """

    var a1: DeviceBuffer[dtype]
    """[M, hidden] after the first layer's ReLU."""
    var a2: DeviceBuffer[dtype]
    """[M, hidden] after the second one's ReLU."""
    var value: DeviceBuffer[dtype]
    """[M, out_dim] the output, with no ReLU."""

    def __init__(out self, ctx: DeviceContext, max_batch: Int, hidden: Int,
                 out_dim: Int) raises:
        self.a1 = zero_buffer[dtype](ctx, max_batch * hidden)
        self.a2 = zero_buffer[dtype](ctx, max_batch * hidden)
        self.value = zero_buffer[dtype](ctx, max_batch * out_dim)


def critic_forward(ctx: DeviceContext, params: CriticParams,
                   cache: CriticCache, x: DeviceBuffer[dtype], m: Int) raises:
    """V(s) for m inputs. Leaves the activations in `cache`.

    Three layers enqueued on the same stream, so they execute in order without
    needing to synchronise between them (Puzzle 14's lesson).
    """
    linear_forward(ctx, cache.a1, x, params.w1, params.b1,
                   m, params.in_dim, params.hidden)
    relu(ctx, cache.a1, m * params.hidden)

    linear_forward(ctx, cache.a2, cache.a1, params.w2, params.b2,
                   m, params.hidden, params.hidden)
    relu(ctx, cache.a2, m * params.hidden)

    # The last layer has NO ReLU: V is a signed value, and clipping it to 0 would
    # make bad positions unrepresentable.
    linear_forward(ctx, cache.value, cache.a2, params.w3, params.b3,
                   m, params.hidden, params.out_dim)


# ---------------------------------------------------------------------------
# The critic's backward. It chains the linear layer's (which is already verified
# against autodiff) through the two ReLU masks:
#
#     L  ──►  dV  ──►  layer3  ──►  da2  ──►  relu'  ──►  layer2  ──►  da1
#                                                             ──►  relu'  ──►  layer1
#
# The ReLU's gradient is the only new thing here, and it is easy: the derivative
# of max(x,0) is 1 where x was positive and 0 where it was not. Since the ReLU was
# applied in place in the forward, the STORED activation is used to tell
# (post-relu > 0 if and only if pre-relu > 0), without having had to store the
# previous value.
# ---------------------------------------------------------------------------


def value_loss_grad_kernel(dv: GlobalF32, value: GlobalF32, target: GlobalF32,
                           n: Int, scale: Scalar[dtype]):
    """dL/dV for L = mean of 0.5*(V - target)^2, that is (V - target)/n.

    It is the same loss Stoix uses for the critic (`rlax.l2_loss` followed by
    `.mean()`); the 0.5 is there so that the derivative comes out clean, with no 2
    in front. `scale` is 1/n, computed on the host so as not to divide in every
    thread.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n:
        dv[i] = (value[i] - target[i]) * scale


def relu_backward_kernel(grad: GlobalF32, activation: GlobalF32, n: Int):
    """Passes the gradient through the ReLU, in place: grad *= (activation > 0).

    Where the ReLU clipped, the output did not depend on the input, so no gradient
    passes through there. Mind the edge: at exactly 0 the derivative does not
    exist and 0 is taken, which is the usual convention (and matches the forward's
    `<`).
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n:
        if activation[i] <= Scalar[dtype](0):
            grad[i] = Scalar[dtype](0)


struct CriticGrads(Movable):
    """The weights' gradients, with the same shape as `CriticParams`."""

    var dw1: DeviceBuffer[dtype]
    var db1: DeviceBuffer[dtype]
    var dw2: DeviceBuffer[dtype]
    var db2: DeviceBuffer[dtype]
    var dw3: DeviceBuffer[dtype]
    var db3: DeviceBuffer[dtype]

    def __init__(out self, ctx: DeviceContext, in_dim: Int, hidden: Int,
                 out_dim: Int) raises:
        self.dw1 = zero_buffer[dtype](ctx, in_dim * hidden)
        self.db1 = zero_buffer[dtype](ctx, hidden)
        self.dw2 = zero_buffer[dtype](ctx, hidden * hidden)
        self.db2 = zero_buffer[dtype](ctx, hidden)
        self.dw3 = zero_buffer[dtype](ctx, hidden * out_dim)
        self.db3 = zero_buffer[dtype](ctx, out_dim)


struct CriticScratch(Movable):
    """The intermediate gradients that travel backwards between layers.

    They are working buffers, not results: `dvalue` is what comes in from above
    and `da2`/`da1` what goes down. `dx` is computed but nobody uses it (the input
    is not a trainable parameter); it is allocated because the layer's backward
    always produces it and having two different paths would cost more than a
    buffer.
    """

    var dvalue: DeviceBuffer[dtype]
    var da2: DeviceBuffer[dtype]
    var da1: DeviceBuffer[dtype]
    var dx: DeviceBuffer[dtype]

    def __init__(out self, ctx: DeviceContext, max_batch: Int, in_dim: Int,
                 hidden: Int, out_dim: Int) raises:
        self.dvalue = zero_buffer[dtype](ctx, max_batch * out_dim)
        self.da2 = zero_buffer[dtype](ctx, max_batch * hidden)
        self.da1 = zero_buffer[dtype](ctx, max_batch * hidden)
        self.dx = zero_buffer[dtype](ctx, max_batch * in_dim)


def critic_backward(ctx: DeviceContext, params: CriticParams,
                    cache: CriticCache, grads: CriticGrads,
                    scratch: CriticScratch, x: DeviceBuffer[dtype],
                    target: DeviceBuffer[dtype], m: Int) raises:
    """Gradients of the 6 weight tensors for the L2 loss against `target`.

    It assumes `cache` comes from a `critic_forward` with the SAME `x` and the
    same weights: the backward reuses those activations instead of recomputing
    them (Puzzle 22). If it were called with a stale cache, the gradients would
    come out wrong without anything failing.
    """
    n_out = m * params.out_dim

    # 1. Where everything comes in: the derivative of the loss with respect to V.
    ctx.enqueue_function[value_loss_grad_kernel, value_loss_grad_kernel](
        scratch.dvalue.unsafe_ptr(), cache.value.unsafe_ptr(),
        target.unsafe_ptr(), n_out, Scalar[dtype](1) / Scalar[dtype](n_out),
        grid_dim=(n_out + TPB_NET - 1) // TPB_NET, block_dim=TPB_NET)

    # 2..6: the rest does not depend on WHICH loss it is, only on the gradient
    # coming down.
    backward_from_dvalue(ctx, params, cache, grads, scratch, x, m)


def backward_from_dvalue(ctx: DeviceContext, params: CriticParams,
                         cache: CriticCache, grads: CriticGrads,
                         scratch: CriticScratch, x: DeviceBuffer[dtype],
                         m: Int) raises:
    """The network's backward starting from `scratch.dvalue`, already computed.

    It was split off from `critic_backward` when the actor arrived: the network is
    the same and so is the backward path, the only thing that changes is WHERE the
    gradient comes in. The critic comes in with d(L2)/dV = (V - target)/n and the
    actor with dL/dlogits = (pi - q)/n. Everything from here down is identical.

    It still assumes `cache` comes from a forward with the SAME x and the same
    weights.
    """
    n_hidden = m * params.hidden

    # 2. Third layer: its input was a2.
    linear_backward(ctx, grads.dw3, grads.db3, scratch.da2,
                    cache.a2, params.w3, scratch.dvalue,
                    m, params.hidden, params.out_dim)

    # 3. The second layer's ReLU, on the gradient that has just come down.
    ctx.enqueue_function[relu_backward_kernel, relu_backward_kernel](
        scratch.da2.unsafe_ptr(), cache.a2.unsafe_ptr(), n_hidden,
        grid_dim=(n_hidden + TPB_NET - 1) // TPB_NET, block_dim=TPB_NET)

    # 4. Second layer: its input was a1.
    linear_backward(ctx, grads.dw2, grads.db2, scratch.da1,
                    cache.a1, params.w2, scratch.da2,
                    m, params.hidden, params.hidden)

    # 5. The first one's ReLU.
    ctx.enqueue_function[relu_backward_kernel, relu_backward_kernel](
        scratch.da1.unsafe_ptr(), cache.a1.unsafe_ptr(), n_hidden,
        grid_dim=(n_hidden + TPB_NET - 1) // TPB_NET, block_dim=TPB_NET)

    # 6. First layer: its input was the observation.
    linear_backward(ctx, grads.dw1, grads.db1, scratch.dx,
                    x, params.w1, scratch.da1,
                    m, params.in_dim, params.hidden)
