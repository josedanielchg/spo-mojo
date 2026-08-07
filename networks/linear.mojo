"""The linear layer: y = x @ W + b, with Puzzle 16's tiled matmul.

It is the brick of every M-step network (the critic and the actor are stacks of
these). That is why it is worth doing well once: every layer of every forward,
and later the two matmuls of every backward, go through here.

    x  [M, K]   M = how many boards at a time (particles or envs)
                K = the layer's inputs
    W  [K, N]   N = the layer's outputs
    b  [N]
    y  [M, N]

Why tiled and not the naive version: in the naive one each thread reads its row of
x and its column of W from global memory, so each element of x gets re-read N
times across the whole grid. By tiling, a block loads its chunk of x and of W into
shared memory ONCE and reuses them TILE times. It is the same change Puzzle 16
measures as going from ~1.5% of peak to something reasonable: it does not change
the arithmetic, it changes how many times memory is crossed.

The guards are not decoration: our real dimensions (18 inputs, a batch of
particles) are almost never multiples of the tile, so the edges are always
straddled. Zeros are loaded out of range so that the accumulator does not see
garbage.
"""

from std.gpu import barrier, block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import stack_allocation, AddressSpace

from ops.common import dtype, GlobalF32

comptime TILE = 16
"""Side of the square tile. 16x16 = 256 threads per block, which is a comfortable
size and leaves each shared-memory tile at 1 KB (16*16*4 bytes)."""


def linear_forward_kernel[T: Int](y: GlobalF32, x: GlobalF32, w: GlobalF32,
                                  bias: GlobalF32, m: Int, k: Int, n: Int):
    """y = x @ W + b. One thread per output element, blocks of TxT.

    The loop walks the K dimension in strides of T. On each turn the whole block
    cooperates to load a tile of x and one of W into shared memory, and then each
    thread accumulates the partial product of its row by its column.

    Both `barrier()` calls are OUTSIDE any conditional, which is Puzzle 9's rule:
    if one thread does not reach a barrier the others do reach, the block hangs
    forever. The first waits for the tile to be loaded before using it; the second
    waits for everyone to have finished using it before overwriting it on the next
    turn. Removing the second one is the classic bug: it works almost always and
    fails once in a while.
    """
    x_tile = stack_allocation[T * T, Scalar[dtype],
                              address_space = AddressSpace.SHARED]()
    w_tile = stack_allocation[T * T, Scalar[dtype],
                              address_space = AddressSpace.SHARED]()

    ty = Int(thread_idx.y)
    tx = Int(thread_idx.x)
    row = Int(block_idx.y) * T + ty     # which row of the output
    col = Int(block_idx.x) * T + tx     # which column of the output

    acc = Scalar[dtype](0)
    num_tiles = (k + T - 1) // T

    for t in range(num_tiles):
        # Load x's tile: row `row`, columns [t*T, t*T+T).
        xc = t * T + tx
        x_tile[ty * T + tx] = x[row * k + xc] if (row < m and xc < k) \
                              else Scalar[dtype](0)
        # And W's: rows [t*T, t*T+T), column `col`.
        wr = t * T + ty
        w_tile[ty * T + tx] = w[wr * n + col] if (wr < k and col < n) \
                              else Scalar[dtype](0)
        barrier()

        # The partial product. The padding zeros contribute nothing, so there is
        # no need to shorten the loop at the edges.
        for i in range(T):
            acc += x_tile[ty * T + i] * w_tile[i * T + tx]
        barrier()

    if row < m and col < n:
        y[row * n + col] = acc + bias[col]


def linear_forward(ctx: DeviceContext, y: DeviceBuffer[dtype],
                   x: DeviceBuffer[dtype], w: DeviceBuffer[dtype],
                   bias: DeviceBuffer[dtype], m: Int, k: Int, n: Int) raises:
    """Enqueues the layer: one TILExTILE block for each tile of the output."""
    grid_x = (n + TILE - 1) // TILE
    grid_y = (m + TILE - 1) // TILE
    ctx.enqueue_function[linear_forward_kernel[TILE],
                         linear_forward_kernel[TILE]](
        y.unsafe_ptr(), x.unsafe_ptr(), w.unsafe_ptr(), bias.unsafe_ptr(),
        m, k, n, grid_dim=(grid_x, grid_y), block_dim=(TILE, TILE))


# ---------------------------------------------------------------------------
# The backward. With no autodiff in Mojo, the three derivatives are written by
# hand.
#
# Starting from  y = x @ W + b  and from dy = dL/dy (what arrives from the layer
# above), the chain rule gives exactly three things:
#
#     dL/dW[k,n] = sum_m  x[m,k] * dy[m,n]        that is  dW = x^T @ dy   [K,N]
#     dL/db[n]   = sum_m  dy[m,n]                 the column-wise sum      [N]
#     dL/dx[m,k] = sum_n  dy[m,n] * W[k,n]        that is  dx = dy @ W^T   [M,K]
#
# The two transposes are NOT materialised: they are applied in the indexing
# (reading `x[m*k + kk]` while walking m is reading a column of x, which is a row
# of x^T). Actually transposing would cost two more buffers and two more kernels,
# for nothing.
#
# They go without tiling on purpose: correctness first. Each thread does a loop
# over the reduced dimension, which here is small (batch or number of neurons). If
# some day profiling says it hurts, they get tiled just like the forward.
# ---------------------------------------------------------------------------


def linear_grad_w_kernel(dw: GlobalF32, x: GlobalF32, dy: GlobalF32,
                         m: Int, k: Int, n: Int):
    """dW = x^T @ dy. One thread per weight (k, n); the loop walks the batch."""
    kk = Int(block_dim.y * block_idx.y + thread_idx.y)
    nn = Int(block_dim.x * block_idx.x + thread_idx.x)
    if kk >= k or nn >= n:
        return

    acc = Scalar[dtype](0)
    for mm in range(m):
        acc += x[mm * k + kk] * dy[mm * n + nn]
    dw[kk * n + nn] = acc


def linear_grad_b_kernel(db: GlobalF32, dy: GlobalF32, m: Int, n: Int):
    """db = column-wise sum of dy. One thread per output.

    The bias goes in added to ALL the batch's rows, so its gradient is the sum of
    what arrives from each row. It is also the check that the bias was added only
    once in the forward: had it been added twice, a factor of 2 would be missing
    here and finite differences would catch it.
    """
    nn = Int(block_dim.x * block_idx.x + thread_idx.x)
    if nn >= n:
        return

    acc = Scalar[dtype](0)
    for mm in range(m):
        acc += dy[mm * n + nn]
    db[nn] = acc


def linear_grad_x_kernel(dx: GlobalF32, dy: GlobalF32, w: GlobalF32,
                         m: Int, k: Int, n: Int):
    """dx = dy @ W^T. One thread per input (m, k); the loop walks the outputs.

    It is the gradient handed back to the previous layer, so a mistake here does
    not break this layer but all the ones below it.
    """
    mm = Int(block_dim.y * block_idx.y + thread_idx.y)
    kk = Int(block_dim.x * block_idx.x + thread_idx.x)
    if mm >= m or kk >= k:
        return

    acc = Scalar[dtype](0)
    for nn in range(n):
        acc += dy[mm * n + nn] * w[kk * n + nn]
    dx[mm * k + kk] = acc


def linear_backward(ctx: DeviceContext, dw: DeviceBuffer[dtype],
                    db: DeviceBuffer[dtype], dx: DeviceBuffer[dtype],
                    x: DeviceBuffer[dtype], w: DeviceBuffer[dtype],
                    dy: DeviceBuffer[dtype], m: Int, k: Int, n: Int) raises:
    """The layer's three gradients starting from dy.

    `dx` could conceptually be skipped in the first layer (nobody uses it), but it
    is computed all the same: it costs little and avoids having two different
    paths.
    """
    ctx.enqueue_function[linear_grad_w_kernel, linear_grad_w_kernel](
        dw.unsafe_ptr(), x.unsafe_ptr(), dy.unsafe_ptr(), m, k, n,
        grid_dim=((n + TILE - 1) // TILE, (k + TILE - 1) // TILE),
        block_dim=(TILE, TILE))

    ctx.enqueue_function[linear_grad_b_kernel, linear_grad_b_kernel](
        db.unsafe_ptr(), dy.unsafe_ptr(), m, n,
        grid_dim=(n + TILE * TILE - 1) // (TILE * TILE),
        block_dim=TILE * TILE)

    ctx.enqueue_function[linear_grad_x_kernel, linear_grad_x_kernel](
        dx.unsafe_ptr(), dy.unsafe_ptr(), w.unsafe_ptr(), m, k, n,
        grid_dim=((k + TILE - 1) // TILE, (m + TILE - 1) // TILE),
        block_dim=(TILE, TILE))
