"""Row-wise reductions: sum, max and argmax.

One block handles one row and each thread one element. That layout is the one SPO
ends up using, where the row is an environment and each thread a particle.

There are two levels here. On one hand the primitives (`block_reduce_sum`,
`block_reduce_max`), which reduce an array that is already in shared memory;
softmax.mojo and rng.mojo reuse them, since they need to reduce halfway through
something else. On the other hand the kernels (`sum_rows`, `max_rows`,
`argmax_rows`), which load the row from global memory and call the primitive.

If the row is longer than the block, each thread accumulates in strides of TPB
before reducing, so row_size can be anything.

TPB has to be a power of two: the tree keeps halving the range and if it is not,
a chunk is left uncombined.
"""

from std.builtin.debug_assert import debug_assert
from std.gpu import block_dim, block_idx, thread_idx, barrier, WARP_SIZE
from std.gpu.primitives.warp import shuffle_xor
from std.memory import stack_allocation, AddressSpace

from ops.common import dtype, idx_dtype, NEG_INF, SharedF32, GlobalF32, GlobalI32


# The two primitives below return the result to every thread and not only to
# thread 0, because the softmax needs everyone to know the row's maximum. They
# expect shared to arrive already filled in and with its barrier done, and they
# leave it clobbered but reusable: they exit with another barrier so that nobody
# overwrites it while a neighbour is still reading it.

def block_reduce_sum[TPB: Int](shared: SharedF32, tid: Int) -> Scalar[dtype]:
    stride = TPB // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        barrier()
        stride //= 2

    total = shared[0]
    barrier()
    return total


def block_reduce_max[TPB: Int](shared: SharedF32, tid: Int) -> Scalar[dtype]:
    stride = TPB // 2
    while stride > 0:
        if tid < stride:
            other = shared[tid + stride]
            if other > shared[tid]:
                shared[tid] = other
        barrier()
        stride //= 2

    biggest = shared[0]
    barrier()
    return biggest


def sum_rows[TPB: Int](out_ptr: GlobalF32, a_ptr: GlobalF32, row_size: Int):
    """out[row] = sum of the row. Launch with grid_dim=num_rows, block_dim=TPB."""
    # shared is sized with TPB, so launching with more threads would write outside.
    debug_assert(Int(block_dim.x) == TPB, "sum_rows: block_dim tiene que ser TPB")

    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    row = Int(block_idx.x)
    base = row * row_size

    # Step 1: each thread eats its share of the row (in strides of TPB).
    acc = Scalar[dtype](0)
    i = tid
    while i < row_size:
        acc += a_ptr[base + i]
        i += TPB
    shared[tid] = acc
    barrier()

    # Step 2: the TPB partials are combined in a tree.
    total = block_reduce_sum[TPB](shared, tid)

    if tid == 0:
        out_ptr[row] = total


def max_rows[TPB: Int](out_ptr: GlobalF32, a_ptr: GlobalF32, row_size: Int):
    """out[row] = maximum of the row."""
    debug_assert(Int(block_dim.x) == TPB, "max_rows: block_dim tiene que ser TPB")

    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    row = Int(block_idx.x)
    base = row * row_size

    # The threads that reach no element at all (row_size < TPB) stay at NEG_INF,
    # which loses against any real value.
    acc = NEG_INF
    i = tid
    while i < row_size:
        v = a_ptr[base + i]
        if v > acc:
            acc = v
        i += TPB
    shared[tid] = acc
    barrier()

    biggest = block_reduce_max[TPB](shared, tid)

    if tid == 0:
        out_ptr[row] = biggest


def argmax_rows[TPB: Int](out_ptr: GlobalI32, a_ptr: GlobalF32, row_size: Int):
    """out[row] = index of the maximum. On ties the LOWER index wins.

    The tie-breaking rule is not a whim: in a tree reduction the order in which
    elements get combined depends on TPB, so without an explicit rule the result
    would change when the block size changed. Exactly the kind of non-determinism
    that makes a test fail one day in ten.

    It does not use block_reduce_max because the index has to be carried along
    with the value.
    """
    debug_assert(Int(block_dim.x) == TPB, "argmax_rows: block_dim tiene que ser TPB")

    shared_val = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    shared_idx = stack_allocation[TPB, Scalar[idx_dtype], address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    row = Int(block_idx.x)
    base = row * row_size

    best = NEG_INF
    best_i = Scalar[idx_dtype](-1)
    i = tid
    while i < row_size:
        v = a_ptr[base + i]
        # Strict comparison: on a tie the one already held stays, and it comes
        # from a lower i because the loop counts up.
        if v > best:
            best = v
            best_i = Scalar[idx_dtype](i)
        i += TPB
    shared_val[tid] = best
    shared_idx[tid] = best_i
    barrier()

    stride = TPB // 2
    while stride > 0:
        if tid < stride:
            other = shared_val[tid + stride]
            mine = shared_val[tid]
            # I give way if the other is strictly greater; on a tie the lower
            # index wins. A thread with no data carries NEG_INF and never wins
            # (its -1 cannot sneak through). Since each half already satisfies
            # "the lowest index among those tied at the maximum", combining them
            # this way preserves it.
            take = other > mine
            if other == mine:
                take = shared_idx[tid + stride] < shared_idx[tid]
            if take:
                shared_val[tid] = other
                shared_idx[tid] = shared_idx[tid + stride]
        barrier()
        stride //= 2

    if tid == 0:
        out_ptr[row] = shared_idx[0]


def warp_sum_rows(out_ptr: GlobalF32, a_ptr: GlobalF32, row_size: Int):
    """Row-wise sum using warp shuffles, with no shared memory and no barriers.

    Inside a warp the 32 lanes run in lockstep and the value travels through
    registers, so synchronising is unnecessary. It comes out considerably shorter
    than the tree version, at the price of requiring row_size <= WARP_SIZE. For
    SPO that restriction does not get in the way, since it is 16 particles, half a
    warp.

    Launch with block_dim=WARP_SIZE.
    """
    debug_assert(row_size <= Int(WARP_SIZE),
                 "warp_sum_rows: la fila tiene que caber en un warp")
    debug_assert(Int(block_dim.x) == Int(WARP_SIZE),
                 "warp_sum_rows: block_dim tiene que ser WARP_SIZE")

    tid = Int(thread_idx.x)
    row = Int(block_idx.x)
    base = row * row_size

    # The leftover lanes contribute 0, which is the sum's neutral element.
    v = a_ptr[base + tid] if tid < row_size else Scalar[dtype](0)

    # shuffle_xor(v, offset) brings the value from lane tid^offset. After
    # log2(WARP_SIZE) rounds every lane holds the total sum, not only lane 0: the
    # exchange is symmetric and does not collapse towards anywhere.
    offset = Int(WARP_SIZE) // 2
    while offset > 0:
        v += shuffle_xor(v, UInt32(offset))
        offset //= 2

    if tid == 0:
        out_ptr[row] = v
