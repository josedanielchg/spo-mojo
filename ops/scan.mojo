"""Row-wise prefix sum, inclusive and exclusive.

In SPO we use the inclusive one to build the resampling CDF. The CDF splits the
particles' weights into intervals. We draw a random number and pick the particle
whose interval contains that number.

Each GPU block handles one row and each thread one position. The whole row has to
fit inside the block, which is why row_size cannot be larger than TPB. In SPO each
row holds one environment's particles, usually 16, so they fit inside a block of
32 threads or more.

First comes block_scan_inclusive, which works directly on a block's shared memory.
Then come inclusive_scan_rows and exclusive_scan_rows, which load the rows, call
the primitive and write the results.
"""

from std.builtin.debug_assert import debug_assert
from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.memory import stack_allocation, AddressSpace

from ops.common import dtype, SharedF32, GlobalF32


def block_scan_inclusive[TPB: Int](shared: SharedF32, tid: Int):
    """In-place inclusive prefix sum over shared[0..TPB).

    Before: shared filled in and with barrier() done.
    After : shared[i] = sum of shared[0..i], and barrier() done.
    """
    offset = 1
    while offset < TPB:
        # Without the barrier in the middle the bug is one of the worst kind: it
        # does not blow up, it just gives a different result depending on how the
        # scheduler orders the warps.
        val = shared[tid - offset] if tid >= offset else Scalar[dtype](0)
        barrier()
        shared[tid] += val
        barrier()
        offset *= 2


def inclusive_scan_rows[TPB: Int](out_ptr: GlobalF32, a_ptr: GlobalF32, row_size: Int):
    """out[i] = a[0] + ... + a[i], row by row."""
    debug_assert(row_size <= TPB, "inclusive_scan_rows: row_size must fit in TPB")
    debug_assert(Int(block_dim.x) == TPB, "inclusive_scan_rows: block_dim must be TPB")

    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    base = Int(block_idx.x) * row_size

    # The leftover threads load 0 (the sum's neutral element) so that the whole
    # block's scan comes out right with no special cases.
    shared[tid] = a_ptr[base + tid] if tid < row_size else Scalar[dtype](0)
    barrier()

    block_scan_inclusive[TPB](shared, tid)

    if tid < row_size:
        out_ptr[base + tid] = shared[tid]


def exclusive_scan_rows[TPB: Int](out_ptr: GlobalF32, a_ptr: GlobalF32, row_size: Int):
    """out[0] = 0, out[i] = a[0] + ... + a[i-1].

    It is the inclusive one shifted one place to the right. The typical use is
    turning counts into write offsets: each bucket knows where its stretch begins.
    """
    debug_assert(row_size <= TPB, "exclusive_scan_rows: row_size must fit in TPB")
    debug_assert(Int(block_dim.x) == TPB, "exclusive_scan_rows: block_dim must be TPB")

    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    base = Int(block_idx.x) * row_size

    shared[tid] = a_ptr[base + tid] if tid < row_size else Scalar[dtype](0)
    barrier()

    block_scan_inclusive[TPB](shared, tid)

    if tid < row_size:
        out_ptr[base + tid] = shared[tid - 1] if tid > 0 else Scalar[dtype](0)
