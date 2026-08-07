"""Stable softmax and row-wise logsumexp.

The trick is subtracting the row's maximum before exponentiating. exp(1000) is inf
and inf/inf is nan, but exp(1000 - 1000) = 1. Mathematically it makes no
difference, because the factor exp(-m) appears in both the numerator and the
denominator and cancels:

    exp(x_i - m) / SUM_j exp(x_j - m)  ==  exp(x_i) / SUM_j exp(x_j)

Numerically it is the difference between working and returning nan. The test has
one row of +1000 and another of -1000 precisely to watch for this.

In SPO it is used in two places: the actor's categorical head, and the conversion
of the SMC weights w/eta into resampling probabilities.

Layout: one block per row (block = env, thread = particle).
TPB has to be a power of two. row_size may be larger than TPB.
"""

from std.builtin.debug_assert import debug_assert
from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.math import exp, log
from std.memory import stack_allocation, AddressSpace

from ops.common import dtype, NEG_INF, SharedF32, GlobalF32
from ops.reductions import block_reduce_sum, block_reduce_max


def _row_max[TPB: Int](shared: SharedF32, a_ptr: GlobalF32,
                       base: Int, row_size: Int, tid: Int) -> Scalar[dtype]:
    """Row maximum, returned to every thread. Leaves shared reusable."""
    acc = NEG_INF
    i = tid
    while i < row_size:
        v = a_ptr[base + i]
        if v > acc:
            acc = v
        i += TPB
    shared[tid] = acc
    barrier()
    return block_reduce_max[TPB](shared, tid)


def _row_sum_exp[TPB: Int](shared: SharedF32, a_ptr: GlobalF32,
                           base: Int, row_size: Int, tid: Int,
                           row_max: Scalar[dtype]) -> Scalar[dtype]:
    """SUM_i exp(a[i] - row_max), returned to every thread."""
    acc = Scalar[dtype](0)
    i = tid
    while i < row_size:
        acc += exp(a_ptr[base + i] - row_max)
        i += TPB
    shared[tid] = acc
    barrier()
    return block_reduce_sum[TPB](shared, tid)


def softmax_rows[TPB: Int](out_ptr: GlobalF32, a_ptr: GlobalF32, row_size: Int):
    """out = softmax(a) row by row."""
    debug_assert(Int(block_dim.x) == TPB, "softmax_rows: block_dim tiene que ser TPB")

    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    row = Int(block_idx.x)
    base = row * row_size

    row_max = _row_max[TPB](shared, a_ptr, base, row_size, tid)
    row_sum = _row_sum_exp[TPB](shared, a_ptr, base, row_size, tid, row_max)

    # The exp is recomputed instead of being kept from the previous pass. Keeping
    # it would call for another shared array of size row_size, and row_size may be
    # larger than TPB (which is exactly what shared measures). I would rather pay
    # for the exp twice than introduce an artificial limit; if it ever hurts, it
    # gets profiled first.
    i = tid
    while i < row_size:
        out_ptr[base + i] = exp(a_ptr[base + i] - row_max) / row_sum
        i += TPB


def logsumexp_rows[TPB: Int](out_ptr: GlobalF32, a_ptr: GlobalF32, row_size: Int):
    """out[row] = log(SUM exp(a)), one scalar per row.

    Identity used: log(SUM exp(x)) = m + log(SUM exp(x - m)).
    In SPO it shows up in the M-step's temperature loss (equation 7 of the paper).
    """
    debug_assert(Int(block_dim.x) == TPB, "logsumexp_rows: block_dim tiene que ser TPB")

    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    row = Int(block_idx.x)
    base = row * row_size

    row_max = _row_max[TPB](shared, a_ptr, base, row_size, tid)
    row_sum = _row_sum_exp[TPB](shared, a_ptr, base, row_size, tid, row_max)

    if tid == 0:
        out_ptr[row] = row_max + log(row_sum)


def log_softmax_rows[TPB: Int](out_ptr: GlobalF32, a_ptr: GlobalF32,
                               row_size: Int):
    """out = log(softmax(a)) row by row, computed without going through the softmax.

    Identity: log softmax(a)[i] = a[i] - (m + log(SUM exp(a - m))), with m the
    row's maximum. It is the same denominator as `logsumexp_rows`.

    Why not just `log(softmax(a))`: the softmax of a very low logit underflows to
    0 and its log would be -inf even though the exact value was perfectly
    representable (-40, say). Computing it directly preserves those values, which
    is precisely what a cross entropy needs: the terms with small probability are
    the ones that weigh most in the log.

    On MASKED cells the input is NEG_INF (the most negative finite float32) and
    the output stays there: subtracting a denominator of order 1 does not move it,
    because at that magnitude the ULP is ~1e31. What comes out is a finite, very
    negative value, not a -inf, which is what we want in order not to propagate
    NaN.
    """
    debug_assert(Int(block_dim.x) == TPB,
                 "log_softmax_rows: block_dim tiene que ser TPB")

    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    row = Int(block_idx.x)
    base = row * row_size

    row_max = _row_max[TPB](shared, a_ptr, base, row_size, tid)
    row_sum = _row_sum_exp[TPB](shared, a_ptr, base, row_size, tid, row_max)
    log_denom = row_max + log(row_sum)

    i = tid
    while i < row_size:
        out_ptr[base + i] = a_ptr[base + i] - log_denom
        i += TPB
