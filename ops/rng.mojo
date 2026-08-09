"""Counter-based RNG and categorical sampling by inverse CDF (Cumulative distribution function).

No global state: the value comes out of hash(seed, stream, counter), like JAX's
keys. Two advantages for SPO:

  - Reproducible whatever the scheduler does. Thread i uses counter = i, so there
    is no shared sequence whose result depends on the order in which the threads
    happen to run. With a stateful RNG, two runs with the same seed would hand
    different numbers to each thread.
  - It "splits" into independent streams per use (one for the root action, another
    for resampling, another for the environment reset) without them colliding.

The mixer is the lowbias32 finaliser: two rounds of xor-shift + multiplication. It
is not cryptographic and does not need to be; it only has to pass the sampling's
statistical tests (mean, variance and chi2 are in test_rng.mojo).

Important design note: the kernels that sample receive the uniforms ALREADY
generated as input, instead of calling the RNG internally. That way the test can
inject uniforms by hand and check EXACT indices, without relying on statistical
tolerances.
"""

from std.builtin.debug_assert import debug_assert
from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.math import exp
from std.memory import stack_allocation, AddressSpace

from ops.common import dtype, idx_dtype, NEG_INF, GlobalF32, GlobalI32
from ops.reductions import block_reduce_max
from ops.scan import block_scan_inclusive

# 1 / 2^24. I keep the top 24 bits because they are exactly the ones that fit in a
# float32's mantissa: that way every representable value in [0,1) comes out with
# the same probability. If I used all 32 bits, the conversion to float would round
# and some values would come out twice as often as others.
comptime INV_2P24 = Scalar[dtype](5.9604645e-8)

# Golden ratio in 32 bits. It is the customary scattering constant (the same one
# Fibonacci hashing uses): it mixes high and low bits well.
comptime GOLDEN32 = UInt32(0x9E3779B9)


def mix32(x_in: UInt32) -> UInt32:
    """lowbias32 finaliser: mixes the 32 bits so that a counter 0,1,2,... turns
    into something that looks random."""
    x = x_in
    x ^= x >> 16
    x *= 0x7FEB352D
    x ^= x >> 15
    x *= 0x846CA68B
    x ^= x >> 16
    return x


def rand_bits(seed: UInt32, stream: UInt32, counter: UInt32) -> UInt32:
    """The three are mixed in cascade so that changing any one of them changes the
    whole result, not just a few bits."""
    return mix32(seed ^ mix32(stream ^ mix32(counter)))


def rand_uniform(seed: UInt32, stream: UInt32, counter: UInt32) -> Scalar[dtype]:
    """Uniform in [0, 1). The 1.0 never comes out, which is what the inverse CDF wants."""
    return Scalar[dtype]((rand_bits(seed, stream, counter) >> 8)) * INV_2P24


@fieldwise_init
struct RngKey(Copyable, Movable):
    """JAX-key-style stream bookkeeping, for the host side.

    The kernels receive (seed, stream) separately rather than an RngKey: passing
    structs as a kernel argument requires them to be DevicePassable and adds
    nothing here. This is the ledger of who uses which stream.
    """

    var seed: UInt32
    var stream: UInt32

    def split(self, which: UInt32) -> RngKey:
        """Derives a child stream: same seed, a different and independent stream."""
        return RngKey(self.seed, mix32(self.stream ^ (which + GOLDEN32)))


def fill_uniform(out_ptr: GlobalF32, seed: UInt32, stream: UInt32, n: Int):
    """Fills out[0..n) with uniforms. One thread per value, with its guard."""
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n:
        # counter = i (the global index), not a shared counter: that is where the reproducibility comes from.
        out_ptr[i] = rand_uniform(seed, stream, UInt32(i))


def categorical_from_logits[TPB: Int](out_ptr: GlobalI32, logits_ptr: GlobalF32,
                                      u_ptr: GlobalF32, row_size: Int):
    """One categorical sample per row, by inverse CDF.

    Recipe: stable softmax -> prefix sum (= CDF) -> first index whose stretch
    contains the uniform. It is the composition of phase 2's three pieces.

    A detail that saves work: the CDF is left UNNORMALISED and instead of dividing
    each element by the total, the uniform is scaled (u * total). That gives one
    division per block instead of one per thread, and the total == 0 case
    disappears along the way.

    Requires row_size <= TPB. Launch with grid_dim=num_rows, block_dim=TPB.
    """
    debug_assert(row_size <= TPB, "categorical: row_size must fit in TPB")
    debug_assert(Int(block_dim.x) == TPB, "categorical: block_dim must be TPB")

    shared = stack_allocation[TPB, Scalar[dtype], address_space = AddressSpace.SHARED]()
    tid = Int(thread_idx.x)
    row = Int(block_idx.x)
    base = row * row_size

    # First the row's maximum, which is what makes the softmax stable.
    shared[tid] = logits_ptr[base + tid] if tid < row_size else NEG_INF
    barrier()
    row_max = block_reduce_max[TPB](shared, tid)

    # Now exp(x - max), unnormalised. The leftover threads put 0 so as not to
    # contribute probability mass.
    shared[tid] = exp(logits_ptr[base + tid] - row_max) if tid < row_size else Scalar[dtype](0)
    barrier()

    # The prefix sum turns that into a CDF, also unnormalised.
    block_scan_inclusive[TPB](shared, tid)

    total = shared[row_size - 1]
    target = u_ptr[row] * total

    # And each thread checks whether the target falls in its stretch
    # [cdf_prev, cdf). Exactly one hits, so there is no race writing out[row]. An
    # element with probability 0 has cdf_prev == cdf, that is, an empty stretch,
    # and therefore cannot be chosen, which is exactly what we want.
    if tid < row_size:
        cdf = shared[tid]
        cdf_prev = shared[tid - 1] if tid > 0 else Scalar[dtype](0)
        hit = cdf_prev <= target and target < cdf

        # Safety net against rounding: the CDF accumulates by adding floats, so
        # the last stretch can end up one ulp below total and leave the target
        # outside every stretch. Without this the row would write nothing and
        # out[row] would be left holding garbage.
        if tid == row_size - 1 and target >= cdf:
            hit = True

        if hit:
            out_ptr[row] = Scalar[idx_dtype](tid)
