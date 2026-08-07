"""Types and constants shared by every kernel.

They were repeated in each ops/ module and ended up drifting out of sync, so I put
them together here. Switching the project to float16 should be a one-line change.
"""

from std.memory import AddressSpace

# We use float32 throughout the project: the laptop GPU has no tensor cores worth
# using for float64 and for CartPole it is far more than enough.
comptime dtype = DType.float32

# Indices (chosen actions, particles selected during resampling).
# int32 rather than int64 because we are never going to have more than 2^31
# particles (on my machine at least) and this way it takes half the space when
# copying from device to host.
comptime idx_dtype = DType.int32

# Neutral element of the max: any real value beats it.
# I use MIN_FINITE and not MIN (which is -inf) on purpose: with -inf, a whole row
# of -inf would give row_max = -inf and then exp(-inf - -inf) = exp(nan) = nan.
# With MIN_FINITE the degenerate case gives 0 instead of propagating a nan.
comptime NEG_INF = Scalar[dtype].MIN_FINITE

# Pointers to shared memory. The full type is long to write and appears in the
# signature of every helper, so I give it a name.
comptime SharedF32 = UnsafePointer[Scalar[dtype], MutAnyOrigin,
                                   address_space = AddressSpace.SHARED]

# Pointers to global memory (the buffers that come from DeviceContext).
comptime GlobalF32 = UnsafePointer[Scalar[dtype], MutAnyOrigin]
comptime GlobalI32 = UnsafePointer[Scalar[idx_dtype], MutAnyOrigin]
