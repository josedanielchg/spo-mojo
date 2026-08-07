"""Flat device-to-device copy, parameterised by dtype.

There used to be two identical kernels, `copy_actions_kernel` (int32) and
`copy_f32_kernel` (float32), because the pointer's type is part of the signature.
Parameterising by dtype makes it a single kernel: `copy_kernel[idx_dtype]` for
actions and `copy_kernel[dtype]` for values.

The search needs it more than it looks, because some fields live in two places at
once: `root_actions` keeps the depth-0 action forever while `next_action` is
overwritten at every depth, and the public output takes its own copy of the gae
and of the root actions.
"""

from std.gpu import block_dim, block_idx, thread_idx


def copy_kernel[dt: DType](dst: UnsafePointer[Scalar[dt], MutAnyOrigin],
                           src: UnsafePointer[Scalar[dt], MutAnyOrigin],
                           n: Int):
    """dst[i] = src[i]. One thread per element, with its guard."""
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < n:
        dst[i] = src[i]
