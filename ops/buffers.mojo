"""Host-side helpers for creating device buffers.

It exists for a boring but real reason: the pair

    buf = ctx.enqueue_create_buffer[dt](n)
    buf.enqueue_fill(0)

appeared about 25 times in a row in the constructors in particles.mojo, and
reading fifty lines of that to find out there are eight fields is wasted time.
With `zero_buffer` each field is one line and the structure is visible at a
glance.

The two lines are kept separate on purpose: `enqueue_fill` returns None, so
chaining `create_buffer(...).enqueue_fill(0)` does not compile (see
docs/api_notes.md).
"""

from std.gpu.host import DeviceContext, DeviceBuffer


def zero_buffer[dt: DType](ctx: DeviceContext, n: Int) raises -> DeviceBuffer[dt]:
    """A buffer of n elements set to zero."""
    buf = ctx.enqueue_create_buffer[dt](n)
    buf.enqueue_fill(0)
    return buf^


def filled_buffer[dt: DType](ctx: DeviceContext, n: Int,
                             value: Scalar[dt]) raises -> DeviceBuffer[dt]:
    """A buffer filled with a value. Useful for initialising an output to -1 so
    that "the kernel wrote -1" can be told from "the kernel wrote nothing"."""
    buf = ctx.enqueue_create_buffer[dt](n)
    buf.enqueue_fill(value)
    return buf^
