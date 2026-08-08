"""Test utilities: uploading/downloading buffers and comparing against expectations.

Without these, every test repeated six lines of create_buffer + fill + map_to_host
before getting to what it was actually testing, and the point got lost among the
scaffolding.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs

from ops.common import dtype


def upload[dt: DType](ctx: DeviceContext,
                      data: List[Scalar[dt]]) raises -> DeviceBuffer[dt]:
    """A device buffer holding `data`'s contents."""
    buf = ctx.enqueue_create_buffer[dt](len(data))
    buf.enqueue_fill(0)
    with buf.map_to_host() as h:
        for i in range(len(data)):
            h[i] = data[i]
    return buf^


def zeros[dt: DType](ctx: DeviceContext, n: Int) raises -> DeviceBuffer[dt]:
    """An output buffer set to zero."""
    buf = ctx.enqueue_create_buffer[dt](n)
    buf.enqueue_fill(0)
    return buf^


def filled[dt: DType](ctx: DeviceContext, n: Int,
                      value: Scalar[dt]) raises -> DeviceBuffer[dt]:
    """A buffer filled with a value. Useful for initialising outputs to -1 and
    detecting that the kernel wrote nothing."""
    buf = ctx.enqueue_create_buffer[dt](n)
    buf.enqueue_fill(value)
    return buf^


def write_into[dt: DType](buf: DeviceBuffer[dt],
                          values: List[Scalar[dt]]) raises:
    """Writes into a buffer that ALREADY exists, instead of creating a new one as
    upload() does.

    It is needed when the buffer is a struct's field (Particles, StepOutputs) and
    what we want is to dictate a starting state to the test."""
    with buf.map_to_host() as h:
        for i in range(len(values)):
            h[i] = values[i]


def download[dt: DType](buf: DeviceBuffer[dt], n: Int) raises -> List[Scalar[dt]]:
    """Copies n elements from device to host."""
    out = List[Scalar[dt]]()
    with buf.map_to_host() as h:
        for i in range(n):
            out.append(h[i])
    return out^


def assert_close(got: Scalar[dtype], want: Scalar[dtype], tol: Scalar[dtype],
                 what: String) raises:
    """Absolute comparison. `what` has to say WHAT failed and WHERE, because it is
    the only thing visible when the test blows up."""
    if abs(got - want) > tol:
        raise Error(what, ": got=", got, " want=", want,
                    " (diff=", abs(got - want), ", tol=", tol, ")")


def assert_eq_int(got: Int, want: Int, what: String) raises:
    if got != want:
        raise Error(what, ": got=", got, " want=", want)
