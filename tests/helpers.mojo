"""Utilidades para los tests: subir/bajar buffers y comparar contra lo esperado.

Sin esto cada test repetia seis lineas de create_buffer + fill + map_to_host
antes de llegar a lo que de verdad estaba probando, y lo importante se perdia
entre el andamiaje.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs

from ops.common import dtype


def upload[dt: DType](ctx: DeviceContext,
                      data: List[Scalar[dt]]) raises -> DeviceBuffer[dt]:
    """Buffer en device con el contenido de `data`."""
    buf = ctx.enqueue_create_buffer[dt](len(data))
    buf.enqueue_fill(0)
    with buf.map_to_host() as h:
        for i in range(len(data)):
            h[i] = data[i]
    return buf^


def zeros[dt: DType](ctx: DeviceContext, n: Int) raises -> DeviceBuffer[dt]:
    """Buffer de salida a cero."""
    buf = ctx.enqueue_create_buffer[dt](n)
    buf.enqueue_fill(0)
    return buf^


def filled[dt: DType](ctx: DeviceContext, n: Int,
                      value: Scalar[dt]) raises -> DeviceBuffer[dt]:
    """Buffer relleno con un valor. Util para inicializar salidas a -1 y detectar
    que el kernel no escribio nada."""
    buf = ctx.enqueue_create_buffer[dt](n)
    buf.enqueue_fill(value)
    return buf^


def download[dt: DType](buf: DeviceBuffer[dt], n: Int) raises -> List[Scalar[dt]]:
    """Copia n elementos del device al host."""
    out = List[Scalar[dt]]()
    with buf.map_to_host() as h:
        for i in range(n):
            out.append(h[i])
    return out^


def assert_close(got: Scalar[dtype], want: Scalar[dtype], tol: Scalar[dtype],
                 what: String) raises:
    """Comparacion absoluta. `what` tiene que decir QUE fallo y DONDE, porque es
    lo unico que se ve cuando el test revienta."""
    if abs(got - want) > tol:
        raise Error(what, ": got=", got, " want=", want,
                    " (diff=", abs(got - want), ", tol=", tol, ")")


def assert_eq_int(got: Int, want: Int, what: String) raises:
    if got != want:
        raise Error(what, ": got=", got, " want=", want)
