"""Helpers de host para crear buffers de device.

Existe por una razon aburrida pero real: el par

    buf = ctx.enqueue_create_buffer[dt](n)
    buf.enqueue_fill(0)

aparecia unas 25 veces seguidas en los constructores de particles.mojo, y leer
cincuenta lineas de eso para enterarte de que hay ocho campos es tiempo perdido.
Con `zero_buffer` cada campo es una linea y se ve la estructura de un vistazo.

Van separadas las dos lineas a proposito: `enqueue_fill` devuelve None, asi que
`create_buffer(...).enqueue_fill(0)` encadenado no compila (ver docs/api_notes.md).
"""

from std.gpu.host import DeviceContext, DeviceBuffer


def zero_buffer[dt: DType](ctx: DeviceContext, n: Int) raises -> DeviceBuffer[dt]:
    """Buffer de n elementos puesto a cero."""
    buf = ctx.enqueue_create_buffer[dt](n)
    buf.enqueue_fill(0)
    return buf^


def filled_buffer[dt: DType](ctx: DeviceContext, n: Int,
                             value: Scalar[dt]) raises -> DeviceBuffer[dt]:
    """Buffer relleno con un valor. Util para inicializar una salida a -1 y
    poder distinguir "el kernel escribio -1" de "el kernel no escribio nada"."""
    buf = ctx.enqueue_create_buffer[dt](n)
    buf.enqueue_fill(value)
    return buf^
