"""Lector de los goldens.

Formato: float32 crudo, tal cual lo escribe numpy con .tofile(). Elegi binario
en vez de CSV porque el decimal redondea y entonces ya no se sabe si una
diferencia viene del kernel o del texto; el .txt de al lado guarda las shapes.
"""

from ops.common import dtype


def read_f32(path: String) raises -> List[Scalar[dtype]]:
    with open(path, "r") as f:
        raw = f.read_bytes()

    if len(raw) % 4 != 0:
        raise Error(path, ": ", len(raw), " bytes no es multiplo de 4, "
                    "no puede ser un array de float32")

    n = len(raw) // 4
    ptr = raw.unsafe_ptr().bitcast[Scalar[dtype]]()
    out = List[Scalar[dtype]]()
    for i in range(n):
        out.append(ptr[i])
    return out^
