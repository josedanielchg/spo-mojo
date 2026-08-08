"""Reader for the goldens.

Format: raw float32, exactly as numpy writes it with .tofile(). I chose binary
rather than CSV because decimal rounds and then there is no telling whether a
difference comes from the kernel or from the text; the .txt next to it stores the
shapes.
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
