"""Lector de los goldens: float32 crudo, tal como los escribe numpy con .tofile()."""

comptime dtype = DType.float32


def read_f32(path: String) raises -> List[Scalar[dtype]]:
    with open(path, "r") as f:
        data = f.read_bytes()
    n = len(data) // 4
    p = data.unsafe_ptr().bitcast[Scalar[dtype]]()
    out = List[Scalar[dtype]]()
    for i in range(n):
        out.append(p[i])
    return out^
