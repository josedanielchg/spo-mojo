"""Tipos y constantes que comparten todos los kernels.

Estaban repetidos en cada modulo de ops/ y acabaron desincronizandose, asi que
los junto aqui. Cambiar el proyecto a float16 deberia ser tocar una linea.
"""

from std.memory import AddressSpace

# El plan dice float32 en todo el proyecto: la GPU de portatil no tiene tensor
# cores utiles para float64 y para CartPole sobra de largo.
comptime dtype = DType.float32

# Indices (acciones elegidas, particulas seleccionadas en el resampling).
# int32 en vez de int64 porque nunca vamos a tener mas de 2^31 particulas y
# asi ocupa la mitad al copiar de device a host.
comptime idx_dtype = DType.int32

# Elemento neutro del max: cualquier valor real le gana.
# Uso MIN_FINITE y no MIN (que es -inf) a proposito: con -inf, una fila entera
# de -inf daria row_max = -inf y luego exp(-inf - -inf) = exp(nan) = nan.
# Con MIN_FINITE el caso degenerado da 0 en vez de propagar un nan.
comptime NEG_INF = Scalar[dtype].MIN_FINITE

# Punteros a shared memory. El tipo completo es largo de escribir y aparece en
# la firma de todos los helpers, asi que le pongo nombre.
comptime SharedF32 = UnsafePointer[Scalar[dtype], MutAnyOrigin,
                                   address_space = AddressSpace.SHARED]

# Punteros a memoria global (los buffers que vienen de DeviceContext).
comptime GlobalF32 = UnsafePointer[Scalar[dtype], MutAnyOrigin]
comptime GlobalI32 = UnsafePointer[Scalar[idx_dtype], MutAnyOrigin]
