"""El replay buffer de trayectorias. Donde se guardan las partidas jugadas.

SPO es off-policy: la busqueda de hace unos cuantos updates sigue siendo un
objetivo valido, asi que las transiciones se guardan y se reutilizan varias veces
(Stoix hace `epochs = 128` pasadas por cada tanda recogida). Sin buffer, cada dato
se usaria una vez y se tiraria.

Vive en el HOST, con arrays planos, tal como decidio el plan: el batch muestreado
se sube a la GPU en cada update. Es trafico host<->device consciente — el tema 2 de
STUDY.md avisa de que el movimiento de datos domina — pero el update ocurre una vez
por epoch, no por paso, asi que aqui prima la claridad. Moverlo a device queda
apuntado como mejora.

**Simplificacion honesta frente a flashbax:** se guardan SECUENCIAS COMPLETAS de
longitud fija T y se muestrean enteras. Flashbax permite empezar en cualquier
posicion, tambien a caballo del puntero de escritura, lo que obliga a llevar
cuidado con las secuencias medio pisadas. Aqui una secuencia esta entera o no
esta. En tres en raya cada secuencia de T=32 cubre unas 8 partidas completas, asi
que no se pierde variedad.

Lo que se guarda es lo que necesita el CRITICO (etapa 1):

    obs            [T, obs_dim]   el tablero en cada paso
    reward         [T]
    done           [T]
    truncated      [T]
    bootstrap_obs  [T, obs_dim]   la observacion siguiente, para el bootstrap

La etapa 2 (el actor) tendra que anadir `sampled_actions` y sus pesos, que es lo
que Stoix llama una transicion "gorda". Anadir un campo aqui es mecanico.
"""

from ops.common import dtype
from ops.rng import rand_bits


struct TrajectoryBuffer(Movable):
    """Ring buffer de secuencias de longitud fija, en memoria de host."""

    var obs: List[Scalar[dtype]]
    """[capacity, t_len, obs_dim] aplanado."""
    var reward: List[Scalar[dtype]]
    """[capacity, t_len]"""
    var done: List[Scalar[dtype]]
    var truncated: List[Scalar[dtype]]
    var bootstrap_obs: List[Scalar[dtype]]
    var q: List[Scalar[dtype]]
    """[capacity, t_len, num_actions] la politica mejorada que produjo la busqueda
    en cada paso. Es el objetivo del actor (ecuacion 11). Con num_actions = 0 no se
    reserva nada: asi el bucle que solo entrena al critico no paga por esto."""

    var capacity: Int
    var t_len: Int
    var obs_dim: Int
    var num_actions: Int
    var write: Int
    """Donde va la proxima secuencia. Da la vuelta al llegar al final."""
    var count: Int
    """Cuantas secuencias validas hay (se queda en capacity al llenarse)."""

    def __init__(out self, capacity: Int, t_len: Int, obs_dim: Int,
                 num_actions: Int = 0):
        self.capacity = capacity
        self.t_len = t_len
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.write = 0
        self.count = 0

        n_obs = capacity * t_len * obs_dim
        n_step = capacity * t_len
        self.obs = List[Scalar[dtype]]()
        self.bootstrap_obs = List[Scalar[dtype]]()
        for _ in range(n_obs):
            self.obs.append(Scalar[dtype](0))
            self.bootstrap_obs.append(Scalar[dtype](0))
        self.reward = List[Scalar[dtype]]()
        self.done = List[Scalar[dtype]]()
        self.truncated = List[Scalar[dtype]]()
        for _ in range(n_step):
            self.reward.append(Scalar[dtype](0))
            self.done.append(Scalar[dtype](0))
            self.truncated.append(Scalar[dtype](0))
        self.q = List[Scalar[dtype]]()
        for _ in range(n_step * num_actions):
            self.q.append(Scalar[dtype](0))

    def size(self) -> Int:
        return self.count

    def is_full(self) -> Bool:
        return self.count >= self.capacity

    def add(mut self, obs: List[Scalar[dtype]], reward: List[Scalar[dtype]],
            done: List[Scalar[dtype]], truncated: List[Scalar[dtype]],
            bootstrap_obs: List[Scalar[dtype]],
            q: List[Scalar[dtype]] = List[Scalar[dtype]]()) raises:
        """Guarda UNA secuencia de t_len pasos. Al llenarse pisa la mas vieja.

        `q` solo hace falta si el buffer se creo con num_actions > 0 (o sea, si se
        va a entrenar al actor). Se valida el tamano en vez de confiar: meter una q
        corta escribiria basura en los pasos que faltan y el actor aprenderia de
        ella sin que nada fallara."""
        if len(reward) != self.t_len or len(done) != self.t_len \
                or len(truncated) != self.t_len:
            raise Error("la secuencia deberia tener ", self.t_len, " pasos")
        if len(obs) != self.t_len * self.obs_dim \
                or len(bootstrap_obs) != self.t_len * self.obs_dim:
            raise Error("las observaciones deberian ser t_len * obs_dim")
        if len(q) != self.t_len * self.num_actions:
            raise Error("q deberia tener t_len * num_actions = ",
                        self.t_len * self.num_actions, " valores, y tiene ",
                        len(q))

        slot = self.write
        obs_base = slot * self.t_len * self.obs_dim
        step_base = slot * self.t_len
        for i in range(self.t_len * self.obs_dim):
            self.obs[obs_base + i] = obs[i]
            self.bootstrap_obs[obs_base + i] = bootstrap_obs[i]
        for i in range(self.t_len):
            self.reward[step_base + i] = reward[i]
            self.done[step_base + i] = done[i]
            self.truncated[step_base + i] = truncated[i]
        q_base = slot * self.t_len * self.num_actions
        for i in range(self.t_len * self.num_actions):
            self.q[q_base + i] = q[i]

        self.write = (self.write + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1

    def sample_indices(self, n: Int, seed: UInt32,
                       stream: UInt32) raises -> List[Int]:
        """`n` indices de secuencias validas, con reemplazo.

        Usa el mismo RNG contador que el resto del proyecto (`rand_bits`), asi que
        con la misma semilla salen los mismos indices: los tests de entrenamiento
        pueden ser deterministas de punta a punta.
        """
        if self.count == 0:
            raise Error("el buffer esta vacio: no hay nada que muestrear")
        out = List[Int]()
        for i in range(n):
            bits = rand_bits(seed, stream, UInt32(i))
            out.append(Int(bits % UInt32(self.count)))
        return out^

    def gather(self, indices: List[Int]) raises -> List[Scalar[dtype]]:
        """Las observaciones de esas secuencias, listas para subir a la GPU.

        Salida [len(indices), t_len, obs_dim] aplanada, en el mismo orden en que
        vienen los indices.
        """
        out = List[Scalar[dtype]]()
        span = self.t_len * self.obs_dim
        for k in range(len(indices)):
            idx = indices[k]
            if idx < 0 or idx >= self.count:
                raise Error("indice de secuencia fuera de rango: ", idx)
            base = idx * span
            for i in range(span):
                out.append(self.obs[base + i])
        return out^

    def gather_steps(self, indices: List[Int], which: Int) raises -> List[Scalar[dtype]]:
        """Un campo por paso de esas secuencias: 0=reward, 1=done, 2=truncated.

        Salida [len(indices), t_len] aplanada, que es justo la forma que espera la
        GAE truncada.
        """
        out = List[Scalar[dtype]]()
        for k in range(len(indices)):
            idx = indices[k]
            if idx < 0 or idx >= self.count:
                raise Error("indice de secuencia fuera de rango: ", idx)
            base = idx * self.t_len
            for i in range(self.t_len):
                if which == 0:
                    out.append(self.reward[base + i])
                elif which == 1:
                    out.append(self.done[base + i])
                else:
                    out.append(self.truncated[base + i])
        return out^

    def gather_bootstrap(self, indices: List[Int]) raises -> List[Scalar[dtype]]:
        """Las bootstrap_obs de esas secuencias, misma forma que `gather`."""
        out = List[Scalar[dtype]]()
        span = self.t_len * self.obs_dim
        for k in range(len(indices)):
            idx = indices[k]
            base = idx * span
            for i in range(span):
                out.append(self.bootstrap_obs[base + i])
        return out^

    def gather_q(self, indices: List[Int]) raises -> List[Scalar[dtype]]:
        """Las q de las secuencias pedidas, en el orden pedido."""
        if self.num_actions == 0:
            raise Error("este buffer se creo sin q (num_actions = 0)")
        span = self.t_len * self.num_actions
        out = List[Scalar[dtype]]()
        for k in range(len(indices)):
            idx = indices[k]
            if idx < 0 or idx >= self.count:
                raise Error("indice fuera de rango: ", idx)
            base = idx * span
            for i in range(span):
                out.append(self.q[base + i])
        return out^
