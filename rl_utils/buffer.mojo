"""The trajectory replay buffer. Where the games played get stored.

SPO is off-policy: the search from a few updates ago is still a valid target, so
the transitions are stored and reused several times (Stoix does `epochs = 128`
passes over each batch collected). Without a buffer, each datum would be used once
and thrown away.

It lives on the HOST, with flat arrays, exactly as the plan decided: the sampled
batch is uploaded to the GPU on every update. It is deliberate host<->device
traffic -- STUDY.md's topic 2 warns that data movement dominates -- but the update
happens once per epoch, not per step, so clarity wins here. Moving it to device is
noted as an improvement.

**An honest simplification with respect to flashbax:** COMPLETE sequences of fixed
length T are stored and sampled whole. Flashbax allows starting at any position,
including straddling the write pointer, which forces care with half-overwritten
sequences. Here a sequence is either whole or absent. In tic-tac-toe each sequence
of T=32 covers about 8 complete games, so no variety is lost.

What is stored is what the CRITIC needs (stage 1):

    obs            [T, obs_dim]   the board at each step
    reward         [T]
    done           [T]
    truncated      [T]
    bootstrap_obs  [T, obs_dim]   the next observation, for the bootstrap

Stage 2 (the actor) will have to add `sampled_actions` and their weights, which is
what Stoix calls a "fat" transition. Adding a field here is mechanical.
"""

from ops.common import dtype
from ops.rng import rand_bits


struct TrajectoryBuffer(Movable):
    """Ring buffer of fixed-length sequences, in host memory."""

    var obs: List[Scalar[dtype]]
    """[capacity, t_len, obs_dim] flattened."""
    var reward: List[Scalar[dtype]]
    """[capacity, t_len]"""
    var done: List[Scalar[dtype]]
    var truncated: List[Scalar[dtype]]
    var bootstrap_obs: List[Scalar[dtype]]
    var q: List[Scalar[dtype]]
    """[capacity, t_len, num_actions] the improved policy the search produced at
    each step. It is the actor's target (equation 11). With num_actions = 0
    nothing is allocated: that way the loop that only trains the critic does not
    pay for this."""

    var capacity: Int
    var t_len: Int
    var obs_dim: Int
    var num_actions: Int
    var write: Int
    """Where the next sequence goes. It wraps around on reaching the end."""
    var count: Int
    """How many valid sequences there are (it stays at capacity once full)."""

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
        """Stores ONE sequence of t_len steps. Once full it overwrites the oldest.

        `q` is only needed if the buffer was created with num_actions > 0 (that
        is, if the actor is going to be trained). The size is validated rather
        than trusted: passing a short q would write garbage into the missing steps
        and the actor would learn from it without anything failing."""
        if len(reward) != self.t_len or len(done) != self.t_len \
                or len(truncated) != self.t_len:
            raise Error("the sequence should have ", self.t_len, " steps")
        if len(obs) != self.t_len * self.obs_dim \
                or len(bootstrap_obs) != self.t_len * self.obs_dim:
            raise Error("the observations should be t_len * obs_dim")
        if len(q) != self.t_len * self.num_actions:
            raise Error("q should have t_len * num_actions = ",
                        self.t_len * self.num_actions, " values, and has ",
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
        """`n` indices of valid sequences, with replacement.

        It uses the same counter-based RNG as the rest of the project
        (`rand_bits`), so with the same seed the same indices come out: the
        training tests can be deterministic end to end.
        """
        if self.count == 0:
            raise Error("the buffer is empty: there is nothing to sample")
        out = List[Int]()
        for i in range(n):
            bits = rand_bits(seed, stream, UInt32(i))
            out.append(Int(bits % UInt32(self.count)))
        return out^

    def gather(self, indices: List[Int]) raises -> List[Scalar[dtype]]:
        """Those sequences' observations, ready to be uploaded to the GPU.

        Output [len(indices), t_len, obs_dim] flattened, in the same order the
        indices arrive in.
        """
        out = List[Scalar[dtype]]()
        span = self.t_len * self.obs_dim
        for k in range(len(indices)):
            idx = indices[k]
            if idx < 0 or idx >= self.count:
                raise Error("sequence index out of range: ", idx)
            base = idx * span
            for i in range(span):
                out.append(self.obs[base + i])
        return out^

    def gather_steps(self, indices: List[Int], which: Int) raises -> List[Scalar[dtype]]:
        """One per-step field of those sequences: 0=reward, 1=done, 2=truncated.

        Output [len(indices), t_len] flattened, which is exactly the shape the
        truncated GAE expects.
        """
        out = List[Scalar[dtype]]()
        for k in range(len(indices)):
            idx = indices[k]
            if idx < 0 or idx >= self.count:
                raise Error("sequence index out of range: ", idx)
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
        """Those sequences' bootstrap_obs, same shape as `gather`."""
        out = List[Scalar[dtype]]()
        span = self.t_len * self.obs_dim
        for k in range(len(indices)):
            idx = indices[k]
            base = idx * span
            for i in range(span):
                out.append(self.bootstrap_obs[base + i])
        return out^

    def gather_q(self, indices: List[Int]) raises -> List[Scalar[dtype]]:
        """The q of the requested sequences, in the requested order."""
        if self.num_actions == 0:
            raise Error("this buffer was created without q (num_actions = 0)")
        span = self.t_len * self.num_actions
        out = List[Scalar[dtype]]()
        for k in range(len(indices)):
            idx = indices[k]
            if idx < 0 or idx >= self.count:
                raise Error("index out of range: ", idx)
            base = idx * span
            for i in range(span):
                out.append(self.q[base + i])
        return out^
