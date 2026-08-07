"""The SPO search's hyperparameters.

This file is only the configuration. The rest of the vocabulary sits next to it:

    particles.mojo      the containers (Particles, StepOutputs, SPOOutput...)
    search_model.mojo   the contract the search asks of a model

The names are Stoix's on purpose (`stoix/configs/system/spo/ff_spo.yaml`) so that
the two files can be put side by side and compared.
"""

from ops.common import dtype


struct SPOConfig(Copyable, Movable):
    """The search's hyperparameters. Default values = ff_spo.yaml."""

    var num_envs: Int
    var num_particles: Int
    var num_actions: Int

    var state_dim: Int
    """The number of values needed to represent one environment's state."""

    var search_depth: Int

    var resample_period: Int
    """How many steps between resamplings (Stoix's 'period' mode)."""

    var temperature: Scalar[dtype]
    """Fixed temperature. Later we will make it dynamic according to what the
    policy learns."""

    var search_gamma: Scalar[dtype]
    """How much future rewards matter:
        gamma = 1.0 means future and present weigh the same
        gamma = 0.9 means each future step weighs 90% of the previous one
    """

    var search_gae_lambda: Scalar[dtype]
    """Controls how much the multi-step TD errors accumulate to compute the advantage.
            low lambda  = advantage based more on nearby steps
            high lambda = takes the whole trajectory more deeply into account
    """

    var dirichlet_alpha: Scalar[dtype]
    """Concentration of the Dirichlet noise at the root. Stoix:
    `root_exploration_dirichlet_alpha = 1.0`. Only alpha = 1 is implemented (a
    symmetric Dirichlet with alpha=1 is uniform over the simplex, which is sampled
    by normalising exponentials); another value would call for a Gamma sampler.
    Since Stoix's default IS 1.0, no more is needed."""

    var dirichlet_fraction: Scalar[dtype]
    """How much noise gets mixed in. Stoix:
    `root_exploration_dirichlet_fraction = 0.0`, that is, OFF by default. With 0
    the search is exactly the one from before."""

    def __init__(out self, num_envs: Int, num_particles: Int, num_actions: Int,
                 state_dim: Int, search_depth: Int, resample_period: Int,
                 temperature: Scalar[dtype], search_gamma: Scalar[dtype],
                 search_gae_lambda: Scalar[dtype],
                 dirichlet_alpha: Scalar[dtype] = 1.0,
                 dirichlet_fraction: Scalar[dtype] = 0.0):
        # Explicit init and not `@fieldwise_init` so that default values can be
        # given to the two new fields: that way the dozens of places that already
        # build an SPOConfig keep compiling untouched.
        self.num_envs = num_envs
        self.num_particles = num_particles
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.search_depth = search_depth
        self.resample_period = resample_period
        self.temperature = temperature
        self.search_gamma = search_gamma
        self.search_gae_lambda = search_gae_lambda
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_fraction = dirichlet_fraction

    def num_search_particles(self) -> Int:
        """P = envs * particles. It is the size of almost every buffer."""
        return self.num_envs * self.num_particles


def default_config(num_envs: Int, state_dim: Int, num_actions: Int) -> SPOConfig:
    """The paper's / ff_spo.yaml's values for the search.

    The temperature is fixed (0.5, Stoix's `fixed_temperature`) until phase 8,
    when the duals exist and it can be learned.
    """
    return SPOConfig(
        num_envs=num_envs,
        num_particles=16,
        num_actions=num_actions,
        state_dim=state_dim,
        search_depth=4,
        resample_period=4,
        temperature=0.5,
        search_gamma=1.0,
        search_gae_lambda=1.0,
    )
