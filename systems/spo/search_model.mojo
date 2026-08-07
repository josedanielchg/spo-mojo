"""The contract the search asks of a model. Only this.

It gets its own file on purpose: it is the system's boundary. What sits on this
side (two methods) is everything that has to be written to plug a new environment
into the search; the rest of the E-step is left untouched.

That it lives here and not in the search's file matters for the direction of the
dependencies: an environment implements `SearchModel` by importing only the
contract and the data types, without depending on the algorithm. That way
`envs/cartpole.mojo` will be usable as a real environment, in the learning loop,
without dragging the E-step along.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from ops.common import dtype
from systems.spo.particles import Particles, StepOutputs
from systems.spo.spo_types import SPOConfig


trait SearchModel:
    """What the search needs a model to know how to do. It is two things.

    It is the equivalent of the abstract `recurrent_fn` that Stoix's SPO class
    receives: the SMC core (weights, GAE, resampling, ESS, readout) does not know
    whether there is a toy MDP, CartPole or an MLP behind it.

    How this works in Mojo 1.0.0b1, which is what took the effort to find out: the
    model is an INSTANCE that always stays on the host, and its `enqueue_function`
    calls live INSIDE its own methods, where the kernel's symbol is concrete. The
    kernel never crosses the generic boundary; only the type does. Trying the
    opposite (passing the kernel as a comptime parameter, or as a device function)
    does not compile -- see docs/api_notes.md.

    That the model is an instance and not a bare type also matters: that way it
    can carry its own state. The toy problem carries three numbers; phase 5's MLP
    will carry the DeviceBuffers of the actor's and the critic's weights.
    """

    def eval_root(self, ctx: DeviceContext, cfg: SPOConfig,
                  root_state: DeviceBuffer[dtype],
                  logits_out: DeviceBuffer[dtype],
                  value_out: DeviceBuffer[dtype]) raises:
        """The prior and the value at the ROOT states, one per environment.

            root_state  [num_envs, state_dim]   input
            logits_out  [num_envs, num_actions] output: the prior's logits
            value_out   [num_envs]              output: V(s_root)
        """
        ...

    def step(self, ctx: DeviceContext, cfg: SPOConfig, particles: Particles,
             outputs: StepOutputs, step_uniforms: DeviceBuffer[dtype]) raises:
        """Advances the P particles by one depth. It is the model's `recurrent_fn`.

        It reads `particles.state` and `outputs.next_action` (the action due to be
        executed) and writes:
            particles.state         the new state, in place
            outputs.reward          [P] the step's reward
            outputs.discount        [P] rec_discount: discount * (1 - truncated)
            outputs.next_value      [P] bootstrap: discount_real * search_gamma * V(s')
            outputs.action_logits   [P, num_actions] the prior at the NEW state

        Folding in gamma and the truncation is the model's responsibility, just as
        in Stoix's `recurrent_fn`, so that the SMC core does not have to know
        anything about the environment.

        `step_uniforms` [P] are uniforms in [0,1), one per particle, for models
        with a STOCHASTIC transition (e.g. TTT against a random rival: they are
        used to pick the rival's cell). A deterministic model ignores them; they
        come from the search's own RNG stream, different at each depth.

        What it does NOT do: touch `particles.value` (the TD error needs the old
        V) nor draw the next action -- that is handled by `sample_next_actions`,
        which is generic and called by the search.
        """
        ...
