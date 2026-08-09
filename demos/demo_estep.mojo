"""Demo 1: the E-step in action.

It shows what SPO does before any neural network exists: starting from a policy
that knows nothing (uniform prior), the SMC search produces an improved policy q
just by simulating. It is Figure 1 of the paper turned into a demo.

    ./run.sh demos/demo_estep.mojo

It prints three things and leaves two CSVs in results/ with the raw numbers:

  1. prior vs q          the improvement, which is the main result
  2. ESS and entropy per depth   how the search degrades and how resampling
                                 recovers it
  3. temperature and particle-count sweeps
                         eta controls how aggressive the improvement is,
                         N controls how reliable the estimate is
"""

from std.gpu.host import DeviceContext
from std.math import log

from ops.common import dtype, idx_dtype
from envs.toy_chain import (default_toy_chain, ToyChain,
                            ACTION_BAD, ACTION_GOOD, NUM_ACTIONS, STATE_DIM)
from systems.spo.particles import SearchWorkspace
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from tests.helpers import upload, download

comptime SEED = UInt32(20260719)
comptime NUM_ENVS = 64
"""Plenty of envs: each one is an independent search from the same state, so
averaging over them gives a stable measurement without repeating the experiment."""


@fieldwise_init
struct SearchStats(Movable):
    """What gets measured from one search configuration."""
    var q_good: Scalar[dtype]
    """Mass the improved policy puts on the good action, averaged."""
    var ess: List[Scalar[dtype]]
    """Mean ESS per depth."""
    var entropy: List[Scalar[dtype]]
    """Mean entropy of the weights per depth."""


def run_search(ctx: DeviceContext, num_particles: Int,
               temperature: Scalar[dtype], depth: Int, period: Int,
               toy: ToyChain) raises -> SearchStats:
    """Runs a complete search and summarises the result."""
    cfg = SPOConfig(
        num_envs=NUM_ENVS, num_particles=num_particles, num_actions=NUM_ACTIONS,
        state_dim=STATE_DIM, search_depth=depth, resample_period=period,
        temperature=temperature, search_gamma=1.0, search_gae_lambda=1.0)

    ws = SearchWorkspace(ctx, cfg)

    root_state = List[Scalar[dtype]]()
    for _ in range(NUM_ENVS):
        root_state.append(0.0)      # all at the starting cell

    search[ToyChain](ctx, ws, cfg, toy, upload[dtype](ctx, root_state), SEED)
    ctx.synchronize()

    p_total = cfg.num_search_particles()
    actions = download[idx_dtype](ws.output.sampled_actions, p_total)
    weights = download[dtype](ws.output.sampled_action_weights, p_total)

    # q(GOOD) = histogram of the root actions weighted by their weights.
    q_total = Scalar[dtype](0)
    for e in range(NUM_ENVS):
        for n in range(num_particles):
            p = e * num_particles + n
            if Int(actions[p]) == ACTION_GOOD:
                q_total += weights[p]
    q_good = q_total / Scalar[dtype](NUM_ENVS)

    ess_raw = download[dtype](ws.output.ess, depth * NUM_ENVS)
    ent_raw = download[dtype](ws.output.entropy, depth * NUM_ENVS)
    ess = List[Scalar[dtype]]()
    entropy = List[Scalar[dtype]]()
    for d in range(depth):
        se = Scalar[dtype](0)
        sh = Scalar[dtype](0)
        for e in range(NUM_ENVS):
            se += ess_raw[d * NUM_ENVS + e]
            sh += ent_raw[d * NUM_ENVS + e]
        ess.append(se / Scalar[dtype](NUM_ENVS))
        entropy.append(sh / Scalar[dtype](NUM_ENVS))

    return SearchStats(q_good, ess^, entropy^)


def bar(value: Scalar[dtype], width: Int) -> String:
    """A text bar, so the result can be shown without leaving the terminal."""
    filled = Int(value * Scalar[dtype](width))
    out = String("")
    for i in range(width):
        out += "#" if i < filled else "."
    return out


def main() raises:
    with DeviceContext() as ctx:
        print("=" * 66)
        print(" Demo 1 - The E-step live (toy MDP, no networks)")
        print("=" * 66)
        print(" The prior is uniform and NOTHING has been trained.")
        print(" Everything that improves the policy comes from simulating.")
        print()

        # The paper's configuration: prior against improved policy.
        short = default_toy_chain()
        # The ESS panel needs a long corridor: with the short one every particle is
        # truncated at depth 4 and from then on their weights are frozen, that is,
        # ESS = N artificially.
        long_chain = ToyChain(chain_length=30, horizon=30, value_scale=1.0)

        base = run_search(ctx, 16, 0.5, 4, 4, short)
        prior_good = Scalar[dtype](1.0) / Scalar[dtype](NUM_ACTIONS)

        print(" 1. The policy before and after the search")
        print("    (16 particles, depth 4, temperature 0.5)")
        print()
        print("      action   prior              improved q")
        print("      BAD      ", bar(prior_good, 20), " ", prior_good,
              "   ", bar(1.0 - base.q_good, 20), " ", 1.0 - base.q_good)
        print("      GOOD     ", bar(prior_good, 20), " ", prior_good,
              "   ", bar(base.q_good, 20), " ", base.q_good)
        print()

        # How the search's health evolves with depth.
        deep = run_search(ctx, 16, 0.5, 8, 4, long_chain)
        print(" 2. Health of the search by depth (ESS out of 16, period 4)")
        print("    (long corridor, so the particles do not die too early)")
        print("    The ESS falls as the particles spread apart and resampling")
        print("    brings it back. The arrow marks where resampling happens.")
        print()
        print("      depth   ESS                          entropy")
        for d in range(len(deep.ess)):
            mark = "  <- resampling" if (d + 1) % 4 == 0 else ""
            print("      ", d, "     ", bar(deep.ess[d] / 16.0, 20), " ",
                  deep.ess[d], "   ", deep.entropy[d], mark)
        print()

        # Temperature sweep.
        print(" 3. What the temperature eta does")
        print("    Low = only the best survive (more improvement, less ESS).")
        print("    High = all alike (less improvement, more ESS).")
        print()
        print("      eta     q(GOOD)                      ESS final")
        temps = List[Scalar[dtype]]()
        temps.append(0.1); temps.append(0.5); temps.append(2.0)
        for i in range(len(temps)):
            st = run_search(ctx, 16, temps[i], 4, 4, short)
            print("      ", temps[i], "  ", bar(st.q_good, 20), " ", st.q_good,
                  "   ", st.ess[len(st.ess) - 1])
        print()

        # Particle-count sweep.
        print(" 4. What the number of particles N does")
        print("    More particles = a better estimate of the improved policy.")
        print()
        print("      N       q(GOOD)")
        counts = List[Int]()
        counts.append(4); counts.append(16); counts.append(64)
        for i in range(len(counts)):
            st = run_search(ctx, counts[i], 0.5, 4, 4, short)
            print("      ", counts[i], "    ", bar(st.q_good, 20), " ", st.q_good)
        print()

        # The raw numbers to disk, in case I want to look at them later.
        with open("results/estep_policy.csv", "w") as f:
            f.write(String("setting,temperature,num_particles,action,probability\n"))
            f.write(String("prior,0.5,16,BAD,", prior_good, "\n"))
            f.write(String("prior,0.5,16,GOOD,", prior_good, "\n"))
            for i in range(len(temps)):
                st = run_search(ctx, 16, temps[i], 4, 4, short)
                f.write(String("search,", temps[i], ",16,BAD,", 1.0 - st.q_good, "\n"))
                f.write(String("search,", temps[i], ",16,GOOD,", st.q_good, "\n"))
            for i in range(len(counts)):
                st = run_search(ctx, counts[i], 0.5, 4, 4, short)
                f.write(String("search,0.5,", counts[i], ",BAD,", 1.0 - st.q_good, "\n"))
                f.write(String("search,0.5,", counts[i], ",GOOD,", st.q_good, "\n"))

        with open("results/estep_ess.csv", "w") as f:
            f.write(String("depth,ess,entropy,num_particles,resample_period\n"))
            for d in range(len(deep.ess)):
                f.write(String(d, ",", deep.ess[d], ",", deep.entropy[d], ",16,4\n"))

        print(" Raw numbers in results/estep_policy.csv and results/estep_ess.csv")
        print("=" * 66)
