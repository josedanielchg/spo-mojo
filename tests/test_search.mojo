"""The full SMC search over the toy problem: BEHAVIOUR tests.

Nothing is dictated here, `search()` is run end to end and what comes out is
inspected. The third one is the one that matters in the whole of phase 3: starting
from a policy that knows nothing (uniform prior) and WITHOUT training anything, the
search has to concentrate the mass on the good action. If that happens, the E-step
works.

The individual pieces (resampling, ESS) are tested in test_resampling.mojo.
"""

from std.gpu.host import DeviceContext

from ops.common import dtype, idx_dtype
from envs.toy_chain import (default_toy_chain, ToyChain,
                            ACTION_BAD, ACTION_GOOD, NUM_ACTIONS, STATE_DIM)
from systems.spo.particles import SearchWorkspace
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from systems.spo.readout import (readout_greedy, readout_expected,
                                 readout_weighted)
from ops.buffers import zero_buffer
from tests.helpers import upload, download, write_into, assert_close

comptime TOL = Scalar[dtype](1e-5)


def make_config(num_envs: Int, num_particles: Int, depth: Int,
                period: Int) -> SPOConfig:
    return SPOConfig(
        num_envs=num_envs, num_particles=num_particles, num_actions=NUM_ACTIONS,
        state_dim=STATE_DIM, search_depth=depth, resample_period=period,
        temperature=0.5, search_gamma=1.0, search_gae_lambda=1.0)


def test_ess_drops_and_recovers_after_resampling(ctx: DeviceContext) raises:
    """Along the rollout the ESS drops and recovers after each resampling.

    With depth 8 and period 4 there is resampling after depths 3 and 7. Since the
    ESS is measured BEFORE resampling, depth 4 (the first after the reset) has to
    look healthier than depth 3.
    """
    cfg = make_config(num_envs=4, num_particles=16, depth=8, period=4)
    toy = ToyChain(chain_length=20, horizon=20, value_scale=1.0)

    ws = SearchWorkspace(ctx, cfg)

    root_state = List[Scalar[dtype]]()
    for _ in range(cfg.num_envs):
        root_state.append(0.0)

    search[ToyChain](ctx, ws, cfg, toy, upload[dtype](ctx, root_state),
                     UInt32(2024))
    ctx.synchronize()

    ess = download[dtype](ws.output.ess, cfg.search_depth * cfg.num_envs)

    # Mean per depth over the envs
    means = List[Scalar[dtype]]()
    for d in range(cfg.search_depth):
        total = Scalar[dtype](0)
        for e in range(cfg.num_envs):
            total += ess[d * cfg.num_envs + e]
        means.append(total / Scalar[dtype](cfg.num_envs))

    print("      ESS by depth:")
    for d in range(cfg.search_depth):
        print("        depth", d, "->", means[d])

    # Falls within the first stretch...
    if means[3] >= means[0]:
        raise Error("the ESS should degrade between resamplings: d0=",
                    means[0], " d3=", means[3])
    # ...and recovers right after depth 3's resampling.
    if means[4] <= means[3]:
        raise Error("the ESS should recover after the resampling: d3=",
                    means[3], " d4=", means[4])
    print("PASS the ESS falls between resamplings and recovers afterwards")


def test_search_improves_a_uniform_prior(ctx: DeviceContext) raises:
    """THE phase's test: the search improves a policy that knows nothing.

    The toy problem's prior is uniform (50/50) and nothing is trained. By
    simulating alone, the improved policy q has to put at least 80% of the mass on
    the good action.

    q is read from the output exactly as the M-step will use it: the histogram of
    `sampled_actions` weighted by `sampled_action_weights`.
    """
    cfg = make_config(num_envs=8, num_particles=16, depth=4, period=4)
    toy = default_toy_chain()

    ws = SearchWorkspace(ctx, cfg)

    root_state = List[Scalar[dtype]]()
    for _ in range(cfg.num_envs):
        root_state.append(0.0)      # all on the starting cell

    search[ToyChain](ctx, ws, cfg, toy, upload[dtype](ctx, root_state),
                     UInt32(1234))
    ctx.synchronize()

    p_total = cfg.num_search_particles()
    actions = download[idx_dtype](ws.output.sampled_actions, p_total)
    weights = download[dtype](ws.output.sampled_action_weights, p_total)
    final_action = download[idx_dtype](ws.output.action, cfg.num_envs)

    worst = Scalar[dtype](1.0)
    for e in range(cfg.num_envs):
        q_good = Scalar[dtype](0)
        q_total = Scalar[dtype](0)
        for n in range(cfg.num_particles):
            p = e * cfg.num_particles + n
            q_total += weights[p]
            if Int(actions[p]) == ACTION_GOOD:
                q_good += weights[p]
        # The weights are a per-env softmax, so they already sum to 1.
        assert_close(q_total, 1.0, Scalar[dtype](1e-4),
                     String("the weights of env ", e, " should sum to 1"))
        if q_good < worst:
            worst = q_good

    print("      minimum q(GOOD) over", cfg.num_envs, "envs:", worst)
    if worst < 0.8:
        raise Error("the search did not improve the prior enough: q(GOOD)=", worst)

    # And the action actually executed has to be valid.
    for e in range(cfg.num_envs):
        a = Int(final_action[e])
        if a != ACTION_BAD and a != ACTION_GOOD:
            raise Error("invalid final action in env ", e, ": ", a)

    print("PASS the search improves the uniform prior: q(GOOD) >=", worst)


def test_search_is_reproducible(ctx: DeviceContext) raises:
    """Same seed, same search. Without this no test is worth anything."""
    cfg = make_config(num_envs=4, num_particles=16, depth=4, period=4)
    toy = default_toy_chain()
    p_total = cfg.num_search_particles()

    root_state = List[Scalar[dtype]]()
    for _ in range(cfg.num_envs):
        root_state.append(0.0)

    first = List[Scalar[dtype]]()
    for run in range(2):
        ws = SearchWorkspace(ctx, cfg)
        search[ToyChain](ctx, ws, cfg, toy, upload[dtype](ctx, root_state),
                         UInt32(555))
        ctx.synchronize()
        w = download[dtype](ws.output.sampled_action_weights, p_total)
        if run == 0:
            first = w^
        else:
            for p in range(p_total):
                assert_close(w[p], first[p], TOL,
                             String("two searches with the same seed differ at ", p))
    print("PASS the search is reproducible with the same seed")


def test_more_particles_is_not_worse(ctx: DeviceContext) raises:
    """Regression: with a large N the search has to keep improving, not get worse.

    This test exists because of a real bug. The kernels whose row is the particle
    dimension (resampling, ESS, the readout's softmax) use ONE block per env, so N
    has to fit in the block. With blocks of 32 and N=64 the search returned
    q(GOOD)=0.75, WORSE than with N=16 (0.99), and without warning about anything.

    The kernel's debug_assert catches it, but only with -D ASSERT=all; that is why
    there is also a host-side check (check_search_config) and this test.
    """
    toy = default_toy_chain()
    q = List[Scalar[dtype]]()

    counts = List[Int]()
    counts.append(4); counts.append(16); counts.append(64)

    for i in range(len(counts)):
        n = counts[i]
        cfg = make_config(num_envs=16, num_particles=n, depth=4, period=4)
        ws = SearchWorkspace(ctx, cfg)

        root_state = List[Scalar[dtype]]()
        for _ in range(cfg.num_envs):
            root_state.append(0.0)

        search[ToyChain](ctx, ws, cfg, toy, upload[dtype](ctx, root_state),
                         UInt32(4321))
        ctx.synchronize()

        p_total = cfg.num_search_particles()
        actions = download[idx_dtype](ws.output.sampled_actions, p_total)
        weights = download[dtype](ws.output.sampled_action_weights, p_total)
        total = Scalar[dtype](0)
        for p in range(p_total):
            if Int(actions[p]) == ACTION_GOOD:
                total += weights[p]
        q.append(total / Scalar[dtype](cfg.num_envs))
        print("      N =", n, "-> q(GOOD) =", q[i])

    if q[1] < q[0] or q[2] < q[1]:
        raise Error("more particles should improve (or at least not worsen): ",
                    q[0], " ", q[1], " ", q[2])
    print("PASS more particles does not worsen the search")


def test_reusing_the_workspace_gives_the_same_result(ctx: DeviceContext) raises:
    """Regression: reusing the workspace has to give the SAME result.

    This test exists because of a real bug, and an expensive one: `root_fn` seeded
    the particles but did not zero the accumulators (weight, gae, terminal, depth).
    With a freshly created workspace it did not show, because they are born at zero;
    but the SearchWorkspace exists precisely to be allocated ONCE and reused at
    every environment step, and there the second search inherited `terminal = 1`
    from the first. update_particles' mask then froze the weight of ALL the
    particles from depth 0, the weights stayed at zero, the readout's softmax came
    out uniform and the search degenerated into choosing at random.

    The worst of it is that nothing failed: the search kept returning valid actions,
    only bad ones. It was detected because a planner on tic-tac-toe won no more
    games than playing at random.

    The check is direct: two identical searches, one with a fresh workspace and one
    reusing a workspace that has already run, have to give exactly the same thing.
    """
    # No resampling (period > depth) and a long corridor, on purpose: that way the
    # weights reach the readout with values that differ from one another and it can
    # be demanded that they NOT be uniform. With resampling the weights are reset to
    # zero by design (the information moves into the multiplicity), and that check
    # would not discriminate the bug.
    cfg = make_config(num_envs=4, num_particles=16, depth=6, period=99)
    toy = ToyChain(chain_length=20, horizon=20, value_scale=1.0)
    p_total = cfg.num_search_particles()

    root_state = List[Scalar[dtype]]()
    for _ in range(cfg.num_envs):
        root_state.append(0.0)

    # Reference: a fresh workspace, the search we care about.
    fresh = SearchWorkspace(ctx, cfg)
    search[ToyChain](ctx, fresh, cfg, toy, upload[dtype](ctx, root_state),
                     UInt32(8080))
    ctx.synchronize()
    want_w = download[dtype](fresh.output.sampled_action_weights, p_total)
    want_a = download[idx_dtype](fresh.output.action, cfg.num_envs)

    # And now the same workspace after having run a different search.
    reused = SearchWorkspace(ctx, cfg)
    search[ToyChain](ctx, reused, cfg, toy, upload[dtype](ctx, root_state),
                     UInt32(1111))          # any search at all, to dirty it
    ctx.synchronize()
    search[ToyChain](ctx, reused, cfg, toy, upload[dtype](ctx, root_state),
                     UInt32(8080))          # and now the one that has to match
    ctx.synchronize()
    got_w = download[dtype](reused.output.sampled_action_weights, p_total)
    got_a = download[idx_dtype](reused.output.action, cfg.num_envs)

    for p in range(p_total):
        assert_close(got_w[p], want_w[p], TOL,
                     String("the reused workspace differs at particle ", p))
    for e in range(cfg.num_envs):
        if Int(got_a[e]) != Int(want_a[e]):
            raise Error("the reused workspace chose a different action in env ", e)

    # And that the weights are not all equal: if the bug were present they would
    # all come out at zero and the check above would pass anyway by being identical.
    spread = False
    for p in range(1, p_total):
        if got_w[p] != got_w[0]:
            spread = True
    if not spread:
        raise Error("all the weights came out identical: the readout is uniform, ",
                    "which is exactly the symptom of the accumulator bug")
    print("PASS reusing the workspace gives the same result as a fresh one")


def test_greedy_readout_takes_the_mode_of_q(ctx: DeviceContext) raises:
    """The greedy readout takes the action with the most q MASS, not the particle
    with the most weight.

    The distinction matters and is easy to confuse. q is a weighted histogram:
    several particles share a root action and their weights are SUMMED. An "argmax
    over particles" would take the heaviest individual particle, which need not
    belong to the most-voted action.

    The two envs are set up precisely to separate the two things, and moreover with
    different answers so that a mix-up between envs also shows:

        env 0   action 0: one particle of 0.40  <- the heaviest particle
                action 1: 0.16+0.16+0.16+0.12 = 0.60  <- the mode, the correct one
        env 1   action 0: 0.20+0.20+0.20 = 0.60   <- the mode, the correct one
                action 1: one particle of 0.30 and another of 0.10

    The toy problem has 2 actions, so q is [env0_a0, env0_a1, env1_a0, env1_a1].
    """
    num_envs = 2
    num_particles = 5
    cfg = make_config(num_envs, num_particles, 2, 2)
    ws = SearchWorkspace(ctx, cfg)
    q_buf = zero_buffer[dtype](ctx, num_envs * NUM_ACTIONS)

    roots = List[Scalar[idx_dtype]]()
    weights = List[Scalar[dtype]]()

    # The two lists in parallel: the particle's root action and its weight.
    acts = List[Int]();     ws_ = List[Float64]()
    acts.append(0); ws_.append(0.40)          # env 0, the heaviest particle
    acts.append(1); ws_.append(0.16)
    acts.append(1); ws_.append(0.16)
    acts.append(1); ws_.append(0.16)
    acts.append(1); ws_.append(0.12)          # action 1 sums to 0.60
    acts.append(1); ws_.append(0.30)          # env 1, the heaviest particle
    acts.append(0); ws_.append(0.20)
    acts.append(0); ws_.append(0.20)
    acts.append(0); ws_.append(0.20)          # action 0 sums to 0.60
    acts.append(1); ws_.append(0.10)
    for i in range(len(acts)):
        roots.append(Scalar[idx_dtype](acts[i]))
        weights.append(Scalar[dtype](ws_[i]))

    write_into[idx_dtype](ws.particles.root_actions, roots)
    write_into[dtype](ws.output.sampled_action_weights, weights)
    ctx.synchronize()

    readout_greedy(ctx, ws.particles, ws.output, cfg, q_buf)
    ctx.synchronize()

    # First the aggregated q, which is what we really want to check.
    q = download[dtype](q_buf, num_envs * NUM_ACTIONS)
    assert_close(q[0], Scalar[dtype](0.40), TOL, "q[env0, action 0]")
    assert_close(q[1], Scalar[dtype](0.60), TOL, "q[env0, action 1]")
    assert_close(q[2], Scalar[dtype](0.60), TOL, "q[env1, action 0]")
    assert_close(q[3], Scalar[dtype](0.40), TOL, "q[env1, action 1]")

    got = download[idx_dtype](ws.output.action, num_envs)
    if Int(got[0]) != 1:
        raise Error("env 0 should pick action 1 (mass 0.60) and not ",
                    Int(got[0]), "; if 0 came out it is looking at individual "
                    "particles instead of aggregating per action")
    if Int(got[1]) != 0:
        raise Error("env 1 should pick action 0 (mass 0.60), got ",
                    Int(got[1]))
    print("PASS the greedy readout takes the mode of q, not the largest particle")


def test_greedy_is_deterministic(ctx: DeviceContext) raises:
    """Two greedy readouts in a row give the same action.

    It is the difference from `readout_weighted`, which draws: if we evaluate in
    greedy mode, the same state has to give the same move every time, or the
    comparison's numbers would not be reproducible.
    """
    num_envs = 3
    num_particles = 8
    cfg = make_config(num_envs, num_particles, 3, 3)
    ws = SearchWorkspace(ctx, cfg)
    q_buf = zero_buffer[dtype](ctx, num_envs * NUM_ACTIONS)
    model = default_toy_chain()

    roots = List[Scalar[dtype]]()
    for _ in range(num_envs * STATE_DIM):
        roots.append(Scalar[dtype](0))
    root_state = upload[dtype](ctx, roots)

    search[ToyChain](ctx, ws, cfg, model, root_state, UInt32(7))
    readout_greedy(ctx, ws.particles, ws.output, cfg, q_buf)
    ctx.synchronize()
    a = download[idx_dtype](ws.output.action, num_envs)

    readout_greedy(ctx, ws.particles, ws.output, cfg, q_buf)
    ctx.synchronize()
    b = download[idx_dtype](ws.output.action, num_envs)

    for e in range(num_envs):
        if Int(a[e]) != Int(b[e]):
            raise Error("the greedy readout should be deterministic, env ", e)
    print("PASS the greedy readout is deterministic")


def test_expected_readout_punishes_risk(ctx: DeviceContext) raises:
    """The variant penalises risk; SPO's readout does not. Same weights, different
    move.

    It is the test that pins down the losses audit's finding, so the setup
    reproduces the real situation: a RISKY action (one splendid particle and three
    disastrous ones) against a SAFE one (four mediocre but equal particles).

        action 0   weights  1.0, -1.0, -1.0, -1.0    mean -0.50
        action 1   weights  0.2,  0.2,  0.2,  0.2    mean  0.20

    SPO does q(a) = SUM exp(weight/tau), which with tau=0.5 gives

        action 0   e^2 + 3*e^-2 = 7.39 + 0.41 = 7.80   <- the risky one wins
        action 1   4 * e^0.4                  = 5.97

    because the three bad particles contribute almost zero but do NOT SUBTRACT. The
    variant averages first, and then the safe one wins. Both readouts are correct
    for what they estimate: SPO's estimates E[exp(A/tau)] and the variant
    exp(E[A]/tau). They coincide if the environment is deterministic; with a random
    rival, they do not.
    """
    num_envs = 1
    num_particles = 8
    cfg = make_config(num_envs, num_particles, 2, 2)   # temperatura 0.5
    ws = SearchWorkspace(ctx, cfg)
    q_buf = zero_buffer[dtype](ctx, num_envs * NUM_ACTIONS)
    logits_buf = zero_buffer[dtype](ctx, num_envs * NUM_ACTIONS)
    us = zero_buffer[dtype](ctx, num_particles)

    roots = List[Scalar[idx_dtype]]()
    w = List[Scalar[dtype]]()
    for _ in range(4):
        roots.append(Scalar[idx_dtype](0))
    for _ in range(4):
        roots.append(Scalar[idx_dtype](1))
    w.append(Scalar[dtype](1.0))                       # la unica buena
    for _ in range(3): w.append(Scalar[dtype](-1.0))
    for _ in range(4): w.append(Scalar[dtype](0.2))
    write_into[idx_dtype](ws.particles.root_actions, roots)
    write_into[dtype](ws.particles.resample_td_weights, w)
    ctx.synchronize()

    # 1. SPO's readout: it goes for the risky one.
    readout_weighted(ctx, ws.particles, ws.scratch, ws.output, cfg, us)
    readout_greedy(ctx, ws.particles, ws.output, cfg, q_buf)
    ctx.synchronize()
    spo = download[idx_dtype](ws.output.action, num_envs)
    if Int(spo[0]) != 0:
        raise Error("with SPO's readout action 0 should come out (the one from the "
                    "good particle), got ", Int(spo[0]))

    # 2. The variant: it goes for the safe one.
    readout_expected(ctx, ws.particles, ws.output, cfg, logits_buf, q_buf, us,
                     True)
    ctx.synchronize()
    exp_a = download[idx_dtype](ws.output.action, num_envs)
    if Int(exp_a[0]) != 1:
        raise Error("with the variant action 1 should come out (larger mean), "
                    "got ", Int(exp_a[0]))

    # And the means, which are what the variant actually computes.
    logits = download[dtype](logits_buf, num_envs * NUM_ACTIONS)
    tau = Scalar[dtype](0.5)
    assert_close(logits[0] * tau, Scalar[dtype](-0.5), TOL, "mean of action 0")
    assert_close(logits[1] * tau, Scalar[dtype](0.2), TOL, "mean of action 1")
    print("PASS the variant penalises risk where SPO's readout cannot")


def test_expected_readout_ignores_unsampled_actions(ctx: DeviceContext) raises:
    """An action no particle tried is left with q = 0.

    Without this, the mean of zero particles would be 0/0 and could come out NaN
    or, worse, a 0 that competes on equal terms with actions of negative mean and
    would end up selecting a move the search never evaluated.
    """
    num_envs = 1
    num_particles = 4
    cfg = make_config(num_envs, num_particles, 2, 2)
    ws = SearchWorkspace(ctx, cfg)
    q_buf = zero_buffer[dtype](ctx, num_envs * NUM_ACTIONS)
    logits_buf = zero_buffer[dtype](ctx, num_envs * NUM_ACTIONS)
    us = zero_buffer[dtype](ctx, num_particles)

    # Every particle plays action 1, and with a NEGATIVE mean at that: if action 0
    # (untried) counted as 0, it would beat it.
    roots = List[Scalar[idx_dtype]]()
    w = List[Scalar[dtype]]()
    for _ in range(4):
        roots.append(Scalar[idx_dtype](1))
        w.append(Scalar[dtype](-2.0))
    write_into[idx_dtype](ws.particles.root_actions, roots)
    write_into[dtype](ws.particles.resample_td_weights, w)
    ctx.synchronize()

    readout_expected(ctx, ws.particles, ws.output, cfg, logits_buf, q_buf, us,
                     True)
    ctx.synchronize()

    q = download[dtype](q_buf, num_envs * NUM_ACTIONS)
    assert_close(q[0], Scalar[dtype](0), TOL,
                 "an action nobody tried has to stay at q = 0")
    assert_close(q[1], Scalar[dtype](1), TOL,
                 "all the mass goes to the only action tried")
    got = download[idx_dtype](ws.output.action, num_envs)
    if Int(got[0]) != 1:
        raise Error("should play 1 even though its mean is negative: it is the "
                    "only one the search evaluated, got ", Int(got[0]))
    print("PASS untried actions do not compete with evaluated ones")


def main() raises:
    with DeviceContext() as ctx:
        test_ess_drops_and_recovers_after_resampling(ctx)
        test_search_improves_a_uniform_prior(ctx)
        test_more_particles_is_not_worse(ctx)
        test_search_is_reproducible(ctx)
        test_reusing_the_workspace_gives_the_same_result(ctx)
        test_greedy_readout_takes_the_mode_of_q(ctx)
        test_greedy_is_deterministic(ctx)
        test_expected_readout_punishes_risk(ctx)
        test_expected_readout_ignores_unsampled_actions(ctx)
