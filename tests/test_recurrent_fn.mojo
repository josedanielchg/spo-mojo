"""One full depth of the search, with values dictated by hand.

I do not use root_fn here on purpose: I place the particles where I want and
dictate their action, so as to be able to check the three endings separately and
with exact numbers. It is the plan's equivalent of "one particle with hand-set
values".

With L=8 and horizon 4, V(pos) = 8 - pos.
"""

from std.gpu.host import DeviceContext
from std.math import abs, log

from ops.common import dtype, idx_dtype
from envs.toy_chain import (toy_value, default_toy_chain,
                            ACTION_BAD, ACTION_GOOD, NUM_ACTIONS, STATE_DIM)
from systems.spo.particles import Particles, StepOutputs
from systems.spo.spo_types import SPOConfig
from systems.spo.root import sample_next_actions
from tests.helpers import (upload, download, write_into, zeros, assert_close,
                           assert_eq_int)

comptime TOL = Scalar[dtype](1e-6)


def make_config(num_particles: Int) -> SPOConfig:
    """A single env with `num_particles` particles: that way index p is the particle."""
    return SPOConfig(
        num_envs=1, num_particles=num_particles, num_actions=NUM_ACTIONS,
        state_dim=STATE_DIM, search_depth=4, resample_period=4,
        temperature=0.5, search_gamma=1.0, search_gae_lambda=1.0)


def test_one_depth_all_three_endings(ctx: DeviceContext) raises:
    """The three particles cover the three possible paths of one step."""
    toy = default_toy_chain()
    cfg = make_config(3)
    p_total = cfg.num_search_particles()

    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    # [0] normal step : pos 0, GOOD -> 1
    # [1] truncation  : pos 3, GOOD -> 4 == horizon
    # [2] terminal    : pos 0, BAD
    positions = List[Scalar[dtype]]()
    positions.append(0.0); positions.append(3.0); positions.append(0.0)
    actions = List[Scalar[idx_dtype]]()
    actions.append(Scalar[idx_dtype](ACTION_GOOD))
    actions.append(Scalar[idx_dtype](ACTION_GOOD))
    actions.append(Scalar[idx_dtype](ACTION_BAD))

    # Each particle's old value: V(0)=8, V(3)=5, V(0)=8.
    values = List[Scalar[dtype]]()
    values.append(8.0); values.append(5.0); values.append(8.0)

    write_into[dtype](particles.state, positions)
    write_into[idx_dtype](outputs.next_action, actions)
    write_into[dtype](particles.value, values)

    # Fixed uniforms: 0.1 falls in action 0 with a uniform prior (CDF [0.5, 1.0]).
    us = List[Scalar[dtype]]()
    for _ in range(p_total):
        us.append(0.1)
    uniforms = upload[dtype](ctx, us)

    step_us = zeros[dtype](ctx, p_total)
    toy.step(ctx, cfg, particles, outputs, step_us)
    sample_next_actions(ctx, outputs, cfg, uniforms)
    ctx.synchronize()

    state = download[dtype](particles.state, p_total)
    reward = download[dtype](outputs.reward, p_total)
    discount = download[dtype](outputs.discount, p_total)
    next_value = download[dtype](outputs.next_value, p_total)
    old_value = download[dtype](particles.value, p_total)

    # The first takes a normal step: it stays alive and carries V(1) = 7.
    assert_close(state[0], 1.0, TOL, "normal: posicion")
    assert_close(reward[0], 1.0, TOL, "normal: reward")
    assert_close(discount[0], 1.0, TOL, "normal: sigue viva")
    assert_close(next_value[0], 7.0, TOL, "normal: bootstrap = V(1)")

    # The second is truncated: it stops simulating but keeps V(4) = 4.
    assert_close(state[1], 4.0, TOL, "truncated: position")
    assert_close(reward[1], 1.0, TOL, "truncated: the step gave reward")
    assert_close(discount[1], 0.0, TOL, "truncated: rec_discount 0")
    assert_close(next_value[1], 4.0, TOL,
                 "truncated: the bootstrap is not 0, there was a future")

    # The third really dies: it stops simulating and also loses the future.
    assert_close(reward[2], 0.0, TOL, "terminal: no reward")
    assert_close(discount[2], 0.0, TOL, "terminal: rec_discount 0")
    assert_close(next_value[2], 0.0, TOL, "terminal: bootstrap 0")

    # The last two share rec_discount but not the bootstrap: that contrast is what
    # separates "time ran out" from "you died".
    if discount[1] != discount[2]:
        raise Error("truncated and terminal should share rec_discount")
    if next_value[1] == next_value[2]:
        raise Error("truncated and terminal should NOT share bootstrap")

    # And the old value has to stay intact, because the TD error needs it.
    assert_close(old_value[0], 8.0, TOL, "recurrent_fn must not touch particles.value")
    assert_close(old_value[1], 5.0, TOL, "recurrent_fn must not touch particles.value")

    print("PASS one depth: normal / truncated / terminal")


def test_next_action_is_sampled_and_scored(ctx: DeviceContext) raises:
    """After the step, each particle has a valid next action and its log-prob.

    The toy problem's prior is uniform over 2 actions, so the log-prob has to be
    log(0.5) whichever action comes out.
    """
    toy = default_toy_chain()
    cfg = make_config(8)
    p_total = cfg.num_search_particles()

    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    positions = List[Scalar[dtype]]()
    actions = List[Scalar[idx_dtype]]()
    for _ in range(p_total):
        positions.append(0.0)
        actions.append(Scalar[idx_dtype](ACTION_GOOD))
    write_into[dtype](particles.state, positions)
    write_into[idx_dtype](outputs.next_action, actions)

    # Spread-out uniforms: some below 0.5 (action 0) and some above.
    us = List[Scalar[dtype]]()
    for p in range(p_total):
        us.append(Scalar[dtype](p) / Scalar[dtype](p_total))
    uniforms = upload[dtype](ctx, us)

    step_us = zeros[dtype](ctx, p_total)
    toy.step(ctx, cfg, particles, outputs, step_us)
    sample_next_actions(ctx, outputs, cfg, uniforms)
    ctx.synchronize()

    next_actions = download[idx_dtype](outputs.next_action, p_total)
    logps = download[dtype](outputs.next_prior_logits, p_total)

    want_logp = log(Scalar[dtype](0.5))
    seen_zero = False
    seen_one = False
    for p in range(p_total):
        a = Int(next_actions[p])
        if a < 0 or a >= NUM_ACTIONS:
            raise Error("invalid next action at ", p, ": ", a)
        if a == 0:
            seen_zero = True
        else:
            seen_one = True
        assert_close(logps[p], want_logp, Scalar[dtype](1e-5),
                     String("log-prob of particle ", p))

    # With uniforms spread on both sides of the cut, both actions have to come
    # out; if only one did, the sampling would be broken even if the log-probs
    # lined up.
    if not seen_zero or not seen_one:
        raise Error("sampling always returned the same action")
    print("PASS next action sampled and scored with the prior")


def test_dead_particle_keeps_walking_is_not_a_problem(ctx: DeviceContext) raises:
    """An already-dead particle gets overwritten again, and it does not matter.

    The model does no auto-reset and does not check `terminal`: it keeps applying
    the dynamics. I document it with a test because it looks like a bug and is not:
    the SMC core's terminal mask freezes its weight, so what happens to it after
    dying does not enter the search. Here I only check that the step does not blow
    up nor produce strange values.
    """
    toy = default_toy_chain()
    cfg = make_config(2)
    p_total = cfg.num_search_particles()

    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    # Both already past the horizon, advancing all the same.
    positions = List[Scalar[dtype]]()
    positions.append(6.0); positions.append(9.0)
    actions = List[Scalar[idx_dtype]]()
    actions.append(Scalar[idx_dtype](ACTION_GOOD))
    actions.append(Scalar[idx_dtype](ACTION_GOOD))
    write_into[dtype](particles.state, positions)
    write_into[idx_dtype](outputs.next_action, actions)

    us = List[Scalar[dtype]]()
    us.append(0.1); us.append(0.9)
    uniforms = upload[dtype](ctx, us)

    step_us = zeros[dtype](ctx, p_total)
    toy.step(ctx, cfg, particles, outputs, step_us)
    sample_next_actions(ctx, outputs, cfg, uniforms)
    ctx.synchronize()

    next_value = download[dtype](outputs.next_value, p_total)
    # V is clipped at 0, so going past the end gives no negative values.
    assert_close(next_value[0], toy_value(7.0, toy.chain_length,
                                          toy.value_scale), TOL,
                 "V(7) after passing the horizon")
    assert_close(next_value[1], 0.0, TOL, "V cannot be negative past the end")
    print("PASS going past the end produces no negative values")


def main() raises:
    with DeviceContext() as ctx:
        test_one_depth_all_three_endings(ctx)
        test_next_action_is_sampled_and_scored(ctx)
        test_dead_particle_keeps_walking_is_not_a_problem(ctx)
