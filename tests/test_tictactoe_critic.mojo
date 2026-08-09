"""The search model with a critic: that V comes from the network and reaches where it should.

What is checked here is not whether the critic is GOOD -- the E1.11 experiment
measures that by playing games. What is checked is the wiring, which is where this
kind of code breaks silently: if `next_value` stayed at 0, or if the value were
read from the wrong buffer, the search would go on working and giving reasonable
results, only without using the critic. Nothing would fail.

The trick in every test is the same: weights chosen so that V(s) is a KNOWN number,
and comparing against that number. With w1 = w2 = w3 = 0 and b3 = c:

    a1 = x@0 + b1 = b1  ->  relu
    a2 = a1@0 + b2 = b2 ->  relu
    V  = a2@0 + b3 = c              for any board

so V is c and does not depend on the input. That makes it possible to predict by
hand what has to appear in each buffer.
"""

from std.gpu.host import DeviceContext

from ops.common import dtype, idx_dtype
from envs.tictactoe import (NUM_ACTIONS, STATE_DIM, NUM_CELLS, OBS_DIM,
                            CELL_EMPTY, NEG_INF)
from envs.tictactoe_critic import TicTacToeCritic
from networks.mlp import zero_critic_params
from systems.spo.particles import Particles, StepOutputs
from systems.spo.spo_types import SPOConfig
from tests.helpers import (upload, download, write_into, zeros, filled,
                           assert_close)

comptime TOL = Scalar[dtype](1e-5)
comptime HIDDEN = 8
comptime GAMMA = Scalar[dtype](0.9)


def board9(c0: Int, c1: Int, c2: Int, c3: Int, c4: Int,
           c5: Int, c6: Int, c7: Int, c8: Int) -> List[Scalar[dtype]]:
    """A readable board: 9 cell codes in the order 0..8."""
    out = List[Scalar[dtype]]()
    out.append(Scalar[dtype](c0)); out.append(Scalar[dtype](c1))
    out.append(Scalar[dtype](c2)); out.append(Scalar[dtype](c3))
    out.append(Scalar[dtype](c4)); out.append(Scalar[dtype](c5))
    out.append(Scalar[dtype](c6)); out.append(Scalar[dtype](c7))
    out.append(Scalar[dtype](c8))
    return out^


def cfg_for(num_envs: Int, num_particles: Int) -> SPOConfig:
    return SPOConfig(num_envs=num_envs, num_particles=num_particles,
                     num_actions=NUM_ACTIONS, state_dim=STATE_DIM,
                     search_depth=4, resample_period=4, temperature=0.5,
                     search_gamma=GAMMA, search_gae_lambda=1.0)


def constant_model(ctx: DeviceContext, max_batch: Int,
                   c: Scalar[dtype]) raises -> TicTacToeCritic:
    """A model whose critic always returns `c`, whatever the board."""
    model = TicTacToeCritic(ctx, max_batch, HIDDEN, Scalar[dtype](1))
    b3 = List[Scalar[dtype]](); b3.append(c)
    write_into[dtype](model.params.b3, b3)      # the rest is already zero
    ctx.synchronize()
    return model^


def test_eval_root_uses_the_network(ctx: DeviceContext) raises:
    """`eval_root` writes the network's V(s), not 0 and not whatever was there before.

    The output buffer is filled with 99 on purpose: if the model wrote nothing, the
    test would see 99 and fail. With `TicTacToe` (no critic) 0 would come out here.
    """
    num_envs = 3
    cfg = cfg_for(num_envs, 4)
    model = constant_model(ctx, cfg.num_search_particles(), Scalar[dtype](0.375))

    boards = List[Scalar[dtype]]()
    for b in board9(1,0,-1, 0,1,0, 0,0,0): boards.append(b)
    for b in board9(0,0,0, 0,0,0, 0,0,0): boards.append(b)
    for b in board9(1,1,-1, -1,-1,1, 1,-1,0): boards.append(b)
    root_state = upload[dtype](ctx, boards)

    logits = zeros[dtype](ctx, num_envs * NUM_ACTIONS)
    value = filled[dtype](ctx, num_envs, Scalar[dtype](99))
    model.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()

    got_value = download[dtype](value, num_envs)
    got_logits = download[dtype](logits, num_envs * NUM_ACTIONS)
    for e in range(num_envs):
        assert_close(got_value[e], Scalar[dtype](0.375), TOL,
                     String("V of the root of env ", e))
        # And the prior is still masked just as without a critic: adding the
        # value cannot have broken the part that already worked.
        for c in range(NUM_ACTIONS):
            want = Scalar[dtype](0) if boards[e * NUM_CELLS + c] == CELL_EMPTY \
                   else NEG_INF
            assert_close(got_logits[e * NUM_ACTIONS + c], want, TOL,
                         String("prior env ", e, " cell ", c))
    print("PASS eval_root takes V from the network and keeps the prior masked")


def run_step(ctx: DeviceContext, model: TicTacToeCritic, cfg: SPOConfig,
             boards: List[Scalar[dtype]], actions: List[Int],
             uniforms: List[Scalar[dtype]],
             depths: List[Int] = List[Int]()) raises -> List[Scalar[dtype]]:
    """One model step; returns [next_value..., discount...] concatenated."""
    p_total = cfg.num_search_particles()
    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    if len(depths) > 0:
        ds = List[Scalar[idx_dtype]]()
        for d in depths: ds.append(Scalar[idx_dtype](d))
        write_into[idx_dtype](particles.depth, ds)

    write_into[dtype](particles.state, boards)
    acts = List[Scalar[idx_dtype]]()
    for a in actions: acts.append(Scalar[idx_dtype](a))
    write_into[idx_dtype](outputs.next_action, acts)
    # next_value marked: if the model did not write it, the -7 would show.
    outputs.next_value.enqueue_fill(Scalar[dtype](-7))
    step_us = upload[dtype](ctx, uniforms)

    model.step(ctx, cfg, particles, outputs, step_us)
    ctx.synchronize()

    out = download[dtype](outputs.next_value, p_total)
    for d in download[dtype](outputs.discount, p_total): out.append(d)
    return out^


def test_bootstrap_formula(ctx: DeviceContext) raises:
    """The bootstrap is discount * search_gamma * V(s'), whether the game ends or not.

    Two particles on purpose:
      - number 0 stays alive -> discount 1 -> next_value = gamma * c
      - number 1 wins and ends -> discount 0 -> next_value = 0, even though V is c

    The second case is the one that really matters: if the discount were not folded
    in, a terminal particle would carry future value that does not exist and the
    search would overvalue the lines that end.
    """
    cfg = cfg_for(1, 2)
    c = Scalar[dtype](0.5)
    model = constant_model(ctx, cfg.num_search_particles(), c)

    boards = List[Scalar[dtype]]()
    # p0: empty board, plays 0 -> no win, the rival answers, it goes on.
    for b in board9(0,0,0, 0,0,0, 0,0,0): boards.append(b)
    # p1: X en 0 y 1, juega la 2 -> linea 0-1-2 -> victoria, terminal.
    for b in board9(1,1,0, -1,-1,0, 0,0,0): boards.append(b)

    actions = List[Int](); actions.append(0); actions.append(2)
    us = List[Scalar[dtype]]()
    us.append(Scalar[dtype](0.1)); us.append(Scalar[dtype](0.1))

    got = run_step(ctx, model, cfg, boards, actions, us)
    nv0 = got[0]; nv1 = got[1]; d0 = got[2]; d1 = got[3]

    assert_close(d0, Scalar[dtype](1), TOL, "particle 0 should still be alive")
    assert_close(d1, Scalar[dtype](0), TOL, "particle 1 should have ended")
    assert_close(nv0, GAMMA * c, TOL, "next_value of the live particle")
    assert_close(nv1, Scalar[dtype](0), TOL,
                 "next_value of a terminal particle has to be 0")
    print("PASS the bootstrap is discount * gamma * V(s') and respects the terminal")


def test_depth_discounted_bootstrap(ctx: DeviceContext) raises:
    """In the coherent mode the bootstrap carries gamma_r^(d+1), not `search_gamma`.

    It is the mode that fixes the scale mismatch: the reward the dynamics produces
    already comes with gamma_r^d, so the value has to bring gamma_r^(d+1) for
    summing them to mean anything. It is checked at several depths at once because
    the typical fault is getting one wrong: using d instead of d+1.

    Mind what must NOT happen: `search_gamma` (0.9 in these tests) cannot appear in
    the result. If it did, the model would be using the usual kernel and the new
    mode would be decorative.
    """
    n_p = 4
    cfg = cfg_for(1, n_p)
    c = Scalar[dtype](0.5)
    gamma_r = Scalar[dtype](0.7)

    model = TicTacToeCritic(ctx, cfg.num_search_particles(), HIDDEN, gamma_r,
                            depth_discounted=True)
    b3 = List[Scalar[dtype]](); b3.append(c)
    write_into[dtype](model.params.b3, b3)
    ctx.synchronize()

    boards = List[Scalar[dtype]]()
    actions = List[Int]()
    us = List[Scalar[dtype]]()
    depths = List[Int]()
    for p in range(n_p):
        for b in board9(0,0,0, 0,0,0, 0,0,0): boards.append(b)
        actions.append(p)                 # empty board: no move ends the game
        us.append(Scalar[dtype](0.1))
        depths.append(p)                  # profundidades 0, 1, 2, 3

    got = run_step(ctx, model, cfg, boards, actions, us, depths)
    for p in range(n_p):
        want = c
        for _ in range(p + 1):            # gamma^(d+1), a mano
            want *= gamma_r
        assert_close(got[p], want, TOL,
                     String("bootstrap at depth ", p))
    # 0.7^1*0.5 = 0.35, and not 0.9*0.5 = 0.45: search_gamma has not slipped in.
    assert_close(got[0], Scalar[dtype](0.35), TOL,
                 "at depth 0 the factor should be gamma_r, not search_gamma")
    print("PASS the coherent bootstrap discounts by depth with gamma_r^(d+1)")


def test_value_depends_on_the_board(ctx: DeviceContext) raises:
    """With real weights, two different boards give different Vs.

    It is what the constant-value test cannot see: if `eval_root` always encoded the
    same board (by misreading the buffer, or by not rewriting `obs`), V would come
    out identical for every env and with a constant b3 nobody would notice. Here the
    network DOES look at the input, so a repeated V gives the fault away.

    The weights are hand-made and simple: w1 takes each cell of the "mine" plane to
    a different neuron, and w3 sums them. With that, V(s) = number of my marks,
    which can be counted by looking at the board.
    """
    num_envs = 2
    cfg = cfg_for(num_envs, 2)
    model = TicTacToeCritic(ctx, cfg.num_search_particles(), HIDDEN,
                            Scalar[dtype](1))

    # w1 is [OBS_DIM, HIDDEN]: cell i of the "mine" plane activates neuron i.
    # There are only HIDDEN=8 neurons for 9 cells, so cell 8 is left out; it does
    # not matter, the test's boards do not use it.
    w1 = List[Scalar[dtype]]()
    for i in range(OBS_DIM):
        for h in range(HIDDEN):
            w1.append(Scalar[dtype](1) if (i < HIDDEN and i == h)
                      else Scalar[dtype](0))
    write_into[dtype](model.params.w1, w1)

    # w2 = identity, so that the activations pass intact into the second layer.
    w2 = List[Scalar[dtype]]()
    for i in range(HIDDEN):
        for h in range(HIDDEN):
            w2.append(Scalar[dtype](1) if i == h else Scalar[dtype](0))
    write_into[dtype](model.params.w2, w2)

    # w3 = all ones: V = sum of the activations = number of my marks.
    w3 = List[Scalar[dtype]]()
    for _ in range(HIDDEN): w3.append(Scalar[dtype](1))
    write_into[dtype](model.params.w3, w3)
    ctx.synchronize()

    boards = List[Scalar[dtype]]()
    for b in board9(1,1,1, 0,0,0, 0,0,0): boards.append(b)   # 3 fichas mias
    for b in board9(1,-1,0, -1,0,0, 0,0,0): boards.append(b) # 1 ficha mia
    root_state = upload[dtype](ctx, boards)

    logits = zeros[dtype](ctx, num_envs * NUM_ACTIONS)
    value = filled[dtype](ctx, num_envs, Scalar[dtype](99))
    model.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()

    got = download[dtype](value, num_envs)
    assert_close(got[0], Scalar[dtype](3), TOL, "V of the board with 3 of my pieces")
    assert_close(got[1], Scalar[dtype](1), TOL, "V of the board with 1 of my pieces")
    print("PASS V depends on the board: the observation reaches the network intact")


def test_sync_from_copies_the_weights(ctx: DeviceContext) raises:
    """`sync_from` brings in another critic's weights and changes what the model predicts.

    Without this there would be no way to get the TRAINED critic into the search:
    the model would keep the constructor's zeros and V would be 0, that is, the
    usual planner in disguise.
    """
    cfg = cfg_for(1, 2)
    model = constant_model(ctx, cfg.num_search_particles(), Scalar[dtype](0.1))

    src = zero_critic_params(ctx, OBS_DIM, HIDDEN, 1)
    b3 = List[Scalar[dtype]](); b3.append(Scalar[dtype](0.8))
    write_into[dtype](src.b3, b3)
    ctx.synchronize()

    model.sync_from(ctx, src)
    ctx.synchronize()

    boards = board9(1,0,-1, 0,1,0, 0,0,0)
    root_state = upload[dtype](ctx, boards)
    logits = zeros[dtype](ctx, NUM_ACTIONS)
    value = filled[dtype](ctx, 1, Scalar[dtype](99))
    model.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()

    got = download[dtype](value, 1)
    assert_close(got[0], Scalar[dtype](0.8), TOL,
                 "after sync_from, V should come from the copied weights")

    # And the copy is independent: touching the source afterwards does not move the model.
    b3b = List[Scalar[dtype]](); b3b.append(Scalar[dtype](-0.4))
    write_into[dtype](src.b3, b3b)
    ctx.synchronize()
    model.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()
    got2 = download[dtype](value, 1)
    assert_close(got2[0], Scalar[dtype](0.8), TOL,
                 "the model has its own copy; the source no longer affects it")

    # And an incompatible shape is rejected instead of copying garbage.
    bad = zero_critic_params(ctx, OBS_DIM, HIDDEN + 1, 1)
    failed = False
    try:
        model.sync_from(ctx, bad)
    except:
        failed = True
    if not failed:
        raise Error("sync_from should reject a critic with a different shape")
    print("PASS sync_from copies the weights, is independent and validates the shape")


def test_many_particles_multi_block(ctx: DeviceContext) raises:
    """More than one block and a non-round size: 5 envs x 13 particles = 65.

    The kernels in this file carry guards, but a guard is only exercised if it is
    ever launched with a size that does not line up with the block. Every test above
    uses tiny sizes that fit in one block, so without this case the `if p < n` would
    never really be tested.
    """
    cfg = cfg_for(5, 13)
    p_total = cfg.num_search_particles()          # 65
    c = Scalar[dtype](0.25)
    model = constant_model(ctx, p_total, c)

    boards = List[Scalar[dtype]]()
    actions = List[Int]()
    us = List[Scalar[dtype]]()
    for p in range(p_total):
        # Empty boards: none ends in one turn, so discount = 1 in all of them and
        # next_value has to be gamma*c across all 65.
        for b in board9(0,0,0, 0,0,0, 0,0,0): boards.append(b)
        actions.append(p % 9)
        us.append(Scalar[dtype](0.1))

    got = run_step(ctx, model, cfg, boards, actions, us)
    for p in range(p_total):
        assert_close(got[p_total + p], Scalar[dtype](1), TOL,
                     String("discount of particle ", p))
        assert_close(got[p], GAMMA * c, TOL,
                     String("next_value of particle ", p))
    print("PASS with 65 particles (several blocks, non-round size) it comes out the same")


def main() raises:
    with DeviceContext() as ctx:
        test_eval_root_uses_the_network(ctx)
        test_bootstrap_formula(ctx)
        test_depth_discounted_bootstrap(ctx)
        test_value_depends_on_the_board(ctx)
        test_sync_from_copies_the_weights(ctx)
        test_many_particles_multi_block(ctx)
