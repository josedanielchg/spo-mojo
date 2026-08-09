"""The full SMC search over Tic-Tac-Toe: BEHAVIOUR tests.

It is test_search.mojo's equivalent but with the TTT model: nothing is dictated,
`search[TicTacToe]()` is run end to end and what comes out is inspected. The first
integration of everything from phase A with the SMC core.

What gets checked, from most basic to most demanding:
  1. the search runs and produces no NaN,
  2. the chosen action is ALWAYS legal (the masked prior does its job),
  3. same seed -> same search, bit for bit,
  4. the ESS stays in the healthy range [1, N].

What is NOT checked here: whether the search PLAYS well. That needs real games
against an opponent, and it belongs to phase A' (the real loop and the demo).
"""

from std.gpu.host import DeviceContext
from std.math import isnan

from ops.common import dtype, idx_dtype
from envs.tictactoe import (TicTacToe, default_tictactoe, NUM_CELLS,
                            NUM_ACTIONS, STATE_DIM, CELL_EMPTY)
from systems.spo.particles import SearchWorkspace
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from tests.helpers import upload, download, assert_close

comptime TOL = Scalar[dtype](1e-5)


def make_config(num_envs: Int, num_particles: Int, depth: Int,
                period: Int) -> SPOConfig:
    """Search config for TTT: 9 actions, 9 state floats."""
    return SPOConfig(
        num_envs=num_envs, num_particles=num_particles, num_actions=NUM_ACTIONS,
        state_dim=STATE_DIM, search_depth=depth, resample_period=period,
        temperature=0.5, search_gamma=1.0, search_gae_lambda=1.0)


def board9(c0: Int, c1: Int, c2: Int, c3: Int, c4: Int,
           c5: Int, c6: Int, c7: Int, c8: Int) -> List[Scalar[dtype]]:
    """A readable board: 9 cell codes (1=X agent, -1=O rival, 0=empty)."""
    out = List[Scalar[dtype]]()
    out.append(Scalar[dtype](c0)); out.append(Scalar[dtype](c1))
    out.append(Scalar[dtype](c2)); out.append(Scalar[dtype](c3))
    out.append(Scalar[dtype](c4)); out.append(Scalar[dtype](c5))
    out.append(Scalar[dtype](c6)); out.append(Scalar[dtype](c7))
    out.append(Scalar[dtype](c8))
    return out^


def mixed_roots(num_envs: Int) -> List[Scalar[dtype]]:
    """Varied root states: empty, mid-game and nearly full, in rotation.

    Mixing positions is worthwhile because each exercises a different path: from
    the empty board the search has 9 actions and depth to spare; from a nearly full
    one almost every particle reaches terminal in one or two steps.
    """
    roots = List[Scalar[dtype]]()
    for e in range(num_envs):
        which = e % 3
        b = board9(0,0,0, 0,0,0, 0,0,0)              # empty: X's turn
        if which == 1:
            b = board9(1,0,-1, 0,1,0, -1,0,0)         # mid-game
        elif which == 2:
            b = board9(1,-1,1, -1,1,0, -1,1,0)        # nearly full: only 5 and 8 free
        for c in range(NUM_CELLS):
            roots.append(b[c])
    return roots^


def test_search_runs_without_nan(ctx: DeviceContext) raises:
    """The search runs end to end and no output carries a NaN.

    It is the real smoke test: with the masked prior there is NEG_INF in the logits
    of the occupied cells, and that is precisely where a NaN would sneak in
    (exp(-inf - -inf) in the softmax) if the masking convention were wrong.
    """
    cfg = make_config(num_envs=6, num_particles=16, depth=6, period=3)
    model = default_tictactoe()
    ws = SearchWorkspace(ctx, cfg)
    p_total = cfg.num_search_particles()

    search[TicTacToe](ctx, ws, cfg, model,
                      upload[dtype](ctx, mixed_roots(cfg.num_envs)), UInt32(7))
    ctx.synchronize()

    weights = download[dtype](ws.output.sampled_action_weights, p_total)
    advantages = download[dtype](ws.output.sampled_advantages, p_total)
    values = download[dtype](ws.output.value, cfg.num_envs)
    ess = download[dtype](ws.output.ess, cfg.search_depth * cfg.num_envs)

    for p in range(p_total):
        if isnan(weights[p]):
            raise Error("NaN weight at particle ", p)
        if isnan(advantages[p]):
            raise Error("NaN advantage at particle ", p)
    for e in range(cfg.num_envs):
        if isnan(values[e]):
            raise Error("NaN value in env ", e)
    for i in range(len(ess)):
        if isnan(ess[i]):
            raise Error("NaN ESS at index ", i)

    # The weights are a per-env softmax: they have to sum to 1.
    for e in range(cfg.num_envs):
        total = Scalar[dtype](0)
        for n in range(cfg.num_particles):
            total += weights[e * cfg.num_particles + n]
        assert_close(total, 1.0, Scalar[dtype](1e-4),
                     String("the weights of env ", e, " should sum to 1"))
    print("PASS the search runs on TTT with no NaN and the weights sum to 1")


def test_search_only_picks_legal_actions(ctx: DeviceContext) raises:
    """THE phase-A test: the chosen action never lands on an occupied cell.

    It is what justifies the masked prior. Both the final action per env and ALL
    the sampled root actions are checked: if a single particle could have chosen an
    occupied cell, the masking would not be working.
    """
    cfg = make_config(num_envs=6, num_particles=16, depth=6, period=3)
    model = default_tictactoe()
    ws = SearchWorkspace(ctx, cfg)
    p_total = cfg.num_search_particles()

    roots = mixed_roots(cfg.num_envs)
    search[TicTacToe](ctx, ws, cfg, model, upload[dtype](ctx, roots), UInt32(99))
    ctx.synchronize()

    final_action = download[idx_dtype](ws.output.action, cfg.num_envs)
    sampled = download[idx_dtype](ws.output.sampled_actions, p_total)

    for e in range(cfg.num_envs):
        a = Int(final_action[e])
        if a < 0 or a >= NUM_ACTIONS:
            raise Error("action out of range in env ", e, ": ", a)
        if roots[e * NUM_CELLS + a] != CELL_EMPTY:
            raise Error("env ", e, " chose the OCCUPIED cell ", a)

    for e in range(cfg.num_envs):
        for n in range(cfg.num_particles):
            a = Int(sampled[e * cfg.num_particles + n])
            if a < 0 or a >= NUM_ACTIONS:
                raise Error("root action out of range in env ", e, ": ", a)
            if roots[e * NUM_CELLS + a] != CELL_EMPTY:
                raise Error("particle ", n, " of env ", e,
                            " sampled the OCCUPIED cell ", a)
    print("PASS every action (final and sampled) is legal")


def test_search_is_reproducible(ctx: DeviceContext) raises:
    """Same seed, same search, even though the model is stochastic.

    It matters more than in the toy problem: TTT's step draws the rival's move, so
    if the step's RNG stream were not deterministic the result would change between
    runs.
    """
    cfg = make_config(num_envs=4, num_particles=16, depth=6, period=3)
    model = default_tictactoe()
    p_total = cfg.num_search_particles()
    roots = mixed_roots(cfg.num_envs)

    first = List[Scalar[dtype]]()
    first_actions = List[Scalar[idx_dtype]]()
    for run in range(2):
        ws = SearchWorkspace(ctx, cfg)
        search[TicTacToe](ctx, ws, cfg, model, upload[dtype](ctx, roots),
                          UInt32(31337))
        ctx.synchronize()
        w = download[dtype](ws.output.sampled_action_weights, p_total)
        a = download[idx_dtype](ws.output.action, cfg.num_envs)
        if run == 0:
            first = w^
            first_actions = a^
        else:
            for p in range(p_total):
                assert_close(w[p], first[p], TOL,
                             String("two searches with the same seed differ at ", p))
            for e in range(cfg.num_envs):
                if Int(a[e]) != Int(first_actions[e]):
                    raise Error("the action of env ", e, " changed between runs")
    print("PASS the search on TTT is reproducible (random opponent included)")


def test_ess_stays_in_range(ctx: DeviceContext) raises:
    """The ESS stays in [1, N] at every depth and env.

    I do not demand the toy problem's sawtooth: in TTT many particles reach
    terminal and freeze their weight, so the curve's shape depends heavily on the
    starting position. What always has to hold is the range: the ESS of N particles
    cannot be less than 1 nor greater than N.
    """
    cfg = make_config(num_envs=6, num_particles=16, depth=6, period=3)
    model = default_tictactoe()
    ws = SearchWorkspace(ctx, cfg)

    search[TicTacToe](ctx, ws, cfg, model,
                      upload[dtype](ctx, mixed_roots(cfg.num_envs)), UInt32(2718))
    ctx.synchronize()

    ess = download[dtype](ws.output.ess, cfg.search_depth * cfg.num_envs)
    lo = Scalar[dtype](1) - Scalar[dtype](1e-4)
    hi = Scalar[dtype](cfg.num_particles) + Scalar[dtype](1e-4)

    print("      mean ESS by depth:")
    for d in range(cfg.search_depth):
        total = Scalar[dtype](0)
        for e in range(cfg.num_envs):
            v = ess[d * cfg.num_envs + e]
            if v < lo or v > hi:
                raise Error("ESS out of range at depth ", d, " env ", e, ": ", v)
            total += v
        print("        depth", d, "->", total / Scalar[dtype](cfg.num_envs))
    print("PASS the ESS stays in [1,", cfg.num_particles, "] at every depth")


def main() raises:
    with DeviceContext() as ctx:
        test_search_runs_without_nan(ctx)
        test_search_only_picks_legal_actions(ctx)
        test_search_is_reproducible(ctx)
        test_ess_stays_in_range(ctx)
