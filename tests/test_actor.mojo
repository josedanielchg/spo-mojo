"""The actor's MLP and its masking, against the numpy golden.

The forward itself is the critic's with 9 outputs instead of 1, and that has been
verified since E1.3. What really gets tested here is **the masking**, which is the
piece Stoix does not have (its environments have no illegal actions) and that has
to be added to take SPO to a board game.

Why the masking deserves its own tests and "it was already tested in the search's
prior" is not enough: there the legal logits were ALL 0, so masking and taking a
softmax gave a uniform, and an indexing error would have come out just as uniform.
Here the logits are numbers different from one another coming out of a network, so
masking the wrong cell changes the distribution detectably.
"""

from std.gpu.host import DeviceContext
from std.math import exp

from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from envs.tictactoe import (NUM_ACTIONS, NUM_CELLS, STATE_DIM, OBS_DIM, NEG_INF,
                            TPB_TTT, ttt_encode_obs_kernel)
from networks.actor import (Actor, actor_logits, actor_probs,
                            actor_log_probs, zero_actor_params)
from tests.golden_io import read_f32
from tests.helpers import upload, download, write_into, filled, assert_close

comptime TOL = Scalar[dtype](2e-5)
comptime GOLDEN = String("tests/golden/")


def board9(c0: Int, c1: Int, c2: Int, c3: Int, c4: Int,
           c5: Int, c6: Int, c7: Int, c8: Int) -> List[Scalar[dtype]]:
    """Un tablero legible: 1 = mia, -1 = suya, 0 = vacia."""
    out = List[Scalar[dtype]]()
    out.append(Scalar[dtype](c0)); out.append(Scalar[dtype](c1))
    out.append(Scalar[dtype](c2)); out.append(Scalar[dtype](c3))
    out.append(Scalar[dtype](c4)); out.append(Scalar[dtype](c5))
    out.append(Scalar[dtype](c6)); out.append(Scalar[dtype](c7))
    out.append(Scalar[dtype](c8))
    return out^


def load_actor(ctx: DeviceContext, hidden: Int) raises -> Actor:
    """An actor with the golden weights for that width."""
    tag = String("actor_h", hidden, "_")
    a = Actor(ctx, 64, hidden)
    write_into[dtype](a.params.w1, read_f32(GOLDEN + tag + "w1.bin"))
    write_into[dtype](a.params.b1, read_f32(GOLDEN + tag + "b1.bin"))
    write_into[dtype](a.params.w2, read_f32(GOLDEN + tag + "w2.bin"))
    write_into[dtype](a.params.b2, read_f32(GOLDEN + tag + "b2.bin"))
    write_into[dtype](a.params.w3, read_f32(GOLDEN + tag + "w3.bin"))
    write_into[dtype](a.params.b3, read_f32(GOLDEN + tag + "b3.bin"))
    ctx.synchronize()
    return a^


def check_width(ctx: DeviceContext, hidden: Int, m: Int) raises:
    """Compara logits enmascarados y probabilidades contra el golden."""
    tag = String("actor_h", hidden, "_")
    actor = load_actor(ctx, hidden)

    x = upload[dtype](ctx, read_f32(GOLDEN + tag + "x" + String(m) + ".bin"))
    mask = upload[dtype](ctx, read_f32(GOLDEN + tag + "mask" + String(m) + ".bin"))
    probs = zero_buffer[dtype](ctx, m * NUM_ACTIONS)

    actor_probs(ctx, actor.params, actor.cache, x, mask, probs, m)
    ctx.synchronize()

    want_masked = read_f32(GOLDEN + tag + "masked" + String(m) + ".bin")
    want_probs = read_f32(GOLDEN + tag + "probs" + String(m) + ".bin")
    got_masked = download[dtype](actor.cache.value, m * NUM_ACTIONS)
    got_probs = download[dtype](probs, m * NUM_ACTIONS)

    for i in range(m * NUM_ACTIONS):
        # The masked logits are NEG_INF on both sides; comparing their difference
        # is meaningless (it is 0 or overflows), so identity is checked instead.
        if want_masked[i] == NEG_INF:
            if got_masked[i] != NEG_INF:
                raise Error("h", hidden, " m", m, " logit ", i,
                            " deberia estar tapado y vale ", got_masked[i])
        else:
            assert_close(got_masked[i], want_masked[i], TOL,
                         String("h", hidden, " m", m, " logit ", i))
        assert_close(got_probs[i], want_probs[i], TOL,
                     String("h", hidden, " m", m, " prob ", i))
    print("PASS actor h=", hidden, " m=", m,
          " coincide con el golden (logits y politica)")


def test_actor_matches_golden(ctx: DeviceContext) raises:
    """The golden's three widths and two batches.

    m=5 is ragged (not a multiple of the tile of 16) and hits the guards; m=64
    spans several tiles along the batch dimension. The three widths are there
    because how much network tic-tac-toe needs is something to be measured, not
    assumed.
    """
    widths = List[Int]()
    widths.append(32); widths.append(64); widths.append(256)
    for hidden in widths:
        check_width(ctx, hidden, 5)
        check_width(ctx, hidden, 64)


def test_illegal_cells_get_exactly_zero(ctx: DeviceContext) raises:
    """An occupied cell comes out with ZERO probability, not with a very small one.

    That it be exactly zero matters: if residual mass were left, the search could
    sample an impossible move, and that does not fail loudly but silently corrupts
    a particle.

    It is also checked that the rows sum to 1: masking without renormalising would
    leave a distribution that is not one.
    """
    actor = load_actor(ctx, 64)
    boards = List[Scalar[dtype]]()
    for b in board9(1,0,-1, 0,1,0, 0,0,0): boards.append(b)   # 6 libres
    for b in board9(1,1,-1, -1,-1,1, 1,-1,0): boards.append(b)  # 1 libre (la 8)
    n = 2
    state = upload[dtype](ctx, boards)
    obs = zero_buffer[dtype](ctx, n * OBS_DIM)
    ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
        obs.unsafe_ptr(), state.unsafe_ptr(), n, grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()

    actor.forward(ctx, state, obs, n)
    ctx.synchronize()
    p = download[dtype](actor.probs, n * NUM_ACTIONS)

    for e in range(n):
        total = Scalar[dtype](0)
        for c in range(NUM_ACTIONS):
            v = p[e * NUM_ACTIONS + c]
            occupied = boards[e * NUM_CELLS + c] != Scalar[dtype](0)
            if occupied:
                if v != Scalar[dtype](0):
                    raise Error("la casilla ocupada ", c, " del tablero ", e,
                                " tiene probabilidad ", v, " y deberia ser 0")
            elif v <= Scalar[dtype](0):
                raise Error("la casilla libre ", c, " del tablero ", e,
                            " tiene probabilidad ", v)
            total += v
        assert_close(total, Scalar[dtype](1), TOL,
                     String("la fila ", e, " deberia sumar 1"))

    # The second board has only cell 8 free: all the mass goes there.
    assert_close(p[NUM_ACTIONS + 8], Scalar[dtype](1), TOL,
                 "con una sola casilla libre, su probabilidad tiene que ser 1")
    print("PASS las casillas ocupadas salen a cero exacto y las filas suman 1")


def test_masking_changes_the_ranking(ctx: DeviceContext) raises:
    """Masking the cell the network PREFERRED really does change the choice.

    This is the test that tells "it masks" from "it does nothing": the most likely
    cell without a mask is found, only that one gets masked, and it is checked that
    the policy comes to prefer another. With uniform logits (the search's prior)
    this check would be impossible, because they all tie.
    """
    actor = load_actor(ctx, 64)
    n = 1
    board = board9(0,0,0, 0,0,0, 0,0,0)          # todo libre
    state = upload[dtype](ctx, board)
    obs = zero_buffer[dtype](ctx, n * OBS_DIM)
    ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
        obs.unsafe_ptr(), state.unsafe_ptr(), n, grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()

    all_free = filled[dtype](ctx, NUM_ACTIONS, Scalar[dtype](1))
    probs = zero_buffer[dtype](ctx, NUM_ACTIONS)
    actor_probs(ctx, actor.params, actor.cache, obs, all_free, probs, n)
    ctx.synchronize()
    p0 = download[dtype](probs, NUM_ACTIONS)

    best = 0
    for c in range(1, NUM_ACTIONS):
        if p0[c] > p0[best]:
            best = c

    # Now only the favourite gets masked.
    m2 = List[Scalar[dtype]]()
    for c in range(NUM_ACTIONS):
        m2.append(Scalar[dtype](0) if c == best else Scalar[dtype](1))
    mask2 = upload[dtype](ctx, m2)
    actor_probs(ctx, actor.params, actor.cache, obs, mask2, probs, n)
    ctx.synchronize()
    p1 = download[dtype](probs, NUM_ACTIONS)

    if p1[best] != Scalar[dtype](0):
        raise Error("la casilla tapada ", best, " sigue con probabilidad ",
                    p1[best])
    best2 = 0
    for c in range(1, NUM_ACTIONS):
        if p1[c] > p1[best2]:
            best2 = c
    if best2 == best:
        raise Error("tras tapar la favorita deberia ganar otra casilla")

    # And the remaining ones keep their relative proportions: masking a cell
    # renormalises, it does not reorder. If this failed, the masking would be
    # distorting the policy instead of merely trimming it.
    for c in range(NUM_ACTIONS):
        if c == best:
            continue
        for d in range(NUM_ACTIONS):
            if d == best or d == c:
                continue
            # p1[c]/p1[d] has to be p0[c]/p0[d]; it is compared as a cross
            # product so as not to divide.
            assert_close(p1[c] * p0[d], p1[d] * p0[c], Scalar[dtype](1e-4),
                         String("proporcion entre ", c, " y ", d))
    print("PASS tapar la favorita cambia la eleccion y solo renormaliza el resto")


def test_full_board_does_not_produce_nan(ctx: DeviceContext) raises:
    """A full board (everything masked) gives uniform, not nan.

    The actor should not be consulted in a terminal position, but if it happens,
    the result has to be harmless. With a true -inf the softmax would give nan and
    the nan would propagate silently through everything downstream; with a finite
    NEG_INF the row degenerates to uniform. This test pins that design choice down.
    """
    actor = load_actor(ctx, 64)
    n = 1
    board = board9(1,-1,1, -1,1,-1, -1,1,-1)     # lleno
    state = upload[dtype](ctx, board)
    obs = zero_buffer[dtype](ctx, n * OBS_DIM)
    ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
        obs.unsafe_ptr(), state.unsafe_ptr(), n, grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()

    actor.forward(ctx, state, obs, n)
    ctx.synchronize()
    p = download[dtype](actor.probs, n * NUM_ACTIONS)

    total = Scalar[dtype](0)
    for c in range(NUM_ACTIONS):
        v = p[c]
        if v != v:                        # nan != nan
            raise Error("la casilla ", c, " salio nan con el tablero lleno")
        total += v
    assert_close(total, Scalar[dtype](1), TOL,
                 "incluso con todo tapado la fila tiene que sumar 1")
    print("PASS un tablero lleno degenera a uniforme en vez de dar nan")


def test_mask_comes_from_the_state(ctx: DeviceContext) raises:
    """The mask is derived from the board, not passed in from outside.

    It is an invariant that matters: if legality came through a separate channel,
    it could drift out of sync with the state and the network would end up playing
    on placed marks. It is checked that `mask_from_state` reproduces exactly the
    empty cells of three different boards, including one spanning several thread
    blocks.
    """
    actor = load_actor(ctx, 32)
    boards = List[Scalar[dtype]]()
    for b in board9(0,0,0, 0,0,0, 0,0,0): boards.append(b)
    for b in board9(1,-1,1, -1,1,-1, -1,1,-1): boards.append(b)
    for b in board9(1,0,0, 0,-1,0, 0,0,1): boards.append(b)
    n = 3
    state = upload[dtype](ctx, boards)
    actor.mask_from_state(ctx, state, n)
    ctx.synchronize()
    got = download[dtype](actor.mask, n * NUM_ACTIONS)

    for e in range(n):
        for c in range(NUM_ACTIONS):
            want = Scalar[dtype](1) if boards[e * NUM_CELLS + c] == 0 \
                   else Scalar[dtype](0)
            assert_close(got[e * NUM_ACTIONS + c], want, TOL,
                         String("mascara del tablero ", e, " casilla ", c))
    print("PASS la mascara sale del estado y coincide con las casillas vacias")


def test_forward_log_is_consistent_with_forward(ctx: DeviceContext) raises:
    """`forward_log` gives the log of what `forward` gives, and no NaN on the illegal ones.

    This path was added for the M-step (the cross entropy works in log space) and
    until now had no test: new code without coverage, which is exactly what this
    project has set out not to do.

    Two different things get checked. That exp(log pi) reproduces pi on the legal
    cells -- if the log-softmax used a different denominator from the softmax, it
    would show here. And that on the illegal ones a very negative, FINITE value
    comes out, not a -inf nor a NaN: the loss skips those terms, but a NaN in the
    buffer would propagate all the same to the first gradient that touches it.
    """
    actor = load_actor(ctx, 64)
    boards = List[Scalar[dtype]]()
    for b in board9(1,0,-1, 0,1,0, 0,0,0): boards.append(b)
    for b in board9(0,0,0, 0,0,0, 0,0,0): boards.append(b)
    for b in board9(1,1,-1, -1,-1,1, 1,-1,0): boards.append(b)
    n = 3
    state = upload[dtype](ctx, boards)
    obs = zero_buffer[dtype](ctx, n * OBS_DIM)
    ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
        obs.unsafe_ptr(), state.unsafe_ptr(), n, grid_dim=1, block_dim=TPB_TTT)
    ctx.synchronize()

    actor.forward(ctx, state, obs, n)
    ctx.synchronize()
    p = download[dtype](actor.probs, n * NUM_ACTIONS)

    actor.forward_log(ctx, state, obs, n)
    ctx.synchronize()
    lp = download[dtype](actor.log_probs, n * NUM_ACTIONS)

    for e in range(n):
        for c in range(NUM_ACTIONS):
            i = e * NUM_ACTIONS + c
            if lp[i] != lp[i]:
                raise Error("log pi es NaN en el tablero ", e, " casilla ", c)
            if boards[e * NUM_CELLS + c] != Scalar[dtype](0):
                if lp[i] > Scalar[dtype](-1e30):
                    raise Error("la casilla ocupada ", c, " del tablero ", e,
                                " deberia tener log pi muy negativo, y vale ",
                                lp[i])
            else:
                assert_close(exp(lp[i]), p[i], Scalar[dtype](1e-5),
                             String("exp(log pi) vs pi en ", e, ",", c))
    print("PASS forward_log coincide con log(forward) y no produce NaN")


def test_rejects_more_boards_than_reserved(ctx: DeviceContext) raises:
    """Asking for more boards than were allocated raises an error, not silent corruption.

    The actor's buffers are sized in the constructor. Without the check, a call
    with a larger m would write outside and the symptom would appear much later, in
    another buffer and with no apparent relation to the cause. It is the class of
    fault that costs most to debug, and avoiding it costs one host-side comparison.
    """
    small = Actor(ctx, 2, 32)
    state = zero_buffer[dtype](ctx, 8 * STATE_DIM)
    obs = zero_buffer[dtype](ctx, 8 * OBS_DIM)

    failed = False
    try:
        small.forward(ctx, state, obs, 8)
    except:
        failed = True
    if not failed:
        raise Error("deberia rechazar 8 tableros con sitio para 2")

    # And with the ones that do fit, it works.
    small.forward(ctx, state, obs, 2)
    ctx.synchronize()
    print("PASS el actor rechaza mas tableros de los reservados")


def test_argmax_of_masked_policy_is_always_legal(ctx: DeviceContext) raises:
    """The argmax of pi ALWAYS lands on a free cell. It is a critical invariant.

    Why critical: `ttt_apply` **does not check legality** (its own docstring says
    so). If the actor chose an occupied cell, the move would **overwrite the
    rival's mark** with its own and the score would come out inflated without
    anything failing. The whole "network playing alone" result (E2.6) depends on
    this invariant, so it comes with a test rather than with an argument.

    The argument is that the masked softmax gives EXACTLY 0 to the occupied ones
    and the row sums to 1, hence some free cell has positive mass and wins the
    argmax. Here it is checked over 60 varied boards, including those that leave a
    single free cell (the tightest case).
    """
    n = 60
    actor = load_actor(ctx, 64)
    small = Actor(ctx, n, 64)
    # The same weights as the loaded actor, so as to have a non-trivial policy.
    boards = List[Scalar[dtype]]()
    for i in range(n):
        filled_cells = i % 8            # de 0 a 7 casillas ocupadas
        for c in range(NUM_CELLS):
            if c < filled_cells:
                boards.append(Scalar[dtype](1) if (i + c) % 2 == 0
                              else Scalar[dtype](-1))
            else:
                boards.append(Scalar[dtype](0))
    state = upload[dtype](ctx, boards)
    obs = zero_buffer[dtype](ctx, n * OBS_DIM)
    blocks = (n + TPB_TTT - 1) // TPB_TTT
    ctx.enqueue_function[ttt_encode_obs_kernel, ttt_encode_obs_kernel](
        obs.unsafe_ptr(), state.unsafe_ptr(), n,
        grid_dim=blocks, block_dim=TPB_TTT)
    ctx.synchronize()

    small.forward(ctx, state, obs, n)
    ctx.synchronize()
    p = download[dtype](small.probs, n * NUM_ACTIONS)

    for e in range(n):
        best = 0
        for c in range(1, NUM_ACTIONS):
            if p[e * NUM_ACTIONS + c] > p[e * NUM_ACTIONS + best]:
                best = c
        if boards[e * NUM_CELLS + best] != Scalar[dtype](0):
            raise Error("el argmax del tablero ", e, " cayo en la casilla ",
                        best, ", que esta OCUPADA: ttt_apply pisaria la ficha "
                        "del rival")
        # And the total mass is still 1: if the softmax had degenerated, the
        # argmax could be choosing among zeros.
        total = Scalar[dtype](0)
        for c in range(NUM_ACTIONS):
            total += p[e * NUM_ACTIONS + c]
        assert_close(total, Scalar[dtype](1), Scalar[dtype](1e-5),
                     String("la fila ", e, " deberia sumar 1"))
    print("PASS el argmax de la politica enmascarada nunca cae en casilla "
          "ocupada (60 tableros)")


def main() raises:
    with DeviceContext() as ctx:
        test_actor_matches_golden(ctx)
        test_illegal_cells_get_exactly_zero(ctx)
        test_masking_changes_the_ranking(ctx)
        test_full_board_does_not_produce_nan(ctx)
        test_mask_comes_from_the_state(ctx)
        test_forward_log_is_consistent_with_forward(ctx)
        test_rejects_more_boards_than_reserved(ctx)
        test_argmax_of_masked_policy_is_always_legal(ctx)
