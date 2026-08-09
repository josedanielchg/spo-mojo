"""The trajectory replay buffer: FIFO, fields and sampling.

There is no interesting maths here, but there is a typical way of breaking
silently: losing or mixing up sequences when wrapping around. So it is checked with
marked values and exact numbers, not with tolerances.
"""

from ops.common import dtype
from rl_utils.buffer import TrajectoryBuffer

comptime T_LEN = 4
comptime OBS_DIM = 3
comptime N_ACT = 4


def seq_obs(value: Int) -> List[Scalar[dtype]]:
    """A sequence of observations marked with `value`, so as to recognise them."""
    out = List[Scalar[dtype]]()
    for i in range(T_LEN * OBS_DIM):
        out.append(Scalar[dtype](value * 100 + i))
    return out^


def q_for(value: Int) -> List[Scalar[dtype]]:
    """A q marked with `value`, so as to recognise it on the way out. At module
    level: a nested function cannot capture variables from the enclosing scope."""
    out = List[Scalar[dtype]]()
    for i in range(T_LEN * N_ACT):
        out.append(Scalar[dtype](value * 1000 + i))
    return out^


def seq_steps(value: Int) -> List[Scalar[dtype]]:
    out = List[Scalar[dtype]]()
    for i in range(T_LEN):
        out.append(Scalar[dtype](value * 10 + i))
    return out^


def add_marked(mut buf: TrajectoryBuffer, value: Int) raises:
    """Puts in a whole sequence marked with `value`."""
    buf.add(seq_obs(value), seq_steps(value), seq_steps(value),
            seq_steps(value), seq_obs(value + 1000))


def test_buffer_fifo_and_wraparound() raises:
    """Once full, the new sequences overwrite the oldest ones, in order.

    With capacity 3 and five sequences put in (1..5), the last three (3, 4, 5) have
    to remain, and in the right slots. It is where a badly written ring buffer
    loses data or leaves it out of order.
    """
    buf = TrajectoryBuffer(3, T_LEN, OBS_DIM)
    if buf.size() != 0:
        raise Error("a freshly created buffer should be empty")

    for v in range(1, 4):
        add_marked(buf, v)
    if buf.size() != 3 or not buf.is_full():
        raise Error("after 3 sequences with capacity 3 it should be full")

    # Two more: they overwrite 1 and 2.
    add_marked(buf, 4)
    add_marked(buf, 5)
    if buf.size() != 3:
        raise Error("the capacity cannot grow: ", buf.size())

    # Slot 0 now holds 4, slot 1 holds 5, and slot 2 still holds 3.
    want = List[Int](); want.append(4); want.append(5); want.append(3)
    for slot in range(3):
        idx = List[Int](); idx.append(slot)
        got = buf.gather(idx)
        expected = seq_obs(want[slot])
        for i in range(T_LEN * OBS_DIM):
            if got[i] != expected[i]:
                raise Error("slot ", slot, " should hold sequence ",
                            want[slot], " but the value ", i, " is ", got[i])
    print("PASS the buffer is FIFO and wraps without mixing sequences")


def test_buffer_fields_dont_cross() raises:
    """Each field is stored in its own: reward, done and truncated do not mix.

    They go in with different values on purpose so that a crossover shows.
    """
    buf = TrajectoryBuffer(2, T_LEN, OBS_DIM)
    r = List[Scalar[dtype]](); d = List[Scalar[dtype]](); tr = List[Scalar[dtype]]()
    for i in range(T_LEN):
        r.append(Scalar[dtype](i) + 0.5)      # 0.5, 1.5, 2.5, 3.5
        d.append(Scalar[dtype](1) if i == T_LEN - 1 else Scalar[dtype](0))
        tr.append(Scalar[dtype](0))
    buf.add(seq_obs(7), r, d, tr, seq_obs(8))

    idx = List[Int](); idx.append(0)
    got_r = buf.gather_steps(idx, 0)
    got_d = buf.gather_steps(idx, 1)
    got_t = buf.gather_steps(idx, 2)
    for i in range(T_LEN):
        if got_r[i] != r[i]:
            raise Error("reward[", i, "] came out ", got_r[i], " and was ", r[i])
        if got_d[i] != d[i]:
            raise Error("done[", i, "] came out ", got_d[i], " and was ", d[i])
        if got_t[i] != tr[i]:
            raise Error("truncated[", i, "] came out ", got_t[i])

    # And the bootstrap_obs are not the obs.
    got_obs = buf.gather(idx)
    got_boot = buf.gather_bootstrap(idx)
    if got_obs[0] == got_boot[0]:
        raise Error("obs and bootstrap_obs should not match in this setup")
    print("PASS the buffer's fields do not cross over into each other")


def test_buffer_sampling_is_deterministic() raises:
    """Same seed, same indices; and always within what is stored."""
    buf = TrajectoryBuffer(8, T_LEN, OBS_DIM)
    for v in range(1, 6):        # 5 sequences, capacity 8: it does not wrap
        add_marked(buf, v)

    a = buf.sample_indices(20, UInt32(123), UInt32(0))
    b = buf.sample_indices(20, UInt32(123), UInt32(0))
    c = buf.sample_indices(20, UInt32(999), UInt32(0))

    for i in range(20):
        if a[i] != b[i]:
            raise Error("the same seed should give the same indices")
        if a[i] < 0 or a[i] >= buf.size():
            raise Error("index outside the valid sequences: ", a[i])

    same = True
    for i in range(20):
        if a[i] != c[i]:
            same = False
    if same:
        raise Error("two different seeds gave exactly the same "
                    "indices: the seed is not being used")
    print("PASS sampling is deterministic, depends on the seed and stays in range")


def test_gather_respects_order_and_repeats() raises:
    """`gather` with SEVERAL indices, in the requested order and with repeats.

    The previous tests only asked for one index at a time, so they checked neither
    the order nor that a repeated index comes out twice. And there will be repeats:
    the sampling is WITH REPLACEMENT, so in a real batch sequences repeat.
    """
    buf = TrajectoryBuffer(4, T_LEN, OBS_DIM)
    for v in range(1, 5):
        add_marked(buf, v)          # sequences 1..4 in slots 0..3

    # A deliberately shuffled order, with 2 repeated.
    idx = List[Int]()
    idx.append(3); idx.append(0); idx.append(2); idx.append(2)

    got = buf.gather(idx)
    span = T_LEN * OBS_DIM
    if len(got) != len(idx) * span:
        raise Error("gather should return ", len(idx), " sequences")

    want_values = List[Int]()
    want_values.append(4); want_values.append(1); want_values.append(3)
    want_values.append(3)
    for k in range(len(idx)):
        expected = seq_obs(want_values[k])
        for i in range(span):
            if got[k * span + i] != expected[i]:
                raise Error("position ", k, " of the batch should be "
                            "sequence ", want_values[k], ", but the value ", i,
                            " is ", got[k * span + i])

    # The same for the per-step fields and for the bootstrap_obs.
    got_r = buf.gather_steps(idx, 0)
    if len(got_r) != len(idx) * T_LEN:
        raise Error("gather_steps should return ", len(idx), "x", T_LEN)
    for k in range(len(idx)):
        expected = seq_steps(want_values[k])
        for i in range(T_LEN):
            if got_r[k * T_LEN + i] != expected[i]:
                raise Error("gather_steps posicion ", k, " valor ", i)

    got_b = buf.gather_bootstrap(idx)
    for k in range(len(idx)):
        expected = seq_obs(want_values[k] + 1000)
        for i in range(span):
            if got_b[k * span + i] != expected[i]:
                raise Error("gather_bootstrap posicion ", k, " valor ", i)
    print("PASS gather respects the requested order and accepts repeated indices")


def test_sampling_more_than_stored() raises:
    """Asking for more samples than stored sequences works (there is replacement).

    In real training the batch (32) may be larger than what is there at the start,
    so this has to be supported and must not go out of range.
    """
    buf = TrajectoryBuffer(16, T_LEN, OBS_DIM)
    add_marked(buf, 1)
    add_marked(buf, 2)          # only 2 sequences stored

    idx = buf.sample_indices(32, UInt32(5), UInt32(0))
    if len(idx) != 32:
        raise Error("should return 32 indices")
    for i in range(32):
        if idx[i] < 0 or idx[i] >= 2:
            raise Error("index ", idx[i], " outside the 2 valid sequences")

    # And the gather of that large batch does not blow up.
    got = buf.gather(idx)
    if len(got) != 32 * T_LEN * OBS_DIM:
        raise Error("the big batch's gather does not have the expected size")
    print("PASS more samples than sequences can be requested (with replacement)")


def test_buffer_rejects_bad_input() raises:
    """A sequence with the wrong size is rejected instead of corrupting things.

    Without this check, putting in a short sequence would write garbage into the
    missing steps and nobody would find out until training looked odd.
    """
    buf = TrajectoryBuffer(2, T_LEN, OBS_DIM)
    short = List[Scalar[dtype]]()
    for _ in range(T_LEN - 1):
        short.append(Scalar[dtype](0))

    failed = False
    try:
        buf.add(seq_obs(1), short, seq_steps(1), seq_steps(1), seq_obs(2))
    except:
        failed = True
    if not failed:
        raise Error("should reject a sequence with fewer steps than required")

    # And an empty buffer does not allow sampling.
    empty = TrajectoryBuffer(2, T_LEN, OBS_DIM)
    failed2 = False
    try:
        _ = empty.sample_indices(1, UInt32(0), UInt32(0))
    except:
        failed2 = True
    if not failed2:
        raise Error("sampling from an empty buffer should raise")
    print("PASS the buffer rejects malformed sequences and sampling when empty")


def test_q_roundtrip_and_validation() raises:
    """The q goes in and comes out intact, and a malformed q is rejected.

    The q is the actor's target (equation 11). If it were stored wrongly, the actor
    would learn from garbage and NOTHING would fail: the loss would go down all the
    same, against the wrong target. Hence the exact value is checked and not a
    tolerance.
    """
    buf = TrajectoryBuffer(3, T_LEN, OBS_DIM, N_ACT)

    for v in range(1, 4):
        buf.add(seq_obs(v), seq_steps(v), seq_steps(v), seq_steps(v),
                seq_obs(v + 1000), q_for(v))

    idx = List[Int](); idx.append(2); idx.append(0)
    got = buf.gather_q(idx)
    want_v = List[Int](); want_v.append(3); want_v.append(1)
    span = T_LEN * N_ACT
    for k in range(2):
        expected = q_for(want_v[k])
        for i in range(span):
            if got[k * span + i] != expected[i]:
                raise Error("q of sequence ", want_v[k], " valor ", i,
                            ": ", got[k * span + i], " != ", expected[i])

    # A q with the wrong size is rejected.
    short = List[Scalar[dtype]]()
    for _ in range(span - 1):
        short.append(Scalar[dtype](0))
    failed = False
    try:
        buf.add(seq_obs(9), seq_steps(9), seq_steps(9), seq_steps(9),
                seq_obs(9), short)
    except:
        failed = True
    if not failed:
        raise Error("should reject a q with fewer values than required")

    # And a buffer without q does not allow asking for it.
    plain = TrajectoryBuffer(2, T_LEN, OBS_DIM)
    add_marked(plain, 1)
    failed2 = False
    try:
        _ = plain.gather_q(idx)
    except:
        failed2 = True
    if not failed2:
        raise Error("a buffer created without q should not allow gather_q")
    print("PASS the buffer's q goes in and comes back intact, and its size is validated")


def main() raises:
    test_buffer_fifo_and_wraparound()
    test_buffer_fields_dont_cross()
    test_buffer_sampling_is_deterministic()
    test_gather_respects_order_and_repeats()
    test_sampling_more_than_stored()
    test_buffer_rejects_bad_input()
    test_q_roundtrip_and_validation()
