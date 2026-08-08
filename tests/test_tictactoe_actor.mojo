"""The search model with the ACTOR's prior: letting the network steer the search.

As with the critic, what is checked here is not whether the prior is GOOD -- E2.6
measures that by playing games. What is checked is the wiring, and this one in
particular can break in an especially treacherous way: if the actor's logits did
not arrive, the model would keep the constructor's zeros, which after masking give
**exactly the same uniform prior** as always. That is, the search would go on
working just as well and the EM loop would not close, without anything failing or
any number changing.

Hence the central test is not "the prior has such a value" but **"the prior is NOT
uniform"**, with weights that produce a clear preference.
"""

from std.gpu.host import DeviceContext

from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from envs.tictactoe import (NUM_ACTIONS, NUM_CELLS, STATE_DIM, OBS_DIM, NEG_INF,
                            CELL_EMPTY)
from envs.tictactoe_actor import TicTacToeActor
from networks.actor import zero_actor_params
from networks.mlp import zero_critic_params
from systems.spo.particles import Particles, StepOutputs
from systems.spo.spo_types import SPOConfig
from tests.helpers import (upload, download, write_into, zeros, filled,
                           assert_close)

comptime TOL = Scalar[dtype](1e-5)
comptime HIDDEN = 8


def board9(c0: Int, c1: Int, c2: Int, c3: Int, c4: Int,
           c5: Int, c6: Int, c7: Int, c8: Int) -> List[Scalar[dtype]]:
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
                     search_depth=4, resample_period=99, temperature=0.5,
                     search_gamma=1.0, search_gae_lambda=1.0)


def biased_model(ctx: DeviceContext, max_batch: Int) raises -> TicTacToeActor:
    """A model whose actor clearly prefers cell 4 (the centre).

    With w1 = w2 = 0 the network ignores the board and outputs b3, so setting b3
    with a peak at 4 gives a fixed, predictable prior. It serves to check that the
    logits ARRIVE, which is what can break silently.
    """
    m = TicTacToeActor(ctx, max_batch, HIDDEN, Scalar[dtype](0.9))
    b3 = List[Scalar[dtype]]()
    for a in range(NUM_ACTIONS):
        b3.append(Scalar[dtype](3) if a == 4 else Scalar[dtype](0))
    write_into[dtype](m.params.b3, b3)
    ctx.synchronize()
    return m^


def test_root_prior_comes_from_the_network(ctx: DeviceContext) raises:
    """The root's prior comes from the network and is NOT uniform.

    It is the test that separates "the actor steers the search" from "the actor is
    connected but has no influence". With the weights at zero the masked prior
    would be uniform, indistinguishable from the usual one; here the network
    prefers the centre and that has to show.
    """
    num_envs = 2
    cfg = cfg_for(num_envs, 4)
    model = biased_model(ctx, cfg.num_search_particles())

    boards = List[Scalar[dtype]]()
    for b in board9(0,0,0, 0,0,0, 0,0,0): boards.append(b)      # todo libre
    for b in board9(1,0,-1, 0,0,0, 0,0,0): boards.append(b)     # 0 y 2 ocupadas
    root_state = upload[dtype](ctx, boards)

    logits = filled[dtype](ctx, num_envs * NUM_ACTIONS, Scalar[dtype](99))
    value = filled[dtype](ctx, num_envs, Scalar[dtype](77))
    model.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()

    got = download[dtype](logits, num_envs * NUM_ACTIONS)
    got_v = download[dtype](value, num_envs)

    for e in range(num_envs):
        for c in range(NUM_ACTIONS):
            v = got[e * NUM_ACTIONS + c]
            if boards[e * NUM_CELLS + c] != CELL_EMPTY:
                if v != NEG_INF:
                    raise Error("la casilla ocupada ", c, " del env ", e,
                                " deberia estar tapada y vale ", v)
            elif c == 4:
                assert_close(v, Scalar[dtype](3), TOL,
                             String("el centro del env ", e))
            else:
                assert_close(v, Scalar[dtype](0), TOL,
                             String("la casilla ", c, " del env ", e))
        # And the value is still 0, like the planner.
        assert_close(got_v[e], Scalar[dtype](0), TOL,
                     String("V del env ", e, " deberia ser 0"))

    # The check that really matters: it is NOT uniform.
    all_equal = True
    for c in range(1, NUM_ACTIONS):
        if got[c] != got[0]:
            all_equal = False
    if all_equal:
        raise Error("el prior salio uniforme: los logits del actor no estan "
                    "llegando, y el bucle EM no se cerraria")
    print("PASS el prior de la raiz sale de la red y no es uniforme")


def test_step_prior_uses_the_new_state(ctx: DeviceContext) raises:
    """After advancing, the prior is evaluated on the NEW board.

    It is where the next action is sampled, so evaluating it on the old state would
    leave the search proposing moves on cells that have just been taken. The case
    is chosen so that 4 (the network's favourite) ends up OCCUPIED after the step:
    if the prior looked at the old state, 4 would still carry its logit of 3
    instead of being masked.
    """
    cfg = cfg_for(1, 1)
    model = biased_model(ctx, cfg.num_search_particles())
    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    # X plays 4 (the centre). The rival will answer on some free cell.
    write_into[dtype](particles.state, board9(1,-1,1, -1,0,0, 0,0,0))
    acts = List[Scalar[idx_dtype]](); acts.append(Scalar[idx_dtype](4))
    write_into[idx_dtype](outputs.next_action, acts)
    us = List[Scalar[dtype]](); us.append(Scalar[dtype](0.1))
    step_us = upload[dtype](ctx, us)

    model.step(ctx, cfg, particles, outputs, step_us)
    ctx.synchronize()

    new_board = download[dtype](particles.state, NUM_CELLS)
    got = download[dtype](outputs.action_logits, NUM_ACTIONS)

    if new_board[4] == CELL_EMPTY:
        raise Error("la casilla 4 deberia estar ocupada tras el paso")
    if got[4] != NEG_INF:
        raise Error("la casilla 4 esta ocupada en el estado NUEVO, asi que su "
                    "logit deberia estar tapado; vale ", got[4],
                    " (el prior esta mirando el estado viejo)")
    for c in range(NUM_ACTIONS):
        want_masked = new_board[c] != CELL_EMPTY
        if want_masked and got[c] != NEG_INF:
            raise Error("la casilla ", c, " esta ocupada y su logit vale ", got[c])
        if not want_masked:
            assert_close(got[c], Scalar[dtype](0), TOL,
                         String("logit de la casilla libre ", c))
    print("PASS el prior del step se evalua en el estado nuevo")


def test_sync_from_brings_the_trained_actor(ctx: DeviceContext) raises:
    """`sync_from` brings the weights in and changes the prior; without it, it would
    be uniform.

    If it were not called, the model would keep the constructor's zeros and the
    masked prior would be EXACTLY the usual uniform one. That is, the EM loop would
    look closed and would not be, without any shape test noticing. This test
    compares both cases explicitly.
    """
    num_envs = 1
    cfg = cfg_for(num_envs, 2)
    fresh = TicTacToeActor(ctx, cfg.num_search_particles(), HIDDEN,
                           Scalar[dtype](0.9))
    boards = board9(0,0,0, 0,0,0, 0,0,0)
    root_state = upload[dtype](ctx, boards)
    logits = zeros[dtype](ctx, NUM_ACTIONS)
    value = zeros[dtype](ctx, num_envs)

    # Without syncing: uniform, and that is why the test is needed.
    fresh.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()
    before = download[dtype](logits, NUM_ACTIONS)
    for c in range(1, NUM_ACTIONS):
        assert_close(before[c], before[0], TOL,
                     "sin sincronizar, el prior deberia ser uniforme")

    # A pretend-trained actor: it prefers corner 8.
    src = zero_actor_params(ctx, HIDDEN)
    b3 = List[Scalar[dtype]]()
    for a in range(NUM_ACTIONS):
        b3.append(Scalar[dtype](2.5) if a == 8 else Scalar[dtype](0))
    write_into[dtype](src.b3, b3)
    ctx.synchronize()

    fresh.sync_from(ctx, src)
    fresh.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()
    after = download[dtype](logits, NUM_ACTIONS)
    assert_close(after[8], Scalar[dtype](2.5), TOL,
                 "tras sync_from el prior deberia venir de los pesos copiados")
    assert_close(after[0], Scalar[dtype](0), TOL, "y el resto a 0")

    # The copy is independent: touching the source afterwards does not move the model.
    b3b = List[Scalar[dtype]]()
    for a in range(NUM_ACTIONS):
        b3b.append(Scalar[dtype](-9) if a == 8 else Scalar[dtype](0))
    write_into[dtype](src.b3, b3b)
    ctx.synchronize()
    fresh.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()
    again = download[dtype](logits, NUM_ACTIONS)
    assert_close(again[8], Scalar[dtype](2.5), TOL,
                 "el modelo tiene su propia copia; el origen ya no le afecta")

    # And an incompatible shape is rejected.
    bad = zero_actor_params(ctx, HIDDEN + 1)
    failed = False
    try:
        fresh.sync_from(ctx, bad)
    except:
        failed = True
    if not failed:
        raise Error("sync_from deberia rechazar un actor con otra forma")
    print("PASS sync_from trae el actor entrenado, es independiente y valida")


def test_many_particles_multi_block(ctx: DeviceContext) raises:
    """5 envs x 13 particles = 65: more than one block and a non-round size.

    The kernels carry guards, but a guard is only tested if it is ever launched
    with a size that does not line up. It is the blind spot I have already been
    bitten by four times in this project, so it goes in from day one.
    """
    cfg = cfg_for(5, 13)
    p_total = cfg.num_search_particles()
    model = biased_model(ctx, p_total)
    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)

    boards = List[Scalar[dtype]]()
    acts = List[Scalar[idx_dtype]]()
    us = List[Scalar[dtype]]()
    for p in range(p_total):
        for b in board9(0,0,0, 0,0,0, 0,0,0): boards.append(b)
        acts.append(Scalar[idx_dtype](p % 9))
        us.append(Scalar[dtype](0.1))
    write_into[dtype](particles.state, boards)
    write_into[idx_dtype](outputs.next_action, acts)
    step_us = upload[dtype](ctx, us)

    model.step(ctx, cfg, particles, outputs, step_us)
    ctx.synchronize()

    new_boards = download[dtype](particles.state, p_total * NUM_CELLS)
    got = download[dtype](outputs.action_logits, p_total * NUM_ACTIONS)
    for p in range(p_total):
        for c in range(NUM_ACTIONS):
            v = got[p * NUM_ACTIONS + c]
            if new_boards[p * NUM_CELLS + c] != CELL_EMPTY:
                if v != NEG_INF:
                    raise Error("particula ", p, " casilla ", c,
                                " ocupada pero su logit vale ", v)
            elif c == 4:
                assert_close(v, Scalar[dtype](3), TOL,
                             String("particula ", p, " centro"))
            else:
                assert_close(v, Scalar[dtype](0), TOL,
                             String("particula ", p, " casilla ", c))
    print("PASS con 65 particulas (varios bloques, tamano no redondo) sale igual")


def test_rejects_more_boards_than_reserved(ctx: DeviceContext) raises:
    """Asking for more boards than were allocated raises an error, not silent corruption."""
    cfg = cfg_for(4, 8)                        # 32 particulas
    small = TicTacToeActor(ctx, 4, HIDDEN, Scalar[dtype](0.9))
    particles = Particles(ctx, cfg)
    outputs = StepOutputs(ctx, cfg)
    us = zero_buffer[dtype](ctx, cfg.num_search_particles())

    failed = False
    try:
        small.step(ctx, cfg, particles, outputs, us)
    except:
        failed = True
    if not failed:
        raise Error("deberia rechazar 32 particulas con sitio para 4")
    print("PASS el modelo rechaza mas tableros de los reservados")


def test_critic_value_reaches_the_search(ctx: DeviceContext) raises:
    """With `use_critic`, V comes from the network and the bootstrap respects terminal.

    It is the critic's reconnection, which had been switched off because of an
    E1.11 measurement made under other conditions. V is in equation 10 of the paper
    and in Stoix's `_critic_loss_fn`, so holding it at 0 was our deviation.

    The usual trick: with w1 = w2 = w3 = 0 and b3 = c, the network gives V = c for
    any board, so the expected value is known by hand.
    """
    num_envs = 2
    cfg = cfg_for(num_envs, 3)
    c = Scalar[dtype](0.6)
    model = TicTacToeActor(ctx, cfg.num_search_particles(), HIDDEN,
                           Scalar[dtype](1.0), Scalar[dtype](0),
                           use_critic=True)
    src = zero_critic_params(ctx, OBS_DIM, HIDDEN, 1)
    b3 = List[Scalar[dtype]](); b3.append(c)
    write_into[dtype](src.b3, b3)
    ctx.synchronize()
    model.sync_critic_from(ctx, src)
    ctx.synchronize()

    boards = List[Scalar[dtype]]()
    for b in board9(1,0,-1, 0,1,0, 0,0,0): boards.append(b)
    for b in board9(0,0,0, 0,0,0, 0,0,0): boards.append(b)
    root_state = upload[dtype](ctx, boards)
    logits = zeros[dtype](ctx, num_envs * NUM_ACTIONS)
    value = filled[dtype](ctx, num_envs, Scalar[dtype](99))

    model.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()
    got = download[dtype](value, num_envs)
    for e in range(num_envs):
        assert_close(got[e], c, TOL, String("V de la raiz del env ", e))

    # And the step's bootstrap: gamma * V if still alive, 0 if it ended.
    cfg2 = cfg_for(1, 2)
    m2 = TicTacToeActor(ctx, cfg2.num_search_particles(), HIDDEN,
                        Scalar[dtype](1.0), Scalar[dtype](0), use_critic=True)
    m2.sync_critic_from(ctx, src)
    ctx.synchronize()
    particles = Particles(ctx, cfg2)
    outputs = StepOutputs(ctx, cfg2)
    st = List[Scalar[dtype]]()
    for b in board9(0,0,0, 0,0,0, 0,0,0): st.append(b)      # sigue viva
    for b in board9(1,1,0, -1,-1,0, 0,0,0): st.append(b)    # gana con la 2
    write_into[dtype](particles.state, st)
    acts = List[Scalar[idx_dtype]]()
    acts.append(Scalar[idx_dtype](0)); acts.append(Scalar[idx_dtype](2))
    write_into[idx_dtype](outputs.next_action, acts)
    us = List[Scalar[dtype]]()
    us.append(Scalar[dtype](0.1)); us.append(Scalar[dtype](0.1))
    step_us = upload[dtype](ctx, us)
    outputs.next_value.enqueue_fill(Scalar[dtype](-7))

    m2.step(ctx, cfg2, particles, outputs, step_us)
    ctx.synchronize()
    nv = download[dtype](outputs.next_value, 2)
    dsc = download[dtype](outputs.discount, 2)
    assert_close(dsc[1], Scalar[dtype](0), TOL, "la particula 1 deberia acabar")
    # search_gamma in cfg_for is 1.0, so the live one's bootstrap is c.
    assert_close(nv[0], c, TOL, "bootstrap de la particula viva")
    assert_close(nv[1], Scalar[dtype](0), TOL,
                 "una particula terminal no arrastra valor futuro")

    # And without use_critic, V goes back to 0: both modes coexist.
    plain = TicTacToeActor(ctx, cfg.num_search_particles(), HIDDEN,
                           Scalar[dtype](1.0))
    plain.sync_critic_from(ctx, src)
    plain.eval_root(ctx, cfg, root_state, logits, value)
    ctx.synchronize()
    got0 = download[dtype](value, num_envs)
    for e in range(num_envs):
        assert_close(got0[e], Scalar[dtype](0), TOL,
                     String("sin use_critic, V del env ", e, " deberia ser 0"))

    # Forma incompatible: se rechaza.
    bad = zero_critic_params(ctx, OBS_DIM, HIDDEN, 2)
    failed = False
    try:
        model.sync_critic_from(ctx, bad)
    except:
        failed = True
    if not failed:
        raise Error("sync_critic_from deberia rechazar otra forma")
    print("PASS el V del critico llega a la busqueda y respeta el terminal")


def main() raises:
    with DeviceContext() as ctx:
        test_root_prior_comes_from_the_network(ctx)
        test_step_prior_uses_the_new_state(ctx)
        test_sync_from_brings_the_trained_actor(ctx)
        test_many_particles_multi_block(ctx)
        test_rejects_more_boards_than_reserved(ctx)
        test_critic_value_reaches_the_search(ctx)
