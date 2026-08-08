"""Checks the seat alternation against the exact references.

Tic-tac-toe is small enough to be solved by recursion. Against a uniform opponent,
a policy that is itself uniform obtains exactly:

    opening              0.6484
    answering            0.3516
    half and half        0.5000     <- by the game's symmetry

That last number is the most severe test there is: it depends on no measurement at
all, only on the game being symmetric and the seat split being exact. If it does
not come out, it is the setup that is wrong, not the agent.

We also check that the two seats taken SEPARATELY recover their own values. An
error that swapped the two seats would leave the mean correct while inverting the
halves: without this second control it would go unnoticed.

CAREFUL -- the mean is computed as the UNWEIGHTED mean of the two seat scores,
never by aggregating all the games into a single counter. The two seats do not
produce the same number of games at equal campaign length: when the opponent
opens, it consumes a cell, the game is shorter, and more of them finish.
Aggregating would therefore bias the result towards that seat. Measurement to
prove it: 12,233 games against 14,801 over the same campaign, and an aggregated
mean of 0.4852 where the exact value is 0.5000.
"""

from std.gpu.host import DeviceContext
from ops.buffers import zero_buffer
from ops.common import dtype, idx_dtype
from ops.rng import fill_uniform
from envs.tictactoe import (ttt_reset_alt_kernel, ttt_auto_reset_alt_kernel,
                            ttt_env_step_kernel, ttt_random_policy_kernel,
                            ttt_seat_opens_first, STATE_DIM, TPB_TTT)
from envs.tictactoe_runner import RNG_POLICY, RNG_RIVAL

comptime SEED = UInt32(20260804)
comptime RNG_OPEN = UInt32(5_000_000)
comptime NUM_ENVS = 256
comptime NUM_STEPS = 400

comptime REF_PREMIER = 0.6484
comptime REF_SECOND = 0.3516
comptime REF_MOYENNE = 0.5000


def main() raises:
    with DeviceContext() as ctx:
        blocks = (NUM_ENVS + TPB_TTT - 1) // TPB_TTT
        state = zero_buffer[dtype](ctx, NUM_ENVS * STATE_DIM)
        action = zero_buffer[idx_dtype](ctx, NUM_ENVS)
        reward = zero_buffer[dtype](ctx, NUM_ENVS)
        done = zero_buffer[idx_dtype](ctx, NUM_ENVS)
        u_pol = zero_buffer[dtype](ctx, NUM_ENVS)
        u_riv = zero_buffer[dtype](ctx, NUM_ENVS)
        u_open = zero_buffer[dtype](ctx, NUM_ENVS)

        ctx.enqueue_function[fill_uniform, fill_uniform](
            u_open.unsafe_ptr(), SEED, RNG_OPEN, NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)
        ctx.enqueue_function[ttt_reset_alt_kernel, ttt_reset_alt_kernel](
            state.unsafe_ptr(), u_open.unsafe_ptr(), NUM_ENVS,
            grid_dim=blocks, block_dim=TPB_TTT)

        # One counter per seat: even indices (the agent opens) and odd ones.
        var g = InlineArray[Int, 2](fill=0)
        var e_ = InlineArray[Int, 2](fill=0)
        var d = InlineArray[Int, 2](fill=0)

        for step in range(NUM_STEPS):
            ctx.enqueue_function[fill_uniform, fill_uniform](
                u_pol.unsafe_ptr(), SEED, RNG_POLICY + UInt32(step), NUM_ENVS,
                grid_dim=blocks, block_dim=TPB_TTT)
            ctx.enqueue_function[ttt_random_policy_kernel, ttt_random_policy_kernel](
                action.unsafe_ptr(), state.unsafe_ptr(), u_pol.unsafe_ptr(),
                NUM_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
            ctx.enqueue_function[fill_uniform, fill_uniform](
                u_riv.unsafe_ptr(), SEED, RNG_RIVAL + UInt32(step), NUM_ENVS,
                grid_dim=blocks, block_dim=TPB_TTT)
            ctx.enqueue_function[ttt_env_step_kernel, ttt_env_step_kernel](
                state.unsafe_ptr(), action.unsafe_ptr(), u_riv.unsafe_ptr(),
                reward.unsafe_ptr(), done.unsafe_ptr(), NUM_ENVS,
                grid_dim=blocks, block_dim=TPB_TTT)
            ctx.synchronize()

            with reward.map_to_host() as rh:
                with done.map_to_host() as dh:
                    for e in range(NUM_ENVS):
                        if Int(dh[e]) != 0:
                            k = 0 if ttt_seat_opens_first(e) else 1
                            r = rh[e]
                            if r > Scalar[dtype](0.75): g[k] += 1
                            elif r > Scalar[dtype](0.25): e_[k] += 1
                            else: d[k] += 1

            ctx.enqueue_function[fill_uniform, fill_uniform](
                u_open.unsafe_ptr(), SEED, RNG_OPEN + UInt32(step) + 1, NUM_ENVS,
                grid_dim=blocks, block_dim=TPB_TTT)
            ctx.enqueue_function[ttt_auto_reset_alt_kernel, ttt_auto_reset_alt_kernel](
                state.unsafe_ptr(), done.unsafe_ptr(), u_open.unsafe_ptr(),
                NUM_ENVS, grid_dim=blocks, block_dim=TPB_TTT)
        ctx.synchronize()

        print("  siege        parties     score    reference")
        var scores = InlineArray[Float64, 2](fill=0.0)
        var refs = InlineArray[Float64, 2](fill=0.0)
        refs[0] = REF_PREMIER
        refs[1] = REF_SECOND
        for k in range(2):
            n = g[k] + e_[k] + d[k]
            scores[k] = (Float64(g[k]) + 0.5 * Float64(e_[k])) / Float64(n if n > 0 else 1)
            nom = String("l'agent ouvre") if k == 0 else String("le rival ouvre")
            print("  " + nom + "   " + String(n) + "    "
                  + String(scores[k]) + "   " + String(refs[k]))

        # UNWEIGHTED mean of the two seats. See the warning in the header.
        n_tot = g[0] + g[1] + e_[0] + e_[1] + d[0] + d[1]
        moy = 0.5 * (scores[0] + scores[1])
        print("  moyenne       " + String(n_tot) + "    " + String(moy)
              + "   " + String(REF_MOYENNE))

        # Three sigma on a proportion of variance at most 0.25.
        tol0 = 3.0 * (0.25 / Float64(g[0] + e_[0] + d[0])) ** 0.5
        tol1 = 3.0 * (0.25 / Float64(g[1] + e_[1] + d[1])) ** 0.5
        tolm = 0.5 * (tol0 * tol0 + tol1 * tol1) ** 0.5
        ok = (abs(scores[0] - REF_PREMIER) < tol0
              and abs(scores[1] - REF_SECOND) < tol1
              and abs(moy - REF_MOYENNE) < tolm)
        print()
        if ok:
            print("PASS les deux sieges et leur moyenne retrouvent les references exactes")
        else:
            raise Error("les scores mesures ne retrouvent pas les references exactes")
