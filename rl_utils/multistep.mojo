"""Truncated GAE: the targets the critic learns.

A port of `batch_truncated_generalized_advantage_estimation` from
`stoix/utils/multistep.py`. It is the ONLY place where the library GAE is used; do
not confuse it with the `calculate_gae` inside the search, which is a different
thing (that one runs FORWARDS and feeds the temperature dual).

The problem it solves: what number should the critic aim at in each position? The
naive answer ("the reward that arrived at the end") has a lot of variance; the
other naive one ("what the critic already predicts") adds no information. The GAE
mixes the two with a lambda parameter, and does it by accumulating backwards:

    delta_t = r + gamma*v_t - v_tm1                 the one-step error
    acc     = delta + gamma*lambda*acc*(1 - trunc)  accumulated BACKWARDS
    target  = v_tm1 + acc                           what the critic will learn

`discount` already arrives with done folded in: in Stoix it is computed as
`(1 - done) * gamma`, so on an episode's last step it is 0 and cuts the bootstrap
by itself.

**The detail that has to be respected**, and it is commented the same way in
Stoix: on a truncated step the accumulator is reset (`* (1 - truncation)`) BUT
that same step's delta is still used. Truncating does not erase the step's
information, it cuts the backwards propagation. Confusing the two is RL's classic
silent bug: training keeps running, it just learns badly computed targets.

In tic-tac-toe truncation never fires (games end on their own in five moves at
most, there is no time limit), but it is implemented all the same: it is part of
the reference function and the test exercises it.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from ops.common import dtype, GlobalF32

comptime TPB_GAE = 128


def gae_kernel(adv_out: GlobalF32, target_out: GlobalF32, reward: GlobalF32,
               discount: GlobalF32, v_tm1: GlobalF32, v_t: GlobalF32,
               truncation: GlobalF32, batch: Int, t_len: Int,
               lambda_: Scalar[dtype]):
    """One sequence per thread, walked backwards in time.

    The parallelism is in the BATCH and not in time, because the recurrence is
    sequential by nature (each step needs the next one's accumulator). With T = 32
    the loop is short and it does not pay to complicate it with a parallel scan.

    Batch-major layout [B, T], which is the one Stoix uses by default
    (`time_major=False`).
    """
    b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= batch:
        return

    acc = Scalar[dtype](0)
    for i in range(t_len):
        t = t_len - 1 - i               # backwards
        idx = b * t_len + t

        delta = reward[idx] + discount[idx] * v_t[idx] - v_tm1[idx]
        # The delta DOES go in even if the step is truncated; what gets cut is the
        # accumulator coming from further ahead.
        acc = delta + discount[idx] * lambda_ * acc \
              * (Scalar[dtype](1) - truncation[idx])

        adv_out[idx] = acc
        target_out[idx] = v_tm1[idx] + acc


def truncated_gae(ctx: DeviceContext, adv: DeviceBuffer[dtype],
                  targets: DeviceBuffer[dtype], reward: DeviceBuffer[dtype],
                  discount: DeviceBuffer[dtype], v_tm1: DeviceBuffer[dtype],
                  v_t: DeviceBuffer[dtype], truncation: DeviceBuffer[dtype],
                  batch: Int, t_len: Int, lambda_: Scalar[dtype]) raises:
    """Advantages and critic targets for a batch of sequences [B, T]."""
    ctx.enqueue_function[gae_kernel, gae_kernel](
        adv.unsafe_ptr(), targets.unsafe_ptr(), reward.unsafe_ptr(),
        discount.unsafe_ptr(), v_tm1.unsafe_ptr(), v_t.unsafe_ptr(),
        truncation.unsafe_ptr(), batch, t_len, lambda_,
        grid_dim=(batch + TPB_GAE - 1) // TPB_GAE, block_dim=TPB_GAE)
