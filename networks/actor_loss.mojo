"""The M-step loss: weighted cross entropy (equation 11 of the paper).

    max_theta  E_{s~mu} [ E_{a~q(.|s)} [ log pi(a|s,theta) ] ]

that is, minimise  -SUM_a q(a) log pi(a)  averaged over the states. The paper
describes it as "projecting the non-parametric policy q back to the space of
parametrisable policies", just as AlphaZero does: the search produces an improved
policy and the network learns to imitate it.

**Dense form and particle form are the same quantity.** Stoix implements it as a
Monte Carlo estimator over the N particles,

    loss = -SUM_n w_n log pi(a_n)            (compute_cross_entropy_loss)

and since the root actions repeat across particles, grouping by action

    SUM_n w_n log pi(a_n) = SUM_a ( SUM_{n: a_n=a} w_n ) log pi(a) = SUM_a q(a) log pi(a)

This is not an approximation: it is the same sum regrouped. The golden proves it
by calling Stoix's actual function and comparing (diff ~1e-8).

The dense form is used here for two reasons. Our readout already produces q as a
vector [B, 9], so summing over 9 actions is cheaper than gathering log-probs from
512 particles with repetitions; and by not sampling, it carries no sampling
variance. In tic-tac-toe the action space is tiny, so the dense form is always
viable -- in an environment with thousands of actions one would have to go back to
the particle form, and that is why the golden keeps both.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import log
from std.gpu.host import DeviceContext, DeviceBuffer

from ops.common import dtype, GlobalF32
from networks.mlp import (CriticParams, CriticCache, CriticGrads,
                          CriticScratch, backward_from_dvalue)
from systems.spo.launch import TPB, blocks_for


def cross_entropy_rows_kernel(loss_out: GlobalF32, q: GlobalF32,
                              log_pi: GlobalF32, n_rows: Int,
                              num_actions: Int):
    """loss[row] = -SUM_a q[a] * log_pi[a]. One thread per state.

    One thread per row and not per (row, action) because num_actions is 9:
    splitting it up would cost more in reduction than in arithmetic.

    **The `if w != 0` is not an optimisation, it is correctness.** On an illegal
    cell q is 0 and log_pi is minus infinity (or the most negative finite value,
    depending on how the masking was done). The product 0 * (-inf) is NaN in IEEE,
    and a NaN here would propagate silently into the loss, the gradient and the
    weights, with nothing failing loudly. Skipping the term gives the correct
    value -- which is 0, because an action with zero target probability
    contributes nothing -- whatever the representation of the masking.
    """
    row = Int(block_dim.x * block_idx.x + thread_idx.x)
    if row >= n_rows:
        return
    acc = Scalar[dtype](0)
    base = row * num_actions
    for a in range(num_actions):
        w = q[base + a]
        if w != Scalar[dtype](0):
            acc += w * log_pi[base + a]
    loss_out[row] = -acc


def cross_entropy_rows(ctx: DeviceContext, loss_out: DeviceBuffer[dtype],
                       q: DeviceBuffer[dtype], log_pi: DeviceBuffer[dtype],
                       n_rows: Int, num_actions: Int) raises:
    """Each state's loss, unaveraged.

    It is left unaveraged on purpose: the average is a reduction the training loop
    already does on the host for reporting (just like the critic in E1.10), and the
    gradient needs the 1/n factor folded in somewhere else.
    """
    ctx.enqueue_function[cross_entropy_rows_kernel, cross_entropy_rows_kernel](
        loss_out.unsafe_ptr(), q.unsafe_ptr(), log_pi.unsafe_ptr(), n_rows,
        num_actions, grid_dim=blocks_for(n_rows), block_dim=TPB)


# ---------------------------------------------------------------------------
# The diagnostic: separating the floor from what the network actually learns.
#
# The cross entropy decomposes exactly into
#
#     H(q, pi)  =  H(q)  +  KL(q || pi)
#
# and H(q) does NOT depend on pi. It is the floor: when the actor reaches q, the
# KL is 0 and the loss stays at H(q), which can be anything at all.
#
# Why it matters to report the KL and not the raw loss. In the real loop q is
# produced by the search and CHANGES between iterations, so the floor moves: the
# loss can go up while the actor is learning better, purely because q became more
# spread out. And when comparing two readouts it is worse still -- we measured
# H(q) = 1.178 with SPO's readout and ~0 with the variant (its q is nearly
# one-hot), that is, 1.18 nats of difference in the floor alone. The raw losses of
# the two arms would not be comparable.
#
# The KL is: its zero means "the actor reproduces exactly what the search says",
# and that zero is the same across every configuration.
# ---------------------------------------------------------------------------


def entropy_rows_kernel(h_out: GlobalF32, q: GlobalF32, n_rows: Int,
                        num_actions: Int):
    """H(q)[row] = -SUM_a q[a] log q[a]. One thread per state.

    The `if w > 0` is the 0*log(0) = 0 convention, and it also avoids the NaN:
    log(0) is -inf and 0 * (-inf) gives NaN. With illegal actions q is exactly 0,
    so this case ALWAYS happens, it is not defensive.
    """
    row = Int(block_dim.x * block_idx.x + thread_idx.x)
    if row >= n_rows:
        return
    acc = Scalar[dtype](0)
    base = row * num_actions
    for a in range(num_actions):
        w = q[base + a]
        if w > Scalar[dtype](0):
            acc += w * log(w)
    h_out[row] = -acc


def kl_rows_kernel(kl_out: GlobalF32, cross_entropy: GlobalF32,
                   entropy: GlobalF32, n_rows: Int):
    """KL(q||pi) = H(q,pi) - H(q). One thread per state."""
    row = Int(block_dim.x * block_idx.x + thread_idx.x)
    if row < n_rows:
        kl_out[row] = cross_entropy[row] - entropy[row]


def entropy_rows(ctx: DeviceContext, h_out: DeviceBuffer[dtype],
                 q: DeviceBuffer[dtype], n_rows: Int,
                 num_actions: Int) raises:
    """H(q) of each state: the floor the loss cannot go below."""
    ctx.enqueue_function[entropy_rows_kernel, entropy_rows_kernel](
        h_out.unsafe_ptr(), q.unsafe_ptr(), n_rows, num_actions,
        grid_dim=blocks_for(n_rows), block_dim=TPB)


def kl_rows(ctx: DeviceContext, kl_out: DeviceBuffer[dtype],
            cross_entropy: DeviceBuffer[dtype], entropy: DeviceBuffer[dtype],
            n_rows: Int) raises:
    """The KL from the cross entropy and the entropy, both already computed."""
    ctx.enqueue_function[kl_rows_kernel, kl_rows_kernel](
        kl_out.unsafe_ptr(), cross_entropy.unsafe_ptr(), entropy.unsafe_ptr(),
        n_rows, grid_dim=blocks_for(n_rows), block_dim=TPB)


# ---------------------------------------------------------------------------
# The backward. The gradient of equation 11 with respect to the LOGITS comes out
# clean:
#
#     L = -SUM_a q(a) log pi(a),     pi = softmax(z)
#     d log pi(a) / d z_j = delta_aj - pi(j)
#     dL/dz_j = -SUM_a q(a) (delta_aj - pi(j)) = pi(j) - q(j)
#
# (using SUM_a q(a) = 1). It is the same result as the plan's particle form,
# SUM_n w_n (softmax - onehot(a_n)), because SUM_n w_n onehot(a_n) = q.
#
# Useful reading: the gradient is "what the network says MINUS what the search
# says". It pushes pi towards q and vanishes when they coincide, which is exactly
# what equation 11 asks for.
# ---------------------------------------------------------------------------


def logits_grad_kernel(dz: GlobalF32, pi: GlobalF32, q: GlobalF32,
                       mask: GlobalF32, n: Int, scale: Scalar[dtype]):
    """dL/dz = (pi - q) * scale, and ZERO where the action is illegal.

    `scale` is normally 1/n_rows, so that the gradient corresponds to the MEAN
    over the batch and not to the sum. It is passed from outside just as in
    `value_loss_grad_kernel`.

    The mask deserves an explanation, because the gradient would already come out
    0 on its own there: on an illegal cell pi is exactly 0 (exp(NEG_INF - max)
    underflows to zero) and so is q, so pi - q = 0. It is forced all the same for
    two reasons. One, that relying on an underflow to zero for a gradient to be
    correct is fragile: change the masking and it stops holding. And two, that
    logit **is not a parameter**: the forward overwrites it with NEG_INF, so the
    network does not influence it and its derivative has to be 0 by construction,
    not by numerical accident.
    """
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= n:
        return
    if mask[i] == Scalar[dtype](0):
        dz[i] = Scalar[dtype](0)
    else:
        dz[i] = (pi[i] - q[i]) * scale


def actor_backward(ctx: DeviceContext, params: CriticParams,
                   cache: CriticCache, grads: CriticGrads,
                   scratch: CriticScratch, x: DeviceBuffer[dtype],
                   pi: DeviceBuffer[dtype], q: DeviceBuffer[dtype],
                   mask: DeviceBuffer[dtype], m: Int) raises:
    """Gradients of the actor's 6 tensors for the equation-11 loss.

    `pi` has to come from a forward with the SAME x and the same weights that
    produced `cache`; otherwise the gradients come out wrong without anything
    failing. It is the same precondition as `critic_backward`.

    The backward path from the logits onwards is identical to the critic's -- the
    network is the same -- so `backward_from_dvalue` is reused, and it has been
    verified against JAX's autodiff since E1.5.
    """
    n_out = m * params.out_dim
    ctx.enqueue_function[logits_grad_kernel, logits_grad_kernel](
        scratch.dvalue.unsafe_ptr(), pi.unsafe_ptr(), q.unsafe_ptr(),
        mask.unsafe_ptr(), n_out, Scalar[dtype](1) / Scalar[dtype](m),
        grid_dim=blocks_for(n_out), block_dim=TPB)
    backward_from_dvalue(ctx, params, cache, grads, scratch, x, m)
