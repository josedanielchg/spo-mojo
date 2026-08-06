"""Verifie l'alternance de siege contre les references exactes.

Le morpion est assez petit pour etre resolu par recursion. Contre un adversaire
uniforme, une politique elle-meme uniforme obtient exactement :

    en ouvrant           0.6484
    en repondant         0.3516
    moitie-moitie        0.5000     <- par symetrie du jeu

Ce dernier nombre est le test le plus severe qui soit : il ne depend d'aucune
mesure, seulement du fait que le jeu est symetrique et que le partage des sieges
est exact. S'il ne sort pas, c'est le montage qui est faux, pas l'agent.

On verifie aussi que les deux sieges pris SEPAREMENT retrouvent leurs valeurs
propres. Une erreur qui echangerait les deux sieges laisserait la moyenne
correcte tout en inversant les moities : sans ce second controle elle passerait
inapercue.

ATTENTION -- la moyenne se calcule comme la moyenne NON PONDEREE des deux scores
de siege, jamais en agregeant toutes les parties dans un seul compteur. Les deux
sieges ne produisent pas le meme nombre de parties a duree de campagne egale :
quand l'adversaire ouvre, il consomme une case, la partie est plus courte, et il
s'en termine davantage. Agreger biaiserait donc le resultat vers ce siege-la.
Mesure a l'appui : 12 233 parties contre 14 801 sur une meme campagne, et une
moyenne agregee de 0.4845 la ou la valeur exacte est 0.5000.
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

        # Un compteur par siege : indices pairs (l'agent ouvre) et impairs.
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

        # Moyenne NON PONDEREE des deux sieges. Voir l'avertissement en tete.
        n_tot = g[0] + g[1] + e_[0] + e_[1] + d[0] + d[1]
        moy = 0.5 * (scores[0] + scores[1])
        print("  moyenne       " + String(n_tot) + "    " + String(moy)
              + "   " + String(REF_MOYENNE))

        # Trois sigma sur une proportion de variance au plus 0.25.
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
