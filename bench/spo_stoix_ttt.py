"""H5: evaluates SPO-Stoix on tic-tac-toe and writes the common CSV's row.

Run from the root of the Stoix repo (it needs its venv and its imports):

    PYTHONPATH=. .venv/bin/python mojo_spo/bench/spo_stoix_ttt.py \
        --checkpoint-uid ttt_run1 --games 6000 --out mojo_spo/results/bench_spo_stoix.csv

It writes in the SAME 17-column schema used by MCTS-Mojo and SMC-Mojo
(`mojo_spo/bench/metrics.mojo`), so that the three legs can be concatenated in
Milestone 4.

**It measures with BOTH protocols, and that is the delicate point.** Stoix's
evaluator plays `search_output.action`, which is an action SAMPLED from q
(`stoix/systems/search/evaluator.py:57`). Our MCTS plays its visit argmax
(`MCTS-mojo-tictactoe/src/mcts.mojo:56`: "return the most-visited root action").
Comparing one's argmax against the other's draw would be comparing exploitation
against exploration, so two labelled rows are emitted:

    spo_stoix_moda        the argmax of q -> comparable with MCTS and SMC-Mojo
    spo_stoix_muestreada  the draw from q -> what Stoix's evaluator does

The mode has to be built here because Stoix does not expose it: q is implicit (the
root actions with their weights, with repetitions), so it is aggregated per action
and the maximum taken -- the same operation as `q_histogram` in the Mojo version.

The script does NOT train. Training happens beforehand through the normal entry
point with `logger.checkpointing.save_model=True`, and here it is only loaded: that
way the measurement is reproducible and does not depend on training coming out the
same twice.
"""

import argparse
import csv
import os
import time
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# Exact tic-tac-toe references, by recursion over all states.
# Exact references by recursion over all reachable states.
# The agent opens half the games and answers the other half, and the two seats are
# NOT the same problem: against the same uniform rival, score-maximising play gets
# 0.9974 opening and 0.9624 answering. And above all, it loses 0.00% of the ones it
# opens against 0.42% of the ones it answers -- when answering, maximising the
# score REQUIRES accepting losses.
RANDOM_FIRST, RANDOM_SECOND, RANDOM_MEAN = 0.6484, 0.3516, 0.5000
OPTIMAL_FIRST, OPTIMAL_SECOND, OPTIMAL_MEAN = 0.9974, 0.9624, 0.9799

CSV_HEADER = [
    "language", "mode", "games", "iterations", "exploration", "seed",
    "total_runtime_s", "total_moves", "mcts_decisions", "total_simulations",
    "total_nodes", "x_wins", "o_wins", "draws",
    "simulations_per_second", "nodes_per_second", "avg_decision_time_s",
]


def dense_q(sampled_actions: jnp.ndarray, weights: jnp.ndarray,
            num_actions: int) -> jnp.ndarray:
    """q as a [num_actions] vector from (root actions, weights).

    It is equation 6's regrouping `SUM_n w_n = SUM_a q(a)`: the root actions repeat
    across particles and their weights get summed. It changes nothing, only the
    shape -- but it is needed in order to take the maximum, because on the implicit
    form one cannot.
    """
    one_hot = jax.nn.one_hot(sampled_actions, num_actions)  # [N, A]
    return jnp.einsum("n,na->a", weights, one_hot)


def _select(mask: jnp.ndarray, when_true: Any, when_false: Any) -> Any:
    """Chooses per env between two pytrees, broadcasting the mask to each leaf."""

    def pick(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        m = mask.reshape((mask.shape[0],) + (1,) * (jnp.ndim(a) - 1))
        return jnp.where(m, a, b)

    return jax.tree_util.tree_map(pick, when_true, when_false)


def _replace_rng(state: Any, keys: jnp.ndarray) -> Any:
    """Replaces `rng_key` at whichever level of the state has it."""
    if hasattr(state, "rng_key"):
        return state.replace(rng_key=keys)
    if hasattr(state, "base_env_state"):
        return state.replace(base_env_state=_replace_rng(state.base_env_state, keys))
    raise ValueError("no encuentro rng_key en el estado")


def build(cfg: DictConfig, untrained: bool = False) -> Tuple[Any, ...]:
    """Builds the environment, the networks and the search, and loads the trained weights.

    With `untrained=True` the checkpoint is NOT loaded and play happens with
    freshly initialised weights. It serves to compare the SEARCH on its own against
    another implementation's: an untrained network gives a practically uniform
    prior and an uninformed critic, so the only thing left working is the SMC. It is
    the only way to sweep the search knobs and have them move anything -- with the
    trained network the agent solves tic-tac-toe on its own and everything
    saturates.
    """
    from stoix.systems.spo.ff_spo import learner_setup
    from stoix.utils.checkpointing import Checkpointer
    from stoix.utils.make_env import make as make_env

    env, eval_env = make_env(cfg)
    keys = jax.random.split(jax.random.PRNGKey(cfg.arch.seed), 3)
    _, root_fn, search_apply_fn, learner_state = learner_setup(
        env, (keys[0], keys[1], keys[2]), cfg, eval_env
    )

    # The params come replicated per device and per update_batch: the first of each
    # axis is taken to get a "flat" set to play with.
    params = jax.tree_util.tree_map(lambda x: x[0][0], learner_state.params)

    if untrained:
        return eval_env, root_fn, search_apply_fn, params

    ckpt = Checkpointer(
        model_name=cfg.system.system_name, **cfg.logger.checkpointing.load_args
    )
    restored, _ = ckpt.restore_params(input_params=params)
    return eval_env, root_fn, search_apply_fn, restored


def play(cfg: DictConfig, eval_env: Any, root_fn: Any, search_apply_fn: Any,
         params: Any, num_envs: int, steps: int, greedy: bool,
         seed: int, random_policy: bool = False,
         break_rng_sharing: bool = False) -> Dict[str, Any]:
    """Plays `num_envs` games in parallel for `steps` rounds.

    `random_policy` skips the search and plays uniformly among the legal moves. It
    is the validation of the MEASUREMENT LOOP: it has to give random play's exact
    0.6484. Without it, a fault in the loop gets mistaken for a result of the agent
    -- and it did happen: the first version gave a score of 1.0000, above the
    theoretical optimum.
    """
    num_actions = cfg.system.action_dim if "action_dim" in cfg.system else 9

    def decide(params: Any, obs: Any, env_state: Any, key: jax.Array) -> jnp.ndarray:
        # `env_state` DIRECTLY, not `env_state.unwrapped_state`. Both appear in
        # Stoix and the difference is which environment is used: the LEARNER runs
        # with the full env (with core wrappers) and unwraps the state
        # (`ff_spo.py:1101`); the EVALUATOR runs with `eval_env`, which does not
        # carry them, and passes it as is (`search/evaluator.py:55`). Here we are
        # evaluating, so it is the latter.
        if random_policy:
            mask = obs.action_mask
            return jax.random.categorical(key, jnp.where(mask > 0, 0.0, -jnp.inf))
        root = root_fn(params, obs, env_state, key)
        out = search_apply_fn(params, key, root)
        if not greedy:
            return out.action
        # The mode: aggregate q per action and take the maximum. vmap over the batch.
        q = jax.vmap(dense_q, in_axes=(0, 0, None))(
            out.sampled_actions, out.sampled_action_weights, num_actions
        )
        return jnp.argmax(q, axis=-1)

    decide_jit = jax.jit(decide, static_argnums=())

    keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    reset = jax.jit(jax.vmap(eval_env.reset))
    step = jax.jit(jax.vmap(eval_env.step))

    state, ts = reset(keys)
    # The seat is deduced from the freshly reset board, with no extra plumbing:
    # nine legal cells means the agent opens, eight that the rival has already
    # played. It has to be recomputed after EVERY reset, because the draw is per
    # game.
    seat = _seat_of(ts)
    wins = np.zeros(2, dtype=np.int64)
    draws = np.zeros(2, dtype=np.int64)
    losses = np.zeros(2, dtype=np.int64)
    moves = 0
    key = jax.random.PRNGKey(seed + 1)
    reset_counter = num_envs

    # A warm-up pass off the clock: the first call pays for compilation, and that
    # is not play time.
    key, sub = jax.random.split(key)
    _ = decide_jit(params, ts.observation, state, sub).block_until_ready()

    start = time.perf_counter()
    for _ in range(steps):
        key, sub = jax.random.split(key)
        action = decide_jit(params, ts.observation, state, sub)

        if break_rng_sharing:
            # The search uses the REAL ENVIRONMENT as its model, and the rival's
            # randomness comes from `state.rng_key` (the adapter does
            # `split(state.rng_key)`). That is, the model derives the SAME rival
            # move that is actually going to happen: the search is not planning
            # under uncertainty, it sees the future. Here the key is re-randomised
            # AFTER deciding, so that the real rival uses one the search did not
            # see.
            key, rk = jax.random.split(key)
            fresh = jax.random.split(rk, num_envs)
            state = _replace_rng(state, fresh)

        state, ts = step(state, action)

        r = np.asarray(ts.reward)
        done = np.asarray(ts.last())
        moves += num_envs
        for k in (0, 1):
            fin = done & (seat == k)
            wins[k] += int(((r > 0.75) & fin).sum())
            draws[k] += int((((r > 0.25) & (r < 0.75)) & fin).sum())
            losses[k] += int(((r < 0.25) & fin).sum())

        # `eval_env` does NOT carry auto-reset (the core wrappers go only on
        # `env`), so the finished games have to be reset by hand. Without this the
        # same game keeps being counted at every step, and since pgx sets the
        # rewards to 0 in a terminal state, `(0+1)/2 = 0.5` turns them into DRAWS:
        # ~99% draws come out, which is this fault's signature.
        if done.any():
            fresh_keys = jax.random.split(
                jax.random.fold_in(jax.random.PRNGKey(seed), reset_counter),
                num_envs,
            )
            reset_counter += 1
            fresh_state, fresh_ts = reset(fresh_keys)
            mask = jnp.asarray(done)
            state, ts = _select(mask, fresh_state, state), _select(mask, fresh_ts, ts)
            # The reset envs have drawn a seat again: it has to be re-read.
            seat = np.where(done, _seat_of(ts), seat)
    jax.block_until_ready(state)
    runtime = time.perf_counter() - start

    decisions = num_envs * steps
    particles = int(cfg.system.num_particles)
    depth = int(cfg.system.search_depth)
    # One entry per seat. Time, decisions and moves are split by the seat's SHARE
    # OF ENVIRONMENTS, not by games: each environment decides once per step
    # whatever its seat, whereas the games where the rival opens are shorter and
    # more of them finish.
    out = {}
    for k, nom in ((0, "1er"), (1, "2e")):
        n_env = int((seat == k).sum())
        part = n_env / num_envs if num_envs else 0.0
        out[nom] = {
            "games": int(wins[k] + draws[k] + losses[k]),
            "iterations": particles,
            "exploration": float(cfg.system.temperature.fixed_temperature),
            "seed": seed,
            "total_runtime_s": runtime * part,
            "total_moves": int(moves * part),
            "mcts_decisions": int(decisions * part),
            "total_simulations": int(decisions * part) * particles * depth,
            "x_wins": int(wins[k]),
            "o_wins": int(losses[k]),
            "draws": int(draws[k]),
        }
    return out


def _seat_of(ts: Any) -> Any:
    """0 if the agent opens, 1 if it answers, per environment.

    It is read from the number of legal cells right after the reset: nine free is a
    pristine board, eight means the rival has already opened. No flag needs to be
    propagated through the state tree -- which matters, because adding a layer of
    state would break the search's particle broadcast.
    """
    libres = np.asarray(ts.observation.action_mask).sum(axis=-1)
    return (libres < 9).astype(np.int64)


def score_and_se(m: Dict[str, Any]) -> Tuple[float, float]:
    """Mean score per game and its EMPIRICAL standard error.

    The per-game score is 1 / 0.5 / 0, so its variance is `E[s^2] - E[s]^2` with
    `E[s^2] = (wins + 0.25*draws)/n`. It has to be computed for real and not
    bounded by the worst case (0.25, which comes from p=0.5): near the ceiling
    almost every game is a win, the real variance drops to ~1e-4 and the worst-case
    bound is ~1500 times wider. With it, the ceiling guard gave a tolerance of
    4 sigma = 0.023 and swallowed a 0.9997 without complaint -- exactly the leak it
    was meant to detect.
    """
    n = max(m["games"], 1)
    score = (m["x_wins"] + 0.5 * m["draws"]) / n
    var = max((m["x_wins"] + 0.25 * m["draws"]) / n - score * score, 0.0)
    return score, (var / n) ** 0.5


def row(mode: str, m: Dict[str, Any]) -> Dict[str, Any]:
    games = max(m["games"], 1)
    rt = max(m["total_runtime_s"], 1e-9)
    return {
        "language": "jax-gpu",
        "mode": mode,
        "games": m["games"],
        "iterations": m["iterations"],
        "exploration": f"{m['exploration']:.6f}",
        "seed": m["seed"],
        "total_runtime_s": f"{m['total_runtime_s']:.6f}",
        "total_moves": m["total_moves"],
        "mcts_decisions": m["mcts_decisions"],
        "total_simulations": m["total_simulations"],
        # The SMC search builds no tree, so the node columns are 0 across all three
        # SPO legs (the MCTS does fill them in).
        "total_nodes": 0,
        "x_wins": m["x_wins"],
        "o_wins": m["o_wins"],
        "draws": m["draws"],
        "simulations_per_second": f"{m['total_simulations'] / rt:.6f}",
        "nodes_per_second": f"{0.0:.6f}",
        "avg_decision_time_s": f"{rt / max(m['mcts_decisions'], 1):.6f}",
    }


def score_moyen(d: Dict[str, Any]) -> Tuple[float, float]:
    """UNWEIGHTED mean of the two seats, and its standard error.

    Pooling both seats' games into a single counter would bias the result: where
    the rival opens, the game is shorter and more of them finish, so the pooled
    average leans that way. Measured in Mojo with the same setup: 12,233 games
    against 14,801, and a pooled mean of 0.4845 where the exact truth of random
    play is 0.5000. The bias is not noise and no error bar would give it away.
    """
    s0, e0 = score_and_se(d["1er"])
    s1, e1 = score_and_se(d["2e"])
    return 0.5 * (s0 + s1), 0.5 * (e0 * e0 + e1 * e1) ** 0.5


def perte_moyenne(d: Dict[str, Any]) -> float:
    return 0.5 * sum(
        100.0 * d[k]["o_wins"] / max(d[k]["games"], 1) for k in ("1er", "2e")
    )


def show(mode: str, d: Dict[str, Any]) -> None:
    """The two seats and their unweighted mean."""
    trozos = []
    for k in ("1er", "2e"):
        m = d[k]
        n = max(m["games"], 1)
        trozos.append(
            f"{k} {(m['x_wins'] + 0.5 * m['draws']) / n:.4f}"
            f"/{100 * m['o_wins'] / n:.2f}%"
        )
    moy, _ = score_moyen(d)
    n_tot = d["1er"]["games"] + d["2e"]["games"]
    print(f"  {mode:28} " + "   ".join(trozos)
          + f"   moy {moy:.4f}/{perte_moyenne(d):.2f}%   n={n_tot}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-uid", required=True)
    ap.add_argument("--config-name", default="default_ff_spo_ttt")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--out", default="mojo_spo/results/bench_spo_stoix.csv")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--untrained", action="store_true",
                    help="Juega con pesos sin entrenar: aisla la busqueda.")
    ap.add_argument("--tag", default=None,
                    help="Sufijo para la etiqueta `mode`. Por defecto el uid del "
                         "checkpoint, para que dos configuraciones distintas no se "
                         "confundan en el CSV comun.")
    args = ap.parse_args()

    # The same config_path ff_spo's entry point uses
    # (`config_path="../../configs/default/anakin"`), or hydra will not find the
    # primary config.
    cfg_dir = os.path.abspath("stoix/configs/default/anakin")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(
            config_name=args.config_name,
            overrides=[
                f"arch.total_num_envs={args.num_envs}",
                # CAREFUL: `learner_setup` loads the checkpoint BY ITSELF when this
                # is True (`ff_spo.py:1836`), so with `--untrained` it has to be
                # switched off here. Skipping `build`'s `Checkpointer` is not
                # enough: the weights would already come restored inside
                # `learner_state.params` and the flag would do nothing.
                f"logger.checkpointing.load_model={not args.untrained}",
                f"logger.checkpointing.load_args.checkpoint_uid={args.checkpoint_uid}",
                *args.overrides,
            ],
        )
    OmegaConf.set_struct(cfg, False)
    # `arch.num_envs` and `num_devices` are not in the composed config: Stoix
    # derives them at run time (total_timestep_checker.py:59), and `make_env` needs
    # them. The same checker the entry point uses is called so that the config ends
    # up EXACTLY as it was during training.
    from stoix.utils.total_timestep_checker import check_total_timesteps

    cfg.num_devices = len(jax.devices())
    cfg = check_total_timesteps(cfg)

    print("=== SPO-Stoix en tres en raya ===")
    print(f"   particulas {cfg.system.num_particles}  profundidad "
          f"{cfg.system.search_depth}  periodo "
          f"{cfg.system.resampling.period}  temperatura "
          f"{'adaptativa' if cfg.system.temperature.adaptive else cfg.system.temperature.fixed_temperature}")
    print(f"   referencias exactas (1er / 2e / media) :"
          f" azar {RANDOM_FIRST}/{RANDOM_SECOND}/{RANDOM_MEAN}"
          f"   optimo {OPTIMAL_FIRST}/{OPTIMAL_SECOND}/{OPTIMAL_MEAN}")
    print()

    eval_env, root_fn, search_apply_fn, params = build(cfg, args.untrained)

    # Loop validation. BOTH seats are checked separately, not just the mean:
    # swapping them would leave the mean intact by inverting the halves, and that
    # fault would go unnoticed.
    chk = play(cfg, eval_env, root_fn, search_apply_fn, params, args.num_envs,
               args.steps, False, args.seed, random_policy=True)
    show("VALIDACION azar", chk)
    for k, esperado in (("1er", RANDOM_FIRST), ("2e", RANDOM_SECOND)):
        m = chk[k]
        nk = max(m["games"], 1)
        sk = (m["x_wins"] + 0.5 * m["draws"]) / nk
        if abs(sk - esperado) > 0.03:
            raise SystemExit(
                f"\n*** El bucle esta MAL en el asiento {k}: una politica aleatoria "
                f"deberia dar {esperado} y da {sk:.4f}. No se escribe CSV. ***"
            )
    chk_moy, _ = score_moyen(chk)
    if abs(chk_moy - RANDOM_MEAN) > 0.02:
        raise SystemExit(
            f"\n*** El bucle esta MAL: la media de los dos asientos deberia dar "
            f"{RANDOM_MEAN} y da {chk_moy:.4f}. No se escribe CSV. ***"
        )
    print(f"   (bucle validado: media {chk_moy:.4f} vs {RANDOM_MEAN} exacto)\n")

    tag = args.tag if args.tag is not None else args.checkpoint_uid
    rows = []
    for greedy, mode in ((True, "spo_stoix_moda"), (False, "spo_stoix_muestreada")):
        m = play(cfg, eval_env, root_fn, search_apply_fn, params,
                 args.num_envs, args.steps, greedy, args.seed)
        show(mode, m)
        score, se = score_moyen(m)

        # CEILING GUARD, PER SEAT. Beating the exact optimum is not a good result:
        # it is proof that the agent sees something it should not. This is how the
        # RNG leak was discovered (0.9996 against 0.9974, eleven sigmas).
        # Each seat has ITS OWN ceiling -- 0.9974 opening, 0.9624 answering -- and
        # conflating them would let an impossible value through on the second's
        # side.
        for k, techo in (("1er", OPTIMAL_FIRST), ("2e", OPTIMAL_SECOND)):
            sk, sek = score_and_se(m[k])
            if sk > techo + 4.0 * sek:
                raise SystemExit(
                    f"\n*** `{mode}` puntua {sk:.4f} +- {sek:.4f} en el asiento {k}, "
                    f"por encima del techo exacto {techo} "
                    f"({(sk - techo) / max(sek, 1e-12):.1f} sigma). Eso es IMPOSIBLE "
                    f"jugando limpio: hay una fuga de informacion. No se escribe CSV. ***"
                )

        # Control for the RNG leak, already fixed in `make_root_fn`
        # (`_rekey_particles`). It is measured again with the key sharing BROKEN by
        # hand: if the score changes, the search was still reading the real
        # environment's randomness.
        m2 = play(cfg, eval_env, root_fn, search_apply_fn, params,
                  args.num_envs, args.steps, greedy, args.seed,
                  break_rng_sharing=True)
        show(mode + " (control sin fuga)", m2)
        score2, se2 = score_moyen(m2)
        se_diff = (se * se + se2 * se2) ** 0.5
        sigmas = abs(score - score2) / max(se_diff, 1e-12)
        if sigmas > 4.0:
            raise SystemExit(
                f"\n*** `{mode}` puntua {score:.4f} pero {score2:.4f} al romper la "
                f"comparticion de RNG ({sigmas:.1f} sigma). La busqueda sigue leyendo "
                f"el azar del entorno real. No se escribe CSV. ***"
            )
        print(f"   (control: {score:.4f} vs {score2:.4f}, {sigmas:.1f} sigma "
              f"-> no hay fuga)")
        # One row per seat. The mean is computed at analysis time, not at
        # measurement time.
        rows.append(row(f"{mode}_{tag}_1er", m["1er"]))
        rows.append(row(f"{mode}_{tag}_2e", m["2e"]))

    # IT APPENDS. Milestone 4's common CSV has to hold the three legs and several
    # configurations at once, so it is appended to and the header is only written
    # if the file does not exist yet. Truncating here silently erased the previous
    # leg.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new_file = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if new_file:
            w.writeheader()
        w.writerows(rows)
    print(f"\n   {len(rows)} filas anadidas a {args.out}")
    print("   La fila `moda` es la comparable con MCTS y SMC-Mojo; la")
    print("   `muestreada` es lo que hace el evaluador de Stoix.")


if __name__ == "__main__":
    main()
