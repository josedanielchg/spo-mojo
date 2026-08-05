"""H5: evalua SPO-Stoix en tres en raya y escribe la fila del CSV comun.

Correr desde la raiz del repo de Stoix (necesita su venv y sus imports):

    PYTHONPATH=. .venv/bin/python mojo_spo/bench/spo_stoix_ttt.py \
        --checkpoint-uid ttt_run1 --games 6000 --out mojo_spo/results/bench_spo_stoix.csv

Escribe en el MISMO esquema de 17 columnas que usan MCTS-Mojo y SMC-Mojo
(`mojo_spo/bench/metrics.mojo`), para que las tres patas se puedan concatenar en el
Milestone 4.

**Mide con los DOS protocolos, y eso es el punto delicado.** El evaluador de Stoix
juega `search_output.action`, que es una accion MUESTREADA de q
(`stoix/systems/search/evaluator.py:57`). Nuestro MCTS juega su argmax de visitas
(`MCTS-mojo-tictactoe/src/mcts.mojo:56`: "return the most-visited root action").
Comparar el argmax de uno contra el sorteo del otro seria comparar explotacion
contra exploracion, asi que se emiten dos filas etiquetadas:

    spo_stoix_moda        el argmax de q  -> comparable con MCTS y con SMC-Mojo
    spo_stoix_muestreada  el sorteo de q  -> lo que hace el evaluador de Stoix

La moda hay que construirla aqui porque Stoix no la expone: q es implicita (las
acciones raiz con sus pesos, con repeticiones), asi que se agrega por accion y se
coge el maximo -- la misma operacion que `q_histogram` en la version de Mojo.

El script NO entrena. Se entrena antes con el punto de entrada normal y
`logger.checkpointing.save_model=True`, y aqui solo se carga: asi la medicion es
reproducible y no depende de que el entrenamiento salga igual dos veces.
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

# Referencias exactas de tres en raya, por recursion sobre todos los estados.
RANDOM_SCORE = 0.6484
OPTIMAL_SCORE = 0.9974

CSV_HEADER = [
    "language", "mode", "games", "iterations", "exploration", "seed",
    "total_runtime_s", "total_moves", "mcts_decisions", "total_simulations",
    "total_nodes", "x_wins", "o_wins", "draws",
    "simulations_per_second", "nodes_per_second", "avg_decision_time_s",
]


def dense_q(sampled_actions: jnp.ndarray, weights: jnp.ndarray,
            num_actions: int) -> jnp.ndarray:
    """q como vector [num_actions] a partir de (acciones raiz, pesos).

    Es la reagrupacion `SUMA_n w_n = SUMA_a q(a)` de la ecuacion 6: las acciones
    raiz se repiten entre particulas y sus pesos se suman. No cambia nada, solo la
    forma -- pero hace falta para poder coger el maximo, porque sobre la forma
    implicita no se puede.
    """
    one_hot = jax.nn.one_hot(sampled_actions, num_actions)  # [N, A]
    return jnp.einsum("n,na->a", weights, one_hot)


def _select(mask: jnp.ndarray, when_true: Any, when_false: Any) -> Any:
    """Elige por env entre dos pytrees, difundiendo la mascara a cada hoja."""

    def pick(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        m = mask.reshape((mask.shape[0],) + (1,) * (jnp.ndim(a) - 1))
        return jnp.where(m, a, b)

    return jax.tree_util.tree_map(pick, when_true, when_false)


def _replace_rng(state: Any, keys: jnp.ndarray) -> Any:
    """Sustituye `rng_key` en el nivel del estado que la tenga."""
    if hasattr(state, "rng_key"):
        return state.replace(rng_key=keys)
    if hasattr(state, "base_env_state"):
        return state.replace(base_env_state=_replace_rng(state.base_env_state, keys))
    raise ValueError("no encuentro rng_key en el estado")


def build(cfg: DictConfig) -> Tuple[Any, ...]:
    """Monta el entorno, las redes y la busqueda, y carga los pesos entrenados."""
    from stoix.systems.spo.ff_spo import learner_setup
    from stoix.utils.checkpointing import Checkpointer
    from stoix.utils.make_env import make as make_env

    env, eval_env = make_env(cfg)
    keys = jax.random.split(jax.random.PRNGKey(cfg.arch.seed), 3)
    _, root_fn, search_apply_fn, learner_state = learner_setup(
        env, (keys[0], keys[1], keys[2]), cfg, eval_env
    )

    # Los params vienen replicados por dispositivo y por update_batch: se coge el
    # primero de cada eje para tener un juego "plano" con el que jugar.
    params = jax.tree_util.tree_map(lambda x: x[0][0], learner_state.params)

    ckpt = Checkpointer(
        model_name=cfg.system.system_name, **cfg.logger.checkpointing.load_args
    )
    restored, _ = ckpt.restore_params(input_params=params)
    return eval_env, root_fn, search_apply_fn, restored


def play(cfg: DictConfig, eval_env: Any, root_fn: Any, search_apply_fn: Any,
         params: Any, num_envs: int, steps: int, greedy: bool,
         seed: int, random_policy: bool = False,
         break_rng_sharing: bool = False) -> Dict[str, Any]:
    """Juega `num_envs` partidas en paralelo durante `steps` rondas.

    `random_policy` salta la busqueda y juega uniformemente entre las legales. Es
    la validacion del BUCLE DE MEDIDA: tiene que dar el 0.6484 exacto del azar. Sin
    ella, un fallo del bucle se confunde con un resultado del agente -- y de hecho
    paso: la primera version daba score 1.0000, por encima del optimo teorico.
    """
    num_actions = cfg.system.action_dim if "action_dim" in cfg.system else 9

    def decide(params: Any, obs: Any, env_state: Any, key: jax.Array) -> jnp.ndarray:
        # `env_state` DIRECTO, no `env_state.unwrapped_state`. Los dos aparecen en
        # Stoix y la diferencia es cual entorno se usa: el LEARNER va con el env
        # completo (con core wrappers) y desnuda el estado (`ff_spo.py:1101`); el
        # EVALUADOR va con `eval_env`, que no los lleva, y lo pasa tal cual
        # (`search/evaluator.py:55`). Aqui se evalua, asi que toca lo segundo.
        if random_policy:
            mask = obs.action_mask
            return jax.random.categorical(key, jnp.where(mask > 0, 0.0, -jnp.inf))
        root = root_fn(params, obs, env_state, key)
        out = search_apply_fn(params, key, root)
        if not greedy:
            return out.action
        # La moda: agregar q por accion y coger el maximo. vmap sobre el batch.
        q = jax.vmap(dense_q, in_axes=(0, 0, None))(
            out.sampled_actions, out.sampled_action_weights, num_actions
        )
        return jnp.argmax(q, axis=-1)

    decide_jit = jax.jit(decide, static_argnums=())

    keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
    reset = jax.jit(jax.vmap(eval_env.reset))
    step = jax.jit(jax.vmap(eval_env.step))

    state, ts = reset(keys)
    wins = draws = losses = moves = 0
    key = jax.random.PRNGKey(seed + 1)
    reset_counter = num_envs

    # Una pasada de calentamiento fuera del reloj: la primera llamada paga la
    # compilacion, y eso no es tiempo de juego.
    key, sub = jax.random.split(key)
    _ = decide_jit(params, ts.observation, state, sub).block_until_ready()

    start = time.perf_counter()
    for _ in range(steps):
        key, sub = jax.random.split(key)
        action = decide_jit(params, ts.observation, state, sub)

        if break_rng_sharing:
            # La busqueda usa el ENTORNO REAL como modelo, y el azar del rival sale
            # de `state.rng_key` (el adaptador hace `split(state.rng_key)`). O sea
            # que el modelo deriva la MISMA jugada del rival que va a ocurrir de
            # verdad: la busqueda no planifica bajo incertidumbre, ve el futuro.
            # Aqui se re-randomiza la clave DESPUES de decidir, para que el rival
            # real use una que la busqueda no vio.
            key, rk = jax.random.split(key)
            fresh = jax.random.split(rk, num_envs)
            state = _replace_rng(state, fresh)

        state, ts = step(state, action)

        r = np.asarray(ts.reward)
        done = np.asarray(ts.last())
        moves += num_envs
        wins += int(((r > 0.75) & done).sum())
        draws += int((((r > 0.25) & (r < 0.75)) & done).sum())
        losses += int(((r < 0.25) & done).sum())

        # `eval_env` NO lleva auto-reset (los core wrappers van solo en `env`), asi
        # que hay que reiniciar a mano las partidas acabadas. Sin esto se sigue
        # contando la misma partida en cada paso, y como pgx pone las recompensas a
        # 0 en un estado terminal, `(0+1)/2 = 0.5` las convierte en TABLAS: sale
        # ~99% de empates, que es la firma de este fallo.
        if done.any():
            fresh_keys = jax.random.split(
                jax.random.fold_in(jax.random.PRNGKey(seed), reset_counter),
                num_envs,
            )
            reset_counter += 1
            fresh_state, fresh_ts = reset(fresh_keys)
            mask = jnp.asarray(done)
            state, ts = _select(mask, fresh_state, state), _select(mask, fresh_ts, ts)
    jax.block_until_ready(state)
    runtime = time.perf_counter() - start

    decisions = num_envs * steps
    particles = int(cfg.system.num_particles)
    depth = int(cfg.system.search_depth)
    return {
        "games": wins + draws + losses,
        "iterations": particles,
        "exploration": float(cfg.system.temperature.fixed_temperature),
        "seed": seed,
        "total_runtime_s": runtime,
        "total_moves": moves,
        "mcts_decisions": decisions,
        "total_simulations": decisions * particles * depth,
        "x_wins": wins,
        "o_wins": losses,
        "draws": draws,
    }


def score_and_se(m: Dict[str, Any]) -> Tuple[float, float]:
    """Score medio por partida y su error estandar EMPIRICO.

    El score por partida vale 1 / 0.5 / 0, asi que su varianza es
    `E[s^2] - E[s]^2` con `E[s^2] = (ganadas + 0.25*empates)/n`. Hay que calcularla
    de verdad y no acotarla por el peor caso (0.25, que sale de p=0.5): cerca del
    techo casi todas las partidas son victorias, la varianza real cae a ~1e-4 y la
    cota del peor caso es ~1500 veces mas ancha. Con ella, la guarda del techo daba
    una tolerancia de 4 sigma = 0.023 y se tragaba un 0.9997 sin protestar --
    exactamente la fuga que se pretendia detectar.
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
        # La busqueda SMC no construye arbol, asi que las columnas de nodos van a 0
        # en las tres patas de SPO (el MCTS si las rellena).
        "total_nodes": 0,
        "x_wins": m["x_wins"],
        "o_wins": m["o_wins"],
        "draws": m["draws"],
        "simulations_per_second": f"{m['total_simulations'] / rt:.6f}",
        "nodes_per_second": f"{0.0:.6f}",
        "avg_decision_time_s": f"{rt / max(m['mcts_decisions'], 1):.6f}",
    }


def show(mode: str, m: Dict[str, Any]) -> None:
    n = max(m["games"], 1)
    score = (m["x_wins"] + 0.5 * m["draws"]) / n
    print(
        f"  {mode:22} n={n:6}  gana {100*m['x_wins']/n:6.2f}%"
        f"  empata {100*m['draws']/n:5.2f}%  pierde {100*m['o_wins']/n:6.2f}%"
        f"  score {score:.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-uid", required=True)
    ap.add_argument("--config-name", default="default_ff_spo_ttt")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--out", default="mojo_spo/results/bench_spo_stoix.csv")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--tag", default=None,
                    help="Sufijo para la etiqueta `mode`. Por defecto el uid del "
                         "checkpoint, para que dos configuraciones distintas no se "
                         "confundan en el CSV comun.")
    args = ap.parse_args()

    # El mismo config_path que usa el punto de entrada de ff_spo
    # (`config_path="../../configs/default/anakin"`), o hydra no encuentra
    # la config primaria.
    cfg_dir = os.path.abspath("stoix/configs/default/anakin")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(
            config_name=args.config_name,
            overrides=[
                f"arch.total_num_envs={args.num_envs}",
                "logger.checkpointing.load_model=True",
                f"logger.checkpointing.load_args.checkpoint_uid={args.checkpoint_uid}",
                *args.overrides,
            ],
        )
    OmegaConf.set_struct(cfg, False)
    # `arch.num_envs` y `num_devices` no estan en la config compuesta: Stoix los
    # deriva en tiempo de ejecucion (total_timestep_checker.py:59), y `make_env`
    # los necesita. Se llama al mismo checker que usa el punto de entrada para que
    # la config quede EXACTAMENTE como en el entrenamiento.
    from stoix.utils.total_timestep_checker import check_total_timesteps

    cfg.num_devices = len(jax.devices())
    cfg = check_total_timesteps(cfg)

    print("=== SPO-Stoix en tres en raya ===")
    print(f"   particulas {cfg.system.num_particles}  profundidad "
          f"{cfg.system.search_depth}  periodo "
          f"{cfg.system.resampling.period}  temperatura "
          f"{'adaptativa' if cfg.system.temperature.adaptive else cfg.system.temperature.fixed_temperature}")
    print(f"   referencias exactas: azar {RANDOM_SCORE}  optimo {OPTIMAL_SCORE}")
    print()

    eval_env, root_fn, search_apply_fn, params = build(cfg)

    # Primero la validacion del bucle: politica aleatoria -> 0.6484 exacto.
    chk = play(cfg, eval_env, root_fn, search_apply_fn, params, args.num_envs,
               args.steps, False, args.seed, random_policy=True)
    n = max(chk["games"], 1)
    chk_score = (chk["x_wins"] + 0.5 * chk["draws"]) / n
    show("VALIDACION azar", chk)
    if abs(chk_score - RANDOM_SCORE) > 0.02:
        raise SystemExit(
            f"\n*** El bucle de medida esta MAL: una politica aleatoria deberia "
            f"dar {RANDOM_SCORE} y da {chk_score:.4f}. No se escribe CSV. ***"
        )
    print(f"   (bucle validado: {chk_score:.4f} vs {RANDOM_SCORE} exacto)\n")

    tag = args.tag if args.tag is not None else args.checkpoint_uid
    rows = []
    for greedy, mode in ((True, "spo_stoix_moda"), (False, "spo_stoix_muestreada")):
        m = play(cfg, eval_env, root_fn, search_apply_fn, params,
                 args.num_envs, args.steps, greedy, args.seed)
        show(mode, m)
        score, se = score_and_se(m)

        # GUARDA DEL TECHO. Contra un rival uniforme, el juego optimo puntua
        # exactamente 0.9974 (recursion sobre todos los estados). Superarlo no es un
        # buen resultado, es la prueba de que el agente tiene informacion que no
        # deberia tener. Asi se descubrio la fuga de RNG: 0.9996, once sigmas por
        # encima. Aborta en vez de escribir un numero imposible en el CSV.
        if score > OPTIMAL_SCORE + 4.0 * se:
            raise SystemExit(
                f"\n*** `{mode}` puntua {score:.4f} +- {se:.4f}, por encima del techo "
                f"exacto {OPTIMAL_SCORE} ({(score - OPTIMAL_SCORE) / max(se, 1e-12):.1f} "
                f"sigma). Eso es IMPOSIBLE jugando limpio: hay una fuga de "
                f"informacion. No se escribe CSV. ***"
            )

        # Control de la fuga de RNG, ya corregida en `make_root_fn` (`_rekey_particles`).
        # Se vuelve a medir con la comparticion de claves ROTA a mano: si el score
        # cambia, la busqueda seguia leyendo el azar del entorno real.
        m2 = play(cfg, eval_env, root_fn, search_apply_fn, params,
                  args.num_envs, args.steps, greedy, args.seed,
                  break_rng_sharing=True)
        show(mode + " (control sin fuga)", m2)
        score2, se2 = score_and_se(m2)
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
        rows.append(row(f"{mode}_{tag}", m))

    # ACUMULA. El CSV comun del Milestone 4 tiene que sostener las tres patas y
    # varias configuraciones a la vez, asi que se anade al final y la cabecera solo
    # se escribe si el fichero todavia no existe. Truncar aqui borraba la pata
    # anterior en silencio.
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
