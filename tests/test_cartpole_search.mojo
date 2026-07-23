"""La busqueda SMC corriendo sobre CartPole (etapa 3.2).

Aqui no se prueba la fisica (eso es test_cartpole.mojo) sino que CartPole enchufa
bien a la busqueda generica: corre sin NaN, es reproducible, el ESS se comporta
como debe, y el flag de truncacion se propaga de verdad hasta el resultado.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import isnan, abs

from ops.common import dtype, idx_dtype
from envs.cartpole import CartPole, default_cartpole, STATE_DIM, NUM_ACTIONS
from systems.spo.particles import SearchWorkspace
from systems.spo.spo_types import SPOConfig
from systems.spo.search import search
from tests.helpers import upload, download, assert_close


def make_config(num_envs: Int, depth: Int, period: Int) -> SPOConfig:
    return SPOConfig(
        num_envs=num_envs, num_particles=16, num_actions=NUM_ACTIONS,
        state_dim=STATE_DIM, search_depth=depth, resample_period=period,
        temperature=0.5, search_gamma=1.0, search_gae_lambda=1.0)


def upload_root(ctx: DeviceContext, num_envs: Int, x: Scalar[dtype],
                x_dot: Scalar[dtype], theta: Scalar[dtype],
                theta_dot: Scalar[dtype], time: Scalar[dtype]) raises -> DeviceBuffer[dtype]:
    root = List[Scalar[dtype]]()
    for _ in range(num_envs):
        root.append(x); root.append(x_dot); root.append(theta)
        root.append(theta_dot); root.append(time)
    return upload[dtype](ctx, root)


def test_search_runs(ctx: DeviceContext) raises:
    """Sin NaN, acciones validas, pesos que suman 1. Profundidad 16 (estres)."""
    cfg = make_config(num_envs=4, depth=16, period=4)
    model = default_cartpole()
    ws = SearchWorkspace(ctx, cfg)

    search[CartPole](ctx, ws, cfg, model,
                     upload_root(ctx, 4, 0.0, 0.0, 0.02, 0.0, 0.0), UInt32(1))
    ctx.synchronize()

    p_total = cfg.num_search_particles()
    actions = download[idx_dtype](ws.output.sampled_actions, p_total)
    weights = download[dtype](ws.output.sampled_action_weights, p_total)

    for p in range(p_total):
        a = Int(actions[p])
        if a < 0 or a >= NUM_ACTIONS:
            raise Error("accion invalida en ", p, ": ", a)
    for e in range(cfg.num_envs):
        total = Scalar[dtype](0)
        for n in range(cfg.num_particles):
            w = weights[e * cfg.num_particles + n]
            if isnan(w):
                raise Error("peso NaN en env ", e, " particula ", n)
            total += w
        assert_close(total, 1.0, 1e-4, String("pesos del env ", e, " suman 1"))
    print("PASS la busqueda corre sobre CartPole a profundidad 16 (sin NaN)")


def test_search_reproducible(ctx: DeviceContext) raises:
    """Misma seed -> misma busqueda, bit a bit."""
    cfg = make_config(num_envs=4, depth=8, period=4)
    model = default_cartpole()
    p_total = cfg.num_search_particles()

    first = List[Scalar[dtype]]()
    for run in range(2):
        ws = SearchWorkspace(ctx, cfg)
        search[CartPole](ctx, ws, cfg, model,
                         upload_root(ctx, 4, 0.0, 0.0, 0.02, 0.0, 0.0), UInt32(99))
        ctx.synchronize()
        w = download[dtype](ws.output.sampled_action_weights, p_total)
        if run == 0:
            first = w^
        else:
            for p in range(p_total):
                assert_close(w[p], first[p], 1e-7,
                             String("dos busquedas con la misma seed difieren en ", p))
    print("PASS la busqueda sobre CartPole es reproducible")


def test_ess_curve(ctx: DeviceContext) raises:
    """El ESS por profundidad: la forma de sierra del SMC.

    Con V ≡ 0 el peso es el retorno acumulado, y desde casi-vertical las
    particulas tardan varios pasos en caer, asi que el ESS se mantiene alto al
    principio (todas sobreviven igual) y baja despues, cuando unas caen y otras
    no. Por eso la sierra se ve con profundidad de sobra: a profundidad 7 el ESS
    ya se degrado, y el resampling de esa profundidad lo recupera en la 8.
    """
    cfg = make_config(num_envs=8, depth=12, period=4)   # resamples en 3, 7, 11
    model = default_cartpole()
    ws = SearchWorkspace(ctx, cfg)

    # theta 0.15 con velocidad hacia el umbral: divergencia de supervivencia.
    search[CartPole](ctx, ws, cfg, model,
                     upload_root(ctx, 8, 0.0, 0.0, 0.15, 0.3, 0.0), UInt32(7))
    ctx.synchronize()

    ess = download[dtype](ws.output.ess, cfg.search_depth * cfg.num_envs)
    means = List[Scalar[dtype]]()
    for d in range(cfg.search_depth):
        total = Scalar[dtype](0)
        for e in range(cfg.num_envs):
            total += ess[d * cfg.num_envs + e]
        means.append(total / Scalar[dtype](cfg.num_envs))

    print("      ESS por profundidad:")
    for d in range(cfg.search_depth):
        mark = "  <- resample" if (d + 1) % 4 == 0 else ""
        print("        depth", d, "->", means[d], mark)

    # ESS siempre en [1, N].
    for d in range(cfg.search_depth):
        if means[d] < 1.0 or means[d] > Scalar[dtype](cfg.num_particles) + 1e-3:
            raise Error("ESS fuera de [1, N] en depth ", d, ": ", means[d])
    # Se degrado de verdad para la profundidad 7 (si no, no hay sierra que ver).
    if means[7] >= means[0]:
        raise Error("el ESS deberia haberse degradado para depth 7: d0=",
                    means[0], " d7=", means[7])
    # Y el resampling de la profundidad 7 lo recupera en la 8 (medido antes de
    # resamplear otra vez: la 8 arranca con pesos reseteados).
    if means[8] <= means[7]:
        raise Error("el ESS deberia recuperarse tras el resampling: d7=",
                    means[7], " d8=", means[8])
    print("PASS ESS en rango, se degrada y se recupera tras el resampling")


def test_flag_changes_search(ctx: DeviceContext) raises:
    """El flag de truncacion se propaga hasta el resultado de la busqueda.

    Que esto se vea costo entenderlo, y el porque es instructivo. El tiempo avanza
    igual para TODAS las particulas (no depende de la accion), asi que todas cruzan
    el paso 500 en la MISMA profundidad. Con V constante, el flag les suma entonces
    el mismo offset a todas, y softmax es invariante a sumar una constante a todos
    los logits -> se cancela y q no cambia. El flag solo se ve si hay
    HETEROGENEIDAD: unas particulas mueren (el palo cae) antes del paso 500 y otras
    llegan, de modo que el flag las afecta de forma desigual.

    Dos condiciones, las dos necesarias:
      - theta 0.16 (precario, cerca del umbral): unas caen antes de 500, otras no.
      - SIN resampling (period > depth): el resampling homogeneiza la poblacion
        antes del cruce y volveria a cancelar el efecto. Sin el, la heterogeneidad
        sobrevive hasta la lectura. (Con critico de verdad, en vez de V constante,
        el flag se veria aunque todas crucen juntas, porque V(s') diferiria por
        particula; con V constante hace falta este montaje.)
    """
    cfg = make_config(num_envs=4, depth=8, period=99)   # 99 > 8: sin resampling
    p_total = cfg.num_search_particles()

    model_a = CartPole(value_scale=7.0, truncate_on_step_limit=False)
    model_b = CartPole(value_scale=7.0, truncate_on_step_limit=True)

    # theta=0.16 (unas caen) + time=495 (las que sobreviven cruzan el paso 500).
    ws_a = SearchWorkspace(ctx, cfg)
    search[CartPole](ctx, ws_a, cfg, model_a,
                     upload_root(ctx, 4, 0.0, 0.0, 0.16, 0.4, 495.0), UInt32(3))
    ctx.synchronize()
    wa = download[dtype](ws_a.output.sampled_action_weights, p_total)

    ws_b = SearchWorkspace(ctx, cfg)
    search[CartPole](ctx, ws_b, cfg, model_b,
                     upload_root(ctx, 4, 0.0, 0.0, 0.16, 0.4, 495.0), UInt32(3))
    ctx.synchronize()
    wb = download[dtype](ws_b.output.sampled_action_weights, p_total)

    max_diff = Scalar[dtype](0)
    for p in range(p_total):
        d = abs(wa[p] - wb[p])
        if d > max_diff:
            max_diff = d
    print("      max |peso_A - peso_B| =", max_diff)
    if max_diff < 1e-4:
        raise Error("el flag no cambia la busqueda: A y B dan lo mismo (", max_diff, ")")
    print("PASS el flag de truncacion se propaga hasta el resultado")


def main() raises:
    with DeviceContext() as ctx:
        test_search_runs(ctx)
        test_search_reproducible(ctx)
        test_ess_curve(ctx)
        test_flag_changes_search(ctx)
