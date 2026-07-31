"""Genera el golden del MLP del ACTOR: 18 -> H -> H -> 9, con enmascarado.

Correr desde la raiz de mojo_spo:
    ../.venv/bin/python tests/golden/gen/gen_actor.py

La red es la MISMA que la del critico salvo en la salida (9 logits en vez de 1
valor), asi que el forward reutiliza el codigo ya verificado en E1.3. Lo que este
golden anade de verdad es la parte nueva: **el enmascarado de casillas ilegales**.

Por que el enmascarado necesita golden propio y no basta con "ya lo probamos en el
prior": en el prior de la busqueda los logits legales valian todos 0, asi que
enmascarar y luego hacer softmax daba una uniforme y cualquier error de indexado se
habria visto igual de uniforme. Aqui los logits son numeros distintos entre si
salidos de una red, asi que un fallo de indexado (tapar la casilla equivocada)
cambia la distribucion de forma detectable.

Se usa NEG_INF = el float32 finito mas negativo, no -inf, por la misma razon que
`ttt_prior_logits_kernel`: una fila entera tapada (tablero lleno) daria nan con
-inf, y con MIN_FINITE degenera a uniforme, que es inofensivo. El golden reproduce
esa eleccion para que el test compare contra lo que el kernel hace de verdad.

Tres anchos, como en el critico: el tamano de la red no lo fija ni el paper ni
Stoix, asi que se mide.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

IN_DIM = 18
OUT_DIM = 9
HIDDENS = [32, 64, 256]
BATCHES = [5, 64]

# El mismo valor que usa envs/tictactoe.mojo: el float32 finito mas negativo.
NEG_INF = np.float32(np.finfo(np.float32).min)

rng = np.random.default_rng(29)


def he_init(fan_in, fan_out):
    """Inicializacion tipo He, igual que en el critico."""
    return rng.normal(0.0, 1.0 / np.sqrt(fan_in),
                      size=(fan_in, fan_out)).astype(np.float32)


def make_boards(m):
    """m tableros en formato de red (dos planos 0/1) y su mascara de legales.

    La mascara sale del MISMO tablero: una casilla es legal si no esta en ninguno
    de los dos planos. Asi el test comprueba de paso que la mascara y la
    observacion son coherentes, que es justo lo que se romperia si alguien tocara
    la codificacion.

    Se fuerza que ninguna fila quede sin casillas legales: un tablero lleno es una
    partida terminada y al actor no se le pregunta ahi. El caso degenerado se
    prueba aparte, a mano, en el test.
    """
    x = np.zeros((m, IN_DIM), dtype=np.float32)
    mask = np.zeros((m, OUT_DIM), dtype=np.float32)
    for r in range(m):
        while True:
            x[r] = 0.0
            for c in range(9):
                who = rng.integers(0, 3)      # 0 vacia, 1 mia, 2 suya
                if who == 1:
                    x[r, c] = 1.0
                elif who == 2:
                    x[r, 9 + c] = 1.0
            free = 1.0 - (x[r, :9] + x[r, 9:])
            if free.sum() > 0:
                mask[r] = free
                break
    return x, mask


def masked_softmax(logits, mask):
    """softmax sobre los logits con las casillas ilegales puestas a NEG_INF.

    Se resta el maximo por fila antes de exponenciar (softmax estable), que es lo
    que hace `softmax_rows` en Mojo. Importa reproducirlo: con NEG_INF sin restar
    el maximo, exp() desbordaria a 0 en toda la fila.
    """
    masked = np.where(mask > 0, logits, NEG_INF).astype(np.float32)
    shifted = masked - masked.max(axis=1, keepdims=True)
    e = np.exp(shifted.astype(np.float64))
    return masked, (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


lines = []
for hidden in HIDDENS:
    tag = f"h{hidden}"
    w1 = he_init(IN_DIM, hidden)
    b1 = rng.normal(0.0, 0.1, size=(hidden,)).astype(np.float32)
    w2 = he_init(hidden, hidden)
    b2 = rng.normal(0.0, 0.1, size=(hidden,)).astype(np.float32)
    w3 = he_init(hidden, OUT_DIM)
    b3 = rng.normal(0.0, 0.1, size=(OUT_DIM,)).astype(np.float32)

    for name, arr in [("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2),
                      ("w3", w3), ("b3", b3)]:
        arr.tofile(os.path.join(OUT, f"actor_{tag}_{name}.bin"))

    n_params = (IN_DIM * hidden + hidden * hidden + hidden * OUT_DIM
                + 2 * hidden + OUT_DIM)

    for m in BATCHES:
        x, mask = make_boards(m)
        a1 = np.maximum(x @ w1 + b1, 0.0).astype(np.float32)
        a2 = np.maximum(a1 @ w2 + b2, 0.0).astype(np.float32)
        raw = (a2 @ w3 + b3).astype(np.float32)
        masked, probs = masked_softmax(raw, mask)

        x.tofile(os.path.join(OUT, f"actor_{tag}_x{m}.bin"))
        mask.tofile(os.path.join(OUT, f"actor_{tag}_mask{m}.bin"))
        raw.tofile(os.path.join(OUT, f"actor_{tag}_raw{m}.bin"))
        masked.tofile(os.path.join(OUT, f"actor_{tag}_masked{m}.bin"))
        probs.tofile(os.path.join(OUT, f"actor_{tag}_probs{m}.bin"))

        # Comprobaciones del propio generador: si el golden estuviera mal, el test
        # de Mojo lo daria por bueno y no nos enterariamos.
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), "las filas no suman 1"
        assert (probs[mask == 0] == 0.0).all(), "una casilla ilegal tiene masa"
        legal_per_row = mask.sum(axis=1)
        lines.append(f"hidden {hidden} batch {m}  pesos {n_params}  "
                     f"legales/fila min {legal_per_row.min():.0f} "
                     f"max {legal_per_row.max():.0f}")
        print(f"  hidden {hidden:3} batch {m:2}: {n_params:6} pesos, "
              f"logits en [{raw.min():7.4f}, {raw.max():7.4f}], "
              f"legales/fila {legal_per_row.min():.0f}-{legal_per_row.max():.0f}, "
              f"p_max {probs.max():.4f}")

with open(os.path.join(OUT, "actor.txt"), "w") as f:
    f.write(f"in_dim {IN_DIM}\nout_dim {OUT_DIM}\n")
    f.write(f"hiddens {' '.join(str(h) for h in HIDDENS)}\n")
    f.write(f"batches {' '.join(str(b) for b in BATCHES)}\n")
    f.write(f"neg_inf {NEG_INF!r}\n")
    f.write("los ficheros llevan el tag h<HIDDEN>:\n")
    f.write("actor_h<H>_w1.bin      float32 in_dim x H\n")
    f.write("actor_h<H>_b1.bin      float32 H\n")
    f.write("actor_h<H>_w2.bin      float32 H x H\n")
    f.write("actor_h<H>_b2.bin      float32 H\n")
    f.write("actor_h<H>_w3.bin      float32 H x out_dim\n")
    f.write("actor_h<H>_b3.bin      float32 out_dim\n")
    f.write("actor_h<H>_x<M>.bin      float32 M x in_dim  (dos planos 0/1)\n")
    f.write("actor_h<H>_mask<M>.bin   float32 M x out_dim (1 legal, 0 ocupada)\n")
    f.write("actor_h<H>_raw<M>.bin    float32 M x out_dim (logits SIN enmascarar)\n")
    f.write("actor_h<H>_masked<M>.bin float32 M x out_dim (logits enmascarados)\n")
    f.write("actor_h<H>_probs<M>.bin  float32 M x out_dim (softmax enmascarado)\n")
    f.write("\n".join(lines) + "\n")

print("golden del actor escrito en", OUT)
