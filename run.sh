#!/usr/bin/env bash
# Wrapper para no tener que acordarme del PATH de pixi ni del -I.
#
#   ./run.sh tests/test_scan.mojo        # normal
#   ASSERTS=1 ./run.sh tests/test_scan.mojo   # con debug_assert activo
#
# Los debug_assert de los kernels estan compilados fuera por defecto (no cuestan
# nada en la GPU). Con ASSERTS=1 se activan y ademas dicen que bloque y que hilo
# fallo, que es justo lo que hace falta cuando algo peta dentro de un kernel.
set -euo pipefail

export PATH="$HOME/.pixi/bin:$PATH"

if [ "$#" -lt 1 ]; then
  echo "usage: [ASSERTS=1] ./run.sh <file.mojo> [args...]" >&2
  exit 1
fi

# -I con la raiz del proyecto para que los `from ops...` resuelvan.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FLAGS=()
if [ "${ASSERTS:-0}" != "0" ]; then
  FLAGS+=(-D ASSERT=all)
fi

mojo run -I "$ROOT" "${FLAGS[@]}" "$@"
