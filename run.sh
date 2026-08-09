#!/usr/bin/env bash
# Wrapper so as not to have to remember pixi's PATH nor the -I.
#
#   ./run.sh tests/test_scan.mojo        # normal
#   ASSERTS=1 ./run.sh tests/test_scan.mojo   # with debug_assert active
#
# The kernels' debug_asserts are compiled out by default (they cost nothing on the
# GPU). With ASSERTS=1 they are switched on and additionally say which block and
# which thread failed, which is exactly what is needed when something blows up
# inside a kernel.
set -euo pipefail

export PATH="$HOME/.pixi/bin:$PATH"

if [ "$#" -lt 1 ]; then
  echo "usage: [ASSERTS=1] ./run.sh <file.mojo> [args...]" >&2
  exit 1
fi

# -I with the project's root so that the `from ops...` imports resolve.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FLAGS=()
if [ "${ASSERTS:-0}" != "0" ]; then
  FLAGS+=(-D ASSERT=all)
fi

mojo run -I "$ROOT" "${FLAGS[@]}" "$@"
