#!/usr/bin/env bash
# Wrapper so I don't forget the pixi PATH every time.
# Usage: ./run.sh <file.mojo> [args...]
set -euo pipefail

export PATH="$HOME/.pixi/bin:$PATH"

if [ "$#" -lt 1 ]; then
  echo "usage: ./run.sh <file.mojo> [args...]" >&2
  exit 1
fi

# -I con la raiz del proyecto para que los `from ops...` resuelvan.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mojo run -I "$ROOT" "$@"
