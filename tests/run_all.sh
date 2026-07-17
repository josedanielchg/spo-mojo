#!/usr/bin/env bash
# Corre todos los tests/test_*.mojo. Falla (exit 1) si alguno falla.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
failed=0

for t in "$ROOT"/tests/test_*.mojo; do
  name="$(basename "$t")"
  echo "--- $name"
  if ! "$ROOT/run.sh" "$t"; then
    echo "FAILED: $name"
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "some tests failed"
  exit 1
fi

echo "all tests passed"
