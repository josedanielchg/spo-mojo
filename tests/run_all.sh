#!/usr/bin/env bash
# Corre todos los tests/test_*.mojo. Sale con 1 si alguno falla.
#
# Va con ASSERTS=1 a proposito: los debug_assert de las precondiciones
# (block_dim == TPB, row_size <= TPB...) solo existen si se compilan con ellos,
# y este es el sitio donde quiero que existan.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
failed=0
passed=0

for t in "$ROOT"/tests/test_*.mojo; do
  name="$(basename "$t")"
  echo "--- $name"
  if ASSERTS=1 "$ROOT/run.sh" "$t"; then
    passed=$((passed + 1))
  else
    echo "FAILED: $name"
    failed=$((failed + 1))
  fi
done

echo
if [ "$failed" -ne 0 ]; then
  echo "$failed archivo(s) de test fallaron, $passed ok"
  exit 1
fi

echo "todo en verde ($passed archivos)"
