#!/usr/bin/env bash
# Runs every tests/test_*.mojo. Exits with 1 if any of them fails.
#
# It goes with ASSERTS=1 on purpose: the precondition debug_asserts
# (block_dim == TPB, row_size <= TPB...) only exist if compiled in, and this is
# the place where I want them to exist.
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
