#!/usr/bin/env bash
# Execute the AVX-512 branch using SIMDe's portable intrinsic semantics.
# SIMD hardware performance must still be measured with a native binary.
# Usage: FASTPOLY_SIMDE_DIR=/path/to/simde-0.8.2 ./scripts/run-tests-simde.sh
# Set FASTPOLY_TEST_SINGLE_HEADER=1 to validate the generated header as well.
set -euo pipefail
cd "$(dirname "$0")/.."
SIMDE_DIR="${FASTPOLY_SIMDE_DIR:?Set FASTPOLY_SIMDE_DIR to a SIMDe checkout (tested with v0.8.2)}"
test -f "$SIMDE_DIR/simde/x86/avx512.h"
CXX="${CXX:-clang++}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/include"
cat > "$TMP/include/immintrin.h" <<'SHIM'
#pragma once
#define SIMDE_NO_NATIVE
#define SIMDE_ENABLE_NATIVE_ALIASES
#include "simde/x86/avx512.h"
// Select the backend after SIMDe has chosen portable types and implementations.
#define __AVX512F__ 1
#define __AVX512VL__ 1
using _MM_PERM_ENUM = int;
SHIM
INC=(-I"$TMP/include" -I"$SIMDE_DIR" -Iinclude)
if [ "${FASTPOLY_TEST_SINGLE_HEADER:-0}" = 1 ]; then
  mkdir -p "$TMP/include/fastpoly"
  for header in modint simd memory ntt poly fastpoly; do
    printf '#include "%s/fastpoly.hpp"\n' "$PWD" > "$TMP/include/fastpoly/$header.hpp"
  done
fi
for t in test_modint test_ntt test_poly test_memory; do
  "$CXX" -std=c++20 -O2 -pthread "${INC[@]}" -include "$TMP/include/immintrin.h" \
    "tests/$t.cpp" -o "$TMP/$t"
  printf '\n== AVX-512 portable model | %s ==\n' "$t"
  "$TMP/$t"
done
printf '\nAVX-512 MODEL PASSED\n'
