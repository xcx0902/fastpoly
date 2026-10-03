#!/usr/bin/env bash
# Build and run the fastpoly test suite for every backend reachable from this
# machine:
#
#   * native arch, default flags                (NEON on arm64, SSE2 on x86-64)
#   * native arch, -march=native                (AVX2/AVX-512 when available)
#   * native arch, ASan + UBSan
#   * x86-64 with -mavx2, run through Rosetta    (Apple Silicon only)
#
# The last one is how the AVX2 kernels get exercised on an Apple Silicon Mac:
# the macOS SDK is universal, so clang can emit x86-64 and Rosetta can run it.
set -uo pipefail

cd "$(dirname "$0")/.."
CXX="${CXX:-clang++}"
INC="-Iinclude"
STD="-std=c++20"
TESTS="test_modint test_ntt test_poly"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
status=0

run() {  # run <label> <command...>
  local label="$1"; shift
  printf '\n\033[1m== %s ==\033[0m\n' "$label"
  if ! "$@"; then status=1; fi
}

build_and_run() {  # build_and_run <label> <flags...>
  local label="$1"; shift
  local ok=1
  for t in $TESTS; do
    # shellcheck disable=SC2086
    $CXX $STD -O2 $INC "$@" "tests/$t.cpp" -o "$TMP/${t}.bin" 2>"$TMP/log" || {
      printf '\033[31mcompile failed for %s (%s)\033[0m\n' "$t" "$label"
      sed -n '1,15p' "$TMP/log"
      status=1; ok=0; break
    }
  done
  [ "$ok" = 1 ] || return
  for t in $TESTS; do run "$label | $t" "$TMP/${t}.bin"; done
}

ARCH="$(uname -m)"
if [ "$ARCH" = "x86_64" ]; then
  build_and_run "x86-64 baseline" ""
  build_and_run "x86-64 -march=native" -march=native
else
  build_and_run "arm64 (NEON)" ""
  build_and_run "arm64 -march=native" -march=native
fi

printf '\n\033[1m== sanitizers (ASan + UBSan) ==\033[0m\n'
for t in $TESTS; do
  if $CXX $STD -O1 -g $INC -fsanitize=address,undefined -fno-omit-frame-pointer \
        "tests/$t.cpp" -o "$TMP/${t}.san" 2>"$TMP/log"; then
    run "sanitized | $t" "$TMP/${t}.san"
  else
    printf '\033[31mcompile failed for %s (sanitizers)\033[0m\n' "$t"; status=1
  fi
done

if [ "$ARCH" = "arm64" ] && arch -x86_64 /usr/bin/true >/dev/null 2>&1; then
  printf '\n\033[1m== x86-64 cross-build (AVX2 / SSE2), run via Rosetta ==\033[0m\n'
  build_and_run "x86-64 AVX2" -arch x86_64 -mavx2
  for t in $TESTS; do
    if $CXX $STD -O2 $INC -arch x86_64 "tests/$t.cpp" -o "$TMP/${t}.b" 2>/dev/null; then
      run "x86-64 baseline(SSE2) | $t" arch -x86_64 "$TMP/${t}.b"
    fi
  done
fi

printf '\n'
if [ "$status" = 0 ]; then
  printf '\033[32mALL BACKENDS PASSED\033[0m\n'
else
  printf '\033[31mFAILURES PRESENT\033[0m\n'
fi
exit "$status"
