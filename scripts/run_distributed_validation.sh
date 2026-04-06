#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
MPI_PROCS="${MPI_PROCS:-2}"
CTEST_JOBS="${CTEST_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
BENCHMARK_ARGS="${BENCHMARK_ARGS:---all --tier 1 --dry-run}"

echo "[distributed-validation] root=$ROOT_DIR"
echo "[distributed-validation] build=$BUILD_DIR"
echo "[distributed-validation] ctest jobs=$CTEST_JOBS"
echo "[distributed-validation] mpi procs=$MPI_PROCS"
echo "[distributed-validation] benchmark args=$BENCHMARK_ARGS"

if [[ ! -d "$BUILD_DIR" ]]; then
  echo "ERROR: build directory not found: $BUILD_DIR" >&2
  echo "Hint: cmake -S . -B build && cmake --build build" >&2
  exit 2
fi

if ! command -v ctest >/dev/null 2>&1; then
  echo "ERROR: ctest not found in PATH." >&2
  exit 2
fi

if ! command -v mpirun >/dev/null 2>&1; then
  echo "ERROR: mpirun not found in PATH." >&2
  exit 2
fi

echo
echo "==> Running C++ test suite in parallel"
ctest --test-dir "$BUILD_DIR" --output-on-failure -j "$CTEST_JOBS"

echo
echo "==> Running distributed benchmark smoke pass"
(
  cd "$ROOT_DIR"
  mpirun -n "$MPI_PROCS" python -m benchmarks.run $BENCHMARK_ARGS --distributed --nodes "$MPI_PROCS" --workers "$CTEST_JOBS"
)

echo
echo "[distributed-validation] Completed successfully."
