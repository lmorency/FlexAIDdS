#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
CTEST_JOBS="${CTEST_JOBS:-$(nproc)}"
MPI_RANKS="${MPI_RANKS:-4}"
BENCH_WORKERS="${BENCH_WORKERS:-${CTEST_JOBS}}"
BENCH_TIER="${BENCH_TIER:-1}"
SKIP_BUILD="${SKIP_BUILD:-0}"

log() {
  printf '%s\n' "$*"
}

have_python_deps() {
  python - <<'PY' >/dev/null 2>&1
import numpy  # noqa: F401
import scipy  # noqa: F401
import yaml   # noqa: F401
PY
}

run_benchmark() {
  if command -v mpirun >/dev/null 2>&1; then
    log "Running benchmark with MPI ranks=${MPI_RANKS} and workers=${BENCH_WORKERS}"
    mpirun -n "${MPI_RANKS}" python -m benchmarks.run --all --tier "${BENCH_TIER}" \
      --distributed --nodes "${MPI_RANKS}" --workers "${BENCH_WORKERS}" --dry-run
  else
    log "mpirun not found; running multi-worker benchmark dry-run instead"
    python -m benchmarks.run --all --tier "${BENCH_TIER}" --workers "${BENCH_WORKERS}" --dry-run
  fi
}

if [[ "${SKIP_BUILD}" != "1" ]]; then
  log "[1/4] Configuring C++ tests in ${BUILD_DIR}"
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release

  log "[2/4] Building with parallel workers (${CTEST_JOBS})"
  cmake --build "${BUILD_DIR}" --parallel "${CTEST_JOBS}"

  log "[3/4] Running ctest in parallel"
  ctest --test-dir "${BUILD_DIR}" --output-on-failure --parallel "${CTEST_JOBS}"
else
  log "[1/4] SKIP_BUILD=1, skipping configure/build/ctest"
fi

log "[4/4] Running benchmark dry-run"
if have_python_deps; then
  run_benchmark
else
  log "Missing benchmark Python dependencies (numpy/scipy/pyyaml)."
  log "Install with: python -m pip install numpy scipy pyyaml"
  exit 3
fi
