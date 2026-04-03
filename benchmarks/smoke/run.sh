#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-smoke"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLEXAIDS_USE_CUDA=OFF \
  -DFLEXAIDS_USE_METAL=OFF \
  -DBUILD_PYTHON_BINDINGS=OFF \
  -DBUILD_TESTING=ON

cmake --build "${BUILD_DIR}" --parallel
ctest --test-dir "${BUILD_DIR}" --output-on-failure

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  PYTHON_BIN=python
fi

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install pytest numpy scipy
PYTHONPATH="${ROOT_DIR}/python" "${PYTHON_BIN}" -m pytest -q \
  "${ROOT_DIR}/python/tests/test_statmech_smoke.py" \
  "${ROOT_DIR}/python/tests/test_cli.py" \
  "${ROOT_DIR}/python/tests/test_version.py"
