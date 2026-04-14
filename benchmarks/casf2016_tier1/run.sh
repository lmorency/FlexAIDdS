#!/usr/bin/env bash
# run.sh — Execute CASF-2016 tier-1 benchmark and validate against baselines
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
RESULTS_DIR="${SCRIPT_DIR}/results"
BUILD_DIR="${ROOT_DIR}/build"
MANIFEST="${SCRIPT_DIR}/manifest.yaml"

TARGETS=(1a30 1gpk 1hvr 2cgr 2hb1)

# --- Parse baselines from manifest.yaml (portable, no yq needed) ---
parse_baseline() {
    local key="$1"
    grep "^${key}:" "${MANIFEST}" | head -1 | awk '{print $2}'
}

TOL=$(parse_baseline "baseline_tolerance" || echo "0.05")
PEARSON_R=$(parse_baseline "scoring_power_pearson_r" || echo "0.88")
RMSE=$(parse_baseline "scoring_power_rmse" || echo "1.20")
DOCK_TOP1=$(parse_baseline "docking_power_top1" || echo "0.82")
ENTROPY_RESCUE=$(parse_baseline "entropy_rescue_rate" || echo "0.35")

echo "=== CASF-2016 Tier-1 Benchmark ==="
echo "Targets: ${TARGETS[*]}"
echo "Baselines: Pearson r=${PEARSON_R}, RMSE=${RMSE}, Dock top1=${DOCK_TOP1}, Entropy rescue=${ENTROPY_RESCUE}"
echo "Tolerance: ${TOL}"
echo ""

# --- Build FlexAIDdS if needed ---
if [[ ! -x "${BUILD_DIR}/FlexAIDdS" ]]; then
    echo "[..] Building FlexAIDdS..."
    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${BUILD_DIR}" --parallel
fi

FLEXAIDDS="${BUILD_DIR}/FlexAIDdS"

# --- Download data if needed ---
if [[ ! -d "${DATA_DIR}" ]] || [[ ! -f "${DATA_DIR}/1a30.pdb" ]]; then
    echo "[..] Running download.sh..."
    bash "${SCRIPT_DIR}/download.sh"
fi

# --- Run docking on each target ---
mkdir -p "${RESULTS_DIR}"
PASS=0
FAIL=0
SKIP=0

for pdb in "${TARGETS[@]}"; do
    receptor="${DATA_DIR}/${pdb}.pdb"
    ligand="${DATA_DIR}/${pdb}/${pdb}_ligand.mol2"
    out_dir="${RESULTS_DIR}/${pdb}"

    if [[ ! -f "${receptor}" ]]; then
        echo "[SKIP] ${pdb}: receptor PDB missing"
        SKIP=$((SKIP + 1))
        continue
    fi

    if [[ ! -f "${ligand}" ]]; then
        echo "[SKIP] ${pdb}: ligand MOL2 missing (needs PDBbind package)"
        SKIP=$((SKIP + 1))
        continue
    fi

    echo "[..] Docking ${pdb}..."
    mkdir -p "${out_dir}"
    if "${FLEXAIDDS}" "${receptor}" "${ligand}" -o "${out_dir}" 2>&1 | tail -3; then
        echo "[OK] ${pdb} completed"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] ${pdb} docking error" >&2
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=== Results Summary ==="
echo "Passed: ${PASS}, Failed: ${FAIL}, Skipped: ${SKIP}"

# --- Generate metrics report ---
REPORT="${RESULTS_DIR}/report.md"
cat > "${REPORT}" <<HEREDOC
# CASF-2016 Tier-1 Report

## Execution
- Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- Targets: ${TARGETS[*]}
- Passed: ${PASS}, Failed: ${FAIL}, Skipped: ${SKIP}

## Metrics (placeholder — requires DatasetRunner integration)

| Metric | Baseline | Tolerance | Measured | Status |
|:-------|:--------:|:---------:|:--------:|:------:|
| scoring_power_pearson_r | ${PEARSON_R} | ±${TOL} | — | pending |
| scoring_power_rmse | ${RMSE} | ±${TOL} | — | pending |
| docking_power_top1 | ${DOCK_TOP1} | ±${TOL} | — | pending |
| entropy_rescue_rate | ${ENTROPY_RESCUE} | ±${TOL} | — | pending |

> Metrics will be populated once DatasetRunner computes them from pose output.
HEREDOC

echo "Report: ${REPORT}"

# --- Exit code ---
if [[ ${FAIL} -gt 0 ]]; then
    echo "RESULT: FAIL (${FAIL} targets failed)"
    exit 1
elif [[ ${PASS} -eq 0 ]]; then
    echo "RESULT: SKIP (no targets executed — missing ligand data?)"
    echo "This is expected in CI without PDBbind cached artifacts."
    exit 0
else
    echo "RESULT: OK (${PASS} targets docked successfully)"
    exit 0
fi
