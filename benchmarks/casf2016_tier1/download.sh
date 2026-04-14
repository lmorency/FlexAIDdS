#!/usr/bin/env bash
# download.sh — Acquire CASF-2016 tier-1 structures
# Fetches receptor PDB files from RCSB. Ligand MOL2 files require the
# PDBbind v2016 refined set (registration at pdbbind.org.cn).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
mkdir -p "${DATA_DIR}"

TARGETS=(1a30 1gpk 1hvr 2cgr 2hb1)
RCSB_BASE="https://files.rcsb.org/download"

echo "=== CASF-2016 Tier-1 Data Download ==="
echo "Targets: ${TARGETS[*]}"
echo ""

for pdb in "${TARGETS[@]}"; do
    pdb_file="${DATA_DIR}/${pdb}.pdb"
    if [[ -f "${pdb_file}" ]]; then
        echo "[OK] ${pdb}.pdb already exists"
    else
        echo "[..] Downloading ${pdb}.pdb from RCSB..."
        if curl -fsSL "${RCSB_BASE}/${pdb}.pdb" -o "${pdb_file}"; then
            echo "[OK] ${pdb}.pdb downloaded ($(wc -l < "${pdb_file}") lines)"
        else
            echo "[FAIL] Could not download ${pdb}.pdb" >&2
            exit 1
        fi
    fi
done

# Check for PDBbind ligand files
LIGAND_MISSING=0
for pdb in "${TARGETS[@]}"; do
    if [[ ! -f "${DATA_DIR}/${pdb}/${pdb}_ligand.mol2" ]]; then
        LIGAND_MISSING=1
    fi
done

if [[ ${LIGAND_MISSING} -eq 1 ]]; then
    echo ""
    echo "=== Ligand Files ==="
    echo "Ligand MOL2 files are NOT available from RCSB."
    echo "To obtain them, download the PDBbind v2016 refined set:"
    echo "  1. Register at http://www.pdbbind.org.cn/casf.asp"
    echo "  2. Download PDBbind_v2016_refined.tar.gz"
    echo "  3. Extract and copy the ${TARGETS[*]} directories into ${DATA_DIR}/"
    echo ""
    echo "For CI: pre-cache the ligand files as workflow artifacts."
fi

echo ""
echo "Data directory: ${DATA_DIR}"
echo "Done."
