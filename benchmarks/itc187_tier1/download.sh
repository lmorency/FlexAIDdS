#!/usr/bin/env bash
# download.sh — Acquire ITC-187 tier-1 structures from RCSB PDB
# ITC thermodynamic values (ΔG, ΔH, −TΔS) are from curated literature;
# the full 187-complex dataset will be deposited to Zenodo upon publication.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
mkdir -p "${DATA_DIR}"

TARGETS=(1a4g 2rh1 3dzy 2hz4 1p62)
RCSB_BASE="https://files.rcsb.org/download"

echo "=== ITC-187 Tier-1 Data Download ==="
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
    echo "Ligand MOL2 files require the PDBbind refined set."
    echo "Register at http://www.pdbbind.org.cn/casf.asp and copy the"
    echo "${TARGETS[*]} directories into ${DATA_DIR}/"
    echo "For CI: pre-cache ligand files as workflow artifacts."
fi

echo ""
echo "Data directory: ${DATA_DIR}"
echo "Done."
