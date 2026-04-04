"""
benchmarks/ligand_prep.py
=========================
SMILES → Mol2 → FlexAID .inp pipeline with FlexAID atom-type validation.

The pipeline:
  1. Canonicalise SMILES (RDKit preferred, OpenBabel fallback).
  2. Generate 3-D coordinates (ETKDG / distance geometry).
  3. Write Mol2 with SYBYL atom types.
  4. Validate that every atom type is recognised by FlexAID.
  5. Emit a FlexAID ligand-section .inp snippet.

No GPL dependencies are introduced.  RDKit (BSD) and OpenBabel (GPL-isolated
via subprocess only — never imported) are optional; the module degrades
gracefully when neither is present (raises PrepError with a clear message).
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FlexAID SYBYL atom-type whitelist
# Derived from FlexAID's own type assignment table (assign_types.cpp).
# ---------------------------------------------------------------------------

FLEXAID_ATOM_TYPES: frozenset[str] = frozenset(
    {
        # Carbon
        "C.3", "C.2", "C.1", "C.ar", "C.cat",
        # Nitrogen
        "N.3", "N.2", "N.1", "N.ar", "N.am", "N.pl3", "N.4",
        # Oxygen
        "O.3", "O.2", "O.co2", "O.spc", "O.t3p",
        # Sulphur
        "S.3", "S.2", "S.O", "S.O2",
        # Phosphorus
        "P.3",
        # Halogens
        "F", "Cl", "Br", "I",
        # Misc
        "H", "Du", "Du.C",
        # Metal ions (common)
        "Ca", "Fe", "Zn", "Mg", "Mn", "Co", "Ni", "Cu",
    }
)

# Broad fallback mapping for common atom types that differ between toolkits
_TYPE_ALIASES: dict[str, str] = {
    "C.ar.1": "C.ar",
    "N.pl": "N.pl3",
    "O.t": "O.3",
    "S.O1": "S.O",
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PrepResult:
    """Outcome of a successful ligand preparation."""
    smiles: str
    mol2_path: Path
    inp_path: Path
    n_atoms: int
    n_heavy: int
    unknown_types: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class PrepError(RuntimeError):
    """Raised when ligand preparation fails and cannot be recovered."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rdkit_available() -> bool:
    try:
        import importlib
        importlib.import_module("rdkit.Chem")
        return True
    except ImportError:
        return False


def _obabel_available() -> bool:
    return shutil.which("obabel") is not None


def _smiles_to_mol2_rdkit(smiles: str, out_path: Path, mol_name: str) -> int:
    """Generate 3-D mol2 via RDKit.  Returns heavy-atom count."""
    from rdkit import Chem  # type: ignore[import]
    from rdkit.Chem import AllChem, Draw  # noqa: F401 -- ensure allchem loaded

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise PrepError(f"RDKit could not parse SMILES: {smiles!r}")

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        # Fallback: distance geometry
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if result != 0:
        raise PrepError(f"RDKit 3-D embedding failed for SMILES: {smiles!r}")

    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
        tmp_sdf = Path(tmp.name)

    writer = Chem.SDWriter(str(tmp_sdf))
    mol.SetProp("_Name", mol_name)
    writer.write(mol)
    writer.close()

    _sdf_to_mol2_obabel(tmp_sdf, out_path)
    tmp_sdf.unlink(missing_ok=True)

    n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
    return n_heavy


def _sdf_to_mol2_obabel(sdf_path: Path, out_path: Path) -> None:
    """Convert SDF → Mol2 via OpenBabel subprocess (GPL-isolated)."""
    if not _obabel_available():
        raise PrepError(
            "obabel not found; install OpenBabel for SDF→Mol2 conversion."
        )
    cmd = [
        "obabel",
        str(sdf_path),
        "-O", str(out_path),
        "--mol2",
        "-xn",          # keep atom names
        "--partialcharge", "mmff94",
        "--gen3D",       # in case 3-D is missing
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise PrepError(
            f"obabel conversion failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )


def _smiles_to_mol2_obabel(smiles: str, out_path: Path, mol_name: str) -> int:
    """Generate mol2 directly from SMILES via obabel (fallback path)."""
    if not _obabel_available():
        raise PrepError(
            "Neither RDKit nor obabel is available.  "
            "Install at least one to prepare ligands."
        )
    cmd = [
        "obabel",
        f"-:{smiles}",
        "-O", str(out_path),
        "--mol2",
        "--gen3D",
        "--partialcharge", "mmff94",
        "--title", mol_name,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise PrepError(
            f"obabel SMILES→Mol2 failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    # Count heavy atoms from mol2
    if not out_path.exists():
        raise PrepError("obabel produced no output file")
    n_heavy = _count_heavy_mol2(out_path)
    return n_heavy


def _count_heavy_mol2(mol2_path: Path) -> int:
    """Return number of heavy (non-hydrogen) atoms in a mol2 file."""
    in_atom_block = False
    count = 0
    with mol2_path.open() as fh:
        for line in fh:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_block = True
                continue
            if in_atom_block and line.startswith("@<TRIPOS>"):
                break
            if in_atom_block and line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    atom_type = parts[5]
                    if not atom_type.upper().startswith("H"):
                        count += 1
    return count


def _read_atom_types_mol2(mol2_path: Path) -> list[str]:
    """Extract SYBYL atom types from a mol2 ATOM block."""
    types: list[str] = []
    in_atom_block = False
    with mol2_path.open() as fh:
        for line in fh:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_block = True
                continue
            if in_atom_block and line.startswith("@<TRIPOS>"):
                break
            if in_atom_block and line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    types.append(parts[5])
    return types


def validate_atom_types(mol2_path: Path) -> list[str]:
    """Return list of atom types in *mol2_path* not in FLEXAID_ATOM_TYPES.

    Applies _TYPE_ALIASES normalisation before checking.
    """
    raw_types = _read_atom_types_mol2(mol2_path)
    unknown: list[str] = []
    for t in raw_types:
        normalised = _TYPE_ALIASES.get(t, t)
        if normalised not in FLEXAID_ATOM_TYPES:
            unknown.append(t)
    return unknown


def patch_atom_types(mol2_path: Path) -> list[str]:
    """Rewrite unrecognised atom types in-place using alias table.

    Returns a list of types that still could not be resolved.
    """
    content = mol2_path.read_text()
    still_unknown: list[str] = []
    in_atom_block = False
    lines_out: list[str] = []

    for line in content.splitlines(keepends=True):
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom_block = True
            lines_out.append(line)
            continue
        if in_atom_block and line.startswith("@<TRIPOS>"):
            in_atom_block = False

        if in_atom_block and line.strip():
            parts = line.split()
            if len(parts) >= 6:
                orig_type = parts[5]
                new_type = _TYPE_ALIASES.get(orig_type, orig_type)
                if new_type != orig_type:
                    line = line.replace(orig_type, new_type, 1)
                elif new_type not in FLEXAID_ATOM_TYPES:
                    still_unknown.append(orig_type)

        lines_out.append(line)

    mol2_path.write_text("".join(lines_out))
    return list(set(still_unknown))


# ---------------------------------------------------------------------------
# .inp generation
# ---------------------------------------------------------------------------

_INP_LIGAND_TEMPLATE = """\
# FlexAID ligand section — auto-generated by ligand_prep.py
# Ligand: {mol_name}
# SMILES: {smiles}
# Heavy atoms: {n_heavy}

LIGAND          {mol2_path}
LIGAND_NAME     {mol_name}
"""


def write_inp(
    mol_name: str,
    smiles: str,
    mol2_path: Path,
    n_heavy: int,
    out_path: Path,
    extra_lines: Optional[dict[str, str]] = None,
) -> None:
    """Write a minimal FlexAID .inp snippet for a prepared ligand.

    Parameters
    ----------
    mol_name:
        Short identifier used as LIGAND_NAME.
    smiles:
        Original SMILES string (recorded as comment).
    mol2_path:
        Absolute path to the prepared mol2 file.
    n_heavy:
        Number of heavy atoms.
    out_path:
        Destination .inp file path.
    extra_lines:
        Optional dict of additional KEY→VALUE entries to append.
    """
    body = _INP_LIGAND_TEMPLATE.format(
        mol_name=mol_name,
        smiles=smiles,
        n_heavy=n_heavy,
        mol2_path=mol2_path.resolve(),
    )
    if extra_lines:
        body += "\n".join(f"{k:<16}{v}" for k, v in extra_lines.items()) + "\n"
    out_path.write_text(body)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_ligand(
    smiles: str,
    output_dir: Path,
    mol_name: str = "LIG",
    extra_inp: Optional[dict[str, str]] = None,
) -> PrepResult:
    """Full SMILES → Mol2 → .inp preparation pipeline.

    Parameters
    ----------
    smiles:
        Input SMILES string (may contain stereo / charge annotations).
    output_dir:
        Directory where mol2 and inp files will be written.
    mol_name:
        Molecule identifier (used in filenames and .inp).
    extra_inp:
        Extra KEY→VALUE pairs to add to the .inp file.

    Returns
    -------
    PrepResult on success; raises PrepError on failure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitise mol_name to safe filename
    safe_name = re.sub(r"[^\w\-]", "_", mol_name)
    mol2_path = output_dir / f"{safe_name}.mol2"
    inp_path = output_dir / f"{safe_name}.inp"

    # Generate 3-D mol2
    n_heavy: int
    if _rdkit_available():
        logger.debug("Using RDKit for 3-D generation: %s", safe_name)
        n_heavy = _smiles_to_mol2_rdkit(smiles, mol2_path, mol_name)
    elif _obabel_available():
        logger.debug("Using obabel for 3-D generation: %s", safe_name)
        n_heavy = _smiles_to_mol2_obabel(smiles, mol2_path, mol_name)
    else:
        raise PrepError(
            "No 3-D generation backend available.  "
            "Install RDKit (conda install -c conda-forge rdkit) or OpenBabel."
        )

    # Atom-type validation and patching
    unknown = patch_atom_types(mol2_path)
    warnings: list[str] = []
    if unknown:
        warnings.append(
            f"Unresolved atom types (will be passed as-is): {', '.join(set(unknown))}"
        )
        logger.warning("%s: unresolved atom types: %s", safe_name, unknown)

    n_atoms_total = _count_heavy_mol2(mol2_path)  # after patch

    # Write .inp
    write_inp(
        mol_name=mol_name,
        smiles=smiles,
        mol2_path=mol2_path,
        n_heavy=n_heavy,
        out_path=inp_path,
        extra_lines=extra_inp,
    )

    return PrepResult(
        smiles=smiles,
        mol2_path=mol2_path,
        inp_path=inp_path,
        n_atoms=n_atoms_total,
        n_heavy=n_heavy,
        unknown_types=unknown,
        warnings=warnings,
    )


def prepare_batch(
    smiles_list: list[tuple[str, str]],
    output_dir: Path,
    fail_fast: bool = False,
) -> tuple[list[PrepResult], list[tuple[str, str, str]]]:
    """Prepare a batch of ligands.

    Parameters
    ----------
    smiles_list:
        List of (mol_name, smiles) tuples.
    output_dir:
        Root output directory; each ligand gets a sub-directory.
    fail_fast:
        If True, raise PrepError on the first failure; otherwise collect
        failures and continue.

    Returns
    -------
    (successes, failures) where failures is a list of (mol_name, smiles, error_msg).
    """
    successes: list[PrepResult] = []
    failures: list[tuple[str, str, str]] = []

    for mol_name, smiles in smiles_list:
        lig_dir = output_dir / mol_name
        try:
            result = prepare_ligand(smiles, lig_dir, mol_name)
            successes.append(result)
        except PrepError as exc:
            msg = str(exc)
            logger.error("Preparation failed for %s: %s", mol_name, msg)
            if fail_fast:
                raise
            failures.append((mol_name, smiles, msg))

    return successes, failures
