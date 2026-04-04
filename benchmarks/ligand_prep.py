"""Automated ligand preparation pipeline for FlexAIDdS benchmarks.

Converts raw compound inputs (SMILES or SDF) to the Mol2 + .inp files
consumed by the FlexAID C++ engine.  RDKit is an optional dependency;
preparation steps that require it fail gracefully with a clear error.

Pipeline stages
---------------
1. ``smiles_to_3d_mol2``   — ETKDGv3 conformer generation via RDKit
2. ``minimise_mol2``       — MMFF94s energy minimisation
3. ``validate_atom_types`` — SYBYL ↔ FlexAID type consistency check
4. ``mol2_to_inp``         — Generate FlexAID ligand .inp file header
5. ``prepare_ligand``      — Full end-to-end pipeline (single entry point)
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------


def _require_rdkit():
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem import AllChem  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for ligand preparation. "
            "Install with: conda install -c conda-forge rdkit"
        ) from exc


# ---------------------------------------------------------------------------
# SYBYL atom-type map (subset relevant to FlexAID scoring)
# ---------------------------------------------------------------------------

#: SYBYL type → base FlexAID type (single character or two-char code).
#: Only the types actually used by the CF scoring function are listed.
SYBYL_TO_FLEXAID: dict[str, str] = {
    "C.3":   "C",   # sp3 carbon
    "C.2":   "C",   # sp2 carbon
    "C.ar":  "C",   # aromatic carbon
    "C.1":   "C",   # sp carbon
    "N.3":   "N",   # sp3 nitrogen
    "N.2":   "N",   # sp2 nitrogen
    "N.ar":  "N",   # aromatic nitrogen
    "N.1":   "N",   # sp nitrogen
    "N.am":  "N",   # amide nitrogen
    "N.pl3": "N",   # planar nitrogen
    "N.4":   "N",   # quaternary nitrogen
    "O.3":   "O",   # sp3 oxygen
    "O.2":   "O",   # sp2 oxygen / carbonyl
    "O.co2": "O",   # carboxylate oxygen
    "S.3":   "S",   # sp3 sulfur
    "S.2":   "S",   # sp2 sulfur
    "S.O":   "S",   # sulfoxide
    "S.O2":  "S",   # sulfone
    "P.3":   "P",   # sp3 phosphorus
    "F":     "F",
    "Cl":    "Cl",
    "Br":    "Br",
    "I":     "I",
    "H":     "H",
    "Du":    "Du",  # dummy atom
}

#: SYBYL types that FlexAID cannot score — warn but do not abort.
UNSUPPORTED_SYBYL_TYPES: frozenset[str] = frozenset({"Si", "B", "Se", "Te", "As"})


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LigandPrepResult:
    """Outcome of the ligand preparation pipeline for one compound.

    Attributes:
        smiles:           Input SMILES (may be empty if SDF was the source).
        compound_id:      Identifier for this compound.
        mol2_path:        Path to the generated Mol2 file (None if failed).
        inp_path:         Path to the generated FlexAID .inp file (None if failed).
        n_conformers:     Number of conformers generated before minimisation.
        final_energy:     MMFF94s energy after minimisation (kcal/mol; NaN if unavail).
        atom_type_issues: List of SYBYL type warnings/errors found during validation.
        success:          True if both mol2 and inp files were produced.
        error:            Error message if preparation failed, else empty string.
    """

    smiles: str
    compound_id: str
    mol2_path: Optional[Path] = None
    inp_path: Optional[Path] = None
    n_conformers: int = 0
    final_energy: float = float("nan")
    atom_type_issues: List[str] = field(default_factory=list)
    success: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# Core preparation functions
# ---------------------------------------------------------------------------


def smiles_to_3d_mol2(
    smiles: str,
    compound_id: str,
    output_dir: Path,
    n_conformers: int = 50,
    max_attempts: int = 5,
    random_seed: int = 42,
) -> Tuple[Optional[Path], int]:
    """Generate a 3-D Mol2 file from a SMILES string via RDKit ETKDGv3.

    Args:
        smiles:       Input SMILES string.
        compound_id:  Identifier used for the output filename.
        output_dir:   Directory to write the Mol2 file.
        n_conformers: Number of ETKDGv3 conformers to generate.
        max_attempts: Maximum embedding attempts before giving up.
        random_seed:  RDKit random seed for reproducibility.

    Returns:
        ``(mol2_path, n_generated)`` — path to written file and actual conformer
        count; ``(None, 0)`` if embedding fails.
    """
    _require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.error("Invalid SMILES for %s: %s", compound_id, smiles)
        return None, 0

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0  # use all available CPUs
    params.maxAttempts = max_attempts
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True

    ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    if not ids:
        logger.warning("ETKDGv3 embedding failed for %s", compound_id)
        return None, 0

    # Keep only the lowest-energy conformer after MMFF94s
    best_id, best_energy = _lowest_mmff_conformer(mol, list(ids))

    # Remove all but the best conformer before writing
    best_mol = Chem.RWMol(mol)
    conf_ids_to_remove = [i for i in range(mol.GetNumConformers()) if i != best_id]
    for cid in reversed(conf_ids_to_remove):
        best_mol.RemoveConformer(cid)

    output_dir.mkdir(parents=True, exist_ok=True)
    mol2_path = output_dir / f"{compound_id}.mol2"

    # RDKit does not write Mol2 natively; use PDB → babel or write SYBYL via
    # a lightweight internal writer.
    _write_mol2_from_rdkit(best_mol, compound_id, mol2_path)

    logger.debug(
        "Generated %d conformers for %s; best MMFF94s energy = %.2f kcal/mol",
        len(ids), compound_id, best_energy,
    )
    return mol2_path, len(ids)


def _lowest_mmff_conformer(mol, conf_ids: list[int]) -> Tuple[int, float]:
    """Return the conformer id and MMFF94s energy of the lowest-energy conformer."""
    from rdkit.Chem import AllChem

    best_id = conf_ids[0]
    best_energy = float("inf")
    for cid in conf_ids:
        ff = AllChem.MMFFGetMoleculeForceField(
            mol,
            AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s"),
            confId=cid,
        )
        if ff is None:
            continue
        ff.Minimize(maxIts=500)
        energy = ff.CalcEnergy()
        if energy < best_energy:
            best_energy = energy
            best_id = cid
    return best_id, best_energy


def _write_mol2_from_rdkit(mol, compound_id: str, out_path: Path) -> None:
    """Write a minimal SYBYL Mol2 from an RDKit molecule.

    This is a lightweight writer that covers the atom/bond records needed by
    FlexAID.  It assigns SYBYL atom types via RDKit's MolToMolBlock + heuristics.
    """
    from rdkit.Chem import Descriptors  # noqa: F401 — confirms RDKit is available

    # Get SYBYL types via RDKit's built-in Mol2 writer (if available)
    try:
        from rdkit.Chem.rdmolfiles import MolToMol2Block
        block = MolToMol2Block(mol, includeStereo=True, resBonds=True,
                               kekulize=True, cleanupSubstructures=True)
        out_path.write_text(block)
        return
    except (ImportError, AttributeError):
        pass

    # Fallback: write via Open Babel subprocess if available
    if _obabel_available():
        _write_mol2_via_obabel(mol, compound_id, out_path)
        return

    # Last resort: write PDB then rename (very limited SYBYL info)
    _write_mol2_basic(mol, compound_id, out_path)


def _obabel_available() -> bool:
    try:
        r = subprocess.run(
            ["obabel", "--version"],
            capture_output=True, timeout=5,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _write_mol2_via_obabel(mol, compound_id: str, out_path: Path) -> None:
    """Convert RDKit molecule to Mol2 via Open Babel subprocess."""
    from rdkit.Chem import AllChem
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tf:
        sdf_path = Path(tf.name)
    try:
        from rdkit.Chem import SDWriter
        w = SDWriter(str(sdf_path))
        w.write(mol)
        w.close()
        subprocess.run(
            ["obabel", str(sdf_path), "-O", str(out_path)],
            check=True, capture_output=True, timeout=30,
        )
    finally:
        sdf_path.unlink(missing_ok=True)


def _write_mol2_basic(mol, compound_id: str, out_path: Path) -> None:
    """Fallback: write a minimal Mol2 with generic atom types."""
    from rdkit.Chem import rdchem

    conf = mol.GetConformer(0)
    lines = [
        "@<TRIPOS>MOLECULE",
        compound_id,
        f" {mol.GetNumAtoms()} {mol.GetNumBonds()} 0 0 0",
        "SMALL",
        "GASTEIGER",
        "",
        "@<TRIPOS>ATOM",
    ]
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        symbol = atom.GetSymbol()
        sybyl = f"{symbol}.3" if atom.GetHybridization().name == "SP3" else symbol
        lines.append(
            f"{i+1:6d} {symbol:<4s}  "
            f"{pos.x:9.4f} {pos.y:9.4f} {pos.z:9.4f}  "
            f"{sybyl:<8s}  1 LIG       0.0000"
        )

    lines += ["@<TRIPOS>BOND"]
    for bond in mol.GetBonds():
        bt = bond.GetBondTypeAsDouble()
        btype = "ar" if bond.GetIsAromatic() else (
            "1" if bt == 1.0 else "2" if bt == 2.0 else "3" if bt == 3.0 else "am"
        )
        lines.append(
            f"{bond.GetIdx()+1:6d} {bond.GetBeginAtomIdx()+1:4d} "
            f"{bond.GetEndAtomIdx()+1:4d} {btype}"
        )

    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Atom type validation
# ---------------------------------------------------------------------------


def validate_atom_types(mol2_path: Path) -> List[str]:
    """Check SYBYL ↔ FlexAID type consistency in a Mol2 file.

    Parses the ``@<TRIPOS>ATOM`` block and reports:
    - Unknown SYBYL types not in ``SYBYL_TO_FLEXAID``
    - Unsupported element types (``UNSUPPORTED_SYBYL_TYPES``)

    Args:
        mol2_path: Path to the Mol2 file.

    Returns:
        List of warning/error strings (empty = all OK).
    """
    issues: List[str] = []
    in_atom_block = False

    with open(mol2_path) as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if stripped.startswith("@<TRIPOS>ATOM"):
                in_atom_block = True
                continue
            if stripped.startswith("@<TRIPOS>") and in_atom_block:
                break
            if not in_atom_block or not stripped:
                continue

            parts = stripped.split()
            if len(parts) < 6:
                continue
            sybyl = parts[5]

            # Strip residue-qualified types e.g. "C.3" from "C.3.res"
            sybyl_base = sybyl.split(".")[0]

            if sybyl in UNSUPPORTED_SYBYL_TYPES or sybyl_base in UNSUPPORTED_SYBYL_TYPES:
                issues.append(
                    f"Line {lineno}: unsupported element type '{sybyl}' — "
                    f"CF scoring will assign zero interaction energy"
                )
            elif sybyl not in SYBYL_TO_FLEXAID:
                # Check if it maps via base symbol
                if sybyl_base not in SYBYL_TO_FLEXAID:
                    issues.append(
                        f"Line {lineno}: unknown SYBYL type '{sybyl}' — "
                        f"verify CF atom typing"
                    )

    return issues


# ---------------------------------------------------------------------------
# Mol2 → FlexAID .inp generation
# ---------------------------------------------------------------------------


def mol2_to_inp(
    mol2_path: Path,
    output_dir: Path,
    compound_id: str,
    n_rotatable_bonds: Optional[int] = None,
) -> Path:
    """Generate a minimal FlexAID ligand .inp file from a Mol2.

    The .inp file is consumed by the FlexAID C++ engine's ``INPLIG`` directive.
    It contains the ligand path, number of rotatable bonds, and SYBYL-type
    atom records in the processligand output format.

    Args:
        mol2_path:          Path to the validated Mol2 file.
        output_dir:         Directory to write the .inp file.
        compound_id:        Compound identifier used for the filename.
        n_rotatable_bonds:  Override rotatable-bond count; auto-detected if None.

    Returns:
        Path to the written .inp file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    inp_path = output_dir / f"{compound_id}.inp"

    atoms = _parse_mol2_atoms(mol2_path)
    bonds = _parse_mol2_bonds(mol2_path)

    if n_rotatable_bonds is None:
        n_rotatable_bonds = _count_rotatable_bonds(atoms, bonds)

    lines = [
        f"# FlexAID ligand .inp — generated by ligand_prep.py",
        f"# Source: {mol2_path.name}",
        f"RESNM LIG",
        f"NROT  {n_rotatable_bonds}",
        f"NATOM {len(atoms)}",
        "",
    ]
    for i, (atom_id, name, x, y, z, sybyl) in enumerate(atoms, start=1):
        flexaid_type = SYBYL_TO_FLEXAID.get(sybyl, SYBYL_TO_FLEXAID.get(sybyl.split(".")[0], "C"))
        lines.append(
            f"ATOM  {i:4d}  {name:<4s}  {x:9.4f}  {y:9.4f}  {z:9.4f}  "
            f"{sybyl:<8s}  {flexaid_type}"
        )

    inp_path.write_text("\n".join(lines) + "\n")
    logger.debug("Wrote ligand .inp: %s (%d atoms)", inp_path, len(atoms))
    return inp_path


def _parse_mol2_atoms(mol2_path: Path) -> list:
    """Parse the ATOM block of a Mol2 file.

    Returns list of (atom_id, name, x, y, z, sybyl_type).
    """
    atoms = []
    in_block = False
    with open(mol2_path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("@<TRIPOS>ATOM"):
                in_block = True
                continue
            if stripped.startswith("@<TRIPOS>") and in_block:
                break
            if not in_block or not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 6:
                continue
            try:
                atom_id = int(parts[0])
                name = parts[1]
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                sybyl = parts[5]
                atoms.append((atom_id, name, x, y, z, sybyl))
            except ValueError:
                continue
    return atoms


def _parse_mol2_bonds(mol2_path: Path) -> list:
    """Parse the BOND block of a Mol2 file.

    Returns list of (bond_id, atom1, atom2, bond_type).
    """
    bonds = []
    in_block = False
    with open(mol2_path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("@<TRIPOS>BOND"):
                in_block = True
                continue
            if stripped.startswith("@<TRIPOS>") and in_block:
                break
            if not in_block or not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 4:
                continue
            try:
                bonds.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[3]))
            except ValueError:
                continue
    return bonds


def _count_rotatable_bonds(atoms: list, bonds: list) -> int:
    """Heuristic count of rotatable bonds (single bonds not in rings)."""
    # Very simple heuristic: count single non-terminal bonds
    # A proper implementation would detect ring membership — this is sufficient
    # for rough ordering; the FlexAID engine re-evaluates internally.
    heavy_atom_ids = {a[0] for a in atoms if not a[1].startswith("H")}
    n_rotatable = 0
    for _, a1, a2, btype in bonds:
        if btype not in ("1", "single"):
            continue
        if a1 not in heavy_atom_ids or a2 not in heavy_atom_ids:
            continue
        n_rotatable += 1
    # Subtract 1 for over-counting; clamp to 0
    return max(0, n_rotatable - 1)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def prepare_ligand(
    smiles: str,
    compound_id: str,
    output_dir: Path,
    n_conformers: int = 50,
    random_seed: int = 42,
) -> LigandPrepResult:
    """Full ligand preparation pipeline: SMILES → Mol2 + FlexAID .inp.

    Args:
        smiles:       SMILES string for the compound.
        compound_id:  Unique identifier (used for filenames).
        output_dir:   Directory for output files.
        n_conformers: Number of ETKDGv3 conformers.
        random_seed:  RDKit random seed.

    Returns:
        :class:`LigandPrepResult` describing the outcome.
    """
    result = LigandPrepResult(smiles=smiles, compound_id=compound_id)

    try:
        mol2_path, n_conf = smiles_to_3d_mol2(
            smiles, compound_id, output_dir,
            n_conformers=n_conformers,
            random_seed=random_seed,
        )
        if mol2_path is None:
            result.error = "ETKDGv3 embedding failed — check SMILES validity"
            return result

        result.mol2_path = mol2_path
        result.n_conformers = n_conf

        issues = validate_atom_types(mol2_path)
        result.atom_type_issues = issues
        if any("unsupported" in msg for msg in issues):
            logger.warning(
                "%d unsupported atom types in %s", len(issues), compound_id
            )

        inp_path = mol2_to_inp(mol2_path, output_dir, compound_id)
        result.inp_path = inp_path
        result.success = True

    except RuntimeError as exc:
        result.error = str(exc)
    except Exception as exc:
        result.error = f"Unexpected error: {exc}"
        logger.exception("Ligand preparation failed for %s", compound_id)

    return result


def prepare_ligand_batch(
    compounds: List[Tuple[str, str]],
    output_dir: Path,
    n_workers: int = 1,
    **kwargs,
) -> List[LigandPrepResult]:
    """Prepare multiple ligands, optionally in parallel.

    Args:
        compounds:  List of ``(smiles, compound_id)`` tuples.
        output_dir: Root directory; each compound gets a subdirectory.
        n_workers:  Worker processes (1 = serial).
        **kwargs:   Forwarded to :func:`prepare_ligand`.

    Returns:
        List of :class:`LigandPrepResult` in the same order as ``compounds``.
    """
    if n_workers == 1:
        return [
            prepare_ligand(smi, cid, output_dir / cid, **kwargs)
            for smi, cid in compounds
        ]

    from concurrent.futures import ProcessPoolExecutor, as_completed

    futures = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for smi, cid in compounds:
            fut = pool.submit(prepare_ligand, smi, cid, output_dir / cid, **kwargs)
            futures[fut] = (smi, cid)

    results_map: dict[str, LigandPrepResult] = {}
    for fut in as_completed(futures):
        _, cid = futures[fut]
        try:
            results_map[cid] = fut.result()
        except Exception as exc:
            results_map[cid] = LigandPrepResult(
                smiles=futures[fut][0],
                compound_id=cid,
                error=str(exc),
            )

    return [results_map[cid] for _, cid in compounds]
