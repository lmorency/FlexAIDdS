"""PDB I/O and REMARK-header parsing for FlexAID∆S docking output files.

FlexAID∆S writes thermodynamic quantities (free energy, entropy, heat
capacity, etc.) as ``REMARK`` records in each output PDB file.  This module
provides low-level parsers that extract those values and infer binding-mode
and pose-rank identifiers from both REMARK content and file names.

Public API:
    - :func:`parse_remark_map` – parse all REMARK lines into a key→value dict.
    - :func:`infer_mode_id` – determine which binding mode a PDB file belongs to.
    - :func:`infer_pose_rank` – determine the pose rank within its mode.
    - :func:`parse_pose_result` – full parse of one PDB file into a
      :class:`~flexaidds.models.PoseResult`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .models import PoseResult

_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_FILE_MODE_PATTERNS = [
    re.compile(r"binding[_-]?mode[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"\bmode[_-]?(\d+)\b", re.IGNORECASE),
    re.compile(r"cluster[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"\bbm[_-]?(\d+)\b", re.IGNORECASE),
]
_FILE_POSE_PATTERNS = [
    re.compile(r"pose[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"conformer[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"model[_-]?(\d+)", re.IGNORECASE),
]


def _normalize_key(raw: str) -> str:
    """Normalise a raw REMARK key to a canonical snake_case identifier.

    Steps:
    1. Strip whitespace and convert to lowercase.
    2. Replace runs of non-alphanumeric characters with ``_``.
    3. Apply a fixed alias table to unify variant spellings
       (e.g. ``dg`` → ``free_energy``, ``cv`` → ``heat_capacity``).

    Args:
        raw: The raw key string extracted from a REMARK line.

    Returns:
        Normalised key string (may differ from the input).
    """
    key = raw.strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")
    aliases = {
        "bindingmode": "binding_mode",
        "binding_mode_id": "binding_mode",
        "modeid": "binding_mode",
        "clusterid": "cluster_id",
        "cf_app": "cf_app",
        "cfapp": "cf_app",
        "app_cf": "cf_app",
        "delta_g": "free_energy",
        "dg": "free_energy",
        "freeenergy": "free_energy",
        "enthalpy_like": "enthalpy",
        "energy_std": "std_energy",
        "sigma_energy": "std_energy",
        "cv": "heat_capacity",
        "temp": "temperature",
    }
    return aliases.get(key, key)


def _coerce_value(raw: str) -> Any:
    """Convert a raw REMARK value string to an appropriate Python type.

    Conversion priority:
    1. Numeric string → ``float`` (or ``int`` if the value is a whole number).
    2. ``"true"``/``"false"`` (case-insensitive) → ``bool``.
    3. Everything else → stripped ``str``.

    Args:
        raw: The raw value string from a REMARK line.

    Returns:
        Coerced Python value.
    """
    value = raw.strip().strip(";,")
    if not value:
        return value
    if _NUMERIC_RE.match(value):
        number = float(value)
        return int(number) if number.is_integer() else number
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return value


def parse_remark_map(lines: Iterable[str]) -> Dict[str, Any]:
    """Parse ``REMARK`` records from PDB lines into a key→value dictionary.

    Supported REMARK formats:
    - ``REMARK Key = value`` (``=`` or ``:`` delimiter)
    - ``REMARK Key value`` (space delimiter, first-seen wins)

    Keys are normalised via :func:`_normalize_key`; values are coerced via
    :func:`_coerce_value`.  Non-REMARK lines are silently ignored.

    Args:
        lines: Iterable of PDB file lines (strings).

    Returns:
        Dictionary mapping normalised keys to coerced values.  If the same
        key appears multiple times the first occurrence wins.
    """
    remarks: Dict[str, Any] = {}
    for line in lines:
        if not line.startswith("REMARK"):
            continue
        payload = line[6:].strip()
        if not payload:
            continue

        match = re.match(r"([A-Za-z][A-Za-z0-9 ._\-/]*)\s*(?:=|:)\s*(.+)", payload)
        if match:
            key = _normalize_key(match.group(1))
            remarks[key] = _coerce_value(match.group(2))
            continue

        match = re.match(r"([A-Za-z][A-Za-z0-9_\-/]*)\s+(.+)", payload)
        if match:
            key = _normalize_key(match.group(1))
            if key not in remarks:
                remarks[key] = _coerce_value(match.group(2))
    return remarks


def infer_mode_id(path: Path, remarks: Dict[str, Any]) -> int:
    """Determine the binding-mode ID for a docking output PDB file.

    Lookup order:
    1. REMARK keys ``binding_mode``, ``mode``, ``cluster_id``, ``cluster``.
    2. Regex patterns applied to the file stem
       (e.g. ``mode3``, ``binding_mode_2``, ``cluster1``).
    3. Fallback: ``1``.

    Args:
        path: Path to the PDB file (used to extract filename patterns).
        remarks: Parsed REMARK map from :func:`parse_remark_map`.

    Returns:
        Integer binding-mode identifier (≥ 1).
    """
    for key in ("binding_mode", "mode", "cluster_id", "cluster"):
        value = remarks.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    name = path.stem
    for pattern in _FILE_MODE_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return 1


def infer_pose_rank(path: Path, remarks: Dict[str, Any]) -> int:
    """Determine the pose rank within a binding mode for a docking output PDB.

    Lookup order:
    1. REMARK keys ``pose_rank``, ``rank``, ``pose``, ``model``.
    2. Regex patterns applied to the file stem
       (e.g. ``pose5``, ``conformer_2``, ``model3``).
    3. Fallback: ``1``.

    Args:
        path: Path to the PDB file (used to extract filename patterns).
        remarks: Parsed REMARK map from :func:`parse_remark_map`.

    Returns:
        Integer pose rank (≥ 1).
    """
    for key in ("pose_rank", "rank", "pose", "model"):
        value = remarks.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    name = path.stem
    for pattern in _FILE_POSE_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return 1


def _first_float(remarks: Dict[str, Any], *keys: str) -> Optional[float]:
    """Return the first numeric value found for any of the given REMARK *keys*.

    Args:
        remarks: Parsed REMARK dictionary.
        *keys: One or more candidate key names, tried in order.

    Returns:
        The value cast to ``float``, or ``None`` if no key is present.
    """
    for key in keys:
        value = remarks.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def parse_pose_result(path: Path) -> PoseResult:
    """Parse one FlexAID∆S output PDB file into a :class:`~flexaidds.models.PoseResult`.

    Reads the file, extracts all REMARK fields, infers the binding-mode ID
    and pose rank, then returns a fully populated (immutable) ``PoseResult``.

    Args:
        path: Absolute or relative path to the PDB output file.

    Returns:
        :class:`~flexaidds.models.PoseResult` with all available fields
        populated from the PDB REMARK section.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    remarks = parse_remark_map(text.splitlines())
    mode_id = infer_mode_id(path, remarks)
    pose_rank = infer_pose_rank(path, remarks)

    cf_app = _first_float(remarks, "cf_app", "cfapp", "apparent_cf")
    cf = _first_float(remarks, "cf", "complementarity_function", "score")
    if cf is None:
        cf = cf_app

    return PoseResult(
        path=path,
        mode_id=mode_id,
        pose_rank=pose_rank,
        cf=cf,
        cf_app=cf_app,
        rmsd_raw=_first_float(remarks, "rmsd_raw", "rmsd", "rmsd_unsym"),
        rmsd_sym=_first_float(remarks, "rmsd_sym", "rmsd_symmetric", "sym_rmsd"),
        free_energy=_first_float(remarks, "free_energy", "f"),
        enthalpy=_first_float(remarks, "enthalpy", "h", "mean_energy"),
        entropy=_first_float(remarks, "entropy", "s"),
        heat_capacity=_first_float(remarks, "heat_capacity", "cv"),
        std_energy=_first_float(remarks, "std_energy", "sigma_energy"),
        temperature=_first_float(remarks, "temperature", "temp"),
        remarks=remarks,
    )
