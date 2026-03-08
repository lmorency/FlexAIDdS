from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .models import PoseResult

_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_FILE_MODE_PATTERNS = [
    re.compile(r"binding[_-]?mode[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"\bmode[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"cluster[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"\bbm[_-]?(\d+)\b", re.IGNORECASE),
]
_FILE_POSE_PATTERNS = [
    re.compile(r"pose[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"conformer[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"model[_-]?(\d+)", re.IGNORECASE),
]


def _normalize_key(raw: str) -> str:
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
    for key in keys:
        value = remarks.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def parse_pose_result(path: Path) -> PoseResult:
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
