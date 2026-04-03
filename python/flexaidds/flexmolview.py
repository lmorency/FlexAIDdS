"""Experimental FlexMolView prototype for FlexAIDdS.

This module is a deliberately narrow molecular-view state layer intended for a
future fast viewer. It is pure Python for now, but it is structured so the hot
path can migrate to free-threaded Python, C++/nanobind, Rust/PyO3, Swift, or a
browser frontend without rewriting the public object model.

What this module is:
- a small, testable molecular scene model
- a PDB-backed object store using existing ``flexaidds.io`` parsing
- a minimal PyMOL-like selection subset
- representation visibility state (show/hide/as)
- simple geometry utilities (counts, center, bounds, distances)

What this module is not:
- a GPU renderer
- a promise of "no GIL" inside stock CPython
- a substitute for native hot-path geometry kernels

The design goal is to keep orchestration in Python while leaving the compute and
render backends replaceable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .io import Atom, PDBStructure, is_ion, read_pdb

_ALLOWED_REPRESENTATIONS = {"lines", "sticks", "spheres", "cartoon", "surface", "labels"}
_SOLVENT_NAMES = {"HOH", "WAT", "DOD", "SOL"}


@dataclass(slots=True)
class ViewerObject:
    """One loaded molecular object inside the experimental scene."""

    name: str
    path: Path
    structure: PDBStructure
    visible_representations: Set[str] = field(default_factory=lambda: {"lines"})

    @property
    def atoms(self) -> List[Atom]:
        return self.structure.atoms


@dataclass(slots=True)
class SelectionResult:
    """Materialized atom selection over a specific object."""

    object_name: str
    expression: str
    atom_indices: List[int]

    @property
    def count(self) -> int:
        return len(self.atom_indices)


class FlexMolView:
    """Experimental scene/state controller for molecular inspection.

    The API is intentionally tiny and command-oriented so it can be wrapped by a
    CLI, a Qt/Swift shell, or a web bridge later.
    """

    def __init__(self) -> None:
        self.objects: Dict[str, ViewerObject] = {}
        self.named_selections: Dict[str, SelectionResult] = {}

    def load_pdb(self, path: str, object_name: Optional[str] = None) -> str:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"PDB not found: {path}")
        name = object_name or p.stem
        if name in self.objects:
            raise ValueError(f"Object already exists: {name}")
        self.objects[name] = ViewerObject(name=name, path=p, structure=read_pdb(str(p)))
        return name

    def delete(self, object_name: str) -> None:
        self._require_object(object_name)
        del self.objects[object_name]
        stale = [key for key, sel in self.named_selections.items() if sel.object_name == object_name]
        for key in stale:
            del self.named_selections[key]

    def list_objects(self) -> List[str]:
        return sorted(self.objects)

    def show(self, representation: str, object_name: str) -> None:
        rep = self._normalize_representation(representation)
        obj = self._require_object(object_name)
        obj.visible_representations.add(rep)

    def hide(self, representation: str, object_name: str) -> None:
        rep = self._normalize_representation(representation)
        obj = self._require_object(object_name)
        obj.visible_representations.discard(rep)

    def as_representation(self, representation: str, object_name: str) -> None:
        rep = self._normalize_representation(representation)
        obj = self._require_object(object_name)
        obj.visible_representations = {rep}

    def get_visible_representations(self, object_name: str) -> Set[str]:
        return set(self._require_object(object_name).visible_representations)

    def select(self, expression: str, object_name: str, *, name: Optional[str] = None) -> SelectionResult:
        obj = self._require_object(object_name)
        indices = sorted(self._evaluate_expression(obj, expression))
        result = SelectionResult(object_name=object_name, expression=expression, atom_indices=indices)
        if name:
            self.named_selections[name] = result
        return result

    def get_selection(self, name: str) -> SelectionResult:
        if name not in self.named_selections:
            raise KeyError(f"Unknown selection: {name}")
        return self.named_selections[name]

    def count_atoms(self, object_name: str, expression: str = "all") -> int:
        return self.select(expression, object_name).count

    def count_residues(self, object_name: str, expression: str = "all") -> int:
        obj = self._require_object(object_name)
        indices = self._evaluate_expression(obj, expression)
        residues = {
            (obj.atoms[i].chainid, obj.atoms[i].resseq, obj.atoms[i].icode)
            for i in indices
        }
        return len(residues)

    def center_of_geometry(self, object_name: str, expression: str = "all") -> Tuple[float, float, float]:
        obj = self._require_object(object_name)
        indices = self._non_empty_indices(obj, expression)
        sx = sy = sz = 0.0
        for i in indices:
            atom = obj.atoms[i]
            sx += atom.x
            sy += atom.y
            sz += atom.z
        n = float(len(indices))
        return (sx / n, sy / n, sz / n)

    def bounds(self, object_name: str, expression: str = "all") -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        obj = self._require_object(object_name)
        indices = self._non_empty_indices(obj, expression)
        xs = [obj.atoms[i].x for i in indices]
        ys = [obj.atoms[i].y for i in indices]
        zs = [obj.atoms[i].z for i in indices]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    def distance_by_serial(self, object_name: str, serial_a: int, serial_b: int) -> float:
        obj = self._require_object(object_name)
        atom_a = next((a for a in obj.atoms if a.serial == serial_a), None)
        atom_b = next((a for a in obj.atoms if a.serial == serial_b), None)
        if atom_a is None or atom_b is None:
            raise KeyError("Both atom serials must exist in the object")
        dx = atom_a.x - atom_b.x
        dy = atom_a.y - atom_b.y
        dz = atom_a.z - atom_b.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def summary(self, object_name: str) -> Dict[str, object]:
        obj = self._require_object(object_name)
        chains = sorted({a.chainid for a in obj.atoms if a.chainid})
        residues = {(a.chainid, a.resseq, a.icode) for a in obj.atoms}
        ligands = {
            (a.chainid, a.resname, a.resseq)
            for a in obj.atoms
            if self._match_keyword(a, "ligand")
        }
        return {
            "object": obj.name,
            "path": str(obj.path),
            "atom_count": len(obj.atoms),
            "residue_count": len(residues),
            "chains": chains,
            "ligands": sorted(ligands),
            "visible_representations": sorted(obj.visible_representations),
        }

    def _require_object(self, object_name: str) -> ViewerObject:
        if object_name not in self.objects:
            raise KeyError(f"Unknown object: {object_name}")
        return self.objects[object_name]

    def _non_empty_indices(self, obj: ViewerObject, expression: str) -> List[int]:
        indices = sorted(self._evaluate_expression(obj, expression))
        if not indices:
            raise ValueError(f"Selection is empty: {expression}")
        return indices

    def _normalize_representation(self, representation: str) -> str:
        rep = representation.strip().lower()
        if rep not in _ALLOWED_REPRESENTATIONS:
            allowed = ", ".join(sorted(_ALLOWED_REPRESENTATIONS))
            raise ValueError(f"Unsupported representation '{representation}'. Allowed: {allowed}")
        return rep

    def _evaluate_expression(self, obj: ViewerObject, expression: str) -> Set[int]:
        tokens = self._tokenize(expression)
        if not tokens:
            raise ValueError("Selection expression is empty")
        value, next_pos = self._parse_or(obj, tokens, 0)
        if next_pos != len(tokens):
            raise ValueError(f"Unexpected trailing tokens in selection: {' '.join(tokens[next_pos:])}")
        return value

    def _parse_or(self, obj: ViewerObject, tokens: List[str], pos: int) -> Tuple[Set[int], int]:
        left, pos = self._parse_and(obj, tokens, pos)
        while pos < len(tokens) and tokens[pos].lower() == "or":
            right, pos = self._parse_and(obj, tokens, pos + 1)
            left |= right
        return left, pos

    def _parse_and(self, obj: ViewerObject, tokens: List[str], pos: int) -> Tuple[Set[int], int]:
        left, pos = self._parse_not(obj, tokens, pos)
        while pos < len(tokens) and tokens[pos].lower() == "and":
            right, pos = self._parse_not(obj, tokens, pos + 1)
            left &= right
        return left, pos

    def _parse_not(self, obj: ViewerObject, tokens: List[str], pos: int) -> Tuple[Set[int], int]:
        if pos < len(tokens) and tokens[pos].lower() == "not":
            inner, pos = self._parse_not(obj, tokens, pos + 1)
            return set(range(len(obj.atoms))) - inner, pos
        return self._parse_primary(obj, tokens, pos)

    def _parse_primary(self, obj: ViewerObject, tokens: List[str], pos: int) -> Tuple[Set[int], int]:
        if pos >= len(tokens):
            raise ValueError("Unexpected end of selection expression")
        token = tokens[pos]
        if token == "(":
            value, pos = self._parse_or(obj, tokens, pos + 1)
            if pos >= len(tokens) or tokens[pos] != ")":
                raise ValueError("Missing closing parenthesis in selection expression")
            return value, pos + 1
        return self._parse_predicate(obj, tokens, pos)

    def _parse_predicate(self, obj: ViewerObject, tokens: List[str], pos: int) -> Tuple[Set[int], int]:
        key = tokens[pos].lower()
        if key in {"all", "polymer", "ligand", "solvent"}:
            indices = {i for i, atom in enumerate(obj.atoms) if self._match_keyword(atom, key)}
            return indices, pos + 1
        if key in {"chain", "resi", "resn", "name", "elem"}:
            if pos + 1 >= len(tokens):
                raise ValueError(f"Selection token '{key}' requires a value")
            value = tokens[pos + 1]
            indices = {
                i for i, atom in enumerate(obj.atoms)
                if self._match_field(atom, key, value)
            }
            return indices, pos + 2
        raise ValueError(f"Unsupported selection token: {tokens[pos]}")

    def _tokenize(self, expression: str) -> List[str]:
        raw = expression.strip()
        if not raw:
            return []
        spaced = raw.replace("(", " ( ").replace(")", " ) ")
        return [tok for tok in spaced.split() if tok]

    def _match_keyword(self, atom: Atom, key: str) -> bool:
        if key == "all":
            return True
        if key == "polymer":
            return atom.record == "ATOM"
        if key == "solvent":
            return atom.resname.strip().upper() in _SOLVENT_NAMES
        if key == "ligand":
            return atom.record == "HETATM" and not is_ion(atom) and atom.resname.strip().upper() not in _SOLVENT_NAMES
        raise ValueError(f"Unsupported keyword: {key}")

    def _match_field(self, atom: Atom, key: str, value: str) -> bool:
        if key == "chain":
            return atom.chainid.strip().upper() == value.strip().upper()
        if key == "resn":
            return atom.resname.strip().upper() == value.strip().upper()
        if key == "name":
            return atom.name.strip().upper() == value.strip().upper()
        if key == "elem":
            elem = atom.element.strip() or atom.name.strip()[0:1]
            return elem.upper() == value.strip().upper()
        if key == "resi":
            return self._match_resi(atom, value)
        raise ValueError(f"Unsupported field selector: {key}")

    def _match_resi(self, atom: Atom, value: str) -> bool:
        spec = value.strip()
        if re.fullmatch(r"-?\d+", spec):
            return atom.resseq == int(spec)
        if re.fullmatch(r"-?\d+-\d+", spec):
            start_s, end_s = spec.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            lo, hi = (start, end) if start <= end else (end, start)
            return lo <= atom.resseq <= hi
        raise ValueError(f"Unsupported resi selector: {value}")


__all__ = ["FlexMolView", "SelectionResult", "ViewerObject"]
