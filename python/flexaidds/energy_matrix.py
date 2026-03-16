"""Energy matrix I/O and 256-type projection for FlexAID∆S.

Bridges between:
  - Legacy FlexAID ``.dat`` format (upper-triangle, scalar or density-function entries)
  - 256×256 binary blob format (SHNN magic header, float32)
  - NumPy 2D array representation

The 256-type encoding uses 8 bits: bits 0–4 = base type (32 classes extending
SYBYL), bits 5–6 = AM1-BCC charge bin (4 levels), bit 7 = H-bond donor/acceptor
flag.  The ``project_to_40()`` method collapses a 256×256 matrix back to the
40-type SYBYL system used by the C++ Voronoi contact function.

Example:
    >>> mat = EnergyMatrix.from_dat_file("WRK/nrg_mat_BEST_012012.dat")
    >>> print(mat.ntypes, mat.matrix.shape)
    10 (10, 10)
    >>> mat.to_dat_file("/tmp/roundtrip.dat")
"""

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _validate_matrix(matrix: "np.ndarray", label: str = "matrix") -> None:
    """Raise ValueError if matrix contains NaN or Inf."""
    if np is not None and not np.all(np.isfinite(matrix)):
        n_nan = int(np.sum(np.isnan(matrix)))
        n_inf = int(np.sum(np.isinf(matrix)))
        raise ValueError(
            f"{label} contains non-finite values: {n_nan} NaN, {n_inf} Inf"
        )


# ── constants ────────────────────────────────────────────────────────────────

SHNN_MAGIC = b"SHNN"  # binary blob magic header
SHNN_VERSION = 1
MATRIX_256_SIZE = 256

# SYBYL atom type names (1-indexed in C++, 0-indexed here)
SYBYL_TYPE_NAMES: List[str] = [
    "C.1", "C.2", "C.3", "C.AR", "C.CAT",          # 1-5
    "N.1", "N.2", "N.3", "N.4", "N.AR", "N.AM", "N.PL3",  # 6-12
    "O.2", "O.3", "O.CO2", "O.AR",                  # 13-16
    "S.2", "S.3", "S.O", "S.O2", "S.AR",            # 17-21
    "P.3", "F", "CL", "BR", "I", "SE",              # 22-27
    "MG", "SR", "CU", "MN", "HG", "CD", "NI", "ZN", "CA", "FE", "CO.OH",  # 28-38
    "DUMMY", "SOLVENT",                               # 39-40
]

SYBYL_RADII: Dict[int, float] = {
    1: 1.88, 2: 1.72, 3: 1.88, 4: 1.76, 5: 1.88,
    6: 1.64, 7: 1.64, 8: 1.64, 9: 1.64, 10: 1.64, 11: 1.64, 12: 1.64,
    13: 1.42, 14: 1.46, 15: 1.46, 16: 1.46,
    17: 1.782, 18: 1.782, 19: 1.782, 20: 1.782, 21: 1.782,
    22: 1.871, 23: 1.560, 24: 1.735, 25: 1.978, 26: 2.094, 27: 1.9,
    28: 0.72, 29: 1.18, 30: 1.18, 31: 0.73, 32: 1.02, 33: 0.95,
    34: 0.69, 35: 0.74, 36: 1.00, 37: 0.61, 38: 0.65,
    39: 2.0, 40: 0.0,
}

# Roman numeral labels used in legacy 10-type .dat files
_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
          "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]


# ── 256-type encoding helpers ────────────────────────────────────────────────

def encode_256_type(base_type: int, charge_bin: int = 0,
                    hbond_flag: bool = False) -> int:
    """Encode a 256-type atom index from components.

    Args:
        base_type:  Base type index (0–31).  Types 0–31 extend SYBYL's 40
                    classes into 32 canonical base types.
        charge_bin: AM1-BCC partial charge quantile (0–3).
        hbond_flag: True if atom is an H-bond donor or acceptor.

    Returns:
        8-bit integer (0–255).
    """
    base_type = max(0, min(31, int(base_type)))
    charge_bin = max(0, min(3, int(charge_bin)))
    return (int(hbond_flag) << 7) | (charge_bin << 5) | base_type


def decode_256_type(code: int) -> Tuple[int, int, bool]:
    """Decode a 256-type code into (base_type, charge_bin, hbond_flag)."""
    code = int(code) & 0xFF
    base_type = code & 0x1F
    charge_bin = (code >> 5) & 0x03
    hbond_flag = bool((code >> 7) & 0x01)
    return base_type, charge_bin, hbond_flag


# ── 256 → 40 SYBYL projection mapping ───────────────────────────────────────

# Maps each of 32 base types to its SYBYL parent (1-indexed, matching C++).
# Base types 0–31 are a superset of SYBYL types with context-aware splits
# (e.g., C_ar_hetadj, C_pi_bridging).  Types beyond the first 22 collapse
# onto nearby SYBYL parents.
_BASE_TO_SYBYL: List[int] = [
    # 0–4: carbon variants → SYBYL carbon types
    1,   # 0: C_sp  → C.1
    2,   # 1: C_sp2 → C.2
    3,   # 2: C_sp3 → C.3
    4,   # 3: C_ar  → C.AR
    5,   # 4: C_cat → C.CAT
    # 5–11: nitrogen variants
    6,   # 5: N_sp  → N.1
    7,   # 6: N_sp2 → N.2
    8,   # 7: N_sp3 → N.3
    9,   # 8: N_quat → N.4
    10,  # 9: N_ar  → N.AR
    11,  # 10: N_am → N.AM
    12,  # 11: N_pl3 → N.PL3
    # 12–15: oxygen variants
    13,  # 12: O_sp2 → O.2
    14,  # 13: O_sp3 → O.3
    15,  # 14: O_co2 → O.CO2
    16,  # 15: O_ar  → O.AR
    # 16–20: sulfur variants
    17,  # 16: S_sp2 → S.2
    18,  # 17: S_sp3 → S.3
    19,  # 18: S_oxide → S.O
    20,  # 19: S_dioxide → S.O2
    21,  # 20: S_ar → S.AR
    # 21: phosphorus
    22,  # 21: P_sp3 → P.3
    # 22–25: halogens
    23,  # 22: F → F
    24,  # 23: Cl → CL
    25,  # 24: Br → BR
    26,  # 25: I → I
    # 26–27: extended carbon (NATURaL-critical π-system refinements)
    4,   # 26: C_ar_hetadj → C.AR (aromatic C adjacent to heteroatom)
    2,   # 27: C_pi_bridging → C.2 (π-bridging carbon, tryptamine/indole)
    # 28–30: metals (collapsed)
    35,  # 28: Zn → ZN
    36,  # 29: Ca → CA
    37,  # 30: Fe → FE
    # 31: solvent / dummy
    40,  # 31: solvent → SOLVENT
]


def base_to_sybyl(base_type: int) -> int:
    """Map a base type (0–31) to its SYBYL parent (1–40)."""
    if 0 <= base_type < len(_BASE_TO_SYBYL):
        return _BASE_TO_SYBYL[base_type]
    return 39  # DUMMY fallback


def sybyl_to_base(sybyl_type: int) -> int:
    """Map a SYBYL type (1–40) to the canonical base type (0–31).

    This is a many-to-one inverse of base_to_sybyl; context-aware
    refinements (C_ar_hetadj, C_pi_bridging) are NOT applied here —
    they require structural context.  Use ``refine_base_type()`` for that.
    """
    # Build reverse mapping (first match)
    for i, s in enumerate(_BASE_TO_SYBYL):
        if s == sybyl_type:
            return i
    return 31  # solvent fallback


# ── density function entry ───────────────────────────────────────────────────

@dataclass
class DensityPoint:
    """Single (x, y) point in a piecewise-linear density function."""
    x: float
    y: float


@dataclass
class MatrixEntry:
    """Energy matrix entry for a pair of atom types.

    For scalar entries (``is_scalar=True``), ``scalar_value`` holds the
    single energy value.  For density-function entries, ``density_points``
    stores the piecewise-linear (x, y) curve.
    """
    type1: int
    type2: int
    is_scalar: bool = True
    scalar_value: float = 0.0
    density_points: List[DensityPoint] = field(default_factory=list)

    def evaluate(self, relative_area: float) -> float:
        """Evaluate the energy for a given normalised contact area.

        Mirrors ``get_yval()`` in ``vcfunction.cpp``.
        """
        if self.is_scalar:
            return self.scalar_value

        if not self.density_points:
            return 0.0

        pts = self.density_points
        # Find bracketing interval
        if relative_area <= pts[0].x:
            return 0.0  # no left-bound data
        for k in range(len(pts) - 1):
            if relative_area <= pts[k + 1].x:
                # linear interpolation
                dx = pts[k + 1].x - pts[k].x
                if abs(dx) < 1e-12:
                    return pts[k].y
                frac = (relative_area - pts[k].x) / dx
                return pts[k].y + frac * (pts[k + 1].y - pts[k].y)
        # beyond last point
        return pts[-1].y


# ── EnergyMatrix class ───────────────────────────────────────────────────────

class EnergyMatrix:
    """Read, write, and manipulate FlexAID energy matrices.

    Supports both legacy N×N scalar matrices and 256×256 binary blobs.

    Attributes:
        ntypes:  Number of atom types (e.g. 10, 40, or 256).
        matrix:  NumPy float64 array of shape ``(ntypes, ntypes)``.
                 Symmetric: ``matrix[i,j] == matrix[j,i]``.
        entries: Optional dict of ``(i,j) → MatrixEntry`` for density-function
                 entries.  Only populated when loaded from a ``.dat`` file that
                 contains density functions.
    """

    def __init__(self, ntypes: int,
                 matrix: "np.ndarray" = None,
                 entries: Optional[Dict[Tuple[int, int], MatrixEntry]] = None):
        if np is None:
            raise RuntimeError("NumPy is required for EnergyMatrix")
        self.ntypes = int(ntypes)
        if self.ntypes <= 0:
            raise ValueError(f"ntypes must be positive, got {self.ntypes}")
        if matrix is None:
            self.matrix = np.zeros((self.ntypes, self.ntypes), dtype=np.float64)
        else:
            self.matrix = np.asarray(matrix, dtype=np.float64)
        if self.matrix.shape != (self.ntypes, self.ntypes):
            raise ValueError(
                f"Matrix shape {self.matrix.shape} does not match "
                f"ntypes={self.ntypes}; expected ({self.ntypes}, {self.ntypes})"
            )
        _validate_matrix(self.matrix, "EnergyMatrix")
        self.entries = entries or {}

    # ── legacy .dat I/O ──────────────────────────────────────────────────

    @classmethod
    def from_dat_file(cls, path: str) -> "EnergyMatrix":
        """Parse a legacy FlexAID ``.dat`` energy matrix file.

        Format: upper-triangle, N*(N+1)/2 lines.  Each line is either a
        single scalar or space-separated x,y pairs (density function).
        Lines may have a ``TYPE-TYPE = value`` prefix.

        Matches the C++ parser in ``read_emat.cpp``.
        """
        path = str(path)
        with open(path) as fh:
            raw_lines = fh.readlines()

        # Combine continuation lines (lines that don't contain a newline-
        # terminated complete entry are appended to the previous line).
        lines: List[str] = []
        current = ""
        for raw in raw_lines:
            current += raw
            if "\n" in current:
                lines.append(current)
                current = ""
        if current.strip():
            lines.append(current)

        nlines = len(lines)
        # Solve N*(N+1)/2 = nlines  →  N = (-1 + sqrt(1 + 8*nlines)) / 2
        discriminant = 1.0 + 8.0 * nlines
        ntypes = int((-1.0 + math.sqrt(discriminant)) / 2.0 + 0.001)

        if ntypes * (ntypes + 1) // 2 != nlines:
            raise ValueError(
                f"Invalid line count {nlines} in energy matrix file {path}; "
                f"expected N*(N+1)/2 lines for some integer N"
            )

        matrix = np.zeros((ntypes, ntypes), dtype=np.float64)
        entries: Dict[Tuple[int, int], MatrixEntry] = {}
        line_idx = 0

        for i in range(ntypes):
            for j in range(i, ntypes):
                line = lines[line_idx]
                line_idx += 1

                # Strip TYPE-TYPE prefix if present
                if "=" in line:
                    line = line[line.index("=") + 1:]
                tokens = line.split()

                entry = MatrixEntry(type1=i, type2=j)

                if len(tokens) == 1:
                    # Scalar entry
                    val = float(tokens[0])
                    entry.is_scalar = True
                    entry.scalar_value = val
                    matrix[i, j] = val
                    matrix[j, i] = val
                elif len(tokens) % 2 == 0:
                    # Density function (x,y pairs)
                    entry.is_scalar = False
                    for k in range(0, len(tokens), 2):
                        entry.density_points.append(
                            DensityPoint(x=float(tokens[k]),
                                         y=float(tokens[k + 1])))
                    # Store the mean y-value as the matrix scalar approximation
                    if entry.density_points:
                        mean_y = sum(p.y for p in entry.density_points) / len(entry.density_points)
                    else:
                        mean_y = 0.0
                    matrix[i, j] = mean_y
                    matrix[j, i] = mean_y
                else:
                    raise ValueError(
                        f"Invalid token count {len(tokens)} for pair ({i},{j}) "
                        f"in {path}"
                    )

                entries[(i, j)] = entry
                if i != j:
                    entries[(j, i)] = MatrixEntry(
                        type1=j, type2=i,
                        is_scalar=entry.is_scalar,
                        scalar_value=entry.scalar_value,
                        density_points=entry.density_points,
                    )

        return cls(ntypes, matrix, entries)

    def to_dat_file(self, path: str, labels: Optional[List[str]] = None) -> None:
        """Write legacy ``.dat`` format consumable by C++ ``read_emat.cpp``.

        Args:
            path:   Output file path.
            labels: Optional list of type labels (e.g. Roman numerals).
                    If None, Roman numerals are used for ntypes <= 20,
                    otherwise numeric indices.
        """
        if labels is None:
            if self.ntypes <= len(_ROMAN):
                labels = _ROMAN[:self.ntypes]
            else:
                labels = [str(i + 1) for i in range(self.ntypes)]

        with open(path, "w") as fh:
            for i in range(self.ntypes):
                for j in range(i, self.ntypes):
                    key = (i, j)
                    label = f"{labels[i]}-{labels[j]}"
                    # Right-align label to 10 chars
                    label_padded = label.rjust(10)

                    if key in self.entries and not self.entries[key].is_scalar:
                        # Density function
                        pts = self.entries[key].density_points
                        vals = " ".join(
                            f"{p.x:.4f} {p.y:.4f}" for p in pts)
                        fh.write(f"{label_padded} = {vals}\n")
                    else:
                        val = self.matrix[i, j]
                        fh.write(f"{label_padded} = {val:10.4f}\n")

    # ── 256×256 binary I/O ───────────────────────────────────────────────

    @classmethod
    def from_binary(cls, path: str) -> "EnergyMatrix":
        """Load a 256×256 matrix from binary blob (SHNN magic header).

        Binary format:
          - 4 bytes: ``SHNN`` magic
          - 4 bytes: uint32 version
          - 4 bytes: uint32 matrix dimension (256)
          - 256*256*4 bytes: float32 matrix data (row-major)
        """
        path = str(path)
        with open(path, "rb") as fh:
            magic = fh.read(4)
            if magic != SHNN_MAGIC:
                raise ValueError(
                    f"Invalid magic header: expected {SHNN_MAGIC!r}, got {magic!r}")
            version, dim = struct.unpack("<II", fh.read(8))
            if version != SHNN_VERSION:
                import warnings
                warnings.warn(
                    f"Binary matrix version {version} does not match expected "
                    f"version {SHNN_VERSION}; data may be incompatible",
                    UserWarning,
                    stacklevel=2,
                )
            if dim != MATRIX_256_SIZE:
                raise ValueError(
                    f"Expected {MATRIX_256_SIZE}×{MATRIX_256_SIZE} matrix, "
                    f"got {dim}×{dim}")
            data = fh.read(dim * dim * 4)
            if len(data) != dim * dim * 4:
                raise ValueError("Truncated binary matrix data")

        flat = np.frombuffer(data, dtype=np.float32).astype(np.float64)
        matrix = flat.reshape((dim, dim))
        return cls(dim, matrix)

    def to_binary(self, path: str) -> None:
        """Write 256×256 binary blob with SHNN magic header."""
        if self.ntypes != MATRIX_256_SIZE:
            raise ValueError(
                f"Binary format requires {MATRIX_256_SIZE}×{MATRIX_256_SIZE} "
                f"matrix, got {self.ntypes}×{self.ntypes}")
        _validate_matrix(self.matrix, "EnergyMatrix.to_binary")
        with open(path, "wb") as fh:
            fh.write(SHNN_MAGIC)
            fh.write(struct.pack("<II", SHNN_VERSION, self.ntypes))
            fh.write(self.matrix.astype(np.float32).tobytes())

    # ── projection ───────────────────────────────────────────────────────

    def project_to_40(self) -> "EnergyMatrix":
        """Project a 256×256 matrix to 40×40 SYBYL via base_to_sybyl_parent().

        For each SYBYL type pair (s1, s2), averages over all 256-type codes
        whose base type maps to s1 and s2 respectively.

        Returns:
            New ``EnergyMatrix`` with ``ntypes=40``.
        """
        if self.ntypes != MATRIX_256_SIZE:
            raise ValueError("project_to_40 requires a 256×256 matrix")

        out = np.zeros((40, 40), dtype=np.float64)
        counts = np.zeros((40, 40), dtype=np.float64)

        for code_i in range(MATRIX_256_SIZE):
            base_i, _, _ = decode_256_type(code_i)
            sybyl_i = base_to_sybyl(base_i) - 1  # 0-indexed

            for code_j in range(MATRIX_256_SIZE):
                base_j, _, _ = decode_256_type(code_j)
                sybyl_j = base_to_sybyl(base_j) - 1

                if 0 <= sybyl_i < 40 and 0 <= sybyl_j < 40:
                    out[sybyl_i, sybyl_j] += self.matrix[code_i, code_j]
                    counts[sybyl_i, sybyl_j] += 1.0

        # Average where counts > 0
        mask = counts > 0
        out[mask] /= counts[mask]

        return EnergyMatrix(40, out)

    # ── lookup ───────────────────────────────────────────────────────────

    def lookup(self, type_i: int, type_j: int) -> float:
        """O(1) matrix lookup.

        Args:
            type_i: First atom type index (0-indexed).
            type_j: Second atom type index (0-indexed).

        Returns:
            Interaction energy value.
        """
        return float(self.matrix[type_i, type_j])

    def evaluate(self, type_i: int, type_j: int,
                 relative_area: float = 0.0) -> float:
        """Evaluate interaction energy, using density function if available.

        For scalar entries, returns the scalar value.
        For density-function entries, interpolates at ``relative_area``.
        """
        key = (type_i, type_j)
        if key in self.entries:
            return self.entries[key].evaluate(relative_area)
        return self.lookup(type_i, type_j)

    # ── utilities ────────────────────────────────────────────────────────

    @property
    def is_symmetric(self) -> bool:
        """Check if matrix is symmetric within floating-point tolerance."""
        return bool(np.allclose(self.matrix, self.matrix.T, atol=1e-6))

    def symmetrise(self) -> None:
        """Force symmetry: M = (M + M^T) / 2."""
        self.matrix = (self.matrix + self.matrix.T) / 2.0

    def __repr__(self) -> str:
        sym = "symmetric" if self.is_symmetric else "asymmetric"
        return f"<EnergyMatrix ntypes={self.ntypes} {sym}>"


# ── convenience functions ────────────────────────────────────────────────────

def parse_dat_file(path: str) -> Tuple[int, "np.ndarray"]:
    """Parse legacy ``.dat`` file and return ``(ntypes, matrix)``."""
    mat = EnergyMatrix.from_dat_file(path)
    return mat.ntypes, mat.matrix


def write_dat_file(path: str, ntypes: int, matrix: "np.ndarray",
                   labels: Optional[List[str]] = None) -> None:
    """Write a NumPy matrix as a legacy ``.dat`` file."""
    mat = EnergyMatrix(ntypes, matrix)
    mat.to_dat_file(path, labels=labels)


# ── training infrastructure ─────────────────────────────────────────────────

import json


@dataclass
class ContactTable:
    """Pre-computed contact frequency table from a set of structures.

    Attributes:
        ntypes:          Number of atom types.
        counts:          2D array (ntypes, ntypes) of observed contact counts.
        type_totals:     1D array (ntypes,) of total atoms per type.
        n_structures:    Number of structures processed.
        distance_cutoff: Contact distance cutoff in Angstroms.
    """
    ntypes: int
    counts: "np.ndarray"
    type_totals: "np.ndarray"
    n_structures: int = 0
    distance_cutoff: float = 6.0

    def save(self, path: Union[str, Path]) -> None:
        """Save contact table as JSON for reproducibility."""
        data = {
            "ntypes": self.ntypes,
            "counts": self.counts.tolist(),
            "type_totals": self.type_totals.tolist(),
            "n_structures": self.n_structures,
            "distance_cutoff": self.distance_cutoff,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ContactTable":
        """Load a previously saved contact table."""
        data = json.loads(Path(path).read_text())
        required_keys = {"ntypes", "counts", "type_totals", "n_structures", "distance_cutoff"}
        missing = required_keys - data.keys()
        if missing:
            raise ValueError(
                f"ContactTable JSON missing required keys: {sorted(missing)}"
            )
        try:
            return cls(
                ntypes=data["ntypes"],
                counts=np.array(data["counts"], dtype=np.int64),
                type_totals=np.array(data["type_totals"], dtype=np.int64),
                n_structures=data["n_structures"],
                distance_cutoff=data["distance_cutoff"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid ContactTable JSON data: {exc}") from exc


class KnowledgeBasedTrainer:
    """Derive Sippl-style statistical potentials from structural data.

    The inverse Boltzmann approach converts observed contact frequencies
    into pseudo-energies::

        E(i,j) = -kT * ln( f_obs(i,j) / f_ref(i,j) )

    where f_ref uses a volume-corrected random mixing reference state.

    Args:
        ntypes:          Number of atom types.
        temperature:     Reference temperature for kT (default 300 K).
        distance_cutoff: Contact distance threshold in Angstroms (default 6.0).
        pseudocount:     Smoothing count added to each bin to avoid log(0) (default 1).
    """

    def __init__(
        self,
        ntypes: int,
        temperature: float = 300.0,
        distance_cutoff: float = 6.0,
        pseudocount: int = 1,
    ) -> None:
        if np is None:
            raise RuntimeError("NumPy is required for KnowledgeBasedTrainer")
        if ntypes <= 0:
            raise ValueError(f"ntypes must be positive, got {ntypes}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if pseudocount < 0:
            raise ValueError(f"pseudocount must be non-negative, got {pseudocount}")
        if distance_cutoff <= 0:
            raise ValueError(f"distance_cutoff must be positive, got {distance_cutoff}")
        self._ntypes = ntypes
        self._temperature = temperature
        self._kT = 0.001987206 * temperature  # kcal/mol
        self._cutoff = distance_cutoff
        self._pseudocount = pseudocount
        self._counts = np.zeros((ntypes, ntypes), dtype=np.int64)
        self._type_totals = np.zeros(ntypes, dtype=np.int64)
        self._n_structures = 0

    def add_structure(
        self,
        coords: "np.ndarray",
        atom_types: "np.ndarray",
        *,
        chain_mask: Optional["np.ndarray"] = None,
    ) -> None:
        """Count contacts from a single structure.

        Args:
            coords:     (N, 3) atom coordinates.
            atom_types: (N,) integer atom types, 0-based.
            chain_mask: (N,) chain IDs (int or str).  If provided, only
                        inter-chain contacts are counted.
        """
        try:
            from scipy.spatial.distance import cdist
        except ImportError:
            raise ImportError(
                "scipy is required for add_structure(). "
                "Install it with: pip install scipy"
            ) from None

        coords = np.asarray(coords, dtype=np.float64)
        atom_types = np.asarray(atom_types, dtype=np.int64)
        n = len(coords)

        # Accumulate type totals
        for t in range(self._ntypes):
            self._type_totals[t] += int(np.sum(atom_types == t))

        # Compute all pairwise distances
        dists = cdist(coords, coords)

        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] > self._cutoff:
                    continue
                if chain_mask is not None and chain_mask[i] == chain_mask[j]:
                    continue
                ti = int(atom_types[i])
                tj = int(atom_types[j])
                if 0 <= ti < self._ntypes and 0 <= tj < self._ntypes:
                    self._counts[ti, tj] += 1
                    self._counts[tj, ti] += 1

        self._n_structures += 1

    def add_contact_table(self, table: ContactTable) -> None:
        """Merge a pre-computed ContactTable into the running counts."""
        if table.ntypes != self._ntypes:
            raise ValueError(
                f"ContactTable ntypes ({table.ntypes}) does not match "
                f"trainer ntypes ({self._ntypes})"
            )
        self._counts += table.counts
        self._type_totals += table.type_totals
        self._n_structures += table.n_structures

    def get_contact_table(self) -> ContactTable:
        """Return the current accumulated contact table."""
        return ContactTable(
            ntypes=self._ntypes,
            counts=self._counts.copy(),
            type_totals=self._type_totals.copy(),
            n_structures=self._n_structures,
            distance_cutoff=self._cutoff,
        )

    def derive_potential(self) -> EnergyMatrix:
        """Compute the statistical potential from accumulated counts.

        Reference state (volume-corrected random mixing)::

            f_ref(i,j) = N_i * N_j / N_total²           (i ≠ j)
            f_ref(i,i) = N_i * (N_i - 1) / (2 * N_total²)

        Returns:
            EnergyMatrix with scalar values.
        """
        total_contacts = np.sum(self._counts) / 2.0  # upper triangle
        n_total = float(np.sum(self._type_totals))
        if n_total == 0 or total_contacts == 0:
            return EnergyMatrix(self._ntypes)

        matrix = np.zeros((self._ntypes, self._ntypes), dtype=np.float64)

        for i in range(self._ntypes):
            for j in range(i, self._ntypes):
                # Observed frequency
                obs = self._counts[i, j] + self._pseudocount
                f_obs = obs / (total_contacts + self._pseudocount * self._ntypes * (self._ntypes + 1) / 2)

                # Reference frequency
                ni = float(self._type_totals[i])
                nj = float(self._type_totals[j])
                if i == j:
                    f_ref = ni * (ni - 1) / (2.0 * n_total * n_total) if ni > 1 else 1e-12
                else:
                    f_ref = ni * nj / (n_total * n_total) if ni > 0 and nj > 0 else 1e-12

                f_ref = max(f_ref, 1e-12)  # prevent log(0)

                energy = -self._kT * math.log(f_obs / f_ref)
                matrix[i, j] = energy
                matrix[j, i] = energy

        _validate_matrix(matrix, "derive_potential output")
        return EnergyMatrix(self._ntypes, matrix)

    @staticmethod
    def parse_pdb_contacts(
        pdb_path: Union[str, Path],
        type_mapping: Dict[str, int],
        distance_cutoff: float = 6.0,
    ) -> Tuple["np.ndarray", "np.ndarray", Optional["np.ndarray"]]:
        """Parse a PDB file and return (coords, atom_types, chain_mask).

        Args:
            pdb_path:     Path to the PDB file.
            type_mapping: Maps ``"RESNAME:ATOMNAME"`` → type index (0-based).
            distance_cutoff: Not used directly; passed through for reference.

        Returns:
            Tuple of (coordinates, atom_types, chain_ids) numpy arrays.
        """
        coords_list: List[List[float]] = []
        types_list: List[int] = []
        chains_list: List[int] = []

        chain_ids: Dict[str, int] = {}

        with open(pdb_path) as fh:
            for line in fh:
                record = line[:6].strip()
                if record not in ("ATOM", "HETATM"):
                    continue

                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21:22].strip() or "A"
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError) as exc:
                    raise ValueError(
                        f"Malformed PDB coordinates at line: {line.rstrip()!r}"
                    ) from exc

                key = f"{res_name}:{atom_name}"
                if key not in type_mapping:
                    continue

                coords_list.append([x, y, z])
                types_list.append(type_mapping[key])

                if chain_id not in chain_ids:
                    chain_ids[chain_id] = len(chain_ids)
                chains_list.append(chain_ids[chain_id])

        coords = np.array(coords_list, dtype=np.float64)
        atom_types = np.array(types_list, dtype=np.int64)
        chain_mask = np.array(chains_list, dtype=np.int64) if len(chains_list) > 0 else None

        return coords, atom_types, chain_mask


# ── gradient-free optimization ──────────────────────────────────────────────

import concurrent.futures
import subprocess
import tempfile


@dataclass
class DockingBenchmarkCase:
    """A single receptor-ligand pair for enrichment evaluation.

    Attributes:
        receptor:    Path to receptor PDB.
        ligand:      Path to ligand MOL2/SDF.
        is_active:   True if this is a known active, False if decoy.
        native_rmsd: RMSD threshold for "near-native" (optional).
        config:      Optional path to pre-built FlexAID config.
    """
    receptor: Union[str, Path] = ""
    ligand: Union[str, Path] = ""
    is_active: bool = True
    native_rmsd: Optional[float] = None
    config: Optional[Union[str, Path]] = None


@dataclass
class OptimizationResult:
    """Result of energy matrix optimization.

    Attributes:
        best_matrix:        The optimized EnergyMatrix.
        best_score:         Objective function value (higher is better for AUC).
        history:            List of (iteration, score) tuples.
        n_evaluations:      Total number of docking evaluations performed.
        convergence_reason: Why optimization stopped.
    """
    best_matrix: EnergyMatrix
    best_score: float
    history: List[Tuple[int, float]] = field(default_factory=list)
    n_evaluations: int = 0
    convergence_reason: str = ""

    def to_dict(self) -> Dict:
        """Serialize for JSON output (excluding the matrix itself)."""
        return {
            "best_score": self.best_score,
            "n_evaluations": self.n_evaluations,
            "convergence_reason": self.convergence_reason,
            "history": self.history,
        }


class EnergyMatrixOptimizer:
    """Optimize energy matrix parameters to maximize docking performance.

    Supports two objective functions:

    - **auc**: Maximize area under ROC curve for active/decoy discrimination.
    - **rmsd**: Minimize fraction of cases where top-ranked pose RMSD > threshold.

    Supports two optimizers:

    - **de**: ``scipy.optimize.differential_evolution`` (default).
    - **cma**: CMA-ES via the ``cma`` package (optional, BSD-3).

    Args:
        reference_matrix: Starting EnergyMatrix (defines ntypes and structure).
        benchmark:        List of DockingBenchmarkCase objects.
        objective:        ``"auc"`` or ``"rmsd"`` (default ``"auc"``).
        optimizer:        ``"de"`` or ``"cma"`` (default ``"de"``).
        n_workers:        Number of parallel docking processes (default 1).
        flexaid_binary:   Path to FlexAID executable (auto-detected if None).
        work_dir:         Directory for temporary files (default tempdir).
        temperature:      Docking temperature in Kelvin (default 300).
    """

    def __init__(
        self,
        reference_matrix: EnergyMatrix,
        benchmark: List[DockingBenchmarkCase],
        objective: str = "auc",
        optimizer: str = "de",
        n_workers: int = 1,
        flexaid_binary: Optional[str] = None,
        work_dir: Optional[Union[str, Path]] = None,
        temperature: float = 300.0,
    ) -> None:
        # Verify the matrix is all-scalar (density functions can't be parameterised)
        if reference_matrix.entries:
            for entry in reference_matrix.entries.values():
                if not entry.is_scalar:
                    raise ValueError(
                        "Optimization requires an all-scalar energy matrix. "
                        "Density-function entries cannot be optimized."
                    )

        if not benchmark:
            raise ValueError("benchmark must be a non-empty list of DockingBenchmarkCase")

        self._reference = reference_matrix
        self._benchmark = benchmark
        self._objective = objective
        self._optimizer_name = optimizer
        self._n_workers = n_workers
        self._flexaid_binary = flexaid_binary or self._find_flexaid()
        self._work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="emat_opt_"))
        self._temperature = temperature
        self._eval_count = 0

    @staticmethod
    def _find_flexaid() -> str:
        """Try to locate FlexAID binary."""
        import shutil
        for name in ("FlexAID", "flexaid", "flexaids"):
            path = shutil.which(name)
            if path:
                return path
        # Check relative to this file's location
        project_bin = Path(__file__).resolve().parents[2] / "BIN" / "FlexAID"
        if project_bin.is_file():
            return str(project_bin)
        raise FileNotFoundError(
            "FlexAID binary not found. Provide the path explicitly via "
            "flexaid_binary parameter, or ensure FlexAID is on PATH."
        )

    def _matrix_to_vector(self, matrix: EnergyMatrix) -> "np.ndarray":
        """Flatten upper-triangle scalar values to a 1D parameter vector.

        Order: (0,0), (0,1), ..., (0,N-1), (1,1), (1,2), ..., (N-1,N-1).
        """
        n = matrix.ntypes
        vec = []
        for i in range(n):
            for j in range(i, n):
                vec.append(float(matrix.matrix[i, j]))
        return np.array(vec, dtype=np.float64)

    def _vector_to_matrix(self, vector: "np.ndarray") -> EnergyMatrix:
        """Reconstruct an EnergyMatrix from a flattened parameter vector."""
        n = self._reference.ntypes
        mat = np.zeros((n, n), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(i, n):
                mat[i, j] = vector[idx]
                mat[j, i] = vector[idx]
                idx += 1
        return EnergyMatrix(n, mat)

    def _evaluate_single(
        self, matrix: EnergyMatrix, case: DockingBenchmarkCase
    ) -> float:
        """Run docking for one case and return the score.

        Writes the matrix to a temp .dat file, runs FlexAID, and parses
        the best CF score from output.
        """
        import uuid
        import shutil as _shutil
        case_dir = self._work_dir / f"eval_{uuid.uuid4().hex[:12]}"
        case_dir.mkdir(parents=True, exist_ok=True)

        try:
            mat_path = case_dir / "emat.dat"
            matrix.to_dat_file(str(mat_path))

            # Build minimal config
            config_lines = [
                f"PDBNAM {case.receptor}",
                f"INPLIG {case.ligand}",
                f"IMATRX {mat_path}",
                f"TEMPER {int(self._temperature)}",
                "METOPT GA",
                "COMPLF VCT",
                "NRGOUT 1",
            ]
            config_path = case_dir / "dock.inp"
            config_path.write_text("\n".join(config_lines) + "\n")

            try:
                result = subprocess.run(
                    [self._flexaid_binary, str(config_path)],
                    capture_output=True, text=True, timeout=300,
                    cwd=str(case_dir),
                )
                # Parse best CF score from output
                best_score = float("inf")
                for line in result.stdout.splitlines():
                    if "CF=" in line or "cf=" in line.lower():
                        try:
                            parts = line.split()
                            for part in parts:
                                if part.replace("-", "").replace(".", "").isdigit():
                                    val = float(part)
                                    if val < best_score:
                                        best_score = val
                        except ValueError:
                            continue
                return best_score
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                return float("inf")
        finally:
            _shutil.rmtree(case_dir, ignore_errors=True)

    def _evaluate_objective(self, vector: "np.ndarray") -> float:
        """Full objective function evaluation.

        Returns negative score for minimization (scipy convention).
        """
        matrix = self._vector_to_matrix(vector)
        self._eval_count += 1

        if self._n_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self._n_workers
            ) as pool:
                futures = {
                    pool.submit(self._evaluate_single, matrix, case): case
                    for case in self._benchmark
                }
                scores = {}
                for future in concurrent.futures.as_completed(futures):
                    case = futures[future]
                    scores[id(case)] = future.result()
        else:
            scores = {
                id(case): self._evaluate_single(matrix, case)
                for case in self._benchmark
            }

        active_scores = [
            scores[id(c)] for c in self._benchmark if c.is_active
        ]
        decoy_scores = [
            scores[id(c)] for c in self._benchmark if not c.is_active
        ]

        if self._objective == "auc":
            auc = self._compute_auc(active_scores, decoy_scores)
            return -auc  # negative for minimization
        elif self._objective == "rmsd":
            # Fraction of actives with RMSD below threshold
            # (lower CF scores should correlate with native-like poses)
            n_good = sum(1 for s in active_scores if s < 0)
            frac = n_good / max(len(active_scores), 1)
            return -frac
        else:
            raise ValueError(f"Unknown objective: {self._objective}")

    @staticmethod
    def _compute_auc(
        active_scores: List[float], decoy_scores: List[float]
    ) -> float:
        """Compute ROC AUC from active and decoy CF scores.

        Uses the Wilcoxon-Mann-Whitney U-statistic (no sklearn dependency).
        Lower CF scores are better, so actives should score lower than decoys.
        """
        n_a = len(active_scores)
        n_d = len(decoy_scores)
        if n_a == 0 or n_d == 0:
            return 0.5

        # Sort-merge O(n log n) approach.
        # Lower CF = better.  Active should score lower than decoy.
        # AUC = P(active_score < decoy_score).
        # Sort ascending.  For each decoy, count how many actives precede it.
        combined = [(s, 0) for s in active_scores] + [(s, 1) for s in decoy_scores]
        combined.sort(key=lambda x: x[0])

        u = 0.0
        active_count = 0
        for score, label in combined:
            if label == 0:  # active
                active_count += 1
            else:  # decoy
                u += active_count  # actives with lower (better) scores

        return u / (n_a * n_d)

    def optimize(
        self,
        max_evaluations: int = 5000,
        sigma0: float = 0.5,
        bounds: Optional[Tuple[float, float]] = (-10.0, 10.0),
        seed: Optional[int] = None,
        callback: Optional[object] = None,
    ) -> OptimizationResult:
        """Run the optimization loop.

        Args:
            max_evaluations: Maximum number of objective evaluations.
            sigma0:          Initial step size for CMA-ES (ignored for DE).
            bounds:          (lower, upper) bounds for each parameter.
            seed:            Random seed for reproducibility.
            callback:        Optional callable(iteration, best_score, best_vector).

        Returns:
            OptimizationResult with the best matrix found.
        """
        x0 = self._matrix_to_vector(self._reference)
        n_params = len(x0)
        history: List[Tuple[int, float]] = []

        if self._optimizer_name == "cma":
            try:
                import cma
            except ImportError:
                raise ImportError(
                    "CMA-ES requires the 'cma' package. "
                    "Install it with: pip install cma"
                )

            opts = {
                "maxfevals": max_evaluations,
                "seed": seed if seed is not None else 0,
                "verbose": -1,
            }
            if bounds is not None:
                opts["bounds"] = [bounds[0], bounds[1]]

            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            iteration = 0
            while not es.stop():
                solutions = es.ask()
                values = [self._evaluate_objective(s) for s in solutions]
                es.tell(solutions, values)
                best_val = es.result.fbest
                history.append((iteration, -best_val))
                if callback is not None:
                    callback(iteration, -best_val, es.result.xbest)
                iteration += 1

            best_vec = es.result.xbest
            best_score = -es.result.fbest
            reason = "CMA-ES converged"

        else:  # differential_evolution
            from scipy.optimize import differential_evolution

            param_bounds = [(bounds[0], bounds[1])] * n_params if bounds else [(-10, 10)] * n_params

            iteration_counter = [0]

            def de_callback(xk, convergence=0):
                val = self._evaluate_objective(xk)
                history.append((iteration_counter[0], -val))
                if callback is not None:
                    callback(iteration_counter[0], -val, xk)
                iteration_counter[0] += 1

            result = differential_evolution(
                self._evaluate_objective,
                bounds=param_bounds,
                x0=x0,
                maxiter=max_evaluations // 15,  # DE uses ~15 evals per iteration
                seed=seed,
                callback=de_callback,
                tol=1e-6,
                workers=1,  # we handle parallelism ourselves
            )

            best_vec = result.x
            best_score = -result.fun
            reason = f"DE: {result.message}"

        best_matrix = self._vector_to_matrix(best_vec)
        return OptimizationResult(
            best_matrix=best_matrix,
            best_score=best_score,
            history=history,
            n_evaluations=self._eval_count,
            convergence_reason=reason,
        )

    @staticmethod
    def load_benchmark(
        benchmark_dir: Union[str, Path],
        actives_subdir: str = "actives",
        decoys_subdir: str = "decoys",
    ) -> List[DockingBenchmarkCase]:
        """Load a benchmark dataset from a directory structure.

        Expected layout::

            benchmark_dir/
                receptor.pdb
                actives/
                    lig1.mol2
                    lig2.mol2
                decoys/
                    decoy1.mol2
                    decoy2.mol2

        Returns:
            List of DockingBenchmarkCase objects.
        """
        bdir = Path(benchmark_dir)

        # Find receptor
        receptor = None
        for ext in ("*.pdb", "*.PDB"):
            candidates = list(bdir.glob(ext))
            if candidates:
                receptor = candidates[0]
                break
        if receptor is None:
            raise FileNotFoundError(f"No receptor PDB found in {bdir}")

        cases: List[DockingBenchmarkCase] = []

        # Actives
        actives_dir = bdir / actives_subdir
        if actives_dir.is_dir():
            for lig in sorted(actives_dir.iterdir()):
                if lig.suffix.lower() in (".mol2", ".sdf"):
                    cases.append(DockingBenchmarkCase(
                        receptor=str(receptor),
                        ligand=str(lig),
                        is_active=True,
                    ))

        # Decoys
        decoys_dir = bdir / decoys_subdir
        if decoys_dir.is_dir():
            for lig in sorted(decoys_dir.iterdir()):
                if lig.suffix.lower() in (".mol2", ".sdf"):
                    cases.append(DockingBenchmarkCase(
                        receptor=str(receptor),
                        ligand=str(lig),
                        is_active=False,
                    ))

        return cases
