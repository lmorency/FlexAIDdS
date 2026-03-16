"""flexaidds: Python bindings and analysis tools for FlexAID∆S."""

from .models import BindingModeResult, DockingResult, PoseResult
from .results import load_results
from .docking import Docking, BindingMode, BindingPopulation, Pose
from .encom import ENCoMEngine, NormalMode, VibrationalEntropy
from .tencm import (
    TorsionalENM, TorsionalNormalMode, Conformer, FullThermoResult,
    compute_shannon_entropy, compute_torsional_vibrational_entropy,
    run_shannon_thermo_stack,
)
from .__version__ import __version__ as __version__

# Pure-Python thermodynamics (always available)
from .thermodynamics import StatMechEngine, Thermodynamics, kB_kcal, kB_SI

# C++ extension — optional: pure-Python helpers work without it
try:
    from ._core import (
        BoltzmannLUT,
        Replica,
        State,
        TIPoint,
        WHAMBin,
        kB_kcal,  # noqa: F811  (more precise C++ value)
        kB_SI,    # noqa: F811
    )
    # Override with C++ StatMechEngine when available
    from ._core import StatMechEngine as _CppStatMechEngine  # noqa: F811
    from ._core import Thermodynamics as _CppThermodynamics  # noqa: F811
    from ._core import ENCoMEngine as _CppENCoMEngine        # noqa: F811
    from ._core import NormalMode as _CppNormalMode           # noqa: F811
    from ._core import VibrationalEntropy as _CppVibrationalEntropy  # noqa: F811
    HAS_CORE_BINDINGS = True
except ImportError:
    # Fallback when C++ extension is not built
    from ._fallback_types import BoltzmannLUT, Replica, State, TIPoint, WHAMBin
    kB_kcal = 0.001987206   # kcal mol⁻¹ K⁻¹
    kB_SI = 1.380649e-23    # J K⁻¹
    HAS_CORE_BINDINGS = False

from .supercluster import SuperCluster
from .tencom_results import FlexModeResult, FlexPopulationResult, parse_tencom_pdb, parse_tencom_json


def dock(
    receptor: str,
    ligand: str,
    *,
    binding_site: str = "auto",
    compute_entropy: bool = True,
    temperature: float = 300.0,
    config: str = "",
    binary: str = None,
    timeout: int = 3600,
) -> BindingPopulation:
    """High-level docking interface.

    Generates a FlexAID config, runs the C++ engine, and returns results
    as a :class:`BindingPopulation`.

    Args:
        receptor:        Path to receptor PDB file.
        ligand:          Path to ligand MOL2 file.
        binding_site:    Binding site specification (default ``"auto"``).
        compute_entropy: Enable thermodynamic entropy calculation.
        temperature:     Simulation temperature in Kelvin.
        config:          Optional path to a FlexAID ``.inp`` config file.
                         When provided, ``receptor`` and ``ligand`` are ignored
                         and the config is used directly.
        binary:          Path to the FlexAID executable (auto-detected if None).
        timeout:         Wall-clock timeout in seconds.

    Returns:
        :class:`BindingPopulation` with ranked binding modes.

    Example:
        >>> results = flexaidds.dock('receptor.pdb', 'ligand.mol2')
        >>> for mode in results.rank_by_free_energy():
        ...     print(f"Mode: ΔG={mode.free_energy:.2f} kcal/mol")
    """
    import tempfile
    from pathlib import Path

    if config:
        docking = Docking(config)
        return docking.run(binary=binary, timeout=timeout)

    # Generate a temporary .inp config from receptor + ligand
    receptor_path = Path(receptor).resolve()
    ligand_path = Path(ligand).resolve()
    if not receptor_path.is_file():
        raise FileNotFoundError(f"Receptor not found: {receptor}")
    if not ligand_path.is_file():
        raise FileNotFoundError(f"Ligand not found: {ligand}")

    lines = [
        f"PDBNAM {receptor_path}",
        f"INPLIG {ligand_path}",
        f"TEMPER {int(temperature)}",
        "METOPT GA",
        "COMPLF VCT",
    ]
    if not compute_entropy:
        lines.append("TEMPER 0")
    if binding_site != "auto":
        lines.append(f"RNGOPT {binding_site}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="flexaidds_"))
    cfg_path = tmp_dir / "dock.inp"
    cfg_path.write_text("\n".join(lines) + "\n")

    docking = Docking(str(cfg_path))
    return docking.run(binary=binary, timeout=timeout)

__all__ = [
    # High-level API
    "dock",
    "Docking",
    "BindingMode",
    "BindingPopulation",
    "Pose",
    # Result I/O (always available)
    "PoseResult",
    "BindingModeResult",
    "DockingResult",
    "load_results",
    # Thermodynamics (always available — pure Python or C++)
    "StatMechEngine",
    "Thermodynamics",
    # ENCoM (always available — pure Python or C++)
    "ENCoMEngine",
    "NormalMode",
    "VibrationalEntropy",
    # tENCoM results (always available, pure Python)
    "FlexModeResult",
    "FlexPopulationResult",
    "parse_tencom_pdb",
    "parse_tencom_json",
    # TorsionalENM / ShannonThermoStack (always available — pure Python or C++)
    "TorsionalENM",
    "TorsionalNormalMode",
    "Conformer",
    "FullThermoResult",
    "compute_shannon_entropy",
    "compute_torsional_vibrational_entropy",
    "run_shannon_thermo_stack",
    # Super-cluster extraction
    "SuperCluster",
    # Physical constants
    "kB_kcal",
    "kB_SI",
    # Availability flag
    "HAS_CORE_BINDINGS",
    # Core types (C++ when available, pure-Python fallback otherwise)
    "StatMechEngine",
    "Thermodynamics",
    "State",
    "BoltzmannLUT",
    "Replica",
    "WHAMBin",
    "TIPoint",
    "ENCoMEngine",
    "NormalMode",
    "VibrationalEntropy",
]

# C++ core extras (only available when compiled)
if HAS_CORE_BINDINGS:
    __all__.extend([
        "State",
        "BoltzmannLUT",
        "Replica",
        "WHAMBin",
        "TIPoint",
    ])
