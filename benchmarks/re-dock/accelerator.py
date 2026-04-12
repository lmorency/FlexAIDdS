"""
RE-DOCK Hardware Acceleration Layer (v7)
==========================================

NumPy/SciPy optimized core with optional CuPy GPU and Numba JIT backends.

Auto-detects available acceleration and selects the fastest backend:

    CuPy (GPU) → Numba JIT (CPU SIMD) → NumPy/SciPy (CPU BLAS)

All public functions accept and return NumPy arrays regardless of backend.
GPU transfers happen transparently inside the acceleration layer.

Components
----------
- **AcceleratorBackend**: Auto-detecting backend selector
- **vectorized_metropolis_batch()**: Batch Metropolis acceptance (vectorized)
- **vectorized_work_accumulation()**: Batch work computation (vectorized)
- **vectorized_boltzmann_weights()**: Log-sum-exp stable Boltzmann weights
- **vectorized_shannon_entropy()**: Batch Shannon entropy over multiple ensembles
- **vectorized_bar_fermi()**: Vectorized Fermi function for BAR iterations
- **vectorized_kde_grid()**: Batch KDE evaluation for Crooks intersection
- **mmap_array()**: Memory-mapped array factory for large campaigns

Le Bonhomme Pharma / Najmanovich Research Group
"""

from __future__ import annotations

import math
import mmap
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Optional backend imports with graceful fallback
# ---------------------------------------------------------------------------

_CUPY_AVAILABLE = False
_NUMBA_AVAILABLE = False

try:
    import cupy as cp  # type: ignore[import-untyped]
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None

try:
    from numba import njit, prange  # type: ignore[import-untyped]
    _NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    prange = None


# ---------------------------------------------------------------------------
# Backend enumeration
# ---------------------------------------------------------------------------

class Backend(str, Enum):
    """Available acceleration backends."""
    CUPY = "cupy"
    NUMBA = "numba"
    NUMPY = "numpy"


# ---------------------------------------------------------------------------
# AcceleratorBackend — auto-detecting dispatch layer
# ---------------------------------------------------------------------------

@dataclass
class AcceleratorBackend:
    """Hardware acceleration backend with auto-detection.

    Attributes
    ----------
    active_backend : Backend
        Currently selected backend.
    device_name : str
        Human-readable device description.
    """
    active_backend: Backend
    device_name: str

    @classmethod
    def auto_detect(cls, prefer_gpu: bool = True) -> "AcceleratorBackend":
        """Auto-detect the fastest available backend.

        Parameters
        ----------
        prefer_gpu : bool
            If True, prefer CuPy GPU when available.

        Returns
        -------
        AcceleratorBackend
        """
        if prefer_gpu and _CUPY_AVAILABLE:
            try:
                device = cp.cuda.Device(0)
                name = f"CUDA GPU: {device.attributes.get('DeviceName', 'unknown')}"
                # Quick smoke test
                _ = cp.array([1.0, 2.0])
                return cls(active_backend=Backend.CUPY, device_name=name)
            except Exception:
                pass

        if _NUMBA_AVAILABLE:
            return cls(
                active_backend=Backend.NUMBA,
                device_name="Numba JIT (CPU SIMD)",
            )

        return cls(
            active_backend=Backend.NUMPY,
            device_name="NumPy/SciPy (CPU BLAS)",
        )

    def __repr__(self) -> str:
        return f"AcceleratorBackend({self.active_backend.value}: {self.device_name})"


# ---------------------------------------------------------------------------
# Numba-accelerated kernels (compiled lazily on first call)
# ---------------------------------------------------------------------------

if _NUMBA_AVAILABLE and njit is not None:

    @njit(cache=True, fastmath=True)  # type: ignore[misc]
    def _numba_metropolis_batch(
        beta_i: np.ndarray,
        beta_j: np.ndarray,
        energy_i: np.ndarray,
        energy_j: np.ndarray,
        rand_vals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Numba JIT: batch Metropolis acceptance with SIMD vectorization."""
        n = len(beta_i)
        accepted = np.empty(n, dtype=np.bool_)
        deltas = np.empty(n, dtype=np.float64)
        for k in range(n):
            delta = (beta_i[k] - beta_j[k]) * (energy_i[k] - energy_j[k])
            deltas[k] = delta
            if delta <= 0.0:
                accepted[k] = True
            else:
                accepted[k] = rand_vals[k] < math.exp(-delta)
        return accepted, deltas

    @njit(cache=True, fastmath=True)  # type: ignore[misc]
    def _numba_work_accumulation(
        beta_i: np.ndarray,
        beta_j: np.ndarray,
        energy_i: np.ndarray,
        energy_j: np.ndarray,
    ) -> np.ndarray:
        """Numba JIT: batch non-equilibrium work computation."""
        n = len(beta_i)
        work = np.empty(n, dtype=np.float64)
        for k in range(n):
            work[k] = -((beta_j[k] - beta_i[k]) * (energy_j[k] - energy_i[k]))
        return work

    @njit(cache=True, fastmath=True)  # type: ignore[misc]
    def _numba_bar_fermi(
        arg: np.ndarray,
    ) -> np.ndarray:
        """Numba JIT: log-Fermi function log(1/(1+exp(x))) = -log1p(exp(x))."""
        n = len(arg)
        result = np.empty(n, dtype=np.float64)
        for k in range(n):
            x = arg[k]
            if x > 30.0:
                result[k] = -x
            elif x < -30.0:
                result[k] = 0.0
            else:
                result[k] = -math.log(1.0 + math.exp(x))
        return result

else:
    _numba_metropolis_batch = None
    _numba_work_accumulation = None
    _numba_bar_fermi = None


# ---------------------------------------------------------------------------
# Vectorized Metropolis batch
# ---------------------------------------------------------------------------

def vectorized_metropolis_batch(
    beta_i: NDArray[np.float64],
    beta_j: NDArray[np.float64],
    energy_i: NDArray[np.float64],
    energy_j: NDArray[np.float64],
    rng: np.random.Generator | None = None,
    backend: Backend = Backend.NUMPY,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    r"""Batch Metropolis acceptance criterion for N exchange pairs.

    .. math::

        \Delta_k = (\beta_i^k - \beta_j^k)(E_i^k - E_j^k)

    Accept if :math:`\Delta_k \leq 0` or :math:`u_k < \exp(-\Delta_k)`.

    Parameters
    ----------
    beta_i, beta_j : NDArray[np.float64]
        Inverse temperatures of pairs (shape (N,)).
    energy_i, energy_j : NDArray[np.float64]
        Current energies of pairs (shape (N,)).
    rng : np.random.Generator, optional
        Random number generator.
    backend : Backend
        Acceleration backend to use.

    Returns
    -------
    accepted : NDArray[np.bool_]
        Acceptance mask (shape (N,)).
    deltas : NDArray[np.float64]
        Delta values (shape (N,)).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(beta_i)
    rand_vals = rng.random(n)

    if backend == Backend.NUMBA and _numba_metropolis_batch is not None:
        return _numba_metropolis_batch(beta_i, beta_j, energy_i, energy_j, rand_vals)

    if backend == Backend.CUPY and _CUPY_AVAILABLE:
        bi = cp.asarray(beta_i)
        bj = cp.asarray(beta_j)
        ei = cp.asarray(energy_i)
        ej = cp.asarray(energy_j)
        rv = cp.asarray(rand_vals)

        deltas_gpu = (bi - bj) * (ei - ej)
        probs = cp.exp(cp.minimum(0.0, -deltas_gpu))
        accepted_gpu = (deltas_gpu <= 0) | (rv < probs)
        return cp.asnumpy(accepted_gpu), cp.asnumpy(deltas_gpu)

    # NumPy fallback
    deltas = (beta_i - beta_j) * (energy_i - energy_j)
    probs = np.exp(np.minimum(0.0, -deltas))
    accepted = (deltas <= 0) | (rand_vals < probs)
    return accepted, deltas


# ---------------------------------------------------------------------------
# Vectorized work accumulation
# ---------------------------------------------------------------------------

def vectorized_work_accumulation(
    beta_i: NDArray[np.float64],
    beta_j: NDArray[np.float64],
    energy_i: NDArray[np.float64],
    energy_j: NDArray[np.float64],
    backend: Backend = Backend.NUMPY,
) -> NDArray[np.float64]:
    r"""Batch non-equilibrium work computation.

    .. math::

        W_k = -(\beta_j^k - \beta_i^k)(E_j^k - E_i^k)

    Parameters
    ----------
    beta_i, beta_j : NDArray[np.float64]
        Inverse temperatures (shape (N,)).
    energy_i, energy_j : NDArray[np.float64]
        Energies (shape (N,)).
    backend : Backend
        Acceleration backend.

    Returns
    -------
    NDArray[np.float64]
        Work values (shape (N,)) in kcal/mol.
    """
    if backend == Backend.NUMBA and _numba_work_accumulation is not None:
        return _numba_work_accumulation(beta_i, beta_j, energy_i, energy_j)

    if backend == Backend.CUPY and _CUPY_AVAILABLE:
        bi = cp.asarray(beta_i)
        bj = cp.asarray(beta_j)
        ei = cp.asarray(energy_i)
        ej = cp.asarray(energy_j)
        work_gpu = -((bj - bi) * (ej - ei))
        return cp.asnumpy(work_gpu)

    # NumPy
    return -((beta_j - beta_i) * (energy_j - energy_i))


# ---------------------------------------------------------------------------
# Vectorized Boltzmann weights (log-sum-exp stable)
# ---------------------------------------------------------------------------

def vectorized_boltzmann_weights(
    energies: NDArray[np.float64],
    beta: float,
    backend: Backend = Backend.NUMPY,
) -> tuple[NDArray[np.float64], float]:
    r"""Compute Boltzmann weights and log-partition function.

    .. math::

        \log p_i = -\beta E_i - \log Z, \quad
        \log Z = \mathrm{LSE}(-\beta E)

    Parameters
    ----------
    energies : NDArray[np.float64]
        Energies (shape (N,)) in kcal/mol.
    beta : float
        Inverse temperature 1/(R*T) in (kcal/mol)^-1.
    backend : Backend
        Acceleration backend.

    Returns
    -------
    log_weights : NDArray[np.float64]
        Log Boltzmann weights log(p_i), normalized (shape (N,)).
    log_Z : float
        Log partition function.
    """
    if backend == Backend.CUPY and _CUPY_AVAILABLE:
        E_gpu = cp.asarray(energies)
        neg_bE = -beta * E_gpu
        max_val = cp.max(neg_bE)
        log_Z = float(max_val + cp.log(cp.sum(cp.exp(neg_bE - max_val))))
        log_w = cp.asnumpy(neg_bE) - log_Z
        return log_w, log_Z

    # NumPy (also used by Numba backend since this is already fast in NumPy)
    neg_bE = -beta * energies
    max_val = np.max(neg_bE)
    log_Z = float(max_val + np.log(np.sum(np.exp(neg_bE - max_val))))
    log_w = neg_bE - log_Z
    return log_w, log_Z


# ---------------------------------------------------------------------------
# Vectorized Shannon entropy over multiple ensembles
# ---------------------------------------------------------------------------

def vectorized_shannon_entropy(
    energy_sets: list[NDArray[np.float64]],
    temperatures: NDArray[np.float64],
    backend: Backend = Backend.NUMPY,
) -> NDArray[np.float64]:
    r"""Batch Shannon configurational entropy for multiple temperature replicas.

    .. math::

        S_k = -R \sum_i p_i \log p_i \quad \text{at temperature } T_k

    Parameters
    ----------
    energy_sets : list[NDArray[np.float64]]
        List of energy arrays, one per replica.
    temperatures : NDArray[np.float64]
        Temperature of each replica (K).
    backend : Backend
        Acceleration backend.

    Returns
    -------
    NDArray[np.float64]
        Shannon entropies (shape (n_replicas,)) in kcal/(mol·K).
    """
    R = 1.987204e-3  # kcal/(mol·K)
    n = len(energy_sets)
    entropies = np.zeros(n, dtype=np.float64)

    for k in range(n):
        E = energy_sets[k]
        if len(E) == 0:
            continue
        beta = 1.0 / (R * temperatures[k])

        if backend == Backend.CUPY and _CUPY_AVAILABLE:
            E_gpu = cp.asarray(E)
            neg_bE = -beta * E_gpu
            max_val = cp.max(neg_bE)
            log_Z = max_val + cp.log(cp.sum(cp.exp(neg_bE - max_val)))
            log_p = neg_bE - log_Z
            p = cp.exp(log_p)
            S = float(-R * cp.sum(p * log_p))
        else:
            neg_bE = -beta * E
            max_val = np.max(neg_bE)
            log_Z = max_val + np.log(np.sum(np.exp(neg_bE - max_val)))
            log_p = neg_bE - log_Z
            p = np.exp(log_p)
            S = float(-R * np.sum(p * log_p))

        entropies[k] = S

    return entropies


# ---------------------------------------------------------------------------
# Vectorized BAR Fermi function
# ---------------------------------------------------------------------------

def vectorized_bar_fermi(
    arg: NDArray[np.float64],
    backend: Backend = Backend.NUMPY,
) -> NDArray[np.float64]:
    r"""Numerically stable log-Fermi function for BAR iterations.

    .. math::

        \log f(x) = \log\frac{1}{1 + e^x} = -\log(1 + e^x)

    Uses softplus decomposition for numerical stability.

    Parameters
    ----------
    arg : NDArray[np.float64]
        Input array.
    backend : Backend
        Acceleration backend.

    Returns
    -------
    NDArray[np.float64]
        log-Fermi values.
    """
    if backend == Backend.NUMBA and _numba_bar_fermi is not None:
        return _numba_bar_fermi(arg)

    if backend == Backend.CUPY and _CUPY_AVAILABLE:
        x = cp.asarray(arg)
        result = -cp.logaddexp(cp.zeros_like(x), x)
        return cp.asnumpy(result)

    # NumPy: -log(1 + exp(x)) = -logaddexp(0, x)
    return -np.logaddexp(0.0, arg)


# ---------------------------------------------------------------------------
# Vectorized KDE for Crooks intersection
# ---------------------------------------------------------------------------

def vectorized_kde_grid(
    data: NDArray[np.float64],
    grid: NDArray[np.float64],
    bandwidth: float,
    backend: Backend = Backend.NUMPY,
) -> NDArray[np.float64]:
    r"""Gaussian KDE evaluated on a grid, vectorized.

    .. math::

        \hat{f}(x) = \frac{1}{n h} \sum_{i=1}^n
        \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(x - x_i)^2}{2 h^2}\right)

    Parameters
    ----------
    data : NDArray[np.float64]
        Data points (shape (n,)).
    grid : NDArray[np.float64]
        Evaluation grid (shape (m,)).
    bandwidth : float
        KDE bandwidth h.
    backend : Backend
        Acceleration backend.

    Returns
    -------
    NDArray[np.float64]
        Density estimates (shape (m,)).
    """
    if len(data) == 0 or bandwidth <= 0:
        return np.zeros_like(grid)

    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

    if backend == Backend.CUPY and _CUPY_AVAILABLE:
        d_gpu = cp.asarray(data)
        g_gpu = cp.asarray(grid)
        z = (g_gpu[:, cp.newaxis] - d_gpu[cp.newaxis, :]) / bandwidth
        kernel = cp.exp(-0.5 * z * z) * (inv_sqrt_2pi / bandwidth)
        return cp.asnumpy(cp.mean(kernel, axis=1))

    # NumPy: outer operation
    z = (grid[:, np.newaxis] - data[np.newaxis, :]) / bandwidth
    kernel = np.exp(-0.5 * z * z) * (inv_sqrt_2pi / bandwidth)
    return np.mean(kernel, axis=1)


# ---------------------------------------------------------------------------
# Memory-mapped arrays for large campaigns
# ---------------------------------------------------------------------------

def mmap_array(
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type = np.float64,
    path: Path | str | None = None,
) -> np.ndarray:
    """Create a memory-mapped NumPy array for large campaign data.

    For campaigns with millions of work samples, memory-mapping avoids
    loading everything into RAM at once.

    Parameters
    ----------
    shape : tuple[int, ...]
        Array shape.
    dtype : np.dtype or type
        Data type (default: float64).
    path : Path or str, optional
        File path for the memory map. If None, uses a temporary file.

    Returns
    -------
    np.ndarray
        Memory-mapped array (read-write).
    """
    if path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mmap", delete=False)
        path = tmp.name
        tmp.close()

    path = Path(path)
    return np.memmap(str(path), dtype=dtype, mode="w+", shape=shape)


# ---------------------------------------------------------------------------
# Backend status query
# ---------------------------------------------------------------------------

def available_backends() -> dict[str, bool]:
    """Query which acceleration backends are available.

    Returns
    -------
    dict[str, bool]
        Backend name → availability.
    """
    return {
        "numpy": True,
        "numba": _NUMBA_AVAILABLE,
        "cupy": _CUPY_AVAILABLE,
    }
