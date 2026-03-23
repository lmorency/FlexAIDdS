"""Boltz-2 NIM client for protein-ligand structure and affinity prediction.

Provides a Python interface to the NVIDIA BioNeMo Boltz-2 NIM API for
deep-learning-based biomolecular structure prediction and binding affinity
estimation.  Uses only the standard library for HTTP (no external dependencies).

API reference: https://docs.nvidia.com/nim/bionemo/boltz2/1.0.0/api-reference.html
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class Boltz2Error(Exception):
    """Error returned by the Boltz-2 NIM API."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


# ---------------------------------------------------------------------------
# Request data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Boltz2Polymer:
    """A polymer chain (protein, DNA, or RNA) for structure prediction.

    Attributes:
        id: Chain identifier — single letter (A–Z) or 4-char PDB-style ID.
        molecule_type: One of ``"protein"``, ``"dna"``, ``"rna"``.
        sequence: Residue/nucleotide sequence (1–4096 characters).
        cyclic: Whether the polymer is cyclic (default ``False``).
    """

    id: str
    molecule_type: str
    sequence: str
    cyclic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "molecule_type": self.molecule_type,
            "sequence": self.sequence,
        }
        if self.cyclic:
            d["cyclic"] = True
        return d


@dataclass(frozen=True)
class Boltz2Ligand:
    """A small-molecule ligand for protein-ligand complex prediction.

    Exactly one of *smiles* or *ccd* must be provided.

    Attributes:
        id: Ligand identifier (e.g. ``"L1"``).
        smiles: SMILES string representation.
        ccd: Chemical Component Dictionary code (1–3 alphanumeric chars).
        predict_affinity: Enable binding affinity prediction for this ligand.
            At most one ligand per request may have this set to ``True``.
    """

    id: str
    smiles: Optional[str] = None
    ccd: Optional[str] = None
    predict_affinity: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"id": self.id}
        if self.smiles is not None:
            d["smiles"] = self.smiles
        if self.ccd is not None:
            d["ccd"] = self.ccd
        if self.predict_affinity:
            d["predict_affinity"] = True
        return d


@dataclass(frozen=True)
class PocketContact:
    """A residue contact defining part of a binding pocket.

    Attributes:
        id: Polymer chain ID that contains this residue.
        residue_index: 1-based residue index within the chain.
    """

    id: str
    residue_index: int

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "residue_index": self.residue_index}


@dataclass(frozen=True)
class PocketConstraint:
    """Pocket constraint restricting a ligand to a specific binding site.

    Attributes:
        binder: ID of the ligand being constrained.
        contacts: Residues forming the binding pocket.
    """

    binder: str
    contacts: Tuple[PocketContact, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_type": "pocket",
            "binder": self.binder,
            "contacts": [c.to_dict() for c in self.contacts],
        }


# ---------------------------------------------------------------------------
# Response data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Boltz2AffinityResult:
    """Binding affinity prediction for a single ligand.

    Attributes:
        ligand_id: ID of the ligand this result corresponds to.
        pic50: Predicted pIC50 values (one per diffusion sample).
        pred_value: Predicted log IC50 values.
        probability_binary: Binding probability [0, 1] per sample.
    """

    ligand_id: str
    pic50: Tuple[float, ...]
    pred_value: Tuple[float, ...]
    probability_binary: Tuple[float, ...]


@dataclass(frozen=True)
class Boltz2PredictionResult:
    """Result from a Boltz-2 structure/affinity prediction.

    Attributes:
        structures: Predicted structure strings (one per diffusion sample),
            in the requested output format (default mmCIF).
        affinities: Mapping of ligand ID to affinity results.  Empty when
            no ligand requested affinity prediction.
        scores: Optional model confidence scores.
        runtime: Optional runtime performance metrics.
    """

    structures: Tuple[str, ...]
    affinities: Dict[str, Boltz2AffinityResult] = field(default_factory=dict)
    scores: Optional[Dict[str, Any]] = None
    runtime: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_PREDICT_PATH = "/biology/mit/boltz2/predict"
_HEALTH_PATH = "/v1/health/ready"

_MAX_POLYMERS = 12
_MAX_LIGANDS = 20
_MAX_SEQUENCE_LENGTH = 4096


def _validate_inputs(
    polymers: List[Boltz2Polymer],
    ligands: Optional[List[Boltz2Ligand]],
) -> None:
    """Validate request inputs before sending to the API."""
    if not polymers:
        raise ValueError("At least one polymer is required.")
    if len(polymers) > _MAX_POLYMERS:
        raise ValueError(
            f"Maximum {_MAX_POLYMERS} polymers allowed, got {len(polymers)}."
        )

    for p in polymers:
        if p.molecule_type not in ("protein", "dna", "rna"):
            raise ValueError(
                f"Invalid molecule_type {p.molecule_type!r} for polymer {p.id!r}. "
                "Must be 'protein', 'dna', or 'rna'."
            )
        if not p.sequence or len(p.sequence) > _MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"Polymer {p.id!r} sequence must be 1–{_MAX_SEQUENCE_LENGTH} "
                f"characters, got {len(p.sequence)}."
            )

    if ligands:
        if len(ligands) > _MAX_LIGANDS:
            raise ValueError(
                f"Maximum {_MAX_LIGANDS} ligands allowed, got {len(ligands)}."
            )
        affinity_count = sum(1 for lig in ligands if lig.predict_affinity)
        if affinity_count > 1:
            raise ValueError(
                "At most one ligand may have predict_affinity=True, "
                f"got {affinity_count}."
            )
        for lig in ligands:
            has_smiles = lig.smiles is not None
            has_ccd = lig.ccd is not None
            if has_smiles == has_ccd:
                raise ValueError(
                    f"Ligand {lig.id!r} must have exactly one of 'smiles' or "
                    "'ccd', not both or neither."
                )


def _build_payload(
    polymers: List[Boltz2Polymer],
    ligands: Optional[List[Boltz2Ligand]],
    constraints: Optional[List[PocketConstraint]],
    *,
    recycling_steps: int,
    sampling_steps: int,
    diffusion_samples: int,
    step_scale: float,
    output_format: str,
    sampling_steps_affinity: int,
    diffusion_samples_affinity: int,
    affinity_mw_correction: bool,
    write_full_pae: bool,
) -> Dict[str, Any]:
    """Build the JSON request payload."""
    payload: Dict[str, Any] = {
        "polymers": [p.to_dict() for p in polymers],
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "step_scale": step_scale,
        "output_format": output_format,
    }

    if ligands:
        payload["ligands"] = [lig.to_dict() for lig in ligands]
        has_affinity = any(lig.predict_affinity for lig in ligands)
        if has_affinity:
            payload["sampling_steps_affinity"] = sampling_steps_affinity
            payload["diffusion_samples_affinity"] = diffusion_samples_affinity
            if affinity_mw_correction:
                payload["affinity_mw_correction"] = True

    if constraints:
        payload["constraints"] = [c.to_dict() for c in constraints]

    if write_full_pae:
        payload["write_full_pae"] = True

    return payload


def _parse_response(data: Dict[str, Any]) -> Boltz2PredictionResult:
    """Parse the JSON response into a Boltz2PredictionResult."""
    # Extract structures
    structures_raw = data.get("output", data.get("structures", []))
    if isinstance(structures_raw, list):
        structures = []
        for item in structures_raw:
            if isinstance(item, dict):
                structures.append(item.get("structure", str(item)))
            else:
                structures.append(str(item))
    else:
        structures = [str(structures_raw)] if structures_raw else []

    # Extract affinities
    affinities: Dict[str, Boltz2AffinityResult] = {}
    raw_affinities = data.get("affinities", {})
    if isinstance(raw_affinities, dict):
        for lig_id, aff_data in raw_affinities.items():
            if isinstance(aff_data, dict):
                affinities[lig_id] = Boltz2AffinityResult(
                    ligand_id=lig_id,
                    pic50=tuple(aff_data.get("affinity_pic50", [])),
                    pred_value=tuple(aff_data.get("affinity_pred_value", [])),
                    probability_binary=tuple(
                        aff_data.get("affinity_probability_binary", [])
                    ),
                )

    return Boltz2PredictionResult(
        structures=tuple(structures),
        affinities=affinities,
        scores=data.get("scores"),
        runtime=data.get("runtime"),
    )


class Boltz2Client:
    """Client for the NVIDIA BioNeMo Boltz-2 NIM prediction API.

    Supports both local NIM deployment (``http://localhost:8000``) and
    the NVIDIA cloud API (``https://integrate.api.nvidia.com/v1``).

    Example::

        >>> client = Boltz2Client()
        >>> result = client.predict_protein_ligand(
        ...     protein_sequence="MALWMRLLPLLALLALWGPDPAAAFVN...",
        ...     ligand_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        ...     predict_affinity=True,
        ... )
        >>> print(result.structures[0][:80])
        >>> if result.affinities:
        ...     aff = result.affinities["L1"]
        ...     print(f"pIC50: {aff.pic50}")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 300,
    ):
        """Initialize the Boltz-2 NIM client.

        Args:
            base_url: NIM server URL.  Defaults to ``http://localhost:8000``
                for local deployment.  Use
                ``https://integrate.api.nvidia.com/v1`` for the NVIDIA
                cloud API.
            api_key: NVIDIA API key (``nvapi-...``).  Falls back to the
                ``NVIDIA_API_KEY`` environment variable.  Required for
                cloud endpoints.
            timeout: HTTP request timeout in seconds (default 300).
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self._timeout = timeout

    def _make_request(
        self, path: str, payload: Optional[Dict[str, Any]] = None, method: str = "GET"
    ) -> Dict[str, Any]:
        """Send an HTTP request and return the parsed JSON response."""
        url = f"{self._base_url}{path}"
        headers = {"User-Agent": "FlexAIDdS-Boltz2Client"}

        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        if payload is not None:
            method = "POST"
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        else:
            body = None

        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            resp_body = None
            try:
                resp_body = exc.read().decode("utf-8")
            except Exception:
                pass
            raise Boltz2Error(
                f"Boltz-2 API returned HTTP {exc.code}: {exc.reason}",
                status_code=exc.code,
                response_body=resp_body,
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise ConnectionError(
                f"Failed to connect to Boltz-2 NIM at {url}: {exc}"
            ) from exc

    def health_check(self) -> bool:
        """Check if the NIM service is ready.

        Returns:
            ``True`` if the health endpoint responds successfully.
        """
        try:
            self._make_request(_HEALTH_PATH)
            return True
        except (Boltz2Error, ConnectionError):
            return False

    def predict(
        self,
        polymers: List[Boltz2Polymer],
        ligands: Optional[List[Boltz2Ligand]] = None,
        constraints: Optional[List[PocketConstraint]] = None,
        *,
        recycling_steps: int = 3,
        sampling_steps: int = 50,
        diffusion_samples: int = 1,
        step_scale: float = 1.638,
        output_format: str = "mmcif",
        sampling_steps_affinity: int = 200,
        diffusion_samples_affinity: int = 5,
        affinity_mw_correction: bool = False,
        write_full_pae: bool = False,
    ) -> Boltz2PredictionResult:
        """Run a Boltz-2 structure prediction.

        This is the full-featured prediction method.  For simple
        protein-ligand cases, see :meth:`predict_protein_ligand`.

        Args:
            polymers: List of polymer chains (1–12).
            ligands: Optional list of ligands (up to 20).
            constraints: Optional pocket constraints.
            recycling_steps: Iterative refinement steps (1–10, default 3).
            sampling_steps: Diffusion sampling steps (default 50).
            diffusion_samples: Independent structural samples (default 1).
            step_scale: Diffusion step scaling factor (default 1.638).
            output_format: Output structure format (default ``"mmcif"``).
            sampling_steps_affinity: Sampling steps for affinity (default 200).
            diffusion_samples_affinity: Affinity diffusion samples (default 5).
            affinity_mw_correction: Molecular weight correction (default False).
            write_full_pae: Return full PAE matrix (default False).

        Returns:
            :class:`Boltz2PredictionResult` with structures and affinities.

        Raises:
            ValueError: Invalid inputs (too many polymers, bad sequence, etc.).
            Boltz2Error: API returned an error response.
            ConnectionError: Network connection failure.
        """
        _validate_inputs(polymers, ligands)

        payload = _build_payload(
            polymers,
            ligands,
            constraints,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            diffusion_samples=diffusion_samples,
            step_scale=step_scale,
            output_format=output_format,
            sampling_steps_affinity=sampling_steps_affinity,
            diffusion_samples_affinity=diffusion_samples_affinity,
            affinity_mw_correction=affinity_mw_correction,
            write_full_pae=write_full_pae,
        )

        data = self._make_request(_PREDICT_PATH, payload)
        return _parse_response(data)

    def predict_protein_ligand(
        self,
        protein_sequence: str,
        ligand_smiles: str,
        *,
        protein_id: str = "A",
        ligand_id: str = "L1",
        predict_affinity: bool = False,
        pocket_residues: Optional[List[int]] = None,
        recycling_steps: int = 3,
        sampling_steps: int = 50,
        diffusion_samples: int = 1,
        step_scale: float = 1.638,
        output_format: str = "mmcif",
        sampling_steps_affinity: int = 200,
        diffusion_samples_affinity: int = 5,
    ) -> Boltz2PredictionResult:
        """Predict a protein-ligand complex structure and optional affinity.

        Convenience wrapper around :meth:`predict` for the common case of
        a single protein chain and a single small-molecule ligand.

        Args:
            protein_sequence: Amino acid sequence (1–4096 residues).
            ligand_smiles: SMILES string for the ligand.
            protein_id: Chain ID for the protein (default ``"A"``).
            ligand_id: ID for the ligand (default ``"L1"``).
            predict_affinity: Predict binding affinity (default ``False``).
            pocket_residues: Optional list of 1-based residue indices
                defining the binding pocket.  When provided, a pocket
                constraint is added to guide ligand placement.
            recycling_steps: Iterative refinement steps (1–10, default 3).
            sampling_steps: Diffusion sampling steps (default 50).
            diffusion_samples: Independent structural samples (default 1).
            step_scale: Diffusion step scaling factor (default 1.638).
            output_format: Output structure format (default ``"mmcif"``).
            sampling_steps_affinity: Sampling steps for affinity (default 200).
            diffusion_samples_affinity: Affinity diffusion samples (default 5).

        Returns:
            :class:`Boltz2PredictionResult` with predicted complex structure
            and optional affinity scores.

        Example::

            >>> client = Boltz2Client()
            >>> result = client.predict_protein_ligand(
            ...     protein_sequence="MKTAYIAKQ...",
            ...     ligand_smiles="c1ccccc1",
            ...     predict_affinity=True,
            ...     pocket_residues=[10, 15, 42],
            ... )
        """
        polymers = [
            Boltz2Polymer(
                id=protein_id,
                molecule_type="protein",
                sequence=protein_sequence,
            )
        ]
        ligands = [
            Boltz2Ligand(
                id=ligand_id,
                smiles=ligand_smiles,
                predict_affinity=predict_affinity,
            )
        ]
        constraints = None
        if pocket_residues:
            constraints = [
                PocketConstraint(
                    binder=ligand_id,
                    contacts=tuple(
                        PocketContact(id=protein_id, residue_index=idx)
                        for idx in pocket_residues
                    ),
                )
            ]

        return self.predict(
            polymers,
            ligands,
            constraints,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            diffusion_samples=diffusion_samples,
            step_scale=step_scale,
            output_format=output_format,
            sampling_steps_affinity=sampling_steps_affinity,
            diffusion_samples_affinity=diffusion_samples_affinity,
        )
