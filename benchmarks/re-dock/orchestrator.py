"""
RE-DOCK Campaign Orchestrator (v7)
====================================

Distributed campaign management for multi-target replica exchange docking
across heterogeneous AI sandbox worker pools.

v7 additions: bidirectional round-trip mode with Crooks/BAR analysis.

Components
----------
- **WorkerType**: Enum of supported worker backends (CLAUDE_MAX, CODEX, etc.)
- **WorkerProfile**: Worker capabilities (GPU, memory, temperature, endpoint)
- **DockingTarget**: PDB ID + receptor/ligand paths + experimental ΔG
- **DockingChunk**: Work packet for a single (target, temperature, generation)
  with to_worker_script() for sandbox execution
- **BenchmarkCampaign**: Full campaign state machine with checkpoint/resume
  and v7 bidirectional exchange + BAR analysis methods

Workflow
--------
1. ``initialize``: Load targets, assign temperature ladder, create replicas
2. ``generate_chunks_for_generation``: Split work into per-worker packets
3. ``dispatch_generation``: Send chunks to available workers
4. ``process_chunk_result``: Ingest completed chunk, update replica state
5. ``run_exchange_round``: Metropolis exchange between adjacent temperatures
6. ``run_bidirectional_round``: v7 — forward + reverse legs with work recording
7. ``run_bar_analysis``: v7 — BAR free energy from bidirectional work
8. ``run_vant_hoff``: Van't Hoff decomposition on accumulated ΔG(T)
9. ``check_convergence``: Monitor ΔG convergence across generations
10. ``check_bidirectional_convergence``: v7 — σ_irr convergence
11. ``save_checkpoint / load_checkpoint``: Fault-tolerant persistence

Le Bonhomme Pharma / Najmanovich Research Group
"""

from __future__ import annotations

import json
import time
import hashlib
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

import numpy as np

from .thermodynamics import (
    DockingPose,
    ReplicaState,
    VantHoffResult,
    geometric_temperature_ladder,
    attempt_exchanges,
    attempt_exchanges_with_work,
    van_t_hoff_analysis,
    shannon_entropy_of_ensemble,
)
from .crooks import (
    BidirectionalExchange,
    BidirectionalResult,
)


# ---------------------------------------------------------------------------
# Worker types and profiles
# ---------------------------------------------------------------------------

class WorkerType(str, Enum):
    """Supported distributed worker backends."""
    CLAUDE_MAX = "claude_max"       # Claude Code Max sandbox (200K context)
    CODEX = "codex"                 # OpenAI Codex sandbox
    PERPLEXITY = "perplexity"       # Perplexity deep research sandbox
    GROK = "grok"                   # Grok sandbox
    HF_SPACE = "hf_space"           # HuggingFace Space persistent worker
    K8S_POD = "k8s_pod"             # Kubernetes pod
    LOCAL = "local"                 # Local execution


@dataclass
class WorkerProfile:
    """Capabilities and state of a single worker node."""
    worker_id: str
    worker_type: WorkerType
    endpoint_url: str = ""
    gpu_type: str = "none"          # e.g., "T4", "A100", "M1-GPU"
    memory_gb: float = 8.0
    assigned_temperature: float = 298.15
    replica_index: int = 0
    is_available: bool = True
    last_heartbeat: float = 0.0
    completed_chunks: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["worker_type"] = self.worker_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WorkerProfile":
        d["worker_type"] = WorkerType(d["worker_type"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Docking targets
# ---------------------------------------------------------------------------

@dataclass
class DockingTarget:
    """A single benchmark target (PDB complex)."""
    pdb_id: str
    receptor_name: str = ""
    ligand_name: str = ""
    resolution_A: float = 0.0
    exp_dG_kcal: float = float("nan")
    receptor_path: str = ""
    ligand_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DockingTarget":
        # Accept either flat dict or nested JSON target format
        return cls(
            pdb_id=d.get("pdb_id", ""),
            receptor_name=d.get("receptor", d.get("receptor_name", "")),
            ligand_name=d.get("ligand", d.get("ligand_name", "")),
            resolution_A=d.get("resolution_A", 0.0),
            exp_dG_kcal=d.get("exp_dG_kcal", float("nan")),
            receptor_path=d.get("receptor_path", ""),
            ligand_path=d.get("ligand_path", ""),
        )


# ---------------------------------------------------------------------------
# Work packets
# ---------------------------------------------------------------------------

@dataclass
class DockingChunk:
    """Work packet: one (target, temperature, generation) unit."""
    chunk_id: str
    target: DockingTarget
    temperature: float
    replica_index: int
    generation: int
    ga_params: Dict[str, Any] = field(default_factory=lambda: {
        "population_size": 200,
        "num_generations": 50,
        "crossover_rate": 0.8,
        "mutation_rate": 0.02,
    })
    status: str = "pending"  # pending, dispatched, completed, failed
    assigned_worker: str = ""
    result_energies: List[float] = field(default_factory=list)
    result_best_energy: float = float("inf")
    dispatched_at: float = 0.0
    completed_at: float = 0.0

    def to_worker_script(self) -> str:
        """Generate a self-contained Python script for sandbox execution.

        The script runs a simulated FlexAID∆S GA at the assigned temperature
        and writes results to stdout as JSON.
        """
        script = f'''#!/usr/bin/env python3
"""RE-DOCK worker script — chunk {self.chunk_id}
Target: {self.target.pdb_id} | T={self.temperature:.1f}K | Gen={self.generation}
"""
import json, math, random, hashlib

# Parameters
PDB_ID = "{self.target.pdb_id}"
TEMPERATURE = {self.temperature}
GENERATION = {self.generation}
POP_SIZE = {self.ga_params.get("population_size", 200)}
N_GENS = {self.ga_params.get("num_generations", 50)}
CROSSOVER = {self.ga_params.get("crossover_rate", 0.8)}
MUTATION = {self.ga_params.get("mutation_rate", 0.02)}

R_KCAL = 1.987204e-3
beta = 1.0 / (R_KCAL * TEMPERATURE)

# Deterministic seed from chunk parameters
seed = int(hashlib.md5(f"{{PDB_ID}}_{{TEMPERATURE}}_{{GENERATION}}".encode()).hexdigest()[:8], 16)
random.seed(seed)

# Simulated GA run (synthetic Boltzmann-distributed energies)
base_energy = -8.0 + random.gauss(0, 2)
energies = []
for g in range(N_GENS):
    gen_energies = [base_energy + random.gauss(0, 1.0 / math.sqrt(beta)) for _ in range(POP_SIZE)]
    best = min(gen_energies)
    if best < base_energy:
        base_energy = best
    energies.extend(gen_energies)

result = {{
    "chunk_id": "{self.chunk_id}",
    "pdb_id": PDB_ID,
    "temperature": TEMPERATURE,
    "generation": GENERATION,
    "best_energy": min(energies),
    "mean_energy": sum(energies) / len(energies),
    "energies_sample": sorted(energies)[:20],
    "n_poses": len(energies),
}}
print(json.dumps(result))
'''
        return script

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target"] = self.target.to_dict()
        return d


# ---------------------------------------------------------------------------
# Campaign state machine
# ---------------------------------------------------------------------------

class BenchmarkCampaign:
    """Full distributed docking benchmark campaign with checkpoint/resume.

    Manages the lifecycle of a multi-target, multi-temperature replica
    exchange docking benchmark across heterogeneous worker pools.
    """

    def __init__(
        self,
        campaign_id: str = "",
        campaign_dir: str = "./campaign",
        T_min: float = 298.0,
        T_max: float = 600.0,
        n_replicas: int = 8,
    ):
        self.campaign_id = campaign_id or hashlib.md5(
            str(time.time()).encode()
        ).hexdigest()[:12]
        self.campaign_dir = Path(campaign_dir)
        self.T_min = T_min
        self.T_max = T_max
        self.n_replicas = n_replicas

        self.temperatures: List[float] = []
        self.targets: List[DockingTarget] = []
        self.workers: Dict[str, WorkerProfile] = {}
        self.replicas: Dict[str, List[ReplicaState]] = {}  # target_id -> replicas
        self.chunks: List[DockingChunk] = []
        self.current_generation: int = 0
        self.vant_hoff_results: Dict[str, VantHoffResult] = {}
        self.created_at: float = time.time()
        self.exchange_history: List[dict] = []

        # v7: Bidirectional analysis state
        self.bidirectional_engines: Dict[str, BidirectionalExchange] = {}
        self.bidirectional_results: Dict[str, BidirectionalResult] = {}
        self.bidirectional_history: List[dict] = []

    def initialize(self, targets_file: str) -> None:
        """Initialize campaign from a targets JSON file.

        Loads targets, generates temperature ladder, creates initial
        replica states for each target.
        """
        with open(targets_file) as f:
            data = json.load(f)

        self.targets = [DockingTarget.from_dict(t) for t in data.get("targets", [])]
        self.temperatures = geometric_temperature_ladder(
            self.T_min, self.T_max, self.n_replicas
        )

        # Initialize replica states for each target
        for target in self.targets:
            self.replicas[target.pdb_id] = [
                ReplicaState(
                    replica_index=i,
                    temperature=self.temperatures[i],
                )
                for i in range(self.n_replicas)
            ]

        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        self.save_checkpoint()

    def generate_chunks_for_generation(self, generation: int) -> List[DockingChunk]:
        """Generate work packets for all (target, replica) pairs at a generation."""
        chunks = []
        for target in self.targets:
            for i, T in enumerate(self.temperatures):
                chunk_id = f"{target.pdb_id}_T{T:.0f}_g{generation}"
                chunk = DockingChunk(
                    chunk_id=chunk_id,
                    target=target,
                    temperature=T,
                    replica_index=i,
                    generation=generation,
                )
                chunks.append(chunk)
        self.chunks.extend(chunks)
        return chunks

    def process_chunk_result(self, chunk_id: str, result: dict) -> None:
        """Ingest a completed chunk result into campaign state."""
        # Find the chunk
        chunk = next((c for c in self.chunks if c.chunk_id == chunk_id), None)
        if chunk is None:
            raise ValueError(f"Unknown chunk: {chunk_id}")

        chunk.status = "completed"
        chunk.completed_at = time.time()
        chunk.result_energies = result.get("energies_sample", [])
        chunk.result_best_energy = result.get("best_energy", float("inf"))

        # Update replica state
        pdb_id = chunk.target.pdb_id
        if pdb_id in self.replicas:
            replica = self.replicas[pdb_id][chunk.replica_index]
            for e in chunk.result_energies:
                replica.add_pose(DockingPose(
                    energy_kcal=e,
                    generation=chunk.generation,
                    replica_index=chunk.replica_index,
                ))

    def run_exchange_round(
        self,
        target_id: str,
        rng: Optional[np.random.Generator] = None,
    ) -> List[dict]:
        """Run Metropolis replica exchange for a single target.

        Alternates even/odd pairing each call.
        """
        if target_id not in self.replicas:
            return []

        replicas = self.replicas[target_id]
        even_odd = len(self.exchange_history) % 2

        results = attempt_exchanges(replicas, even_odd=even_odd, rng=rng)

        exchange_record = {
            "target_id": target_id,
            "generation": self.current_generation,
            "timestamp": time.time(),
            "exchanges": [
                {"i": i, "j": j, "accepted": acc, "delta": d}
                for i, j, acc, d in results
            ],
        }
        self.exchange_history.append(exchange_record)
        return exchange_record["exchanges"]

    def run_vant_hoff(self, target_id: str, fit_dCp: bool = False) -> Optional[VantHoffResult]:
        """Run Van't Hoff analysis on accumulated data for a target."""
        if target_id not in self.replicas:
            return None

        replicas = self.replicas[target_id]
        temperatures = []
        free_energies = []

        for replica in replicas:
            if not replica.poses:
                continue

            energies = [p.energy_kcal for p in replica.poses]
            # Approximate ΔG from Boltzmann-weighted ensemble
            beta = replica.beta
            E_arr = np.array(energies)
            max_neg_bE = np.max(-beta * E_arr)
            log_Z = max_neg_bE + np.log(np.sum(np.exp(-beta * E_arr - max_neg_bE)))
            dG = -log_Z / beta

            temperatures.append(replica.temperature)
            free_energies.append(dG)

        if len(temperatures) < 3:
            return None

        result = van_t_hoff_analysis(
            temperatures, free_energies, fit_dCp=fit_dCp
        )
        self.vant_hoff_results[target_id] = result
        return result

    def check_convergence(
        self,
        target_id: str,
        window: int = 5,
        threshold_kcal: float = 0.1,
    ) -> bool:
        """Check if ΔG estimates have converged over recent generations.

        Convergence criterion: standard deviation of ΔG over the last
        ``window`` generations is below ``threshold_kcal``.
        """
        if target_id not in self.replicas:
            return False

        # Collect best energies per generation from the lowest-T replica
        replica_0 = self.replicas[target_id][0]
        if len(replica_0.poses) < window:
            return False

        recent = [p.energy_kcal for p in replica_0.poses[-window:]]
        return float(np.std(recent)) < threshold_kcal

    # -------------------------------------------------------------------
    # v7: Bidirectional round-trip methods
    # -------------------------------------------------------------------

    def run_bidirectional_round(
        self,
        target_id: str,
        n_sweeps: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run a full bidirectional round-trip (forward + reverse) for a target.

        1. Forward leg: heating exchanges T_low → T_high, recording W_fwd
        2. Reverse leg: cooling exchanges T_high → T_low, recording W_rev

        Parameters
        ----------
        target_id : str
            PDB ID of the target.
        n_sweeps : int
            Number of exchange sweeps per leg.
        rng : np.random.Generator, optional

        Returns
        -------
        dict or None
            Summary of the round-trip including work statistics.
        """
        if target_id not in self.replicas:
            return None

        replicas = self.replicas[target_id]

        # Lazy-initialize bidirectional engine
        if target_id not in self.bidirectional_engines:
            self.bidirectional_engines[target_id] = BidirectionalExchange(
                temperatures=[r.temperature for r in replicas],
                reference_temperature=replicas[0].temperature,
            )

        engine = self.bidirectional_engines[target_id]

        # Forward leg (heating)
        fwd = engine.run_forward_leg(replicas, n_sweeps=n_sweeps, rng=rng)

        # Reverse leg (cooling)
        rev = engine.run_reverse_leg(replicas, n_sweeps=n_sweeps, rng=rng)

        record = {
            "target_id": target_id,
            "generation": self.current_generation,
            "timestamp": time.time(),
            "forward_mean_work": fwd.mean_work,
            "reverse_mean_work": rev.mean_work,
            "forward_n_samples": fwd.n_samples,
            "reverse_n_samples": rev.n_samples,
            "forward_acceptance": fwd.acceptance_rate,
            "reverse_acceptance": rev.acceptance_rate,
        }
        self.bidirectional_history.append(record)
        return record

    def run_bar_analysis(
        self,
        target_id: str,
    ) -> Optional[BidirectionalResult]:
        """Run BAR analysis on accumulated bidirectional work for a target.

        Requires at least one bidirectional round to have been completed.

        Parameters
        ----------
        target_id : str
            PDB ID of the target.

        Returns
        -------
        BidirectionalResult or None
        """
        if target_id not in self.bidirectional_engines:
            return None

        engine = self.bidirectional_engines[target_id]
        try:
            result = engine.analyze()
        except ValueError:
            return None

        self.bidirectional_results[target_id] = result
        return result

    def check_bidirectional_convergence(
        self,
        target_id: str,
        threshold: float = 1e-3,
    ) -> bool:
        """Check bidirectional convergence via σ_irr → 0.

        Parameters
        ----------
        target_id : str
            PDB ID of the target.
        threshold : float
            σ_irr convergence threshold in kcal/(mol·K).

        Returns
        -------
        bool
            True if σ_irr < threshold.
        """
        result = self.bidirectional_results.get(target_id)
        if result is None:
            return False
        return result.converged

    def save_checkpoint(self) -> Path:
        """Serialize campaign state to JSON checkpoint."""
        checkpoint = {
            "campaign_id": self.campaign_id,
            "created_at": self.created_at,
            "current_generation": self.current_generation,
            "T_min": self.T_min,
            "T_max": self.T_max,
            "n_replicas": self.n_replicas,
            "temperatures": self.temperatures,
            "targets": [t.to_dict() for t in self.targets],
            "workers": {k: v.to_dict() for k, v in self.workers.items()},
            "replicas": {
                k: [r.to_dict() for r in v]
                for k, v in self.replicas.items()
            },
            "chunks": [c.to_dict() for c in self.chunks[-100:]],  # Keep last 100
            "vant_hoff_results": {
                k: v.to_dict() for k, v in self.vant_hoff_results.items()
            },
            "exchange_history": self.exchange_history[-50:],  # Keep last 50
            # v7: Bidirectional state
            "bidirectional_results": {
                k: v.to_dict() for k, v in self.bidirectional_results.items()
            },
            "bidirectional_history": self.bidirectional_history[-50:],
        }

        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        path = self.campaign_dir / "checkpoint.json"
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        return path

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str) -> "BenchmarkCampaign":
        """Restore campaign from JSON checkpoint."""
        with open(checkpoint_path) as f:
            data = json.load(f)

        campaign = cls(
            campaign_id=data["campaign_id"],
            campaign_dir=str(Path(checkpoint_path).parent),
            T_min=data["T_min"],
            T_max=data["T_max"],
            n_replicas=data["n_replicas"],
        )
        campaign.created_at = data["created_at"]
        campaign.current_generation = data["current_generation"]
        campaign.temperatures = data["temperatures"]
        campaign.targets = [DockingTarget.from_dict(t) for t in data["targets"]]
        campaign.workers = {
            k: WorkerProfile.from_dict(v) for k, v in data.get("workers", {}).items()
        }
        campaign.replicas = {
            k: [ReplicaState.from_dict(r) for r in v]
            for k, v in data.get("replicas", {}).items()
        }
        campaign.vant_hoff_results = {
            k: VantHoffResult(**v) for k, v in data.get("vant_hoff_results", {}).items()
        }
        campaign.exchange_history = data.get("exchange_history", [])

        return campaign

    def dispatch_generation(self, generation: int) -> List[DockingChunk]:
        """Generate and prepare chunks for dispatch.

        Returns list of ready-to-dispatch chunks with worker scripts.
        Workers are assigned round-robin from available pool.
        """
        self.current_generation = generation
        chunks = self.generate_chunks_for_generation(generation)

        available = [w for w in self.workers.values() if w.is_available]

        for i, chunk in enumerate(chunks):
            if available:
                worker = available[i % len(available)]
                chunk.assigned_worker = worker.worker_id
                chunk.status = "dispatched"
                chunk.dispatched_at = time.time()

        self.save_checkpoint()
        return chunks
