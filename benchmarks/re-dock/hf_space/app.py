"""
RE-DOCK HuggingFace Space Worker Node
=======================================

FastAPI + Gradio application serving as a persistent replica exchange
worker node in the distributed docking benchmark system.

Endpoints
---------
  POST /api/dock       Submit a docking job (target PDB + ligand + temperature)
  POST /api/exchange   Accept replica exchange proposal from coordinator
  GET  /api/status     Worker health, temperature, and current job status
  GET  /api/results    Retrieve completed docking results

Gradio UI
---------
  Real-time docking progress display with energy traces.

Each HF Space instance represents one temperature in the replica exchange
ladder. The worker runs FlexAID∆S at its assigned temperature, stores
results locally, and responds to exchange proposals from the coordinator.

Le Bonhomme Pharma / Najmanovich Research Group
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import gradio as gr
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

REPLICA_INDEX = int(os.environ.get("REPLICA_INDEX", "0"))
REPLICA_TEMPERATURE = float(os.environ.get("REPLICA_TEMPERATURE", "298.15"))
COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:8080")
WORKER_ID = os.environ.get("WORKER_ID", f"hf-worker-{REPLICA_INDEX}")

R_KCAL = 1.987204e-3  # kcal/(mol·K)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class DockRequest(BaseModel):
    pdb_id: str
    temperature: float = REPLICA_TEMPERATURE
    generation: int = 0
    population_size: int = 200
    num_generations: int = 50


class ExchangeProposal(BaseModel):
    partner_index: int
    partner_temperature: float
    partner_energy: float


class DockResult(BaseModel):
    pdb_id: str
    temperature: float
    generation: int
    best_energy: float
    mean_energy: float
    energies_sample: List[float]
    n_poses: int
    elapsed_sec: float


# ---------------------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------------------

@dataclass
class WorkerState:
    replica_index: int = REPLICA_INDEX
    temperature: float = REPLICA_TEMPERATURE
    current_energy: float = float("inf")
    best_energy: float = float("inf")
    total_jobs: int = 0
    total_exchanges: int = 0
    exchanges_accepted: int = 0
    is_busy: bool = False
    last_job_time: float = 0.0
    energy_history: List[float] = field(default_factory=list)


state = WorkerState()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title=f"RE-DOCK Worker {REPLICA_INDEX}", version="1.0.0")


@app.get("/api/status")
def get_status():
    return {
        "worker_id": WORKER_ID,
        "replica_index": state.replica_index,
        "temperature": state.temperature,
        "current_energy": state.current_energy,
        "best_energy": state.best_energy,
        "is_busy": state.is_busy,
        "total_jobs": state.total_jobs,
        "exchange_rate": (state.exchanges_accepted / state.total_exchanges
                          if state.total_exchanges > 0 else 0.0),
        "uptime": time.time() - state.last_job_time if state.last_job_time else 0,
    }


@app.post("/api/dock")
def submit_dock(req: DockRequest) -> DockResult:
    """Run a simulated docking job at this worker's temperature."""
    if state.is_busy:
        raise HTTPException(status_code=503, detail="Worker busy")

    state.is_busy = True
    t0 = time.time()

    try:
        rng = np.random.default_rng(
            hash(f"{req.pdb_id}_{req.temperature}_{req.generation}") % (2**32)
        )

        beta = 1.0 / (R_KCAL * req.temperature)
        sigma = math.sqrt(R_KCAL * req.temperature) * 2.0

        base_energy = -8.0 + rng.normal(0, 2)
        all_energies = []

        for g in range(req.num_generations):
            gen_e = rng.normal(base_energy, sigma, size=req.population_size)
            best_gen = float(np.min(gen_e))
            if best_gen < base_energy:
                base_energy = best_gen
            all_energies.extend(gen_e.tolist())

        best = min(all_energies)
        mean = sum(all_energies) / len(all_energies)

        state.current_energy = best
        if best < state.best_energy:
            state.best_energy = best
        state.total_jobs += 1
        state.energy_history.append(best)

        elapsed = time.time() - t0

        return DockResult(
            pdb_id=req.pdb_id,
            temperature=req.temperature,
            generation=req.generation,
            best_energy=best,
            mean_energy=mean,
            energies_sample=sorted(all_energies)[:20],
            n_poses=len(all_energies),
            elapsed_sec=elapsed,
        )
    finally:
        state.is_busy = False
        state.last_job_time = time.time()


@app.post("/api/exchange")
def exchange(proposal: ExchangeProposal):
    """Evaluate Metropolis exchange proposal."""
    state.total_exchanges += 1

    beta_self = 1.0 / (R_KCAL * state.temperature)
    beta_partner = 1.0 / (R_KCAL * proposal.partner_temperature)
    delta = (beta_self - beta_partner) * (state.current_energy - proposal.partner_energy)

    accepted = delta <= 0 or np.random.random() < math.exp(-delta)

    if accepted:
        state.exchanges_accepted += 1
        old_energy = state.current_energy
        state.current_energy = proposal.partner_energy
        return {"accepted": True, "swapped_energy": old_energy, "delta": delta}

    return {"accepted": False, "delta": delta}


@app.get("/api/results")
def get_results():
    return {
        "worker_id": WORKER_ID,
        "energy_history": state.energy_history[-100:],
        "best_energy": state.best_energy,
        "total_jobs": state.total_jobs,
    }


# ---------------------------------------------------------------------------
# Gradio monitoring UI
# ---------------------------------------------------------------------------

def status_display():
    return json.dumps(get_status(), indent=2)


def energy_plot():
    if not state.energy_history:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(state.energy_history, "b-", linewidth=1)
    ax.set_xlabel("Job")
    ax.set_ylabel("Best Energy (kcal/mol)")
    ax.set_title(f"Worker {REPLICA_INDEX} — T={state.temperature:.1f}K")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


with gr.Blocks(title=f"RE-DOCK Worker {REPLICA_INDEX}") as demo:
    gr.Markdown(f"# RE-DOCK Worker {REPLICA_INDEX} (T={REPLICA_TEMPERATURE:.1f}K)")

    with gr.Row():
        status_box = gr.JSON(label="Worker Status")
        refresh_btn = gr.Button("Refresh")

    plot_output = gr.Plot(label="Energy History")

    refresh_btn.click(fn=status_display, outputs=status_box)
    refresh_btn.click(fn=energy_plot, outputs=plot_output)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
