"""
RE-DOCK Vercel Serverless Coordinator
======================================

Serverless API coordinating the distributed replica exchange campaign.

Endpoints
---------
  POST /api/campaign/init       Initialize campaign
  POST /api/campaign/exchange   Trigger Metropolis exchange round
  GET  /api/campaign/status     Campaign status
  POST /api/workers/register    Register a worker node
  GET  /api/workers/health      Worker pool health
  GET  /api/dashboard           Redirect to status

State is stored in-memory per invocation (use Vercel KV for persistence
in production).

Le Bonhomme Pharma / Najmanovich Research Group
"""

from __future__ import annotations

import json
import math
import os
import time
from http.server import BaseHTTPRequestHandler
from typing import Dict, List, Any

import numpy as np

# ---------------------------------------------------------------------------
# In-memory state (per cold-start; use Vercel KV for persistence)
# ---------------------------------------------------------------------------

R_KCAL = 1.987204e-3
SECRET = os.environ.get("REDOCK_SECRET", "dev")

campaign_state: Dict[str, Any] = {
    "initialized": False,
    "workers": {},
    "temperatures": [],
    "exchange_history": [],
}


def geometric_ladder(T_min: float, T_max: float, n: int) -> List[float]:
    if n < 2:
        return [T_min]
    ratio = T_max / T_min
    return [T_min * ratio ** (i / (n - 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class handler(BaseHTTPRequestHandler):
    """Vercel Python serverless handler."""

    def _send_json(self, status: int, data: Any) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_GET(self) -> None:
        path = self.path.rstrip("/")

        if path == "/api/campaign/status":
            self._send_json(200, {
                "initialized": campaign_state["initialized"],
                "n_workers": len(campaign_state["workers"]),
                "temperatures": campaign_state["temperatures"],
                "exchange_rounds": len(campaign_state["exchange_history"]),
            })

        elif path == "/api/workers/health":
            now = time.time()
            health = {}
            for wid, info in campaign_state["workers"].items():
                age = now - info.get("last_seen", 0)
                health[wid] = {
                    "temperature": info.get("temperature", 0),
                    "status": "healthy" if age < 600 else "stale",
                    "last_seen_sec_ago": round(age, 1),
                }
            self._send_json(200, health)

        elif path == "/api/dashboard":
            self.send_response(302)
            self.send_header("Location", "/api/campaign/status")
            self.end_headers()

        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.rstrip("/")
        body = self._read_body()

        if path == "/api/campaign/init":
            n_replicas = body.get("n_replicas", 8)
            T_min = body.get("T_min", 298.0)
            T_max = body.get("T_max", 600.0)
            campaign_state["initialized"] = True
            campaign_state["temperatures"] = geometric_ladder(T_min, T_max, n_replicas)
            self._send_json(200, {
                "status": "initialized",
                "temperatures": campaign_state["temperatures"],
            })

        elif path == "/api/workers/register":
            wid = body.get("worker_id", "")
            if not wid:
                self._send_json(400, {"error": "worker_id required"})
                return
            campaign_state["workers"][wid] = {
                "temperature": body.get("temperature", 298.15),
                "endpoint": body.get("endpoint", ""),
                "last_seen": time.time(),
            }
            self._send_json(200, {"registered": wid})

        elif path == "/api/campaign/exchange":
            # Simplified exchange: expects worker energies in body
            energies = body.get("energies", {})  # {replica_index: energy}
            temps = campaign_state["temperatures"]

            if len(energies) < 2:
                self._send_json(400, {"error": "need at least 2 replica energies"})
                return

            # Sort by replica index
            sorted_indices = sorted(energies.keys(), key=int)
            results = []
            even_odd = len(campaign_state["exchange_history"]) % 2

            for idx in range(even_odd, len(sorted_indices) - 1, 2):
                i_str = sorted_indices[idx]
                j_str = sorted_indices[idx + 1]
                i, j = int(i_str), int(j_str)

                if i >= len(temps) or j >= len(temps):
                    continue

                beta_i = 1.0 / (R_KCAL * temps[i])
                beta_j = 1.0 / (R_KCAL * temps[j])
                E_i = energies[i_str]
                E_j = energies[j_str]
                delta = (beta_i - beta_j) * (E_i - E_j)

                accepted = delta <= 0 or np.random.random() < math.exp(-min(delta, 500))
                results.append({
                    "i": i, "j": j,
                    "accepted": bool(accepted),
                    "delta": float(delta),
                })

            campaign_state["exchange_history"].append({
                "timestamp": time.time(),
                "results": results,
            })

            self._send_json(200, {"exchanges": results})

        else:
            self._send_json(404, {"error": "not found"})

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
