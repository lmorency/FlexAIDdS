# RE-DOCK Architecture

**Replica Exchange Distributed Orchestrated Docking Kit**
*FlexAID∆S Thermodynamic Benchmarking System*

Le Bonhomme Pharma / Najmanovich Research Group

## Core Insight

Replica Exchange Molecular Dynamics (RE-MD) maps directly onto distributed AI
sandbox workers. The constraints of AI services (rate limits, sandbox timeouts)
become the exchange interval of the replica exchange protocol. Van't Hoff
thermodynamic decomposition comes free from the temperature ladder.

| RE-MD Concept         | RE-DOCK Mapping                          |
|-----------------------|------------------------------------------|
| Replica at temp T_i   | Worker running FlexAID∆S with softening  |
| Exchange interval     | Rate limit window / sandbox timeout      |
| Configuration swap    | Best-pose JSON transfer between workers  |
| Temperature ladder    | Geometric T spacing across services      |
| Production replica    | Cold replica (298K) on strongest worker  |

## Mathematical Framework

### Van't Hoff Thermodynamic Decomposition

Linear model:

    ln(K) = -ΔH/(RT) + ΔS/R

Slope of ln(K) vs 1/T gives -ΔH/R, intercept gives ΔS/R.

Nonlinear extension (ΔCp ≠ 0):

    ln K(T) = ln K(T_ref) - (ΔH_ref/R)(1/T - 1/T_ref)
              + (ΔCp/R)(T_ref/T - 1 + ln(T/T_ref))

Curvature captures ΔCp — the heat capacity change on binding.
Large negative ΔCp is THE thermodynamic fingerprint of hydrophobic burial.

### Metropolis Exchange Criterion

    P(swap i↔j) = min(1, exp[(β_i - β_j)(E_i - E_j)])

where β = 1/(k_B T). When a hot replica finds a low-energy pose and a cold
replica is trapped, exchange probability is high — escaping kinetic traps
without brute-forcing.

### Shannon Configurational Entropy

    S_config = -Σ p_i ln(p_i)

over Boltzmann-weighted pose ensemble at each temperature. This is the ∆S
metric — Shannon's Energy Collapse. High S_config = entropic binding (many
degenerate poses). Low S_config = enthalpic lock (single dominant pose).

### WHAM Free Energy Surface

Weighted Histogram Analysis reweights samples from all replicas to construct
the unbiased free energy profile at 298K:

    w_i = exp(-(β_ref - β_i) × E_i)
    F(x) = -kT ln P(x)

## System Architecture

```
 ┌──────────────────────────────────────────────────────────┐
 │                  VERCEL COORDINATOR                      │
 │  Campaign init │ Job dispatch │ Exchange │ Van't Hoff    │
 └────────┬───────┬───────┬───────┬────────────────────────┘
          │       │       │       │
     ┌────▼──┐ ┌──▼───┐ ┌▼─────┐ ┌▼─────┐  ┌───────┐
     │Claude │ │Codex │ │Pplx  │ │Grok  │  │ K8s   │
     │Max    │ │      │ │Pro×4 │ │      │  │ Pods  │
     │298K   │ │350K  │ │425K  │ │525K  │  │×8     │
     │Worker │ │Worker│ │Worker│ │Worker│  │Worker │
     └───┬───┘ └──┬───┘ └──┬───┘ └──┬───┘  └──┬────┘
         │        │        │        │          │
         ▼        ▼        ▼        ▼          ▼
 ┌──────────────────────────────────────────────────────────┐
 │               GOOGLE DRIVE CHECKPOINT STORE              │
 │  campaigns/{id}/checkpoints/  results/  exchanges/       │
 │  campaigns/{id}/vanthoff/     poses/                     │
 │  workers/{id}_heartbeat.json                             │
 └──────────────────────────────────────────────────────────┘
         │
         ▼
 ┌──────────────────────────────────────────────────────────┐
 │  HuggingFace Spaces (persistent, 24/7, no rate limits)  │
 │  8 Spaces × FastAPI+Gradio = 8 permanent docking nodes  │
 └──────────────────────────────────────────────────────────┘
```

## Temperature Ladder Design

Geometric spacing ensures approximately equal exchange acceptance between
adjacent replicas:

    T_i = T_min × (T_max/T_min)^(i/(n-1))

For 8 replicas (298–600K):

| Replica | T (K)  | Softening | Worker Tier         | Resources      |
|---------|--------|-----------|---------------------|----------------|
| R0      | 298.15 | 1.000     | Cold (production)   | 2 CPU / 4 GB   |
| R1      | 329.47 | 1.104     | Cold                | 2 CPU / 4 GB   |
| R2      | 364.09 | 1.218     | Warm                | 1 CPU / 2 GB   |
| R3      | 402.34 | 1.345     | Warm                | 1 CPU / 2 GB   |
| R4      | 444.64 | 1.485     | Hot (exploration)   | 1 CPU / 2 GB   |
| R5      | 491.34 | 1.640     | Hot                 | 1 CPU / 1 GB   |
| R6      | 542.96 | 1.811     | Hot                 | 1 CPU / 1 GB   |
| R7      | 600.00 | 2.000     | Hot                 | 1 CPU / 1 GB   |

Cold replicas get more resources (precision matters). Hot replicas get less
(broad exploration, sampling noise tolerated).

## Infrastructure

### HuggingFace Spaces (Persistent Workers)
- 8 Spaces, each a FastAPI + Gradio app running FlexAID∆S at fixed T
- Endpoints: `/dock`, `/exchange`, `/health`, `/history`
- Free tier, runs 24/7, no rate limits, no sandbox timeouts
- The actual compute backbone

### Kubernetes (Production Cluster)
- 8 worker Deployments with health/readiness probes
- Headless Service for DNS-based worker discovery
- Exchange CronJob (every 5 min): Metropolis criterion across all replicas
- Van't Hoff CronJob (every 15 min): thermodynamic analysis + checkpoint
- PersistentVolumeClaims: 50 GB data + 10 GB checkpoints

### Vercel (Serverless Coordinator)
- Campaign initialization and management
- Job dispatch to workers
- Result ingestion and exchange orchestration
- Live Van't Hoff endpoint for monitoring

### Google Drive (Distributed State)
- Shared checkpoint store accessible by all workers
- Campaign folder hierarchy: checkpoints, results, exchanges, vanthoff, poses
- Worker heartbeats for health monitoring
- Service account auth (headless) or OAuth2 (interactive)

## Deployment

```bash
# Deploy everything
./deploy.sh all

# Or individually
./deploy.sh hf       # 8 HuggingFace Space workers
./deploy.sh k8s      # Kubernetes cluster
./deploy.sh vercel   # Coordinator API
```

## Module Reference

| Module              | Purpose                                              |
|---------------------|------------------------------------------------------|
| thermodynamics.py   | Van't Hoff, Metropolis, Shannon entropy, WHAM        |
| orchestrator.py     | Campaign management, worker dispatch, checkpointing  |
| cli.py              | Human-in-the-loop CLI                                |
| visualization.py    | Chart.js HTML dashboards                             |
| demo_run.py         | Simulated campaign with synthetic thermodynamics     |
| gdrive_store.py     | Google Drive distributed checkpoint store            |
| hf_space/app.py     | HuggingFace Space worker node                        |
| vercel_coordinator/ | Serverless coordinator API                           |
| k8s/                | Kubernetes manifests                                 |
| deploy.sh           | Multi-target deployment script                       |
