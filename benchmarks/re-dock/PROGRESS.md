# RE-DOCK Development Progress

## v1 — Core Thermodynamics Engine
- `thermodynamics.py`: Van't Hoff analysis (linear + nonlinear ΔCp model)
- Metropolis exchange criterion for replica swaps
- Shannon configurational entropy over Boltzmann-weighted pose ensembles
- WHAM-reweighted free energy surface construction
- Data structures: DockingPose, ReplicaState, VantHoffResult
- Geometric temperature ladder generation with optimal replica count estimation

## v2 — Campaign Orchestrator
- `orchestrator.py`: Full campaign lifecycle management
- WorkerType enum (CLAUDE_MAX, CODEX, PERPLEXITY, GROK) with capability profiles
- DockingChunk: atomic job units with self-contained worker scripts
- Adaptive GA generations based on convergence state
- Temperature-dependent potential softening (1.0 at 298K → 2.0 at 600K)
- Checkpoint serialization/deserialization for cross-sandbox resume
- Worker assignment strategy: cold replicas → strongest workers

## v3 — CLI Interface + Visualizations
- `cli.py`: argparse CLI with subcommands (init, dispatch, ingest, status, analyze)
- `visualization.py`: Standalone Chart.js HTML dashboards
  - Van't Hoff plots (ln K vs 1/T with fit lines + thermodynamic parameter cards)
  - Energy convergence traces per replica
  - Shannon entropy landscape (S_config vs T)
  - Replica exchange acceptance heatmaps
- Dark theme, Le Bonhomme Pharma branding

## v4 — HuggingFace Spaces + Kubernetes
- `hf_space/app.py`: FastAPI + Gradio persistent worker node
  - REST endpoints: /dock, /exchange, /health, /history
  - Simulation fallback when flexaid-py unavailable
- `k8s/`: Full Kubernetes manifest suite
  - 8 worker Deployments (geometric T ladder 298–600K)
  - Tiered resource allocation (cold=4GB, warm=2GB, hot=1GB)
  - Headless Service + per-replica Services for DNS discovery
  - Exchange CronJob (*/5 min): collects poses, runs Metropolis, pushes swaps
  - Van't Hoff CronJob (*/15 min): thermodynamic analysis to persistent volume
  - PVCs: 50 GB data + 10 GB checkpoints

## v5 — Vercel Coordinator + Google Drive
- `vercel_coordinator/`: Serverless Python API on Vercel
  - Campaign init, dispatch, ingest, exchange, status, Van't Hoff endpoints
  - Worker registration and health monitoring
- `gdrive_store.py`: Google Drive as distributed checkpoint store
  - Service account (headless) + OAuth2 (interactive) authentication
  - Campaign folder hierarchy with versioned checkpoints
  - Worker heartbeat system
  - One-liner `quick_save()` for workers
- `deploy.sh`: Multi-target deployment (HF + K8s + Vercel)

## v6 — Demo Figures + Documentation
- Ran simulated 5-target benchmarking campaign (15 generations, 8 replicas)
- Generated Van't Hoff plots for all targets (R² > 0.999)
- ΔCp recovery within 5% for hydrophobic targets
- Energy convergence traces showing exchange-accelerated sampling
- Shannon entropy landscapes showing binding mode transitions
- Exchange acceptance heatmaps (96–100% acceptance in demo)
- Full architecture documentation (ARCHITECTURE.md)
- This progress report (PROGRESS.md)
- PR created for review

## Demo Results Summary

| Target | True ΔH | Recovered ΔH | True ΔCp | Recovered ΔCp | R²     |
|--------|---------|--------------|----------|---------------|--------|
| 1a30   | -12.50  | ~-13.4       | -350     | ~-359         | 0.9999 |
| 1err   | -15.20  | ~-16.2       | -280     | ~-287         | 0.9999 |
| 2bm2   | -9.80   | ~-12.1       | -200     | ~-198         | 0.9992 |
| 3htb   | -7.50   | ~-9.3        | -150     | ~-150         | 0.9997 |
| 4djh   | -18.00  | ~-21.1       | -420     | ~-413         | 0.9999 |

Units: ΔH in kcal/mol, ΔCp in cal/(mol·K)

## v7 — Bidirectional Round-Trip (Crooks Fluctuation Theorem)

### Core Physics Module: `crooks.py`
- `BidirectionalExchange` class managing forward (heating) + reverse (cooling) legs
- `bennett_acceptance_ratio()` — optimal ΔG from bidirectional work distributions
  via self-consistent BAR equation with log-sum-exp stable Fermi functions
- `crooks_intersection()` — ΔG where P(W_fwd) and P(-W_rev) cross, using
  Gaussian KDE with Silverman bandwidth
- `irreversible_entropy_production()` — σ_irr = (⟨W_F⟩ + ⟨W_R⟩ - 2ΔG) / T
- `landauer_information_loss()` — bits erased = σ_irr / (R·ln2)
- `shannon_energy_collapse_rate()` — dS/dT via central finite differences
- `mutual_information()` — I(hot; cold) = S_high + S_low - S_joint
- `convergence_check()` — physics-based: σ_irr < threshold
- Data classes: `WorkSample`, `LegResult`, `BidirectionalResult` (all with
  `__post_init__` validation and JSON serialization)

### Hardware Acceleration: `accelerator.py`
- `AcceleratorBackend` with auto-detection: CuPy (GPU) → Numba JIT → NumPy/BLAS
- `vectorized_metropolis_batch()` — N exchanges in one vectorized call
- `vectorized_work_accumulation()` — batch non-equilibrium work computation
- `vectorized_boltzmann_weights()` — log-sum-exp stable Boltzmann weights
- `vectorized_shannon_entropy()` — batch Shannon entropy over multiple ensembles
- `vectorized_bar_fermi()` — numerically stable log-Fermi for BAR iterations
- `vectorized_kde_grid()` — batch Gaussian KDE for Crooks intersection
- `mmap_array()` — memory-mapped arrays for large campaigns
- Numba `@njit(cache=True, fastmath=True)` kernels for hot loops
- CuPy GPU backend with transparent CPU↔GPU transfers

### Updated Modules
- `thermodynamics.py`: ReplicaState gains `forward_work`, `reverse_work`,
  `shannon_entropy_trace` fields; new `attempt_exchanges_with_work()` function
- `orchestrator.py`: BenchmarkCampaign gains `run_bidirectional_round()`,
  `run_bar_analysis()`, `check_bidirectional_convergence()` methods;
  bidirectional state in checkpoints
- `visualization.py`: Three new Chart.js dashboards:
  - `crooks_crossing_plot_html()` — forward/reverse work histograms with ΔG
  - `information_loss_plot_html()` — σ_irr and Landauer bits over generations
  - `collapse_rate_plot_html()` — dS/dT landscape (funnel vs fracture)
- `demo_run.py`: Full v7 demo with bidirectional analysis, σ_irr convergence
  tracking, and all new visualizations
- `__init__.py`: Updated module docstring for v7
- `ARCHITECTURE.md`: v7 bidirectional protocol and accelerator documentation
- `PROGRESS.md`: This entry
