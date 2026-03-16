# Bonhomme Fleet — Usability & Enhancement Proposals

> **Date**: 2026-03-14
> **Scope**: Swift FleetScheduler, TypeScript BonhommeViewer PWA, C++ hardware dispatch
> **Status**: Proposal

---

## 1. Executive Summary

Bonhomme Fleet is a distributed molecular docking system that leverages Apple's ecosystem (iCloud Drive, HealthKit, Metal, Apple Intelligence) to coordinate GA chromosome evaluation across multiple devices. The architecture is well-founded — actor-based concurrency, ChaChaPoly encryption, thermal-aware scheduling, and 1:1 Swift/TypeScript model alignment — but several gaps limit real-world usability. This document proposes actionable enhancements grouped by priority.

---

## 2. Critical Usability Fixes (Short-Term)

### 2.1 Chunk Retry & Orphan Recovery

**Problem**: `WorkChunk` has a `.failed` status but no retry logic. If a device goes offline mid-computation, chunks are permanently orphaned.

**Proposal**:
- Add a `timeoutSeconds` field to `WorkChunk` (default: 3600)
- Add a `claimedAt: Date?` timestamp
- In `FleetScheduler`, run a periodic sweep (`checkOrphanedChunks()`) that reclaims chunks where `now - claimedAt > timeout`
- Implement exponential backoff: retry up to 3 times, then mark `.permanentlyFailed`
- Emit a `ChunkOrphanedEvent` for the dashboard

**Files to modify**:
- `swift/Sources/FleetScheduler/WorkChunk.swift` — add timeout, claimedAt, retryCount fields
- `swift/Sources/FleetScheduler/FleetScheduler.swift` — add orphan sweep timer

### 2.2 Battery-Aware Scheduling

**Problem**: `DeviceCapability` tracks thermal state but ignores battery level. A device at 5% battery may accept a 2-hour chunk and die mid-computation.

**Proposal**:
- Add `batteryLevel: Float` and `isCharging: Bool` to `DeviceCapability`
- Apply a battery multiplier to `computeWeight`:
  - `< 20%` and not charging → weight × 0.0 (exclude)
  - `20-50%` and not charging → weight × 0.5
  - `> 50%` or charging → weight × 1.0
- Use `UIDevice.current.batteryLevel` (iOS) or `IOPSCopyPowerSourcesInfo` (macOS)

**Files to modify**:
- `swift/Sources/FleetScheduler/DeviceCapability.swift`

### 2.3 Graceful iCloud Unavailability

**Problem**: If iCloud is unreachable, the entire fleet halts with no fallback.

**Proposal**:
- Add a `FleetMode` enum: `.distributed` (iCloud fleet), `.localOnly` (single device)
- On iCloud failure, automatically fall back to `.localOnly` with a user notification
- Queue work locally; when iCloud reconnects, optionally redistribute pending chunks
- Add `isICloudAvailable()` health check to `FleetScheduler`

**Files to modify**:
- `swift/Sources/FleetScheduler/FleetScheduler.swift`

---

## 3. Fleet Intelligence Enhancements (Medium-Term)

### 3.1 Dynamic Work Rebalancing

**Problem**: Work is split proportionally once at job start. If a fast device finishes early while a slow device is still computing, the fast device sits idle.

**Proposal**:
- Implement work-stealing: when a device completes all its chunks, it can claim unclaimed or split large pending chunks from slow devices
- Track per-device throughput (chromosomes/second) and update `computeWeight` dynamically
- Add `splitChunk(chunk: WorkChunk, into: Int) -> [WorkChunk]` to subdivide work

**Files to modify**:
- `swift/Sources/FleetScheduler/FleetScheduler.swift`
- `swift/Sources/FleetScheduler/WorkChunk.swift` — add `parentChunkId` for split tracking

### 3.2 Distributed Replica Exchange (REMD)

**Problem**: Each device runs an independent GA subpopulation with no cross-device thermostat coupling, limiting conformational sampling.

**Proposal**:
- Assign each device a temperature from a ladder (e.g., 300K, 350K, 400K, 500K)
- After N generations, exchange chromosomes between adjacent-temperature devices via iCloud
- Accept/reject exchanges using Metropolis criterion: `P = min(1, exp((β_i - β_j)(E_i - E_j)))`
- This leverages fleet parallelism for enhanced sampling, not just throughput

**New files**:
- `swift/Sources/FleetScheduler/ReplicaExchange.swift`

### 3.3 Fleet Result Aggregation Pipeline

**Problem**: No code exists for merging `ChunkResult` populations across devices into a unified `BindingPopulation`.

**Proposal**:
- Collect all device poses into a global population
- Run FOPTICS/density-peak clustering across the merged set
- Recalculate partition function, Boltzmann weights, and thermodynamics globally
- Detect and deduplicate poses from retry/resubmission scenarios (RMSD-based)

**New files**:
- `swift/Sources/FleetScheduler/FleetAggregator.swift`

### 3.4 Oracle Learning Loop

**Problem**: Both `RuleBasedOracle` and `FoundationModels` oracle produce stateless analyses with no learning from past docking campaigns.

**Proposal**:
- Store past oracle analyses in a local Core Data store keyed by (receptor, ligand, binding mode)
- When analyzing a new population, retrieve similar past results as context
- For FoundationModels oracle: include historical context in the prompt
- For rule-based oracle: track which binding modes improved/worsened over campaigns and highlight trends

**Files to modify**:
- `swift/Sources/Intelligence/IntelligenceOracle.swift`

---

## 4. PWA / BonhommeViewer Enhancements

### 4.1 Real-Time Fleet Dashboard

**Problem**: `FleetDashboard.tsx` has placeholder polling. No live data flows from the fleet.

**Proposal**:
- **Option A (Simple)**: Expose a `/fleet/status.json` file in iCloud that the orchestrator updates every 10s. The PWA polls this file.
- **Option B (Real-time)**: Add a lightweight WebSocket relay (Node.js, < 100 LOC) that the orchestrator publishes to. PWA subscribes for live updates.
- Display per-device: chunk progress bar, chromosomes evaluated, estimated time remaining, thermal state icon

**Files to modify**:
- `typescript/apps/viewer/src/FleetDashboard.tsx`

### 4.2 Mol* 3D Viewer Integration

**Problem**: The Mol* viewer is referenced but not fully integrated.

**Proposal**:
- Load the top binding mode's PDB into Mol* on population load
- Color residues by Boltzmann-weighted contact frequency
- Animate between binding modes (morph interpolation)
- Overlay entropy heatmap on protein surface (Shannon S per residue)

**Files to modify**:
- `typescript/apps/viewer/src/App.tsx` — add `<MolstarViewer>` component

### 4.3 Comparative Analysis View

**Problem**: The PWA shows one population at a time. Users often want to compare two ligands or two receptor conformations side-by-side.

**Proposal**:
- Add a split-pane comparison mode
- Load two `BindingPopulation` results
- Show delta thermodynamics: ΔΔF, ΔΔS between the two
- Highlight binding modes unique to each population
- Oracle produces comparative analysis ("Ligand A binds 2.3 kcal/mol stronger but with lower entropy")

### 4.4 Export & Reporting

**Problem**: Results are viewed in-app only. No export for publications or reports.

**Proposal**:
- Add "Export PDF" with thermodynamic summary, top binding modes, oracle analysis, and Mol* screenshot
- Add "Export CSV" for pose-level data (energy, RMSD, Boltzmann weight, binding mode assignment)
- Add "Copy BibTeX" for citing FlexAIDdS in publications

---

## 5. C++ Hardware Dispatch Enhancements

### 5.1 Per-Chunk Performance Telemetry

**Problem**: `DispatchReport` is generated once at init. No per-evaluation telemetry exists for fleet performance tracking.

**Proposal**:
- Add a lightweight `DispatchTelemetry` struct: `{ backend, wall_time_ms, elements_processed, throughput_gflops }`
- Return from `boltzmann_weights_batch()` and `log_sum_exp()`
- Aggregate per-chunk for fleet dashboard metrics

**Files to modify**:
- `LIB/hardware_dispatch.h` / `LIB/hardware_dispatch.cpp`

### 5.2 Adaptive Backend Selection

**Problem**: Backend priority is fixed (CUDA > Metal > AVX-512 > ...). On some workloads, AVX-512 may outperform Metal for small batch sizes due to kernel launch overhead.

**Proposal**:
- Run a micro-benchmark at init (100 elements, each backend, 10 iterations)
- Select the fastest backend for each batch size range
- Cache the decision in `HardwareCapabilities`

**Files to modify**:
- `LIB/hardware_dispatch.cpp`

---

## 6. Security & Privacy

### 6.1 Result Encryption at Rest

**Problem**: Work chunks are encrypted in transit (ChaChaPoly) but results are written as plaintext JSON to iCloud.

**Proposal**:
- Encrypt `ChunkResult` files with the same symmetric key
- Key derivation: use a per-job key derived from a user passphrase via HKDF
- Store the key in Keychain, not in iCloud

### 6.2 Device Authorization

**Problem**: Any device with iCloud container access can claim chunks. No device pairing or whitelisting.

**Proposal**:
- On first fleet join, device generates an Ed25519 keypair (CryptoKit)
- Orchestrator maintains an `authorized_devices.json` with public keys
- Chunks include a challenge; only authorized devices can sign the claim
- Revocation: remove public key from the list

---

## 7. Operational Improvements

### 7.1 Structured Logging

**Proposal**:
- Add OSLog categories: `.fleet`, `.scheduler`, `.oracle`, `.health`
- Log all chunk state transitions: `pending → claimed → running → completed`
- Include device ID, chunk ID, wall time, thermal state at completion
- Export logs for debugging distributed issues

### 7.2 Fleet Health Dashboard Metrics

**Proposal**: Track and display:
- **Chunk success rate** (completed / total)
- **Mean chunk wall time** by device
- **Thermal throttle events** (count, duration)
- **iCloud sync latency** (submission → visibility on other devices)
- **Population quality** over time (best F, mean S, number of binding modes)

### 7.3 Configuration File

**Problem**: Constants like iCloud paths ("FleetJobs/", "FleetResults/"), entropy thresholds (0.1), and timeout values are hardcoded.

**Proposal**:
- Create a `FleetConfig` struct loaded from `fleet_config.json`
- Allow per-job override of: timeout, max retries, thermal threshold, temperature ladder, chunk size
- Validate config on load; reject invalid values with descriptive errors

---

## 8. Testing Gaps to Address

| Area | Current | Needed |
|------|---------|--------|
| FleetScheduler | Basic split/thermal tests | Orphan recovery, retry, rebalancing |
| WorkChunk encryption | Likely untested | Round-trip encrypt/decrypt, tampered data rejection |
| DeviceCapability | Snapshot test | Simulated thermal transitions, battery edge cases |
| IntelligenceOracle | Not visible | Threshold boundary tests, FoundationModels mock |
| FleetDashboard (TS) | None | Component render tests, mock fleet data |
| Result aggregation | None (not implemented) | Merge correctness, deduplication, partition function accuracy |
| Hardware dispatch telemetry | None (not implemented) | Throughput measurement accuracy |

---

## 9. Priority Roadmap

| Phase | Timeline | Items |
|-------|----------|-------|
| **P0** | 1-4 weeks | 2.1 Chunk retry, 2.2 Battery-aware, 2.3 iCloud fallback, 7.3 Config file |
| **P1** | 1-3 months | 3.1 Dynamic rebalancing, 3.3 Result aggregation, 4.1 Live dashboard, 7.1 Logging |
| **P2** | 3-6 months | 3.2 Distributed REMD, 4.2 Mol* integration, 5.1 Telemetry, 6.1 Encryption at rest |
| **P3** | 6-12 months | 3.4 Oracle learning, 4.3 Comparative view, 4.4 Export/PDF, 5.2 Adaptive backends, 6.2 Device auth |

---

## 10. Summary

The Bonhomme Fleet architecture is sound for research-scale distributed docking (2-10 Apple devices). The most impactful near-term improvements are **chunk retry/recovery** (prevents lost work), **battery-aware scheduling** (prevents device death mid-computation), and **result aggregation** (enables the fleet output to actually be used). Medium-term, **distributed REMD** and a **live dashboard** would transform the fleet from a throughput multiplier into a genuine enhanced-sampling platform.
