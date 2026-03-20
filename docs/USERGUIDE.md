# User Guide

Complete reference for using FlexAID∆S — from zero-config docking to advanced thermodynamic analysis.

---

## Overview

FlexAID∆S is an entropy-driven molecular docking engine. It extends the FlexAID genetic algorithm with canonical ensemble thermodynamics, computing the Helmholtz free energy *F* = *H* − *TS* from the full conformational ensemble.

**Design philosophy**: All parameters have sensible defaults. A typical docking run requires only a receptor PDB and a ligand MOL2 file — no configuration needed.

---

## Zero-Config Docking

### Command Line

```bash
# Full flexibility + entropy at 300 K (all defaults)
./FlexAIDdS receptor.pdb ligand.mol2
```

This single command:
1. Detects binding site automatically (SURFNET cavity detection)
2. Runs genetic algorithm with full ligand flexibility
3. Computes configurational + vibrational entropy via ShannonThermoStack
4. Clusters poses and ranks binding modes by Helmholtz free energy
5. Outputs ranked binding modes as PDB files

### Python

```python
import flexaidds as fd

results = fd.dock(
    receptor='receptor.pdb',
    ligand='ligand.mol2',
)
for mode in results.rank_by_free_energy():
    print(f"Mode {mode.rank}: ΔG={mode.free_energy:.2f} kcal/mol")
```

---

## Command-Line Reference

### FlexAIDdS (Docking)

```
FlexAIDdS <receptor.pdb> <ligand.mol2> [options]
```

| Flag | Description |
|:-----|:------------|
| `-c <file>` | JSON config file (overrides defaults) |
| `-o <prefix>` | Output prefix for result files |
| `--rigid` | Disable all flexibility and entropy (enthalpy-only scoring) |
| `--folded` | Skip NATURaL co-translational chain growth |
| `--legacy` | Legacy 3-argument mode: `config.inp ga.inp output_prefix` |
| `--version` | Print version and exit |

### tENCoM (Vibrational Entropy)

```
tENCoM <reference.pdb> <target1.pdb> [target2.pdb ...] [options]
```

| Flag | Description |
|:-----|:------------|
| `-T <temp>` | Temperature in Kelvin (default: 300) |
| `-r <cutoff>` | Contact distance cutoff in Å |
| `-k <k0>` | Spring constant |
| `-o <prefix>` | Output prefix |

### Python CLI Inspector

```bash
python -m flexaidds /path/to/results/              # Summary table
python -m flexaidds /path/to/results/ --top 5      # Top 5 modes
python -m flexaidds /path/to/results/ --json        # JSON output
python -m flexaidds /path/to/results/ --csv out.csv  # CSV export
```

---

## JSON Configuration

All keys are optional — defaults enable full flexibility at 300 K. Override only what you need.

### Complete Example

```json
{
  "scoring": {
    "function": "VCT",
    "self_consistency": "MAX",
    "solvent_penalty": 0.0
  },
  "optimization": {
    "translation_step": 0.25,
    "angle_step": 5.0,
    "dihedral_step": 5.0,
    "flexible_step": 10.0,
    "grid_spacing": 0.375
  },
  "flexibility": {
    "ligand_torsions": true,
    "intramolecular": true,
    "ring_conformers": true,
    "chirality": true,
    "permeability": 1.0,
    "dee_clash": 0.5
  },
  "thermodynamics": {
    "temperature": 300,
    "clustering_algorithm": "CF",
    "cluster_rmsd": 2.0
  },
  "ga": {
    "num_chromosomes": 1000,
    "num_generations": 500,
    "crossover_rate": 0.8,
    "mutation_rate": 0.03,
    "fitness_model": "PSHARE",
    "reproduction_model": "BOOM",
    "seed": 0
  },
  "output": {
    "max_results": 10,
    "htp_mode": false
  },
  "protein": {
    "keep_ions": true,
    "keep_structural_waters": true,
    "structural_water_bfactor_max": 20.0
  },
  "advanced": {
    "assume_folded": false
  }
}
```

### Section Reference

#### `scoring`

| Key | Default | Options | Description |
|:----|:--------|:--------|:------------|
| `function` | `"VCT"` | `VCT`, `SPH` | Voronoi contact function or sphere approximation |
| `self_consistency` | `"MAX"` | — | Contact area handling mode |
| `solvent_penalty` | `0.0` | float | Solvent exposure penalty weight |

#### `optimization`

| Key | Default | Units | Description |
|:----|:--------|:------|:------------|
| `translation_step` | `0.25` | Å | Translation delta per GA step |
| `angle_step` | `5.0` | degrees | Bond angle delta |
| `dihedral_step` | `5.0` | degrees | Dihedral angle delta |
| `flexible_step` | `10.0` | degrees | Side-chain rotamer delta |
| `grid_spacing` | `0.375` | Å | Binding site grid resolution |

#### `flexibility`

| Key | Default | Description |
|:----|:--------|:------------|
| `ligand_torsions` | `true` | DEE torsion sampling |
| `intramolecular` | `true` | Intramolecular scoring |
| `ring_conformers` | `true` | Chair/boat/twist ring sampling |
| `chirality` | `true` | Explicit R/S discrimination |
| `permeability` | `1.0` | VDW permeability factor |
| `dee_clash` | `0.5` | DEE clash threshold |

#### `thermodynamics`

| Key | Default | Description |
|:----|:--------|:------------|
| `temperature` | `300` | Temperature in K (0 disables entropy) |
| `clustering_algorithm` | `"CF"` | Clustering: `CF` (centroid-first), `DP` (Density Peak), or `FO` (FastOPTICS) |
| `cluster_rmsd` | `2.0` | RMSD threshold for clustering (Å) |

#### `ga` (Genetic Algorithm)

| Key | Default | Description |
|:----|:--------|:------------|
| `num_chromosomes` | `1000` | Population size |
| `num_generations` | `500` | Number of generations |
| `crossover_rate` | `0.8` | Crossover probability |
| `mutation_rate` | `0.03` | Mutation probability |
| `fitness_model` | `"PSHARE"` | Fitness sharing model |
| `reproduction_model` | `"BOOM"` | Reproduction strategy |
| `seed` | `0` | RNG seed (0 = time-based) |

#### `protein`

| Key | Default | Description |
|:----|:--------|:------------|
| `keep_ions` | `true` | Retain metal ions for scoring (Mg²⁺, Zn²⁺, etc.) |
| `keep_structural_waters` | `true` | Retain ordered crystallographic waters |
| `structural_water_bfactor_max` | `20.0` | B-factor cutoff (Å²) for water selection |

---

## Common Workflows

### Virtual Screening (High-Throughput)

```json
{
  "ga": { "num_chromosomes": 500, "num_generations": 200 },
  "output": { "max_results": 3, "htp_mode": true },
  "thermodynamics": { "clustering_algorithm": "CF" }
}
```

```bash
for lig in ligands/*.mol2; do
    ./FlexAIDdS receptor.pdb "$lig" -c htp.json -o results/$(basename "$lig" .mol2)
done
```

### Accurate Binding Mode (Full Entropy)

```json
{
  "ga": { "num_chromosomes": 2000, "num_generations": 1000 },
  "thermodynamics": { "temperature": 300, "clustering_algorithm": "DP" },
  "flexibility": { "ring_conformers": true, "chirality": true }
}
```

### Enthalpy-Only Ranking (No Entropy)

```bash
./FlexAIDdS receptor.pdb ligand.mol2 --rigid
```

### Metalloprotein Docking

Metal ions are scored automatically when present in the PDB. Ensure ions are listed as HETATM records:

```json
{
  "protein": { "keep_ions": true }
}
```

### Docking with Structural Waters

Crystallographic waters with B-factor < 20 Å² are retained by default. To adjust:

```json
{
  "protein": {
    "keep_structural_waters": true,
    "structural_water_bfactor_max": 25.0
  }
}
```

### Co-Translational Assembly (NATURaL)

Activates automatically for nucleotide ligands or nucleic acid receptors:

```bash
./FlexAIDdS ribosome.pdb atp_analog.mol2
```

Skip chain growth for pre-folded structures:

```bash
./FlexAIDdS ribosome.pdb ligand.mol2 --folded
```

---

## Python API

### Docking

```python
import flexaidds as fd

results = fd.dock(
    receptor='receptor.pdb',
    ligand='ligand.mol2',
    binding_site='auto',
    compute_entropy=True,
)

for mode in results.binding_modes:
    print(f"Mode {mode.rank}: ΔG={mode.free_energy:.2f} kcal/mol, "
          f"S={mode.entropy:.3f} kcal/(mol·K)")
```

### Loading Existing Results

```python
docking = fd.load_results('output_prefix')
for mode in docking.binding_modes:
    print(f"Mode {mode.rank}: ΔG={mode.free_energy:.2f}")
```

### StatMechEngine

```python
from flexaidds import StatMechEngine

engine = StatMechEngine(temperature=300)
engine.add_energies(pose_energies)
thermo = engine.compute()

print(f"F  = {thermo.free_energy:.2f} kcal/mol")
print(f"S  = {thermo.entropy:.4f} kcal/(mol·K)")
print(f"Cv = {thermo.heat_capacity:.4f} kcal/(mol·K²)")
```

### Vibrational Entropy

```python
from flexaidds import ENCoMEngine, TorsionalENM, run_shannon_thermo_stack

# ENCoM: apo vs holo comparison
delta_s = ENCoMEngine.compute_delta_s('apo.pdb', 'holo.pdb')

# Full entropy pipeline
tenm = TorsionalENM()
tenm.build_from_pdb('receptor.pdb')
result = run_shannon_thermo_stack(
    energies=pose_energies,
    tencm_model=tenm,
    base_deltaG=-12.5,
    temperature_K=300.0,
)
print(f"ΔG = {result.deltaG:.4f} kcal/mol")
print(f"S_vib = {result.torsionalVibEntropy:.6f} kcal/(mol·K)")
```

### Module Index

| Module | Key Classes & Functions |
|:-------|:-----------------------|
| `docking` | `dock()`, `Docking`, `BindingMode`, `BindingPopulation`, `Pose` |
| `thermodynamics` | `StatMechEngine`, `Thermodynamics`, Boltzmann LUT |
| `encom` | `ENCoMEngine`, `NormalMode`, `VibrationalEntropy` |
| `tencm` | `TorsionalENM`, `compute_shannon_entropy`, `run_shannon_thermo_stack` |
| `energy_matrix` | `EnergyMatrix` I/O, 256-type projection |
| `results` | `load_results()` file parser |
| `models` | `PoseResult`, `BindingModeResult`, `DockingResult` |
| `io` | PDB/MOL2/config I/O, `is_ion()` classifier |
| `visualization` | PyMOL integration helpers |

---

## PyMOL Plugin

### Installation

1. PyMOL → Plugin Manager → Install New Plugin
2. Select the `pymol_plugin/` directory
3. Restart PyMOL — access via Plugin → FlexAID∆S

Requires: `pip install -e python/`

### Commands

| Command | Description |
|:--------|:------------|
| `flexaids_load <dir> [temp]` | Load results from output directory |
| `flexaids_show_ensemble <mode>` | Display all poses in a binding mode |
| `flexaids_color_boltzmann <mode>` | Color poses by Boltzmann weight |
| `flexaids_thermo <mode>` | Print thermodynamic properties |
| `flexaids_entropy_heatmap <mode>` | Spatial entropy density heatmap |
| `flexaids_animate <m1> <m2>` | Interpolated animation between modes |
| `flexaids_itc_plot` | Enthalpy-entropy compensation plot |
| `flexaids_itc_compare <csv>` | Compare predictions with ITC data |
| `flexaids_dock <obj> <lig>` | Interactive docking from PyMOL |

---

## Bonhomme Fleet (Distributed Docking)

Fleet distributes docking workloads across Apple devices via iCloud Drive coordination.

### Architecture

- **FleetScheduler** (Swift actor) — coordinates work chunks across devices
- **Device-aware weighting** — adjusts for battery level, thermal state, and TFLOPS
- **Orphan recovery** — timed-out chunks reclaimed with exponential backoff
- **Encrypted transit** — ChaChaPoly encryption for secure computation
- **TypeScript PWA** — real-time Fleet dashboard with Mol* 3D viewer

### Requirements

- macOS 14+ / iOS 17+ (Swift package)
- iCloud Drive enabled across participating devices
- FlexAID∆S built with Metal acceleration recommended

---

## Tips & Best Practices

### Performance

- **CUDA/Metal acceleration** provides 3575×/412× speedup for Shannon entropy computation
- Use `-march=native` (enabled by default in LTO builds) for best CPU performance
- For virtual screening, use `htp_mode: true` with reduced GA parameters

### Accuracy

- **Always use entropy** — the `--rigid` flag is for quick screening only; entropy recovers correct binding modes that enthalpy-only scoring misses
- **Keep structural waters** — ordered waters contribute 0.4–3 kcal/mol to binding thermodynamics
- **Keep metal ions** — critical for metalloprotein binding geometry
- **Density Peak clustering** (`"DP"`) produces more distinct binding modes than centroid-first

### Supported Input Formats

| Format | Extension | Notes |
|:-------|:----------|:------|
| PDB | `.pdb` | Receptor and reference structures |
| MOL2 (Tripos) | `.mol2` | Ligand (preferred) |
| SDF/MOL V2000 | `.sdf`, `.mol` | Ligand (alternative) |
| INP (legacy) | `.inp` | Legacy FlexAID ligand format |

---

## Next Steps

- [Installation Guide](INSTALLATION.md) — build instructions and troubleshooting
- [Benchmarks](BENCHMARKS.md) — performance and accuracy data
- [Configuration Reference](../README.md#json-config) — full JSON config schema in README
