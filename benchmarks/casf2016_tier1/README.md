# CASF-2016 Tier-1 Benchmark Bundle

Level-1 reproducibility bundle for fast CI-viable validation of FlexAIDdS
scoring and docking power on 5 CASF-2016 protein-ligand complexes.

## Purpose

Validates that FlexAIDdS produces binding affinity predictions and pose
rankings consistent with published baselines on a representative subset of
the PDBbind Core Set 2016. This is a **tier-1** (fast) subset — the full
285-complex validation runs as tier-2 separately.

## Targets

| PDB  | Protein                 |
|:-----|:------------------------|
| 1a30 | Neuraminidase           |
| 1gpk | Carbonic anhydrase II   |
| 1hvr | HIV-1 protease          |
| 2cgr | CDK2                    |
| 2hb1 | Thrombin                |

## Usage

```bash
# 1. Download structures
bash benchmarks/casf2016_tier1/download.sh

# 2. Run benchmark
bash benchmarks/casf2016_tier1/run.sh

# 3. Check results against expected baselines
cat benchmarks/casf2016_tier1/results/report.md
```

## Expected Baselines

Defined in `manifest.yaml` under `expected_baselines`. Regression is flagged
when any metric drops below `baseline * (1 - baseline_tolerance)`.

## Full Dataset

The complete CASF-2016 benchmark (285 complexes) is configured in
`benchmarks/datasets/casf2016.yaml`. The full PDBbind v2016 refined set
requires registration at http://www.pdbbind.org.cn/casf.asp.
