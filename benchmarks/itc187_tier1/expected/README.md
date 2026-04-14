# Expected Outputs

This directory will contain reference results after the first validated
benchmark run. Expected files per target:

- `{PDB}/best_pose.pdb` — Top-ranked docking pose
- `{PDB}/binding_modes.json` — Clustered binding modes with thermodynamic properties

Aggregated metrics (computed by DatasetRunner) must match the baselines
defined in `../manifest.yaml` within `baseline_tolerance`.
