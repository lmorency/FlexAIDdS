# Python API Reference

## Core Modules

### `flexaidds.dock()`

High-level docking interface.

```python
result = flexaidds.dock("receptor.pdb", "ligand.mol2")
```

### `flexaidds.StatMechEngine`

Statistical mechanics engine for ensemble thermodynamics.

### `flexaidds.ENCoMEngine`

Elastic Network Contact Model for vibrational entropy.

### `flexaidds.GAOptimizer`

Automated GA hyperparameter tuning.

### `flexaidds.load_results()`

Load and parse docking results from a directory.

## ML Rescoring

### `flexaidds.VoronoiGraphExtractor`

Extract contact graphs from docking results.

### `flexaidds.ShannonProfileExtractor`

Extract Shannon entropy trajectories from GA logs.

### `flexaidds.FeatureBuilder`

Build ML feature vectors from docking components.

### `flexaidds.MLRescorer`

Apply user-provided ML models for rescoring.
