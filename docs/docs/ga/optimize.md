# Hyperparameter Tuning

## Overview

The `GAOptimizer` class uses `scipy.optimize.differential_evolution` to automatically find optimal GA parameters for a specific receptor-ligand system.

## Usage

```python
from flexaidds import GAOptimizer

optimizer = GAOptimizer(
    receptor="receptor.pdb",
    ligand="ligand.mol2",
    cleft="cleft.grid",
)

result = optimizer.optimize(n_iterations=50)
print(f"Best params: {result.best_params}")
print(f"Best score: {result.best_score}")
```

## Parameter Space

| Parameter | Range | Description |
|-----------|-------|-------------|
| `num_chromosomes` | 100–5000 | Population size |
| `num_generations` | 50–2000 | Max generations |
| `crossover_rate` | 0.5–0.99 | Crossover probability |
| `mutation_rate` | 0.005–0.15 | Mutation probability |
| `sharing_alpha` | 0.5–2.0 | Niche sharing exponent |
| `entropy_weight` | 0.0–1.0 | Boltzmann blending |
