# Configuration

FlexAIDΔS uses a JSON configuration file with sensible defaults. Override only what you need.

## Scoring

```json
{
  "scoring": {
    "function": "VCT",
    "hbond_enabled": false,
    "hbond_optimal_distance": 2.8,
    "hbond_optimal_angle": 180.0,
    "hbond_sigma_distance": 0.4,
    "hbond_sigma_angle": 30.0,
    "hbond_weight": -2.5,
    "hbond_salt_bridge_weight": -5.0,
    "gist_enabled": false,
    "gist_dx_file": "",
    "gist_weight": 1.0
  }
}
```

## Genetic Algorithm

```json
{
  "ga": {
    "num_chromosomes": 1000,
    "num_generations": 500,
    "crossover_rate": 0.8,
    "mutation_rate": 0.03,
    "fitness_model": "SMFREE",
    "entropy_weight": 0.5,
    "diversity_monitoring": false,
    "diversity_check_interval": 10,
    "diversity_collapse_threshold": 0.3,
    "catastrophic_mutation_fraction": 0.2
  }
}
```

## Distributed Computing

```json
{
  "distributed": {
    "backend": "thread"
  }
}
```

Valid backends: `"thread"` (default), `"mpi"`.
