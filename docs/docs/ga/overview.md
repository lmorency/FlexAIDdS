# Genetic Algorithm

The FlexAIDΔS genetic algorithm (`gaboom`) explores the conformational and orientational search space using entropy-driven fitness evaluation.

## Fitness Models

- **SMFREE**: StatMech free-energy-weighted fitness with niche sharing (recommended)
- **PSHARE**: Rank-based fitness with phenotypic niche sharing
- **LINEAR**: Simple rank-based fitness

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_chromosomes` | 1000 | Population size |
| `num_generations` | 500 | Maximum generations |
| `crossover_rate` | 0.8 | Crossover probability |
| `mutation_rate` | 0.03 | Mutation probability |
| `entropy_weight` | 0.5 | Boltzmann vs rank blending |
