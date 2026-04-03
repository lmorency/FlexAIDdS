# Diversity Monitoring

## Entropy Collapse Detection

Premature convergence occurs when the population loses genetic diversity too early. FlexAIDΔS monitors the Shannon entropy of allele frequency distributions across all gene dimensions.

## How It Works

1. Every `diversity_check_interval` generations, compute normalized Shannon entropy per gene
2. If mean allele entropy drops below `diversity_collapse_threshold`, trigger catastrophic mutation
3. The bottom `catastrophic_mutation_fraction` of the population (by fitness) is re-randomized

## Configuration

```json
{
  "ga": {
    "diversity_monitoring": true,
    "diversity_check_interval": 10,
    "diversity_collapse_threshold": 0.3,
    "catastrophic_mutation_fraction": 0.2
  }
}
```
