# Cross-Docking Validation

Cross-docking tests docking a ligand into a receptor conformation crystallized with a different ligand, testing the engine's ability to handle induced fit.

## Running

```bash
python tests/benchmarks/crossdock/run_crossdock.py \
  --results /path/to/results \
  --pairs crossdock_pairs.csv
```

## Metrics

- **Success rate**: Fraction of pairs with RMSD ≤ 2Å
- **Chi-angle deviation**: Side-chain rotamer accuracy
