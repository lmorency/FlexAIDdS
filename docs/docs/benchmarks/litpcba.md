# LIT-PCBA Benchmark

LIT-PCBA provides unbiased virtual screening evaluation using 15 targets with confirmed actives and inactives from PubChem dose-response assays.

## Running

```bash
export LITPCBA_DATA=/path/to/LIT-PCBA
python tests/benchmarks/litpcba/run_litpcba.py --results /path/to/results
```

## Metrics

- **EF1%**: Enrichment factor at 1% of the ranked list
- **AUROC**: Area under the ROC curve
- **BEDROC**: Boltzmann-Enhanced Discrimination of ROC
