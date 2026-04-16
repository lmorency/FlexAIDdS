# FreeNRG Integration

This branch adds the FreeNRG unified free energy framework that bridges
FlexAID-deltaS entropy-aware docking with NRGRank virtual screening.

## FreeNRG Repository

The FreeNRG Python package is maintained at:
https://github.com/LeBonhommePharma/FreeNRG

## What FreeNRG Provides

FreeNRG is a Python package that ports key FlexAID-deltaS thermodynamic
components into a reusable library:

- **StatMechEngine** - Python port of `statmech.cpp` (partition functions, WHAM, TI)
- **ShannonThermoStack** - Python port of `ShannonThermoStack.cpp` (Shannon entropy)
- **TorsionalENM** - Python port of `tencm.cpp` (backbone flexibility)
- **CFScorer** - Python port of `cffunction.cpp` (complementarity scoring)
- **FlexAIDBridge** - Subprocess wrapper for the FlexAID C++ binary
- **NRGRankBridge** - Direct Python integration with NRGRank

## Usage with FlexAID

```python
from freenrg.pipeline import FreeNRGPipeline, FreeNRGConfig, DockingMode

config = FreeNRGConfig(
    mode=DockingMode.FLEXAID,
    flexaid_binary="/path/to/FlexAID",
    receptor_pdb="receptor.inp.pdb",
    ligand_inp="ligand.inp",
    binding_site="cleft.pdb",
)

pipeline = FreeNRGPipeline()
result = pipeline.run(config)
print(f"deltaG = {result.delta_G:.2f} kcal/mol")
```

## Install

```bash
pip install freenrg
```
