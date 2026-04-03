# GIST Desolvation

## Overview

Grid Inhomogeneous Solvation Theory (GIST) provides voxel-level thermodynamic properties of water molecules in the binding cavity. FlexAIDΔS reads pre-computed `.dx` grids and applies trilinear interpolation.

## Integration

$$E_{GIST} = w_{GIST} \sum_i \Delta G_{solv}(x_i, y_i, z_i)$$

Where the sum runs over all scored atoms and $\Delta G_{solv}$ is interpolated from the GIST grid.

## Usage

1. Run MD simulation and compute GIST grids using tools like `cpptraj`
2. Configure FlexAIDΔS:

```json
{
  "scoring": {
    "gist_enabled": true,
    "gist_dx_file": "/path/to/gist_dG.dx",
    "gist_weight": 1.0
  }
}
```
