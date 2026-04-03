# Angular-Dependent H-Bond Potential

## Overview

The H-bond potential uses a Gaussian bell function that accounts for both donor-acceptor distance and D-H...A angle geometry.

## Mathematical Formulation

$$E_{hb} = w \cdot \exp\left(-\frac{(d - d_0)^2}{2\sigma_d^2}\right) \cdot \exp\left(-\frac{(\theta - \theta_0)^2}{2\sigma_\theta^2}\right)$$

Where:
- $d$ = donor-acceptor distance (Å)
- $d_0$ = optimal distance (default 2.8 Å)
- $\theta$ = D-H...A angle (degrees)
- $\theta_0$ = optimal angle (default 180°)
- $\sigma_d$, $\sigma_\theta$ = Gaussian widths

## Salt Bridge Detection

Salt bridges are automatically detected when one atom is anionic (charge bin < -0.25) and the other cationic (charge bin ≥ 0.25). A stronger weight is applied.

## Configuration

```json
{
  "scoring": {
    "hbond_enabled": true,
    "hbond_weight": -2.5,
    "hbond_salt_bridge_weight": -5.0
  }
}
```
