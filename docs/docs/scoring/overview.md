# Scoring Functions

FlexAIDΔS uses a multi-component scoring function:

$$E_{total} = E_{CF} + E_{wall} + E_{SAS} + E_{elec} + E_{hbond} + E_{GIST}$$

- **E_CF**: Voronoi contact function (surface complementarity)
- **E_wall**: Steric clash penalty (repulsive wall potential)
- **E_SAS**: Solvent accessible surface contribution
- **E_elec**: Coulomb electrostatics with RESP charges
- **E_hbond**: Angular-dependent hydrogen bond potential
- **E_GIST**: Grid Inhomogeneous Solvation Theory desolvation

See [H-Bond Potential](hbond.md) and [GIST Desolvation](gist.md) for details.
