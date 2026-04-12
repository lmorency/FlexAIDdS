# FlexMolViewPrototype (Swift)

Experimental standalone Swift package for a future Apple-native molecular viewer shell.

Current scope:
- minimal PDB parsing
- tiny selection subset (`all`, `polymer`, `ligand`, `chain`, `resn`, `resi`, `name`)
- intentionally thin model layer

Non-goals for this prototype:
- GPU rendering
- PyMOL feature parity
- full session model
- docking thermodynamics UI

Build:

```bash
cd swift/FlexMolViewPrototype
swift test
```
