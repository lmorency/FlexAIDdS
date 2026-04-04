# flexmolview-prototype (TypeScript)

Experimental browser-facing prototype for a future FlexMolView shell.

Current scope:
- PDB line parsing into a small atom model
- tiny selection subset (`all`, `polymer`, `ligand`, `chain`, `resn`, `resi`, `name`)
- no rendering stack yet

Build check:

```bash
cd typescript/flexmolview-prototype
npm install
npm run check
```

This package is intentionally standalone so it does not perturb the supported Core 1.0 surface.
