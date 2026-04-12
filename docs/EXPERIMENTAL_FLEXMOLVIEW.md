# Experimental FlexMolView prototype

This document defines a deliberately small, non-supported prototype surface for a future molecular viewer around FlexAIDdS.

## Why multiple languages

The viewer problem splits naturally into different concerns:

- **Python** for orchestration, analysis, scripting, and rapid iteration.
- **Swift** for an Apple-native shell and concurrency-safe UI/state management.
- **TypeScript** for browser-facing dashboards and remote visualization surfaces.
- **Native compute/render backends** for hot-path geometry, picking, and drawing.

## Important constraint

A pure CPython rewrite does **not** remove the GIL by itself. The Python prototype exists to stabilize the object model and selection grammar, not to claim that Python alone is the final high-performance renderer.

## Prototype contents

### Python
- `python/flexaidds/flexmolview.py`
- `python/tests/test_flexmolview.py`

This is the most functional prototype today. It supports:
- PDB-backed object loading through existing FlexAIDdS I/O
- minimal selection grammar (`all`, `polymer`, `ligand`, `solvent`, `chain`, `resi`, `resn`, `name`, `elem`, `and`, `or`, `not`)
- representation visibility state (`show`, `hide`, `as_representation`)
- simple geometry summaries and distance calculations

### Swift
- `swift/FlexMolViewPrototype/`

Standalone package proving the same data-model idea can live in a native Apple surface without touching the supported Swift package.

### TypeScript
- `typescript/flexmolview-prototype/`

Standalone package proving the same PDB/selection nucleus can be exposed to a browser-facing shell.

## Non-goals in this prototype

- GPU rendering
- full PyMOL command compatibility
- session format parity
- movie support
- plugin ecosystem
- docking thermodynamics overlays in the UI

## Recommended next step

Keep Python as the command-and-analysis layer, then migrate the hot path to native code while letting Swift and TypeScript remain thin shells over the same scene model.
