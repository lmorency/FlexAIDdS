# Known Limitations

This file documents limitations that matter for installation trust, scientific interpretation, and release planning.

## Product-scope limitations

- The repository contains more code than the supported 1.0 surface. Not every visible feature is part of the supported release contract.
- Experimental surfaces may compile or exist in-tree without being considered support-guaranteed.

## Reproducibility limitations

- Benchmark tables and scientific claims should not be assumed replayable unless a corresponding bundle exists under `benchmarks/`.
- Some documented results may still be preliminary or manuscript-bound rather than repository-reproducible.

## Backend limitations

- Backend breadth does not imply release-grade parity.
- Scalar CPU, OpenMP, and core CPU workflows are the primary supported base.
- Backend-specific accelerators may remain optional or experimental until covered by release validation.

## Documentation limitations

- Historical README and site language may reflect the full research vision rather than the narrow supported Core 1.0 surface.
- Public-facing documentation should be interpreted together with `PRODUCT.md`, `docs/SUPPORT_MATRIX.md`, and `docs/EXPERIMENTAL_CAPABILITIES.md`.

## Security limitations

- Legacy C and C++ parsing surfaces require continued hardening and regression testing.
- Presence of an audit or policy document does not itself imply closure; fixes and automation are the real closure criteria.

## Scientific interpretation limitations

- Entropy-aware docking can be useful without implying universal superiority on every target class.
- Claims should be evaluated against the specific benchmark bundle, dataset scope, preprocessing rules, and metric definition used.
