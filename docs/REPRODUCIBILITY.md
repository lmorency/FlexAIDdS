# Reproducibility Policy

This document defines what must exist before a benchmark or scientific performance claim is treated as **repository-reproducible**.

## Levels of claim maturity

### 1. Replayable from repository artifacts

A claim reaches this level only when all of the following are present:

- dataset provenance or acquisition script
- checksums or immutable identifiers
- preprocessing steps
- exact command lines
- fixed seeds where applicable
- expected outputs and metric calculation scripts
- recorded git SHA and environment details

### 2. Preliminary

A claim is preliminary if it appears in documentation but is not yet backed by a replayable bundle in the repository.

Preliminary claims must be clearly labeled as such.

### 3. External / published

A claim may also be labeled external or published when it is validated in a peer-reviewed publication or equivalent external artifact.

## Required benchmark bundle layout

Every replayable benchmark should include a bundle under `benchmarks/` with at least:

- `README.md`
- `manifest.yaml`
- `download.sh` or equivalent acquisition instructions
- `run.sh` or equivalent execution script
- `expected/` outputs or metric snapshots
- `environment.txt` or machine metadata template

## Minimum manifest fields

Each benchmark manifest should describe:

- benchmark name
- dataset source
- dataset version or immutable identifier
- preprocessing steps
- executable and config used
- seed(s)
- output artifact paths
- metric definitions
- known limitations

## What must not happen

- no benchmark table should imply full reproducibility if the corresponding bundle is missing
- no metric should be called final if the replay scripts do not exist
- no mixed reporting of exploratory and release-grade numbers without explicit labels

## Immediate repository direction

The first reproducibility target for Core 1.0 should be a small smoke benchmark bundle that can run in CI, followed by larger manually triggered bundles for full dataset evaluation.
