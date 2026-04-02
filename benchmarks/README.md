# Benchmarks

This directory is the repository home for benchmark reproducibility bundles.

A benchmark should not be treated as repository-reproducible until it has a bundle here.

## Expected bundle structure

Each benchmark should provide at minimum:

- `README.md`
- `manifest.yaml`
- acquisition instructions or download script
- execution instructions or run script
- expected output layout
- metric definition notes

## Intended split

- `smoke/` for small, fast benchmark exercises suitable for CI or frequent validation
- dataset-specific directories for larger manually triggered or scheduled runs

## Current status

This directory is scaffolding for Core 1.0 reproducibility discipline.
