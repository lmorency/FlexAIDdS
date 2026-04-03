# Smoke validation bundle

This directory is the first concrete reproducibility bundle for the Core 1.0 hardening track.

It is intentionally small. The purpose is **not** to prove scientific superiority on a large external dataset. The purpose is to provide a replayable, low-cost validation path that checks the supported build-and-test surface in a deterministic way.

## What this bundle validates

- supported C++ configure/build path without optional GPU backends
- supported C++ test execution path
- supported Python smoke-test path for the package surface

## Intended use

Use this bundle when you need a quick answer to the question:

> can a technically literate user clone the repo, run a documented command sequence, and observe the expected validation behavior on the supported Core 1.0 surface?

## Commands

From repository root:

```bash
bash benchmarks/smoke/run.sh
```

## Expected behavior

- a local build directory is created
- the core test suite runs through `ctest`
- the Python smoke test runs through `pytest`
- the script exits non-zero on failure

## Notes

This is a **repository smoke-validation bundle**, not a scientific benchmark dataset release.
Scientific benchmark claims still require their own dataset-specific bundles.
