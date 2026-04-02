# Product Definition

## Supported product for 1.0

**FlexAIDdS Core 1.0** is the only supported product surface for production and benchmark-facing use.

Supported components:

- `FlexAIDdS` command-line executable
- `FlexAID` legacy-compatible command-line executable
- `tENCoM` command-line executable
- JSON configuration parser and documented configuration schema
- `flexaidds` Python package
- Benchmark runner and reproducibility artifacts under `benchmarks/`
- Repository documentation required to install, validate, and reproduce supported workflows

## Explicitly out of 1.0 support scope

The following surfaces may remain in the repository, but they are **not part of the supported 1.0 contract** unless and until they are explicitly promoted here:

- Swift packages and Apple-device orchestration
- TypeScript/PWA dashboards and browser-facing viewers
- Bonhomme Fleet and iCloud-based distributed execution
- NATURaL and any co-translational / co-transcriptional workflows
- Experimental backend permutations not covered by required CI
- Any benchmark claim not backed by a replayable bundle in `benchmarks/`

## Release gate for supported status

A capability is considered supported only if all of the following are true:

1. It is documented in `docs/VALIDATED_CAPABILITIES.md`.
2. It is covered by required CI.
3. It has an installation path documented in the repository.
4. It has at least one automated test or reproducibility artifact.
5. It is not listed in `docs/EXPERIMENTAL_CAPABILITIES.md`.

## Why this file exists

FlexAIDdS has grown into a broad research platform. This file exists to separate:

- what is **shipping-grade and supportable now**
- what is **valuable but still experimental**

That separation is necessary for installation trust, release discipline, and scientific credibility.
