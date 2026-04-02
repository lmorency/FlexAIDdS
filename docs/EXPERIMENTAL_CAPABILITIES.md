# Experimental Capabilities

This file lists visible repository surfaces that should be treated as experimental until they are explicitly promoted into `docs/VALIDATED_CAPABILITIES.md`.

Experimental means one or more of the following:

- incomplete CI coverage
- unstable API or UX contract
- incomplete installation documentation
- incomplete reproducibility artifacts
- performance path exists but is not release-gated

## Experimental surfaces

At the current stage, the following should be treated as experimental:

- Swift packages and Apple-platform integration layers
- TypeScript, PWA, dashboard, and browser-facing viewers
- Bonhomme Fleet and iCloud-driven distributed execution
- NATURaL and related co-translational or co-transcriptional workflows
- backend-specific acceleration paths not required by the Core 1.0 support matrix
- benchmark claims not yet backed by a repository reproducibility bundle

## Promotion rule

A capability should remain experimental until it has:

1. documentation sufficient for external use
2. automated validation or release validation coverage
3. an explicit place in the support matrix
4. an unambiguous ownership in the product boundary defined by `PRODUCT.md`
