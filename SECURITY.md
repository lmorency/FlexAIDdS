# Security Policy

## Scope

This policy applies to the **supported 1.0 product surface** defined in `PRODUCT.md`.

Security reports involving unsupported experimental surfaces are still welcome, but remediation priority is determined by whether the affected code is part of the supported release contract.

## Supported branches

| Branch | Status |
|:--|:--|
| `master` | Active development |
| latest tagged release | Security fixes expected |

Until the first formal 1.0 release is cut, `master` is the active hardening branch.

## Reporting a vulnerability

Please do **not** open a public issue for suspected vulnerabilities involving memory safety, secret leakage, supply-chain risk, or arbitrary code execution.

Report privately to the maintainer through GitHub security reporting if available, or through the primary repository contact listed in the project metadata.

When reporting, include:

- affected commit SHA or branch
- impacted file(s) and function(s)
- reproduction steps
- expected impact
- proof-of-concept input if available
- whether the issue affects supported or experimental surfaces

## Response goals

Target response goals for supported 1.0 surfaces:

- initial triage: within 7 days
- severity assignment: within 14 days
- fix or mitigation plan for confirmed high-severity issues: as soon as possible

These are goals, not guarantees.

## Security baseline for supported surfaces

The supported product should converge toward the following baseline:

- zero known critical vulnerabilities
- zero known high-severity memory-safety vulnerabilities
- required CI coverage for supported build matrix
- static analysis and dependency scanning in CI
- reproducible release artifacts
- documented third-party licensing

## Current hardening priorities

1. Close the findings tracked in `SECURITY_AUDIT_BUFFER_OVERFLOW.md`.
2. Add automated regression tests for previously identified unsafe parsing patterns.
3. Add CodeQL and dependency update automation.
4. Promote sanitizer and fuzz coverage for supported parsers and input handling.

## Out of scope

The following are not considered supported attack surfaces until explicitly promoted:

- prototype UI layers
- device-fleet orchestration layers
- research-only execution paths
- unpublished benchmark claims
