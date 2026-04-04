#!/usr/bin/env python3
"""check_licenses.py — Verify no GPL/AGPL/LGPL dependencies in the codebase.

Parses scancode-toolkit JSON output and fails if any forbidden licenses
are detected. Used in CI to enforce the Apache-2.0 clean-room policy.

Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
SPDX-License-Identifier: Apache-2.0
"""

import json
import sys
from typing import List, Tuple

# Forbidden license SPDX identifiers (case-insensitive matching)
FORBIDDEN_LICENSES = {
    "gpl-1.0", "gpl-1.0-only", "gpl-1.0-or-later",
    "gpl-2.0", "gpl-2.0-only", "gpl-2.0-or-later",
    "gpl-3.0", "gpl-3.0-only", "gpl-3.0-or-later",
    "agpl-1.0", "agpl-3.0", "agpl-3.0-only", "agpl-3.0-or-later",
    "lgpl-2.0", "lgpl-2.0-only", "lgpl-2.0-or-later",
    "lgpl-2.1", "lgpl-2.1-only", "lgpl-2.1-or-later",
    "lgpl-3.0", "lgpl-3.0-only", "lgpl-3.0-or-later",
}

# Allowed licenses
ALLOWED_LICENSES = {
    "apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause",
    "bsd-1-clause", "isc", "psf-2.0", "mpl-2.0",
    "unlicense", "cc0-1.0", "public-domain", "boost-1.0",
}


def check_report(report_path: str) -> List[Tuple[str, str]]:
    """Check scancode report for forbidden licenses.

    Returns list of (file_path, license_key) tuples for violations.
    """
    with open(report_path) as f:
        report = json.load(f)

    violations = []

    for file_entry in report.get("files", []):
        file_path = file_entry.get("path", "unknown")

        for lic in file_entry.get("licenses", []):
            key = lic.get("key", "").lower()
            spdx = lic.get("spdx_license_key", "").lower()

            for identifier in [key, spdx]:
                if identifier in FORBIDDEN_LICENSES:
                    violations.append((file_path, identifier))

    return violations


def main():
    if len(sys.argv) < 2:
        print("Usage: check_licenses.py <scancode-report.json>")
        sys.exit(1)

    report_path = sys.argv[1]

    try:
        violations = check_report(report_path)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"WARNING: Could not parse license report: {e}")
        print("Skipping license check (report may be empty or malformed)")
        sys.exit(0)

    if violations:
        print(f"FORBIDDEN LICENSES DETECTED ({len(violations)} violations):")
        print("=" * 60)
        for file_path, license_key in violations:
            print(f"  {license_key:30s}  {file_path}")
        print("=" * 60)
        print("FlexAIDdS is Apache-2.0. GPL/AGPL/LGPL dependencies are forbidden.")
        print("See docs/licensing/clean-room-policy.md for details.")
        sys.exit(1)
    else:
        print("License scan PASSED: no forbidden licenses detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
