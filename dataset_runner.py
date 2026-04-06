#!/usr/bin/env python3
"""Legacy shim — use ``python -m flexaidds.dataset_runner`` instead."""

import sys

from flexaidds.dataset_runner.cli import main

if __name__ == "__main__":
    sys.exit(main())
