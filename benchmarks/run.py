"""Backward-compatibility shim. Real code lives in flexaidds.dataset_runner.cli."""

import sys

from flexaidds.dataset_runner.cli import main

if __name__ == "__main__":
    sys.exit(main())
