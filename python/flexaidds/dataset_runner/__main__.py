"""Allow ``python -m flexaidds.dataset_runner``."""

import sys

from .cli import main

sys.exit(main())
