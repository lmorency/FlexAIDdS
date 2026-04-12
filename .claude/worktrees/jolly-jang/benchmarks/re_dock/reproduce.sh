#!/bin/bash
# RE-DOCK benchmark reproduction script
# Usage: ./reproduce.sh [tier]
#   tier 0: 1STP single-system validation (default)
#   tier 1: 5-system validation set
#   tier 2: Astex diverse 85

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TIER=${1:-0}
OUTPUT="${PROJECT_ROOT}/results/re-dock"

echo "=== RE-DOCK Benchmark ==="
echo "Tier: $TIER"
echo "Output: $OUTPUT"
echo ""

cd "$PROJECT_ROOT"
python -m benchmarks.re_dock --tier "$TIER" --output "$OUTPUT"

echo ""
echo "=== Complete ==="
