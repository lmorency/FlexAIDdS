#!/usr/bin/env bash
# prune_stale_branches.sh — Delete all merged remote branches from FlexAIDdS
# Run from inside the FlexAIDdS repo clone on your Mac.
# Usage: chmod +x prune_stale_branches.sh && ./prune_stale_branches.sh

set -euo pipefail

BRANCHES=(
  "claude/claude-md-mmkf3751rp6l5b4n-sWyvM"
  "claude/continue-refine-implementation-LMtiE"
  "claude/fix-breaking-bug-CrO1T"
  "claude/fix-ci-build-errors-FvOtk"
  "claude/fix-compile-errors-y1yxt"
  "claude/fix-cross-platform-build-XIbZA"
  "claude/implement-todo-item-AJIMS"
  "claude/implement-todo-item-XFpes"
  "claude/implement-todo-item-XMkoY"
  "claude/implement-todo-item-Y9Vnq"
  "claude/implement-todo-item-Yq24u"
  "claude/implement-todo-item-ZWoRs"
  "claude/implement-todo-item-aBbOD"
  "claude/implement-todo-item-eQMWY"
  "claude/implement-todo-item-jVW4q"
  "claude/implement-todo-item-jak9Z"
  "claude/implement-todo-item-mwLyW"
  "claude/implement-todo-item-tdCxx"
  "claude/merge-branches-build-test-tXswo"
  "claude/merge-flexaids-v1.5-patches-bTUmk"
  "claude/pull-and-test-WJR2e"
  "claude/v1.5-test-bTUmk"
  "claude/write-implementation-MglRZ"
  "feature/full-thermodynamic-accel-v14"
  "feature/implement-thermodynamic-refactoring"
  "feature/native-cavity-detection-task13"
  "feature/phase1-phase2-production-grade"
  "feature/statmech-integration"
  "windows-ffs"
)

echo "Deleting ${#BRANCHES[@]} stale remote branches..."
echo ""

deleted=0
failed=0

for branch in "${BRANCHES[@]}"; do
  printf "  %-55s " "$branch"
  if git push origin --delete "$branch" 2>/dev/null; then
    echo "OK"
    ((deleted++))
  else
    echo "FAILED"
    ((failed++))
  fi
done

echo ""
echo "Done: $deleted deleted, $failed failed."

# Clean up local tracking refs
git remote prune origin
echo "Local tracking refs pruned."
