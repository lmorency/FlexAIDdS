#!/usr/bin/env bash
# mirror_to_gdrive.sh — Async RAID-1 mirror from iCloud to Google Drive
#
# Syncs benchmark_data/, results/, and logs/ from iCloud (primary) to
# Google Drive (mirror). Skips build/ (too volatile, rebuild is cheaper).
#
# Usage:
#   ./benchmarks/m3pro/mirror_to_gdrive.sh          # foreground
#   nohup ./benchmarks/m3pro/mirror_to_gdrive.sh &  # background
#
# Exit codes:
#   0 — all syncs succeeded
#   1 — partial failure (some dirs failed, logged)
#   2 — total failure (env not set or both paths missing)
#
# Apache-2.0 (c) 2026 NRGlab, Universite de Montreal

set -uo pipefail

# ─── Load environment ────────────────────────────────────────────────────────

ENV_FILE="$HOME/.flexaidds_env"
if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
else
    echo "[ERROR] Environment file not found: $ENV_FILE" >&2
    echo "        Run setup_cloud_storage.sh first." >&2
    exit 2
fi

# Validate required vars
if [[ -z "${FLEXAIDDS_ICLOUD:-}" ]] || [[ -z "${FLEXAIDDS_GDRIVE:-}" ]]; then
    echo "[ERROR] FLEXAIDDS_ICLOUD or FLEXAIDDS_GDRIVE not set." >&2
    exit 2
fi

if [[ ! -d "$FLEXAIDDS_ICLOUD" ]]; then
    echo "[ERROR] iCloud path does not exist: $FLEXAIDDS_ICLOUD" >&2
    exit 2
fi

if [[ ! -d "$FLEXAIDDS_GDRIVE" ]]; then
    echo "[ERROR] Google Drive path does not exist: $FLEXAIDDS_GDRIVE" >&2
    exit 2
fi

# ─── Logging ─────────────────────────────────────────────────────────────────

TIMESTAMP="$(date +%Y%m%dT%H%M%S)"
LOG_DIR="$FLEXAIDDS_ICLOUD/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/mirror_${TIMESTAMP}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }

log "Mirror started: iCloud -> Google Drive"
log "Source:      $FLEXAIDDS_ICLOUD"
log "Destination: $FLEXAIDDS_GDRIVE"

# ─── Sync directories ───────────────────────────────────────────────────────

DIRS_TO_SYNC=("benchmark_data" "results" "logs")
FAILURES=0
TOTAL_BYTES=0

for dir in "${DIRS_TO_SYNC[@]}"; do
    SRC="$FLEXAIDDS_ICLOUD/$dir/"
    DST="$FLEXAIDDS_GDRIVE/$dir/"

    if [[ ! -d "$SRC" ]]; then
        log "SKIP: $dir/ (source does not exist)"
        continue
    fi

    mkdir -p "$DST"

    log "SYNC: $dir/ ..."
    SYNC_START=$(date +%s)

    # rsync with:
    #   -a  archive mode (preserves permissions, timestamps)
    #   -v  verbose (for logging)
    #   -z  compress during transfer (helps over cloud FS)
    #   --delete  remove files from dst that don't exist in src
    #   --stats   show transfer statistics
    if rsync_output=$(rsync -avz --delete --stats "$SRC" "$DST" 2>&1); then
        SYNC_END=$(date +%s)
        DURATION=$((SYNC_END - SYNC_START))

        # Extract bytes transferred from rsync stats
        bytes=$(echo "$rsync_output" | grep -o 'Total transferred file size: [0-9,]*' | grep -o '[0-9,]*' | tr -d ',' || echo "0")
        TOTAL_BYTES=$((TOTAL_BYTES + ${bytes:-0}))

        log "  OK: $dir/ synced in ${DURATION}s (${bytes:-0} bytes)"
    else
        FAILURES=$((FAILURES + 1))
        log "  FAIL: $dir/ rsync returned $?"
        echo "$rsync_output" >> "$LOG_FILE"
    fi
done

# ─── Post-sync integrity verification ────────────────────────────────────────

MANIFEST="$LOG_DIR/sync_manifest_${TIMESTAMP}.json"
VERIFY_FAILURES=0

log ""
log "Verifying sync integrity..."

{
    echo "{"
    echo "  \"timestamp\": \"${TIMESTAMP}\","
    echo "  \"directories\": {"
    FIRST_DIR=true
    for dir in "${DIRS_TO_SYNC[@]}"; do
        SRC="$FLEXAIDDS_ICLOUD/$dir/"
        DST="$FLEXAIDDS_GDRIVE/$dir/"
        [[ ! -d "$SRC" ]] && continue

        SRC_COUNT=$(find "$SRC" -type f 2>/dev/null | wc -l | tr -d ' ')
        DST_COUNT=$(find "$DST" -type f 2>/dev/null | wc -l | tr -d ' ')

        if [[ "$SRC_COUNT" -ne "$DST_COUNT" ]]; then
            log "  VERIFY FAIL: $dir/ — source=$SRC_COUNT files, mirror=$DST_COUNT files"
            VERIFY_FAILURES=$((VERIFY_FAILURES + 1))
            MATCH="false"
        else
            log "  VERIFY OK: $dir/ — $SRC_COUNT files match"
            MATCH="true"
        fi

        if [[ "$FIRST_DIR" == true ]]; then
            FIRST_DIR=false
        else
            echo ","
        fi
        printf '    "%s": {"source_files": %d, "mirror_files": %d, "match": %s}' \
            "$dir" "$SRC_COUNT" "$DST_COUNT" "$MATCH"
    done
    echo ""
    echo "  },"
    echo "  \"verify_failures\": $VERIFY_FAILURES"
    echo "}"
} > "$MANIFEST"

log "Manifest: $MANIFEST"

if [[ $VERIFY_FAILURES -gt 0 ]]; then
    log "WARNING: $VERIFY_FAILURES dir(s) have mismatched file counts"
    FAILURES=$((FAILURES + VERIFY_FAILURES))
fi

# ─── Summary ─────────────────────────────────────────────────────────────────

log ""
log "Mirror complete: ${#DIRS_TO_SYNC[@]} dirs attempted, $FAILURES failures"
log "Total bytes synced: $TOTAL_BYTES"
log "Log: $LOG_FILE"

if [[ $FAILURES -eq ${#DIRS_TO_SYNC[@]} ]]; then
    log "STATUS: TOTAL FAILURE"
    exit 2
elif [[ $FAILURES -gt 0 ]]; then
    log "STATUS: PARTIAL FAILURE ($FAILURES/${#DIRS_TO_SYNC[@]} dirs failed)"
    exit 1
else
    log "STATUS: SUCCESS"
    exit 0
fi
