#!/usr/bin/env bash
# run_benchmarks.sh — Master benchmark runner for MacBook Pro M3 Pro 18GB
#
# Runs all benchmark tiers sequentially with memory-aware worker counts:
#   Phase 1: C++ kernel benchmarks (dispatch, vcfbatch, tencom)
#   Phase 2: Tier-1 dataset benchmark (CASF-2016, 5 targets, ~5 min)
#   Phase 3: Tier-2 dataset benchmarks (all 7 datasets, sequential, hours)
#   Phase 4: Final report consolidation + blocking mirror sync
#
# Each phase triggers an async rsync mirror to Google Drive.
#
# Usage:
#   ./benchmarks/m3pro/run_benchmarks.sh                # all phases
#   ./benchmarks/m3pro/run_benchmarks.sh --kernels-only  # phase 1 only
#   ./benchmarks/m3pro/run_benchmarks.sh --tier1-only    # phase 2 only
#   ./benchmarks/m3pro/run_benchmarks.sh --tier2-only    # phase 3 only
#
# Apache-2.0 (c) 2026 NRGlab, Universite de Montreal

set -euo pipefail

# ─── Colors ──────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { printf "${CYAN}[INFO]${NC}  %s\n" "$*"; }
ok()    { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
phase() { printf "\n${BOLD}════════════════════════════════════════════════════════${NC}\n"; printf "${BOLD}  %s${NC}\n" "$*"; printf "${BOLD}════════════════════════════════════════════════════════${NC}\n\n"; }
die()   { printf "${RED}[ERROR]${NC} %s\n" "$*" >&2; exit 1; }

# ─── Load environment ────────────────────────────────────────────────────────

ENV_FILE="$HOME/.flexaidds_env"
if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
else
    die "Environment file not found: $ENV_FILE — run setup_cloud_storage.sh first"
fi

REPO="${FLEXAIDDS_REPO:?not set}"
BUILD="${FLEXAIDDS_BUILD:?not set}"
RESULTS="${FLEXAIDDS_RESULTS:?not set}"
LOGS="${FLEXAIDDS_LOGS:?not set}"
DATA="${FLEXAIDDS_BENCHMARK_DATA:?not set}"
BINARY="${FLEXAIDDS_BINARY:?not set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MIRROR_SCRIPT="$SCRIPT_DIR/mirror_to_gdrive.sh"

# ─── Parse arguments ─────────────────────────────────────────────────────────

RUN_KERNELS=true
RUN_TIER1=true
RUN_TIER2=true

for arg in "$@"; do
    case "$arg" in
        --kernels-only) RUN_TIER1=false; RUN_TIER2=false ;;
        --tier1-only)   RUN_KERNELS=false; RUN_TIER2=false ;;
        --tier2-only)   RUN_KERNELS=false; RUN_TIER1=false ;;
        --help|-h)
            echo "Usage: $0 [--kernels-only|--tier1-only|--tier2-only]"
            exit 0
            ;;
        *) die "Unknown argument: $arg" ;;
    esac
done

# ─── Validate build ─────────────────────────────────────────────────────────

if [[ ! -f "$BINARY" ]]; then
    die "FlexAID binary not found: $BINARY — run build_m3pro.sh first"
fi

# ─── Hardware detection ──────────────────────────────────────────────────────

TIMESTAMP="$(date +%Y%m%dT%H%M%S)"
MASTER_LOG="$LOGS/benchmark_run_${TIMESTAMP}.log"
mkdir -p "$LOGS"

{
    echo "FlexAIDdS Benchmark Run — $TIMESTAMP"
    echo "========================================"
    echo ""
    echo "Hardware Profile:"
    echo "  CPU:     $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
    echo "  Cores:   $(sysctl -n hw.ncpu) total ($(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo '?') P + $(sysctl -n hw.perflevel1.logicalcpu 2>/dev/null || echo '?') E)"
    echo "  RAM:     $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
    echo "  GPU:     $(system_profiler SPDisplaysDataType 2>/dev/null | grep 'Chipset Model' | sed 's/.*: //' || echo 'unknown')"
    echo "  OS:      $(sw_vers -productName 2>/dev/null || echo macOS) $(sw_vers -productVersion 2>/dev/null || echo '')"
    echo "  Storage: iCloud (primary) + Google Drive (mirror)"
    echo ""
    echo "Git SHA:   $(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    echo "Branch:    $(cd "$REPO" && git branch --show-current 2>/dev/null || echo 'unknown')"
    echo ""
} | tee "$MASTER_LOG"

GLOBAL_START=$(date +%s)

# ─── Thermal throttle detection ──────────────────────────────────────────────
# M3 Pro in 14" chassis throttles under sustained load. Log thermal state
# before and after each phase to detect if results are affected.

check_thermal() {
    local label="$1"
    local cpu_temp
    # pmset thermalPressureLevel: 0=nominal, 1=moderate, 2=heavy, 3=trapping, 4=sleeping
    local thermal_level
    thermal_level=$(pmset -g therm 2>/dev/null | grep -o 'CPU_Scheduler_Limit.*' | head -1 || echo "unavailable")
    local speed_limit
    speed_limit=$(pmset -g therm 2>/dev/null | grep -o 'CPU_Speed_Limit.*' | head -1 || echo "unavailable")

    # powermetrics requires sudo; fall back to pmset thermal data
    local pressure
    pressure=$(pmset -g therm 2>/dev/null | grep -i 'thermal' | head -1 || echo "unknown")

    echo "  Thermal [$label]: $pressure | $thermal_level | $speed_limit" | tee -a "$MASTER_LOG"

    # Check for throttling (speed limit < 100 means throttled)
    if echo "$speed_limit" | grep -qE '[0-9]+' 2>/dev/null; then
        local limit_val
        limit_val=$(echo "$speed_limit" | grep -o '[0-9]*' | head -1)
        if [[ -n "$limit_val" ]] && [[ "$limit_val" -lt 100 ]]; then
            warn "THERMAL THROTTLING DETECTED: CPU speed limited to ${limit_val}%"
            warn "Results from this phase may show degraded performance."
            echo "  WARNING: THROTTLED to ${limit_val}%" >> "$MASTER_LOG"
            return 1
        fi
    fi
    return 0
}

# ─── Memory pressure guard ───────────────────────────────────────────────────
# 18GB unified RAM shared with Metal GPU. Detect swap pressure before starting
# benchmarks and between phases.

check_memory_pressure() {
    local label="$1"
    local page_size
    page_size=$(pagesize 2>/dev/null || echo 16384)

    # vm_stat gives page counts; compute swap and compressed memory
    local vm_out
    vm_out=$(vm_stat 2>/dev/null || true)

    local swap_used_bytes=0
    if [[ -n "$vm_out" ]]; then
        local swapins swapouts
        swapins=$(echo "$vm_out" | awk '/Swapins/{gsub(/\./,"",$2); print $2}')
        swapouts=$(echo "$vm_out" | awk '/Swapouts/{gsub(/\./,"",$2); print $2}')
        local compressed
        compressed=$(echo "$vm_out" | awk '/stored in compressor/{gsub(/\./,"",$NF); print $NF}')

        # memory_pressure command (macOS) gives system/user pressure
        local pressure_level
        pressure_level=$(memory_pressure 2>/dev/null | grep -o 'System-wide.*level' | head -1 || echo "unknown")

        local free_pages
        free_pages=$(echo "$vm_out" | awk '/Pages free/{gsub(/\./,"",$NF); print $NF}')
        local free_mb=0
        if [[ -n "$free_pages" ]]; then
            free_mb=$(( (free_pages * page_size) / 1048576 ))
        fi

        echo "  Memory [$label]: free=${free_mb}MB, compressed=${compressed:-?} pages, swapins=${swapins:-0}, pressure=${pressure_level:-unknown}" | tee -a "$MASTER_LOG"

        # Warn if free memory is below 1GB (1024MB)
        if [[ "$free_mb" -lt 1024 ]]; then
            warn "LOW MEMORY: Only ${free_mb}MB free — benchmark timings may be unreliable"
            echo "  WARNING: LOW MEMORY ${free_mb}MB" >> "$MASTER_LOG"
            return 1
        fi

        # Warn if significant swap activity
        if [[ -n "$swapins" ]] && [[ "$swapins" -gt 1000 ]]; then
            warn "SWAP PRESSURE: ${swapins} swap-ins detected — results may be unreliable"
            echo "  WARNING: SWAP PRESSURE swapins=$swapins" >> "$MASTER_LOG"
            return 1
        fi
    else
        echo "  Memory [$label]: vm_stat unavailable" | tee -a "$MASTER_LOG"
    fi
    return 0
}

# ─── Pre-flight checks ──────────────────────────────────────────────────────

echo "Pre-flight environment checks:" | tee -a "$MASTER_LOG"
check_thermal "pre-flight"
check_memory_pressure "pre-flight"
echo "" | tee -a "$MASTER_LOG"

# ─── Helper: async mirror ───────────────────────────────────────────────────

trigger_mirror() {
    if [[ -x "$MIRROR_SCRIPT" ]]; then
        info "Triggering async mirror to Google Drive..."
        nohup "$MIRROR_SCRIPT" >> "$LOGS/mirror_bg_${TIMESTAMP}.log" 2>&1 &
        disown
    else
        warn "Mirror script not found or not executable: $MIRROR_SCRIPT"
    fi
}

# ─── Phase 1: C++ Kernel Benchmarks ─────────────────────────────────────────

if [[ "$RUN_KERNELS" == true ]]; then
    phase "Phase 1: C++ Kernel Benchmarks"
    mkdir -p "$RESULTS/kernels"

    KERNEL_REPORT="$RESULTS/kernels/report_${TIMESTAMP}.txt"
    : > "$KERNEL_REPORT"

    # --- benchmark_dispatch ---
    if [[ -f "$BUILD/benchmark_dispatch" ]]; then
        info "Running benchmark_dispatch (size=100000, reps=100)..."
        DISPATCH_OUT="$RESULTS/kernels/dispatch_${TIMESTAMP}.txt"
        "$BUILD/benchmark_dispatch" --size 100000 --reps 100 2>&1 | tee "$DISPATCH_OUT" | tee -a "$KERNEL_REPORT"
        ok "Dispatch benchmark complete"
    else
        warn "benchmark_dispatch not found — skipping"
    fi

    echo "" >> "$KERNEL_REPORT"

    # --- benchmark_vcfbatch ---
    if [[ -f "$BUILD/benchmark_vcfbatch" ]]; then
        info "Running benchmark_vcfbatch (pop=200, genes=20)..."
        VCFBATCH_OUT="$RESULTS/kernels/vcfbatch_${TIMESTAMP}.txt"
        "$BUILD/benchmark_vcfbatch" 200 20 2>&1 | tee "$VCFBATCH_OUT" | tee -a "$KERNEL_REPORT"
        ok "VoronoiCFBatch benchmark complete"
    else
        warn "benchmark_vcfbatch not found — skipping"
    fi

    echo "" >> "$KERNEL_REPORT"

    # --- benchmark_tencom ---
    if [[ -f "$BUILD/benchmark_tencom" ]]; then
        info "Running benchmark_tencom..."
        TENCOM_OUT="$RESULTS/kernels/tencom_${TIMESTAMP}.txt"
        "$BUILD/benchmark_tencom" 2>&1 | tee "$TENCOM_OUT" | tee -a "$KERNEL_REPORT"
        ok "tENCoM benchmark complete"
    else
        warn "benchmark_tencom not found — skipping"
    fi

    ok "Phase 1 complete — kernel results in $RESULTS/kernels/"
    echo "" >> "$MASTER_LOG"
    echo "Phase 1 (Kernels): COMPLETE" >> "$MASTER_LOG"
    cat "$KERNEL_REPORT" >> "$MASTER_LOG"

    check_thermal "post-phase1"
    check_memory_pressure "post-phase1"
    trigger_mirror
fi

# ─── Phase 2: Tier-1 Dataset Benchmark ──────────────────────────────────────

if [[ "$RUN_TIER1" == true ]]; then
    phase "Phase 2: Tier-1 Dataset Benchmark (CASF-2016, 5 targets)"
    mkdir -p "$RESULTS/tier1"

    # 4 workers: ~4.5 GB per worker from 18GB total (3GB OS + 4GB Metal + 2.5GB×4 workers ≈ 17GB)
    info "Running CASF-2016 tier-1 with 4 workers..."
    python -m benchmarks.run \
        --dataset casf2016 \
        --tier 1 \
        --workers 4 \
        --results-dir "$RESULTS/tier1" \
        --data-dir "$DATA" \
        --binary "$BINARY" \
        --report-prefix "$RESULTS/tier1/report_${TIMESTAMP}" \
        2>&1 | tee -a "$MASTER_LOG"

    TIER1_EXIT=$?
    if [[ $TIER1_EXIT -eq 0 ]]; then
        ok "Phase 2 complete — no regressions"
    elif [[ $TIER1_EXIT -eq 1 ]]; then
        warn "Phase 2 complete — REGRESSIONS DETECTED"
    else
        warn "Phase 2 failed with exit code $TIER1_EXIT"
    fi

    echo "Phase 2 (Tier-1): exit=$TIER1_EXIT" >> "$MASTER_LOG"
    check_thermal "post-phase2"
    check_memory_pressure "post-phase2"
    trigger_mirror
fi

# ─── Phase 3: Tier-2 Dataset Benchmarks ─────────────────────────────────────

if [[ "$RUN_TIER2" == true ]]; then
    phase "Phase 3: Tier-2 Dataset Benchmarks (7 datasets, sequential)"
    mkdir -p "$RESULTS/tier2"

    # Sequential execution to stay within 18GB RAM.
    # 2 workers per dataset: ~4.5GB per worker (3GB OS + 4GB Metal + 4.5GB×2 ≈ 16GB)
    DATASETS=(casf2016 itc187 dude37 muv lsd_docking erds_specificity psychopharm23)
    TIER2_REGRESSIONS=0

    for ds in "${DATASETS[@]}"; do
        info "Running $ds (tier-2, 2 workers, bootstrap=5000)..."
        DS_START=$(date +%s)

        python -m benchmarks.run \
            --dataset "$ds" \
            --tier 2 \
            --workers 2 \
            --bootstrap \
            --n-bootstrap 5000 \
            --results-dir "$RESULTS/tier2" \
            --data-dir "$DATA" \
            --binary "$BINARY" \
            --report-prefix "$RESULTS/tier2/${ds}_${TIMESTAMP}" \
            2>&1 | tee -a "$MASTER_LOG"

        DS_EXIT=$?
        DS_END=$(date +%s)
        DS_DURATION=$((DS_END - DS_START))

        if [[ $DS_EXIT -eq 0 ]]; then
            ok "$ds complete in ${DS_DURATION}s — no regressions"
        elif [[ $DS_EXIT -eq 1 ]]; then
            warn "$ds complete in ${DS_DURATION}s — REGRESSION DETECTED"
            TIER2_REGRESSIONS=$((TIER2_REGRESSIONS + 1))
        else
            warn "$ds failed (exit=$DS_EXIT) after ${DS_DURATION}s"
        fi

        echo "  $ds: exit=$DS_EXIT, duration=${DS_DURATION}s" >> "$MASTER_LOG"

        # Check environment health between datasets
        check_thermal "post-$ds"
        check_memory_pressure "post-$ds"

        # Mirror after each dataset for incremental backup
        trigger_mirror
    done

    echo "" >> "$MASTER_LOG"
    echo "Phase 3 (Tier-2): ${#DATASETS[@]} datasets, $TIER2_REGRESSIONS regressions" >> "$MASTER_LOG"
fi

# ─── Phase 4: Final Report + Blocking Mirror ────────────────────────────────

phase "Phase 4: Final Report + Sync"

GLOBAL_END=$(date +%s)
GLOBAL_DURATION=$((GLOBAL_END - GLOBAL_START))

{
    echo ""
    echo "========================================"
    echo "  BENCHMARK RUN COMPLETE"
    echo "========================================"
    echo ""
    echo "  Total wall-clock: ${GLOBAL_DURATION}s ($(( GLOBAL_DURATION / 60 ))m $(( GLOBAL_DURATION % 60 ))s)"
    echo "  Results:          $RESULTS"
    echo "  Log:              $MASTER_LOG"
    echo ""
} | tee -a "$MASTER_LOG"

# ─── Cross-run baseline comparison ───────────────────────────────────────────

BASELINES_DIR="$RESULTS/baselines"
mkdir -p "$BASELINES_DIR"

# Find most recent tier-2 JSON report (if any tier-2 ran)
LATEST_REPORT=""
if [[ "$RUN_TIER2" == true ]]; then
    LATEST_REPORT=$(ls -t "$RESULTS/tier2/"*_${TIMESTAMP}.json 2>/dev/null | head -1 || true)
elif [[ "$RUN_TIER1" == true ]]; then
    LATEST_REPORT=$(ls -t "$RESULTS/tier1/"*_${TIMESTAMP}.json 2>/dev/null | head -1 || true)
fi

if [[ -n "$LATEST_REPORT" ]] && [[ -f "$LATEST_REPORT" ]]; then
    GIT_SHA=$(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")

    # Save current result as baseline for this commit
    BASELINE_FILE="$BASELINES_DIR/baseline_${GIT_SHA}.json"
    cp "$LATEST_REPORT" "$BASELINE_FILE"
    ok "Baseline saved: $BASELINE_FILE"

    # Compare against previous baseline (if one exists)
    PREV_BASELINE=$(ls -t "$BASELINES_DIR"/baseline_*.json 2>/dev/null | grep -v "$GIT_SHA" | head -1 || true)
    if [[ -n "$PREV_BASELINE" ]] && [[ -f "$PREV_BASELINE" ]]; then
        info "Comparing against previous baseline: $(basename "$PREV_BASELINE")"
        python3 -c "
import json, sys
cur = json.load(open('$LATEST_REPORT'))
prev = json.load(open('$PREV_BASELINE'))
drifted = []
for cd, pd in zip(cur.get('datasets',[]), prev.get('datasets',[])):
    if cd.get('dataset') != pd.get('dataset'):
        continue
    for metric, val in cd.get('metrics',{}).items():
        pval = pd.get('metrics',{}).get(metric)
        if pval is None or pval == 0:
            continue
        change = (val - pval) / abs(pval)
        if 'rmse' in metric or 'mae' in metric:
            change = -change
        if change < -0.02:
            drifted.append(f'  {cd[\"dataset\"]}/{metric}: {pval:.4f} -> {val:.4f} ({change*100:+.1f}%)')
if drifted:
    print(f'DRIFT DETECTED ({len(drifted)} metrics):')
    for d in drifted:
        print(d)
    sys.exit(0)
else:
    print(f'No drift (>2%) vs {prev.get(\"git_sha\",\"unknown\")}')
" 2>&1 | tee -a "$MASTER_LOG"
    else
        info "No previous baseline found — this is the first run"
    fi
fi

# Final thermal/memory check
check_thermal "final"
check_memory_pressure "final"

# Final sync: foreground (blocking) to guarantee both copies are complete
info "Running final blocking mirror to Google Drive..."
if [[ -x "$MIRROR_SCRIPT" ]]; then
    "$MIRROR_SCRIPT" 2>&1 | tee -a "$MASTER_LOG"
    ok "Final mirror sync complete — both iCloud and Google Drive are up to date"
else
    warn "Mirror script not available"
fi

echo ""
echo "================================================================"
echo "  All done. Results are on:"
echo "    iCloud:       $RESULTS"
echo "    Google Drive: ${FLEXAIDDS_GDRIVE:-unknown}/results"
echo "    Master log:   $MASTER_LOG"
echo "================================================================"
