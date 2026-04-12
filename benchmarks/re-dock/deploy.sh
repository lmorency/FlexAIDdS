#!/usr/bin/env bash
# ===========================================================================
# RE-DOCK Infrastructure Deployment Script
# ===========================================================================
#
# Deploys the distributed docking benchmark infrastructure:
#   1. HuggingFace Spaces: 8 persistent worker nodes (FastAPI + Gradio)
#   2. Kubernetes: 8-replica temperature-ladder worker deployments
#   3. Vercel: Serverless coordinator API
#
# Prerequisites:
#   - huggingface-cli authenticated (huggingface-cli login)
#   - kubectl configured for target cluster
#   - vercel CLI authenticated (vercel login)
#   - Docker for building worker images
#
# Usage:
#   ./deploy.sh [hf|k8s|vercel|all] [--dry-run] [--namespace NS]
#
# Environment variables:
#   HF_ORG            HuggingFace organization (default: nrglab)
#   HF_SPACE_PREFIX   Space name prefix (default: redock-worker)
#   K8S_NAMESPACE     Kubernetes namespace (default: flexaids-redock)
#   VERCEL_PROJECT    Vercel project name (default: redock-coordinator)
#   DOCKER_REGISTRY   Container registry (default: ghcr.io/nrglab)
#
# Le Bonhomme Pharma / Najmanovich Research Group
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_ORG="${HF_ORG:-nrglab}"
HF_SPACE_PREFIX="${HF_SPACE_PREFIX:-redock-worker}"
K8S_NAMESPACE="${K8S_NAMESPACE:-flexaids-redock}"
VERCEL_PROJECT="${VERCEL_PROJECT:-redock-coordinator}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io/nrglab}"
DRY_RUN=false

# Geometric temperature ladder: 8 replicas, 298K–600K
TEMPERATURES=(298.0 329.3 363.8 401.9 444.0 490.6 542.0 600.0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()   { echo "[RE-DOCK] $*"; }
warn()  { echo "[RE-DOCK] WARNING: $*" >&2; }
die()   { echo "[RE-DOCK] ERROR: $*" >&2; exit 1; }

run_cmd() {
    if $DRY_RUN; then
        echo "  [dry-run] $*"
    else
        eval "$@"
    fi
}

check_tool() {
    command -v "$1" >/dev/null 2>&1 || die "$1 not found — install it first"
}

# ---------------------------------------------------------------------------
# Deploy HuggingFace Spaces (8 workers)
# ---------------------------------------------------------------------------

deploy_hf() {
    log "Deploying HuggingFace Space workers..."
    check_tool huggingface-cli

    for i in "${!TEMPERATURES[@]}"; do
        T="${TEMPERATURES[$i]}"
        SPACE_NAME="${HF_SPACE_PREFIX}-${i}"
        REPO="${HF_ORG}/${SPACE_NAME}"
        log "  Worker $i: T=${T}K → ${REPO}"

        # Create or update Space
        run_cmd "huggingface-cli repo create '${SPACE_NAME}' \
            --organization '${HF_ORG}' \
            --type space \
            --space-sdk gradio \
            2>/dev/null || true"

        # Clone, copy files, push
        TMPDIR=$(mktemp -d)
        run_cmd "git clone 'https://huggingface.co/spaces/${REPO}' '${TMPDIR}/space' 2>/dev/null || \
                 git init '${TMPDIR}/space'"

        run_cmd "cp '${SCRIPT_DIR}/hf_space/app.py' '${TMPDIR}/space/app.py'"
        run_cmd "cp '${SCRIPT_DIR}/hf_space/requirements.txt' '${TMPDIR}/space/requirements.txt'"
        run_cmd "cp '${SCRIPT_DIR}/hf_space/README.md' '${TMPDIR}/space/README.md'"

        # Set environment variables for this replica
        run_cmd "cd '${TMPDIR}/space' && \
                 git add -A && \
                 git commit -m 'Deploy RE-DOCK worker ${i} (T=${T}K)' --allow-empty && \
                 git push origin main 2>/dev/null || true"

        run_cmd "rm -rf '${TMPDIR}'"

        # Set Space secrets/variables
        run_cmd "huggingface-cli repo set-variable '${REPO}' REPLICA_INDEX '${i}' --type space 2>/dev/null || true"
        run_cmd "huggingface-cli repo set-variable '${REPO}' REPLICA_TEMPERATURE '${T}' --type space 2>/dev/null || true"
    done

    log "HuggingFace Spaces deployment complete (${#TEMPERATURES[@]} workers)"
}

# ---------------------------------------------------------------------------
# Deploy Kubernetes manifests
# ---------------------------------------------------------------------------

deploy_k8s() {
    log "Deploying Kubernetes manifests..."
    check_tool kubectl

    K8S_DIR="${SCRIPT_DIR}/k8s"

    run_cmd "kubectl apply -f '${K8S_DIR}/namespace.yaml'"
    run_cmd "kubectl apply -f '${K8S_DIR}/configmap.yaml'"
    run_cmd "kubectl apply -f '${K8S_DIR}/pvc.yaml'"
    run_cmd "kubectl apply -f '${K8S_DIR}/worker-deployment.yaml'"
    run_cmd "kubectl apply -f '${K8S_DIR}/services.yaml'"
    run_cmd "kubectl apply -f '${K8S_DIR}/exchange-cronjob.yaml'"

    log "Waiting for worker pods..."
    if ! $DRY_RUN; then
        kubectl -n "${K8S_NAMESPACE}" rollout status deployment/redock-worker-0 --timeout=120s || \
            warn "Worker 0 rollout not ready (may need image push)"
    fi

    log "Kubernetes deployment complete"
}

# ---------------------------------------------------------------------------
# Deploy Vercel coordinator
# ---------------------------------------------------------------------------

deploy_vercel() {
    log "Deploying Vercel coordinator..."
    check_tool vercel

    VERCEL_DIR="${SCRIPT_DIR}/vercel_coordinator"

    run_cmd "cd '${VERCEL_DIR}' && vercel deploy --prod --yes"

    log "Vercel coordinator deployed: https://${VERCEL_PROJECT}.vercel.app"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

usage() {
    echo "Usage: $0 [hf|k8s|vercel|all] [--dry-run] [--namespace NS]"
    echo ""
    echo "Components:"
    echo "  hf       Deploy HuggingFace Space workers (8 replicas)"
    echo "  k8s      Deploy Kubernetes manifests"
    echo "  vercel   Deploy Vercel serverless coordinator"
    echo "  all      Deploy everything (default)"
    echo ""
    echo "Options:"
    echo "  --dry-run      Show commands without executing"
    echo "  --namespace NS Override K8s namespace"
}

COMPONENT="${1:-all}"
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --namespace) K8S_NAMESPACE="$2"; shift 2 ;;
        --help|-h)   usage; exit 0 ;;
        *)           warn "Unknown option: $1"; shift ;;
    esac
done

log "RE-DOCK Infrastructure Deployment"
log "  Component: ${COMPONENT}"
log "  Dry run:   ${DRY_RUN}"
log ""

case "${COMPONENT}" in
    hf)     deploy_hf ;;
    k8s)    deploy_k8s ;;
    vercel) deploy_vercel ;;
    all)    deploy_hf; deploy_k8s; deploy_vercel ;;
    *)      die "Unknown component: ${COMPONENT}. Use hf|k8s|vercel|all" ;;
esac

log "Deployment complete!"
