#!/bin/bash
# ============================================================
# start_db_boa.sh
# Master startup script for the DB-BOA Financial Security System
#
# Usage:
#   ./start_db_boa.sh            — starts API server + opens dashboard
#   ./start_db_boa.sh --python   — runs Python pipeline directly (no UI)
#   ./start_db_boa.sh --fabric   — also starts Hyperledger Fabric network
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_DIR="$PROJECT_ROOT/db_boa_framework"
FABRIC_DIR="$PROJECT_ROOT/db_boa_fabric"
API_DIR="$FABRIC_DIR/api-server"
DASHBOARD="$API_DIR/index.html"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; NC='\033[0m'

header() { echo -e "\n${BOLD}${CYAN}══ $1 ══${NC}\n"; }
ok()     { echo -e "${GREEN}  ✔  $1${NC}"; }
info()   { echo -e "${YELLOW}  ►  $1${NC}"; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}████████████████████████████████████████████████████████████████████${NC}"
echo -e "${BOLD}${CYAN}█           DB-BOA Financial Security Framework                    █${NC}"
echo -e "${BOLD}${CYAN}█           Prabanand & Thanabal (2025) — Implementation           █${NC}"
echo -e "${BOLD}${CYAN}████████████████████████████████████████████████████████████████████${NC}"
echo ""

# ── Parse args ────────────────────────────────────────────────────────────────
PYTHON_ONLY=false
START_FABRIC=false
QUICK=false

for arg in "$@"; do
    case $arg in
        --python)   PYTHON_ONLY=true ;;
        --fabric)   START_FABRIC=true ;;
        --quick)    QUICK=true ;;
    esac
done

# ── Mode: Python only ─────────────────────────────────────────────────────────
if [ "$PYTHON_ONLY" = true ]; then
    header "Running Python DB-BOA Pipeline"
    cd "$PYTHON_DIR"
    ARGS=""
    [ "$QUICK" = true ] && ARGS="--quick"
    python3 main.py $ARGS
    ok "Pipeline complete. Results in: $PYTHON_DIR/results/"
    exit 0
fi

# ── Mode: Start Fabric network ────────────────────────────────────────────────
if [ "$START_FABRIC" = true ]; then
    header "Starting Hyperledger Fabric Network"

    FABRIC_SAMPLES="$PROJECT_ROOT/fabric/fabric-samples"
    if [ ! -d "$FABRIC_SAMPLES" ]; then
        echo "ERROR: fabric-samples not found at $FABRIC_SAMPLES"
        echo "Please install Hyperledger Fabric first (Checkpoint 1 of your lab)."
        exit 1
    fi

    cd "$FABRIC_SAMPLES/test-network"

    # Always tear down stale network before bringing up fresh
    info "Tearing down any existing Fabric network…"
    ./network.sh down || true
    # Explicitly remove the named ledger volumes (compose project prefix = "compose_")
    # network.sh down looks for wrong prefix "docker_" so these are never auto-deleted
    docker volume rm compose_orderer.example.com \
                      compose_peer0.org1.example.com \
                      compose_peer0.org2.example.com 2>/dev/null || true
    # Also prune any remaining anonymous/unused volumes
    docker volume prune -f
    ok "Previous network stopped and cleaned"

    # Clear stale wallet so enroll/register runs fresh
    rm -rf "$API_DIR/wallet"

    # Copy chaincode into fabric-samples
    CHAINCODE_DEST="$FABRIC_SAMPLES/db-boa"
    mkdir -p "$CHAINCODE_DEST"
    cp -r "$FABRIC_DIR/chaincode/"* "$CHAINCODE_DEST/"

    info "Bringing up Fabric network (Org1 + Org2 + Orderer + CouchDB)…"
    ./network.sh up createChannel -ca -s couchdb

    info "Installing chaincode dependencies…"
    (cd "$CHAINCODE_DEST" && npm install --silent)

    info "Deploying DB-BOA chaincode…"
    ./network.sh deployCC -ccn db-boa -ccp ../db-boa -ccl javascript -ccv 1 -ccs 1

    ok "Fabric network started and chaincode deployed!"
    ok "CouchDB UI: http://localhost:5984/_utils  (admin/adminpw)"

    # Enroll admin + register user against fresh CAs
    cd "$API_DIR"
    npm install --silent
    node enrollAdmin.js
    node registerUser.js
fi

# ── Start API server ──────────────────────────────────────────────────────────
header "Starting DB-BOA REST API Server"
cd "$API_DIR"

if [ ! -d "node_modules" ]; then
    info "Installing npm dependencies…"
    npm install --silent
    ok "Dependencies installed"
fi

info "Starting API server on http://localhost:3001 …"
node server.js &
API_PID=$!
echo "  API server PID: $API_PID"
sleep 2

# ── Open Dashboard ────────────────────────────────────────────────────────────
header "Opening DB-BOA Dashboard"
ok "Dashboard: file://$DASHBOARD"
ok "API:       http://localhost:3001"
ok "Health:    http://localhost:3001/health"

# Try to open browser
if command -v xdg-open &>/dev/null; then
    xdg-open "$DASHBOARD" 2>/dev/null &
elif command -v open &>/dev/null; then
    open "$DASHBOARD" &
fi

echo ""
info "Press Ctrl+C to stop the API server"
echo ""

# ── Keep running ──────────────────────────────────────────────────────────────
trap "kill $API_PID 2>/dev/null; echo ''; echo 'Server stopped.'" EXIT
wait $API_PID
