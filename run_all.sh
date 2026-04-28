#!/usr/bin/env bash
# ============================================================================
# PayPal x Azure AI Workshop — Launch All Streamlit Demo Apps
# ============================================================================
# Starts all 4 Streamlit apps on different ports and opens them in the browser.
#
# Usage: ./run_all.sh
# Stop:  Ctrl+C (kills all background processes)
# ============================================================================

set -e

echo "=============================================="
echo "  PayPal x Azure AI Workshop — Demo Launcher"
echo "=============================================="
echo ""

# Ports for each app
PORT_LLM=8501
PORT_AGENTS=8502
PORT_SEARCH=8503
PORT_VOICE=8504

# Track PIDs for cleanup
PIDS=()

cleanup() {
    echo ""
    echo "  Stopping all demo apps..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "  All apps stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Launch each app
echo "  Starting LLM Track        → http://localhost:${PORT_LLM}"
uv run streamlit run demo/streamlit_llm.py \
    --server.port=${PORT_LLM} \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    > /dev/null 2>&1 &
PIDS+=($!)

echo "  Starting Agents Track     → http://localhost:${PORT_AGENTS}"
uv run streamlit run demo/streamlit_agents.py \
    --server.port=${PORT_AGENTS} \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    > /dev/null 2>&1 &
PIDS+=($!)

echo "  Starting Search+RAG Track → http://localhost:${PORT_SEARCH}"
uv run streamlit run demo/streamlit_search_rag.py \
    --server.port=${PORT_SEARCH} \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    > /dev/null 2>&1 &
PIDS+=($!)

echo "  Starting Voice+CUA Track  → http://localhost:${PORT_VOICE}"
uv run streamlit run demo/streamlit_voice_cua.py \
    --server.port=${PORT_VOICE} \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    > /dev/null 2>&1 &
PIDS+=($!)

# Wait for apps to start
echo ""
echo "  Waiting for apps to start..."
sleep 3

# Open in browser
echo "  Opening browser tabs..."
if command -v open &> /dev/null; then
    # macOS
    open "http://localhost:${PORT_LLM}"
    open "http://localhost:${PORT_AGENTS}"
    open "http://localhost:${PORT_SEARCH}"
    open "http://localhost:${PORT_VOICE}"
elif command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open "http://localhost:${PORT_LLM}"
    xdg-open "http://localhost:${PORT_AGENTS}"
    xdg-open "http://localhost:${PORT_SEARCH}"
    xdg-open "http://localhost:${PORT_VOICE}"
fi

echo ""
echo "=============================================="
echo "  All 4 demo apps running:"
echo "    LLM Track        → http://localhost:${PORT_LLM}"
echo "    Agents Track     → http://localhost:${PORT_AGENTS}"
echo "    Search+RAG Track → http://localhost:${PORT_SEARCH}"
echo "    Voice+CUA Track  → http://localhost:${PORT_VOICE}"
echo ""
echo "  Press Ctrl+C to stop all apps."
echo "=============================================="

# Wait for all background processes
wait
