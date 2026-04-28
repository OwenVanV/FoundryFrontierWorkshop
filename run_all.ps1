# ============================================================================
# PayPal x Azure AI Workshop — Launch All Streamlit Demo Apps (PowerShell)
# ============================================================================
# Starts all 4 Streamlit apps on different ports and opens them in the browser.
#
# Usage: .\run_all.ps1
# Stop:  Ctrl+C in this window (kills all spawned processes)
# ============================================================================

Write-Host "=============================================="
Write-Host "  PayPal x Azure AI Workshop — Demo Launcher"
Write-Host "=============================================="
Write-Host ""

$PORT_LLM    = 8501
$PORT_AGENTS = 8502
$PORT_SEARCH = 8503
$PORT_VOICE  = 8504

$jobs = @()

Write-Host "  Starting LLM Track        -> http://localhost:$PORT_LLM"
$jobs += Start-Process -PassThru -NoNewWindow -FilePath "uv" -ArgumentList @(
    "run", "streamlit", "run", "demo/streamlit_llm.py",
    "--server.port=$PORT_LLM",
    "--server.headless=true",
    "--browser.gatherUsageStats=false"
) -RedirectStandardOutput "NUL" -RedirectStandardError "NUL"

Write-Host "  Starting Agents Track     -> http://localhost:$PORT_AGENTS"
$jobs += Start-Process -PassThru -NoNewWindow -FilePath "uv" -ArgumentList @(
    "run", "streamlit", "run", "demo/streamlit_agents.py",
    "--server.port=$PORT_AGENTS",
    "--server.headless=true",
    "--browser.gatherUsageStats=false"
) -RedirectStandardOutput "NUL" -RedirectStandardError "NUL"

Write-Host "  Starting Search+RAG Track -> http://localhost:$PORT_SEARCH"
$jobs += Start-Process -PassThru -NoNewWindow -FilePath "uv" -ArgumentList @(
    "run", "streamlit", "run", "demo/streamlit_search_rag.py",
    "--server.port=$PORT_SEARCH",
    "--server.headless=true",
    "--browser.gatherUsageStats=false"
) -RedirectStandardOutput "NUL" -RedirectStandardError "NUL"

Write-Host "  Starting Voice+CUA Track  -> http://localhost:$PORT_VOICE"
$jobs += Start-Process -PassThru -NoNewWindow -FilePath "uv" -ArgumentList @(
    "run", "streamlit", "run", "demo/streamlit_voice_cua.py",
    "--server.port=$PORT_VOICE",
    "--server.headless=true",
    "--browser.gatherUsageStats=false"
) -RedirectStandardOutput "NUL" -RedirectStandardError "NUL"

Write-Host ""
Write-Host "  Waiting for apps to start..."
Start-Sleep -Seconds 3

# Open in browser
Write-Host "  Opening browser tabs..."
Start-Process "http://localhost:$PORT_LLM"
Start-Process "http://localhost:$PORT_AGENTS"
Start-Process "http://localhost:$PORT_SEARCH"
Start-Process "http://localhost:$PORT_VOICE"

Write-Host ""
Write-Host "=============================================="
Write-Host "  All 4 demo apps running:"
Write-Host "    LLM Track        -> http://localhost:$PORT_LLM"
Write-Host "    Agents Track     -> http://localhost:$PORT_AGENTS"
Write-Host "    Search+RAG Track -> http://localhost:$PORT_SEARCH"
Write-Host "    Voice+CUA Track  -> http://localhost:$PORT_VOICE"
Write-Host ""
Write-Host "  Press Ctrl+C to stop all apps."
Write-Host "=============================================="

# Cleanup on exit
try {
    # Wait for user to press Ctrl+C
    while ($true) { Start-Sleep -Seconds 1 }
}
finally {
    Write-Host ""
    Write-Host "  Stopping all demo apps..."
    foreach ($job in $jobs) {
        try { Stop-Process -Id $job.Id -Force -ErrorAction SilentlyContinue } catch {}
    }
    Write-Host "  All apps stopped."
}
