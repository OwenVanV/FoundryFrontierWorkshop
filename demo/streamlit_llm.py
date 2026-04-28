"""
============================================================================
Streamlit Demo App — LLM Track
============================================================================
Run: uv run streamlit run demo/streamlit_llm.py
============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from demo.shared import setup_page, render_module_page, PROJECT_ROOT

setup_page("PayPal Workshop — LLM Track", "🔍")

# ============================================================================
# Module definitions
# ============================================================================

MODULES = [
    {
        "id": "01",
        "title": "01 — Data Exploration",
        "file": "llm/01_data_exploration.py",
        "icon": "📊",
        "summary": "Use Azure OpenAI to profile and understand the fraud dataset landscape.",
        "highlights": [
            "Loads 50K transactions, 2K customers, 500 merchants",
            "Computes statistical profiles and sends summaries to GPT-4o",
            "LLM identifies initial anomalies in amount distributions and demographics",
            "Every call streams live with TTFT measurement",
        ],
    },
    {
        "id": "02",
        "title": "02 — Pattern Detection",
        "file": "llm/02_pattern_detection.py",
        "icon": "🎯",
        "summary": "Structured JSON outputs for parseable anomaly detection — smurfing, temporal patterns, shell companies.",
        "highlights": [
            "JSON mode (`response_format`) forces parseable output",
            "Zero-shot smurfing detection in $9K-$10K range",
            "Temporal analysis: coordinated transaction timing",
            "Shell merchant detection via name similarity",
        ],
    },
    {
        "id": "03",
        "title": "03 — Cross-Reference Analysis",
        "file": "llm/03_cross_reference_analysis.py",
        "icon": "🔗",
        "summary": "Multi-turn conversations connecting signals across CSV, JSON, and time-series data.",
        "highlights": [
            "Device fingerprint sharing detection (nested JSON)",
            "Impossible travel detection (Lagos → London in 20 min)",
            "4-turn investigation conversation with chain-of-thought",
            "Context accumulation — token costs grow per turn",
        ],
    },
    {
        "id": "04",
        "title": "04 — Fraud Ring Investigation",
        "file": "llm/04_fraud_ring_investigation.py",
        "icon": "🕵️",
        "summary": "Network graph reasoning to map the full fraud ring and generate a formal investigation report.",
        "highlights": [
            "Transaction network graph (sender → receiver adjacency)",
            "Bidirectional flow and circular pattern detection",
            "LLM-generated formal investigation report",
            "Responsible AI: human oversight discussion",
        ],
    },
    {
        "id": "05",
        "title": "05 — Evaluation Framework",
        "file": "llm/05_evaluation_framework.py",
        "icon": "📏",
        "summary": "Precision, recall, F1 against ground truth. Prompt variant testing.",
        "highlights": [
            "Compare LLM predictions vs. ground truth answer key",
            "Precision/recall/F1 for ring members and shell merchants",
            "4 prompt variants tested: baseline, chain-of-thought, high-precision, high-recall",
            "Error analysis: why false positives and negatives occurred",
        ],
    },
    {
        "id": "06",
        "title": "06 — Observability Dashboard",
        "file": "llm/06_observability_dashboard.py",
        "icon": "📈",
        "summary": "Cost analysis, latency distributions, platform health, and production readiness.",
        "highlights": [
            "Token usage and cost attribution by module",
            "Latency percentile analysis (P50, P90, P99)",
            "Platform service health dashboard",
            "Production readiness scoring with pass/warn/fail checks",
        ],
    },
]

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.title("🔍 LLM Track")
    st.caption("Raw Azure OpenAI — no agents, no frameworks")
    st.divider()

    st.markdown("### Modules")
    selected = None
    for mod in MODULES:
        if st.button(
            f"{mod['icon']} {mod['title']}",
            key=mod["id"],
            use_container_width=True,
        ):
            st.session_state["selected_module"] = mod["id"]

    st.divider()
    st.markdown("### Quick Actions")
    if st.button("🗃️ Regenerate Data", use_container_width=True):
        st.session_state["selected_module"] = "generate_data"

    if st.button("🧪 Run Tests", use_container_width=True):
        st.session_state["selected_module"] = "run_tests"

    st.divider()
    st.markdown("### ⚠️ How to Run")
    st.code("uv sync --extra all\nuv run streamlit run demo/streamlit_llm.py", language="bash")
    st.caption("Always use `uv run streamlit` — never a global streamlit install. The venv must have all workshop dependencies.")

# ============================================================================
# Main content
# ============================================================================

selected_id = st.session_state.get("selected_module", None)

if selected_id == "generate_data":
    st.header("🗃️ Generate Synthetic Data")
    st.info("Generates all 8 dataset files + ground truth answer key. Deterministic (seed=42).")
    from demo.shared import stream_subprocess
    if st.button("▶️ Generate", type="primary"):
        stream_subprocess(["uv", "run", "python", "data/generate_synthetic_data.py"])

elif selected_id == "run_tests":
    st.header("🧪 Test Suite")
    st.info("52 tests: data integrity, imports, syntax, auth, security, fraud patterns.")
    from demo.shared import stream_subprocess
    if st.button("▶️ Run Tests", type="primary"):
        stream_subprocess(["uv", "run", "python", "tests/test_workshop.py"])

elif selected_id:
    mod = next((m for m in MODULES if m["id"] == selected_id), None)
    if mod:
        module_path = PROJECT_ROOT / mod["file"]
        run_cmd = ["uv", "run", "python", mod["file"]]

        # Extra description with highlights
        highlights_md = "\n".join(f"- {h}" for h in mod["highlights"])
        extra = f"### Key Highlights\n{highlights_md}"

        render_module_page(
            module_title=f"{mod['icon']} {mod['title']}",
            module_file=module_path,
            run_cmd=run_cmd,
            extra_description=extra,
        )
else:
    # Landing page
    st.title("🔍 Cracking the Ring — LLM Track")
    st.markdown("""
    **Use Azure OpenAI to uncover a hidden fraud ring buried in 50,000+ synthetic payment transactions.**

    This track demonstrates raw LLM capabilities — no agents, no frameworks.
    Each module streams output live with TTFT (Time to First Token) measurement.

    👈 **Select a module from the sidebar to begin.**
    """)

    st.divider()

    # Module overview cards
    cols = st.columns(3)
    for i, mod in enumerate(MODULES):
        with cols[i % 3]:
            st.markdown(f"### {mod['icon']} {mod['title'].split(' — ')[1]}")
            st.caption(mod["summary"])
