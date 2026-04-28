"""
============================================================================
Streamlit Demo App — Agents Track
============================================================================
Run: uv run streamlit run demo/streamlit_agents.py
============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from demo.shared import setup_page, render_module_page, PROJECT_ROOT

setup_page("PayPal Workshop — Agents Track", "🤖")

# ============================================================================
# Module definitions
# ============================================================================

MODULES = [
    {
        "id": "a01",
        "title": "01 — Foundry Agent Service Setup",
        "file": "agents/01_foundry_agent_setup.py",
        "icon": "🏗️",
        "system": "Foundry Agent Service",
        "summary": "Create a managed Prompt Agent with Code Interpreter. Upload CSVs and let the agent write + execute pandas code in a sandbox.",
        "highlights": [
            "AIProjectClient connects to your Foundry project",
            "Upload transactions.csv, customers.csv, merchants.csv",
            "Versioned Prompt Agent with Code Interpreter tool",
            "Agent writes pandas code autonomously in Hyper-V sandbox",
            "Download generated charts (histograms, scatter plots)",
        ],
    },
    {
        "id": "a02",
        "title": "02 — MAF: First Agent",
        "file": "agents/02_maf_first_agent.py",
        "icon": "🛠️",
        "system": "Microsoft Agent Framework",
        "summary": "Build a local MAF agent with FoundryChatClient and 6 custom @tool fraud investigation functions.",
        "highlights": [
            "FoundryChatClient for Foundry model inference",
            "6 custom @tool functions: query_transactions, lookup_customer, etc.",
            "@tool decorator auto-generates JSON schemas from type hints",
            "Single-shot and multi-turn investigations with AgentSession",
        ],
    },
    {
        "id": "a03",
        "title": "03 — Deep Code Analysis",
        "file": "agents/03_foundry_code_analysis.py",
        "icon": "📊",
        "system": "Foundry Agent Service",
        "summary": "5-phase deep analysis: network graphs, circular flows, time-series anomalies, all via Code Interpreter.",
        "highlights": [
            "Upload ALL datasets for comprehensive analysis",
            "5 sequential analysis phases in one conversation",
            "Agent builds adjacency matrices and detects cycles",
            "Cross-references network with customer demographics",
            "Generates publication-quality network diagrams",
        ],
    },
    {
        "id": "a04",
        "title": "04 — MAF Multi-Turn Investigation",
        "file": "agents/04_maf_multi_turn_investigation.py",
        "icon": "🔍",
        "system": "Microsoft Agent Framework",
        "summary": "4-phase progressive investigation with streaming. Agent builds a case file across turns.",
        "highlights": [
            "4 investigation phases: tip → follow money → identity → summary",
            "Agent-directed: it decides which tools to call",
            "Streaming output for the final case summary",
            "Evidence accumulation with AgentSession",
        ],
    },
    {
        "id": "a05",
        "title": "05 — Workflow Orchestration",
        "file": "agents/05_maf_workflow_orchestration.py",
        "icon": "🔄",
        "system": "Microsoft Agent Framework",
        "summary": "3 specialist agents in a functional workflow: TransactionAnalyst → IdentityInvestigator → ReportWriter.",
        "highlights": [
            "3 agents with focused roles and minimal tools each",
            "Functional workflow: plain async def, no DSL",
            "Data flows: each agent's output feeds the next",
            "ReportWriter synthesizes into a formal investigation report",
        ],
    },
    {
        "id": "a06",
        "title": "06 — Evaluation",
        "file": "agents/06_evaluation.py",
        "icon": "📏",
        "system": "Microsoft Agent Framework",
        "summary": "Compare agent-identified fraud ring vs. ground truth. Precision, recall, F1.",
        "highlights": [
            "Fresh single-shot investigation for clean predictions",
            "Compare against ground_truth.json answer key",
            "Precision/recall/F1 for members and merchants",
            "Compare agent accuracy vs. raw LLM track results",
        ],
    },
]

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.title("🤖 Agents Track")
    st.caption("Foundry Agent Service + Microsoft Agent Framework")
    st.divider()

    st.markdown("### Modules")
    for mod in MODULES:
        badge = "🟦 FAS" if "Foundry" in mod["system"] else "🟩 MAF"
        if st.button(
            f"{mod['icon']} {mod['title']}",
            key=mod["id"],
            use_container_width=True,
            help=f"{badge} — {mod['system']}",
        ):
            st.session_state["selected_module"] = mod["id"]

    st.divider()
    st.markdown("""
    **Legend:**
    - 🟦 FAS = Foundry Agent Service
    - 🟩 MAF = Microsoft Agent Framework
    """)
    st.divider()
    st.markdown("### ⚠️ How to Run")
    st.code("uv sync --extra all\nuv run streamlit run demo/streamlit_agents.py", language="bash")
    st.caption("Always use `uv run streamlit` — never a global streamlit install. The venv must have all workshop dependencies.")

# ============================================================================
# Main content
# ============================================================================

selected_id = st.session_state.get("selected_module", None)

if selected_id:
    mod = next((m for m in MODULES if m["id"] == selected_id), None)
    if mod:
        module_path = PROJECT_ROOT / mod["file"]
        run_cmd = ["uv", "run", "python", mod["file"]]

        # System badge
        badge = "🟦 Foundry Agent Service" if "Foundry" in mod["system"] else "🟩 Microsoft Agent Framework"
        highlights_md = "\n".join(f"- {h}" for h in mod["highlights"])
        extra = f"""
> **System:** {badge}

### Key Highlights
{highlights_md}
"""

        render_module_page(
            module_title=f"{mod['icon']} {mod['title']}",
            module_file=module_path,
            run_cmd=run_cmd,
            extra_description=extra,
        )
else:
    # Landing page
    st.title("🤖 Cracking the Ring — Agents Track")
    st.markdown("""
    **Build intelligent fraud investigators using Foundry Agent Service and Microsoft Agent Framework.**

    Two agent systems, one investigation:

    | System | Package | Modules | Best For |
    |---|---|---|---|
    | 🟦 **Foundry Agent Service** | `azure-ai-projects` | 01, 03 | Sandboxed code execution, managed tools |
    | 🟩 **Microsoft Agent Framework** | `agent-framework` | 02, 04, 05, 06 | Custom tools, orchestration, workflows |

    👈 **Select a module from the sidebar to begin.**
    """)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🟦 Foundry Agent Service")
        st.caption("Managed cloud agents with Code Interpreter")
        for mod in MODULES:
            if "Foundry" in mod["system"]:
                st.markdown(f"- {mod['icon']} **{mod['title'].split(' — ')[1]}** — {mod['summary'][:60]}...")

    with col2:
        st.markdown("### 🟩 Microsoft Agent Framework")
        st.caption("Local SDK with custom @tool functions")
        for mod in MODULES:
            if "MAF" in mod["system"] or "Microsoft Agent" in mod["system"]:
                st.markdown(f"- {mod['icon']} **{mod['title'].split(' — ')[1]}** — {mod['summary'][:60]}...")
