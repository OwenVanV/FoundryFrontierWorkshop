"""
============================================================================
Streamlit Demo App — Search + RAG Track
============================================================================
Run: uv run streamlit run demo/streamlit_search_rag.py
============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from demo.shared import setup_page, render_module_page, PROJECT_ROOT

setup_page("PayPal Workshop — Search + RAG Track", "📄")

MODULES = [
    {
        "id": "s00",
        "title": "00 — Generate Documents",
        "file": "search_rag/data/generate_documents.py",
        "icon": "📝",
        "summary": "Generate 10 synthetic fraud evidence documents — SARs, compliance memos, audit logs, security alerts.",
        "highlights": [
            "10 realistic payment processor internal documents",
            "Each document contains scattered evidence of the fraud ring",
            "Consistent IDs with the main transaction/customer/merchant data",
            "No single document reveals the full ring — investigation required",
        ],
    },
    {
        "id": "s01",
        "title": "01 — Create Analyzer",
        "file": "search_rag/01_create_analyzer.py",
        "icon": "🔧",
        "summary": "Create a custom Content Understanding analyzer with fraud-specific field extraction schema.",
        "highlights": [
            "Custom field schema: merchant_names, customer_ids, amounts, risk_indicators",
            "Three extraction methods: extract (from text), classify (categorize), generate (LLM-synthesized)",
            "Confidence scoring and source grounding enabled",
            "Base: prebuilt-document with GPT-4.1 completion model",
        ],
    },
    {
        "id": "s02",
        "title": "02 — Analyze Documents",
        "file": "search_rag/02_analyze_documents.py",
        "icon": "🔍",
        "summary": "Run the analyzer against all 10 documents. Extract structured fields with confidence scores.",
        "highlights": [
            "begin_analyze_binary for each document → poll → structured result",
            "Extracted: merchant IDs, customer IDs, amounts, risk indicators",
            "Generated: document summaries, risk indicators (LLM-powered)",
            "Classified: document type, priority level",
        ],
    },
    {
        "id": "s03",
        "title": "03 — Build Search Index",
        "file": "search_rag/03_build_search_index.py",
        "icon": "🗂️",
        "summary": "Create an Azure AI Search index with full-text, vector, and filtered search over the analyzed documents.",
        "highlights": [
            "Index with text fields, collection fields, and vector embeddings",
            "text-embedding-3-large for semantic vector search",
            "Filterable facets: document_type, merchant_ids, customer_ids",
            "Test queries: text search, filter, faceted navigation",
        ],
    },
    {
        "id": "s04",
        "title": "04 — Agent Investigation",
        "file": "search_rag/04_agent_investigation.py",
        "icon": "🕵️",
        "summary": "MAF agent searches documents AND transaction data to build a unified fraud investigation brief.",
        "highlights": [
            "search_fraud_evidence tool: hybrid text + vector search with filters",
            "get_document_content tool: retrieve full document text",
            "query_transactions + lookup_customer tools: CSV data access",
            "4-phase investigation: discovery → cross-reference → synthesis → report",
        ],
    },
    {
        "id": "s05",
        "title": "05 — Evaluation",
        "file": "search_rag/05_evaluation.py",
        "icon": "📏",
        "summary": "Compare RAG-powered agent findings vs ground truth. Precision/recall/F1 comparison across all tracks.",
        "highlights": [
            "Fresh RAG investigation for clean predictions",
            "Precision/recall/F1 for ring members and shell merchants",
            "Compare: LLM Track vs Agents Track vs Search+RAG Track",
            "RAG typically achieves higher recall (evidence is pre-analyzed)",
        ],
    },
]

with st.sidebar:
    st.title("📄 Search + RAG Track")
    st.caption("Content Understanding + AI Search + Agent")
    st.divider()

    st.markdown("### Pipeline")
    st.markdown("""
    ```
    Documents → Content Understanding
                    ↓ extract fields
              Azure AI Search
                    ↓ index + embed
              MAF Agent (RAG)
                    ↓ investigate
              Investigation Brief
    ```
    """)
    st.divider()

    st.markdown("### Modules")
    for mod in MODULES:
        if st.button(f"{mod['icon']} {mod['title']}", key=mod["id"], use_container_width=True):
            st.session_state["selected_module"] = mod["id"]

    st.divider()
    st.markdown("### ⚠️ How to Run")
    st.code("uv sync --extra all\nuv run streamlit run demo/streamlit_search_rag.py", language="bash")
    st.caption("Always use `uv run streamlit` — never a global streamlit install.")

selected_id = st.session_state.get("selected_module", None)

if selected_id:
    mod = next((m for m in MODULES if m["id"] == selected_id), None)
    if mod:
        module_path = PROJECT_ROOT / mod["file"]
        run_cmd = ["uv", "run", "python", mod["file"]]
        highlights_md = "\n".join(f"- {h}" for h in mod["highlights"])
        render_module_page(
            module_title=f"{mod['icon']} {mod['title']}",
            module_file=module_path,
            run_cmd=run_cmd,
            extra_description=f"### Key Highlights\n{highlights_md}",
        )
else:
    st.title("📄 Cracking the Ring — Search + RAG Track")
    st.markdown("""
    **Extract evidence from unstructured documents, index it for search, and unleash an agent to investigate.**

    This track demonstrates the conjunction of three Azure services:

    | Service | Role | Module |
    |---|---|---|
    | **Azure Content Understanding** | Extract structured fields from fraud documents | 01, 02 |
    | **Azure AI Search** | Index, embed, and search the extracted evidence | 03 |
    | **Microsoft Agent Framework** | RAG-powered investigation agent | 04, 05 |

    ### The Story
    The fraud investigation team has received 10 internal documents — SARs, compliance memos,
    audit logs, security alerts, and due diligence files. Each contains scattered evidence of
    the fraud ring. No single document tells the whole story.

    Content Understanding extracts merchant names, account IDs, transaction amounts, and
    AI-generated risk indicators. AI Search makes it all searchable. The agent connects the dots.

    👈 **Select a module from the sidebar to begin.**
    """)

    st.divider()

    cols = st.columns(3)
    for i, mod in enumerate(MODULES):
        with cols[i % 3]:
            st.markdown(f"### {mod['icon']} {mod['title'].split(' — ')[1]}")
            st.caption(mod["summary"][:80] + "...")
