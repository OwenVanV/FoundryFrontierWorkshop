"""
============================================================================
SEARCH+RAG MODULE 04 — MAF Agent: Document-Powered Investigation
============================================================================

WORKSHOP NARRATIVE:
    The evidence is indexed. Now we unleash a MAF agent that can SEARCH
    the fraud evidence documents and CROSS-REFERENCE them with the
    transaction data from the main workshop.

    The agent has two types of tools:
    1. search_fraud_evidence — queries Azure AI Search (documents)
    2. query_transactions / lookup_customer — queries the CSV data

    By combining document evidence (SARs, memos, alerts) with structured
    transaction data, the agent can build a more complete picture of
    the fraud ring than either source alone.

LEARNING OBJECTIVES:
    1. Use Azure AI Search as a RAG data source for an agent
    2. Combine document search with structured data tools
    3. Observe how document evidence strengthens the investigation
    4. Understand the RAG pattern: retrieve → augment → generate

ESTIMATED TIME: 20-25 minutes

============================================================================
"""

import os
import sys
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv
_dir = Path(__file__).resolve().parent
load_dotenv(_dir / ".env")
load_dotenv()

# Add project root to path BEFORE any project imports
PROJECT_ROOT = _dir.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "agents"))

from typing import Annotated
from pydantic import Field

from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

# Import auth with fallback
try:
    from utils.auth import use_managed_identity
except ImportError:
    import importlib.util
    _spec = importlib.util.spec_from_file_location("auth", PROJECT_ROOT / "utils" / "auth.py")
    _auth = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_auth)
    use_managed_identity = _auth.use_managed_identity

# ============================================================================
# CONFIGURATION
# ============================================================================
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "fraud-evidence")
FOUNDRY_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT")
MODEL = os.getenv("FOUNDRY_MODEL", "gpt-4o")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
USE_MI = use_managed_identity()


def get_search_client():
    if USE_MI:
        from azure.identity import DefaultAzureCredential
        return SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=DefaultAzureCredential())
    return SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=AzureKeyCredential(SEARCH_KEY))


def get_embedding(text: str) -> list[float]:
    if USE_MI:
        try:
            from utils.auth import get_token_provider
        except ImportError:
            import importlib.util
            _spec = importlib.util.spec_from_file_location("auth", PROJECT_ROOT / "utils" / "auth.py")
            _auth = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_auth)
            get_token_provider = _auth.get_token_provider
        client = AzureOpenAI(azure_endpoint=OPENAI_ENDPOINT, azure_ad_token_provider=get_token_provider(), api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))
    else:
        client = AzureOpenAI(azure_endpoint=OPENAI_ENDPOINT, api_key=OPENAI_KEY, api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))
    response = client.embeddings.create(input=[text[:8000]], model=EMBEDDING_DEPLOYMENT)
    return response.data[0].embedding


# ============================================================================
# SEARCH TOOLS — Query the document evidence index
# ============================================================================

_search_client = None

def _get_search():
    global _search_client
    if _search_client is None:
        _search_client = get_search_client()
    return _search_client


@tool(approval_mode="never_require")
def search_fraud_evidence(
    query: Annotated[str, Field(description="Natural language search query about fraud evidence, e.g. 'structuring below BSA threshold' or 'Apex Digital Solutions'")],
    document_type: Annotated[str | None, Field(description="Filter by document type: suspicious_activity_report, compliance_memo, due_diligence, audit_log, onboarding_summary, security_alert, risk_report, dispute_summary", default=None)] = None,
    merchant_id: Annotated[str | None, Field(description="Filter by merchant ID (MER-XXXX)", default=None)] = None,
    customer_id: Annotated[str | None, Field(description="Filter by customer ID (CUS-XXXX)", default=None)] = None,
) -> str:
    """Search the fraud evidence document index. Returns matching documents with summaries, risk indicators, and extracted fields."""
    print(f"  \U0001f50d Tool: search_fraud_evidence(query='{query[:50]}...'", end="", flush=True)
    if document_type: print(f", type={document_type}", end="", flush=True)
    if merchant_id: print(f", merchant={merchant_id}", end="", flush=True)
    if customer_id: print(f", customer={customer_id}", end="", flush=True)
    print(")", flush=True)
    client = _get_search()

    # Build filter
    filters = []
    if document_type:
        filters.append(f"document_type eq '{document_type}'")
    if merchant_id:
        filters.append(f"merchant_ids/any(m: m eq '{merchant_id}')")
    if customer_id:
        filters.append(f"customer_ids/any(c: c eq '{customer_id}')")
    filter_str = " and ".join(filters) if filters else None

    # Generate embedding for vector search
    embedding = get_embedding(query)
    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=5, fields="embedding")

    results = client.search(
        search_text=query,
        vector_queries=[vector_query],
        filter=filter_str,
        top=5,
        select=["filename", "document_type", "priority_level", "summary",
                "merchant_names", "merchant_ids", "customer_ids",
                "risk_indicators", "recommended_actions"],
    )

    output_lines = []
    for r in results:
        output_lines.append(f"--- {r['filename']} ({r['document_type']}) ---")
        output_lines.append(f"Priority: {r.get('priority_level', 'N/A')}")
        output_lines.append(f"Summary: {r.get('summary', 'N/A')}")
        if r.get("merchant_ids"):
            output_lines.append(f"Merchant IDs: {', '.join(r['merchant_ids'])}")
        if r.get("customer_ids"):
            output_lines.append(f"Customer IDs: {', '.join(r['customer_ids'][:10])}")
        if r.get("risk_indicators"):
            output_lines.append(f"Risk Indicators: {'; '.join(r['risk_indicators'])}")
        if r.get("recommended_actions"):
            output_lines.append(f"Recommended Actions: {'; '.join(r['recommended_actions'])}")
        output_lines.append("")

    if not output_lines:
        print(f"    → 0 results", flush=True)
        return "No documents found matching the search criteria."

    doc_count = output_lines.count("")  # Each doc ends with ""
    print(f"    → {doc_count} documents returned", flush=True)
    return "\n".join(output_lines)


@tool(approval_mode="never_require")
def get_document_content(
    filename: Annotated[str, Field(description="The exact filename to retrieve, e.g. 'SAR-2026-0142_apex_digital.txt'")],
) -> str:
    """Retrieve the full text content of a specific fraud evidence document."""
    print(f"  \U0001f4c4 Tool: get_document_content('{filename}')", flush=True)
    client = _get_search()
    results = client.search(
        search_text="*",
        filter=f"filename eq '{filename}'",
        top=1,
        select=["filename", "content"],
    )
    for r in results:
        content = r.get("content", "")
        return content[:5000]  # Truncate for context window
    return f"Document '{filename}' not found in the index."


# ============================================================================
# TRANSACTION DATA TOOLS — from the main workshop data
# ============================================================================

import csv
DATA_DIR = PROJECT_ROOT / "data"


@tool(approval_mode="never_require")
def query_transactions(
    sender_id: Annotated[str | None, Field(description="Sender customer ID", default=None)] = None,
    merchant_id: Annotated[str | None, Field(description="Merchant ID", default=None)] = None,
    min_amount: Annotated[float | None, Field(description="Minimum amount", default=None)] = None,
    max_amount: Annotated[float | None, Field(description="Maximum amount", default=None)] = None,
) -> str:
    """Query the transaction ledger with filters. Returns summary statistics."""
    print(f"  \U0001f4b3 Tool: query_transactions(", end="", flush=True)
    parts = []
    if sender_id: parts.append(f"sender={sender_id}")
    if merchant_id: parts.append(f"merchant={merchant_id}")
    if min_amount: parts.append(f"min=${min_amount:,.0f}")
    if max_amount: parts.append(f"max=${max_amount:,.0f}")
    print(f"{', '.join(parts) if parts else '*'})", flush=True)
    count, total = 0, 0.0
    senders, merchants = set(), set()
    with open(DATA_DIR / "transactions.csv", "r") as f:
        for row in csv.DictReader(f):
            amt = float(row["amount"])
            if sender_id and row["sender_id"] != sender_id: continue
            if merchant_id and row["merchant_id"] != merchant_id: continue
            if min_amount and amt < min_amount: continue
            if max_amount and amt > max_amount: continue
            count += 1
            total += amt
            senders.add(row["sender_id"])
            merchants.add(row["merchant_id"])
            if count >= 200: break
    if count == 0:
        return "No transactions found."
    return (f"Found {count}+ transactions totaling ${total:,.2f}. "
            f"{len(senders)} unique senders, {len(merchants)} merchants.")


@tool(approval_mode="never_require")
def lookup_customer(
    customer_id: Annotated[str, Field(description="Customer ID (CUS-XXXX)")],
) -> str:
    """Look up a customer profile."""
    print(f"  \U0001f464 Tool: lookup_customer('{customer_id}')", flush=True)
    with open(DATA_DIR / "customers.csv", "r") as f:
        for row in csv.DictReader(f):
            if row["customer_id"] == customer_id:
                return (f"{row['first_name']} {row['last_name']}, email {row['email']}, "
                        f"registered {row['registration_date']}, {row['city']}, {row['state']}, "
                        f"risk score {row['risk_score']}")
    return f"Customer {customer_id} not found."


# ============================================================================
# AGENT SETUP AND INVESTIGATION
# ============================================================================

async def main():
    print("=" * 70)
    print("SEARCH+RAG MODULE 04 — Agent Document Investigation")
    print("=" * 70)
    print()

    client = FoundryChatClient(
        project_endpoint=FOUNDRY_ENDPOINT,
        model=MODEL,
        credential=AzureCliCredential(),
    )

    agent = Agent(
        client=client,
        name="DocumentInvestigator",
        instructions=(
            "You are a senior fraud investigator at a payment processing company. "
            "You have access to TWO types of data sources:\n\n"
            "1. DOCUMENT EVIDENCE (via search_fraud_evidence, get_document_content):\n"
            "   Internal documents including SARs, compliance memos, audit logs,\n"
            "   due diligence files, security alerts, and risk reports.\n\n"
            "2. TRANSACTION DATA (via query_transactions, lookup_customer):\n"
            "   The structured transaction ledger and customer profiles.\n\n"
            "INVESTIGATION APPROACH:\n"
            "- Start by searching documents for relevant evidence\n"
            "- Cross-reference document findings with transaction data\n"
            "- Look for connections between documents (shared IDs, dates, patterns)\n"
            "- Build a complete picture by combining both data sources\n"
            "- Cite specific document names and IDs as evidence\n\n"
            "Be thorough but concise. Cite your sources."
        ),
        tools=[
            search_fraud_evidence,
            get_document_content,
            query_transactions,
            lookup_customer,
        ],
    )

    session = agent.create_session()

    # ================================================================
    # Helper: run a phase with streaming output
    # ================================================================
    async def run_phase(phase_name: str, query: str):
        """Run an investigation phase with streaming output."""
        print("─" * 70)
        print(f"{phase_name}")
        print("─" * 70)
        print()
        print("  ┌─ Agent ", end="", flush=True)
        full_text = []
        async for chunk in agent.run(query, session=session, stream=True):
            if chunk.text:
                full_text.append(chunk.text)
                for char in chunk.text:
                    if char == "\n":
                        print()
                        print("  │ ", end="", flush=True)
                    else:
                        print(char, end="", flush=True)
        print()
        print("  └─────────────────────────────────────────────────────────────")
        print()
        return "".join(full_text)

    # ================================================================
    # Investigation Phase 1: Document Discovery
    # ================================================================
    await run_phase(
        "PHASE 1: Document Discovery",
        "Search the fraud evidence documents for any mentions of 'Apex' merchants "
        "or structuring below the BSA threshold. What documents do we have and "
        "what do they tell us?",
    )

    # ================================================================
    # Investigation Phase 2: Cross-Reference
    # ================================================================
    await run_phase(
        "PHASE 2: Cross-Reference Documents with Transaction Data",
        "Now cross-reference the merchant IDs and customer IDs from the documents "
        "with the actual transaction data. Query transactions for the merchants "
        "mentioned in the SARs. Do the transaction patterns confirm what the "
        "documents describe?",
    )

    # ================================================================
    # Investigation Phase 3: Evidence Synthesis
    # ================================================================
    await run_phase(
        "PHASE 3: Evidence Synthesis",
        "Search for documents about device fingerprints, impossible travel, and "
        "customer onboarding anomalies. Then look up the specific customer IDs "
        "mentioned in those alerts. Build a unified evidence matrix:\n"
        "- Which accounts appear in MULTIPLE documents?\n"
        "- Which evidence types are strongest (highest confidence)?\n"
        "- What's the total estimated financial exposure?\n"
        "- What actions have been recommended across all documents?",
    )

    # ================================================================
    # Investigation Phase 4: Final Report
    # ================================================================
    report = await run_phase(
        "PHASE 4: Final Report",
        "Write a FINAL INVESTIGATION BRIEF that synthesizes ALL evidence from "
        "both document sources and transaction data. Structure it as:\n\n"
        "1. EXECUTIVE SUMMARY (2 sentences)\n"
        "2. EVIDENCE INVENTORY (which documents support which findings)\n"
        "3. FRAUD RING MEMBERS (customer IDs with evidence citations)\n"
        "4. SHELL MERCHANTS (merchant IDs with evidence citations)\n"
        "5. FINANCIAL EXPOSURE\n"
        "6. RECOMMENDED ACTIONS (consolidated from all documents)\n\n"
        "Cite specific document filenames for every claim.",
    )

    # Save the report
    report_path = Path(__file__).parent / "data" / "investigation_brief.md"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  ✓ Report saved: {report_path}")

    print()
    print("=" * 70)
    print("MODULE 04 — Summary")
    print("=" * 70)
    print("""
  The RAG Pattern in Action:
    ├─ RETRIEVE: Agent searched AI Search index for relevant documents
    ├─ AUGMENT: Combined document evidence with transaction data
    └─ GENERATE: Synthesized a unified investigation brief

  Document evidence STRENGTHENS the case because:
    ├─ SARs provide human analyst assessments and context
    ├─ Audit logs show what the rules engine missed and why
    ├─ Security alerts add device/geolocation signals
    └─ Compliance memos reveal the investigation timeline

  NEXT: Run 05_evaluation.py
""")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
