"""
============================================================================
SEARCH+RAG MODULE 05 — Evaluation
============================================================================

WORKSHOP NARRATIVE:
    Does the document-powered agent find the fraud ring more accurately
    than the raw LLM or agent-only approaches? We compare against
    ground truth — same precision/recall/F1 framework from Module 05
    of the LLM track.

ESTIMATED TIME: 10-15 minutes

============================================================================
"""

import os
import sys
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv
_dir = Path(__file__).parent
load_dotenv(_dir / ".env")
load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_ground_truth():
    with open(DATA_DIR / "ground_truth.json", "r") as f:
        return json.load(f)


def compute_metrics(actual: set, predicted: set, label: str) -> dict:
    tp = len(actual & predicted)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "label": label, "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "precision": round(precision, 4),
        "recall": round(recall, 4), "f1_score": round(f1, 4),
    }


async def run_rag_investigation():
    """Run a fresh RAG investigation and extract predicted IDs."""
    from agent_framework import Agent, tool
    from agent_framework.foundry import FoundryChatClient
    from azure.identity import AzureCliCredential
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.models import VectorizedQuery
    from openai import AzureOpenAI
    from typing import Annotated
    from pydantic import Field
    import csv

    SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
    SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
    INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "fraud-evidence")
    USE_MI = os.getenv("USE_MANAGED_IDENTITY", "false").lower() in ("true", "1")

    def _search_client():
        if USE_MI:
            from azure.identity import DefaultAzureCredential
            return SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=DefaultAzureCredential())
        return SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=AzureKeyCredential(SEARCH_KEY))

    sc = _search_client()

    @tool(approval_mode="never_require")
    def search_documents(query: Annotated[str, Field(description="Search query")]) -> str:
        results = sc.search(search_text=query, top=5,
                           select=["filename", "summary", "merchant_ids", "customer_ids", "risk_indicators"])
        lines = []
        for r in results:
            lines.append(f"{r['filename']}: {r.get('summary', '')[:150]}")
            if r.get("merchant_ids"):
                lines.append(f"  Merchants: {', '.join(r['merchant_ids'])}")
            if r.get("customer_ids"):
                lines.append(f"  Customers: {', '.join(r['customer_ids'][:8])}")
        return "\n".join(lines) if lines else "No results."

    @tool(approval_mode="never_require")
    def query_txns(min_amount: Annotated[float | None, Field(default=None)] = None,
                   max_amount: Annotated[float | None, Field(default=None)] = None) -> str:
        count = 0
        with open(DATA_DIR / "transactions.csv", "r") as f:
            for row in csv.DictReader(f):
                amt = float(row["amount"])
                if min_amount and amt < min_amount: continue
                if max_amount and amt > max_amount: continue
                count += 1
        return f"{count} transactions found."

    client = FoundryChatClient(
        project_endpoint=os.getenv("FOUNDRY_PROJECT_ENDPOINT"),
        model=os.getenv("FOUNDRY_MODEL", "gpt-4o"),
        credential=AzureCliCredential(),
    )

    agent = Agent(
        client=client, name="EvalRAGAgent",
        instructions=(
            "You investigate fraud by searching internal documents and transaction data. "
            "Find ALL fraud ring members (CUS-XXX) and shell merchants (MER-XXX). "
            "Return ONLY valid JSON: {\"ring_members\": [...], \"shell_merchants\": [...]}"
        ),
        tools=[search_documents, query_txns],
    )

    result = await agent.run(
        "Search all documents for fraud ring evidence. Find every customer ID "
        "and merchant ID associated with the ring. Return as JSON."
    )

    text = str(result)
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass
    return {"ring_members": [], "shell_merchants": []}


async def main():
    print("=" * 70)
    print("SEARCH+RAG MODULE 05 — Evaluation")
    print("=" * 70)
    print()

    gt = load_ground_truth()
    actual_members = set(gt["fraud_ring"]["member_customer_ids"])
    actual_shells = set(gt["fraud_ring"]["shell_merchant_ids"])

    print(f"  Ground Truth: {len(actual_members)} members, {len(actual_shells)} merchants")
    print()
    print("  Running RAG-powered investigation...")

    predictions = await run_rag_investigation()

    predicted_members = set(predictions.get("ring_members", []))
    predicted_shells = set(predictions.get("shell_merchants", []))

    print(f"\n  Predictions: {len(predicted_members)} members, {len(predicted_shells)} merchants")
    print()

    member_m = compute_metrics(actual_members, predicted_members, "Ring Members")
    merchant_m = compute_metrics(actual_shells, predicted_shells, "Shell Merchants")
    combined_m = compute_metrics(actual_members | actual_shells, predicted_members | predicted_shells, "Combined")

    print("  {:<20} {:>10} {:>10} {:>10} {:>6} {:>6} {:>6}".format(
        "Category", "Precision", "Recall", "F1", "TP", "FP", "FN"))
    print("  " + "─" * 70)
    for m in [member_m, merchant_m, combined_m]:
        print("  {:<20} {:>10.1%} {:>10.1%} {:>10.1%} {:>6} {:>6} {:>6}".format(
            m["label"], m["precision"], m["recall"], m["f1_score"],
            m["true_positives"], m["false_positives"], m["false_negatives"]))

    print()
    print("=" * 70)
    print("MODULE 05 — Summary")
    print("=" * 70)
    print(f"""
  RAG-Powered Detection:
    ├─ Combined F1: {combined_m['f1_score']:.1%}
    ├─ Members found: {member_m['true_positives']}/{len(actual_members)}
    └─ Merchants found: {merchant_m['true_positives']}/{len(actual_shells)}

  Compare with other tracks:
    ├─ LLM Track (Module 05): Raw LLM from data summaries
    ├─ Agents Track (Module 06): MAF agents with @tool functions
    └─ Search+RAG (this): Document evidence + transaction data

  RAG typically achieves higher recall because the documents explicitly
  name fraud ring members and merchants — the evidence is PRE-ANALYZED
  by human investigators (the synthetic document authors).

  WORKSHOP COMPLETE for the Search + RAG track!
""")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
