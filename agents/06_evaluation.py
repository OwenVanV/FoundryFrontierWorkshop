"""
============================================================================
AGENTS MODULE 06 — Evaluation: Agents vs. Ground Truth
============================================================================

WORKSHOP NARRATIVE:
    The agents have produced their findings. But how accurate are they?
    This module compares agent-identified fraud ring members and shell
    merchants against the ground truth, computing precision, recall,
    and F1 — the same framework from Module 05 of the main workshop.

    This lets participants see how agent-based detection compares to
    the raw LLM approach from the earlier modules.

ESTIMATED TIME: 15-20 minutes

============================================================================
"""

import os
import sys
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv
_agents_dir = Path(__file__).parent
load_dotenv(_agents_dir / ".env")
load_dotenv()

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential

sys.path.insert(0, str(Path(__file__).parent))
from utils.fraud_tools import (
    query_transactions, lookup_customer, check_merchant,
    find_similar_merchants, analyze_account_network, check_device_fingerprints,
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_ground_truth() -> dict:
    """Load the ground truth answer key."""
    gt_path = DATA_DIR / "ground_truth.json"
    if not gt_path.exists():
        print("  ✗ ground_truth.json not found. Run data/generate_synthetic_data.py.")
        sys.exit(1)
    with open(gt_path, "r") as f:
        return json.load(f)


def compute_metrics(actual: set, predicted: set, label: str) -> dict:
    """Compute precision, recall, F1."""
    tp = len(actual & predicted)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "label": label,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }


async def run_fresh_investigation() -> dict:
    """Run a quick single-shot investigation to get agent predictions."""
    client = FoundryChatClient(
        project_endpoint=os.getenv("FOUNDRY_PROJECT_ENDPOINT"),
        model=os.getenv("FOUNDRY_MODEL", "gpt-4o"),
        credential=AzureCliCredential(),
    )

    agent = Agent(
        client=client,
        name="EvalInvestigator",
        instructions=(
            "You are a fraud investigator. Use your tools to identify ALL members "
            "of a fraud ring and ALL shell merchants. Be thorough — check for:\n"
            "- Smurfing ($9,200-$9,800 range transactions)\n"
            "- Shell merchant clusters (similar names, same state)\n"
            "- Device fingerprint sharing\n"
            "- Registration date clustering\n\n"
            "You MUST return your findings as valid JSON:\n"
            '{"ring_members": ["CUS-..."], "shell_merchants": ["MER-..."]}'
        ),
        tools=[
            query_transactions, lookup_customer, check_merchant,
            find_similar_merchants, analyze_account_network, check_device_fingerprints,
        ],
    )

    result = await agent.run(
        "Conduct a comprehensive fraud investigation. Start by:\n"
        "1. Finding merchants with 'Apex' in the name\n"
        "2. Querying transactions $9,000-$10,000 to find frequent senders\n"
        "3. Checking device fingerprints for the top senders\n"
        "4. Analyzing networks for circular flows\n\n"
        "Return ALL suspected ring members and shell merchants as JSON."
    )

    # Parse JSON from the response
    text = str(result)
    try:
        # Try to extract JSON from the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end])
            return parsed
    except json.JSONDecodeError:
        pass

    return {"ring_members": [], "shell_merchants": []}


async def main():
    print("=" * 70)
    print("AGENTS MODULE 06 — Evaluation: Agents vs. Ground Truth")
    print("=" * 70)
    print()

    # Load ground truth
    gt = load_ground_truth()
    actual_members = set(gt["fraud_ring"]["member_customer_ids"])
    actual_shells = set(gt["fraud_ring"]["shell_merchant_ids"])

    print(f"  Ground Truth:")
    print(f"    Fraud ring members: {len(actual_members)}")
    print(f"    Shell merchants: {len(actual_shells)}")
    print()

    # Run investigation
    print("  Running agent investigation (this may take a minute)...")
    predictions = await run_fresh_investigation()

    predicted_members = set(predictions.get("ring_members", []))
    predicted_shells = set(predictions.get("shell_merchants", []))

    print(f"\n  Agent Predictions:")
    print(f"    Ring members identified: {len(predicted_members)}")
    print(f"    Shell merchants identified: {len(predicted_shells)}")
    print()

    # Compute metrics
    member_m = compute_metrics(actual_members, predicted_members, "Ring Members")
    merchant_m = compute_metrics(actual_shells, predicted_shells, "Shell Merchants")
    all_actual = actual_members | actual_shells
    all_predicted = predicted_members | predicted_shells
    combined_m = compute_metrics(all_actual, all_predicted, "Combined")

    # Display results
    print("─" * 70)
    print("  EVALUATION RESULTS")
    print("─" * 70)
    print()
    print("  {:<20} {:>10} {:>10} {:>10} {:>6} {:>6} {:>6}".format(
        "Category", "Precision", "Recall", "F1", "TP", "FP", "FN"))
    print("  " + "─" * 70)

    for m in [member_m, merchant_m, combined_m]:
        print("  {:<20} {:>10.1%} {:>10.1%} {:>10.1%} {:>6} {:>6} {:>6}".format(
            m["label"],
            m["precision"], m["recall"], m["f1_score"],
            m["true_positives"], m["false_positives"], m["false_negatives"],
        ))

    print()
    print("=" * 70)
    print("AGENTS MODULE 06 — Summary")
    print("=" * 70)
    print(f"""
  Agent Detection Accuracy:
    ├─ Combined F1: {combined_m['f1_score']:.1%}
    ├─ Members found: {member_m['true_positives']}/{len(actual_members)}
    └─ Merchants found: {merchant_m['true_positives']}/{len(actual_shells)}

  Compare this with the raw LLM results from Module 05 of the main
  workshop. Agent-based approaches often achieve higher recall because
  agents can systematically query data through tools rather than
  reasoning from summaries alone.

  WORKSHOP COMPLETE for the Agents Deep-Dive!
""")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
