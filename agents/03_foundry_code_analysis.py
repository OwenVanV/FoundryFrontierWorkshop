"""
============================================================================
AGENTS MODULE 03 — Foundry Agent Service: Deep Code Analysis
============================================================================

WORKSHOP NARRATIVE:
    Back to Foundry Agent Service. This time we push Code Interpreter
    harder — uploading ALL datasets and asking the agent to build
    network graphs, detect circular flows, and generate comprehensive
    visualizations.

    The key difference from Module 01: we're now asking the agent to
    perform MULTI-STEP analysis within a single conversation — building
    on its own code artifacts across turns.

LEARNING OBJECTIVES:
    1. Multi-step code execution within a single agent conversation
    2. Agent-generated visualizations (network graphs, heatmaps)
    3. Cross-dataset analysis in the sandbox (joining CSVs)
    4. Downloading and reviewing generated artifacts

ESTIMATED TIME: 25-30 minutes

============================================================================
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
_agents_dir = Path(__file__).parent
load_dotenv(_agents_dir / ".env")
load_dotenv()

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition,
    CodeInterpreterTool,
    AutoCodeInterpreterToolParam,
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT")

if not PROJECT_ENDPOINT:
    print("ERROR: Set FOUNDRY_PROJECT_ENDPOINT in your .env file.")
    sys.exit(1)


def main():
    print("=" * 70)
    print("AGENTS MODULE 03 — Foundry Code Interpreter Deep Analysis")
    print("=" * 70)
    print()

    # ================================================================
    # Connect and upload ALL datasets
    # ================================================================
    project = AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
    )
    openai_client = project.get_openai_client()

    # Upload all data files for comprehensive analysis
    all_files = [
        "transactions.csv", "customers.csv", "merchants.csv",
        "velocity_metrics.csv",
    ]

    file_ids = []
    for filename in all_files:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"  ✗ {filename} not found. Run data/generate_synthetic_data.py first.")
            sys.exit(1)
        print(f"  Uploading {filename}...")
        with open(filepath, "rb") as f:
            file_obj = openai_client.files.create(purpose="assistants", file=f)
        file_ids.append(file_obj.id)
        print(f"  ✓ {file_obj.id}")

    print()

    # ================================================================
    # Create the deep analysis agent
    # ================================================================
    agent = project.agents.create_version(
        agent_name="FraudNetworkAnalyst",
        definition=PromptAgentDefinition(
            model=os.getenv("FOUNDRY_MODEL", "gpt-4o"),
            instructions=(
                "You are an expert financial crimes network analyst. You have access to "
                "payment transaction data, customer profiles, merchant records, and "
                "time-series velocity metrics.\n\n"
                "ANALYSIS APPROACH:\n"
                "1. Load all datasets and join them as needed.\n"
                "2. Build transaction network graphs (sender→receiver adjacency).\n"
                "3. Detect circular money flows (A→B→C→D→A patterns).\n"
                "4. Identify bidirectional transaction pairs.\n"
                "5. Cross-reference with customer registration patterns.\n"
                "6. Generate publication-quality visualizations.\n\n"
                "Always write efficient pandas code. Use matplotlib for charts. "
                "Save generated charts as PNG files so they can be downloaded."
            ),
            tools=[
                CodeInterpreterTool(
                    container=AutoCodeInterpreterToolParam(file_ids=file_ids)
                ),
            ],
        ),
        description="Network analyst for deep fraud investigation.",
    )

    print(f"  ✓ Agent: {agent.name} (v{agent.version})")
    print()

    # ================================================================
    # Multi-step analysis conversation
    # ================================================================
    conversation = openai_client.conversations.create()

    analyses = [
        # Step 1: Build the transaction network
        (
            "Network Construction",
            "Load transactions.csv and build a directed transaction network. "
            "Create an adjacency count matrix: for each (sender, receiver) pair, "
            "count the number of transactions and total amount. Find all pairs "
            "where BOTH directions exist (A sends to B AND B sends to A). "
            "List the top 20 bidirectional pairs by total volume. "
            "Generate a bar chart showing bidirectional pair volumes."
        ),

        # Step 2: Detect circular flows
        (
            "Circular Flow Detection",
            "From the bidirectional pairs identified, look for longer cycles: "
            "chains where A→B→C→D→A. Focus on high-value transactions ($5K+). "
            "For each cycle found, calculate the total money that flowed through "
            "the cycle and the time span. Create a network visualization showing "
            "the circular flows with edge weights."
        ),

        # Step 3: Cross-reference with customer data
        (
            "Customer Cross-Reference",
            "Load customers.csv and join with the suspicious accounts from the "
            "network analysis. For the accounts involved in bidirectional flows "
            "and circular patterns, analyze:\n"
            "1. Registration date distribution — any clustering?\n"
            "2. Email domains — any unusual concentrations?\n"
            "3. Geographic distribution — do they cluster geographically?\n"
            "4. Risk scores — are they below the alert threshold?\n"
            "Create charts for each dimension."
        ),

        # Step 4: Time-series anomaly detection
        (
            "Temporal Pattern Analysis",
            "Load velocity_metrics.csv. Focus on the 'high_value_count' and "
            "'high_value_total_usd' columns. Look for time windows where "
            "high-value transaction counts spike above the mean + 2*std. "
            "Cross-reference these spike windows with the suspicious accounts "
            "from the network analysis — do the suspicious accounts' transactions "
            "cluster in the same time windows? Create a time-series chart with "
            "highlighted anomaly windows."
        ),

        # Step 5: Final synthesis
        (
            "Investigation Report",
            "Synthesize ALL findings into a structured report:\n"
            "1. List all suspected fraud ring members (customer IDs) with evidence\n"
            "2. List all suspected shell merchants (merchant IDs) with evidence\n"
            "3. Estimated total financial exposure\n"
            "4. Network diagram showing the ring structure\n"
            "5. Confidence assessment for each finding\n"
            "Save the network diagram as a PNG file."
        ),
    ]

    for i, (label, query) in enumerate(analyses, 1):
        print(f"  ── Analysis {i}/{len(analyses)}: {label} ──")
        print(f"  {query[:80]}...")
        print()

        response = openai_client.responses.create(
            conversation=conversation.id,
            input=query,
            extra_body={
                "agent_reference": {
                    "name": agent.name,
                    "type": "agent_reference",
                }
            },
        )

        # Print response
        last_msg = response.output[-1] if response.output else None
        if last_msg and last_msg.type == "message" and last_msg.content:
            for part in last_msg.content:
                text = getattr(part, "text", None)
                if text:
                    text_str = text if isinstance(text, str) else getattr(text, "value", str(text))
                    print("  ┌─ Agent ───────────────────────────────────────────────────")
                    for line in str(text_str).split("\n")[:20]:
                        print(f"  │ {line}")
                    if len(str(text_str).split("\n")) > 20:
                        print(f"  │ ... ({len(str(text_str).split(chr(10)))} lines)")
                    print("  └────────────────────────────────────────────────────────────")

                # Download generated files
                annotations = getattr(text, "annotations", None) if text else None
                if annotations:
                    for ann in annotations:
                        if getattr(ann, "type", "") == "container_file_citation":
                            print(f"\n  📊 Generated: {ann.filename}")
                            try:
                                content = openai_client.containers.files.content.retrieve(
                                    file_id=ann.file_id,
                                    container_id=ann.container_id,
                                )
                                out_path = Path(__file__).parent / ann.filename
                                with open(out_path, "wb") as f:
                                    f.write(content.read())
                                print(f"  ✓ Saved: {out_path}")
                            except Exception as e:
                                print(f"  ⚠ Download failed: {e}")
        print()

    # ================================================================
    # Cleanup
    # ================================================================
    print("=" * 70)
    print("AGENTS MODULE 03 — Summary")
    print("=" * 70)
    print("""
  What Happened:
    ├─ Code Interpreter executed 5 multi-step analysis phases
    ├─ Built transaction network graphs from 50K+ transactions
    ├─ Detected circular money flows and bidirectional pairs
    ├─ Cross-referenced network with customer demographics
    └─ Generated time-series anomaly visualizations

  KEY TAKEAWAY:
  Code Interpreter shines when you need the agent to perform complex
  data analysis autonomously. It writes pandas code, executes it,
  iterates on errors, and generates artifacts — all in a sandbox.

  NEXT: Run 04_maf_multi_turn_investigation.py
""")
    print("=" * 70)

    try:
        project.agents.delete_version(agent_name=agent.name, agent_version=agent.version)
        print(f"  ✓ Cleaned up: {agent.name} v{agent.version}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
