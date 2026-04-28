"""
============================================================================
AGENTS MODULE 04 — MAF: Deep Multi-Turn Investigation
============================================================================

WORKSHOP NARRATIVE:
    Module 02 introduced MAF agents with tools. Now we use MAF's session
    system for a deep, multi-turn investigation where the agent
    progressively builds a case — each turn adding evidence from
    different data sources, referencing findings from previous turns.

    This module also demonstrates STREAMING responses — watching the
    agent think and act in real-time as tokens arrive.

LEARNING OBJECTIVES:
    1. Deep multi-turn investigation patterns with AgentSession
    2. Streaming agent responses for real-time observation
    3. Progressive evidence accumulation across turns
    4. Agent-driven investigation (agent decides next steps)

ESTIMATED TIME: 20-25 minutes

============================================================================
"""

import os
import sys
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
    query_transactions,
    lookup_customer,
    check_merchant,
    find_similar_merchants,
    analyze_account_network,
    check_device_fingerprints,
)

PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT")
MODEL = os.getenv("FOUNDRY_MODEL", "gpt-4o")

if not PROJECT_ENDPOINT:
    print("ERROR: Set FOUNDRY_PROJECT_ENDPOINT in your .env file.")
    sys.exit(1)


async def main():
    print("=" * 70)
    print("AGENTS MODULE 04 — MAF Deep Multi-Turn Investigation")
    print("=" * 70)
    print()

    # ================================================================
    # Create a highly specialized investigation agent
    # ================================================================
    client = FoundryChatClient(
        project_endpoint=PROJECT_ENDPOINT,
        model=MODEL,
        credential=AzureCliCredential(),
    )

    agent = Agent(
        client=client,
        name="DeepInvestigator",
        instructions=(
            "You are the lead fraud investigator. You conduct systematic, "
            "multi-phase investigations. In each phase:\n\n"
            "1. State your hypothesis clearly\n"
            "2. Explain which tools you're calling and why\n"
            "3. Analyze the results against your hypothesis\n"
            "4. Update your hypothesis based on evidence\n"
            "5. Propose next steps\n\n"
            "You are building a CASE FILE. Reference evidence from prior turns "
            "by citing specific transaction IDs, customer IDs, and merchant IDs.\n\n"
            "CRITICAL: Think like a prosecutor. Every claim needs evidence. "
            "Every connection needs proof. Distinguish between confirmed facts "
            "and circumstantial evidence."
        ),
        tools=[
            query_transactions,
            lookup_customer,
            check_merchant,
            find_similar_merchants,
            analyze_account_network,
            check_device_fingerprints,
        ],
    )

    session = agent.create_session()

    # ================================================================
    # Phase 1: Initial Lead (User provides starting point)
    # ================================================================
    print("─" * 70)
    print("PHASE 1: Initial Lead")
    print("─" * 70)
    print()
    print("  Starting with a tip: search for merchants with 'Apex' in the name.")
    print("  The agent will investigate and decide its own next steps.")
    print()

    result = await agent.run(
        "I received a tip about suspicious merchants with 'Apex' in their name. "
        "Start by finding these merchants, then investigate their transaction "
        "patterns. Look for anything unusual — amounts, timing, connected accounts.",
        session=session,
    )
    _print_result("Phase 1", result)

    # ================================================================
    # Phase 2: Follow the Money (Agent-directed)
    # ================================================================
    print("─" * 70)
    print("PHASE 2: Follow the Money")
    print("─" * 70)
    print()

    result = await agent.run(
        "Good findings. Now follow the money: for the top senders you identified, "
        "analyze their full transaction networks. Look for circular flows where "
        "money moves A→B→C and eventually returns to A. Also check if these "
        "accounts share any device fingerprints.",
        session=session,
    )
    _print_result("Phase 2", result)

    # ================================================================
    # Phase 3: Identity Analysis
    # ================================================================
    print("─" * 70)
    print("PHASE 3: Identity Analysis")
    print("─" * 70)
    print()

    result = await agent.run(
        "Now look at the PEOPLE behind these accounts. For each suspicious "
        "customer ID, pull their profile. Check:\n"
        "- When did they register? Any clustering?\n"
        "- What email domains do they use?\n"
        "- Do their phone area codes match their cities?\n"
        "- What are their risk scores? Are they suspiciously LOW?",
        session=session,
    )
    _print_result("Phase 3", result)

    # ================================================================
    # Phase 4: Streaming — Case Summary
    # ================================================================
    # Here we demonstrate STREAMING. Instead of waiting for the full
    # response, we print tokens as they arrive.
    #
    # WORKSHOP NOTE:
    # "Streaming is essential for long investigations. Without it, the
    # user waits 30-60 seconds for a complex response. With streaming,
    # they see the agent thinking in real-time."
    # ================================================================
    print("─" * 70)
    print("PHASE 4: Case Summary (Streaming)")
    print("─" * 70)
    print()
    print("  Requesting final case summary with streaming enabled...")
    print()

    print("  ┌─ Agent (streaming) ──────────────────────────────────────────")
    line_count = 0
    async for chunk in agent.run(
        "Write the FINAL CASE SUMMARY. Structure it as:\n\n"
        "## Fraud Ring Identification\n"
        "- List ALL suspected ring members with customer IDs\n"
        "- List ALL suspected shell merchants with merchant IDs\n\n"
        "## Evidence Matrix\n"
        "For each member, list which signals apply:\n"
        "- Smurfing (transactions $9,200-$9,800)\n"
        "- Shell merchant connections\n"
        "- Device fingerprint sharing\n"
        "- Registration date clustering\n"
        "- Suspicious email domains\n\n"
        "## Confidence Assessment\n"
        "Rate your overall confidence and cite the strongest evidence.\n\n"
        "## Recommended Actions\n"
        "What should the compliance team do next?",
        session=session,
        stream=True,
    ):
        if chunk.text:
            for char in chunk.text:
                if char == "\n":
                    line_count += 1
                    if line_count <= 40:
                        print()
                        print("  │ ", end="", flush=True)
                else:
                    if line_count <= 40:
                        print(char, end="", flush=True)

    if line_count > 40:
        print()
        print(f"  │ ... (truncated, {line_count} total lines)")
    print()
    print("  └────────────────────────────────────────────────────────────")
    print()

    # ================================================================
    # Summary
    # ================================================================
    print("=" * 70)
    print("AGENTS MODULE 04 — Summary")
    print("=" * 70)
    print("""
  What You Built:
    ├─ 4-phase progressive investigation using AgentSession
    ├─ Agent-directed investigation (agent chose which tools to call)
    ├─ Streaming response for the final case summary
    └─ Evidence accumulation across all turns

  KEY TAKEAWAY:
  Multi-turn sessions let the agent build institutional knowledge
  within a conversation. Each turn references prior findings,
  creating a cohesive investigation narrative.

  NEXT: Run 05_maf_workflow_orchestration.py
""")
    print("=" * 70)


def _print_result(phase: str, result):
    """Print a non-streaming result."""
    text = str(result)
    print(f"  ┌─ {phase} Response ──────────────────────────────────────────")
    for line in text.split("\n")[:25]:
        print(f"  │ {line}")
    if len(text.split("\n")) > 25:
        print(f"  │ ... ({len(text.split(chr(10)))} lines)")
    print("  └────────────────────────────────────────────────────────────")
    print()


if __name__ == "__main__":
    asyncio.run(main())
