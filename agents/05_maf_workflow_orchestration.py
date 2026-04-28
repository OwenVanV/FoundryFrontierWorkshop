"""
============================================================================
AGENTS MODULE 05 — MAF: Workflow Orchestration (Multi-Agent)
============================================================================

WORKSHOP NARRATIVE:
    This is the culmination of the agents deep-dive. Instead of one
    agent doing everything, we create SPECIALIST agents and compose
    them in a WORKFLOW — the MAF functional workflow pattern.

    Three agents, each with their own tools and expertise:
    1. TransactionAnalyst — finds suspicious transaction patterns
    2. IdentityInvestigator — cross-references customer profiles
    3. ReportWriter — synthesizes findings into a formal report

    A workflow function orchestrates them sequentially, passing
    findings from one agent to the next.

LEARNING OBJECTIVES:
    1. MAF functional workflow pattern (async def as workflow)
    2. Multi-agent composition with specialized roles
    3. Data passing between agents in a workflow
    4. Comparing single-agent vs. multi-agent accuracy

ESTIMATED TIME: 25-30 minutes

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

from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential
from typing import Annotated
from pydantic import Field

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


# ============================================================================
# AGENT FACTORY — Create specialist agents
# ============================================================================
# Each agent has a focused role and only the tools it needs.
# This is a production pattern: agents with fewer, relevant tools
# perform better than agents overloaded with every tool.
#
# WORKSHOP DISCUSSION POINT:
# "Why not give one agent all the tools? Because:
# 1. Tool descriptions consume tokens — fewer tools = cheaper
# 2. Focused instructions improve reasoning accuracy
# 3. Specialized agents are easier to evaluate independently
# 4. You can upgrade one agent without affecting others"
# ============================================================================

def create_agents() -> dict:
    """Create the three specialist agents."""
    client = FoundryChatClient(
        project_endpoint=PROJECT_ENDPOINT,
        model=MODEL,
        credential=AzureCliCredential(),
    )

    # Agent 1: Transaction Analyst
    # Tools: query_transactions, find_similar_merchants
    transaction_analyst = Agent(
        client=client,
        name="TransactionAnalyst",
        instructions=(
            "You are a transaction pattern specialist. Your ONLY job is to:\n"
            "1. Find suspicious transaction patterns (smurfing, velocity anomalies)\n"
            "2. Identify merchant clusters with similar names or registration patterns\n"
            "3. Output a structured JSON summary of findings\n\n"
            "OUTPUT FORMAT (you MUST return valid JSON):\n"
            '{"suspicious_senders": ["CUS-..."], "suspicious_merchants": ["MER-..."], '
            '"patterns_found": [{"type": "smurfing|merchant_cluster|...", "details": "..."}]}'
        ),
        tools=[query_transactions, find_similar_merchants, check_merchant],
    )

    # Agent 2: Identity Investigator
    # Tools: lookup_customer, analyze_account_network, check_device_fingerprints
    identity_investigator = Agent(
        client=client,
        name="IdentityInvestigator",
        instructions=(
            "You are an identity and network specialist. You receive a list of "
            "suspicious account IDs from the Transaction Analyst. Your job:\n"
            "1. Look up customer profiles for each suspicious account\n"
            "2. Analyze their transaction networks for connections\n"
            "3. Check device fingerprints for shared devices\n"
            "4. Identify synthetic identity signals (registration clustering, "
            "   email domain patterns, geographic mismatches)\n\n"
            "OUTPUT FORMAT (you MUST return valid JSON):\n"
            '{"confirmed_ring_members": ["CUS-..."], '
            '"device_sharing_pairs": [["CUS-A", "CUS-B"]], '
            '"identity_signals": [{"customer_id": "CUS-...", "signals": ["..."]}], '
            '"network_connections": [{"from": "CUS-A", "to": "CUS-B", "evidence": "..."}]}'
        ),
        tools=[lookup_customer, analyze_account_network, check_device_fingerprints],
    )

    # Agent 3: Report Writer
    # No tools — this agent only synthesizes and writes
    report_writer = Agent(
        client=client,
        name="ReportWriter",
        instructions=(
            "You are a senior compliance report writer. You receive findings from "
            "two specialist agents (Transaction Analyst and Identity Investigator). "
            "Your job is to synthesize their findings into a formal investigation "
            "report suitable for regulatory review.\n\n"
            "REPORT STRUCTURE:\n"
            "# FRAUD RING INVESTIGATION REPORT\n"
            "## Executive Summary (2-3 sentences)\n"
            "## Ring Members (table: ID, Name, Evidence, Confidence)\n"
            "## Shell Merchants (table: ID, Name, Evidence)\n"
            "## Money Flow Pattern (description)\n"
            "## Evidence Matrix (which signals apply to which members)\n"
            "## Recommended Actions (immediate, short-term, long-term)\n"
            "## Confidence Assessment\n\n"
            "Be precise. Cite specific IDs and amounts."
        ),
        tools=[],  # No tools — writing only
    )

    return {
        "transaction_analyst": transaction_analyst,
        "identity_investigator": identity_investigator,
        "report_writer": report_writer,
    }


# ============================================================================
# WORKFLOW: Sequential multi-agent investigation
# ============================================================================
# This is the MAF functional workflow pattern: a plain async function
# that calls agents in sequence, passing data between them.
#
# WORKSHOP KEY INSIGHT:
# "A workflow is just an async function. There's no special DSL or
# configuration — it's Python. This means you can use if/else,
# loops, error handling, and any Python library in your workflow."
# ============================================================================

async def fraud_investigation_workflow(agents: dict) -> str:
    """
    Multi-agent fraud investigation workflow.

    Flow: TransactionAnalyst → IdentityInvestigator → ReportWriter
    Each agent's output feeds into the next agent's input.
    """
    print("─" * 70)
    print("WORKFLOW: Multi-Agent Fraud Investigation")
    print("─" * 70)
    print()

    # ── Phase 1: Transaction Analysis ──
    print("  ┌─ Phase 1: TransactionAnalyst ─────────────────────────────────")
    print("  │ Task: Find smurfing patterns and suspicious merchants")
    print("  │ Tools: query_transactions, find_similar_merchants, check_merchant")
    print("  │")

    tx_result = await agents["transaction_analyst"].run(
        "Investigate the transaction ledger for fraud signals:\n"
        "1. Search for transactions between $9,000 and $10,000 — this is the "
        "   BSA reporting threshold zone. Which senders appear most?\n"
        "2. Search for merchants with 'Apex' in the name. Are there clusters?\n"
        "3. Check each Apex merchant's profile for registration patterns.\n"
        "Return your findings as JSON."
    )

    tx_text = str(tx_result)
    print(f"  │ Result: {len(tx_text)} chars")
    for line in tx_text.split("\n")[:10]:
        print(f"  │   {line[:70]}")
    print("  └────────────────────────────────────────────────────────────────")
    print()

    # ── Phase 2: Identity Investigation ──
    print("  ┌─ Phase 2: IdentityInvestigator ───────────────────────────────")
    print("  │ Task: Cross-reference accounts from Phase 1")
    print("  │ Tools: lookup_customer, analyze_account_network, check_device_fingerprints")
    print("  │")

    id_result = await agents["identity_investigator"].run(
        f"The Transaction Analyst found these suspicious entities:\n\n"
        f"{tx_text}\n\n"
        f"For each suspicious sender ID, investigate:\n"
        f"1. Look up their customer profile\n"
        f"2. Analyze their transaction network for bidirectional flows\n"
        f"3. Check device fingerprints for sharing\n"
        f"Return your findings as JSON."
    )

    id_text = str(id_result)
    print(f"  │ Result: {len(id_text)} chars")
    for line in id_text.split("\n")[:10]:
        print(f"  │   {line[:70]}")
    print("  └────────────────────────────────────────────────────────────────")
    print()

    # ── Phase 3: Report Generation ──
    print("  ┌─ Phase 3: ReportWriter ────────────────────────────────────────")
    print("  │ Task: Synthesize findings into a formal report")
    print("  │ Tools: none (writing only)")
    print("  │")

    report_result = await agents["report_writer"].run(
        f"Write the formal investigation report based on these findings:\n\n"
        f"TRANSACTION ANALYSIS:\n{tx_text}\n\n"
        f"IDENTITY INVESTIGATION:\n{id_text}\n\n"
        f"Synthesize into a comprehensive, regulator-ready report."
    )

    report_text = str(report_result)
    print(f"  │ Report: {len(report_text)} chars")
    print("  └────────────────────────────────────────────────────────────────")
    print()

    # Display the report
    print("  ╔══════════════════════════════════════════════════════════════════")
    print("  ║ INVESTIGATION REPORT")
    print("  ╠══════════════════════════════════════════════════════════════════")
    for line in report_text.split("\n")[:50]:
        print(f"  ║ {line}")
    if len(report_text.split("\n")) > 50:
        print(f"  ║ ... ({len(report_text.split(chr(10)))} total lines)")
    print("  ╚══════════════════════════════════════════════════════════════════")

    # Save report
    report_path = Path(__file__).parent.parent / "data" / "agent_investigation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  ✓ Report saved: {report_path}")

    return report_text


# ============================================================================
# MAIN
# ============================================================================

async def main():
    print("=" * 70)
    print("AGENTS MODULE 05 — MAF Workflow Orchestration")
    print("=" * 70)
    print()
    print("  Creating 3 specialist agents...")

    agents = create_agents()
    for name, agent in agents.items():
        tool_count = len(agent._tools) if hasattr(agent, '_tools') else 0
        print(f"    ✓ {agent.name} ({tool_count} tools)")
    print()

    report = await fraud_investigation_workflow(agents)

    print()
    print("=" * 70)
    print("AGENTS MODULE 05 — Summary")
    print("=" * 70)
    print("""
  What You Built:
    ├─ 3 specialist agents (Analyst, Investigator, Writer)
    ├─ Sequential workflow: each agent's output feeds the next
    ├─ Data transformation between agents (findings → context)
    └─ Publication-quality investigation report

  Multi-Agent vs. Single-Agent:
    ├─ Specialization: each agent has focused instructions + tools
    ├─ Token efficiency: fewer tools per agent = less prompt overhead
    ├─ Composability: swap agents independently
    └─ Evaluability: test each agent's accuracy separately

  NEXT: Run 06_evaluation.py
""")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
