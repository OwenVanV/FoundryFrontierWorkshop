"""
============================================================================
AGENTS MODULE 02 — Microsoft Agent Framework: Your First MAF Agent
============================================================================

WORKSHOP NARRATIVE:
    Module 01 showed Foundry Agent Service — managed agents with server-side
    tools like Code Interpreter. Now we introduce a DIFFERENT system:
    the Microsoft Agent Framework (MAF).

    MAF is an open-source SDK (pip install agent-framework) for building
    agents that run in YOUR code with YOUR custom tools. Where Foundry
    Agent Service excels at sandboxed code execution, MAF excels at
    custom tool integration, multi-agent orchestration, and workflows.

    Think of it this way:
      Foundry Agent Service = "agent as a service" (managed)
      MAF = "agent as a library" (you control everything)

    In this module, you create a MAF agent with FoundryChatClient (which
    uses Foundry's model endpoint for inference) and attach custom @tool
    functions that query the fraud datasets locally.

LEARNING OBJECTIVES:
    1. Understand MAF's Agent + FoundryChatClient pattern
    2. Create custom tools with the @tool decorator
    3. Run an agent with tools and observe function calling
    4. Use AgentSession for multi-turn conversations

PACKAGES:
    - agent-framework (core)
    - agent-framework-foundry (Foundry provider)

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

# ============================================================================
# MAF IMPORTS
# ============================================================================
# agent_framework is the core package.
# agent_framework.foundry provides FoundryChatClient for Azure Foundry models.
# ============================================================================
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential

# Our custom fraud investigation tools
sys.path.insert(0, str(Path(__file__).parent))
from utils.fraud_tools import (
    query_transactions,
    lookup_customer,
    check_merchant,
    find_similar_merchants,
    analyze_account_network,
    check_device_fingerprints,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT")
MODEL = os.getenv("FOUNDRY_MODEL", "gpt-4o")

if not PROJECT_ENDPOINT:
    print("ERROR: Set FOUNDRY_PROJECT_ENDPOINT in your .env file.")
    sys.exit(1)


# ============================================================================
# STEP 1: Create the FoundryChatClient
# ============================================================================
# FoundryChatClient connects to a Foundry project's model endpoint.
# Unlike Foundry Agent Service (Module 01), the agent logic runs
# locally in YOUR code — Foundry just provides the LLM inference.
#
# WORKSHOP DISCUSSION POINT:
# "FoundryChatClient vs. AIProjectClient — when do you use each?
#  - FoundryChatClient (MAF): Your app owns tools, sessions, orchestration
#  - AIProjectClient (Foundry Agent Service): Server owns agent definition
#  Each has its place. MAF gives you more control; Foundry gives you
#  more managed infrastructure."
# ============================================================================

async def step_1_create_client():
    """Create the MAF client and agent."""
    print("=" * 70)
    print("AGENTS MODULE 02 — Microsoft Agent Framework: First Agent")
    print("=" * 70)
    print()

    print("  Creating FoundryChatClient...")
    client = FoundryChatClient(
        project_endpoint=PROJECT_ENDPOINT,
        model=MODEL,
        credential=AzureCliCredential(),
    )
    print(f"  ✓ Client: FoundryChatClient → {PROJECT_ENDPOINT}")
    print(f"  ✓ Model: {MODEL}")
    print()
    return client


# ============================================================================
# STEP 2: Create an Agent with Custom Tools
# ============================================================================
# The Agent class is MAF's core abstraction. It wraps a chat client
# with instructions, tools, and session management.
#
# The @tool-decorated functions from fraud_tools.py are passed directly
# to the agent. MAF automatically:
#   1. Generates JSON schemas from the function signatures
#   2. Sends them to the LLM as available tools
#   3. Executes the function when the LLM calls it
#   4. Feeds the result back to the LLM
#
# WORKSHOP KEY INSIGHT:
# "Notice we pass Python functions as tools — not JSON schemas.
# The @tool decorator inspects type hints and docstrings to generate
# the schema automatically. This is why Annotated[str, Field(...)]
# annotations matter — they become the tool parameter descriptions
# the LLM sees."
# ============================================================================

async def step_2_create_agent(client) -> Agent:
    """Create a MAF agent with fraud investigation tools."""
    print("─" * 70)
    print("STEP 2: Creating MAF Agent with Custom Tools")
    print("─" * 70)
    print()

    tools = [
        query_transactions,
        lookup_customer,
        check_merchant,
        find_similar_merchants,
        analyze_account_network,
        check_device_fingerprints,
    ]

    agent = Agent(
        client=client,
        name="FraudInvestigator",
        instructions=(
            "You are a senior fraud investigator at a major payment processor. "
            "You have access to tools that query the transaction ledger, customer "
            "profiles, merchant records, and device fingerprint data.\n\n"
            "INVESTIGATION PROTOCOL:\n"
            "1. Start broad: use query_transactions to identify suspicious patterns\n"
            "2. Drill down: use lookup_customer and check_merchant for flagged entities\n"
            "3. Map networks: use analyze_account_network for connected accounts\n"
            "4. Cross-reference: use check_device_fingerprints for device sharing\n"
            "5. Find clusters: use find_similar_merchants for shell company detection\n\n"
            "Always cite specific IDs, amounts, and dates. Show your reasoning."
        ),
        tools=tools,
    )

    print(f"  ✓ Agent: {agent.name}")
    print(f"  ✓ Tools: {len(tools)} custom functions")
    for t in tools:
        fname = getattr(t, '__name__', getattr(t, 'name', str(t)))
        print(f"      • {fname}")
    print()
    return agent


# ============================================================================
# STEP 3: Run Single-Shot Investigations
# ============================================================================
# We start with single-shot queries — one question, one answer.
# The agent will call tools autonomously to answer.
#
# OBSERVE THE TOOL CALLS:
# "Watch the output carefully. The agent doesn't just generate text —
# it decides WHICH tools to call and in WHAT ORDER. For the smurfing
# query, it should call query_transactions with min_amount=9000,
# max_amount=10000. For the merchant query, it should call
# find_similar_merchants with 'Apex'."
# ============================================================================

async def step_3_single_shot(agent: Agent):
    """Run single-shot investigation queries."""
    print("─" * 70)
    print("STEP 3: Single-Shot Investigations")
    print("─" * 70)
    print()

    queries = [
        {
            "label": "3a",
            "query": (
                "Find all transactions between $9,000 and $10,000. "
                "How many are there? Which senders appear most frequently?"
            ),
        },
        {
            "label": "3b",
            "query": (
                "Search for merchants with 'Apex' in their name. "
                "What do they have in common? Are their registration dates suspicious?"
            ),
        },
        {
            "label": "3c",
            "query": (
                "Look up customer CUS-A08563C93C2B and analyze their "
                "transaction network. Who do they send money to and receive from? "
                "Check their device fingerprints too."
            ),
        },
    ]

    for q in queries:
        print(f"  [{q['label']}] Query: {q['query'][:80]}...")
        print()

        result = await agent.run(q["query"])

        print("  ┌─ Agent Response ────────────────────────────────────────────")
        for line in str(result).split("\n")[:25]:
            print(f"  │ {line}")
        if len(str(result).split("\n")) > 25:
            print(f"  │ ... ({len(str(result).split(chr(10)))} lines)")
        print("  └────────────────────────────────────────────────────────────")
        print()


# ============================================================================
# STEP 4: Multi-Turn Investigation with AgentSession
# ============================================================================
# Now we use AgentSession to maintain context across multiple turns.
# Each turn builds on the previous one — just like a real investigation.
#
# WORKSHOP KEY CONCEPT:
# "Sessions manage conversation history locally. The agent remembers
# what was said in previous turns. This is different from Foundry
# Agent Service where conversations are managed server-side."
# ============================================================================

async def step_4_multi_turn(agent: Agent):
    """Run a multi-turn investigation conversation."""
    print("─" * 70)
    print("STEP 4: Multi-Turn Investigation")
    print("─" * 70)
    print()

    # Create a session for conversation persistence
    session = agent.create_session()

    turns = [
        "Search for merchants with 'Apex' or 'ADS' in their name. List all matches with their registration dates and locations.",
        "Now look up the transaction patterns through those merchants. Use query_transactions for each merchant ID you found. What amounts are typical?",
        "For the senders you found in those transactions, check their device fingerprints. Are any devices shared between accounts?",
        "Based on everything you've found, provide a structured summary: which accounts and merchants are likely part of a fraud ring, and what evidence supports this?",
    ]

    for i, turn in enumerate(turns, 1):
        print(f"  ── Turn {i} ──")
        print(f"  User: {turn[:80]}...")
        print()

        result = await agent.run(turn, session=session)

        print("  ┌─ Agent Response ────────────────────────────────────────────")
        for line in str(result).split("\n")[:20]:
            print(f"  │ {line}")
        if len(str(result).split("\n")) > 20:
            print(f"  │ ... ({len(str(result).split(chr(10)))} lines)")
        print("  └────────────────────────────────────────────────────────────")
        print()


# ============================================================================
# STEP 5: Summary
# ============================================================================

async def step_5_summary():
    """Print module summary."""
    print("=" * 70)
    print("AGENTS MODULE 02 — Summary")
    print("=" * 70)
    print("""
  What You Built:
    ├─ A MAF Agent with FoundryChatClient
    ├─ 6 custom @tool functions for fraud data queries
    ├─ Single-shot and multi-turn investigation flows
    └─ Local session management with AgentSession

  MAF vs. Foundry Agent Service:
    ┌──────────────────────────────────────────────────────────────────
    │ Feature              │ Foundry Agent Service │ MAF
    │ Tools                │ Code Interpreter,     │ Custom @tool functions,
    │                      │ Web Search (managed)  │ anything you can code
    │ Agent Location       │ Cloud (managed)       │ Your code (local)
    │ Session Management   │ Server-side           │ Client-side (AgentSession)
    │ Best For             │ Data analysis,        │ Custom integrations,
    │                      │ sandboxed code        │ orchestration, workflows
    └──────────────────────────────────────────────────────────────────

  In Module 03, we return to Foundry Agent Service for a deep-dive into
  Code Interpreter's analytical capabilities.

  NEXT: Run 03_foundry_code_analysis.py
""")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

async def main():
    client = await step_1_create_client()
    agent = await step_2_create_agent(client)
    await step_3_single_shot(agent)
    await step_4_multi_turn(agent)
    await step_5_summary()


if __name__ == "__main__":
    asyncio.run(main())
