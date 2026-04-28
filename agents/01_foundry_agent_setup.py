"""
============================================================================
AGENTS MODULE 01 — Foundry Agent Service: Setup & First Agent
============================================================================

WORKSHOP NARRATIVE:
    In the first part of the workshop, we used raw Azure OpenAI calls to
    investigate fraud. Now we level up: we'll use MANAGED AGENTS that
    have built-in tools like Code Interpreter.

    This module introduces the Foundry Agent Service — Microsoft's managed
    agent runtime. You create a "Prompt Agent" with Code Interpreter
    enabled, upload the transaction CSV, and let the agent write and
    execute Python code in a sandboxed environment to analyze the data.

    This is different from MAF (Microsoft Agent Framework, Module 02).
    Foundry Agent Service = managed cloud agents with server-side tools.
    MAF = local SDK for building/orchestrating agents with custom tools.

LEARNING OBJECTIVES:
    1. Understand the difference between Foundry Agent Service and MAF
    2. Create a versioned Prompt Agent with Code Interpreter
    3. Upload files for agent analysis in a sandboxed Python environment
    4. Download generated artifacts (charts, CSVs) from the agent

AZURE SERVICES:
    - Foundry Agent Service (PromptAgentDefinition, CodeInterpreterTool)
    - azure-ai-projects SDK

ESTIMATED TIME: 20-25 minutes

============================================================================
"""

import os
import sys
import asyncio
from pathlib import Path

from dotenv import load_dotenv
# Load agents/.env first, then fall back to root .env
_agents_dir = Path(__file__).parent
load_dotenv(_agents_dir / ".env")
load_dotenv()  # root .env as fallback (won't override existing vars)

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition,
    CodeInterpreterTool,
    AutoCodeInterpreterToolParam,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT")

if not PROJECT_ENDPOINT:
    print("ERROR: Set FOUNDRY_PROJECT_ENDPOINT in your .env file.")
    print("       Format: https://<your-project>.services.ai.azure.com")
    sys.exit(1)


# ============================================================================
# STEP 1: Connect to the Foundry Project
# ============================================================================
# AIProjectClient is the gateway to the Foundry Agent Service.
# It manages agent definitions, file uploads, and conversations.
#
# WORKSHOP NOTE:
# "DefaultAzureCredential tries az login, managed identity, VS Code, etc.
# Run `az login` before this module if you haven't already."
# ============================================================================

def step_1_connect():
    """Establish connection to the Foundry project."""
    print("=" * 70)
    print("AGENTS MODULE 01 — Foundry Agent Service Setup")
    print("=" * 70)
    print()

    print("  Connecting to Foundry project...")
    project = AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
    )
    # Get the OpenAI-compatible client for file uploads and conversations
    openai_client = project.get_openai_client()

    print(f"  ✓ Connected to: {PROJECT_ENDPOINT}")
    print()
    return project, openai_client


# ============================================================================
# STEP 2: Upload Transaction Data
# ============================================================================
# Code Interpreter needs files to analyze. We upload transactions.csv
# to Azure-managed storage. The file gets an ID that we'll attach
# to the agent's tool configuration.
#
# WORKSHOP DISCUSSION POINT:
# "Files uploaded here are stored in Azure-managed storage, not your
# subscription's storage. They're accessible only within the agent's
# sandbox. This is a security feature — the agent can't access your
# network or other Azure resources."
# ============================================================================

def step_2_upload_data(openai_client) -> dict:
    """Upload the fraud datasets for Code Interpreter analysis."""
    print("─" * 70)
    print("STEP 2: Uploading Data Files")
    print("─" * 70)
    print()

    files_to_upload = {
        "transactions": DATA_DIR / "transactions.csv",
        "customers": DATA_DIR / "customers.csv",
        "merchants": DATA_DIR / "merchants.csv",
    }

    uploaded = {}
    for label, filepath in files_to_upload.items():
        if not filepath.exists():
            print(f"  ✗ {label}: {filepath} not found!")
            print(f"    Run: python data/generate_synthetic_data.py")
            sys.exit(1)

        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  Uploading {label} ({size_mb:.1f} MB)...")

        with open(filepath, "rb") as f:
            file_obj = openai_client.files.create(
                purpose="assistants",
                file=f,
            )
        uploaded[label] = file_obj
        print(f"  ✓ {label}: {file_obj.id}")

    print()
    return uploaded


# ============================================================================
# STEP 3: Create a Prompt Agent with Code Interpreter
# ============================================================================
# A Prompt Agent is defined entirely through configuration:
#   - Model (which LLM to use)
#   - Instructions (system prompt)
#   - Tools (code_interpreter, web_search, file_search)
#
# We use create_version() to create a VERSIONED agent. This is a
# production pattern: you can roll back to previous versions if
# a new prompt doesn't work well.
#
# WORKSHOP KEY INSIGHT:
# "Code Interpreter runs Python in an isolated Hyper-V sandbox.
# It has pandas, numpy, matplotlib, seaborn, and other data science
# packages pre-installed. The agent writes the code, executes it,
# and returns the results — including generated files like charts."
# ============================================================================

def step_3_create_agent(project, uploaded_files) -> dict:
    """Create a versioned Prompt Agent with Code Interpreter."""
    print("─" * 70)
    print("STEP 3: Creating Prompt Agent with Code Interpreter")
    print("─" * 70)
    print()

    file_ids = [f.id for f in uploaded_files.values()]

    agent = project.agents.create_version(
        agent_name="FraudDataAnalyst",
        definition=PromptAgentDefinition(
            model=os.getenv("FOUNDRY_MODEL", "gpt-4o"),
            instructions=(
                "You are a senior fraud data analyst at a major payment processor. "
                "You have access to transaction, customer, and merchant datasets via "
                "Code Interpreter. When analyzing data:\n\n"
                "1. Always start by loading and inspecting the CSV files to understand the schema.\n"
                "2. Use pandas for data manipulation and analysis.\n"
                "3. Generate charts with matplotlib when visualizing distributions.\n"
                "4. Focus on identifying:\n"
                "   - Transaction amount clustering (especially $9,000-$10,000 range)\n"
                "   - Unusual merchant registration patterns\n"
                "   - Customer registration date clustering\n"
                "   - Email domain anomalies\n"
                "5. Provide specific numbers, percentages, and statistical measures.\n"
                "6. When you find anomalies, list the specific IDs involved."
            ),
            tools=[
                CodeInterpreterTool(
                    container=AutoCodeInterpreterToolParam(file_ids=file_ids)
                ),
            ],
        ),
        description="Fraud data analyst with Code Interpreter for transaction analysis.",
    )

    print(f"  ✓ Agent created: {agent.name} (version: {agent.version})")
    print(f"  Model: {os.getenv('FOUNDRY_MODEL', 'gpt-4o')}")
    print(f"  Tools: Code Interpreter with {len(file_ids)} files")
    print()
    return agent


# ============================================================================
# STEP 4: Run Analysis Conversations
# ============================================================================
# Now we create a conversation and ask the agent to analyze the data.
# The agent will write Python code, execute it in the sandbox, and
# return results — potentially including generated chart files.
#
# WORKSHOP NOTE:
# "Watch the response carefully. The agent doesn't just describe the
# data — it WRITES AND EXECUTES CODE. You'll see it import pandas,
# load the CSV, compute statistics, and generate visualizations.
# This is the power of Code Interpreter: it's a data scientist in a box."
# ============================================================================

def step_4_run_analysis(project, openai_client, agent) -> list:
    """Run analysis conversations with the Foundry agent."""
    print("─" * 70)
    print("STEP 4: Running Analysis Conversations")
    print("─" * 70)
    print()

    # Create a conversation
    conversation = openai_client.conversations.create()
    print(f"  Conversation created: {conversation.id}")
    print()

    # ----------------------------------------------------------------
    # Analysis 1: Data Overview
    # ----------------------------------------------------------------
    print("  [4a] Requesting data overview...")
    response = openai_client.responses.create(
        conversation=conversation.id,
        input=(
            "Load all three CSV files (transactions, customers, merchants). "
            "For each, show: row count, column names, and basic statistics. "
            "Then create a histogram of transaction amounts with bins at "
            "$0-100, $100-500, $500-1K, $1K-5K, $5K-9K, $9K-10K, $10K+. "
            "Highlight any unusual concentrations."
        ),
        extra_body={
            "agent_reference": {
                "name": agent.name,
                "type": "agent_reference",
            }
        },
    )

    print(f"  ✓ Response received")
    _print_response(response)

    # ----------------------------------------------------------------
    # Analysis 2: Smurfing Detection
    # ----------------------------------------------------------------
    print()
    print("  [4b] Requesting smurfing analysis...")
    response = openai_client.responses.create(
        conversation=conversation.id,
        input=(
            "Focus on the $9,000-$10,000 transaction range. This is the "
            "Bank Secrecy Act reporting threshold zone. Analyze:\n"
            "1. How many transactions fall in this range?\n"
            "2. What percentage of total transactions is this?\n"
            "3. Which sender IDs appear most frequently in this range?\n"
            "4. Which merchants receive these transactions?\n"
            "5. Is there a tighter cluster within this range (e.g., $9,200-$9,800)?\n"
            "Create a scatter plot of amount vs timestamp for these transactions."
        ),
        extra_body={
            "agent_reference": {
                "name": agent.name,
                "type": "agent_reference",
            }
        },
    )

    print(f"  ✓ Response received")
    _print_response(response)

    # ----------------------------------------------------------------
    # Analysis 3: Merchant Registration Patterns
    # ----------------------------------------------------------------
    print()
    print("  [4c] Requesting merchant analysis...")
    response = openai_client.responses.create(
        conversation=conversation.id,
        input=(
            "Analyze merchant registration patterns:\n"
            "1. Plot merchant registration dates over time.\n"
            "2. Identify any clusters of merchants registered within a 2-week window.\n"
            "3. For clustered merchants, compare their business names for similarity.\n"
            "4. Check if clustered merchants share the same state or city.\n"
            "5. Flag any merchants with suspiciously similar names."
        ),
        extra_body={
            "agent_reference": {
                "name": agent.name,
                "type": "agent_reference",
            }
        },
    )

    print(f"  ✓ Response received")
    _print_response(response)

    # Download any generated files
    _download_generated_files(response, openai_client)

    return [response]


def _print_response(response):
    """Print the agent's response content."""
    last_msg = response.output[-1] if response.output else None
    if last_msg and last_msg.type == "message" and last_msg.content:
        for part in last_msg.content:
            if hasattr(part, "text"):
                text = part.text if isinstance(part.text, str) else getattr(part.text, "value", str(part.text))
                print()
                print("  ┌─ Agent Response ────────────────────────────────────────────")
                for line in str(text).split("\n")[:30]:
                    print(f"  │ {line}")
                if len(str(text).split("\n")) > 30:
                    print(f"  │ ... ({len(str(text).split(chr(10)))} total lines)")
                print("  └────────────────────────────────────────────────────────────")


def _download_generated_files(response, openai_client):
    """Download any files generated by Code Interpreter."""
    last_msg = response.output[-1] if response.output else None
    if not last_msg or not hasattr(last_msg, "content"):
        return

    for part in last_msg.content:
        text_obj = getattr(part, "text", None)
        if text_obj is None:
            continue
        annotations = getattr(text_obj, "annotations", None)
        if not annotations:
            continue
        for annotation in annotations:
            if getattr(annotation, "type", "") == "container_file_citation":
                file_id = annotation.file_id
                filename = annotation.filename
                container_id = annotation.container_id
                print(f"\n  📊 Downloading generated file: {filename}")
                try:
                    content = openai_client.containers.files.content.retrieve(
                        file_id=file_id, container_id=container_id,
                    )
                    output_path = Path(__file__).parent / filename
                    with open(output_path, "wb") as f:
                        f.write(content.read())
                    print(f"  ✓ Saved to: {output_path}")
                except Exception as e:
                    print(f"  ⚠ Download failed: {e}")


# ============================================================================
# STEP 5: Cleanup and Summary
# ============================================================================

def step_5_summary(project, agent):
    """Clean up and print summary."""
    print()
    print("=" * 70)
    print("AGENTS MODULE 01 — Summary")
    print("=" * 70)

    print(f"""
  What You Built:
    ├─ A versioned Prompt Agent: {agent.name}
    ├─ With Code Interpreter (sandboxed Python execution)
    ├─ Analyzing 3 uploaded datasets
    └─ Agent writes + executes pandas code autonomously

  Foundry Agent Service vs. Raw OpenAI Calls:
    ├─ Server-managed agent definition (versioned, rollback-able)
    ├─ Code Interpreter runs in isolated Hyper-V sandbox
    ├─ Files stored in Azure-managed storage
    └─ Conversation state managed server-side

  KEY TAKEAWAY:
  Foundry Agent Service excels at data analysis tasks where the agent
  needs to write and execute code. The sandboxed environment means
  the agent can safely run arbitrary Python — including pandas,
  matplotlib, and numpy — without touching your infrastructure.

  In Module 02, we switch to the Microsoft Agent Framework (MAF) for
  agents that call CUSTOM tools — functions you define in your code.

  NEXT: Run 02_maf_first_agent.py
""")

    # Cleanup: delete the agent version
    try:
        project.agents.delete_version(
            agent_name=agent.name,
            agent_version=agent.version,
        )
        print(f"  ✓ Agent version cleaned up: {agent.name} v{agent.version}")
    except Exception:
        print(f"  ℹ Agent cleanup skipped (may need manual deletion)")

    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    project, openai_client = step_1_connect()
    uploaded_files = step_2_upload_data(openai_client)
    agent = step_3_create_agent(project, uploaded_files)
    step_4_run_analysis(project, openai_client, agent)
    step_5_summary(project, agent)


if __name__ == "__main__":
    main()
