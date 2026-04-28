# Agents Deep-Dive — Workshop Walkthrough

> **Use Foundry Agent Service + Microsoft Agent Framework to build intelligent fraud investigators with managed tools and multi-agent workflows.**

This module builds on the synthetic fraud data from the main workshop. Where the main workshop used raw Azure OpenAI calls, here you use **managed agents** that can write code, search the web, and orchestrate multi-step investigations autonomously.

---

## Two Agent Systems, One Workshop

This walkthrough uses **two distinct** Azure agent technologies:

| System | Package | What It Does | Modules |
|---|---|---|---|
| **Foundry Agent Service** | `azure-ai-projects` | Managed cloud agents with server-side tools (Code Interpreter, Web Search) | 01, 03 |
| **Microsoft Agent Framework (MAF)** | `agent-framework` | Local SDK for building agents with custom `@tool` functions, sessions, workflows | 02, 04, 05 |

Think of it this way:
- **Foundry Agent Service** = "agent as a service" — agent definition lives in the cloud, tools run in sandboxed environments
- **MAF** = "agent as a library" — agent runs in your code, tools are your Python functions

---

## Prerequisites

1. **Azure subscription** with a Foundry project
2. **GPT-4o deployment** in your Foundry project
3. **Python 3.10+**
4. Complete the main workshop data generation first:
   ```bash
   python data/generate_synthetic_data.py
   ```

## Setup

```bash
# From the project root (PayPal/), install with the agents extra:
uv sync --extra agents

# Configure
cp agents/.env.example agents/.env
# Edit agents/.env with your Foundry project endpoint

# Authenticate
az login  # Required for both Foundry Agent Service and MAF
```

<details>
<summary>Alternative: pip + manual venv</summary>

```bash
cd agents
pip install -r requirements.txt
cp .env.example .env
az login
```
</details>

### Authentication Modes

| Mode | `.env` Setting | How It Works |
|---|---|---|
| **API Key** (default) | `USE_MANAGED_IDENTITY=false` | Uses `az login` for Foundry, API keys for OpenAI |
| **Managed Identity** | `USE_MANAGED_IDENTITY=true` | Uses `DefaultAzureCredential` everywhere — works on Azure VMs, App Service, AKS |

---

## Module Walkthrough

### Module 01 — Foundry Agent Service: Setup & First Agent
**File:** `01_foundry_agent_setup.py` | **Time:** 20-25 min | **System:** Foundry Agent Service

**What happens:**
1. Connect to your Foundry project via `AIProjectClient`
2. Upload `transactions.csv`, `customers.csv`, `merchants.csv` to Azure-managed storage
3. Create a versioned Prompt Agent with Code Interpreter enabled
4. The agent writes and executes pandas code in a Hyper-V sandbox
5. Agent generates charts (histogram of amounts, scatter plots) — you download them

**Key concept:** Code Interpreter runs arbitrary Python safely. The agent decides what code to write. You upload the data and ask the question.

**Discussion prompt:** *"What are the security implications of letting an LLM write and execute code? How does the Hyper-V sandbox mitigate this?"*

```bash
uv run python agents/01_foundry_agent_setup.py
```

---

### Module 02 — MAF: Your First Agent Framework Agent
**File:** `02_maf_first_agent.py` | **Time:** 20-25 min | **System:** Microsoft Agent Framework

**What happens:**
1. Create a `FoundryChatClient` (uses Foundry for inference, but agent logic is local)
2. Attach 6 custom `@tool` functions that query the fraud datasets
3. Run single-shot queries — watch the agent decide which tools to call
4. Use `AgentSession` for multi-turn investigation

**Key concept:** `@tool` decorator converts Python functions into tool schemas automatically. The LLM sees parameter types and docstrings as tool descriptions.

**Discussion prompt:** *"The MAF agent calls `query_transactions(min_amount=9000, max_amount=10000)` — a human would write the same SQL. What value does the LLM add?"*

```bash
uv run python agents/02_maf_first_agent.py
```

---

### Module 03 — Foundry Agent Service: Deep Code Analysis
**File:** `03_foundry_code_analysis.py` | **Time:** 25-30 min | **System:** Foundry Agent Service

**What happens:**
1. Upload ALL datasets (transactions, customers, merchants, velocity metrics)
2. 5-phase analysis in a single conversation:
   - Network construction (adjacency matrices)
   - Circular flow detection (A→B→C→D→A)
   - Customer cross-reference (registration clustering, email domains)
   - Time-series anomaly detection (velocity spikes)
   - Final investigation report with network diagram
3. Each phase builds on prior code — the sandbox persists state

**Key concept:** Multi-step Code Interpreter conversations. The agent's code from Phase 1 (DataFrames, variables) is available in Phase 2. This is like a data scientist working in a Jupyter notebook.

```bash
uv run python agents/03_foundry_code_analysis.py
```

---

### Module 04 — MAF: Deep Multi-Turn Investigation
**File:** `04_maf_multi_turn_investigation.py` | **Time:** 20-25 min | **System:** MAF

**What happens:**
1. 4-phase progressive investigation using `AgentSession`:
   - Phase 1: Start with a tip ("search for Apex merchants")
   - Phase 2: Follow the money (network analysis via tools)
   - Phase 3: Identity analysis (customer profiles, device fingerprints)
   - Phase 4: Case summary with **streaming** output
2. The agent decides which tools to call in each phase
3. Streaming shows tokens arriving in real-time

**Key concept:** Sessions maintain context across turns. The agent remembers its Phase 1 findings when reasoning in Phase 3.

**Discussion prompt:** *"Streaming adds ~10 lines of code but transforms the UX. When should you use streaming vs. non-streaming in production?"*

```bash
uv run python agents/04_maf_multi_turn_investigation.py
```

---

### Module 05 — MAF: Workflow Orchestration (Multi-Agent)
**File:** `05_maf_workflow_orchestration.py` | **Time:** 25-30 min | **System:** MAF

**What happens:**
1. Create 3 specialist agents:
   - **TransactionAnalyst** — tools: `query_transactions`, `find_similar_merchants`, `check_merchant`
   - **IdentityInvestigator** — tools: `lookup_customer`, `analyze_account_network`, `check_device_fingerprints`
   - **ReportWriter** — no tools (writing only)
2. Functional workflow chains them: Analyst → Investigator → Writer
3. Each agent's output is the next agent's input

**Key concept:** MAF functional workflows are plain `async def` functions. No DSL, no YAML — it's Python. Each agent has focused instructions and minimal tools.

**Discussion prompt:** *"We gave each agent 2-3 tools instead of all 6. Why does specialization improve accuracy?"*

```bash
uv run python agents/05_maf_workflow_orchestration.py
```

---

### Module 06 — Evaluation
**File:** `06_evaluation.py` | **Time:** 15-20 min | **System:** MAF

**What happens:**
1. Run a fresh investigation with a single MAF agent
2. Compare predictions against `ground_truth.json`
3. Compute precision, recall, F1 for ring members and shell merchants
4. Compare agent accuracy to the raw LLM results from the main workshop

```bash
uv run python agents/06_evaluation.py
```

---

## Comparison: When to Use Each System

| Scenario | Use This |
|---|---|
| Data analysis with pandas, matplotlib | **Foundry Agent Service** (Code Interpreter) |
| Custom tool integration (APIs, databases) | **MAF** (`@tool` functions) |
| Agent definition managed in cloud portal | **Foundry Agent Service** (Prompt Agents) |
| Multi-agent orchestration | **MAF** (functional workflows) |
| Production deployment with versioning | **Foundry Agent Service** (versioned agents) |
| Full control over agent behavior | **MAF** (local Agent class) |

---

## Expected Budget

Full run of all 6 modules: ~$1-3 in Azure OpenAI tokens (GPT-4o pricing). Code Interpreter sessions have additional charges (~$0.03/session).
