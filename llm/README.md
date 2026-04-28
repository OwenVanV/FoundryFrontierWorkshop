# LLM Deep-Dive — Workshop Walkthrough

> **Use Azure OpenAI to crack a hidden fraud ring buried in 50,000+ synthetic payment transactions — no agents, no frameworks, just raw LLM power.**

This is the core workshop: 6 progressive modules that demonstrate what large language models can do with the right prompting, structured outputs, and evaluation frameworks. No agent abstractions — you talk to the model directly.

---

## What Makes This Different from the Agents Track

| | LLM Track (this) | Agents Track |
|---|---|---|
| **Interface** | Direct `call_openai()` wrapper | MAF `Agent` + Foundry Agent Service |
| **Tools** | None — LLM reasons over data summaries | `@tool` functions, Code Interpreter |
| **State** | Manual conversation history | `AgentSession`, server-side conversations |
| **Focus** | Prompt engineering, structured outputs, evaluation | Agent orchestration, tool calling, workflows |

Both tracks use the same synthetic data and investigate the same fraud ring. Compare results in Module 05.

---

## Prerequisites

1. **Azure OpenAI** resource with a GPT-4o deployment
2. **Python 3.10+**
3. Generate the synthetic data first:
   ```bash
   python data/generate_synthetic_data.py
   ```

## Setup

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install LLM dependencies
uv sync --extra llm

# Configure
cp .env.example .env
# Edit .env with your Azure OpenAI endpoint and deployment name
```

<details>
<summary>Alternative: pip</summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[llm]"
```
</details>

### Authentication Modes

| Mode | `.env` Setting | How It Works |
|---|---|---|
| **API Key** (default) | `USE_MANAGED_IDENTITY=false` | Set `AZURE_OPENAI_API_KEY` in `.env` |
| **Managed Identity** | `USE_MANAGED_IDENTITY=true` | Run `az login`, or use Managed Identity on Azure. No API key needed. |

The `USE_MANAGED_IDENTITY` flag controls authentication across **all** workshop scenarios — LLM, agents, and Voice_CUA. Set it once in the root `.env`.

---

## Module Walkthrough

### Module 01 — Data Exploration with Azure OpenAI
**File:** `01_data_exploration.py` | **Time:** 20-25 min

**What you learn:**
- Configure and authenticate with Azure OpenAI
- Use LLMs for intelligent data profiling
- Understand token usage and API call economics
- Identify initial anomalies through natural-language analysis

**What happens:**
1. Load `transactions.csv` (50K rows), `customers.csv` (2K), `merchants.csv` (500)
2. Compute statistical profiles: amount distributions, currency mix, time patterns
3. Send profiles to GPT-4o with a "senior payment analyst" system prompt
4. LLM identifies anomalies: amount clustering, email domain concentrations, registration patterns

**Key technique:** Pre-aggregate data → send summaries to LLM. You can't send 50K rows, so you summarize strategically.

**Discussion prompt:** *"We sent statistical summaries, not raw data. What anomalies might we miss with this approach? When would you send raw rows instead?"*

```bash
uv run python llm/01_data_exploration.py
```

---

### Module 02 — Pattern Detection with Structured Outputs
**File:** `02_pattern_detection.py` | **Time:** 25-30 min

**What you learn:**
- Use `response_format={"type": "json_object"}` for structured LLM outputs
- Design zero-shot anomaly detection prompts
- Analyze transaction amount distributions for smurfing
- Detect temporal patterns in transaction timing

**What happens:**
1. Create targeted samples: high-value transactions, near-threshold ($9K-$10K), repeat senders
2. Zero-shot smurfing detection: LLM identifies $9,200-$9,800 clustering
3. Temporal analysis: find time windows with coordinated high-value transactions
4. Merchant shell company detection: similar names, registration clustering

**Key technique:** JSON mode forces parseable output. Each anomaly has a type, confidence score, and evidence citations — ready for downstream automation.

**Discussion prompt:** *"We used zero-shot detection (no examples). How would few-shot prompting change the results? What would good examples look like?"*

```bash
uv run python llm/02_pattern_detection.py
```

---

### Module 03 — Cross-Reference Analysis with Multi-Turn Conversations
**File:** `03_cross_reference_analysis.py` | **Time:** 30-35 min

**What you learn:**
- Design multi-turn investigation conversations
- Cross-reference signals across CSV, JSON, and time-series data
- Use chain-of-thought prompting ("think step by step")
- Understand token cost growth in multi-turn conversations

**What happens:**
1. Device fingerprint cross-reference: find accounts sharing devices (nested JSON)
2. Impossible travel detection: same customer in Lagos and London in 20 minutes
3. 4-turn investigation conversation:
   - Turn 1: Present Module 02 findings
   - Turn 2: Add device fingerprint evidence
   - Turn 3: Add impossible travel evidence
   - Turn 4: Request structured hypothesis

**Key technique:** Multi-turn context accumulation. Each turn includes ALL previous messages — token cost grows linearly. The LLM needs the full picture to connect dots.

**Observe:** Token usage grows ~2x per turn. Turn 4 sends the full conversation history.

```bash
uv run python llm/03_cross_reference_analysis.py
```

---

### Module 04 — Fraud Ring Investigation & Report Generation
**File:** `04_fraud_ring_investigation.py` | **Time:** 25-30 min

**What you learn:**
- Network graph reasoning with LLMs
- Investigation report generation
- Responsible AI considerations in fraud detection

**What happens:**
1. Build a transaction network graph (sender → receiver adjacency)
2. Find bidirectional pairs and high-connectivity hub nodes
3. LLM performs network analysis: identifies ring structure, member roles, shell merchants
4. Generate a formal investigation report suitable for regulatory review

**Key technique:** Present graph data as text (edge lists, node stats). LLMs can reason about network structure from well-formatted text.

**Discussion prompt:** *"This report could trigger account freezes affecting real people. What human oversight is needed before acting on AI-generated findings?"*

```bash
uv run python llm/04_fraud_ring_investigation.py
```

---

### Module 05 — Evaluation Framework
**File:** `05_evaluation_framework.py` | **Time:** 20-25 min

**What you learn:**
- Precision, recall, and F1 for fraud detection
- Compare LLM predictions against ground truth
- Analyze false positives (innocent accounts flagged) and false negatives (fraudsters missed)
- Test prompt variants for detection improvement

**What happens:**
1. Load `ground_truth.json` (the answer key) and Module 04 predictions
2. Compute precision/recall/F1 for ring members and shell merchants
3. LLM-powered error analysis: why did certain errors occur?
4. Test 4 prompt variants: baseline, chain-of-thought, high-precision, high-recall
5. Compare variant accuracy side-by-side

**Key technique:** Systematic prompt evaluation. Small prompt changes can dramatically shift precision vs. recall. Measure, don't guess.

**Discussion prompt:** *"In fraud detection, is it worse to freeze an innocent account (false positive) or miss a fraudster (false negative)? How does your answer change the target metrics?"*

```bash
uv run python llm/05_evaluation_framework.py
```

---

### Module 06 — Observability Dashboard
**File:** `06_observability_dashboard.py` | **Time:** 15-20 min

**What you learn:**
- AI-specific observability requirements
- Cost attribution by module and operation
- Latency percentile analysis
- Production readiness assessment

**What happens:**
1. Aggregate all telemetry from Modules 01-05
2. Display API usage table: calls, tokens, estimated cost per module
3. Latency distribution histogram
4. Platform health dashboard (from `system_telemetry.csv`)
5. Detection quality metrics (from Module 05 evaluation)
6. Production readiness score with pass/warn/fail checks

```bash
uv run python llm/06_observability_dashboard.py
```

---

## The Hidden Fraud Ring — What to Look For

The LLM is hunting for 7 interlocking signals. No single signal is conclusive — the power of the LLM is reasoning across all of them simultaneously:

| Signal | Data Source | What to Look For |
|---|---|---|
| Smurfing | `transactions.csv` | Amounts clustering at $9,200-$9,800 |
| Circular flow | `transactions.csv` | A→B→C→D→A with 48-72hr delays |
| Shell merchants | `merchants.csv` | "Apex Digital Solutions" / "Apex Digitial Svcs" — similar names, Delaware |
| Device sharing | `payment_sessions.json` | 6 accounts share 3 device fingerprints |
| Impossible travel | `payment_sessions.json` | Lagos → London in 20 minutes |
| Synthetic identity | `customers.csv` | 2-week registration burst, `quickinbox.io` / `dropmail.cc` emails |
| Temporal coordination | `transactions.csv` | Ring transactions fire in 3-7 minute windows |

---

## Telemetry & Observability

Every `call_openai()` automatically records:
- Token usage (prompt + completion)
- Latency (milliseconds)
- Module attribution
- Estimated cost

These feed into Module 06's dashboard. When `ENABLE_AZURE_MONITOR=true`, telemetry also exports to Application Insights.

---

## Expected Budget

Full run of all 6 modules: **$0.50 - $2.00** in Azure OpenAI tokens (GPT-4o pricing). Module 05's prompt variant testing adds ~4 extra API calls.
