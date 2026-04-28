# 🔍 Cracking the Ring — PayPal x Azure AI Fraud Investigation Workshop

> **Use Azure OpenAI to uncover a hidden fraud ring buried in 50,000+ synthetic payment transactions.**

This workshop accelerator demonstrates the power of Azure OpenAI, Azure AI Foundry, observability, telemetry, and evaluation frameworks through the lens of a payment processing fraud investigation. Participants work through 6 progressive Python modules, using LLMs to analyze multi-format datasets and uncover a sophisticated fraud ring that traditional rules-based systems missed.

---

## 📋 Workshop Overview

| | |
|---|---|
| **Duration** | 2.5 – 3 hours |
| **Audience** | Engineers, data scientists, product leads |
| **Prerequisites** | Python 3.10+, Azure OpenAI access |
| **Format** | Facilitator-led, hands-on coding |
| **Focus** | LLMs (not agents) — raw model capabilities |

### What Participants Will Do

1. **Explore** 50,000+ payment transactions, 2,000 customers, and 500 merchants using LLM-powered data profiling
2. **Detect** anomalies using structured outputs (JSON mode) — smurfing, temporal coordination, shell companies
3. **Cross-reference** signals across CSV, JSON, and time-series datasets using multi-turn conversations
4. **Investigate** the full fraud ring using network graph reasoning
5. **Evaluate** detection accuracy against ground truth (precision, recall, F1)
6. **Monitor** the entire system through an observability dashboard

### Azure Services Demonstrated

| Service | How It's Used |
|---|---|
| **Azure OpenAI** | Chat completions, JSON mode, multi-turn conversations, prompt engineering |
| **Azure AI Foundry** | Model deployment and management patterns |
| **Azure Monitor** | Application Insights telemetry export |
| **OpenTelemetry** | Distributed tracing, metrics collection, cost tracking |

---

## 🗂️ Project Structure

```
PayPal/
├── README.md                          ← You are here
├── requirements.txt                   ← Python dependencies
├── .env.example                       ← Azure configuration template
├── .gitignore
│
├── data/
│   ├── generate_synthetic_data.py     ← Deterministic data generator (seed=42)
│   ├── transactions.csv               ← ~50,000 transaction records
│   ├── merchants.csv                  ← ~500 merchant profiles
│   ├── customers.csv                  ← ~2,000 customer profiles
│   ├── payment_sessions.json          ← Nested session/device/geo data
│   ├── dispute_cases.json             ← Chargeback cases with evidence
│   ├── risk_signals.json              ← Risk engine scoring breakdowns
│   ├── velocity_metrics.csv           ← Hourly time-series aggregations
│   ├── system_telemetry.csv           ← Platform health metrics
│   └── ground_truth.json              ← Answer key (facilitator only)
│
├── utils/
│   ├── azure_client.py                ← Azure OpenAI client factory + tracking
│   └── telemetry.py                   ← OpenTelemetry + Azure Monitor setup
│
├── 01_data_exploration.py             ← Module 1: LLM-powered data profiling
├── 02_pattern_detection.py            ← Module 2: Structured output anomaly detection
├── 03_cross_reference_analysis.py     ← Module 3: Multi-turn cross-dataset investigation
├── 04_fraud_ring_investigation.py     ← Module 4: Network analysis + report generation
├── 05_evaluation_framework.py         ← Module 5: Precision/recall/F1 evaluation
└── 06_observability_dashboard.py      ← Module 6: Cost, latency, and health dashboard
```

---

## 🚀 Setup Instructions

### 1. Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) (fast Python package manager) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- An Azure subscription with Azure OpenAI access
- A GPT-4o deployment in Azure OpenAI

### 2. Clone and Install

```bash
cd PayPal

# Create venv and install core + LLM dependencies
uv sync --extra llm

# Or install everything (LLM + agents + voice):
# uv sync --extra all
```

<details>
<summary>Alternative: pip + manual venv</summary>

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[llm]"
```
</details>

### 3. Configure Azure Credentials

```bash
cp .env.example .env
# Edit .env with your Azure OpenAI endpoint, API key, and deployment name
```

Required `.env` values:
```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

Or use Managed Identity (no API key needed):
```
USE_MANAGED_IDENTITY=true
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
# Then: az login
```

### 4. Generate Synthetic Data

```bash
uv run python data/generate_synthetic_data.py
```

This creates all 8 dataset files plus the ground truth answer key. The generator is seeded (`SEED=42`) so every participant gets identical data.

### 5. Run Modules Sequentially

```bash
uv run python 01_data_exploration.py
uv run python 02_pattern_detection.py
uv run python 03_cross_reference_analysis.py
uv run python 04_fraud_ring_investigation.py
uv run python 05_evaluation_framework.py
uv run python 06_observability_dashboard.py
```

Each module produces findings that feed into the next. Run them in order.

### 6. Run Tests

```bash
uv run pytest tests/ -v
# Or: uv run python tests/test_workshop.py
```

---

## 🕵️ The Hidden Fraud Ring

Buried in the synthetic data is a coordinated fraud ring consisting of **18 accounts** operating through **4 shell merchants**. The ring represents ~0.84% of all transactions — a true needle in a haystack.

### Why Traditional Rules Miss It

The fraud ring was designed to evade rules-based detection systems. Each individual signal is plausible on its own. Only by cross-referencing multiple signals can you identify the ring:

| # | Signal | Where It Hides | Why It's Subtle |
|---|---|---|---|
| 1 | **Smurfing** | `transactions.csv` | Amounts cluster at $9,200–$9,800 (below $10K BSA threshold) with realistic cents — not round numbers |
| 2 | **Circular Money Flow** | `transactions.csv` | A→B→C→D→A with 48–72hr delays between hops and noise transactions in between |
| 3 | **Shell Merchant Clustering** | `merchants.csv` | 4 Delaware-registered merchants with similar names ("Apex Digital Solutions" / "Apex Digitial Svcs"), registered within 10 days |
| 4 | **Device Fingerprint Sharing** | `payment_sessions.json` | 6 accounts share 3 device fingerprints — hidden in nested JSON among 15,000 sessions |
| 5 | **Impossible Travel** | `payment_sessions.json` | Same customer in Lagos and London within 20 minutes — requires joining timestamps with geolocation |
| 6 | **Synthetic Identity** | `customers.csv` | Ring members registered in a 2-week burst, emails cluster on 2 obscure domains, phone area codes don't match cities |
| 7 | **Temporal Coordination** | `transactions.csv` | Ring transactions fire within 3–7 minute windows, visible only when filtering to ring-connected accounts |

### Why LLMs Can Find It

LLMs excel at this because they can simultaneously reason about:
- **Semantic similarity** (merchant name variations)
- **Temporal patterns** (coordinated timing across accounts)
- **Cross-dataset joins** (device fingerprints + geolocation + transaction flows)
- **Network structure** (circular flows, hub-and-spoke patterns)

This is the core thesis of the workshop: LLMs are investigative partners, not just classifiers.

---

## 📖 Module Guide

### Module 01 — Data Exploration (20–25 min)

**Theme:** "Understand the landscape before investigating."

Participants load the datasets and use Azure OpenAI as an intelligent data profiling tool. Instead of writing pandas aggregations, they feed statistical summaries to the LLM and get natural-language insights.

**Key Concepts:**
- Azure OpenAI client configuration and authentication
- Token usage tracking (prompt vs. completion tokens)
- The tradeoff between summarizing data vs. sending raw rows
- Temperature settings for analytical tasks (0.1 = deterministic)

**Azure Focus:** Azure OpenAI chat completions, OpenTelemetry span tracing

---

### Module 02 — Pattern Detection (25–30 min)

**Theme:** "From free text to structured, actionable outputs."

Participants switch to **JSON mode** (`response_format={"type": "json_object"}`), forcing the LLM to output structured anomaly detections with confidence scores and evidence citations.

**Key Concepts:**
- Structured outputs / JSON mode
- Zero-shot anomaly detection prompting
- Strategic sampling (can't send 50K rows to the LLM)
- Amount distribution analysis for smurfing detection

**Azure Focus:** Azure OpenAI structured outputs, latency histograms

---

### Module 03 — Cross-Reference Analysis (30–35 min)

**Theme:** "Connect the dots across datasets."

Participants run a **multi-turn investigation conversation** where each turn adds evidence from a different data source. The LLM progressively narrows the investigation, connecting transaction patterns to device fingerprints to geolocation anomalies.

**Key Concepts:**
- Multi-turn conversation design
- Chain-of-thought prompting ("think step by step")
- Context accumulation and its cost implications
- Cross-referencing CSV, JSON, and time-series data

**Azure Focus:** Multi-turn chat completions, cumulative token tracking

---

### Module 04 — Fraud Ring Investigation (25–30 min)

**Theme:** "Map the full ring and write the report."

The LLM performs network graph reasoning on transaction flows and generates a formal investigation report suitable for compliance review.

**Key Concepts:**
- Network graph reasoning (nodes = accounts, edges = transactions)
- Bidirectional flow detection and circular pattern identification
- Investigation report generation
- Responsible AI: human oversight requirements

**Azure Focus:** Azure AI Foundry patterns, content safety

---

### Module 05 — Evaluation Framework (20–25 min)

**Theme:** "Trust but verify — measure everything."

Participants compare the LLM's fraud ring identification against the ground truth answer key, calculating precision, recall, and F1 scores. They then test prompt variants to see how instruction style affects accuracy.

**Key Concepts:**
- Precision (false positive control)
- Recall (false negative control)
- F1 score (harmonic mean)
- Prompt variant testing and comparison
- The precision-recall tradeoff in fraud detection

**Azure Focus:** Azure AI Evaluation SDK patterns, systematic quality measurement

---

### Module 06 — Observability Dashboard (15–20 min)

**Theme:** "Is this system ready for production?"

Participants view an operational dashboard aggregating all telemetry: API costs by module, latency distributions, platform health, and a production readiness assessment.

**Key Concepts:**
- AI-specific observability requirements
- Cost attribution and optimization
- Latency percentile analysis
- Operational readiness scoring

**Azure Focus:** Azure Monitor integration, Application Insights patterns

---

## 🎯 Facilitator Notes

### Before the Workshop

1. **Test Azure Access:** Verify all participants have Azure OpenAI access with a GPT-4o deployment
2. **Pre-generate Data:** Run `generate_synthetic_data.py` to verify it completes without errors
3. **Budget Estimate:** Full workshop run costs approximately $0.50–$2.00 in Azure OpenAI tokens (GPT-4o pricing)
4. **Ground Truth:** The `ground_truth.json` file contains the answer key — decide when to reveal it

### During the Workshop

- **Module 01:** Let participants discover anomalies independently before discussing
- **Module 02:** Highlight the difference between free-text and structured outputs
- **Module 03:** Show how token costs grow with each conversation turn
- **Module 04:** Discuss responsible AI — what happens if the LLM is wrong?
- **Module 05:** This is the "aha" moment — measuring AI accuracy quantitatively
- **Module 06:** Connect observability to production deployment readiness

### Discussion Prompts

- "What would happen if we set temperature to 1.0 for fraud detection?"
- "How would you handle this at PayPal scale (billions of transactions)?"
- "When should a human review the LLM's findings before acting?"
- "How would you deploy this — batch processing or real-time streaming?"
- "What if the fraud ring adapts to the LLM's detection patterns?"

### Answer Key

The ground truth is in `data/ground_truth.json`. It contains:
- All 18 fraud ring member customer IDs
- All 4 shell merchant IDs
- The 7 embedded signals
- Device sharing pairs
- Impossible travel accounts
- Circular flow chain definitions

---

## 📊 Dataset Specifications

| File | Format | Records | Key Fields |
|---|---|---|---|
| `transactions.csv` | CSV | ~50,000 | transaction_id, timestamp, sender_id, receiver_id, merchant_id, amount, currency, status |
| `customers.csv` | CSV | 2,000 | customer_id, email, city, registration_date, risk_score, verified_identity |
| `merchants.csv` | CSV | 500 | merchant_id, business_name, dba_name, mcc_code, registration_date, chargeback_rate |
| `payment_sessions.json` | JSON | 15,000 | session_id, customer_id, device.fingerprint, geolocation.city, risk_assessment |
| `dispute_cases.json` | JSON | 350 | dispute_id, transaction_id, reason, evidence_thread[], resolution |
| `risk_signals.json` | JSON | 8,000 | signal_id, transaction_id, overall_risk_score, rule_results{} |
| `velocity_metrics.csv` | CSV | ~2,200/hr | timestamp, transaction_count, total_amount_usd, high_value_count |
| `system_telemetry.csv` | CSV | ~13,000 | timestamp, service, avg_latency_ms, error_rate_pct, requests_per_minute |

---

## 🔒 Security & Responsible AI

- **No real data** is used — all datasets are synthetically generated
- **No PII** — names, emails, and IDs are randomly generated
- `.env` files are gitignored — credentials never enter version control
- The workshop emphasizes **human-in-the-loop** review for all AI-generated findings
- Module 05's evaluation framework demonstrates the importance of measuring AI accuracy before acting on results

---

## License

This workshop accelerator is provided for educational and demonstration purposes.
