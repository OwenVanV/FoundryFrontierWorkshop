<!-- 
============================================================================
PayPal x Azure AI Workshop — Presentation Deck
============================================================================
Format: Each slide is separated by ---
Each slide has: title, content, speaker notes, and style instructions
Use with any Markdown-to-slides tool (Marp, Slidev, reveal.js, etc.)
============================================================================
-->

---

<!-- SLIDE 1: Title -->

# Cracking the Ring
## PayPal x Azure AI Workshop

**Uncover a hidden fraud ring using Azure AI — from raw LLMs to voice-controlled computer agents.**

- 50,000+ synthetic transactions
- 2,000 customers · 500 merchants
- 18 hidden fraud ring members · 4 shell companies
- 7 interlocking signals — no single one is conclusive

*Two-day hands-on workshop*

<!-- SPEAKER NOTES:
Welcome everyone. Over the next two days, we'll use Azure AI to investigate a sophisticated fraud ring hidden in a realistic payment processing dataset. The fraud is designed to evade traditional rules — only AI can crack it by reasoning across multiple signals simultaneously.
-->

<!-- STYLE: Dark navy background (#0a1628). Large bold white title. PayPal blue (#003087) accent line below subtitle. Azure logo in bottom-right corner. Clean, professional, minimal. All slides should follow this dark theme consistently. -->

---

<!-- SLIDE 2: Workshop Overview -->

# Workshop Journey

| Day 1 | Day 2 |
|---|---|
| **Track 1:** LLM Deep-Dive | **Track 4:** Voice + CUA |
| **Track 2:** Agents | *Talk to a computer.* |
| **Track 3:** Search + RAG | *Watch it work.* |

```
Day 1: Investigate the fraud ring
  LLM → Agents → Search + RAG

Day 2: Build the operator
  CUA → Voice Live → Full System
```

<!-- SPEAKER NOTES:
Day 1 is about investigation — three progressively sophisticated approaches to finding the fraud ring. Day 2 is about building a voice-controlled system that can both query data and control a computer. Each track builds on the shared synthetic dataset.
-->

<!-- STYLE: Dark navy background. Two-column layout. Left column "Day 1" in PayPal blue, right column "Day 2" in Azure blue (#0078D4). Use a horizontal timeline arrow connecting the four tracks at the bottom. -->

---

<!-- SLIDE 3: The Dataset -->

# The Payment Landscape

| Dataset | Format | Records | What's Inside |
|---|---|---|---|
| Transactions | CSV | 50,000 | Amounts, senders, receivers, merchants, timestamps |
| Customers | CSV | 2,000 | Profiles, registration dates, emails, risk scores |
| Merchants | CSV | 500 | Business names, MCC codes, registration, chargebacks |
| Payment Sessions | JSON | 15,000 | Device fingerprints, geolocation, session actions |
| Dispute Cases | JSON | 350 | Chargebacks, evidence threads, resolutions |
| Risk Signals | JSON | 8,000 | Rule-by-rule scoring breakdowns |
| Velocity Metrics | CSV | 2,200/hr | Hourly aggregations, high-value counts |
| System Telemetry | CSV | 13,000 | Latency, error rates, throughput by service |

**All deterministically generated. Seed = 42. Every participant gets identical data.**

<!-- SPEAKER NOTES:
The data is realistic — power-law transaction distributions, timezone-aware business hours, real MCC codes, actual PayPal corridor currency mixes. The fraud ring is embedded programmatically, not hand-placed. Let's look at what's hiding in this data.
-->

<!-- STYLE: Dark navy background. Table with alternating dark row shading. Each format type (CSV, JSON) has a small colored badge. Monospace font for numbers. -->

---

<!-- SLIDE 4: The Hidden Fraud Ring -->

# The Hidden Fraud Ring
## 18 accounts · 4 shell merchants · 7 signals

The ring represents **~0.84%** of all transactions.
A needle in a 50,000-transaction haystack.

**Why traditional rules miss it:**
Every individual signal is plausible on its own.
Only cross-referencing reveals the ring.

<!-- SPEAKER NOTES:
This is the key insight of the workshop. The fraud ring was specifically designed to evade rules-based detection. Each signal — the amount clustering, the registration timing, the device sharing — looks innocent in isolation. The power of LLMs is reasoning across all signals simultaneously.
-->

<!-- STYLE: Dark navy background. Center the "18 · 4 · 7" as large highlight numbers in PayPal blue. Use a faint network graph visualization in the background showing interconnected nodes. -->

---

<!-- SLIDE 5: The 7 Signals -->

# 7 Interlocking Signals

| # | Signal | Where It Hides | Why It's Subtle |
|---|---|---|---|
| 1 | **Smurfing** | transactions.csv | $9,200–$9,800 with realistic cents |
| 2 | **Circular Flow** | transactions.csv | A→B→C→D→A with 48-72hr delays |
| 3 | **Shell Merchants** | merchants.csv | "Apex Digital Solutions" / "Apex Digitial Svcs" |
| 4 | **Device Sharing** | sessions.json | 6 accounts share 3 fingerprints |
| 5 | **Impossible Travel** | sessions.json | Lagos → London in 20 minutes |
| 6 | **Synthetic Identity** | customers.csv | 2-week registration burst, obscure email domains |
| 7 | **Temporal Coordination** | transactions.csv | 3-7 minute burst windows |

<!-- SPEAKER NOTES:
Walk through each signal. Emphasize that signal 1 (smurfing) is the most detectable but still hard because the amounts aren't round numbers. Signal 2 (circular flow) requires joining transactions across 48-72 hour windows. Signal 4 (device sharing) is buried in nested JSON. An LLM can reason about all of these simultaneously.
-->

<!-- STYLE: Dark navy background. Table with each signal number in a colored circle. Use a gradient from yellow (easy to detect) to red (hard to detect) across the rows. Small icons next to each signal name. -->

---

<!-- SLIDE 6: Track 1 Header — LLM Deep-Dive -->

# Track 1
## LLM Deep-Dive

**Raw Azure OpenAI — no agents, no frameworks.**

Six modules. Direct API calls. Streaming with TTFT measurement.
Prompt engineering from exploration to evaluation.

```
01 Data Exploration → 02 Pattern Detection → 03 Cross-Reference
    → 04 Investigation → 05 Evaluation → 06 Observability
```

<!-- SPEAKER NOTES:
Track 1 is the foundation. We use Azure OpenAI directly — call_openai() with streaming, token tracking, and telemetry. No agent abstractions. This shows what the raw model can do with good prompt engineering.
-->

<!-- STYLE: Dark navy background. "Track 1" in large font with a "01" watermark in light gray. Azure OpenAI logo. A horizontal flow diagram showing the 6 modules as connected nodes. -->

---

<!-- SLIDE 7: LLM Module 01 — Data Exploration -->

# Module 01: Data Exploration

**Use the LLM as an intelligent data profiling tool.**

- Load 50K transactions, compute statistical profiles
- Send summaries to GPT-4o with a "senior payment analyst" persona
- LLM identifies anomalies in amount distributions, demographics
- Every call streams live with TTFT measurement

**Key Question:** *"What does this data tell us before we even look for fraud?"*

**Azure Services:** Azure OpenAI (chat completions), OpenTelemetry

<!-- SPEAKER NOTES:
The key insight here: LLMs can't process 50K rows, so we pre-aggregate and let the model reason over summaries. The model should notice the $9K-$10K clustering and the email domain concentrations. Watch the TTFT — first token arrives in ~200ms, full response streams over 5-10 seconds.
-->

<!-- STYLE: Dark navy background. Show a mock terminal screenshot with streaming output — the box-drawing characters and TTFT footer line. Code snippet styling for the call_openai() pattern. -->

---

<!-- SLIDE 8: LLM Module 02 — Pattern Detection -->

# Module 02: Pattern Detection

**From free text to structured, parseable JSON.**

- `response_format={"type": "json_object"}` forces structured output
- Zero-shot smurfing detection in the $9K-$10K range
- Temporal analysis: coordinated burst windows
- Merchant shell company detection via name similarity

**Output:** Each anomaly has a `type`, `confidence`, and `evidence` — ready for automation.

```json
{
  "type": "smurfing",
  "confidence": 0.87,
  "evidence": {"sender_ids": ["CUS-..."], "amount_range": {"min": 9200, "max": 9800}}
}
```

<!-- SPEAKER NOTES:
This is where we transition from "interesting analysis" to "actionable intelligence." JSON mode means the output feeds directly into downstream systems. Point out the confidence scores — the model self-assesses its certainty.
-->

<!-- STYLE: Dark navy background. JSON code block with syntax highlighting in the center. Confidence score visualized as a gauge/meter. Use green for high confidence, yellow for medium. -->

---

<!-- SLIDE 9: LLM Module 03 — Cross-Reference -->

# Module 03: Cross-Reference Analysis

**Multi-turn conversations connecting signals across datasets.**

- Device fingerprint sharing (nested JSON parsing)
- Impossible travel detection (Lagos → London, 20 min)
- 4-turn investigation conversation with chain-of-thought
- Token costs **grow linearly** with each turn

**Key Technique:** *"Think step by step"* — chain-of-thought prompting
dramatically improves cross-referencing accuracy.

<!-- SPEAKER NOTES:
Watch the token usage grow with each turn. Turn 1 might be 2K tokens, Turn 4 is 8K+ because the full conversation history is sent every time. This is the cost of context accumulation — essential for investigation, expensive at scale. This is why we'll look at agents in Track 2.
-->

<!-- STYLE: Dark navy background. Show 4 conversation bubbles stacking vertically, each slightly wider than the last (representing growing context). Token counter in the corner incrementing. -->

---

<!-- SLIDE 10: LLM Modules 04-06 -->

# Modules 04-06: Investigate → Evaluate → Observe

**04 — Fraud Ring Investigation**
Network graph reasoning. Formal investigation report generation.
*"This report could freeze real accounts — what oversight is needed?"*

**05 — Evaluation Framework**
Precision / Recall / F1 against ground truth.
4 prompt variants tested side-by-side.
*"Small prompt changes → dramatic accuracy shifts."*

**06 — Observability Dashboard**
Token costs by module. Latency percentiles. Production readiness scoring.
*"Is this system ready for production?"*

<!-- SPEAKER NOTES:
Module 04 generates a formal report — suitable for regulators. Module 05 is the "aha moment" — measuring AI accuracy quantitatively. Module 06 ties it all together with operational metrics. After this track, participants understand what raw LLMs can and can't do. Track 2 adds agent capabilities.
-->

<!-- STYLE: Dark navy background. Three horizontal panels stacked, each with an icon (magnifying glass, ruler, dashboard). Subtle connecting arrows between them. -->

---

<!-- SLIDE 11: Track 2 Header — Agents -->

# Track 2
## Agents Deep-Dive

**Two systems, one investigation.**

| System | Package | What It Does |
|---|---|---|
| 🟦 **Foundry Agent Service** | `azure-ai-projects` | Managed agents with Code Interpreter |
| 🟩 **Microsoft Agent Framework** | `agent-framework` | Local agents with custom @tool functions |

```
Foundry Agent Service = "Agent as a Service" (managed)
MAF = "Agent as a Library" (you control everything)
```

<!-- SPEAKER NOTES:
Track 2 introduces two distinct agent systems. Foundry Agent Service gives you managed agents with server-side tools like Code Interpreter. MAF gives you a local SDK where you define custom tools as Python functions. We use both because they excel at different things.
-->

<!-- STYLE: Dark navy background. Two large badges side by side — blue for Foundry, green for MAF. Each with the package name below. Use a "vs" or "+" symbol between them. -->

---

<!-- SLIDE 12: Agents — Foundry Agent Service -->

# Foundry Agent Service
## Code Interpreter: A Data Scientist in a Box

- Upload CSVs to Azure-managed storage
- Agent writes pandas code autonomously
- Executes in a **Hyper-V sandbox** — safe for untrusted code
- Downloads generated charts and visualizations

```python
agent = project.agents.create_version(
    agent_name="FraudDataAnalyst",
    definition=PromptAgentDefinition(
        model="gpt-4o",
        tools=[CodeInterpreterTool(container=AutoCodeInterpreterToolParam(file_ids=file_ids))],
    ),
)
```

<!-- SPEAKER NOTES:
The agent writes pandas code, executes it, iterates on errors, and generates artifacts — all in a Hyper-V sandbox. It can build network graphs, compute statistics, and generate matplotlib charts. This is what Module 01 and 03 of the agents track demonstrate.
-->

<!-- STYLE: Dark navy background. Code snippet in the center with Python syntax highlighting. Show a small chart thumbnail being "generated" by the sandbox. Blue accent for Foundry branding. -->

---

<!-- SLIDE 13: Agents — MAF -->

# Microsoft Agent Framework
## Custom Tools with @tool

```python
@tool(approval_mode="never_require")
def query_transactions(
    sender_id: Annotated[str, Field(description="Customer ID")],
    min_amount: Annotated[float, Field(description="Minimum amount")],
) -> str:
    """Query the transaction ledger with filters."""
    # Your code runs HERE — query CSV, call API, anything
    ...
```

- `@tool` auto-generates JSON schemas from type hints
- `AgentSession` maintains context across turns
- Functional workflows compose multiple agents
- **TransactionAnalyst → IdentityInvestigator → ReportWriter**

<!-- SPEAKER NOTES:
The @tool decorator is the magic. You write a Python function with type hints and a docstring, and MAF automatically generates the tool schema the LLM sees. The model decides when to call your function and with what arguments. Module 05 chains three specialist agents in a workflow.
-->

<!-- STYLE: Dark navy background. Code block with green MAF accent color. Show three agent icons connected by arrows for the workflow. -->

---

<!-- SLIDE 14: Track 3 Header — Search + RAG -->

# Track 3
## Search + RAG

**Extract evidence from documents. Index it. Unleash an agent.**

```
10 Fraud Documents → Content Understanding → Azure AI Search → MAF Agent
     (SARs, memos,       (extract fields,       (index + embed,     (RAG
      alerts, audits)      classify, generate)    vector search)    investigation)
```

**Three Azure services working together:**
- Azure Content Understanding — extract structured fields from unstructured documents
- Azure AI Search — full-text + vector + filtered search
- Microsoft Agent Framework — RAG-powered investigation agent

<!-- SPEAKER NOTES:
This track demonstrates a real-world RAG pipeline. The fraud team has internal documents — SARs, compliance memos, security alerts — that contain scattered evidence. No single document tells the whole story. Content Understanding extracts fields with confidence scores. AI Search makes it all searchable. The agent connects the dots.
-->

<!-- STYLE: Dark navy background. Horizontal pipeline diagram with three service logos connected by arrows. Document icons on the left flowing into a search index, then into an agent. -->

---

<!-- SLIDE 15: The Fraud Documents -->

# 10 + 4 Fraud Evidence Documents

| Document | Type | Key Evidence |
|---|---|---|
| SAR-2026-0142 | Suspicious Activity Report | Names all 4 shell merchants |
| Compliance Memo Q1 | Internal Memo | Delaware cluster noted, "no action" |
| Due Diligence — Apex | Merchant Onboarding | Registered agent address, not real office |
| Audit Q1 High-Value | Transaction Audit | $9.2K-$9.8K clustering, temporal bursts |
| Onboarding Feb 2026 | Customer Batch | 18 accounts, quickinbox.io emails |
| Device Fingerprint Alert | Security Alert | 3 devices shared across 6 accounts |
| SAR-2026-0198 | SAR | Full circular flow chains with IDs |
| Risk Engine Report | Performance Report | Known gaps in detection rules |
| Impossible Travel Alert | Geolocation Alert | Lagos→London 15 min |
| Dispute Summary | Chargebacks | Third-party victims, ring files zero disputes |
| + 4 AI-generated images | Email, notes, report, chat | Visual evidence via gpt-image-2 |

<!-- SPEAKER NOTES:
Each document is written from a different internal team's perspective — compliance, IT security, fraud analytics, dispute resolution. The evidence is scattered. The SAR names the merchants but not all the customers. The onboarding summary names the customers but doesn't connect them to merchants. The agent must search across all documents.
-->

<!-- STYLE: Dark navy background. Compact table with document type badges (red for SAR, blue for memo, yellow for alert). Small thumbnail images for the 4 AI-generated images at the bottom. -->

---

<!-- SLIDE 16: Content Understanding -->

# Azure Content Understanding
## From Unstructured to Structured

**Custom Analyzer with 11 fields:**

| Field | Method | What It Does |
|---|---|---|
| document_type | classify | Categorize: SAR, memo, alert, audit... |
| merchant_names | extract | Pull merchant names from text |
| customer_ids | extract | Find CUS-XXXX patterns |
| risk_indicators | generate | LLM synthesizes fraud signals |
| summary | generate | One-paragraph investigation relevance |

**Three methods:** `extract` (from text) · `classify` (categorize) · `generate` (LLM-synthesized)

**Confidence scores + source grounding** on every extracted field.

<!-- SPEAKER NOTES:
Content Understanding is the key differentiator. The 'generate' fields are especially powerful — they capture insights that aren't explicitly stated but are inferred by GPT-4.1. The confidence scores tell you how much to trust each extraction. Source grounding tells you exactly where in the document the value came from.
-->

<!-- STYLE: Dark navy background. Table with method types color-coded (green=extract, blue=classify, purple=generate). Show a document with highlighted regions mapping to extracted fields. -->

---

<!-- SLIDE 17: The RAG Agent -->

# The Investigation Agent
## Documents + Transaction Data = Complete Picture

```
Tools:
  🔍 search_fraud_evidence  → Azure AI Search (documents)
  📄 get_document_content   → Full document text
  💳 query_transactions     → CSV transaction data
  👤 lookup_customer        → Customer profiles
```

**4-phase investigation:**
1. Document Discovery — *"What documents mention Apex?"*
2. Cross-Reference — *"Do transactions confirm what the SARs describe?"*
3. Evidence Synthesis — *"Which accounts appear in multiple documents?"*
4. Final Report — *"Write the investigation brief with citations."*

<!-- SPEAKER NOTES:
The agent combines two data sources — document evidence and transaction data — to build a more complete picture than either alone. SARs provide human analyst assessments. Audit logs show what rules missed. The agent cross-references and synthesizes. Every claim in the final report cites a specific document filename.
-->

<!-- STYLE: Dark navy background. Four tool icons on the left connected to a central "Agent" brain icon. Four phase boxes flowing downward on the right. -->

---

<!-- SLIDE 18: Day 2 Separator -->

# Day 2
## From Investigation to Operation

Day 1: We **found** the fraud ring.
Day 2: We **build the operator** that finds the next one.

*Talk to a computer. Watch it work.*

```
Voice Live API + GPT-5.4 Computer Use Agent
= Voice-controlled fraud investigation
```

<!-- SPEAKER NOTES:
Day 1 was about investigation techniques. Day 2 is about building an operational system. We combine Azure Voice Live (speech-to-speech) with a Computer Use Agent (GPT-5.4 controlling a Docker container). By the end, you'll speak to an AI that both queries data and controls a browser.
-->

<!-- STYLE: Dark navy background with a dramatic visual break. Large "Day 2" centered. A waveform audio visualization transitioning into a computer screen. Gold accent color for Day 2 to distinguish from Day 1's blue. -->

---

<!-- SLIDE 19: Track 4 Header — Voice + CUA -->

# Track 4
## Voice + CUA

**Progressive journey: 5 modules, each adds one capability.**

```
Module 01: [CUA Agent] → Docker          ← Type a task, watch it work
     ↓
Module 02: [Voice Live] → Chat           ← Talk to an AI
     ↓
Module 03: [Voice Live] → [Tools]        ← Voice calls functions
     ↓
Module 04: [Voice Live] → [CUA]          ← Voice controls computer
     ↓
Module 05: [Voice + Tools + CUA]         ← Everything unified
```

<!-- SPEAKER NOTES:
Each module adds exactly one capability on top of the previous one. Module 01 is CUA alone — no voice. Module 02 is voice alone — no CUA. Module 03 adds function calling to voice. Module 04 bridges voice to CUA. Module 05 combines everything. This progressive approach makes each new concept digestible.
-->

<!-- STYLE: Dark navy background. Vertical flow diagram with five steps, each step slightly larger/more colorful than the last. Use progressive color intensity from gray to gold. -->

---

<!-- SLIDE 20: CUA — Computer Use Agent -->

# Computer Use Agent (CUA)
## GPT-5.4 Controls a Computer

```
The CUA Loop:
  screenshot → GPT-5.4 → action → screenshot → GPT-5.4 → action → ...
```

- Takes screenshots of a Docker container (Xfce + Firefox)
- Model decides: click, type, scroll, keypress
- Actions executed via `xdotool` in the container
- **SupervisorGPT** (GPT-4.1) monitors and terminates

**Watch it work:** VNC viewer on `localhost:5900`

<!-- SPEAKER NOTES:
The CUA agent sees the screen as an image, reasons about what to do, and takes actions. It's not using DOM or accessibility APIs — it literally looks at the pixels. GPT-5.4 is specifically trained for this. A supervisor agent monitors progress and decides when the task is done. Open the VNC viewer to watch it navigate Firefox in real-time.
-->

<!-- STYLE: Dark navy background. Show a circular loop diagram: Screenshot → Model → Action → Screenshot. Include a small VNC viewer mockup showing a Firefox browser. -->

---

<!-- SLIDE 21: Voice Live API -->

# Azure Voice Live API
## Fully Managed Speech-to-Speech

**What makes Voice Live special:**

| Feature | What It Does |
|---|---|
| `azure_semantic_vad` | Knows when you're done speaking by *meaning*, not silence |
| `remove_filler_words` | Ignores "um", "uh" during active response |
| `server_echo_cancellation` | Doesn't hear its own voice back |
| `azure_deep_noise_suppression` | Works in noisy demo environments |
| HD Voices | `en-US-Ava:DragonHDLatestNeural` — sounds human |
| Function Calling | Standard Realtime API tool pattern |

**No model deployment needed.** Voice Live is fully managed.

<!-- SPEAKER NOTES:
Voice Live is the key enabler. The semantic VAD is a game-changer — it understands that you're mid-sentence even if you pause to think. The echo cancellation means the AI doesn't trigger itself. And the HD voices sound remarkably natural. All over a single WebSocket connection.
-->

<!-- STYLE: Dark navy background. Feature table with small icons for each capability. Show a waveform visualization at the bottom. Azure Speech logo. -->

---

<!-- SLIDE 22: The Bridge -->

# The Bridge: Voice → Computer

```
[Your Voice] → Voice Live API → GPT-5 → function_call
                                              ↓
                                    control_computer(task)
                                              ↓
                                    run_cua() → Docker
                                              ↓
[Your Ears] ← Voice Live API ← GPT-5 ← action log
```

**Say:** *"Go to Google and search for PayPal fraud detection"*

The voice agent calls `control_computer()`, which invokes the CUA agent,
which controls the Docker container, then reports back —
and the voice agent **speaks** the result.

<!-- SPEAKER NOTES:
This is the magic moment. You speak a command. The Voice Live model decides to call the control_computer function. Your code invokes the CUA agent. GPT-5.4 takes screenshots and controls Firefox. The action log flows back through the function call response. The voice agent narrates what happened. All in real-time.
-->

<!-- STYLE: Dark navy background. Architecture diagram with arrows flowing from a microphone icon through Voice Live, to a brain icon, to a Docker container icon, and back to a speaker icon. Use gold arrows for the critical path. -->

---

<!-- SLIDE 23: Full System -->

# The Complete System
## Voice Fraud Investigation Operator

**4 tools unified in one voice interface:**

| Tool | What It Does |
|---|---|
| `control_computer` | Navigate websites via CUA → Docker |
| `query_transactions` | Search 50,000+ transaction database |
| `search_merchants` | Find merchants by name |
| `lookup_customer` | Get customer profile by ID |

**Example session:**
1. *"Search our database for merchants named Apex"* → `search_merchants`
2. *"How many transactions go through those merchants?"* → `query_transactions`
3. *"Open Firefox and Google 'Apex Digital Solutions Delaware'"* → `control_computer`
4. *"Look up customer CUS-A08563C93C2B"* → `lookup_customer`

<!-- SPEAKER NOTES:
This is the culmination. Natural voice conversation with an AI that can both query your data and control a computer. The model decides which tool to use based on what you say. Data queries are instant. Computer tasks take 30-60 seconds while the CUA agent works. The voice agent narrates throughout.
-->

<!-- STYLE: Dark navy background. Four tool cards arranged in a 2x2 grid, each with an icon and description. Below, show a conversation flow with speech bubbles. Gold accent. -->

---

<!-- SLIDE 24: Azure Services Summary -->

# Azure Services Demonstrated

| Service | Where It's Used |
|---|---|
| **Azure OpenAI** | GPT-4o (reasoning), GPT-5.4 (CUA), gpt-image-2 |
| **Azure AI Foundry** | Agent Service, model deployment, project management |
| **Microsoft Agent Framework** | Custom agents, @tool functions, workflows |
| **Voice Live API** | Speech-to-speech, semantic VAD, HD voices |
| **Azure Content Understanding** | Document field extraction, classification |
| **Azure AI Search** | Full-text + vector + filtered search index |
| **Azure Monitor** | Application Insights, telemetry export |
| **OpenTelemetry** | Distributed tracing, metrics collection |

<!-- SPEAKER NOTES:
Across both days, we've demonstrated 8 Azure services working together. The key architectural pattern: each service does one thing well, and they compose into powerful systems. Azure OpenAI provides the reasoning. Foundry provides the infrastructure. Voice Live provides the interface. Content Understanding + AI Search provide the data pipeline.
-->

<!-- STYLE: Dark navy background. Clean table with Azure service logos in the left column. Group by Day 1 (blue rows) and Day 2 (gold rows). -->

---

<!-- SLIDE 25: Key Takeaways -->

# Key Takeaways

1. **LLMs excel at cross-referencing** signals across heterogeneous data — things that are painful to write rules for.

2. **Structured outputs enable automation**, not just analysis — JSON mode makes LLM output machine-readable.

3. **Agents + tools > raw prompting** — custom functions let the model access live data instead of reasoning from summaries.

4. **Evaluation is essential** — without precision/recall/F1, you're flying blind. Measure everything.

5. **Voice is the next interface** — natural language + function calling + computer use = a new category of operator.

6. **Observability is not optional** — instrument every call, track every token, know your costs.

<!-- SPEAKER NOTES:
These are the six things I want everyone to walk away remembering. Each one maps to a specific track in the workshop. LLMs for investigation, agents for tool use, evaluation for trust, voice for the interface, observability for production.
-->

<!-- STYLE: Dark navy background. Six numbered takeaways in a clean list. Each number in a PayPal blue circle. Use a subtle gradient background from dark navy at top to slightly lighter at bottom. -->

---

<!-- SLIDE 26: Getting Started -->

# Get Started

```bash
# Clone and install
cd PayPal
uv sync --extra all

# Generate the data
uv run python data/generate_synthetic_data.py

# Launch a demo app
uv run streamlit run demo/streamlit_llm.py
uv run streamlit run demo/streamlit_agents.py
uv run streamlit run demo/streamlit_search_rag.py
uv run streamlit run demo/streamlit_voice_cua.py
```

**Prerequisites:**
- Python 3.10+ · `uv` package manager
- Azure subscription with OpenAI access
- `az login` for authentication
- Docker (Day 2 only, for CUA)

<!-- SPEAKER NOTES:
Everything runs through uv. One command installs all dependencies. The Streamlit apps provide a visual interface for each track — you can select a module, read the description, see the code, and run it with one click. The stop button lets you abort long-running modules.
-->

<!-- STYLE: Dark navy background. Terminal/code block styling. Large monospace font for the commands. Green checkmarks next to each prerequisite. -->

---

<!-- SLIDE 27: Thank You -->

# Thank You

**PayPal x Azure AI Workshop**
*Cracking the Ring*

Questions? Let's investigate together.

<!-- SPEAKER NOTES:
Open the floor for questions. Have the Streamlit demos ready to show specific modules on demand. The Foundry portal can demonstrate the playground experience for anyone who wants to compare the API approach with the UI approach.
-->

<!-- STYLE: Dark navy background. Simple, clean closing slide. PayPal logo on the left, Azure logo on the right. Subtle fraud ring network graph visualization in the background at very low opacity. -->
