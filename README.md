# 🔍 Cracking the Ring — PayPal x Azure AI Workshop

> **Uncover a hidden fraud ring using Azure AI — from raw LLMs to voice-controlled computer agents.**

A two-day hands-on workshop demonstrating Azure AI capabilities through the lens of payment fraud investigation. 50,000+ synthetic transactions, a hidden 18-member fraud ring, and 4 progressive tracks that build from direct LLM calls to a voice-controlled computer operator.

---

## Workshop Tracks

| Track | Folder | Focus | Azure Services |
|---|---|---|---|
| **LLM Deep-Dive** | [`llm/`](llm/) | Raw Azure OpenAI — streaming, structured outputs, evaluation | Azure OpenAI, OpenTelemetry |
| **Agents** | [`agents/`](agents/) | Foundry Agent Service + Microsoft Agent Framework | Agent Service, MAF, Code Interpreter |
| **Search + RAG** | [`search_rag/`](search_rag/) | Document extraction → search index → RAG agent | Content Understanding, AI Search, MAF |
| **Voice + CUA** | [`Voice_CUA/`](Voice_CUA/) | Voice Live + Computer Use Agent | Voice Live API, GPT-5.4 CUA |

Each track has its own **README** with a detailed walkthrough, setup instructions, and module-by-module guide.

---

## The Hidden Fraud Ring

Buried in the synthetic data: **18 accounts** operating through **4 shell merchants** with **7 interlocking signals**:

| Signal | Where | What |
|---|---|---|
| Smurfing | transactions | $9,200–$9,800 (below $10K BSA threshold) |
| Circular Flow | transactions | A→B→C→D→A with 48-72hr delays |
| Shell Merchants | merchants | "Apex Digital Solutions" / "Apex Digitial Svcs" |
| Device Sharing | sessions | 6 accounts share 3 device fingerprints |
| Impossible Travel | sessions | Lagos → London in 20 minutes |
| Synthetic Identity | customers | 2-week registration burst, obscure email domains |
| Temporal Coordination | transactions | 3-7 minute burst windows |

No single signal is conclusive. Only cross-referencing reveals the ring.

---

## Quick Start

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# or

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install all dependencies
uv sync --extra all

# Generate the synthetic dataset
uv run python data/generate_synthetic_data.py

# Launch all 4 demo apps
./run_all.sh          # macOS/Linux
# .\run_all.ps1       # Windows PowerShell

# Or launch one track at a time
uv run streamlit run demo/streamlit_llm.py        # LLM Track
uv run streamlit run demo/streamlit_agents.py     # Agents Track
uv run streamlit run demo/streamlit_search_rag.py # Search + RAG Track
uv run streamlit run demo/streamlit_voice_cua.py  # Voice + CUA Track
```

> **Windows users:** Install uv with `irm https://astral.sh/uv/install.ps1 | iex`, then restart your terminal. If you see `charmap` encoding errors, set `$env:PYTHONIOENCODING = "utf-8"` before running.

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Azure subscription with OpenAI access
- `az login` for authentication (or set `USE_MANAGED_IDENTITY=true`)
- Docker (Voice + CUA track only)

### Configuration

```bash
# Copy the environment template
cp .env.example .env
# Edit with your Azure OpenAI endpoint and credentials

# Each track has its own .env for track-specific settings
cp agents/.env.example agents/.env
cp search_rag/.env.example search_rag/.env
cp Voice_CUA/.env.example Voice_CUA/.env
```

---

## Project Structure

```
PayPal/
├── README.md                      ← You are here
├── pyproject.toml                 ← Dependencies (uv sync --extra all)
├── run_all.sh / run_all.ps1       ← Launch all 4 Streamlit demos
├── deck.md                        ← Presentation slides (27 slides)
│
├── data/                          ← Synthetic dataset generator
│   └── generate_synthetic_data.py ← Seed=42, deterministic
│
├── utils/                         ← Shared utilities (all tracks)
│   ├── auth.py                    ← Managed identity / API key auth
│   ├── azure_client.py            ← Streaming OpenAI client with TTFT
│   └── telemetry.py               ← OpenTelemetry + Azure Monitor
│
├── llm/                           ← Track 1: LLM Deep-Dive (6 modules)
│   ├── README.md                  ← Track walkthrough
│   ├── 01_data_exploration.py
│   ├── 02_pattern_detection.py
│   ├── 03_cross_reference_analysis.py
│   ├── 04_fraud_ring_investigation.py
│   ├── 05_evaluation_framework.py
│   └── 06_observability_dashboard.py
│
├── agents/                        ← Track 2: Agents (6 modules)
│   ├── README.md                  ← Track walkthrough
│   ├── 01_foundry_agent_setup.py  ← Foundry Agent Service
│   ├── 02_maf_first_agent.py      ← Microsoft Agent Framework
│   ├── 03_foundry_code_analysis.py
│   ├── 04_maf_multi_turn_investigation.py
│   ├── 05_maf_workflow_orchestration.py
│   ├── 06_evaluation.py
│   └── utils/fraud_tools.py       ← @tool-decorated functions
│
├── search_rag/                    ← Track 3: Search + RAG (5 modules)
│   ├── 01_create_analyzer.py      ← Content Understanding
│   ├── 02_analyze_documents.py
│   ├── 03_build_search_index.py   ← Azure AI Search
│   ├── 04_agent_investigation.py  ← RAG agent
│   ├── 05_evaluation.py
│   └── data/generate_documents.py ← 10 text + 4 image docs
│
├── Voice_CUA/                     ← Track 4: Voice + CUA (5 modules)
│   ├── README.md                  ← Track walkthrough
│   ├── 01_cua_basics/             ← CUA only
│   ├── 02_voice_live_intro/       ← Voice only
│   ├── 03_voice_with_functions/   ← Voice + tools
│   ├── 04_voice_cua_bridge/       ← Voice → CUA bridge
│   ├── 05_full_system/            ← Everything unified
│   └── shared/                    ← Audio helpers, CUA client
│
├── computer-use/                  ← CUA Docker container + agent
│   ├── Dockerfile                 ← Xfce + Firefox + VNC
│   ├── cua.py                     ← GPT-5.4 computer use agent
│   ├── docker_c.py                ← Docker container interface
│   └── shadowbox/                 ← Enhanced CUA package
│
├── demo/                          ← Streamlit demo apps
│   ├── streamlit_llm.py
│   ├── streamlit_agents.py
│   ├── streamlit_search_rag.py
│   ├── streamlit_voice_cua.py
│   └── shared.py                  ← Subprocess streaming, UI helpers
│
└── tests/
    └── test_workshop.py           ← 52 tests
```

---

## Azure Services

| Service | Tracks | Purpose |
|---|---|---|
| **Azure OpenAI** | All | GPT-4o reasoning, GPT-5.4 CUA, gpt-image-2 |
| **Azure AI Foundry** | Agents | Agent Service, model deployment |
| **Microsoft Agent Framework** | Agents, Search | Custom agents, @tool functions, workflows |
| **Voice Live API** | Voice+CUA | Speech-to-speech, semantic VAD, HD voices |
| **Content Understanding** | Search | Document field extraction, classification |
| **Azure AI Search** | Search | Full-text + vector + filtered search |
| **Azure Monitor** | LLM | Application Insights, telemetry |
| **OpenTelemetry** | LLM | Distributed tracing, metrics |

---

## Authentication

All tracks support two auth modes via `USE_MANAGED_IDENTITY` in `.env`:

| Mode | Setting | How |
|---|---|---|
| **API Key** | `USE_MANAGED_IDENTITY=false` | Set `AZURE_OPENAI_API_KEY` |
| **Managed Identity** | `USE_MANAGED_IDENTITY=true` | `az login` locally, or auto on Azure |

---

## Tests

```bash
uv run python tests/test_workshop.py    # 52 tests
uv run pytest tests/ -v                 # via pytest
```

---

## License

This workshop accelerator is provided for educational and demonstration purposes.
