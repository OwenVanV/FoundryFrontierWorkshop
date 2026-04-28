# Voice + CUA — Workshop Walkthrough

> **Talk to a computer and watch it work. Combine Azure Voice Live API with a GPT-5.4 Computer Use Agent to build a voice-controlled fraud investigation system.**

This module is a progressive 5-step journey from basic computer control to a fully voice-operated investigation system. Each module adds one capability on top of the last.

---

## The Vision

By Module 05, you will **speak** to an AI operator:

> *"Search for merchants named Apex in our database."*

And the AI will:
1. Call `search_merchants("Apex")` from the fraud dataset
2. Speak: *"I found 4 merchants matching Apex. They're all registered in Delaware within a 10-day window."*
3. You say: *"Now open Firefox and Google those companies."*
4. The CUA agent controls a Docker container, navigates Firefox, types the search, and reports back
5. The voice agent narrates: *"I found their Delaware incorporation records. Apex Digital Solutions LLC was filed January 20th."*

---

## Architecture

```
┌─────────────┐    audio     ┌─────────────────┐     reasoning      ┌─────────────┐
│  Microphone  │ ──────────→ │  Voice Live API  │ ────────────────→  │   GPT-5     │
│  (PCM16)     │             │  (WebSocket)     │                    │  (thinking)  │
└─────────────┘             │  • azure_semantic │    function_call   └──────┬──────┘
                             │    _vad           │ ←────────────────         │
┌─────────────┐    audio     │  • echo cancel   │                    ┌──────▼──────┐
│  Speaker     │ ←────────── │  • HD voice      │                    │  Your Code  │
│  (PCM16)     │             └─────────────────┘                    │  (tools)    │
└─────────────┘                                                      └──────┬──────┘
                                                                            │
                                                          ┌────────────────▼────────────────┐
                                                          │  query_transactions()           │
                                                          │  search_merchants()             │
                                                          │  control_computer() ──→ CUA ──→ │
                                                          │                         Docker  │
                                                          └────────────────────────────────┘
```

---

## Prerequisites

1. **Azure AI Foundry resource** with Voice Live access
2. **Docker** running with the `shadowboxer-vnc` container (for CUA modules)
3. **VNC viewer** connected to `localhost:5900` (to watch CUA actions)
4. **Microphone + speakers** (for voice modules; text fallback available)
5. Synthetic data generated: `python data/generate_synthetic_data.py`

## Setup

### 1. Install dependencies

```bash
# From the project root (PayPal/):
uv sync --extra voice
```

<details>
<summary>Alternative: pip + manual venv</summary>

```bash
cd Voice_CUA
pip install -r requirements.txt
cp .env.example .env
```
</details>

### 2. Configure environment

```bash
cp Voice_CUA/.env.example Voice_CUA/.env
# Edit Voice_CUA/.env with your Azure resource name and API key (or use managed identity)
```

### 3. Build and run the Docker container (required for CUA modules)

```bash
# Build the CUA container (one-time)
docker build -t shadowboxer-vnc ./computer-use

# Run the container (keep it running during the workshop)
docker run -d --rm --name shadowboxer-vnc \
  -p 5900:5900 \
  -e DISPLAY=:99 \
  shadowboxer-vnc

# Verify it's running
docker ps | grep shadowboxer
```

### 4. Connect a VNC viewer to watch the agent work

```bash
# macOS
open vnc://localhost:5900
# Password: secret

# Or use any VNC client (RealVNC, TigerVNC) pointed at localhost:5900
```

### 5. Launch the Streamlit demo

```bash
uv run streamlit run demo/streamlit_voice_cua.py
```

### Authentication Modes

| Mode | `.env` Setting | Voice Live Auth | CUA Auth |
|---|---|---|---|
| **API Key** (default) | `USE_MANAGED_IDENTITY=false` | `api-key` WebSocket header | `api_key` in AzureOpenAI client |
| **Managed Identity** | `USE_MANAGED_IDENTITY=true` | `Bearer` token via DefaultAzureCredential | `azure_ad_token_provider` in AzureOpenAI client |

---

## Progressive Module Walkthrough

### Module 01 — CUA Basics: Direct Computer Control
**File:** `01_cua_basics/run_cua_direct.py` | **Time:** 15-20 min

**What's running:** CUA agent + Docker only. No voice.

**What happens:**
1. You type a task: *"Open Firefox, go to google.com, search for PayPal fraud detection"*
2. GPT-5.4 receives a screenshot of the Docker desktop
3. It decides: "I need to click on the Firefox icon"
4. `xdotool` executes the click in the container
5. New screenshot → model sees Firefox open → types in address bar → ...
6. SupervisorGPT (GPT-4.1) monitors and terminates when done

**Watch:** Open a VNC viewer to `localhost:5900` to see the agent working.

**Key concept:** The CUA loop: `screenshot → model → action → screenshot → ...`

```bash
uv run python Voice_CUA/01_cua_basics/run_cua_direct.py "Open Firefox and go to microsoft.com"
```

---

### Module 02 — Voice Live: Basic Conversation
**File:** `02_voice_live_intro/voice_live_basic.py` | **Time:** 15-20 min

**What's running:** Voice Live WebSocket only. No computer control.

**What happens:**
1. WebSocket connection to `wss://<resource>.services.ai.azure.com/voice-live/realtime`
2. Session configured with:
   - `azure_semantic_vad` — knows when you're done speaking by meaning, not just silence
   - `azure_deep_noise_suppression` — filters background noise
   - `server_echo_cancellation` — doesn't hear its own voice back
   - `en-US-Ava:DragonHDLatestNeural` — HD voice synthesis
3. You talk, it responds. Basic conversation about fraud patterns.

**Key concept:** Voice Live is fully managed. No model deployment needed. The `session.update` event controls everything.

**Text fallback:** If no microphone is available, the module falls back to text input.

```bash
uv run python Voice_CUA/02_voice_live_intro/voice_live_basic.py
```

---

### Module 03 — Voice + Function Calling
**File:** `03_voice_with_functions/voice_functions.py` | **Time:** 20-25 min

**What's running:** Voice Live + function tools (no CUA yet).

**What happens:**
1. Session configured with 3 function tools: `query_fraud_data`, `search_merchants`, `get_customer_info`
2. You say: *"How many transactions are above nine thousand dollars?"*
3. Voice Live model decides to call `query_fraud_data(min_amount=9000)`
4. You receive `response.function_call_arguments.done` event
5. Your code executes the function against `transactions.csv`
6. Result sent back via `conversation.item.create(function_call_output)`
7. Model speaks: *"I found 1,247 transactions above $9,000 totaling $11.3 million."*

**Key concept:** The function call loop:
```
User speaks → Model calls function → Your code runs → Result sent back → Model speaks result
```

**Try these:**
- *"Search for merchants with Apex in the name"*
- *"How many transactions are between 9 and 10 thousand dollars?"*
- *"Look up customer CUS-A08563C93C2B"*

```bash
uv run python Voice_CUA/03_voice_with_functions/voice_functions.py
```

---

### Module 04 — The Bridge: Voice → CUA
**File:** `04_voice_cua_bridge/voice_cua_bridge.py` | **Time:** 25-30 min

**What's running:** Voice Live → function call → CUA → Docker.

**What happens:**
1. A new function tool: `control_computer(task)` 
2. You say: *"Go to Google and search for PayPal security"*
3. Voice Live calls `control_computer(task="Go to Google and search for PayPal security")`
4. Your code calls `run_cua(task)` — the existing shadowbox system
5. GPT-5.4 controls the Docker container (Firefox, typing, clicking)
6. CUA action log returned to Voice Live
7. Voice agent speaks: *"I opened Firefox, navigated to Google, and searched for PayPal security. Here are the top results..."*

**Key concept:** The bridge pattern — Voice Live's function calling invokes the CUA agent as a tool.

```bash
uv run python Voice_CUA/04_voice_cua_bridge/voice_cua_bridge.py
```

---

### Module 05 — Full System: Voice Fraud Investigation Operator
**File:** `05_full_system/voice_cua_operator.py` | **Time:** 20-25 min

**What's running:** Everything unified. Voice + data tools + computer control.

**All tools available:**
| Tool | What It Does |
|---|---|
| `control_computer` | Navigate websites via CUA → Docker |
| `query_transactions` | Query 50,000+ transaction database |
| `search_merchants` | Find merchants by name |
| `lookup_customer` | Get customer profile by ID |

**Example session:**
1. *"Search our database for merchants named Apex"* → calls `search_merchants`
2. *"Check the transactions through those merchants"* → calls `query_transactions`
3. *"Now Google 'Apex Digital Solutions Delaware' on the computer"* → calls `control_computer`
4. *"What's the risk score on customer CUS-A08563C93C2B?"* → calls `lookup_customer`

```bash
uv run python Voice_CUA/05_full_system/voice_cua_operator.py
```

---

## Voice Live API Key Concepts

| Feature | What It Does | Why It Matters |
|---|---|---|
| `azure_semantic_vad` | Detects end-of-speech by meaning, not silence | Users can pause to think without the AI cutting in |
| `remove_filler_words` | Ignores "um", "uh", "ah" during active response | Prevents false barge-in from filler words |
| `server_echo_cancellation` | Removes the AI's own voice from mic input | Speaker playback won't trigger self-conversation |
| `azure_deep_noise_suppression` | Filters background noise | Works in noisy demo environments |
| HD Voices (`DragonHDLatestNeural`) | Natural, expressive synthesis with temperature control | Sounds human, not robotic |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| "No audio device" | Install `portaudio`: `brew install portaudio` (macOS) or `apt install portaudio19-dev` (Linux) |
| CUA not connecting | Ensure Docker container is running: `docker ps \| grep shadowboxer` |
| Voice Live 401 | Check API key or run `az login` with Cognitive Services User role |
| Echo/feedback loop | Enable `server_echo_cancellation` in session config (already set) |
| VAD too aggressive | Increase `silence_duration_ms` or switch to `azure_semantic_vad` |
