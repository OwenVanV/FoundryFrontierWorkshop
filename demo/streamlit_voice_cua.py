"""
============================================================================
Streamlit Demo App — Voice + CUA Track
============================================================================
Run: uv run streamlit run demo/streamlit_voice_cua.py
============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from demo.shared import setup_page, render_module_page, PROJECT_ROOT

setup_page("PayPal Workshop — Voice + CUA Track", "🎤")

# ============================================================================
# Module definitions
# ============================================================================

MODULES = [
    {
        "id": "v01",
        "title": "01 — CUA Basics",
        "file": "Voice_CUA/01_cua_basics/run_cua_direct.py",
        "icon": "🖥️",
        "layer": "CUA Only",
        "summary": "Run the Computer Use Agent directly — type a task, watch GPT-5.4 control a Docker container via screenshots and actions.",
        "highlights": [
            "CUA loop: screenshot → GPT-5.4 → action → screenshot",
            "DockerComputer class: xdotool commands in VNC container",
            "SupervisorGPT (GPT-4.1) monitors and terminates",
            "Watch via VNC viewer on localhost:5900",
        ],
        "note": "⚠️ Requires Docker container `shadowboxer-vnc` running. Open VNC viewer to localhost:5900 to watch.",
    },
    {
        "id": "v02",
        "title": "02 — Voice Live: Basic Chat",
        "file": "Voice_CUA/02_voice_live_intro/voice_live_basic.py",
        "icon": "🎤",
        "layer": "Voice Only",
        "summary": "WebSocket connection to Azure Voice Live API. Basic voice conversation with HD voice, semantic VAD, echo cancellation.",
        "highlights": [
            "WebSocket to wss://<resource>.services.ai.azure.com/voice-live/realtime",
            "azure_semantic_vad — knows when you're done by meaning, not silence",
            "azure_deep_noise_suppression — filters background noise",
            "server_echo_cancellation — doesn't hear its own voice",
            "en-US-Ava:DragonHDLatestNeural — HD voice synthesis",
        ],
        "note": "🎧 Requires microphone and speakers. Falls back to text mode if unavailable.",
    },
    {
        "id": "v03",
        "title": "03 — Voice + Function Calling",
        "file": "Voice_CUA/03_voice_with_functions/voice_functions.py",
        "icon": "🔧",
        "layer": "Voice + Tools",
        "summary": "Voice Live with 3 function tools that query fraud datasets. Ask about transactions and the AI queries the data.",
        "highlights": [
            "3 function tools: query_fraud_data, search_merchants, get_customer_info",
            "response.function_call_arguments.done → execute → function_call_output",
            "Voice narrates the function results naturally",
            "Try: 'How many transactions are above nine thousand dollars?'",
        ],
        "note": "🎧 Voice + data tools. No computer control yet.",
    },
    {
        "id": "v04",
        "title": "04 — Voice → CUA Bridge",
        "file": "Voice_CUA/04_voice_cua_bridge/voice_cua_bridge.py",
        "icon": "🌉",
        "layer": "Voice + CUA",
        "summary": "The magic: Voice Live calls control_computer() which invokes run_cua() which controls Docker. Talk and the computer acts.",
        "highlights": [
            "control_computer(task) function tool bridges voice to CUA",
            "Voice Live → function_call → run_cua() → Docker → action log → voice narration",
            "Say: 'Go to Google and search for PayPal security'",
            "CUA actions stream back for voice narration",
        ],
        "note": "⚠️ Requires Docker + microphone. The bridge module.",
    },
    {
        "id": "v05",
        "title": "05 — Full System",
        "file": "Voice_CUA/05_full_system/voice_cua_operator.py",
        "icon": "🎯",
        "layer": "Everything",
        "summary": "Complete voice-controlled fraud investigation operator. Voice + data tools + computer control unified.",
        "highlights": [
            "4 tools: control_computer, query_transactions, search_merchants, lookup_customer",
            "Natural voice conversation with tool dispatch",
            "CUA narrates actions as they happen",
            "Bidirectional: interrupt, redirect, ask questions mid-task",
        ],
        "note": "⚠️ Requires Docker + microphone. The complete system.",
    },
]

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.title("🎤 Voice + CUA Track")
    st.caption("Talk to a computer. Watch it work.")
    st.divider()

    st.markdown("### Progressive Journey")

    layer_colors = {
        "CUA Only": "🟥",
        "Voice Only": "🟦",
        "Voice + Tools": "🟪",
        "Voice + CUA": "🟧",
        "Everything": "🟩",
    }

    for mod in MODULES:
        badge = layer_colors.get(mod["layer"], "⬜")
        if st.button(
            f"{mod['icon']} {mod['title']}",
            key=mod["id"],
            use_container_width=True,
            help=f"{badge} {mod['layer']}",
        ):
            st.session_state["selected_module"] = mod["id"]

    st.divider()
    st.markdown("""
    **Progression:**
    - 🟥 CUA only (no voice)
    - 🟦 Voice only (no CUA)
    - 🟪 Voice + data tools
    - 🟧 Voice + CUA bridge
    - 🟩 Full unified system
    """)
    st.divider()
    st.markdown("### ⚠️ How to Run")
    st.code("uv sync --extra all\nuv run streamlit run demo/streamlit_voice_cua.py", language="bash")
    st.caption("Always use `uv run streamlit` — never a global streamlit install.")

    st.divider()
    st.markdown("### 🐳 Docker (required for CUA)")
    st.code(
        "# Build (one-time)\n"
        "docker build -t shadowboxer-vnc ./computer-use\n\n"
        "# Run (keep running during workshop)\n"
        "docker run -d --rm --name shadowboxer-vnc \\\n"
        "  -p 5900:5900 -e DISPLAY=:99 shadowboxer-vnc\n\n"
        "# VNC viewer (password: secret)\n"
        "open vnc://localhost:5900",
        language="bash",
    )

# ============================================================================
# Main content
# ============================================================================

selected_id = st.session_state.get("selected_module", None)

if selected_id:
    mod = next((m for m in MODULES if m["id"] == selected_id), None)
    if mod:
        module_path = PROJECT_ROOT / mod["file"]
        run_cmd = ["uv", "run", "python", mod["file"]]

        badge = layer_colors.get(mod["layer"], "⬜")
        highlights_md = "\n".join(f"- {h}" for h in mod["highlights"])
        extra = f"""
> **Layer:** {badge} {mod['layer']}

{f"**{mod['note']}**" if mod.get('note') else ""}

### Key Highlights
{highlights_md}
"""

        render_module_page(
            module_title=f"{mod['icon']} {mod['title']}",
            module_file=module_path,
            run_cmd=run_cmd,
            extra_description=extra,
        )
else:
    # Landing page
    st.title("🎤 Cracking the Ring — Voice + CUA Track")
    st.markdown("""
    **Talk to a computer and watch it work.** Combine Azure Voice Live API with a GPT-5.4 Computer Use Agent
    to build a voice-controlled fraud investigation system.

    Each module adds one capability on top of the last:
    """)

    # Progressive journey visualization
    st.markdown("""
    ```
    Module 01: [CUA Agent] → Docker Container
                    ↓ add voice
    Module 02: [Voice Live] → Conversation
                    ↓ add tools
    Module 03: [Voice Live] → [Function Tools] → Fraud Data
                    ↓ bridge
    Module 04: [Voice Live] → [control_computer()] → [CUA] → Docker
                    ↓ unify
    Module 05: [Voice Live] → [All Tools] → [CUA + Data] → Full System
    ```
    """)

    st.info("👈 **Select a module from the sidebar to begin the progressive journey.**")

    st.divider()

    # Architecture diagram
    st.markdown("### Architecture (Module 05)")
    st.code("""
┌─────────────┐    audio     ┌─────────────────┐    reasoning    ┌───────────┐
│  Microphone  │ ──────────→ │  Voice Live API  │ ─────────────→ │   GPT-5   │
│  (PCM16)     │             │  • semantic VAD  │   function_call │ (thinking) │
└─────────────┘             │  • echo cancel   │ ←───────────── └─────┬─────┘
                             │  • HD voice      │                      │
┌─────────────┐    audio     └─────────────────┘               ┌──────▼──────┐
│  Speaker     │ ←─────────────────────────────────────────────│  Your Code  │
└─────────────┘                                                │  (tools)    │
                                                                └──────┬──────┘
                                                                       │
                                                 ┌─────────────────────▼──────────┐
                                                 │  query_transactions()          │
                                                 │  search_merchants()            │
                                                 │  control_computer() → CUA →    │
                                                 │                      Docker    │
                                                 └────────────────────────────────┘
    """, language="text")
