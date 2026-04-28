"""
============================================================================
VOICE+CUA MODULE 02 — Voice Live API: Basic Voice Conversation
============================================================================

WORKSHOP NARRATIVE:
    Now we introduce the Azure Voice Live API — Microsoft's managed
    speech-to-speech platform. In this module, we establish a WebSocket
    connection to Voice Live and have a basic voice conversation.

    No computer control yet — just talking to an AI. This is the
    foundation we'll build on in Modules 03-05.

    Voice Live handles:
    - Speech recognition (your voice → text for the model)
    - LLM inference (GPT-5 or GPT-4.1 thinking)
    - Speech synthesis (model response → HD voice output)
    - Turn detection (knowing when you're done speaking)
    - Echo cancellation (not hearing its own voice back)

ARCHITECTURE:
    [Microphone] → PCM16 audio → Voice Live WebSocket → Model → Audio → [Speaker]

PREREQUISITES:
    - Microphone and speakers
    - Azure AI Foundry resource with Voice Live access

ESTIMATED TIME: 15-20 minutes

============================================================================
"""

import os
import sys
import json
import asyncio
import signal
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import websockets

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.audio_helpers import MicrophoneStream, AudioPlayer, AUDIO_AVAILABLE


# ============================================================================
# CONFIGURATION
# ============================================================================
RESOURCE_NAME = os.getenv("AZURE_AI_RESOURCE_NAME")
API_KEY = os.getenv("AZURE_AI_API_KEY")
MODEL = os.getenv("VOICE_LIVE_MODEL", "gpt-5")

# Parse resource name — handles full URLs like https://resource.services.ai.azure.com/...
from shared.resource_helpers import parse_resource_name
RESOURCE_NAME = parse_resource_name(RESOURCE_NAME) if RESOURCE_NAME else None

# Managed identity support — import from project root
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
try:
    from utils.auth import get_voice_live_headers, use_managed_identity
except ImportError:
    # Fallback if running standalone
    def get_voice_live_headers():
        return {"api-key": os.getenv("AZURE_AI_API_KEY", "")}
    def use_managed_identity():
        return False

if not RESOURCE_NAME:
    print("ERROR: Set AZURE_AI_RESOURCE_NAME in .env")
    sys.exit(1)

if not use_managed_identity() and not API_KEY:
    print("ERROR: Set AZURE_AI_API_KEY in .env, or set USE_MANAGED_IDENTITY=true")
    sys.exit(1)


# ============================================================================
# Voice Live WebSocket endpoint
# ============================================================================
# The endpoint format from the docs:
#   wss://<resource>.services.ai.azure.com/voice-live/realtime?api-version=2025-10-01&model=<model>
#
# WORKSHOP NOTE:
# "Voice Live is FULLY MANAGED. You don't deploy models — the API handles
# everything. You just specify which model to use and connect."
# ============================================================================

VOICE_LIVE_URI = (
    f"wss://{RESOURCE_NAME}.services.ai.azure.com"
    f"/voice-live/realtime?api-version=2025-10-01&model={MODEL}"
)


async def main():
    print("=" * 70)
    print("VOICE+CUA MODULE 02 — Voice Live: Basic Conversation")
    print("=" * 70)
    print()
    print(f"  Resource: {RESOURCE_NAME}")
    print(f"  Model: {MODEL}")
    print(f"  Endpoint: {VOICE_LIVE_URI[:80]}...")
    print()

    if not AUDIO_AVAILABLE:
        print("  ⚠ Audio I/O not available. Running in text-only simulation mode.")
        print()

    # ================================================================
    # Connect to Voice Live WebSocket
    # ================================================================
    # WORKSHOP NOTE:
    # "The api-key header authenticates the WebSocket connection.
    # In production, use Entra ID bearer tokens instead."
    # ================================================================
    print("  Connecting to Voice Live API...")

    async with websockets.connect(
        VOICE_LIVE_URI,
        additional_headers=get_voice_live_headers(),
    ) as ws:
        print("  ✓ Connected!")
        print()

        # ============================================================
        # Configure the session
        # ============================================================
        # This is the critical session.update message. Key Voice Live
        # enhancements over raw OpenAI Realtime:
        #
        #   azure_semantic_vad: Knows WHEN you're done speaking by
        #     understanding semantic meaning, not just silence duration.
        #
        #   azure_deep_noise_suppression: Filters background noise
        #     before the model processes your speech.
        #
        #   server_echo_cancellation: Prevents the model from hearing
        #     its own voice through your speakers.
        #
        #   HD voices: "en-US-Ava:DragonHDLatestNeural" — natural,
        #     expressive speech synthesis.
        # ============================================================
        session_config = {
            "type": "session.update",
            "session": {
                "instructions": (
                    "You are a friendly AI assistant for a fraud investigation "
                    "workshop. You can discuss payment fraud patterns, transaction "
                    "analysis, and data investigation techniques. Keep responses "
                    "conversational and concise — this is a voice conversation, "
                    "not a written report. Limit responses to 2-3 sentences."
                ),
                "turn_detection": {
                    "type": "azure_semantic_vad",
                    "silence_duration_ms": 500,
                    "remove_filler_words": True,
                    "languages": ["en"],
                    "interrupt_response": True,
                    "auto_truncate": True,
                },
                "input_audio_noise_reduction": {
                    "type": "azure_deep_noise_suppression",
                },
                "input_audio_echo_cancellation": {
                    "type": "server_echo_cancellation",
                },
                "voice": {
                    "name": "en-US-Ava:DragonHDLatestNeural",
                    "type": "azure-standard",
                },
            },
        }

        await ws.send(json.dumps(session_config))
        print("  ✓ Session configured")
        print(f"    VAD: azure_semantic_vad")
        print(f"    Voice: en-US-Ava:DragonHDLatestNeural (HD)")
        print(f"    Noise suppression: ON")
        print(f"    Echo cancellation: ON")
        print()

        # Wait for session.updated confirmation
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "session.updated":
                print("  ✓ Session confirmed by server")
                break
            elif msg["type"] == "error":
                print(f"  ✗ Error: {msg.get('error', {}).get('message', 'Unknown')}")
                return

        print()
        print("  ╔══════════════════════════════════════════════════════════════")
        print("  ║  VOICE CONVERSATION ACTIVE")
        print("  ║  Speak into your microphone. Press Ctrl+C to stop.")
        print("  ╚══════════════════════════════════════════════════════════════")
        print()

        # ============================================================
        # Main conversation loop
        # ============================================================
        # Two concurrent tasks:
        # 1. Capture microphone → send to Voice Live
        # 2. Receive Voice Live events → play audio / print transcript
        # ============================================================

        mic = MicrophoneStream()
        player = AudioPlayer()
        player.start()

        running = True

        def handle_sigint(sig, frame):
            nonlocal running
            running = False
            print("\n  Stopping conversation...")

        signal.signal(signal.SIGINT, handle_sigint)

        async def send_audio():
            """Capture microphone and stream to Voice Live."""
            mic.start()
            while running:
                chunk = await asyncio.get_event_loop().run_in_executor(
                    None, mic.get_chunk, 0.1,
                )
                if chunk:
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": chunk,
                    }))
                await asyncio.sleep(0.01)
            mic.stop()

        async def receive_events():
            """Process events from Voice Live."""
            while running:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break

                event = json.loads(raw)
                event_type = event.get("type", "")

                # ── Speech detection events ──
                if event_type == "input_audio_buffer.speech_started":
                    player.clear()  # Stop playback immediately on barge-in
                    print("  🎤 [interrupting...]", end="", flush=True)

                elif event_type == "input_audio_buffer.speech_stopped":
                    print(" [processing]", flush=True)

                # ── Transcript of what the user said ──
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript", "")
                    if transcript.strip():
                        print(f"  You: {transcript.strip()}")

                # ── Model's audio response ──
                elif event_type == "response.audio.delta":
                    player.play_chunk(event.get("delta", ""))

                # ── Model's text transcript ──
                elif event_type == "response.audio_transcript.delta":
                    pass  # We get the full transcript at .done

                elif event_type == "response.audio_transcript.done":
                    transcript = event.get("transcript", "")
                    if transcript.strip():
                        print(f"  AI: {transcript.strip()}")

                elif event_type == "response.done":
                    player.flush()

                elif event_type == "error":
                    err = event.get("error", {})
                    print(f"  ⚠ Error: {err.get('message', 'Unknown')}")

        # Run both tasks concurrently
        if AUDIO_AVAILABLE:
            await asyncio.gather(
                send_audio(),
                receive_events(),
            )
        else:
            # Text-only mode for environments without audio
            print("  Running in text-only mode (no microphone).")
            print("  Type messages and press Enter. Type 'quit' to stop.")
            print()
            while running:
                try:
                    text = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("  You: "),
                    )
                except EOFError:
                    break
                if text.lower() in ("quit", "exit", "q"):
                    break
                # Send text as a conversation item
                await ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }))
                await ws.send(json.dumps({"type": "response.create"}))

                # Receive the response
                while True:
                    raw = await ws.recv()
                    event = json.loads(raw)
                    if event["type"] == "response.text.done":
                        print(f"  AI: {event.get('text', '')}")
                        break
                    elif event["type"] == "response.done":
                        break
                    elif event["type"] == "error":
                        print(f"  ⚠ Error: {event.get('error', {}).get('message')}")
                        break

    print()
    print("=" * 70)
    print("MODULE 02 — Summary")
    print("=" * 70)
    print("""
  What You Built:
    ├─ WebSocket connection to Azure Voice Live API
    ├─ Session with azure_semantic_vad turn detection
    ├─ HD voice output (en-US-Ava:DragonHDLatestNeural)
    ├─ Echo cancellation + noise suppression
    └─ Bidirectional audio streaming (mic → API → speaker)

  KEY INSIGHT:
  Voice Live is fully managed — no model deployment needed.
  The session.update event controls everything: VAD, voice,
  noise suppression, echo cancellation.

  NEXT: Run 03_voice_with_functions/voice_functions.py
""")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
