"""
============================================================================
VOICE+CUA MODULE 04 — The Bridge: Voice Live → CUA
============================================================================

WORKSHOP NARRATIVE:
    This is the MAGIC module. We connect Voice Live (speech) to the CUA
    system (computer control). The voice agent gets a new function tool:
    "control_computer". When you say "go to Google and search for PayPal",
    the voice agent calls control_computer(), which invokes run_cua(),
    which controls the Docker container, then reports back — and the
    voice agent SPEAKS the result.

    You are literally TALKING TO A COMPUTER and watching it work.

ARCHITECTURE:
    [Mic] → Voice Live → Model → control_computer() → run_cua() → Docker
                                                          ↓
    [Speaker] ← Voice Live ← Model ← result ← action log ←

ESTIMATED TIME: 25-30 minutes

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
from shared.cua_client import run_task

RESOURCE_NAME = os.getenv("AZURE_AI_RESOURCE_NAME")
API_KEY = os.getenv("AZURE_AI_API_KEY")
MODEL = os.getenv("VOICE_LIVE_MODEL", "gpt-5")

from shared.resource_helpers import parse_resource_name
RESOURCE_NAME = parse_resource_name(RESOURCE_NAME) if RESOURCE_NAME else None

# Managed identity support
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
try:
    from utils.auth import get_voice_live_headers, use_managed_identity
except ImportError:
    def get_voice_live_headers():
        return {"api-key": os.getenv("AZURE_AI_API_KEY", "")}
    def use_managed_identity():
        return False

if not RESOURCE_NAME:
    print("ERROR: Set AZURE_AI_RESOURCE_NAME in .env")
    sys.exit(1)

if not use_managed_identity() and not API_KEY:
    print("ERROR: Set AZURE_AI_API_KEY or USE_MANAGED_IDENTITY=true")
    sys.exit(1)

VOICE_LIVE_URI = (
    f"wss://{RESOURCE_NAME}.services.ai.azure.com"
    f"/voice-live/realtime?api-version=2025-10-01&model={MODEL}"
)


# ============================================================================
# THE BRIDGE TOOL — Voice Live calls this, which calls CUA
# ============================================================================

async def execute_computer_task(task: str) -> str:
    """
    Bridge function: takes a task string from Voice Live,
    runs it through the CUA system, returns the result.
    Runs in a separate thread to avoid blocking the asyncio event loop.
    """
    print(f"\n  🖥️  CUA Executing: {task}")
    print("  " + "─" * 60)

    actions_log = []

    def on_action(action_text):
        actions_log.append(action_text)
        print(f"  🖥️  [{len(actions_log)}] {action_text}", flush=True)

    # run_task is async but internally uses blocking OpenAI SDK calls.
    # Wrap the entire thing in a thread so it doesn't freeze the event loop.
    import concurrent.futures
    loop = asyncio.get_event_loop()

    def _run_sync():
        """Run the CUA task synchronously in a thread."""
        return asyncio.run(run_task(
            task=task,
            on_action=on_action,
            max_iterations=20,
            max_user_inputs=2,
        ))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        result = await loop.run_in_executor(pool, _run_sync)

    print("  " + "─" * 60)
    print(f"  🖥️  CUA Complete: {len(result['actions'])} actions taken")
    print()

    # Format result for the voice agent to speak
    summary = f"I completed the task on the computer. I took {len(result['actions'])} actions. "
    if result['actions']:
        summary += f"The key steps were: {'; '.join(result['actions'][-3:])}. "
    if result.get('final_message'):
        summary += f"Final result: {result['final_message'][:200]}"

    return summary


# Tool schemas for Voice Live
TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "control_computer",
        "description": (
            "Execute a task on the computer. The computer has a web browser "
            "and can navigate websites, fill forms, click buttons, and search. "
            "Use this when the user asks you to DO something on the computer, "
            "like navigate to a website, search for information, or fill out a form."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "A clear description of what to do on the computer.",
                },
            },
            "required": ["task"],
        },
    },
    {
        "type": "function",
        "name": "query_fraud_data",
        "description": "Query the fraud transaction database for statistics.",
        "parameters": {
            "type": "object",
            "properties": {
                "min_amount": {"type": "number", "description": "Minimum amount filter"},
                "max_amount": {"type": "number", "description": "Maximum amount filter"},
            },
        },
    },
]


def query_fraud_data_sync(min_amount=None, max_amount=None):
    """Quick fraud data query (non-CUA tool)."""
    import csv
    data_dir = Path(__file__).parent.parent.parent / "data"
    count = 0
    total = 0.0
    with open(data_dir / "transactions.csv", "r") as f:
        for row in csv.DictReader(f):
            amt = float(row["amount"])
            if min_amount and amt < min_amount:
                continue
            if max_amount and amt > max_amount:
                continue
            count += 1
            total += amt
    return f"Found {count} transactions totaling ${total:,.2f}."


async def main():
    print("=" * 70)
    print("VOICE+CUA MODULE 04 — The Bridge: Voice → Computer")
    print("=" * 70)
    print()
    print("  This module connects your VOICE to the COMPUTER.")
    print("  Say: 'Go to Google and search for PayPal security'")
    print("  Watch the Docker container execute the task.")
    print()

    async with websockets.connect(
        VOICE_LIVE_URI,
        additional_headers=get_voice_live_headers(),
    ) as ws:
        print("  ✓ Connected to Voice Live")

        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": (
                    "You are a voice-controlled computer operator. You can control "
                    "a computer through the control_computer tool. When the user asks "
                    "you to do something on the computer (browse, search, navigate), "
                    "use the control_computer tool.\n\n"
                    "IMPORTANT:\n"
                    "- Before calling control_computer, tell the user what you're about to do\n"
                    "- After the tool returns, summarize what happened\n"
                    "- Keep responses concise — this is voice, not text\n"
                    "- You can also query the fraud database with query_fraud_data"
                ),
                "turn_detection": {
                    "type": "azure_semantic_vad",
                    "silence_duration_ms": 600,
                    "remove_filler_words": True,
                    "interrupt_response": True,
                    "auto_truncate": True,
                },
                "input_audio_noise_reduction": {"type": "azure_deep_noise_suppression"},
                "input_audio_echo_cancellation": {"type": "server_echo_cancellation"},
                "voice": {
                    "name": "en-US-Ava:DragonHDLatestNeural",
                    "type": "azure-standard",
                },
                "tools": TOOL_SCHEMAS,
            },
        }))

        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "session.updated":
                print("  ✓ Session configured with CUA bridge")
                break

        print()
        print("  ╔══════════════════════════════════════════════════════════════")
        print("  ║  VOICE → COMPUTER BRIDGE ACTIVE")
        print("  ║  Say: 'Go to Google and search for PayPal fraud detection'")
        print("  ║  Say: 'How many transactions are above nine thousand dollars?'")
        print("  ║  Press Ctrl+C to stop.")
        print("  ╚══════════════════════════════════════════════════════════════")
        print()

        mic = MicrophoneStream()
        player = AudioPlayer()
        player.start()
        running = True

        def handle_stop(sig, frame):
            nonlocal running
            running = False
            mic.stop()
            player.stop()
            print("\n  Stopping...")
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_stop)
        signal.signal(signal.SIGTERM, handle_stop)

        # Track whether a response is currently being generated
        response_active = False
        pending_items = []  # Queue of (item_json, trigger_response) to send when response clears

        async def flush_pending():
            """Send any queued items now that the response slot is free."""
            nonlocal response_active
            while pending_items:
                item_json, trigger = pending_items.pop(0)
                await ws.send(json.dumps(item_json))
                if trigger:
                    response_active = True
                    await ws.send(json.dumps({"type": "response.create"}))

        async def safe_send_and_respond(item_json):
            """Send a conversation item + response.create, queuing if a response is active."""
            nonlocal response_active
            if response_active:
                pending_items.append((item_json, True))
            else:
                await ws.send(json.dumps(item_json))
                response_active = True
                await ws.send(json.dumps({"type": "response.create"}))

        async def send_audio():
            mic.start()
            while running:
                chunk = await asyncio.get_event_loop().run_in_executor(None, mic.get_chunk, 0.1)
                if chunk:
                    await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
                await asyncio.sleep(0.01)
            mic.stop()

        async def receive_events():
            nonlocal response_active
            while running:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    continue

                event = json.loads(raw)
                t = event.get("type", "")

                if t == "input_audio_buffer.speech_started":
                    player.clear()
                    response_active = False  # Interruption cancels active response
                    print("  🎤 [interrupting...]", end="", flush=True)
                elif t == "input_audio_buffer.speech_stopped":
                    print(" [processing]")
                elif t == "conversation.item.input_audio_transcription.completed":
                    text = event.get("transcript", "").strip()
                    if text:
                        print(f"  You: {text}")
                elif t == "response.audio.delta":
                    player.play_chunk(event.get("delta", ""))
                elif t == "response.audio_transcript.done":
                    text = event.get("transcript", "").strip()
                    if text:
                        print(f"  AI: {text}")
                elif t == "response.done":
                    player.flush()
                    response_active = False
                    await flush_pending()

                # ════════════════════════════════════════════════════
                # THE BRIDGE — Voice Live calls our tools here
                # CUA runs NON-BLOCKING in the background so the
                # voice conversation can continue while it works.
                # ════════════════════════════════════════════════════
                elif t == "response.function_call_arguments.done":
                    func_name = event.get("name", "")
                    call_id = event.get("call_id", "")
                    args = json.loads(event.get("arguments", "{}"))

                    # Function calls arrive DURING a response — the response
                    # is now done from Voice Live's perspective
                    response_active = False

                    print(f"\n  🔧 Voice → Tool: {func_name}")

                    if func_name == "control_computer":
                        # NON-BLOCKING: Tell voice agent it's started, run CUA in background
                        await ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": "I've started the computer task. It's running in the background — you can keep talking to me while it works. I'll let you know when it's done.",
                            },
                        }))
                        response_active = True
                        await ws.send(json.dumps({"type": "response.create"}))

                        # Launch CUA in background thread
                        async def _run_cua_background(task_str):
                            print(f"  🖥️  CUA background task started: {task_str[:60]}...", flush=True)
                            try:
                                result = await execute_computer_task(task_str)
                                print(f"  🖥️  CUA background task completed!", flush=True)
                                await safe_send_and_respond({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "message",
                                        "role": "user",
                                        "content": [{"type": "input_text", "text": f"[SYSTEM: The computer task completed. Results: {result}. Tell the user what happened.]"}],
                                    },
                                })
                            except Exception as e:
                                print(f"  🖥️  CUA background task failed: {e}", flush=True)
                                await safe_send_and_respond({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "message",
                                        "role": "user",
                                        "content": [{"type": "input_text", "text": f"[SYSTEM: Computer task failed: {str(e)[:200]}. Let the user know.]"}],
                                    },
                                })

                        asyncio.create_task(_run_cua_background(args.get("task", "")))

                    elif func_name == "query_fraud_data":
                        result = query_fraud_data_sync(**args)
                        await ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {"type": "function_call_output", "call_id": call_id, "output": result},
                        }))
                        response_active = True
                        await ws.send(json.dumps({"type": "response.create"}))
                    else:
                        await ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {"type": "function_call_output", "call_id": call_id, "output": f"Unknown tool: {func_name}"},
                        }))
                        response_active = True
                        await ws.send(json.dumps({"type": "response.create"}))

                elif t == "error":
                    print(f"  ⚠ {event.get('error', {}).get('message', 'Unknown error')}")

        if AUDIO_AVAILABLE:
            await asyncio.gather(send_audio(), receive_events())
        else:
            print("  Text mode. Type commands:")
            while True:
                try:
                    text = input("  You: ")
                except (EOFError, KeyboardInterrupt):
                    break
                if text.lower() in ("quit", "exit"):
                    break
                await ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
                }))
                await ws.send(json.dumps({"type": "response.create"}))
                while True:
                    raw = await ws.recv()
                    ev = json.loads(raw)
                    if ev["type"] == "response.function_call_arguments.done":
                        fn = ev["name"]
                        args = json.loads(ev.get("arguments", "{}"))
                        if fn == "control_computer":
                            res = await execute_computer_task(args.get("task", ""))
                        elif fn == "query_fraud_data":
                            res = query_fraud_data_sync(**args)
                        else:
                            res = "Unknown"
                        await ws.send(json.dumps({"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": ev.get("call_id", ""), "output": res}}))
                        await ws.send(json.dumps({"type": "response.create"}))
                    elif ev["type"] in ("response.text.done", "response.audio_transcript.done"):
                        print(f"  AI: {ev.get('text', ev.get('transcript', ''))}")
                    elif ev["type"] == "response.done":
                        break

    print()
    print("=" * 70)
    print("MODULE 04 — Summary")
    print("=" * 70)
    print("""
  What You Built:
    ├─ Voice Live → control_computer() → run_cua() → Docker
    ├─ Bidirectional: voice commands IN, voice narration OUT
    ├─ CUA action log reported back through the voice channel
    └─ Combined voice + computer control + data query tools

  THE ARCHITECTURE:
    [Your Voice] → Voice Live API → GPT-5 → function_call
         ↓                                       ↓
    [Your Ears] ← Voice Live API ← GPT-5 ← CUA result ← Docker

  NEXT: Run 05_full_system/voice_cua_operator.py
""")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
