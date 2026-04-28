"""
============================================================================
VOICE+CUA MODULE 05 — Full System: Voice Fraud Investigation Operator
============================================================================

WORKSHOP NARRATIVE:
    The complete system. All pieces come together:

    You SPEAK to an AI fraud investigation operator. It can:
    1. Query fraud datasets (function tools)
    2. Control a computer (CUA → Docker)
    3. Narrate what it's doing in real-time
    4. Accept interruptions and course corrections

    This is the "talk to a computer and have it do work" vision.

    Example conversation:
      You: "Search for merchants named Apex on Google"
      AI:  "I'll open the browser and search for Apex merchants..."
           [CUA navigates Firefox, goes to Google, types search]
      AI:  "I found several results. The top result mentions Apex Digital
            Solutions LLC registered in Delaware."
      You: "Now check our transaction database for that merchant"
      AI:  [calls query_fraud_data] "We have 47 transactions through
            Apex-linked merchants, totaling $438,000. The amounts cluster
            between $9,200 and $9,800 — classic structuring pattern."

ESTIMATED TIME: 20-25 minutes (demo/exploration)

============================================================================
"""

import os
import sys
import json
import csv
import asyncio
import signal
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

import websockets

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.audio_helpers import MicrophoneStream, AudioPlayer, AUDIO_AVAILABLE
from shared.cua_client import run_task

RESOURCE_NAME = os.getenv("AZURE_AI_RESOURCE_NAME")
API_KEY = os.getenv("AZURE_AI_API_KEY")
MODEL = os.getenv("VOICE_LIVE_MODEL", "gpt-5")
DATA_DIR = Path(__file__).parent.parent.parent / "data"

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
# COMPLETE TOOL SUITE — All capabilities unified
# ============================================================================

async def execute_computer_task(task: str) -> str:
    """CUA bridge: voice → computer control. Runs in a thread to avoid blocking the event loop."""
    print(f"\n  🖥️  CUA: {task}", flush=True)

    import concurrent.futures
    loop = asyncio.get_event_loop()

    def _run_sync():
        return asyncio.run(run_task(
            task=task, max_iterations=25, max_user_inputs=2,
            on_action=lambda a: print(f"  🖥️  → {a}", flush=True),
        ))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        result = await loop.run_in_executor(pool, _run_sync)

    summary = f"Completed in {len(result['actions'])} steps. "
    if result.get('final_message'):
        summary += result['final_message'][:300]
    return summary


def query_transactions_voice(min_amount=None, max_amount=None, sender_id=None, merchant_id=None):
    """Query transactions with filters."""
    count, total = 0, 0.0
    senders, merchants = Counter(), Counter()
    with open(DATA_DIR / "transactions.csv", "r") as f:
        for row in csv.DictReader(f):
            amt = float(row["amount"])
            if min_amount and amt < min_amount: continue
            if max_amount and amt > max_amount: continue
            if sender_id and row["sender_id"] != sender_id: continue
            if merchant_id and row["merchant_id"] != merchant_id: continue
            count += 1
            total += amt
            senders[row["sender_id"]] += 1
            merchants[row["merchant_id"]] += 1
    if count == 0:
        return "No transactions found matching those criteria."
    top_senders = ", ".join(f"{s}({c})" for s, c in senders.most_common(3))
    top_merchants = ", ".join(f"{m}({c})" for m, c in merchants.most_common(3))
    return (f"Found {count} transactions totaling ${total:,.2f}. "
            f"Top senders: {top_senders}. Top merchants: {top_merchants}.")


def search_merchants_voice(name_fragment):
    """Search merchants by name."""
    matches = []
    with open(DATA_DIR / "merchants.csv", "r") as f:
        for row in csv.DictReader(f):
            if name_fragment.lower() in row["business_name"].lower():
                matches.append(row)
    if not matches:
        return f"No merchants found matching '{name_fragment}'."
    names = [f"{m['business_name']} ({m['merchant_id']}, registered {m['registration_date']}, {m['city']}, {m['state']})" for m in matches]
    return f"Found {len(matches)} merchants: {'; '.join(names)}."


def lookup_customer_voice(customer_id):
    """Look up a customer."""
    with open(DATA_DIR / "customers.csv", "r") as f:
        for row in csv.DictReader(f):
            if row["customer_id"] == customer_id:
                return (f"{row['first_name']} {row['last_name']}, email {row['email']}, "
                        f"registered {row['registration_date']}, {row['city']}, {row['state']}, "
                        f"risk score {row['risk_score']}, verified: {row['verified_identity']}.")
    return f"Customer {customer_id} not found."


TOOL_FUNCTIONS = {
    "control_computer": execute_computer_task,
    "query_transactions": query_transactions_voice,
    "search_merchants": search_merchants_voice,
    "lookup_customer": lookup_customer_voice,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "control_computer",
        "description": "Control the computer to perform tasks like browsing, searching, or navigating websites. Use when the user asks to DO something on screen.",
        "parameters": {"type": "object", "properties": {"task": {"type": "string", "description": "What to do on the computer"}}, "required": ["task"]},
    },
    {
        "type": "function",
        "name": "query_transactions",
        "description": "Query the payment transaction database. Use when the user asks about transaction counts, amounts, or patterns.",
        "parameters": {"type": "object", "properties": {
            "min_amount": {"type": "number", "description": "Minimum amount"},
            "max_amount": {"type": "number", "description": "Maximum amount"},
            "sender_id": {"type": "string", "description": "Filter by sender"},
            "merchant_id": {"type": "string", "description": "Filter by merchant"},
        }},
    },
    {
        "type": "function",
        "name": "search_merchants",
        "description": "Search for merchants by name in the payment platform database.",
        "parameters": {"type": "object", "properties": {"name_fragment": {"type": "string", "description": "Name to search"}}, "required": ["name_fragment"]},
    },
    {
        "type": "function",
        "name": "lookup_customer",
        "description": "Look up a customer's profile by ID.",
        "parameters": {"type": "object", "properties": {"customer_id": {"type": "string", "description": "Customer ID like CUS-XXXX"}}, "required": ["customer_id"]},
    },
]


async def main():
    print("=" * 70)
    print("VOICE+CUA MODULE 05 — Full System: Voice Fraud Operator")
    print("=" * 70)
    print()
    print("  The complete system: Voice + Data Tools + Computer Control")
    print()
    print("  Try these commands:")
    print("    'Search for merchants named Apex in our database'")
    print("    'How many transactions are between 9 and 10 thousand dollars?'")
    print("    'Open Firefox and go to azure.microsoft.com'")
    print("    'Look up customer CUS-A08563C93C2B'")
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
                    "You are the Fraud Investigation Operator — a voice-controlled AI that "
                    "can both query fraud databases and control a computer. You help investigators "
                    "by combining data analysis with hands-on computer work.\n\n"
                    "CAPABILITIES:\n"
                    "- query_transactions: Search the 50,000+ transaction database\n"
                    "- search_merchants: Find merchants by name\n"
                    "- lookup_customer: Get customer profiles\n"
                    "- control_computer: Navigate websites, search, fill forms\n\n"
                    "BEHAVIOR:\n"
                    "- Before taking action, briefly state what you'll do\n"
                    "- After tool results, summarize findings conversationally\n"
                    "- Connect dots across data sources proactively\n"
                    "- Keep responses concise — you're a voice interface\n"
                    "- When computer tasks complete, narrate what happened"
                ),
                "turn_detection": {
                    "type": "azure_semantic_vad",
                    "silence_duration_ms": 500,
                    "remove_filler_words": True,
                    "languages": ["en"],
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
                break

        print("  ✓ Full system configured")
        print()
        print("  ╔══════════════════════════════════════════════════════════════")
        print("  ║  🎤 FRAUD INVESTIGATION OPERATOR — ACTIVE")
        print("  ║  Voice + Data Tools + Computer Control")
        print("  ║  Speak naturally. Press Ctrl+C to stop.")
        print("  ╚══════════════════════════════════════════════════════════════")
        print()

        mic = MicrophoneStream()
        player = AudioPlayer()
        player.start()
        running = True

        signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

        # Track active response to avoid "already has an active response" errors
        response_active = False
        pending_items = []

        async def flush_pending():
            nonlocal response_active
            while pending_items:
                item_json, trigger = pending_items.pop(0)
                await ws.send(json.dumps(item_json))
                if trigger:
                    response_active = True
                    await ws.send(json.dumps({"type": "response.create"}))

        async def safe_send_and_respond(item_json):
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
                    response_active = False
                    print("  🎤 [interrupting...]", end="", flush=True)
                elif t == "input_audio_buffer.speech_stopped":
                    print(" ✓")
                elif t == "conversation.item.input_audio_transcription.completed":
                    text = event.get("transcript", "").strip()
                    if text:
                        print(f"  You: {text}")
                elif t == "response.audio.delta":
                    player.play_chunk(event.get("delta", ""))
                elif t == "response.audio_transcript.done":
                    text = event.get("transcript", "").strip()
                    if text:
                        print(f"  Operator: {text}")
                elif t == "response.done":
                    player.flush()
                    response_active = False
                    await flush_pending()
                elif t == "response.function_call_arguments.done":
                    func_name = event.get("name", "")
                    call_id = event.get("call_id", "")
                    args = json.loads(event.get("arguments", "{}"))
                    response_active = False

                    print(f"\n  🔧 {func_name}({json.dumps(args)[:60]}...)")

                    if func_name == "control_computer":
                        # NON-BLOCKING: CUA runs in background, voice continues
                        await ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": "I've started the computer task in the background. You can keep talking to me — I'll let you know when it's done.",
                            },
                        }))
                        response_active = True
                        await ws.send(json.dumps({"type": "response.create"}))

                        async def _run_cua_bg(task_str):
                            print(f"  🖥️  CUA background: {task_str[:60]}...", flush=True)
                            try:
                                cua_result = await execute_computer_task(task_str)
                                print(f"  🖥️  CUA background: done!", flush=True)
                                await safe_send_and_respond({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "message",
                                        "role": "user",
                                        "content": [{"type": "input_text", "text": f"[SYSTEM: Computer task completed. Results: {cua_result}. Tell the user what happened.]"}],
                                    },
                                })
                            except Exception as e:
                                print(f"  🖥️  CUA background failed: {e}", flush=True)

                        asyncio.create_task(_run_cua_bg(args.get("task", "")))

                    else:
                        # All other tools run synchronously (they're fast)
                        func = TOOL_FUNCTIONS.get(func_name)
                        if func:
                            if asyncio.iscoroutinefunction(func):
                                result = await func(**args)
                            else:
                                result = func(**args)
                        else:
                            result = f"Unknown function: {func_name}"

                        print(f"  🔧 → {str(result)[:100]}...")

                        await ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {"type": "function_call_output", "call_id": call_id, "output": str(result)},
                        }))
                        response_active = True
                        await ws.send(json.dumps({"type": "response.create"}))

                elif t == "error":
                    print(f"  ⚠ {event.get('error', {}).get('message', '')}")

        if AUDIO_AVAILABLE:
            await asyncio.gather(send_audio(), receive_events())
        else:
            print("  Text mode:")
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
                        func = TOOL_FUNCTIONS.get(fn)
                        if func:
                            res = await func(**args) if asyncio.iscoroutinefunction(func) else func(**args)
                        else:
                            res = "Unknown"
                        print(f"  🔧 {fn} → {str(res)[:80]}...")
                        await ws.send(json.dumps({"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": ev.get("call_id", ""), "output": str(res)}}))
                        await ws.send(json.dumps({"type": "response.create"}))
                    elif ev["type"] in ("response.text.done", "response.audio_transcript.done"):
                        print(f"  Operator: {ev.get('text', ev.get('transcript', ''))}")
                    elif ev["type"] == "response.done":
                        break

    print()
    print("=" * 70)
    print("VOICE+CUA MODULE 05 — Workshop Complete!")
    print("=" * 70)
    print("""
  ╔══════════════════════════════════════════════════════════════════════
  ║  WHAT YOU BUILT: A complete voice-controlled fraud investigation
  ║  operator that can:
  ║
  ║    🎤  Understand natural voice commands
  ║    📊  Query a 50,000-transaction fraud database
  ║    🏢  Look up merchant and customer profiles
  ║    🖥️   Control a computer (browse, search, navigate)
  ║    🔊  Narrate findings and actions in real-time
  ║
  ║  ARCHITECTURE:
  ║    Voice Live (azure_semantic_vad, HD voice, echo cancellation)
  ║      → GPT-5 (reasoning, function calling)
  ║        → Custom Tools (fraud data queries)
  ║        → CUA Bridge (GPT-5.4 → Docker container)
  ║      → Voice Live (speech synthesis)
  ║    → Your ears
  ║
  ║  THREE AZURE SERVICES UNIFIED:
  ║    1. Voice Live API — speech-to-speech
  ║    2. Azure OpenAI — reasoning (GPT-5) + vision (GPT-5.4)
  ║    3. Foundry — managed infrastructure
  ╚══════════════════════════════════════════════════════════════════════
""")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
