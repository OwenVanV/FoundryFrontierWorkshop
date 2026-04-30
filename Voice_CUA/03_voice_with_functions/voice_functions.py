"""
============================================================================
VOICE+CUA MODULE 03 — Voice Live with Function Calling
============================================================================

WORKSHOP NARRATIVE:
    Module 02 was voice-only chat. Now we add TOOLS — the Voice Live
    agent can call functions to query our fraud datasets. Same Realtime
    function calling pattern as OpenAI, but with Voice Live's enhanced
    VAD, echo cancellation, and HD voices.

    Ask the voice agent: "How many transactions are above $9,000?"
    and it will call query_fraud_data(), get the answer, and speak it.

ARCHITECTURE:
    [Mic] → Voice Live → Model thinks → function_call → YOUR CODE → result → Model → [Speaker]

LEARNING OBJECTIVES:
    1. Define function tools for Voice Live sessions
    2. Handle response.function_call_arguments.done events
    3. Send function results back via conversation.item.create
    4. Observe the voice agent using tools mid-conversation

ESTIMATED TIME: 20-25 minutes

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
# FUNCTION TOOLS — These are called by Voice Live via function calling
# ============================================================================
# Each function is defined as a tool schema for the session, and
# implemented as a Python function that gets called when Voice Live
# triggers it.
#
# WORKSHOP NOTE:
# "These are the SAME functions from agents/utils/fraud_tools.py but
# simplified for voice — shorter outputs that sound good when spoken."
# ============================================================================

def query_fraud_data(min_amount: float = None, max_amount: float = None, sender_id: str = None) -> str:
    """Query the transaction data with filters."""
    txns = []
    with open(DATA_DIR / "transactions.csv", "r") as f:
        for row in csv.DictReader(f):
            amount = float(row["amount"])
            if min_amount and amount < min_amount:
                continue
            if max_amount and amount > max_amount:
                continue
            if sender_id and row["sender_id"] != sender_id:
                continue
            txns.append(row)

    if not txns:
        return "No transactions found matching those criteria."

    total_amount = sum(float(t["amount"]) for t in txns)
    unique_senders = len(set(t["sender_id"] for t in txns))
    unique_merchants = len(set(t["merchant_id"] for t in txns))

    return (
        f"Found {len(txns)} transactions totaling ${total_amount:,.2f}. "
        f"Involving {unique_senders} unique senders and {unique_merchants} merchants."
    )


def search_merchants(name_fragment: str) -> str:
    """Search for merchants by name."""
    merchants = []
    with open(DATA_DIR / "merchants.csv", "r") as f:
        for row in csv.DictReader(f):
            if name_fragment.lower() in row["business_name"].lower():
                merchants.append(row)

    if not merchants:
        return f"No merchants found matching '{name_fragment}'."

    names = [m["business_name"] for m in merchants]
    states = [m["state"] for m in merchants]
    dates = [m["registration_date"] for m in merchants]

    return (
        f"Found {len(merchants)} merchants matching '{name_fragment}': "
        f"{', '.join(names)}. "
        f"Registered in {', '.join(set(states))} between {min(dates)} and {max(dates)}."
    )


def get_customer_info(customer_id: str) -> str:
    """Look up a customer profile."""
    with open(DATA_DIR / "customers.csv", "r") as f:
        for row in csv.DictReader(f):
            if row["customer_id"] == customer_id:
                return (
                    f"Customer {customer_id}: {row['first_name']} {row['last_name']}, "
                    f"email {row['email']}, registered {row['registration_date']}, "
                    f"located in {row['city']}, {row['state']}, "
                    f"risk score {row['risk_score']}."
                )
    return f"Customer {customer_id} not found."


# Map function names to implementations
TOOL_FUNCTIONS = {
    "query_fraud_data": query_fraud_data,
    "search_merchants": search_merchants,
    "get_customer_info": get_customer_info,
}

# Tool schemas for Voice Live session configuration
TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "query_fraud_data",
        "description": "Query the transaction database with optional filters. Use this when the user asks about transactions, amounts, or patterns.",
        "parameters": {
            "type": "object",
            "properties": {
                "min_amount": {"type": "number", "description": "Minimum transaction amount in USD"},
                "max_amount": {"type": "number", "description": "Maximum transaction amount in USD"},
                "sender_id": {"type": "string", "description": "Filter by sender customer ID"},
            },
        },
    },
    {
        "type": "function",
        "name": "search_merchants",
        "description": "Search for merchants by name. Use when the user asks about specific merchants or companies.",
        "parameters": {
            "type": "object",
            "properties": {
                "name_fragment": {"type": "string", "description": "Part of the merchant name to search for"},
            },
            "required": ["name_fragment"],
        },
    },
    {
        "type": "function",
        "name": "get_customer_info",
        "description": "Look up a customer's profile by their ID. Use when the user asks about a specific customer.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "The customer ID (e.g. CUS-XXXX)"},
            },
            "required": ["customer_id"],
        },
    },
]


async def main():
    print("=" * 70)
    print("VOICE+CUA MODULE 03 — Voice Live with Function Calling")
    print("=" * 70)
    print()
    print(f"  Model: {MODEL}")
    print(f"  Tools: {len(TOOL_SCHEMAS)} functions")
    for t in TOOL_SCHEMAS:
        print(f"    • {t['name']}: {t['description'][:60]}...")
    print()

    async with websockets.connect(
        VOICE_LIVE_URI,
        additional_headers=get_voice_live_headers(),
    ) as ws:
        print("  ✓ Connected to Voice Live")

        # ============================================================
        # Session config WITH tools
        # ============================================================
        # The tools array uses the same format as OpenAI Realtime API
        # function tools. Voice Live processes them identically.
        # ============================================================
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": (
                    "You are a fraud investigation voice assistant. You have tools "
                    "to query a payment transaction database, search for merchants, "
                    "and look up customer profiles. Use these tools to answer the "
                    "user's questions about the data. Keep responses brief and "
                    "conversational — this is a voice interface."
                ),
                "turn_detection": {
                    "type": "azure_semantic_vad",
                    "silence_duration_ms": 500,
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

        # Wait for confirmation
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "session.updated":
                print("  ✓ Session configured with function tools")
                break

        print()
        print("  ╔══════════════════════════════════════════════════════════════")
        print("  ║  VOICE + TOOLS ACTIVE")
        print("  ║  Try: 'How many transactions are above $9,000?'")
        print("  ║  Try: 'Search for merchants with Apex in the name'")
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

        async def send_audio():
            mic.start()
            while running:
                chunk = await asyncio.get_event_loop().run_in_executor(None, mic.get_chunk, 0.1)
                if chunk:
                    await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
                await asyncio.sleep(0.01)
            mic.stop()

        async def receive_events():
            while running:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break

                event = json.loads(raw)
                event_type = event.get("type", "")

                if event_type == "input_audio_buffer.speech_started":
                    player.clear()
                    print("  🎤 [interrupting...]", end="", flush=True)

                elif event_type == "input_audio_buffer.speech_stopped":
                    print(" [processing]")

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    t = event.get("transcript", "").strip()
                    if t:
                        print(f"  You: {t}")

                elif event_type == "response.audio.delta":
                    player.play_chunk(event.get("delta", ""))

                elif event_type == "response.audio_transcript.done":
                    t = event.get("transcript", "").strip()
                    if t:
                        print(f"  AI: {t}")

                elif event_type == "response.done":
                    player.flush()

                # ════════════════════════════════════════════════════════
                # FUNCTION CALL HANDLING — This is the key addition
                # ════════════════════════════════════════════════════════
                # When the model decides to call a function, we get:
                #   response.function_call_arguments.done
                # We execute the function, then send the result back as:
                #   conversation.item.create (function_call_output)
                # Then trigger a new response so the model speaks the result.
                # ════════════════════════════════════════════════════════
                elif event_type == "response.function_call_arguments.done":
                    func_name = event.get("name", "")
                    call_id = event.get("call_id", "")
                    args_str = event.get("arguments", "{}")

                    print(f"  🔧 Tool call: {func_name}({args_str[:60]}...)")

                    # Execute the function
                    try:
                        args = json.loads(args_str)
                        func = TOOL_FUNCTIONS.get(func_name)
                        if func:
                            result = func(**args)
                        else:
                            result = f"Unknown function: {func_name}"
                    except Exception as e:
                        result = f"Error: {str(e)}"

                    print(f"  🔧 Result: {result[:80]}...")

                    # Send the result back to Voice Live
                    await ws.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": result,
                        },
                    }))

                    # Trigger a new response so the model speaks the result
                    await ws.send(json.dumps({"type": "response.create"}))

                elif event_type == "error":
                    print(f"  ⚠ Error: {event.get('error', {}).get('message', 'Unknown')}")

        if AUDIO_AVAILABLE:
            await asyncio.gather(send_audio(), receive_events())
        else:
            print("  Text-only mode. Type messages below:")
            while running:
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
                        func_name = ev.get("name", "")
                        args = json.loads(ev.get("arguments", "{}"))
                        func = TOOL_FUNCTIONS.get(func_name)
                        result = func(**args) if func else "Unknown function"
                        print(f"  🔧 {func_name} → {result[:80]}...")
                        await ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {"type": "function_call_output", "call_id": ev.get("call_id", ""), "output": result},
                        }))
                        await ws.send(json.dumps({"type": "response.create"}))
                    elif ev["type"] == "response.text.done":
                        print(f"  AI: {ev.get('text', '')}")
                        break
                    elif ev["type"] == "response.audio_transcript.done":
                        print(f"  AI: {ev.get('transcript', '')}")
                    elif ev["type"] == "response.done":
                        break

    print()
    print("=" * 70)
    print("MODULE 03 — Summary")
    print("=" * 70)
    print("""
  What You Built:
    ├─ Voice Live session with function tools
    ├─ 3 fraud data query functions callable by voice
    ├─ Real-time function call → execute → response cycle
    └─ Voice narration of function results

  The Function Call Flow:
    User speaks → Model decides to call function →
    response.function_call_arguments.done event →
    Your code executes the function →
    conversation.item.create (function_call_output) →
    response.create → Model speaks the result

  NEXT: Run 04_voice_cua_bridge/voice_cua_bridge.py
""")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
