"""
============================================================================
VOICE+CUA MODULE 01 — CUA Basics: Direct Computer Control
============================================================================

WORKSHOP NARRATIVE:
    Before adding voice, understand the CUA (Computer Use Agent) system
    on its own. This module runs CUA directly — you type a task, and
    GPT-5.4 controls a Docker container via screenshots and actions.

    The CUA loop:
    1. Take a screenshot of the Docker container's display
    2. Send the screenshot to GPT-5.4
    3. Model returns an action (click, type, scroll, keypress)
    4. Execute the action on the container
    5. Repeat until task is complete

    A "SupervisorGPT" (GPT-4.1) monitors the CUA agent and decides
    when the task is complete.

PREREQUISITES:
    - Docker running with the shadowboxer-vnc container
    - VNC viewer connected to localhost:5900 (to watch the agent work)
    - Azure OpenAI access with GPT-5.4 and GPT-4.1 deployments

ESTIMATED TIME: 15-20 minutes

============================================================================
"""

import os
import sys
import asyncio
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.cua_client import run_task


async def main():
    print("=" * 70)
    print("VOICE+CUA MODULE 01 — CUA Basics: Direct Computer Control")
    print("=" * 70)
    print()
    print("  This module runs the CUA agent directly — no voice.")
    print("  Open a VNC viewer to localhost:5900 to watch the agent work.")
    print()

    # ================================================================
    # Demo 1: Simple navigation task
    # ================================================================
    print("─" * 70)
    print("DEMO 1: Simple Web Navigation")
    print("─" * 70)
    print()

    task = "Open Firefox, go to google.com, and search for 'PayPal fraud detection'"

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])

    print(f"  Task: {task}")
    print()
    print("  Running CUA agent... (watch the VNC viewer)")
    print()

    action_count = 0

    def on_action(action_text):
        nonlocal action_count
        action_count += 1
        print(f"  [{action_count}] {action_text}")

    result = await run_task(
        task=task,
        on_action=on_action,
        max_iterations=20,
    )

    print()
    print(f"  Completed: {result['completed']}")
    print(f"  Actions taken: {len(result['actions'])}")
    if result['final_message']:
        print(f"  Final message: {result['final_message'][:200]}")

    # ================================================================
    # Summary
    # ================================================================
    print()
    print("=" * 70)
    print("MODULE 01 — Summary")
    print("=" * 70)
    print("""
  What Happened:
    ├─ GPT-5.4 received screenshots from the Docker container
    ├─ It decided what to click, type, or scroll
    ├─ Actions were executed via xdotool in the container
    └─ SupervisorGPT monitored progress and terminated when done

  The CUA Loop:
    screenshot → model → action → screenshot → model → action → ...

  KEY INSIGHT:
  This is the raw CUA system — powerful but silent. In Module 02,
  we add a voice interface (Voice Live) so you can TALK to an AI.
  In Module 04, we connect voice to CUA — talk and the computer acts.

  NEXT: Run 02_voice_live_intro/voice_live_basic.py
""")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
