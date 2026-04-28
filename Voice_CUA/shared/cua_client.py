"""
============================================================================
Voice + CUA — Shared CUA Client Wrapper
============================================================================

PURPOSE:
    Wraps the existing shadowbox CUA system (computer-use/shadowbox/) into
    a clean async interface that Voice Live function calls can invoke.

    The existing system uses:
    - docker_c.DockerComputer for controlling a VNC Docker container
    - cua.Agent for the GPT-5.4 screenshot→action loop
    - cua.Scaler for resizing screenshots

    This wrapper adds:
    - Clean async def run_task(task) interface
    - Action log capture for Voice Live narration
    - Timeout and error handling

============================================================================
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import openai

# Managed identity support
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
try:
    from utils.auth import use_managed_identity, get_token_provider
except ImportError:
    def use_managed_identity():
        return False
    def get_token_provider():
        return None

# Add the computer-use/shadowbox directory to the path — this is the enhanced
# version of the CUA modules with last_screenshot tracking and run_logger.
# The root computer-use/cua.py is a simpler version without these features.
COMPUTER_USE_DIR = Path(__file__).parent.parent.parent / "computer-use"
SHADOWBOX_DIR = COMPUTER_USE_DIR / "shadowbox"

try:
    sys.path.insert(0, str(SHADOWBOX_DIR))
    sys.path.insert(0, str(COMPUTER_USE_DIR))
    # Import from shadowbox/ first — it has last_screenshot and run_logger
    from shadowbox import cua
    from shadowbox import docker_c
except ImportError:
    try:
        # Fallback: try root-level modules
        sys.path.insert(0, str(COMPUTER_USE_DIR))
        import cua
        import docker_c
    except ImportError:
        print("  ⚠ Could not import CUA modules from computer-use/shadowbox/")
        print("    Make sure the computer-use/shadowbox directory exists.")
        cua = None
        docker_c = None


async def run_task(
    task: str,
    model: str = None,
    endpoint: str = "azure",
    container_name: str = None,
    max_iterations: int = 30,
    max_user_inputs: int = 3,
    on_action: callable = None,
) -> dict:
    """
    Run a CUA task and return the results.

    Args:
        task: The task description for the CUA agent.
        model: Model name (defaults to CUA_MODEL env or gpt-5.4).
        endpoint: "azure" or "openai".
        container_name: Docker container name.
        max_iterations: Max CUA loop iterations.
        max_user_inputs: Max supervisor responses before auto-stop.
        on_action: Optional callback(action_text) called for each CUA action.

    Returns:
        dict with keys: actions (list of action descriptions),
                        completed (bool), final_message (str or None)
    """
    if cua is None or docker_c is None:
        return {
            "actions": [],
            "completed": False,
            "final_message": "CUA modules not available. Check computer-use/ directory.",
        }

    model = model or os.getenv("CUA_MODEL", "gpt-5.4")
    container_name = container_name or os.getenv("DOCKER_CONTAINER_NAME", "shadowboxer-vnc")
    supervisor_model = os.getenv("SUPERVISOR_MODEL", "gpt-4.1")

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    print(f"  ⚙ CUA Config: model={model}, endpoint={endpoint}, container={container_name}", flush=True)
    print(f"  ⚙ Supervisor: {supervisor_model}, max_iter={max_iterations}, max_input={max_user_inputs}", flush=True)

    # Build the OpenAI client
    if endpoint == "azure":
        client_kwargs = {
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
        }
        if use_managed_identity():
            client_kwargs["azure_ad_token_provider"] = get_token_provider()
            print("  ⚙ Auth: Managed Identity", flush=True)
        else:
            client_kwargs["api_key"] = os.getenv("AZURE_OPENAI_API_KEY", "")
            print("  ⚙ Auth: API Key", flush=True)
        openai_client = openai.AzureOpenAI(**client_kwargs)
    else:
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        print("  ⚙ Auth: OpenAI direct", flush=True)

    # Set up the Docker computer
    display = os.getenv("DOCKER_DISPLAY", ":99")
    computer = docker_c.DockerComputer(container_name=container_name, display=display)
    print(f"  🐳 Connecting to Docker container '{container_name}'...", flush=True)
    computer.__enter__()
    print(f"  🐳 Display: {computer.dimensions[0]}x{computer.dimensions[1]}", flush=True)

    # Quick connectivity test — run a harmless xdotool command
    try:
        test_result = computer._exec(f"DISPLAY={display} xdotool getactivewindow getwindowname 2>/dev/null || echo 'no-window'")
        print(f"  🐳 Docker test: active window = '{test_result.strip()}'", flush=True)
    except Exception as e:
        print(f"  ⚠ Docker test failed: {e}", flush=True)
    print(f"  📸 Scaling screenshots for model...", flush=True)
    computer = cua.Scaler(computer)
    print(f"  📸 Scaled to: {computer.dimensions[0]}x{computer.dimensions[1]}", flush=True)

    # Create the CUA agent
    agent = cua.Agent(openai_client, model, computer)
    action_buffer = []

    print(f"  🚀 Starting task: {task[:80]}...", flush=True)
    import time as _time
    t0 = _time.time()
    agent.start_task(
        task,
        instructions=(
            "You are the Shadowboxer, an AI system that executes tasks on a computer. "
            "Complete the task correctly. If errors occur, try alternative steps. "
            "Do not use CTRL+A. When done, provide your final answer."
        ),
    )
    print(f"  🚀 Task started ({_time.time() - t0:.1f}s)", flush=True)

    user_input_count = 0

    for iteration in range(max_iterations):
        user_message = None
        iter_t0 = _time.time()

        # Log the agent's current state
        state_desc = "needs_input" if agent.requires_user_input else "has_actions"
        n_actions = len(agent.state.computer_actions) if agent.state and hasattr(agent.state, 'computer_actions') else 0
        print(f"  [{iteration+1}/{max_iterations}] State: {state_desc} | "
              f"Actions queued: {n_actions} | Buffer: {len(action_buffer)}", flush=True)

        if agent.requires_user_input:
            user_input_count += 1
            if user_input_count > max_user_inputs:
                print(f"  ⏹ Max user inputs reached ({max_user_inputs})", flush=True)
                break

            has_done_work = len(action_buffer) > 0
            print(f"  🤖 Agent requests input (#{user_input_count}): {str(agent.message)[:100]}", flush=True)
            print(f"  🧠 Calling Supervisor ({supervisor_model})...", flush=True)
            sys_msg = {
                "role": "system",
                "content": (
                    f"You are SupervisorGPT. You monitor a CUA agent that controls a computer.\n\n"
                    f"ORIGINAL TASK: {task}\n\n"
                    f"ACTIONS TAKEN SO FAR: {len(action_buffer)}\n"
                    f"ACTION LOG:\n" + "\n".join(f"  - {a}" for a in action_buffer[-5:]) + "\n\n"
                    f"RULES:\n"
                    f"1. If the agent has NOT started working yet (0 actions), tell it to begin immediately.\n"
                    f"2. If the agent HAS completed the task (navigated to the right page, searched, "
                    f"found results, or accomplished what was asked), you MUST call the terminate function.\n"
                    f"3. If the agent is stuck or going in circles, call terminate.\n"
                    f"4. If the agent asks a question, answer it briefly and tell it to continue.\n"
                    f"5. ALWAYS prefer to terminate rather than let the agent keep going unnecessarily.\n"
                    f"6. After {max_iterations // 2}+ actions, strongly consider terminating.\n"
                ),
            }
            user_content = [{"type": "text", "text": str(agent.message)}]
            if agent.last_screenshot:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{agent.last_screenshot}", "detail": "low"},
                })

            # Force terminate after enough work is done
            if has_done_work and len(action_buffer) >= max_iterations // 2:
                tool_choice = {"type": "function", "function": {"name": "terminate"}}
            elif has_done_work:
                tool_choice = "required"
            else:
                tool_choice = "none"

            um = openai_client.chat.completions.create(
                model=supervisor_model,
                messages=[sys_msg, {"role": "user", "content": user_content}],
                tool_choice=tool_choice,
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "terminate",
                        "description": "Call when the task is complete.",
                        "parameters": {
                            "type": "object",
                            "properties": {"final_message": {"type": "string"}},
                            "required": ["final_message"],
                        },
                    },
                }],
            )
            if um.choices[0].message.tool_calls:
                print(f"  ✅ Supervisor: Task complete!", flush=True)
                return {
                    "actions": action_buffer,
                    "completed": True,
                    "final_message": agent.message or "Task completed.",
                }
            user_message = um.choices[0].message.content
            print(f"  🧠 Supervisor says: {str(user_message)[:80]}", flush=True)

        print(f"  ▶ Calling continue_task...", flush=True)
        ct_t0 = _time.time()

        # Log the actual computer actions BEFORE they execute
        if agent.state and agent.state.next_action == "computer_call_output":
            for act in agent.state.computer_actions:
                act_info = {k: v for k, v in vars(act).items() if k != "type"}
                print(f"    🖱 Executing: {act.type} {act_info}", flush=True)

        agent.continue_task(
            user_message,
            instructions=(
                "You are the Shadowboxer. Complete the task correctly. "
                "If errors occur, try alternative steps."
            ),
        )
        print(f"  ▶ continue_task completed ({_time.time() - ct_t0:.1f}s)", flush=True)

        if agent.reasoning_summary:
            action_buffer.append(agent.reasoning_summary)
            print(f"  🎯 Action: {agent.reasoning_summary}", flush=True)
            if on_action:
                on_action(agent.reasoning_summary)

        if agent.message:
            print(f"  💬 Agent: {str(agent.message)[:100]}", flush=True)

        print(f"  ⏱ Iteration {iteration+1} took {_time.time() - iter_t0:.1f}s", flush=True)

    print(f"  📊 Finished: {len(action_buffer)} actions, {user_input_count} supervisor calls", flush=True)
    return {
        "actions": action_buffer,
        "completed": user_input_count <= max_user_inputs,
        "final_message": agent.message,
    }
