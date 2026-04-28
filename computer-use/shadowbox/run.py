#!/usr/bin/env python3
"""
Shadowbox CUA Accelerator — standalone runner.

Usage:
    python -m shadowbox.run "Go to amazon.com and find comfy running shoes"
    python -m shadowbox.run --model gpt-5.4 --endpoint azure "your task here"
    python -m shadowbox.run --help

Reads configuration from .env (or shadowbox/.env). No screen recording,
no COBRA video analysis — just the CUA agent loop.
"""

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

# Load .env from shadowbox dir or project root
_here = os.path.dirname(os.path.abspath(__file__))
for candidate in [os.path.join(_here, ".env"), os.path.join(os.path.dirname(_here), ".env")]:
    if os.path.isfile(candidate):
        load_dotenv(candidate)
        break

import openai
from . import cua
from . import docker_c
from . import local_computer
from .run_logger import RunLogger


async def run(
    task: str,
    model: str = None,
    endpoint: str = None,
    container_name: str = None,
    supervisor_model: str = None,
    max_iterations: int = None,
    max_user_inputs: int = None,
    use_local: bool = False,
):
    """Run a CUA task from start to finish, returning the run directory path."""
    # Resolve config: CLI args > env vars > defaults
    model = model or os.getenv("CUA_MODEL", "gpt-5.4")
    endpoint = endpoint or os.getenv("CUA_ENDPOINT", "azure")
    container_name = container_name or os.getenv("DOCKER_CONTAINER_NAME", "shadowboxer-vnc")
    supervisor_model = supervisor_model or os.getenv("SUPERVISOR_MODEL", "gpt-4.1")
    max_iterations = max_iterations or int(os.getenv("CUA_MAX_ITERATIONS", "50"))
    max_user_inputs = max_user_inputs or int(os.getenv("CUA_MAX_USER_INPUTS", "3"))

    # Logging
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("cua").setLevel(logging.DEBUG)

    # Build the OpenAI client
    use_mi = os.getenv("USE_MANAGED_IDENTITY", "false").lower() in ("true", "1", "yes")

    if endpoint == "azure":
        client_kwargs = {
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
        }
        if use_mi:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
            client_kwargs["azure_ad_token_provider"] = token_provider
        else:
            client_kwargs["api_key"] = os.getenv("AZURE_OPENAI_API_KEY", "")
        openai_client = openai.AzureOpenAI(**client_kwargs)
    else:
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Set up the computer environment
    if use_local:
        computer = local_computer.LocalComputer()
    else:
        display = os.getenv("DOCKER_DISPLAY", ":99")
        computer = docker_c.DockerComputer(container_name=container_name, display=display)
        computer.__enter__()

    computer = cua.Scaler(computer)

    # Create logger + agent
    run_logger = RunLogger(task=task)
    agent = cua.Agent(openai_client, model, computer, run_logger=run_logger)

    print(f"User: {task}")
    original_task = task

    agent.start_task(
        task,
        instructions=(
            "You are the Shadowboxer, an AI system that takes in a task description "
            "from a user and executes it on a computer. You are responsible for ensuring "
            "the task is completed correctly. If errors occur, try alternative steps. "
            "Do not use the CTRL+A hotkey. When you have completed the task, provide "
            "your final answer and do not ask follow-up questions."
        ),
    )
    run_logger.log_event("task_started", {"prompt": original_task})

    action_buffer = []
    user_input_count = 0

    for iteration in range(max_iterations):
        user_message = None

        if agent.requires_user_input:
            user_input_count += 1

            if user_input_count > max_user_inputs:
                print(f"\n[Supervisor] Agent asked for input {user_input_count} times — auto-terminating.")
                run_logger.finish(reason="auto_terminated_max_user_inputs")
                return run_logger.run_dir

            has_done_work = len(action_buffer) > 0

            sys_msg = {
                "role": "system",
                "content": f"""You are SupervisorGPT, guiding another AI that controls a computer.

RULES:
1. NEVER terminate if no actions have been taken (actions so far: {len(action_buffer)}). The agent MUST use the computer first.
2. If the agent hasn't browsed yet, tell it to start NOW. Be direct.
3. If the task IS done after real actions, call terminate.
4. If the agent asks for unspecified preferences, make reasonable choices. Do NOT echo questions.
5. Keep responses to 1-2 sentences. Be decisive.
6. If stuck in a loop after work is done, call terminate.

Original Task: {original_task}

Actions taken ({len(action_buffer)}):
{action_buffer}

Input requests so far: {user_input_count}""",
            }
            # Build user message with the agent's text + latest screenshot
            user_content = [{"type": "text", "text": str(agent.message)}]
            if agent.last_screenshot:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{agent.last_screenshot}", "detail": "low"},
                })
            user_msg = {"role": "user", "content": user_content}
            tool_choice = "none" if not has_done_work else "auto"

            um = openai_client.chat.completions.create(
                model=supervisor_model,
                messages=[sys_msg, user_msg],
                tool_choice=tool_choice,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "terminate",
                            "description": "Call ONLY when the agent has completed computer actions and the task is done.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "final_message": {
                                        "type": "string",
                                        "description": "Brief summary of what was accomplished",
                                    }
                                },
                                "required": ["final_message"],
                            },
                        },
                    }
                ],
            )
            if um.choices[0].message.tool_calls:
                print("\n[Supervisor] Task complete — terminating.")
                run_logger.finish(reason="supervisor_terminated")
                return run_logger.run_dir
            user_message = um.choices[0].message.content
            print("Supervisor:", user_message)
            run_logger.log_supervisor_response(user_message)

        agent.continue_task(
            user_message,
            instructions=(
                "You are the Shadowboxer, an AI system that takes in a task description "
                "from a user and executes it on a computer. You are responsible for ensuring "
                "the task is completed correctly. If errors occur, try alternative steps. "
                "Do not use the CTRL+A hotkey. When you have completed the task, provide "
                "your final answer and do not ask follow-up questions."
            ),
        )

        if agent.reasoning_summary:
            print(f"\nAction: {agent.reasoning_summary}")
            action_buffer.append(agent.reasoning_summary)
            run_logger.log_reasoning(agent.reasoning_summary)

        if agent.message:
            print(f"Agent: {agent.message}\n")
            run_logger.log_agent_message(agent.message)

    run_logger.finish(reason="max_iterations_reached")
    print(f"\n[Supervisor] Reached max iterations ({max_iterations}) — stopping.")
    return run_logger.run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Shadowbox CUA Accelerator — run a computer-use task with GPT-5.4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m shadowbox.run "find comfy running shoes on amazon"
  python -m shadowbox.run --endpoint openai "go to google and search hello"
  python -m shadowbox.run --local "open calculator and compute 42*17"
  python -m shadowbox.run --model gpt-5.4 --max-iterations 30 "your task"
        """,
    )
    parser.add_argument("task", help="The task to execute on the computer")
    parser.add_argument("--model", default=None, help="Model name (default: CUA_MODEL env or gpt-5.4)")
    parser.add_argument("--endpoint", default=None, choices=["azure", "openai"], help="API endpoint")
    parser.add_argument("--container", default=None, help="Docker container name")
    parser.add_argument("--supervisor-model", default=None, help="Model for the supervisor")
    parser.add_argument("--max-iterations", type=int, default=None, help="Max CUA loop iterations")
    parser.add_argument("--max-user-inputs", type=int, default=None, help="Max supervisor responses before auto-stop")
    parser.add_argument("--local", action="store_true", help="Use local computer instead of Docker")

    args = parser.parse_args()

    run_dir = asyncio.run(
        run(
            task=args.task,
            model=args.model,
            endpoint=args.endpoint,
            container_name=args.container,
            supervisor_model=args.supervisor_model,
            max_iterations=args.max_iterations,
            max_user_inputs=args.max_user_inputs,
            use_local=args.local,
        )
    )
    print(f"\nRun data: {run_dir}")


if __name__ == "__main__":
    main()
