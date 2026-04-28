import logging
import os
import asyncio
import sys

import openai
from . import cua
from . import local_computer
from . import docker_c
from .run_logger import RunLogger

from azure.core.credentials import AzureKeyCredential
# from rtclient import RTLowLevelClient, ResponseCreateMessage, ResponseCreateParams

import sounddevice as sd
import numpy as np
import base64
import soundfile as sf
import threading
from pynput import keyboard as pynput_keyboard
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

# Constants for Azure Speech
SPEECH_KEY = "a783459248694cd1a20cda972eb0f03d"
SPEECH_REGION = "westus"

###############################################################################
# Utility Functions
###############################################################################

# async def generate_and_play_audio(text_prompt: str):
#     """Uses your Azure RTLowLevelClient to generate TTS and play via sounddevice."""
#     uri = "wss://azureopenaisim.openai.azure.com"
#     api_key = "15885014a4e04abaa76c0ad198fc1eb5"

#     async with RTLowLevelClient(
#         url=uri,
#         azure_deployment="gpt-4o-realtime-preview",
#         key_credential=AzureKeyCredential(api_key)
#     ) as client:
#         await client.send(
#             ResponseCreateMessage(
#                 response=ResponseCreateParams(
#                     modalities={"audio", "text"},
#                     instructions=text_prompt
#                 )
#             )
#         )

#         audio_bytes = b""
#         done = False
#         while not done:
#             message = await client.recv()
#             if message.type == "response.done":
#                 done = True
#             elif message.type == "error":
#                 print("Error:", message.error)
#                 done = True
#             elif message.type == "response.audio.delta":
#                 audio_bytes += base64.b64decode(message.delta)
#             elif message.type == "response.audio_transcript.delta":
#                 pass

#         if audio_bytes:
#             audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
#             sd.play(audio_array, samplerate=24000)
#             sd.wait()

# def record_audio(filename="current_utterance.wav"):
#     """Records microphone audio while Left Option (Alt) is held down, saves to `filename`."""
#     samplerate = 16000
#     channels = 1
#     audio_frames = []

#     key_pressed = threading.Event()
#     key_released = threading.Event()

#     def on_press(key):
#         if key == pynput_keyboard.Key.alt_l:
#             key_pressed.set()

#     def on_release(key):
#         if key == pynput_keyboard.Key.alt_l:
#             if key_pressed.is_set():
#                 key_released.set()
#                 return False  # Stop listener

#     print("Hold Left Option to record...")
#     listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
#     listener.start()

#     key_pressed.wait()
#     print("Recording started...")

#     def audio_callback(indata, frames, time, status):
#         audio_frames.append(indata.copy())

#     stream = sd.InputStream(samplerate=samplerate, channels=channels, callback=audio_callback)
#     stream.start()

#     key_released.wait()

#     stream.stop()
#     listener.stop()
#     print("Recording stopped.")

#     audio_data = np.concatenate(audio_frames, axis=0)
#     sf.write(filename, audio_data, samplerate)

# def transcribe_audio(audio_file_path):
#     """Uses Azure Cognitive Services to transcribe audio from `audio_file_path`."""
#     speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
#     audio_input = speechsdk.AudioConfig(filename=audio_file_path)
#     speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

#     result = speech_recognizer.recognize_once_async().get()
#     return result.text if result.reason == speechsdk.ResultReason.RecognizedSpeech else ""

###############################################################################
# Main CUA Function
###############################################################################

async def run_cua(
    user_message: str,
    model: str = "gpt-5.4",
    endpoint: str = "azure",
    autoplay: bool = True,
    environment: str = "linux",
    vm_address: str = None
):
    """
    Runs the CUA workflow using the given `user_message` as the initial task prompt.
    
    :param user_message: The user instructions or prompt for the CUA agent.
    :param model: Model name (defaults to "gpt-5.4").
    :param endpoint: Either "azure" or "openai" to configure client usage.
    :param autoplay: Whether to autoplay VM actions (skipping user confirmations).
    :param environment: e.g. "linux" or "windows" (passed to the agent).
    :param vm_address: If using a remote environment, specify the VM address.
    """
    # Configure logging
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    logging.getLogger("cua").setLevel(logging.DEBUG)

    # Choose the client
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
        openai_client = openai.OpenAI()

    # Computer is used to take screenshots and send keystrokes or mouse clicks.
    # If you're actually local, you could use local_computer.LocalComputer()
    # or a Docker container, etc.
    computer = docker_c.DockerComputer()
    computer.__enter__()  # Initialize and detect display dimensions
    # computer = local_computer.LocalComputer()
    # Optional: resize the captured screen to smaller size for model
    computer = cua.Scaler(computer)

    # Create the CUA agent
    run_logger = RunLogger(task=user_message)
    agent = cua.Agent(openai_client, model, computer, run_logger=run_logger)
    action_buffer = []
    # Print user message and start the CUA task
    print(f"User: {user_message}")
    original_user_message = user_message
    agent.start_task(user_message, instructions="You are the Shadowboxer, an AI system that takes in a templatized task description from a user and executes it on a computer. You are responsible for ensuring the task is completed correctly, following the task exactly, unless errors occur in which case try and accomplish the task with different steps. Dont use CTRL + A in firefox it is a bad hotkey")
    run_logger.log_event("task_started", {"prompt": original_user_message})

    # The main loop: continue until the agent no longer has tasks
    max_iterations = 50
    user_input_count = 0
    max_user_inputs = 3  # Max times the supervisor answers before auto-terminating
    for iteration in range(max_iterations):
        user_message = None

        if agent.requires_user_input:
            user_input_count += 1

            # If the agent keeps asking for user input repeatedly, terminate
            if user_input_count > max_user_inputs:
                print(f"\n[Supervisor] Agent asked for input {user_input_count} times — auto-terminating.")
                run_logger.finish(reason="auto_terminated_max_user_inputs")
                return

            # If no actions have been taken yet, don't let supervisor terminate —
            # force the agent to actually do work on the computer first.
            has_done_work = len(action_buffer) > 0

            # We provide a system message with context, then let GPT generate user input
            sys_msg = {
                "role": "system",
                "content": f"""You are SupervisorGPT, an AI system responsible for guiding another AI that is controlling a computer.

IMPORTANT RULES:
1. NEVER terminate if no actions have been taken yet (actions taken so far: {len(action_buffer)}). The agent MUST actually use the computer to accomplish the task before you can consider it done.
2. If the agent has NOT yet browsed/searched/navigated, tell it to start doing the task on the computer NOW. Be direct: "Go to [website] and start the task."
3. If the task HAS been accomplished after real computer actions, call the terminate function.
4. If the agent asks for preferences or details not specified in the original task, make reasonable choices on behalf of the user and tell the agent to proceed. Do NOT echo the questions back.
5. Keep responses very short (1-2 sentences max). Be decisive, not conversational.
6. If the agent seems stuck in a loop asking the same thing after having done work, call terminate.

User Info:
   Name: Owen Van Valkenburg
   Age: 26
   Address: 1259 Twin Oaks Dell, Mississauga, ON, L5H 3J7
   Phone: 416-669-5560
   Cookies: Never
   Current Date: March 12th 2026

Original Task: {original_user_message}

Actions taken so far ({len(action_buffer)}):
{action_buffer}

Times agent has asked for input: {user_input_count}"""
            }
            # Build user message with the agent's text + latest screenshot
            user_content = [{"type": "text", "text": str(agent.message)}]
            if agent.last_screenshot:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{agent.last_screenshot}", "detail": "low"},
                })
            user_msg = {"role": "user", "content": user_content}

            # If no work has been done, prevent the supervisor from terminating
            tool_choice = "none" if not has_done_work else "auto"

            um = openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[sys_msg, user_msg],
                tool_choice=tool_choice,
                tools=[
                   {
                            "type": "function",
                            "function":{
                            "name": "terminate",
                            "description": "Call this ONLY when the agent has actually completed computer actions and the original task is done. NEVER call this if no actions have been taken yet.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "final_message": {"type": "string", "description": "A brief summary of what was accomplished"}
                                },
                                "required": ["final_message"]
                            }
                            }
                        },
                ]
            )
            if um.choices[0].message.tool_calls:
                print(f"\n[Supervisor] Task complete — terminating.")
                run_logger.finish(reason="supervisor_terminated")
                return
            user_message = um.choices[0].message.content
            print("AI Response: ", user_message)
            run_logger.log_supervisor_response(user_message)

        agent.continue_task(user_message, instructions="You are the Shadowboxer, an AI system that takes in a templatized task description from a user and executes it on a computer. You are responsible for ensuring the task is completed correctly, following the task exactly, unless errors occur in which case try and accomplish the task with different steps. Do not use the CTRL + A hotkey, always click and select. When you have completed the task, provide your final answer and do not ask follow-up questions unless the user explicitly asks for more.")

        if agent.reasoning_summary:
            print(f"\nAction: {agent.reasoning_summary}")
            action_buffer.append(agent.reasoning_summary)
            run_logger.log_reasoning(agent.reasoning_summary)

        if agent.message:
            print(f"Agent: {agent.message}\n")
            run_logger.log_agent_message(agent.message)

    run_logger.finish(reason="max_iterations_reached")
    print(f"\n[Supervisor] Reached max iterations ({max_iterations}) — stopping.")
