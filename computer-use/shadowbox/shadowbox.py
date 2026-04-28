import subprocess
import time
import platform

import sys
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from pynput import keyboard as pynput_keyboard
# from cobrapy import VideoClient
# from cobrapy.analysis import BasicSummary,ActionSummary
from .shadowboxer import run_cua
import asyncio
is_recording = False
ffmpeg_process = None
filename = "shadow.mp4"
AZURE_OPENAI_GPT_VISION_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_GPT_VISION_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_GPT_VISION_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_GPT_VISION_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
async def analyze():
    """
    Replace this with your cobrapy logic, for example:
    
        from cobrapy import process_video
        data = process_video(filename)
        print(data)
    """
    client = VideoClient(video_path="./shadow.mp4")
    print(client.manifest)
    client.preprocess_video(
    #output_directory="./output_dir",  # control where to save the manifest and other files
    segment_length=10,  # how long should each segment be in seconds 
    fps=2,  # how many frames per second to sample from the video (i.e. 1 = 1 frame per second, 1/3 = 1 frame per 3 seconds)
    max_workers=2,  # how many threads to use for processing. Default is to use number of cores minus 1.
    allow_partial_segments=True,  # if False, the last segment will be discarded if it is shorter than segment_length
    overwrite_output=True,  # any files in a directory with the same name will be overwritten
)
    action_config = ActionSummary()
    action_config.system_prompt_lens="Analyze this video from the perspective of an employee making notes as they shadow another employee doing a process. Your job is to watch the video of someone peforming a task on the web, and notate down every single action they take, button they click, text they input, etc..."
    ac=action_config.results_template
    ac[0]["Steps"]="A chronologically ordered list of all the steps taken in the scene of the video. Things like Opened page x, navigated to this section, scrolled to this portion, etc..."
    ac[0]["UIActions"]="A chronologically ordered list of all the UI actions taken in the scene of the video. This includes things like clicking buttons, typing text, etc..."
    action_summary = client.analyze_video(analysis_config=action_config,run_async=False,max_concurrent_tasks=7)

    
#     print(action_summary)

    ## Load the action summary from the JSON file as a json object
    import json
    # print("opening json")
    # with open("C:\\Code\computer-use-model\computer-use\shadow_mp4_2.00fps_10sSegs_cobra\_ActionSummary.json" ,"r") as f:
    #     action_summary = json.load(f)
    # print("json opened")
    client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_GPT_VISION_ENDPOINT,
    api_key=AZURE_OPENAI_GPT_VISION_API_KEY,
    api_version=AZURE_OPENAI_GPT_VISION_API_VERSION
    )
    sys={"role":"system","content":"""
    You are Shadowboxer, an AI system that analyzes a JSON array of scenes from a video that represent a task that a worker is peforming on their computer.
    Your job is to look at the scenes and determine the task the worker is performing, as well as every step they are taking to complete the task.
    Your output will be fed to an AI system that can control a computer, so your output should be a an array of all the steps that need to be performed to accomplish the task using the computer.
    Think about the reasoning behind each step and how they constribute to accomplishing the task at hand, when the Computer using AI gets your output, it should be able to perfectly replicate the task.
    Because tasks are often general, you want to extract the general steps, meaning that if in the scenes the worker clicks on a link to open a new page, that new page may be a different target url, but the action is the same.
    
    Output should be within a JSON object, with the following format being followed (example):
    [
    {
    "task": "Initial Setup",
    "description": "Prepare browser and environment before running any consent scenario. (dont close or accept the cookie banner)",
    "steps": [
      "Open a new browser window and navigate to the Manulife.ca.",
      "Open Developer Tools > Storage > Cookies and Clear all cookies for the domain. (right click domain name in the list)",
      "Verify that the cookie banner is visible.",
      "Open Developer Tools > Storage > Cookies and ensure 'OptanonConsent' and 'OptanonAlertBoxClosed' are NOT present. Use the filters to find the values.",
      "Open JavaScript Console and run: window.OnetrustActiveGroups and window.OnetrustConsent. Confirm default or empty return values.",
      "Respond to the user with the default values of OnetrustActiveGroups and OnetrustConsent, and then ask 'i am done, what is next?'"
    ]
  },
    ]
         
    Each JSON Object in the array should represent a task that needs to be performed with CUA, the idea is to group the tasks together into logical buckets, so that each group of actions gets executed then we move onto the next.
    as an example of this, if the user is performing a task that requires them to open a browser, then search for something, then click on a link, you would have a task that is "Open Browser", with steps "Search for X", then "Click on Link Y".
    if the task is longer and more complicated, you can break it down into multiple tasks, with their own consituent steps.
    
    The steps should be general, but the task should be specific to what the user is doing.
    Dont mention any actions that are not visible in the video, and do not make any assumptions about what the user is doing.
    If popups or ads play, dont mention them, just focus on the task at hand.
    Dont include things like pressing alt+9 for screen recording,that is to trigger the process that triggers you.
    The task you output needs to be an end to end process, and the steps and UI actions are how you accomplish that task.
    """}
    # sys={"role":"system","content":"""
    # You are Shadowboxer, an AI system that analyzes a JSON array of scenes from a video that represent a task that a worker is peforming on their computer.
    # Your job is to look at the scenes and determine the task the worker is performing, as well as every step they are taking to complete the task.
    # Your output will be fed to an AI system that can control a computer, so your output should be a an array of all the steps that need to be performed to accomplish the task using the computer.
    # Think about the reasoning behind each step and how they constribute to accomplishing the task at hand, when the Computer using AI gets your output, it should be able to perfectly replicate the task.
    # Because tasks are often general, you want to extract the general steps, meaning that if in the scenes the worker clicks on a link to open a new page, that new page may be a different target url, but the action is the same.
    # Output should be within a JSON object, with a key for "steps", a key for "UIActions", and a key for "task".
    # The steps should be general, but the task should be specific to what the user is doing.
    # Dont mention any actions that are not visible in the video, and do not make any assumptions about what the user is doing.
    # If popups or ads play, dont mention them, just focus on the task at hand.
    # Dont include things like pressing alt+9 for screen recording,that is to trigger the process that triggers you.
    # The task you output needs to be an end to end process, and the steps and UI actions are how you accomplish that task.
    # Add in Templatized placeholders in the task, steps, and UIActions so that the output can be used to generate a script that can be used to automate the task. This is the way you should think of the problem, turn the scenes into templatized automation insturctions another machine could use to automate the task.
    # """}
    print("Generating Plan..")
    user={"role":"user","content":str(action_summary)}
    response= client.chat.completions.create(
    model="gpt-4.1",
    messages=[sys,user],
    
    )
    print("Plan Generated...\n")
    
    print("Response: \n\n",response.choices[0].message.content)
    final=response.choices[0].message.content
#     final="""
# ```json
# {
#   "task": "Go to outlook, and retrieve my latest email from Nate Harris",
#   "steps": [
#     "Open Outlook from the toolbar",
#     "Type 'Nate Harris' into the Bing search bar.",
#     "Initiate the search and wait for the results to load.",
#     "Click on the Latest link from the search results.",
#     "Tell the user the contents",

#   ],
#   "UIActions": [
#     "Open Outlook.",
#     "Type 'Nate Harris' into the Bing search bar.",
#     "Click search button.",
#     "Click on the email link.",
#   ]
# }
# ```
# """
#     final="""
# ```json
# {
#   "task": "Search for and browse content on YouTube, specifically the Veritasium channel.",
#   "steps": [
#     "Open the web browser.",
#     "Type 'YouTube' into the Bing search bar.",
#     "Initiate the search and wait for the results to load.",
#     "Click on the YouTube link from the search results.",
#     "Wait for the YouTube homepage to load.",
#     "Type 'Veritasium' into the YouTube search bar.",
#     "Initiate the search and wait for the results to load.",
#     "Click on the Veritasium channel link from the search results.",
#     "Navigate through the Veritasium channel page.",
#     "Select the 'Latest' filter for videos."
#   ],
#   "UIActions": [
#     "Open web browser.",
#     "Type 'YouTube' into the Bing search bar.",
#     "Click search button.",
#     "Click on the YouTube link.",
#     "Type 'Veritasium' into the YouTube search bar.",
#     "Click enter to initiate search.",
#     "Click Veritasium channel link.",
#     "Select 'Latest' filter for videos."
#   ]
# }
# ```
# """
    with open("shadow.txt","w") as f:
        f.write(final)
    await run_cua(final)
    
    print("Analyze function called.")

def toggle_recording():
    global is_recording, ffmpeg_process

    if not is_recording:
        is_recording = True
        
        if platform.system() == "Darwin":  # macOS
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f", "avfoundation",
                "-framerate", "30",
                "-capture_cursor", "1",
                "-i", "1:none",       # Screen index 1, no audio
                "-vf", "scale=1920:1080",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                filename
            ]
        else:  # Windows
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f", "gdigrab",
                "-framerate", "30",
                "-i", "desktop",
                "-vf", "scale=1920:1080",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                filename
            ]
        
        # Start FFmpeg in the background with an interactive stdin so we can send 'q'
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,         # allow us to send commands
            stdout=subprocess.DEVNULL,     # discard stdout
            stderr=subprocess.DEVNULL,     # discard stderr
            text=True                      # so we can write strings easily
        )
        
        print("Recording started...")
        
    else:
        # Stop recording
        is_recording = False
        
        if ffmpeg_process is not None:
            # Gracefully stop FFmpeg by sending 'q'
            ffmpeg_process.communicate("q")
            ffmpeg_process = None
        
        print("Recording stopped. Output saved to:", filename)
        
        # Now call your analysis logic
        print("Analyzing video...")
        asyncio.run(analyze())
        print("Analysis complete.")

# Set a global hotkey (Ctrl+Shift+R) to toggle recording on/off

if __name__ == "__main__":
    # When run directly, use the __main__.py entry point
    from . import __main__
    # __main__.main()

# analyze()
# run_cua("go to https://www.qbe.com/au/home-insurance/home-contents-insurance and get a home quote")