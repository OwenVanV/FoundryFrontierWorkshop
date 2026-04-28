"""Entry point for `python -m shadowbox`."""
from .shadowbox import toggle_recording, analyze
from .shadowboxer import run_cua
import platform
import sys
from pynput import keyboard as pynput_keyboard
import asyncio


def main():
    # --test-cua "your prompt here"  →  skip recording, just run CUA directly
    if len(sys.argv) >= 2 and sys.argv[1] == "--test-cua":
        prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "go to https://www.google.com and search for 'hello world'"
        print(f"Running CUA with prompt: {prompt}")
        asyncio.run(run_cua(prompt))
        return

    print("Press Ctrl+Shift+R to start/stop screen recording...")
    if platform.system() == "Darwin":
        print(
            "(macOS: Make sure Terminal/IDE has Accessibility permissions in "
            "System Settings > Privacy & Security > Accessibility)"
        )
    with pynput_keyboard.GlobalHotKeys({
        '<ctrl>+<shift>+r': toggle_recording
    }) as hotkey:
        hotkey.join()


if __name__ == "__main__":
    main()
