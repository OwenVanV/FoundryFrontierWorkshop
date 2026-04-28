"""
RunLogger — persists screenshots, actions, and messages for each CUA run.

Creates a timestamped directory under ``runs/`` with:
  screenshots/       – PNG screenshot after every action batch
  run_log.json       – full structured log (actions, messages, reasoning, timing)
  summary.txt        – human-readable run summary
"""

import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path


class RunLogger:
    """Collects runtime data for a single CUA run and writes it to disk."""

    def __init__(self, task: str, runs_dir: str = None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_task = "".join(c if c.isalnum() or c in " -_" else "" for c in task)[:60].strip().replace(" ", "_")
        folder_name = f"{ts}_{safe_task}" if safe_task else ts

        if runs_dir is None:
            # Store runs/ next to the shadowbox package (i.e. project root)
            runs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs")

        self.run_dir = os.path.join(runs_dir, folder_name)
        self.screenshots_dir = os.path.join(self.run_dir, "screenshots")
        os.makedirs(self.screenshots_dir, exist_ok=True)

        self.task = task
        self.start_time = time.time()
        self.step_counter = 0
        self.screenshot_counter = 0
        self.entries: list[dict] = []

        # Write initial metadata
        self._write_json("meta.json", {
            "task": task,
            "started_at": datetime.now().isoformat(),
            "run_dir": self.run_dir,
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_screenshot(self, screenshot_b64: str, label: str = "") -> str:
        """Save a base64 PNG screenshot to disk. Returns the file path."""
        self.screenshot_counter += 1
        filename = f"{self.screenshot_counter:04d}_{label}.png" if label else f"{self.screenshot_counter:04d}.png"
        filepath = os.path.join(self.screenshots_dir, filename)
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(screenshot_b64))
        self._add_entry("screenshot", {"file": filepath, "label": label})
        return filepath

    def log_actions(self, actions: list, step: int = None):
        """Log the batched actions from a computer_call."""
        self.step_counter += 1
        step = step or self.step_counter
        action_dicts = []
        for a in actions:
            try:
                action_dicts.append(vars(a))
            except TypeError:
                action_dicts.append(str(a))
        self._add_entry("actions", {"step": step, "actions": action_dicts})

    def log_reasoning(self, summary: str):
        """Log the model's reasoning summary."""
        if summary:
            self._add_entry("reasoning", {"summary": summary})

    def log_agent_message(self, message: str):
        """Log a message from the CUA agent."""
        if message:
            self._add_entry("agent_message", {"message": message})

    def log_supervisor_response(self, response: str):
        """Log the supervisor's response to the agent."""
        if response:
            self._add_entry("supervisor_response", {"response": response})

    def log_event(self, event_type: str, data: dict = None):
        """Log a generic event."""
        self._add_entry(event_type, data or {})

    def finish(self, reason: str = "completed"):
        """Finalize the run — write the full log and summary to disk."""
        elapsed = time.time() - self.start_time
        self._add_entry("run_finished", {
            "reason": reason,
            "elapsed_seconds": round(elapsed, 2),
            "total_steps": self.step_counter,
            "total_screenshots": self.screenshot_counter,
        })

        # Write full structured log
        self._write_json("run_log.json", {
            "task": self.task,
            "started_at": datetime.fromtimestamp(self.start_time).isoformat(),
            "finished_at": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "total_steps": self.step_counter,
            "total_screenshots": self.screenshot_counter,
            "finish_reason": reason,
            "entries": self.entries,
        })

        # Write human-readable summary
        summary_lines = [
            f"Task: {self.task}",
            f"Started: {datetime.fromtimestamp(self.start_time).isoformat()}",
            f"Finished: {datetime.now().isoformat()}",
            f"Elapsed: {elapsed:.1f}s",
            f"Steps: {self.step_counter}",
            f"Screenshots: {self.screenshot_counter}",
            f"Finish reason: {reason}",
            "",
            "--- Timeline ---",
        ]
        for entry in self.entries:
            ts = entry.get("timestamp", "")
            etype = entry.get("type", "")
            if etype == "actions":
                actions = entry["data"].get("actions", [])
                action_types = [a.get("type", "?") if isinstance(a, dict) else "?" for a in actions]
                summary_lines.append(f"[{ts}] ACTIONS: {' -> '.join(action_types)}")
            elif etype == "reasoning":
                summary_lines.append(f"[{ts}] REASONING: {entry['data'].get('summary', '')[:200]}")
            elif etype == "agent_message":
                summary_lines.append(f"[{ts}] AGENT: {entry['data'].get('message', '')[:200]}")
            elif etype == "supervisor_response":
                summary_lines.append(f"[{ts}] SUPERVISOR: {entry['data'].get('response', '')[:200]}")
            elif etype == "screenshot":
                summary_lines.append(f"[{ts}] SCREENSHOT: {entry['data'].get('file', '')}")
            elif etype == "run_finished":
                summary_lines.append(f"[{ts}] FINISHED: {reason}")
            else:
                summary_lines.append(f"[{ts}] {etype}: {json.dumps(entry.get('data', {}))[:200]}")

        summary_path = os.path.join(self.run_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("\n".join(summary_lines) + "\n")

        print(f"\n[RunLogger] Run data saved to: {self.run_dir}")
        return self.run_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_entry(self, entry_type: str, data: dict):
        self.entries.append({
            "type": entry_type,
            "timestamp": datetime.now().isoformat(),
            "elapsed": round(time.time() - self.start_time, 2),
            "data": data,
        })

    def _write_json(self, filename: str, data: dict):
        filepath = os.path.join(self.run_dir, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
