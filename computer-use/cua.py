
import base64
import io
import logging
import re
import time
import typing

import openai
import PIL.Image

logger = logging.getLogger(__name__)


class State:
    """Tracking and controlling the state for GPT-5.4 computer use."""

    previous_response_id: str
    next_action: typing.Literal["user_interaction", "computer_call_output"] = ""
    previous_computer_id: str = ""
    computer_actions: list = []          # GPT-5.4: batched actions[] array
    pending_safety_checks: list = []
    reasoning_summary: str = ""
    message: str = ""

    def __init__(self, response):
        assert response.status == "completed"
        self.previous_response_id = response.id
        self.computer_actions = []
        self.pending_safety_checks = []
        self.reasoning_summary = ""
        self.message = ""

        for item in response.output:
            if item.type == "computer_call":
                # GPT-5.4 returns a batched actions[] array per computer_call
                self.next_action = "computer_call_output"
                self.previous_computer_id = item.call_id
                self.computer_actions = item.actions
                self.pending_safety_checks = getattr(item, "pending_safety_checks", []) or []
            elif item.type == "reasoning":
                self.reasoning_summary = "".join(
                    [summary.text for summary in item.summary]
                )
            elif item.type == "message":
                self.next_action = "user_interaction"
                self.message += item.content[-1].text
            else:
                logger.debug("Ignoring response output type '%s'.", item.type)


class Scaler:
    """Wrapper for a computer instance that performs resizing and coordinate translation."""

    def __init__(self, computer, dimensions=None):
        self.computer = computer
        self.dimensions = dimensions
        if not self.dimensions:
            # If no dimensions are given, take a screenshot and scale to fit
            # GPT-5.4 docs recommend 1440x900 or 1600x900 for best performance
            image = self._screenshot()
            width, height = image.size
            max_width, max_height = 1600, 900
            if width <= max_width and height <= max_height:
                self.dimensions = (width, height)
            else:
                scale = min(max_width / width, max_height / height)
                self.dimensions = (int(width * scale), int(height * scale))
        self.environment = computer.environment
        self.screen_width = -1
        self.screen_height = -1

    def screenshot(self) -> str:
        # Take a screenshot from the actual computer
        image = self._screenshot()
        # Scale the screenshot
        self.screen_width, self.screen_height = image.size
        width, height = self.dimensions
        ratio = min(width / self.screen_width, height / self.screen_height)
        new_width = int(self.screen_width * ratio)
        new_height = int(self.screen_height * ratio)
        resized_image = image.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
        image = PIL.Image.new("RGB", (width, height), (0, 0, 0))
        image.paste(resized_image, (0, 0))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        data = bytearray(buffer.getvalue())
        return base64.b64encode(data).decode("utf-8")

    def click(self, x: int, y: int, button: str = "left") -> None:
        x, y = self._point_to_screen_coords(x, y)
        self.computer.click(x, y, button=button)

    def double_click(self, x: int, y: int) -> None:
        x, y = self._point_to_screen_coords(x, y)
        self.computer.double_click(x, y)

    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        x, y = self._point_to_screen_coords(x, y)
        self.computer.scroll(x, y, scroll_x, scroll_y)

    def type(self, text: str) -> None:
        self.computer.type(text)

    def wait(self, ms: int = 2000) -> None:
        self.computer.wait(ms)

    def move(self, x: int, y: int) -> None:
        x, y = self._point_to_screen_coords(x, y)
        self.computer.move(x, y)

    def keypress(self, keys: list[str]) -> None:
        self.computer.keypress(keys)

    def drag(self, path: list[dict[str, int]]) -> None:
        for point in path:
            x, y = self._point_to_screen_coords(point.x, point.y)
            point.x = x
            point.y = y
        self.computer.drag(path)

    def _screenshot(self):
        # Take screenshot from the actual computer.
        screenshot = self.computer.screenshot()
        screenshot = base64.b64decode(screenshot)
        buffer = io.BytesIO(screenshot)
        return PIL.Image.open(buffer)

    def _point_to_screen_coords(self, x, y):
        width, height = self.dimensions
        ratio = min(width / self.screen_width, height / self.screen_height)
        x = x / ratio
        y = y / ratio
        return int(x), int(y)


class Agent:
    """CUA agent for GPT-5.4 computer use (batched actions, simplified tool definition)."""

    def __init__(self, client, model, computer):
        self.client = client
        self.model = model
        self.computer = computer
        self.state = None

    def start_task(self, user_message, instructions=None):
        tools = [self.computer_tool()]
        response = self.client.responses.create(
            model=self.model,
            input=user_message,
            tools=tools,
            instructions=instructions,
            truncation="auto",
        )
        self.state = State(response)

    @property
    def requires_user_input(self):
        return self.state.next_action == "user_interaction"

    @property
    def requires_consent(self):
        return self.state.next_action == "computer_call_output"

    @property
    def pending_safety_checks(self):
        return self.state.pending_safety_checks

    @property
    def reasoning_summary(self):
        return self.state.reasoning_summary

    @property
    def message(self):
        return self.state.message

    # -----------------------------------------------------------------
    # Action dispatcher — maps GPT-5.4 batched actions to the computer
    # -----------------------------------------------------------------
    def _execute_action(self, action):
        """Execute a single action from the batched actions[] array."""
        action_type = action.type
        if action_type == "screenshot":
            # No-op; we always capture a screenshot after the whole batch
            return
        elif action_type == "click":
            self.computer.click(
                action.x, action.y,
                button=getattr(action, "button", "left"),
            )
        elif action_type == "double_click":
            self.computer.double_click(action.x, action.y)
        elif action_type == "scroll":
            scroll_x = getattr(action, "scroll_x",
                        getattr(action, "scrollX",
                        getattr(action, "delta_x", 0)))
            scroll_y = getattr(action, "scroll_y",
                        getattr(action, "scrollY",
                        getattr(action, "delta_y", 0)))
            self.computer.scroll(action.x, action.y, scroll_x, scroll_y)
        elif action_type == "type":
            self.computer.type(action.text)
        elif action_type == "keypress":
            self.computer.keypress(action.keys)
        elif action_type == "wait":
            self.computer.wait(getattr(action, "ms", 2000))
        elif action_type == "move":
            self.computer.move(action.x, action.y)
        elif action_type == "drag":
            self.computer.drag(action.path)
        else:
            logger.warning("Unknown action type: %s", action_type)

    def continue_task(self, user_message="", instructions=None):
        screenshot = ""
        previous_response_id = self.state.previous_response_id
        if self.state.next_action == "computer_call_output":
            # GPT-5.4: execute ALL actions in the batch, then screenshot
            for action in self.state.computer_actions:
                logger.info("Action: %s %s", action.type,
                            {k: v for k, v in vars(action).items() if k != "type"})
                self._execute_action(action)
            screenshot = self.computer.screenshot()
        if self.state.next_action == "computer_call_output":
            # GPT-5.4: plain dict input with computer_screenshot output
            next_input = {
                "type": "computer_call_output",
                "call_id": self.state.previous_computer_id,
                "output": {
                    "type": "computer_screenshot",
                    "image_url": f"data:image/png;base64,{screenshot}",
                },
            }
            if self.state.pending_safety_checks:
                next_input["acknowledged_safety_checks"] = (
                    self.state.pending_safety_checks
                )
        else:
            next_input = {
                "role": "user",
                "content": user_message,
            }
        tools = [self.computer_tool()]
        self.state = None
        wait_time = 0
        for _ in range(10):
            try:
                time.sleep(wait_time)
                next_response = self.client.responses.create(
                    model=self.model,
                    input=[next_input],
                    previous_response_id=previous_response_id,
                    tools=tools,
                    instructions=instructions,
                    truncation="auto",
                )
                self.state = State(next_response)
                return
            except openai.RateLimitError as e:
                match = re.search(r"Please try again in (\d+)s", e.message)
                wait_time = int(match.group(1)) if match else 10
                logger.info("Rate limit exceeded. Waiting for %s seconds.", wait_time)
        logger.critical("Max retries exceeded.")

    def computer_tool(self):
        # GPT-5.4: simplified tool definition — no display_width/height/environment
        return {"type": "computer"}
