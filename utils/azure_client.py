"""
============================================================================
PayPal x Azure AI Workshop — Azure OpenAI Client Factory
============================================================================

PURPOSE:
    Centralized client configuration for Azure OpenAI and Azure AI Foundry.
    All workshop modules import from here to ensure consistent configuration,
    retry logic, and telemetry integration.

DESIGN PHILOSOPHY:
    - Single source of truth for Azure credentials and endpoints
    - Automatic retry with exponential backoff (production pattern)
    - Token usage tracking for cost analysis in Module 06
    - OpenTelemetry span integration for every API call

USAGE:
    from utils.azure_client import get_openai_client, call_openai

    client = get_openai_client()
    response = call_openai(
        messages=[{"role": "user", "content": "Analyze this data..."}],
    )

============================================================================
"""

import os
import sys
import time
import json
from typing import Optional

from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Ensure stdout handles UTF-8 on Windows (prevents charmap codec errors
# from box-drawing characters and emojis in streaming output)
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load environment variables from .env file
load_dotenv()

# Import the centralized auth helper
from utils.auth import get_openai_client_args, use_managed_identity

# ============================================================================
# GLOBAL TOKEN TRACKING
# ============================================================================
# We track cumulative token usage across all modules so Module 06 can
# report total cost and efficiency metrics. This is a key observability
# pattern: instrument everything, aggregate later.
# ============================================================================

_token_usage_log: list[dict] = []


def get_token_usage_log() -> list[dict]:
    """Return the cumulative token usage log for cost analysis."""
    return _token_usage_log


def get_total_tokens() -> dict:
    """Return aggregate token counts across all API calls."""
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}
    for entry in _token_usage_log:
        totals["prompt_tokens"] += entry.get("prompt_tokens", 0)
        totals["completion_tokens"] += entry.get("completion_tokens", 0)
        totals["total_tokens"] += entry.get("total_tokens", 0)
        totals["api_calls"] += 1
    return totals


# ============================================================================
# CLIENT FACTORY
# ============================================================================

_client: Optional[AzureOpenAI] = None


def get_openai_client() -> AzureOpenAI:
    """Get or create a singleton Azure OpenAI client.

    The client is created once and reused across all modules in a session.
    This is the recommended pattern — creating multiple clients wastes
    connection pool resources.

    Supports two authentication modes (controlled by USE_MANAGED_IDENTITY):
        - API Key: Uses AZURE_OPENAI_API_KEY (default, simpler for workshops)
        - Managed Identity: Uses DefaultAzureCredential (production pattern)
    """
    global _client
    if _client is None:
        try:
            args = get_openai_client_args()
        except EnvironmentError as e:
            raise EnvironmentError(
                "\n"
                "╔══════════════════════════════════════════════════════════╗\n"
                "║  Azure OpenAI credentials not configured!               ║\n"
                "║                                                         ║\n"
                "║  Option A (API Key — default):                          ║\n"
                "║    1. Copy .env.example to .env                        ║\n"
                "║    2. Fill in AZURE_OPENAI_ENDPOINT and API_KEY        ║\n"
                "║                                                         ║\n"
                "║  Option B (Managed Identity):                           ║\n"
                "║    1. Set USE_MANAGED_IDENTITY=true in .env            ║\n"
                "║    2. Run `az login` locally                            ║\n"
                "║    3. Or use Managed Identity on Azure                  ║\n"
                "╚══════════════════════════════════════════════════════════╝\n"
            ) from e

        _client = AzureOpenAI(**args)

        auth_mode = "Managed Identity" if use_managed_identity() else "API Key"
        # Silent log — only visible when debugging
        import logging
        logging.getLogger("azure_client").debug(f"OpenAI client created with {auth_mode}")

    return _client


# ============================================================================
# CORE API CALL WRAPPER
# ============================================================================
# This wrapper adds:
# 1. Automatic retry with exponential backoff (handles 429 rate limits)
# 2. Token usage tracking for cost analysis
# 3. Latency measurement for performance monitoring
# 4. Structured logging for the telemetry pipeline
# ============================================================================

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
def call_openai(
    messages: list[dict],
    model: Optional[str] = None,
    response_format: Optional[dict] = None,
    module_name: str = "unknown",
    print_stream: bool = True,
) -> dict:
    """Make a STREAMING, tracked, retryable call to Azure OpenAI.

    All responses stream by default. Tokens are printed to stdout as they
    arrive, giving participants real-time visibility into the LLM's output.
    TTFT (Time to First Token) is measured and included in the result.

    Args:
        messages:        Chat messages in OpenAI format
        model:           Deployment name (defaults to AZURE_OPENAI_DEPLOYMENT env var)
        response_format: Optional response format (e.g., {"type": "json_object"})
        module_name:     Which workshop module is calling (for telemetry grouping)
        print_stream:    Whether to print tokens as they arrive (default: True)

    Returns:
        dict with keys: content, usage, latency_ms, ttft_ms, model, module
    """
    client = get_openai_client()
    deployment = model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    # Time the API call
    start_time = time.time()
    ttft_ms = None

    # Build the API call kwargs — always stream
    kwargs = {
        "model": deployment,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if response_format:
        kwargs["response_format"] = response_format

    # Stream the response, collecting content and measuring TTFT
    collected_content = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if print_stream:
        # Print a streaming header
        print("  ┌─ LLM ──────────────────────────────────────────────────────")
        print("  │ ", end="", flush=True)

    line_char_count = 0  # Track characters on current line for wrapping

    with client.chat.completions.create(**kwargs) as stream:
        for chunk in stream:
            # Measure TTFT on the first content chunk
            if ttft_ms is None and chunk.choices and chunk.choices[0].delta.content:
                ttft_ms = round((time.time() - start_time) * 1000, 1)

            # Collect content tokens
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                collected_content.append(token)

                if print_stream:
                    for char in token:
                        if char == "\n":
                            print()
                            print("  │ ", end="", flush=True)
                            line_char_count = 0
                        else:
                            print(char, end="", flush=True)
                            line_char_count += 1
                            # Soft-wrap at 65 chars for readability
                            if line_char_count >= 65 and char == " ":
                                print()
                                print("  │ ", end="", flush=True)
                                line_char_count = 0

            # Extract usage from the final chunk (stream_options includes it)
            if chunk.usage:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }

    elapsed_ms = round((time.time() - start_time) * 1000, 1)
    if ttft_ms is None:
        ttft_ms = elapsed_ms  # Fallback if no content was streamed

    if print_stream:
        print()
        print(f"  └─ TTFT: {ttft_ms}ms │ Total: {elapsed_ms}ms │ "
              f"Tokens: {usage['total_tokens']:,} ─────────")

    full_content = "".join(collected_content)

    # Log to the global token tracker
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "module": module_name,
        "model": deployment,
        "latency_ms": elapsed_ms,
        "ttft_ms": ttft_ms,
        **usage,
    }
    _token_usage_log.append(log_entry)

    return {
        "content": full_content,
        "usage": usage,
        "latency_ms": elapsed_ms,
        "ttft_ms": ttft_ms,
        "model": deployment,
        "module": module_name,
    }


def call_openai_json(
    messages: list[dict],
    model: Optional[str] = None,
    module_name: str = "unknown",
    print_stream: bool = True,
) -> dict:
    """Make a streaming Azure OpenAI call with JSON response format.

    Convenience wrapper that forces JSON output — essential for structured
    data extraction in Modules 02-04 where we need parseable results.
    Streams tokens live and parses the completed JSON.

    Returns:
        dict with keys: parsed (the parsed JSON), content (raw string),
                        usage, latency_ms, ttft_ms, model, module
    """
    result = call_openai(
        messages=messages,
        model=model,
        response_format={"type": "json_object"},
        module_name=module_name,
        print_stream=print_stream,
    )

    # Parse the JSON content
    try:
        parsed = json.loads(result["content"])
    except json.JSONDecodeError:
        parsed = {"error": "Failed to parse JSON response", "raw": result["content"]}

    result["parsed"] = parsed
    return result


# ============================================================================
# SYSTEM PROMPT LIBRARY
# ============================================================================
# Pre-built system prompts for each module. These encode the "fraud analyst"
# persona and domain expertise that guide the LLM's reasoning.
# ============================================================================

SYSTEM_PROMPTS = {
    "data_explorer": (
        "You are a senior payment data analyst at a major payment processor. "
        "You specialize in understanding transaction patterns, merchant behavior, "
        "and customer demographics. When analyzing data, provide specific numbers, "
        "percentages, and highlight anything that seems unusual or warrants further "
        "investigation. Be precise and data-driven in your analysis."
    ),

    "pattern_detector": (
        "You are a fraud detection specialist with 15 years of experience in "
        "payment processing. You are trained to identify structuring (smurfing), "
        "circular fund flows, shell company patterns, and synthetic identity fraud. "
        "When you detect potential anomalies, explain the specific pattern, why it's "
        "suspicious, and what additional data would help confirm or rule out fraud. "
        "Always output your findings as structured JSON with confidence scores."
    ),

    "cross_reference_analyst": (
        "You are a financial intelligence analyst conducting a multi-dataset "
        "investigation. You excel at connecting signals across different data "
        "sources — transactions, device fingerprints, geolocation, customer "
        "profiles, and risk scores. When cross-referencing, explicitly state "
        "which datasets you're joining, what keys you're matching on, and "
        "what the combined evidence suggests. Build your case methodically."
    ),

    "fraud_investigator": (
        "You are the lead investigator on a suspected fraud ring case. You have "
        "access to transaction records, customer profiles, merchant data, device "
        "fingerprints, and risk signals. Your job is to identify the full scope "
        "of the ring — all members, all shell companies, all transaction patterns. "
        "Present your findings as a formal investigation report with evidence "
        "citations, confidence levels, and recommended next steps."
    ),

    "evaluator": (
        "You are a quality assurance specialist reviewing fraud detection results. "
        "Compare predicted fraud indicators against ground truth data. Calculate "
        "precision, recall, and F1 scores. Identify false positives and false "
        "negatives. Provide specific recommendations for improving detection accuracy."
    ),
}
