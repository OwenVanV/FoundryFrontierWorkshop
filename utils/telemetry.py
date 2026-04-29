"""
============================================================================
PayPal x Azure AI Workshop — Telemetry & Observability Setup
============================================================================

PURPOSE:
    Configure OpenTelemetry tracing and metrics collection for the workshop.
    Every Azure OpenAI call, data processing step, and evaluation result
    is captured as telemetry — demonstrating production observability patterns.

WHY THIS MATTERS:
    In production AI systems, you need visibility into:
    - How much each LLM call costs (token usage)
    - How long inference takes (latency percentiles)
    - Whether the model's outputs are consistent (quality metrics)
    - System health during AI-powered processing (error rates)

    This module sets up the telemetry pipeline that feeds Module 06's
    observability dashboard.

MODES:
    1. LOCAL (default): Telemetry exported to console + in-memory store
    2. AZURE MONITOR: Telemetry exported to Application Insights
       (requires APPLICATIONINSIGHTS_CONNECTION_STRING in .env)

USAGE:
    from utils.telemetry import init_telemetry, create_span, record_metric

    init_telemetry(service_name="workshop-module-01")

    with create_span("analyze_transactions") as span:
        span.set_attribute("transaction_count", 50000)
        result = do_analysis()
        record_metric("analysis_duration_ms", elapsed)

============================================================================
"""

import os
import time
import json
from contextlib import contextmanager
from typing import Optional
from datetime import datetime

from dotenv import load_dotenv

# ============================================================================
# OpenTelemetry imports — the industry standard for observability
# ============================================================================
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        SimpleSpanProcessor,
        ConsoleSpanExporter,
        BatchSpanProcessor,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    print("  ⚠ OpenTelemetry not installed. Telemetry will use fallback logging.")

# ============================================================================
# Azure Monitor integration (optional — only if configured)
# ============================================================================
AZURE_MONITOR_AVAILABLE = False
try:
    from azure.monitor.opentelemetry.exporter.export.trace._exporter import AzureMonitorTraceExporter
    AZURE_MONITOR_AVAILABLE = True
except (ImportError, Exception):
    try:
        from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
        AZURE_MONITOR_AVAILABLE = True
    except (ImportError, Exception):
        pass

load_dotenv()

# ============================================================================
# IN-MEMORY TELEMETRY STORE
# ============================================================================
# Even without Azure Monitor, we capture all telemetry in memory so
# Module 06 can build its dashboard from local data. This is a
# common pattern: always instrument, export when ready.
# ============================================================================

_telemetry_store: list[dict] = []
_metric_store: list[dict] = []
_tracer = None
_initialized = False


def get_telemetry_store() -> list[dict]:
    """Return all captured telemetry spans."""
    return _telemetry_store


def get_metric_store() -> list[dict]:
    """Return all captured metrics."""
    return _metric_store


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_telemetry(service_name: str = "paypal-azure-workshop") -> None:
    """Initialize the telemetry pipeline.

    This sets up:
    1. A tracer provider with the service name
    2. Console exporter (always on — for live workshop visibility)
    3. Azure Monitor exporter (if configured — for persistent dashboards)

    Call this once at the top of each workshop module.
    """
    global _tracer, _initialized

    if _initialized:
        return

    if not OTEL_AVAILABLE:
        print(f"  ℹ Telemetry initialized in FALLBACK mode for '{service_name}'")
        _initialized = True
        return

    # Resource identifies this service in telemetry backends
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": "workshop",
    })

    # Set up the tracer provider
    provider = TracerProvider(resource=resource)

    # Always add console exporter so participants see spans in real-time
    console_exporter = ConsoleSpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(console_exporter))

    # Optionally add Azure Monitor exporter
    enable_azure = os.getenv("ENABLE_AZURE_MONITOR", "false").lower() == "true"
    conn_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

    if enable_azure and conn_string and AZURE_MONITOR_AVAILABLE:
        azure_exporter = AzureMonitorTraceExporter(connection_string=conn_string)
        provider.add_span_processor(BatchSpanProcessor(azure_exporter))
        print(f"  ✓ Telemetry: Azure Monitor export ENABLED for '{service_name}'")
    else:
        print(f"  ℹ Telemetry: Console-only mode for '{service_name}'")

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)
    _initialized = True

    # Register atexit handler to flush spans before process exits
    import atexit
    atexit.register(flush_telemetry)


# ============================================================================
# SPAN CREATION — Wraps any operation with timing and attributes
# ============================================================================

@contextmanager
def create_span(
    name: str,
    attributes: Optional[dict] = None,
):
    """Create a telemetry span that tracks an operation.

    Spans capture:
    - Start/end time (for latency calculation)
    - Custom attributes (for filtering and grouping)
    - Success/failure status

    Usage:
        with create_span("llm_call", {"model": "gpt-4o"}) as span:
            result = call_openai(...)
            span.set_attribute("tokens_used", result["usage"]["total_tokens"])

    Even without OpenTelemetry, this captures data to the in-memory store.
    """
    start_time = time.time()
    span_data = {
        "name": name,
        "start_time": datetime.utcnow().isoformat() + "Z",
        "attributes": attributes or {},
        "status": "ok",
    }

    if _tracer:
        # Use real OpenTelemetry span
        with _tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value) if not isinstance(value, (int, float, bool)) else value)

            # Yield a wrapper that records attributes to both OTel and our store
            class SpanWrapper:
                def set_attribute(self, key, value):
                    span.set_attribute(key, str(value) if not isinstance(value, (int, float, bool)) else value)
                    span_data["attributes"][key] = value

            wrapper = SpanWrapper()
            try:
                yield wrapper
                span_data["status"] = "ok"
            except Exception as e:
                span_data["status"] = "error"
                span_data["error"] = str(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                raise
            finally:
                elapsed_ms = round((time.time() - start_time) * 1000, 1)
                span_data["duration_ms"] = elapsed_ms
                span_data["end_time"] = datetime.utcnow().isoformat() + "Z"
                _telemetry_store.append(span_data)
    else:
        # Fallback: no OTel, but still capture to in-memory store
        class FallbackSpan:
            def set_attribute(self, key, value):
                span_data["attributes"][key] = value

        try:
            yield FallbackSpan()
            span_data["status"] = "ok"
        except Exception as e:
            span_data["status"] = "error"
            span_data["error"] = str(e)
            raise
        finally:
            elapsed_ms = round((time.time() - start_time) * 1000, 1)
            span_data["duration_ms"] = elapsed_ms
            span_data["end_time"] = datetime.utcnow().isoformat() + "Z"
            _telemetry_store.append(span_data)


# ============================================================================
# METRIC RECORDING — Point-in-time measurements
# ============================================================================

def record_metric(
    name: str,
    value: float,
    unit: str = "",
    attributes: Optional[dict] = None,
) -> None:
    """Record a metric data point.

    Metrics are different from spans — they capture a single measurement
    at a point in time rather than a duration.

    Common workshop metrics:
    - token_count (prompt_tokens, completion_tokens)
    - api_latency_ms
    - precision, recall, f1_score
    - cost_usd

    Usage:
        record_metric("api_latency_ms", 234.5, unit="ms", attributes={"module": "02"})
    """
    metric_entry = {
        "name": name,
        "value": value,
        "unit": unit,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "attributes": attributes or {},
    }
    _metric_store.append(metric_entry)


# ============================================================================
# COST ESTIMATION — Translate token usage into dollar amounts
# ============================================================================

# Azure OpenAI pricing (approximate, per 1K tokens as of 2026)
# These are for estimation only — actual pricing depends on your agreement
PRICING = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4": {"input": 0.03, "output": 0.06},
}


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "gpt-4o",
) -> float:
    """Estimate the cost of an API call in USD.

    This helps workshop participants understand the economics of LLM-powered
    fraud detection — a key consideration for production deployment.
    """
    pricing = PRICING.get(model, PRICING["gpt-4o"])
    input_cost = (prompt_tokens / 1000) * pricing["input"]
    output_cost = (completion_tokens / 1000) * pricing["output"]
    return round(input_cost + output_cost, 6)


# ============================================================================
# TELEMETRY EXPORT — Save telemetry to disk for Module 06
# ============================================================================

def export_telemetry_to_file(filepath: str = "data/telemetry_export.json") -> str:
    """Export all captured telemetry to a JSON file.

    Module 06 loads this file to build its observability dashboard.
    Call this at the end of each module, or once after running all modules.
    """
    export_data = {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "spans": _telemetry_store,
        "metrics": _metric_store,
        "summary": {
            "total_spans": len(_telemetry_store),
            "total_metrics": len(_metric_store),
            "error_spans": len([s for s in _telemetry_store if s["status"] == "error"]),
        },
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, default=str)

    # Flush any pending Azure Monitor spans before the process exits
    flush_telemetry()

    print(f"  ✓ Telemetry exported to {filepath} ({len(_telemetry_store)} spans, {len(_metric_store)} metrics)")
    return filepath


def flush_telemetry() -> None:
    """Force-flush all pending telemetry spans to Azure Monitor.

    BatchSpanProcessor buffers spans and sends them in batches. If the
    process exits before a batch is flushed, spans are lost. Call this
    at the end of each module to ensure all telemetry reaches App Insights.
    """
    try:
        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush(timeout_millis=10000)
    except Exception:
        pass
