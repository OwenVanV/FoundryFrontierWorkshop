"""
============================================================================
MODULE 06 — Observability Dashboard
============================================================================

WORKSHOP NARRATIVE:
    You've built an AI-powered fraud detection system across 5 modules.
    Now step back and look at the OPERATIONAL picture:

    - How much did all those LLM calls cost?
    - What were the latency percentiles?
    - Which modules used the most tokens?
    - Is the underlying platform healthy?
    - What does production-grade AI observability look like?

    This module aggregates all telemetry from the workshop and the
    platform's system metrics into a comprehensive dashboard — rendered
    right in the terminal using the Rich library.

LEARNING OBJECTIVES:
    1. Understand AI-specific observability requirements
    2. Aggregate and visualize telemetry from LLM-powered systems
    3. Calculate cost attribution by module/operation
    4. Correlate system health with AI performance
    5. Azure Monitor integration patterns for production deployment

AZURE SERVICES USED:
    - Azure Monitor / Application Insights (export patterns)
    - OpenTelemetry (telemetry aggregation)

ESTIMATED TIME: 15-20 minutes

============================================================================
"""

import os
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np

# ============================================================================
# PATH SETUP
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.telemetry import (
    init_telemetry, create_span, record_metric,
    get_telemetry_store, get_metric_store,
    estimate_cost, export_telemetry_to_file,
)

DATA_DIR = PROJECT_ROOT / "data"
MODULE_NAME = "06_observability_dashboard"

# Try to import Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Try to import plotext for terminal charts
try:
    import plotext as plt
    PLOTEXT_AVAILABLE = True
except ImportError:
    PLOTEXT_AVAILABLE = False


# ============================================================================
# STEP 1: Gather All Telemetry
# ============================================================================

def step_1_gather_telemetry() -> dict:
    """Load all available telemetry data from the workshop run."""
    print("=" * 70)
    print("MODULE 06 — Observability Dashboard")
    print("=" * 70)
    print()

    init_telemetry(service_name=MODULE_NAME)

    telemetry = {
        "workshop_spans": [],
        "workshop_metrics": [],
        "system_telemetry": [],
        "module_findings": {},
    }

    # Load exported telemetry from previous modules
    telemetry_path = DATA_DIR / "telemetry_export.json"
    if telemetry_path.exists():
        with open(telemetry_path, "r", encoding="utf-8") as f:
            exported = json.load(f)
        telemetry["workshop_spans"] = exported.get("spans", [])
        telemetry["workshop_metrics"] = exported.get("metrics", [])
        print(f"  ✓ Workshop telemetry: {len(telemetry['workshop_spans'])} spans, "
              f"{len(telemetry['workshop_metrics'])} metrics")
    else:
        print("  ⚠ No workshop telemetry found (telemetry_export.json)")
        print("    Run modules 01-04 to generate telemetry data")

    # Load system telemetry (platform health)
    system_path = DATA_DIR / "system_telemetry.csv"
    if system_path.exists():
        with open(system_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                telemetry["system_telemetry"].append(row)
        print(f"  ✓ System telemetry: {len(telemetry['system_telemetry']):,} records")

    # Load module findings for token usage data
    for module_num in ["02", "03", "04", "05"]:
        findings_path = DATA_DIR / f"module_{module_num}_findings.json"
        if findings_path.exists():
            with open(findings_path, "r", encoding="utf-8") as f:
                telemetry["module_findings"][f"module_{module_num}"] = json.load(f)

    eval_path = DATA_DIR / "module_05_evaluation.json"
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            telemetry["module_findings"]["module_05"] = json.load(f)

    print()
    return telemetry


# ============================================================================
# STEP 2: Workshop API Usage Dashboard
# ============================================================================

def step_2_api_usage_dashboard(telemetry: dict):
    """Display API usage, token counts, and cost analysis."""
    print("─" * 70)
    print("STEP 2: API Usage & Cost Analysis")
    print("─" * 70)
    print()

    # Aggregate token usage by module
    module_usage = defaultdict(lambda: {
        "api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "total_tokens": 0, "total_latency_ms": 0,
    })

    for span in telemetry["workshop_spans"]:
        module = span.get("attributes", {}).get("module", span.get("name", "unknown"))
        tokens = span.get("attributes", {}).get("tokens_used", 0)
        latency = span.get("duration_ms", 0)

        # Try to extract module name from span name
        for mod_name in ["01_data_exploration", "02_pattern_detection",
                          "03_cross_reference_analysis", "04_fraud_ring_investigation",
                          "05_evaluation_framework"]:
            if mod_name in str(module):
                module = mod_name
                break

        if isinstance(tokens, (int, float)) and tokens > 0:
            module_usage[module]["api_calls"] += 1
            module_usage[module]["total_tokens"] += int(tokens)
            module_usage[module]["total_latency_ms"] += latency

    # Also pull from module findings if available
    for mod_key, findings in telemetry["module_findings"].items():
        usage = findings.get("token_usage", {})
        if usage and mod_key not in module_usage:
            module_usage[mod_key] = {
                "api_calls": usage.get("api_calls", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "total_latency_ms": 0,
            }

    if RICH_AVAILABLE:
        console = Console()

        # API Usage Table
        table = Table(
            title="Workshop API Usage by Module",
            box=box.ROUNDED,
            show_lines=True,
        )
        table.add_column("Module", style="cyan", min_width=30)
        table.add_column("API Calls", justify="right", style="green")
        table.add_column("Total Tokens", justify="right", style="yellow")
        table.add_column("Est. Cost", justify="right", style="red")

        total_calls = 0
        total_tokens = 0
        total_cost = 0.0

        for module, usage in sorted(module_usage.items()):
            calls = usage["api_calls"]
            tokens = usage["total_tokens"]
            prompt = usage.get("prompt_tokens", int(tokens * 0.7))
            completion = usage.get("completion_tokens", int(tokens * 0.3))
            cost = estimate_cost(prompt, completion)

            total_calls += calls
            total_tokens += tokens
            total_cost += cost

            table.add_row(
                module,
                str(calls),
                f"{tokens:,}",
                f"${cost:.4f}",
            )

        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{total_calls}[/bold]",
            f"[bold]{total_tokens:,}[/bold]",
            f"[bold]${total_cost:.4f}[/bold]",
            style="bold",
        )

        console.print(table)
        console.print()
    else:
        # Fallback: plain text table
        print("  {:<32} {:>10} {:>14} {:>12}".format(
            "Module", "API Calls", "Total Tokens", "Est. Cost"))
        print("  " + "─" * 70)

        total_calls = 0
        total_tokens = 0
        total_cost = 0.0

        for module, usage in sorted(module_usage.items()):
            calls = usage["api_calls"]
            tokens = usage["total_tokens"]
            prompt = usage.get("prompt_tokens", int(tokens * 0.7))
            completion = usage.get("completion_tokens", int(tokens * 0.3))
            cost = estimate_cost(prompt, completion)

            total_calls += calls
            total_tokens += tokens
            total_cost += cost

            print("  {:<32} {:>10} {:>14,} {:>12}".format(
                module, calls, tokens, f"${cost:.4f}"))

        print("  " + "─" * 70)
        print("  {:<32} {:>10} {:>14,} {:>12}".format(
            "TOTAL", total_calls, total_tokens, f"${total_cost:.4f}"))
        print()


# ============================================================================
# STEP 3: Latency Analysis
# ============================================================================

def step_3_latency_analysis(telemetry: dict):
    """Analyze and display API latency distribution."""
    print("─" * 70)
    print("STEP 3: Latency Analysis")
    print("─" * 70)
    print()

    latencies = []
    for span in telemetry["workshop_spans"]:
        duration = span.get("duration_ms", 0)
        if duration > 0:
            latencies.append(duration)

    if not latencies:
        print("  No latency data available. Run modules 01-04 first.")
        print()
        return

    latencies = np.array(latencies)

    print(f"  Latency Statistics (across {len(latencies)} operations):")
    print(f"    Mean:   {np.mean(latencies):,.1f} ms")
    print(f"    Median: {np.median(latencies):,.1f} ms")
    print(f"    P90:    {np.percentile(latencies, 90):,.1f} ms")
    print(f"    P95:    {np.percentile(latencies, 95):,.1f} ms")
    print(f"    P99:    {np.percentile(latencies, 99):,.1f} ms")
    print(f"    Max:    {np.max(latencies):,.1f} ms")
    print()

    # Terminal histogram with plotext if available
    if PLOTEXT_AVAILABLE:
        plt.clear_figure()
        plt.hist(latencies.tolist(), bins=20)
        plt.title("API Latency Distribution (ms)")
        plt.xlabel("Latency (ms)")
        plt.ylabel("Count")
        plt.theme("dark")
        plt.plot_size(70, 15)
        plt.show()
        print()
    else:
        # Simple ASCII histogram
        hist, bin_edges = np.histogram(latencies, bins=10)
        max_count = max(hist) if max(hist) > 0 else 1
        print("  Latency Distribution:")
        for i in range(len(hist)):
            bar_length = int(hist[i] / max_count * 40)
            label = f"  {bin_edges[i]:>8,.0f}-{bin_edges[i+1]:>8,.0f}ms"
            bar = "█" * bar_length
            print(f"  {label} │{bar} ({hist[i]})")
        print()


# ============================================================================
# STEP 4: System Health Dashboard
# ============================================================================

def step_4_system_health(telemetry: dict):
    """Display platform health metrics from system telemetry."""
    print("─" * 70)
    print("STEP 4: Platform Health Dashboard")
    print("─" * 70)
    print()

    system_data = telemetry["system_telemetry"]
    if not system_data:
        print("  No system telemetry available.")
        print()
        return

    # Aggregate by service
    service_stats = defaultdict(lambda: {
        "latencies": [], "error_rates": [], "throughputs": [],
        "cpu_utils": [], "memory_utils": [],
    })

    for record in system_data:
        service = record.get("service", "unknown")
        service_stats[service]["latencies"].append(float(record.get("avg_latency_ms", 0)))
        service_stats[service]["error_rates"].append(float(record.get("error_rate_pct", 0)))
        service_stats[service]["throughputs"].append(float(record.get("requests_per_minute", 0)))
        service_stats[service]["cpu_utils"].append(float(record.get("cpu_utilization_pct", 0)))
        service_stats[service]["memory_utils"].append(float(record.get("memory_utilization_pct", 0)))

    if RICH_AVAILABLE:
        console = Console()

        table = Table(
            title="Platform Service Health Summary",
            box=box.ROUNDED,
            show_lines=True,
        )
        table.add_column("Service", style="cyan", min_width=22)
        table.add_column("Avg Latency", justify="right")
        table.add_column("P99 Latency", justify="right")
        table.add_column("Avg Error %", justify="right")
        table.add_column("Avg RPM", justify="right")
        table.add_column("Avg CPU %", justify="right")
        table.add_column("Avg Mem %", justify="right")

        for service, stats in sorted(service_stats.items()):
            lats = np.array(stats["latencies"])
            errs = np.array(stats["error_rates"])
            rpms = np.array(stats["throughputs"])
            cpus = np.array(stats["cpu_utils"])
            mems = np.array(stats["memory_utils"])

            # Color code based on health
            err_avg = np.mean(errs)
            err_style = "green" if err_avg < 1 else ("yellow" if err_avg < 5 else "red")

            table.add_row(
                service,
                f"{np.mean(lats):.1f} ms",
                f"{np.percentile(lats, 99):.1f} ms",
                f"[{err_style}]{err_avg:.2f}%[/{err_style}]",
                f"{np.mean(rpms):,.0f}",
                f"{np.mean(cpus):.1f}%",
                f"{np.mean(mems):.1f}%",
            )

        console.print(table)
        console.print()
    else:
        print("  {:<24} {:>12} {:>12} {:>10} {:>10} {:>10}".format(
            "Service", "Avg Lat(ms)", "P99 Lat(ms)", "Err %", "Avg RPM", "CPU %"))
        print("  " + "─" * 80)

        for service, stats in sorted(service_stats.items()):
            lats = np.array(stats["latencies"])
            errs = np.array(stats["error_rates"])
            rpms = np.array(stats["throughputs"])
            cpus = np.array(stats["cpu_utils"])

            print("  {:<24} {:>12.1f} {:>12.1f} {:>10.2f} {:>10,.0f} {:>10.1f}".format(
                service,
                np.mean(lats),
                np.percentile(lats, 99),
                np.mean(errs),
                np.mean(rpms),
                np.mean(cpus),
            ))
        print()


# ============================================================================
# STEP 5: Evaluation Metrics Summary
# ============================================================================

def step_5_evaluation_summary(telemetry: dict):
    """Display evaluation metrics from Module 05 if available."""
    print("─" * 70)
    print("STEP 5: Detection Quality Metrics")
    print("─" * 70)
    print()

    eval_data = telemetry["module_findings"].get("module_05", {})
    if not eval_data:
        print("  No evaluation data available. Run Module 05 first.")
        print()
        return

    metrics = eval_data.get("metrics", {})
    member_m = metrics.get("member_metrics", {})
    merchant_m = metrics.get("merchant_metrics", {})
    combined_m = metrics.get("combined_metrics", {})

    if RICH_AVAILABLE:
        console = Console()

        table = Table(
            title="Fraud Detection Quality Metrics",
            box=box.ROUNDED,
            show_lines=True,
        )
        table.add_column("Category", style="cyan")
        table.add_column("Precision", justify="center")
        table.add_column("Recall", justify="center")
        table.add_column("F1 Score", justify="center")
        table.add_column("TP", justify="right", style="green")
        table.add_column("FP", justify="right", style="red")
        table.add_column("FN", justify="right", style="yellow")

        for label, m in [("Ring Members", member_m), ("Shell Merchants", merchant_m), ("Combined", combined_m)]:
            if m:
                f1 = m.get("f1_score", 0)
                f1_style = "green" if f1 > 0.8 else ("yellow" if f1 > 0.5 else "red")
                table.add_row(
                    f"[bold]{label}[/bold]" if label == "Combined" else label,
                    f"{m.get('precision', 0):.1%}",
                    f"{m.get('recall', 0):.1%}",
                    f"[{f1_style}]{f1:.1%}[/{f1_style}]",
                    str(m.get("true_positives", 0)),
                    str(m.get("false_positives", 0)),
                    str(m.get("false_negatives", 0)),
                )

        console.print(table)
        console.print()
    else:
        print("  {:<20} {:>10} {:>10} {:>10} {:>6} {:>6} {:>6}".format(
            "Category", "Precision", "Recall", "F1", "TP", "FP", "FN"))
        print("  " + "─" * 70)
        for label, m in [("Ring Members", member_m), ("Shell Merchants", merchant_m), ("Combined", combined_m)]:
            if m:
                print("  {:<20} {:>10.1%} {:>10.1%} {:>10.1%} {:>6} {:>6} {:>6}".format(
                    label,
                    m.get("precision", 0), m.get("recall", 0), m.get("f1_score", 0),
                    m.get("true_positives", 0), m.get("false_positives", 0), m.get("false_negatives", 0),
                ))
        print()

    # Prompt variant comparison
    variant_data = eval_data.get("prompt_variant_results", {})
    variant_results = variant_data.get("variant_results", [])
    if variant_results:
        print("  Prompt Variant Performance:")
        for v in variant_results:
            bar_len = int(v.get("f1_score", 0) * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"    {v['variant_name']:<20} {bar} {v.get('f1_score', 0):.1%}")
        print()


# ============================================================================
# STEP 6: Operational Readiness Assessment
# ============================================================================

def step_6_readiness_assessment(telemetry: dict):
    """Generate an operational readiness score for the AI fraud detection system."""
    print("─" * 70)
    print("STEP 6: Production Readiness Assessment")
    print("─" * 70)
    print()

    checks = []

    # Check 1: Do we have evaluation metrics?
    eval_data = telemetry["module_findings"].get("module_05", {})
    if eval_data:
        combined_f1 = eval_data.get("metrics", {}).get("combined_metrics", {}).get("f1_score", 0)
        checks.append({
            "name": "Detection Accuracy (F1)",
            "status": "PASS" if combined_f1 > 0.5 else "WARN" if combined_f1 > 0.3 else "FAIL",
            "value": f"{combined_f1:.1%}",
            "threshold": "> 50%",
        })
    else:
        checks.append({
            "name": "Detection Accuracy (F1)",
            "status": "SKIP",
            "value": "N/A",
            "threshold": "> 50%",
        })

    # Check 2: System health
    system_data = telemetry["system_telemetry"]
    if system_data:
        error_rates = [float(r.get("error_rate_pct", 0)) for r in system_data]
        avg_error = np.mean(error_rates) if error_rates else 0
        checks.append({
            "name": "Platform Error Rate",
            "status": "PASS" if avg_error < 1 else "WARN" if avg_error < 5 else "FAIL",
            "value": f"{avg_error:.2f}%",
            "threshold": "< 1%",
        })

        latencies = [float(r.get("avg_latency_ms", 0)) for r in system_data
                     if r.get("service") == "payment_processor"]
        if latencies:
            p99_lat = np.percentile(latencies, 99)
            checks.append({
                "name": "Payment Processor P99 Latency",
                "status": "PASS" if p99_lat < 500 else "WARN" if p99_lat < 1000 else "FAIL",
                "value": f"{p99_lat:.0f} ms",
                "threshold": "< 500 ms",
            })

    # Check 3: Telemetry pipeline
    spans = telemetry["workshop_spans"]
    checks.append({
        "name": "Telemetry Coverage",
        "status": "PASS" if len(spans) > 10 else "WARN" if len(spans) > 0 else "FAIL",
        "value": f"{len(spans)} spans",
        "threshold": "> 10 spans",
    })

    # Check 4: Cost efficiency
    for mod_key, findings in telemetry["module_findings"].items():
        cost = findings.get("estimated_cost_usd", 0)
        if cost > 0:
            checks.append({
                "name": f"Cost ({mod_key})",
                "status": "PASS" if cost < 1.0 else "WARN" if cost < 5.0 else "FAIL",
                "value": f"${cost:.4f}",
                "threshold": "< $1.00",
            })

    # Display readiness assessment
    pass_count = sum(1 for c in checks if c["status"] == "PASS")
    total_checks = sum(1 for c in checks if c["status"] != "SKIP")
    readiness_pct = (pass_count / total_checks * 100) if total_checks > 0 else 0

    status_symbols = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "○"}
    status_colors_rich = {"PASS": "green", "WARN": "yellow", "FAIL": "red", "SKIP": "dim"}

    if RICH_AVAILABLE:
        console = Console()

        table = Table(
            title=f"Production Readiness: {readiness_pct:.0f}%",
            box=box.ROUNDED,
            show_lines=True,
        )
        table.add_column("Check", style="cyan", min_width=30)
        table.add_column("Status", justify="center", min_width=8)
        table.add_column("Value", justify="right")
        table.add_column("Threshold", justify="right", style="dim")

        for check in checks:
            color = status_colors_rich[check["status"]]
            symbol = status_symbols[check["status"]]
            table.add_row(
                check["name"],
                f"[{color}]{symbol} {check['status']}[/{color}]",
                check["value"],
                check["threshold"],
            )

        console.print(table)
        console.print()
    else:
        print(f"  Production Readiness Score: {readiness_pct:.0f}%")
        print()
        print("  {:<35} {:>8} {:>15} {:>12}".format("Check", "Status", "Value", "Threshold"))
        print("  " + "─" * 72)
        for check in checks:
            symbol = status_symbols[check["status"]]
            print("  {:<35} {:>8} {:>15} {:>12}".format(
                check["name"],
                f"{symbol} {check['status']}",
                check["value"],
                check["threshold"],
            ))
        print()


# ============================================================================
# STEP 7: Final Workshop Summary
# ============================================================================

def step_7_workshop_summary():
    """Print the final workshop summary and next steps."""
    print("=" * 70)
    print("WORKSHOP COMPLETE — Final Summary")
    print("=" * 70)
    print("""
  Congratulations! You've completed the "Cracking the Ring" workshop.

  WHAT YOU BUILT:
  ┌──────────────────────────────────────────────────────────────────────
  │ Module 01: Data Exploration
  │   → Used Azure OpenAI for intelligent data profiling
  │   → Identified initial anomalies from statistical summaries
  │
  │ Module 02: Pattern Detection
  │   → Structured outputs (JSON mode) for parseable anomaly detection
  │   → Detected smurfing, temporal coordination, shell companies
  │
  │ Module 03: Cross-Reference Analysis
  │   → Multi-turn conversations for progressive investigation
  │   → Cross-referenced devices, geolocation, and transactions
  │
  │ Module 04: Fraud Ring Investigation
  │   → Network graph reasoning to map the full ring
  │   → Generated formal investigation report
  │
  │ Module 05: Evaluation Framework
  │   → Precision/recall/F1 against ground truth
  │   → Prompt variant testing for accuracy improvement
  │
  │ Module 06: Observability Dashboard
  │   → Cost analysis, latency monitoring, system health
  │   → Production readiness assessment
  └──────────────────────────────────────────────────────────────────────

  AZURE SERVICES DEMONSTRATED:
    • Azure OpenAI (GPT-4o) — Chat completions, JSON mode, multi-turn
    • Azure AI Foundry — Model deployment and management
    • Azure Monitor / Application Insights — Telemetry export
    • OpenTelemetry — Distributed tracing and metrics

  KEY TAKEAWAYS:
    1. LLMs excel at cross-referencing signals across heterogeneous data
    2. Structured outputs enable automation, not just analysis
    3. Multi-turn conversations allow progressive investigation
    4. Evaluation frameworks are ESSENTIAL for production trust
    5. Observability is not optional — instrument everything

  NEXT STEPS FOR PRODUCTION:
    • Implement streaming for real-time transaction monitoring
    • Add human-in-the-loop review for high-confidence alerts
    • Deploy evaluation as CI/CD pipeline gates
    • Set up Azure Monitor dashboards and alerting rules
    • Consider fine-tuning for domain-specific fraud patterns
""")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Execute Module 06: Observability Dashboard."""
    telemetry = step_1_gather_telemetry()
    step_2_api_usage_dashboard(telemetry)
    step_3_latency_analysis(telemetry)
    step_4_system_health(telemetry)
    step_5_evaluation_summary(telemetry)
    step_6_readiness_assessment(telemetry)
    step_7_workshop_summary()


if __name__ == "__main__":
    main()
