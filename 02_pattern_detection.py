"""
============================================================================
MODULE 02 — Pattern Detection with Structured Outputs
============================================================================

WORKSHOP NARRATIVE:
    Module 01 gave us a bird's-eye view of the data. Now we drill down.

    In this module, you'll use Azure OpenAI's STRUCTURED OUTPUT capability
    (JSON mode) to extract specific, parseable anomaly detections from
    transaction data. Instead of getting free-text analysis, we'll force
    the LLM to output structured JSON with confidence scores, evidence
    citations, and categorized anomaly types.

    This is how production systems work: the LLM's output feeds directly
    into downstream pipelines, dashboards, and alerting systems. Free text
    is great for humans; structured JSON is essential for automation.

LEARNING OBJECTIVES:
    1. Use JSON response format for structured LLM outputs
    2. Design effective zero-shot and few-shot anomaly detection prompts
    3. Analyze transaction amount distributions for structuring (smurfing)
    4. Detect temporal patterns in transaction timing
    5. Understand prompt engineering tradeoffs for detection accuracy

AZURE SERVICES USED:
    - Azure OpenAI (structured outputs / JSON mode)
    - OpenTelemetry (latency histograms, token tracking)

ESTIMATED TIME: 25-30 minutes

============================================================================
"""

import os
import sys
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np

# ============================================================================
# PATH SETUP
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.azure_client import call_openai, call_openai_json, SYSTEM_PROMPTS, get_total_tokens
from utils.telemetry import init_telemetry, create_span, record_metric

DATA_DIR = PROJECT_ROOT / "data"
MODULE_NAME = "02_pattern_detection"


# ============================================================================
# STEP 1: Load Transaction Data and Prepare Samples
# ============================================================================
# We can't send 50,000 transactions to the LLM in one shot — it would
# exceed context limits and cost a fortune. Instead, we use STRATEGIC
# SAMPLING: we select subsets that maximize the chance of surfacing
# anomalies.
#
# WORKSHOP DISCUSSION POINT:
# "How does sampling strategy affect detection accuracy? What biases
# might we introduce by sampling differently?"
# ============================================================================

def step_1_load_and_sample() -> dict:
    """Load transactions and create targeted samples for LLM analysis."""
    print("=" * 70)
    print("MODULE 02 — Pattern Detection with Structured Outputs")
    print("=" * 70)
    print()

    init_telemetry(service_name=MODULE_NAME)

    with create_span("load_all_data") as span:
        # Load transactions
        transactions = []
        with open(DATA_DIR / "transactions.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["amount"] = float(row["amount"])
                transactions.append(row)

        # Load merchants for cross-reference
        merchants = {}
        with open(DATA_DIR / "merchants.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                merchants[row["merchant_id"]] = row

        span.set_attribute("transaction_count", len(transactions))
        span.set_attribute("merchant_count", len(merchants))

    print(f"  Loaded {len(transactions):,} transactions, {len(merchants):,} merchants")
    print()

    # ----------------------------------------------------------------
    # Create targeted samples for different analysis angles
    # ----------------------------------------------------------------
    print("─" * 70)
    print("STEP 1: Creating Targeted Samples")
    print("─" * 70)

    samples = {}

    # Sample 1: HIGH-VALUE TRANSACTIONS ($5,000+)
    # This is where smurfing lives — amounts structured below $10K
    high_value = [t for t in transactions if t["amount"] >= 5000]
    samples["high_value"] = high_value[:200]  # Cap at 200 for context limits
    print(f"  High-value ($5K+): {len(high_value):,} total, sampled {len(samples['high_value'])}")

    # Sample 2: NEAR-THRESHOLD TRANSACTIONS ($9,000-$10,000)
    # This specific range is where smurfing concentrates
    near_threshold = [t for t in transactions if 9000 <= t["amount"] <= 10000]
    samples["near_threshold"] = near_threshold[:200]
    print(f"  Near-threshold ($9K-$10K): {len(near_threshold):,} total, sampled {len(samples['near_threshold'])}")

    # Sample 3: SAME-DAY REPEATED SENDERS
    # Look for accounts that transact multiple times per day
    sender_daily = defaultdict(list)
    for t in transactions:
        day = t["timestamp"][:10]
        sender_daily[(t["sender_id"], day)].append(t)
    repeat_senders = {k: v for k, v in sender_daily.items() if len(v) >= 3}
    repeat_samples = []
    for (sender, day), txns in list(repeat_senders.items())[:50]:
        repeat_samples.extend(txns)
    samples["repeat_senders"] = repeat_samples[:200]
    print(f"  Repeat senders (3+/day): {len(repeat_senders):,} sender-days, sampled {len(samples['repeat_senders'])}")

    # Sample 4: MERCHANT-CONCENTRATED TRANSACTIONS
    # Transactions grouped by merchant to detect shell company patterns
    merchant_txns = defaultdict(list)
    for t in transactions:
        merchant_txns[t["merchant_id"]].append(t)

    # Find merchants with unusual transaction patterns
    merchant_stats = {}
    for mid, txns in merchant_txns.items():
        amounts = [t["amount"] for t in txns]
        merchant_stats[mid] = {
            "count": len(txns),
            "mean": np.mean(amounts),
            "std": np.std(amounts),
            "min": min(amounts),
            "max": max(amounts),
            "unique_senders": len(set(t["sender_id"] for t in txns)),
        }

    # Select merchants with suspiciously tight amount distributions
    # (low std relative to mean = possible structuring)
    suspicious_merchants = []
    for mid, stats in merchant_stats.items():
        if stats["count"] >= 10 and stats["mean"] > 5000:
            cv = stats["std"] / stats["mean"] if stats["mean"] > 0 else 0
            if cv < 0.15:  # Coefficient of variation < 15% = very tight clustering
                suspicious_merchants.append((mid, stats))

    merchant_samples = []
    for mid, _ in suspicious_merchants[:10]:
        merchant_samples.extend(merchant_txns[mid][:20])
    samples["suspicious_merchants"] = merchant_samples
    print(f"  Suspicious merchants (tight CV): {len(suspicious_merchants):,}, sampled {len(samples['suspicious_merchants'])} txns")

    print()
    return {
        "samples": samples,
        "transactions": transactions,
        "merchants": merchants,
        "merchant_stats": merchant_stats,
    }


# ============================================================================
# STEP 2: Zero-Shot Anomaly Detection with Structured Output
# ============================================================================
# This is the core technique: we give the LLM transaction data and a
# detection task, then FORCE it to respond in structured JSON.
#
# Zero-shot means we don't provide examples of what anomalies look like —
# we rely on the model's training data knowledge of fraud patterns.
#
# WORKSHOP NOTE:
# "response_format={"type": "json_object"}" is the key parameter.
# When this is set, the model MUST respond with valid JSON. If you
# don't include a request for JSON in the prompt, the model may produce
# an empty JSON object — always mention JSON in your prompt too."
# ============================================================================

def step_2_zero_shot_detection(data: dict) -> list[dict]:
    """Run zero-shot anomaly detection on high-value transaction sample."""
    print("─" * 70)
    print("STEP 2: Zero-Shot Anomaly Detection (Structured Output)")
    print("─" * 70)
    print()

    # Prepare the data payload — include relevant fields only
    sample_data = []
    for t in data["samples"]["near_threshold"]:
        sample_data.append({
            "transaction_id": t["transaction_id"],
            "timestamp": t["timestamp"],
            "sender_id": t["sender_id"],
            "receiver_id": t["receiver_id"],
            "merchant_id": t["merchant_id"],
            "amount": t["amount"],
            "currency": t["currency"],
            "payment_method": t["payment_method"],
            "status": t["status"],
            "description": t["description"],
        })

    print(f"  Analyzing {len(sample_data)} near-threshold transactions ($9K-$10K)...")

    with create_span("zero_shot_detection", {"sample_size": len(sample_data)}) as span:
        response = call_openai_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["pattern_detector"]},
                {"role": "user", "content": (
                    "Analyze the following transactions that fall in the $9,000-$10,000 range. "
                    "The Bank Secrecy Act requires reporting of transactions over $10,000. "
                    "Structuring (smurfing) is the practice of breaking transactions into "
                    "amounts just below this threshold to avoid reporting.\n\n"
                    "For each suspicious pattern you find, output a JSON object with this structure:\n"
                    "{\n"
                    '  "anomalies": [\n'
                    "    {\n"
                    '      "type": "smurfing|velocity|circular_flow|other",\n'
                    '      "confidence": 0.0 to 1.0,\n'
                    '      "description": "what you found",\n'
                    '      "evidence": {\n'
                    '        "transaction_ids": ["TXN-..."],\n'
                    '        "sender_ids": ["CUS-..."],\n'
                    '        "merchant_ids": ["MER-..."],\n'
                    '        "amount_range": {"min": 0, "max": 0},\n'
                    '        "time_window": "description of timing"\n'
                    "      },\n"
                    '      "recommended_action": "what to do next"\n'
                    "    }\n"
                    "  ],\n"
                    '  "summary": "overall assessment"\n'
                    "}\n\n"
                    f"Transactions:\n{json.dumps(sample_data, indent=2)}"
                )},
            ],
            module_name=MODULE_NAME,
        )

        span.set_attribute("tokens_used", response["usage"]["total_tokens"])
        span.set_attribute("anomalies_found", len(response["parsed"].get("anomalies", [])))
        record_metric("tokens_used", response["usage"]["total_tokens"], "tokens", {"step": "2"})
        record_metric("ttft_ms", response["ttft_ms"], "ms", {"step": "2"})

    anomalies = response["parsed"].get("anomalies", [])
    print()
    print(f"  ✓ Parsed {len(anomalies)} anomalies from JSON response")
    print()

    return anomalies


# ============================================================================
# STEP 3: Temporal Pattern Analysis
# ============================================================================
# Now we look at WHEN transactions happen, not just HOW MUCH they are.
# The fraud ring coordinates transactions within 3-7 minute windows —
# but this pattern is only visible when you look at specific account
# clusters.
#
# WORKSHOP DISCUSSION POINT:
# "Temporal analysis is one of the hardest things to do with rules engines.
# You need to define arbitrary time windows, handle timezone differences,
# and account for legitimate burst behavior (e.g., a business paying
# multiple invoices). LLMs can reason about time naturally."
# ============================================================================

def step_3_temporal_analysis(data: dict) -> list[dict]:
    """Analyze temporal patterns in transaction timing."""
    print("─" * 70)
    print("STEP 3: Temporal Pattern Analysis")
    print("─" * 70)
    print()

    transactions = data["transactions"]

    # Group transactions by 10-minute windows and look for bursts
    time_windows = defaultdict(list)
    for t in transactions:
        ts = datetime.strptime(t["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
        # Round to 10-minute window
        window = ts.replace(minute=(ts.minute // 10) * 10, second=0)
        time_windows[window].append(t)

    # Find windows with unusually many high-value transactions
    suspicious_windows = []
    for window, txns in time_windows.items():
        high_val = [t for t in txns if t["amount"] > 9000]
        if len(high_val) >= 3:  # 3+ high-value txns in a 10-min window is unusual
            suspicious_windows.append({
                "window_start": window.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_transactions": len(txns),
                "high_value_transactions": len(high_val),
                "transactions": [
                    {
                        "transaction_id": t["transaction_id"],
                        "timestamp": t["timestamp"],
                        "sender_id": t["sender_id"],
                        "receiver_id": t["receiver_id"],
                        "merchant_id": t["merchant_id"],
                        "amount": t["amount"],
                    }
                    for t in high_val
                ],
            })

    suspicious_windows.sort(key=lambda w: w["high_value_transactions"], reverse=True)
    print(f"  Found {len(suspicious_windows)} time windows with 3+ high-value transactions")

    # Send the most suspicious windows to the LLM
    analysis_windows = suspicious_windows[:15]  # Top 15 most suspicious

    print(f"  Sending top {len(analysis_windows)} windows to LLM for temporal analysis...")
    print()

    with create_span("temporal_analysis", {"windows_analyzed": len(analysis_windows)}) as span:
        response = call_openai_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["pattern_detector"]},
                {"role": "user", "content": (
                    "Analyze these time windows where multiple high-value ($9,000+) "
                    "transactions occurred within 10 minutes of each other. Look for:\n\n"
                    "1. COORDINATED TIMING: Are the same senders/receivers appearing "
                    "   across multiple windows? This suggests coordination.\n"
                    "2. MERCHANT CONCENTRATION: Are these transactions routing through "
                    "   the same merchants? Shell companies often show this pattern.\n"
                    "3. AMOUNT PRECISION: Are amounts suspiciously similar (suggesting "
                    "   structured payments) or naturally varied?\n"
                    "4. NETWORK PATTERNS: Do the sender/receiver pairs form chains "
                    "   (A→B, B→C, C→D) suggesting circular money flow?\n\n"
                    "Output your findings as JSON:\n"
                    "{\n"
                    '  "temporal_anomalies": [\n'
                    "    {\n"
                    '      "pattern_type": "coordinated_burst|circular_flow|merchant_concentration",\n'
                    '      "confidence": 0.0 to 1.0,\n'
                    '      "windows_involved": ["2026-01-..."],\n'
                    '      "key_accounts": ["CUS-..."],\n'
                    '      "key_merchants": ["MER-..."],\n'
                    '      "description": "detailed finding",\n'
                    '      "network_connections": [{"from": "CUS-...", "to": "CUS-...", "count": N}]\n'
                    "    }\n"
                    "  ],\n"
                    '  "summary": "overall temporal assessment"\n'
                    "}\n\n"
                    f"Suspicious Time Windows:\n{json.dumps(analysis_windows, indent=2)}"
                )},
            ],
            module_name=MODULE_NAME,
        )

        temporal_anomalies = response["parsed"].get("temporal_anomalies", [])
        span.set_attribute("anomalies_found", len(temporal_anomalies))
        record_metric("tokens_used", response["usage"]["total_tokens"], "tokens", {"step": "3"})
        record_metric("ttft_ms", response["ttft_ms"], "ms", {"step": "3"})

    print()
    print(f"  ✓ Parsed {len(temporal_anomalies)} temporal patterns from JSON response")
    print()

    return temporal_anomalies


# ============================================================================
# STEP 4: Merchant Anomaly Detection
# ============================================================================
# Shell companies are a critical component of money laundering operations.
# In this step, we analyze merchant registration patterns and transaction
# profiles to identify potential shell companies.
#
# The fraud ring uses 4 merchants with similar names registered in Delaware
# within 10 days — a classic shell company pattern.
# ============================================================================

def step_4_merchant_analysis(data: dict) -> list[dict]:
    """Analyze merchant profiles for shell company indicators."""
    print("─" * 70)
    print("STEP 4: Merchant Shell Company Detection")
    print("─" * 70)
    print()

    merchants = data["merchants"]
    merchant_stats = data["merchant_stats"]

    # Prepare merchant profiles with transaction statistics
    merchant_profiles = []
    for mid, merchant in merchants.items():
        stats = merchant_stats.get(mid, {})
        merchant_profiles.append({
            "merchant_id": mid,
            "business_name": merchant["business_name"],
            "dba_name": merchant["dba_name"],
            "mcc_code": merchant["mcc_code"],
            "mcc_description": merchant["mcc_description"],
            "city": merchant["city"],
            "state": merchant["state"],
            "country": merchant["country"],
            "registration_date": merchant["registration_date"],
            "verified": merchant["verified"],
            "account_age_days": merchant.get("account_age_days", "N/A"),
            "transaction_count": stats.get("count", 0),
            "avg_transaction_amount": round(stats.get("mean", 0), 2),
            "amount_std_dev": round(stats.get("std", 0), 2),
            "unique_senders": stats.get("unique_senders", 0),
        })

    # Sort by registration date to help the LLM spot clustering
    merchant_profiles.sort(key=lambda m: m["registration_date"])

    # Take a representative sample — newest registrations + any with high avg amounts
    recent_merchants = [m for m in merchant_profiles if m["registration_date"] >= "2025-10-01"]
    high_avg = [m for m in merchant_profiles if m["avg_transaction_amount"] > 5000]
    analysis_set = {m["merchant_id"]: m for m in recent_merchants + high_avg}
    analysis_merchants = list(analysis_set.values())[:100]

    print(f"  Analyzing {len(analysis_merchants)} merchants (recent + high-value)")
    print()

    with create_span("merchant_analysis", {"merchants_analyzed": len(analysis_merchants)}) as span:
        response = call_openai_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["pattern_detector"]},
                {"role": "user", "content": (
                    "Analyze these merchant profiles for shell company indicators. "
                    "Look for:\n\n"
                    "1. NAME SIMILARITY: Companies with suspiciously similar names "
                    "   (common typosquatting, abbreviation, or DBA variations)\n"
                    "2. REGISTRATION CLUSTERING: Multiple companies registered within "
                    "   a short time window, especially in the same state\n"
                    "3. GEOGRAPHIC CONCENTRATION: Multiple entities in states known "
                    "   for shell company formation (Delaware, Nevada, Wyoming)\n"
                    "4. MCC MISMATCH: Merchant category doesn't match their "
                    "   transaction patterns (e.g., 'Business Services' merchant "
                    "   processing amounts typical of electronics)\n"
                    "5. UNUSUAL METRICS: Very tight transaction amount distributions "
                    "   (low std dev) suggesting structured payments\n\n"
                    "Output JSON:\n"
                    "{\n"
                    '  "shell_company_indicators": [\n'
                    "    {\n"
                    '      "merchant_ids": ["MER-..."],\n'
                    '      "names": ["..."],\n'
                    '      "indicator_type": "name_similarity|registration_cluster|...",\n'
                    '      "confidence": 0.0 to 1.0,\n'
                    '      "evidence": "specific details",\n'
                    '      "risk_level": "low|medium|high|critical"\n'
                    "    }\n"
                    "  ],\n"
                    '  "summary": "overall merchant risk assessment"\n'
                    "}\n\n"
                    f"Merchant Profiles:\n{json.dumps(analysis_merchants, indent=2)}"
                )},
            ],
            module_name=MODULE_NAME,
        )

        indicators = response["parsed"].get("shell_company_indicators", [])
        span.set_attribute("indicators_found", len(indicators))
        record_metric("tokens_used", response["usage"]["total_tokens"], "tokens", {"step": "4"})
        record_metric("ttft_ms", response["ttft_ms"], "ms", {"step": "4"})

    print()
    print(f"  ✓ Parsed {len(indicators)} shell company indicators from JSON response")
    print()
    print()

    return indicators


# ============================================================================
# STEP 5: Summary and Findings Export
# ============================================================================

def step_5_summary(
    smurfing_anomalies: list,
    temporal_anomalies: list,
    merchant_indicators: list,
):
    """Consolidate findings and export for Module 03."""
    print("=" * 70)
    print("MODULE 02 — Summary")
    print("=" * 70)

    totals = get_total_tokens()

    # Consolidate all detected entities
    all_suspicious_accounts = set()
    all_suspicious_merchants = set()

    for a in smurfing_anomalies:
        evidence = a.get("evidence", {})
        for sid in evidence.get("sender_ids", []):
            all_suspicious_accounts.add(sid)
        for mid in evidence.get("merchant_ids", []):
            all_suspicious_merchants.add(mid)

    for a in temporal_anomalies:
        for acc in a.get("key_accounts", []):
            all_suspicious_accounts.add(acc)
        for mid in a.get("key_merchants", []):
            all_suspicious_merchants.add(mid)

    for ind in merchant_indicators:
        for mid in ind.get("merchant_ids", []):
            all_suspicious_merchants.add(mid)

    # Export findings for Module 03
    findings = {
        "module": "02_pattern_detection",
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "suspicious_accounts": sorted(all_suspicious_accounts),
        "suspicious_merchants": sorted(all_suspicious_merchants),
        "smurfing_anomalies": smurfing_anomalies,
        "temporal_anomalies": temporal_anomalies,
        "merchant_indicators": merchant_indicators,
        "token_usage": totals,
    }

    output_path = DATA_DIR / "module_02_findings.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, default=str)

    print(f"""
  Detection Summary:
    ├─ Smurfing anomalies:    {len(smurfing_anomalies)}
    ├─ Temporal patterns:     {len(temporal_anomalies)}
    ├─ Shell company signals: {len(merchant_indicators)}
    ├─ Suspicious accounts:   {len(all_suspicious_accounts)}
    └─ Suspicious merchants:  {len(all_suspicious_merchants)}

  API Usage:
    ├─ API Calls:             {totals['api_calls']}
    ├─ Total Tokens:          {totals['total_tokens']:,}
    └─ Findings exported to:  {output_path}

  KEY TAKEAWAY:
  Structured outputs let us extract parseable, actionable results from
  the LLM. Each anomaly has a type, confidence score, and evidence
  citations — ready for downstream processing.

  In Module 03, we'll CROSS-REFERENCE these findings across multiple
  datasets using multi-turn conversations to build a comprehensive
  investigation.

  NEXT: Run 03_cross_reference_analysis.py
""")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Execute Module 02: Pattern Detection."""
    data = step_1_load_and_sample()
    smurfing_anomalies = step_2_zero_shot_detection(data)
    temporal_anomalies = step_3_temporal_analysis(data)
    merchant_indicators = step_4_merchant_analysis(data)
    step_5_summary(smurfing_anomalies, temporal_anomalies, merchant_indicators)


if __name__ == "__main__":
    main()
