"""
============================================================================
MODULE 01 — Data Exploration with Azure OpenAI
============================================================================

WORKSHOP NARRATIVE:
    Welcome to the PayPal x Azure AI fraud investigation workshop!

    You've just been handed a massive dataset from the payment processing
    platform. Before diving into fraud detection, you need to UNDERSTAND
    the data landscape. What does the transaction distribution look like?
    How many merchants are active? What are the typical patterns?

    In this module, you'll use Azure OpenAI as an intelligent data profiling
    tool. Instead of writing complex SQL queries or pandas aggregations,
    you'll feed data summaries to the LLM and get natural-language insights
    that highlight potential areas of concern.

LEARNING OBJECTIVES:
    1. Configure and authenticate with Azure OpenAI
    2. Use LLMs for data profiling and summarization
    3. Understand token usage and API call patterns
    4. Identify initial anomalies through exploratory analysis
    5. Set up telemetry tracing for API calls

AZURE SERVICES USED:
    - Azure OpenAI (chat completions)
    - OpenTelemetry (span tracing, token counting)

ESTIMATED TIME: 20-25 minutes

============================================================================
"""

import os
import sys
import json
import csv
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np

# ============================================================================
# PATH SETUP — Ensure imports work regardless of where you run this from
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.azure_client import call_openai, SYSTEM_PROMPTS, get_total_tokens
from utils.telemetry import init_telemetry, create_span, record_metric

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = PROJECT_ROOT / "data"
MODULE_NAME = "01_data_exploration"


# ============================================================================
# STEP 1: Initialize Telemetry
# ============================================================================
# Every module starts by initializing the telemetry pipeline. This ensures
# all API calls, processing steps, and results are captured for the
# observability dashboard in Module 06.
#
# WORKSHOP NOTE FOR FACILITATORS:
# Point out that init_telemetry() is idempotent — calling it multiple times
# is safe. This is a production pattern: components shouldn't need to know
# if telemetry is already configured.
# ============================================================================

def step_1_initialize():
    """Initialize telemetry and verify data files exist."""
    print("=" * 70)
    print("MODULE 01 — Data Exploration with Azure OpenAI")
    print("=" * 70)
    print()

    init_telemetry(service_name=MODULE_NAME)

    # Verify all data files exist
    required_files = [
        "transactions.csv", "merchants.csv", "customers.csv",
        "payment_sessions.json", "velocity_metrics.csv",
    ]

    print("  Checking data files:")
    for filename in required_files:
        filepath = DATA_DIR / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"    ✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"    ✗ {filename} — MISSING! Run data/generate_synthetic_data.py first.")
            sys.exit(1)

    print()
    return True


# ============================================================================
# STEP 2: Load and Profile Transaction Data
# ============================================================================
# We load the transaction CSV and compute statistical summaries that we'll
# feed to the LLM. The key insight here: LLMs can't process 50,000 rows
# directly (context window limitations), so we pre-aggregate and let the
# LLM reason over the summaries.
#
# WORKSHOP DISCUSSION POINT:
# "What are the tradeoffs of summarizing data before sending it to the LLM
# vs. sending raw rows? When would you choose each approach?"
# ============================================================================

def step_2_profile_transactions() -> dict:
    """Load transactions and compute statistical profiles."""
    print("─" * 70)
    print("STEP 2: Profiling Transaction Data")
    print("─" * 70)

    with create_span("load_transactions") as span:
        transactions = []
        with open(DATA_DIR / "transactions.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["amount"] = float(row["amount"])
                transactions.append(row)
        span.set_attribute("transaction_count", len(transactions))

    print(f"  Loaded {len(transactions):,} transactions")

    # ----------------------------------------------------------------
    # Compute statistical summaries for the LLM
    # ----------------------------------------------------------------
    with create_span("compute_transaction_stats") as span:
        amounts = [t["amount"] for t in transactions]
        currencies = Counter(t["currency"] for t in transactions)
        statuses = Counter(t["status"] for t in transactions)
        payment_methods = Counter(t["payment_method"] for t in transactions)
        tx_types = Counter(t["transaction_type"] for t in transactions)

        # Amount distribution analysis
        amount_stats = {
            "count": len(amounts),
            "mean": round(np.mean(amounts), 2),
            "median": round(np.median(amounts), 2),
            "std": round(np.std(amounts), 2),
            "min": round(min(amounts), 2),
            "max": round(max(amounts), 2),
            "p25": round(np.percentile(amounts, 25), 2),
            "p75": round(np.percentile(amounts, 75), 2),
            "p90": round(np.percentile(amounts, 90), 2),
            "p95": round(np.percentile(amounts, 95), 2),
            "p99": round(np.percentile(amounts, 99), 2),
        }

        # Amount histogram (binned distribution)
        bins = [0, 10, 50, 100, 500, 1000, 5000, 9000, 9500, 10000, 50000, 100000, float('inf')]
        bin_labels = ["$0-10", "$10-50", "$50-100", "$100-500", "$500-1K",
                      "$1K-5K", "$5K-9K", "$9K-9.5K", "$9.5K-10K", "$10K-50K", "$50K-100K", "$100K+"]
        hist_counts = [0] * len(bin_labels)
        for a in amounts:
            for j in range(len(bins) - 1):
                if bins[j] <= a < bins[j + 1]:
                    hist_counts[j] += 1
                    break

        amount_histogram = dict(zip(bin_labels, hist_counts))

        # Time distribution
        hours = [int(t["timestamp"][11:13]) for t in transactions]
        hour_distribution = Counter(hours)

        # International vs domestic
        international_count = sum(1 for t in transactions if t["is_international"] == "True")
        international_pct = round(international_count / len(transactions) * 100, 2)

        # Risk flags
        risk_flagged = sum(1 for t in transactions if t["risk_flag"] == "True")
        risk_pct = round(risk_flagged / len(transactions) * 100, 2)

        profile = {
            "total_transactions": len(transactions),
            "amount_statistics": amount_stats,
            "amount_histogram": amount_histogram,
            "currency_distribution": dict(currencies.most_common()),
            "status_distribution": dict(statuses.most_common()),
            "payment_method_distribution": dict(payment_methods.most_common()),
            "transaction_type_distribution": dict(tx_types.most_common()),
            "international_percentage": international_pct,
            "risk_flagged_percentage": risk_pct,
            "hour_distribution": dict(sorted(hour_distribution.items())),
        }

        span.set_attribute("profile_keys", len(profile))

    print(f"  Transaction profile computed ({len(profile)} dimensions)")
    print(f"  Amount range: ${amount_stats['min']:,.2f} — ${amount_stats['max']:,.2f}")
    print(f"  Median transaction: ${amount_stats['median']:,.2f}")
    print(f"  International: {international_pct}%  |  Risk-flagged: {risk_pct}%")
    print()

    return profile


# ============================================================================
# STEP 3: Profile Customer and Merchant Data
# ============================================================================

def step_3_profile_entities() -> dict:
    """Load and profile customer and merchant datasets."""
    print("─" * 70)
    print("STEP 3: Profiling Customer & Merchant Data")
    print("─" * 70)

    # --- Customer profiling ---
    with create_span("load_customers") as span:
        customers = []
        with open(DATA_DIR / "customers.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                customers.append(row)
        span.set_attribute("customer_count", len(customers))

    # Analyze customer demographics
    countries = Counter(c["country"] for c in customers)
    account_tiers = Counter(c["account_tier"] for c in customers)
    account_statuses = Counter(c["account_status"] for c in customers)
    email_domains = Counter(c["email"].split("@")[1] for c in customers)

    # Registration date distribution (by month)
    reg_months = Counter(c["registration_date"][:7] for c in customers)

    # Identity verification rate
    verified_count = sum(1 for c in customers if c["verified_identity"] == "True")
    verified_pct = round(verified_count / len(customers) * 100, 2)

    customer_profile = {
        "total_customers": len(customers),
        "country_distribution": dict(countries.most_common(10)),
        "account_tier_distribution": dict(account_tiers),
        "account_status_distribution": dict(account_statuses),
        "email_domain_distribution": dict(email_domains.most_common(15)),
        "registration_month_distribution": dict(sorted(reg_months.items())),
        "identity_verified_percentage": verified_pct,
    }

    print(f"  Customers: {len(customers):,}")
    print(f"  Top countries: {', '.join(f'{k}({v})' for k, v in countries.most_common(5))}")
    print(f"  Verified: {verified_pct}%")

    # --- Merchant profiling ---
    with create_span("load_merchants") as span:
        merchants = []
        with open(DATA_DIR / "merchants.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                merchants.append(row)
        span.set_attribute("merchant_count", len(merchants))

    mcc_categories = Counter(m["mcc_description"] for m in merchants)
    merchant_countries = Counter(m["country"] for m in merchants)
    risk_tiers = Counter(m["industry_risk_tier"] for m in merchants)

    # Registration clustering analysis
    merchant_reg_dates = [m["registration_date"] for m in merchants]
    merchant_reg_months = Counter(d[:7] for d in merchant_reg_dates)

    merchant_profile = {
        "total_merchants": len(merchants),
        "mcc_category_distribution": dict(mcc_categories.most_common(10)),
        "country_distribution": dict(merchant_countries.most_common(10)),
        "risk_tier_distribution": dict(risk_tiers),
        "registration_month_distribution": dict(sorted(merchant_reg_months.items())),
    }

    print(f"  Merchants: {len(merchants):,}")
    print(f"  MCC categories: {len(mcc_categories)}")
    print(f"  Risk tiers: {dict(risk_tiers)}")
    print()

    return {
        "customer_profile": customer_profile,
        "merchant_profile": merchant_profile,
    }


# ============================================================================
# STEP 4: LLM-Powered Data Exploration
# ============================================================================
# Here's where Azure OpenAI enters the picture. We send our pre-computed
# data profiles to the LLM and ask it to identify patterns, anomalies,
# and areas that warrant deeper investigation.
#
# WORKSHOP DISCUSSION POINT:
# "Notice how we're using temperature=0.1 — why not 0.0? A tiny amount of
# temperature prevents the model from getting stuck in repetitive patterns
# while still being highly deterministic. This is a production best practice."
#
# OBSERVE THE TELEMETRY:
# Every call_openai() invocation automatically records:
# - Token usage (prompt + completion)
# - Latency (time to first token, total time)
# - The module name (for cost attribution)
# These metrics feed directly into Module 06's dashboard.
# ============================================================================

def step_4_llm_exploration(transaction_profile: dict, entity_profiles: dict) -> dict:
    """Use Azure OpenAI to analyze the data profiles and identify anomalies."""
    print("─" * 70)
    print("STEP 4: LLM-Powered Data Exploration")
    print("─" * 70)
    print()

    results = {}

    # ----------------------------------------------------------------
    # Analysis 1: Transaction Pattern Overview
    # ----------------------------------------------------------------
    # We send the transaction profile to the LLM and ask for a
    # high-level assessment. The system prompt establishes the
    # "senior payment analyst" persona.
    # ----------------------------------------------------------------
    print("  [4a] Requesting transaction pattern analysis...")

    with create_span("llm_transaction_analysis", {"analysis_type": "transaction_overview"}) as span:
        response = call_openai(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["data_explorer"]},
                {"role": "user", "content": (
                    "I'm analyzing a payment processing dataset. Here is the statistical "
                    "profile of all transactions. Please provide:\n"
                    "1. An overall assessment of the transaction landscape\n"
                    "2. Any unusual patterns in the amount distribution\n"
                    "3. Observations about the currency and payment method mix\n"
                    "4. Any red flags or areas that warrant deeper investigation\n\n"
                    f"Transaction Profile:\n{json.dumps(transaction_profile, indent=2)}"
                )},
            ],
            module_name=MODULE_NAME,
        )

        span.set_attribute("tokens_used", response["usage"]["total_tokens"])
        span.set_attribute("latency_ms", response["latency_ms"])
        span.set_attribute("ttft_ms", response["ttft_ms"])
        record_metric("api_latency_ms", response["latency_ms"], "ms", {"step": "4a"})
        record_metric("ttft_ms", response["ttft_ms"], "ms", {"step": "4a"})
        record_metric("tokens_used", response["usage"]["total_tokens"], "tokens", {"step": "4a"})

    results["transaction_analysis"] = response["content"]
    print()

    # ----------------------------------------------------------------
    # Analysis 2: Customer & Merchant Demographics
    # ----------------------------------------------------------------
    # Now we send entity profiles. The LLM should notice:
    # - Unusual email domain concentrations
    # - Registration date clustering
    # - Geographic anomalies
    # These are early signals that something might be off.
    # ----------------------------------------------------------------
    print("  [4b] Requesting entity analysis...")

    with create_span("llm_entity_analysis", {"analysis_type": "entity_overview"}) as span:
        response = call_openai(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["data_explorer"]},
                {"role": "user", "content": (
                    "Here are the customer and merchant profiles for a payment platform. "
                    "Please analyze:\n"
                    "1. Customer demographic patterns — anything unusual?\n"
                    "2. Email domain distribution — any anomalous concentrations?\n"
                    "3. Registration timing patterns — any suspicious clustering?\n"
                    "4. Merchant category and risk tier distribution\n"
                    "5. Any signals that suggest synthetic identities or shell companies?\n\n"
                    f"Entity Profiles:\n{json.dumps(entity_profiles, indent=2)}"
                )},
            ],
            module_name=MODULE_NAME,
        )

        span.set_attribute("tokens_used", response["usage"]["total_tokens"])
        record_metric("api_latency_ms", response["latency_ms"], "ms", {"step": "4b"})
        record_metric("ttft_ms", response["ttft_ms"], "ms", {"step": "4b"})
        record_metric("tokens_used", response["usage"]["total_tokens"], "tokens", {"step": "4b"})

    results["entity_analysis"] = response["content"]
    print()

    # ----------------------------------------------------------------
    # Analysis 3: Initial Anomaly Identification
    # ----------------------------------------------------------------
    # We combine both profiles and explicitly ask the LLM to identify
    # the top areas of concern. This primes the investigation for Module 02.
    # ----------------------------------------------------------------
    print("  [4c] Requesting combined anomaly identification...")

    with create_span("llm_anomaly_identification", {"analysis_type": "combined_anomalies"}) as span:
        response = call_openai(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["data_explorer"]},
                {"role": "user", "content": (
                    "Based on the following transaction AND entity profiles, identify the "
                    "top 5 anomalies or areas of concern that should be investigated for "
                    "potential fraud. For each anomaly:\n"
                    "- Describe the specific data point(s) that are unusual\n"
                    "- Explain why it's concerning in a payment processing context\n"
                    "- Suggest what additional data you would need to investigate further\n\n"
                    f"Transaction Profile:\n{json.dumps(transaction_profile, indent=2)}\n\n"
                    f"Entity Profiles:\n{json.dumps(entity_profiles, indent=2)}"
                )},
            ],
            module_name=MODULE_NAME,
        )

        span.set_attribute("tokens_used", response["usage"]["total_tokens"])
        record_metric("api_latency_ms", response["latency_ms"], "ms", {"step": "4c"})
        record_metric("ttft_ms", response["ttft_ms"], "ms", {"step": "4c"})
        record_metric("tokens_used", response["usage"]["total_tokens"], "tokens", {"step": "4c"})

    results["anomaly_identification"] = response["content"]
    print()

    return results


# ============================================================================
# STEP 5: Summary and Transition
# ============================================================================

def step_5_summary(results: dict):
    """Print module summary and token usage report."""
    print("=" * 70)
    print("MODULE 01 — Summary")
    print("=" * 70)

    totals = get_total_tokens()
    print(f"""
  API Calls Made:        {totals['api_calls']}
  Total Tokens Used:     {totals['total_tokens']:,}
    ├─ Prompt Tokens:    {totals['prompt_tokens']:,}
    └─ Completion Tokens:{totals['completion_tokens']:,}

  KEY TAKEAWAY:
  The LLM has identified initial anomalies in the data — but it's working
  from summaries only. In Module 02, we'll feed it actual transaction
  samples and use STRUCTURED OUTPUTS (JSON mode) to get parseable,
  actionable anomaly detection results.

  NEXT: Run 02_pattern_detection.py
""")
    print("=" * 70)


# ============================================================================
# MAIN — Run all steps in sequence
# ============================================================================

def main():
    """Execute Module 01: Data Exploration."""
    step_1_initialize()
    transaction_profile = step_2_profile_transactions()
    entity_profiles = step_3_profile_entities()
    results = step_4_llm_exploration(transaction_profile, entity_profiles)
    step_5_summary(results)


if __name__ == "__main__":
    main()
