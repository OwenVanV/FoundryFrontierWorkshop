"""
============================================================================
MODULE 03 — Cross-Reference Analysis with Multi-Turn Conversations
============================================================================

WORKSHOP NARRATIVE:
    Module 02 identified suspicious patterns within individual datasets.
    Now comes the real investigative work: CROSS-REFERENCING signals
    across multiple data sources.

    A smurfing pattern in transactions is concerning. But when you combine
    it with device fingerprint sharing from payment sessions, impossible
    travel from geolocation data, and shell company indicators from
    merchant profiles — the picture becomes clear.

    In this module, you'll use MULTI-TURN CONVERSATIONS with Azure OpenAI.
    Each turn builds on previous findings, simulating how a human fraud
    analyst would progressively narrow an investigation.

LEARNING OBJECTIVES:
    1. Design multi-turn investigation conversations
    2. Cross-reference signals across CSV, JSON, and time-series data
    3. Use chain-of-thought prompting for complex reasoning
    4. Understand how conversation context accumulates (and costs)
    5. Track cumulative token usage across conversation turns

AZURE SERVICES USED:
    - Azure OpenAI (multi-turn chat completions)
    - OpenTelemetry (conversation turn tracking, cumulative cost)

KEY TECHNIQUE: CHAIN-OF-THOUGHT
    We explicitly ask the LLM to "think step by step" and show its
    reasoning. This dramatically improves accuracy for complex
    cross-referencing tasks where the answer isn't immediately obvious.

ESTIMATED TIME: 30-35 minutes

============================================================================
"""

import os
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

# ============================================================================
# PATH SETUP
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.azure_client import call_openai, call_openai_json, SYSTEM_PROMPTS, get_total_tokens
from utils.telemetry import init_telemetry, create_span, record_metric, estimate_cost

DATA_DIR = PROJECT_ROOT / "data"
MODULE_NAME = "03_cross_reference_analysis"


# ============================================================================
# STEP 1: Load All Data Sources & Module 02 Findings
# ============================================================================
# We load EVERY dataset here — transactions, customers, merchants,
# payment sessions, risk signals, and the findings from Module 02.
# This is the first time we're working with the full data landscape.
# ============================================================================

def step_1_load_all_data() -> dict:
    """Load all datasets and Module 02 findings."""
    print("=" * 70)
    print("MODULE 03 — Cross-Reference Analysis")
    print("=" * 70)
    print()

    init_telemetry(service_name=MODULE_NAME)

    data = {}

    with create_span("load_all_datasets") as span:
        # Load transactions
        transactions = []
        with open(DATA_DIR / "transactions.csv", "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["amount"] = float(row["amount"])
                transactions.append(row)
        data["transactions"] = transactions

        # Load customers
        customers = {}
        with open(DATA_DIR / "customers.csv", "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                customers[row["customer_id"]] = row
        data["customers"] = customers

        # Load merchants
        merchants = {}
        with open(DATA_DIR / "merchants.csv", "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                merchants[row["merchant_id"]] = row
        data["merchants"] = merchants

        # Load payment sessions (JSON — nested structure)
        with open(DATA_DIR / "payment_sessions.json", "r", encoding="utf-8") as f:
            data["sessions"] = json.load(f)

        # Load risk signals
        with open(DATA_DIR / "risk_signals.json", "r", encoding="utf-8") as f:
            data["risk_signals"] = json.load(f)

        # Load dispute cases
        with open(DATA_DIR / "dispute_cases.json", "r", encoding="utf-8") as f:
            data["disputes"] = json.load(f)

        # Load Module 02 findings if available
        findings_path = DATA_DIR / "module_02_findings.json"
        if findings_path.exists():
            with open(findings_path, "r", encoding="utf-8") as f:
                data["module_02_findings"] = json.load(f)
            print("  ✓ Module 02 findings loaded")
        else:
            data["module_02_findings"] = {
                "suspicious_accounts": [],
                "suspicious_merchants": [],
            }
            print("  ⚠ Module 02 findings not found — using empty set")
            print("    Run 02_pattern_detection.py first for best results")

        span.set_attribute("datasets_loaded", 7)

    print(f"  Transactions: {len(data['transactions']):,}")
    print(f"  Customers: {len(data['customers']):,}")
    print(f"  Merchants: {len(data['merchants']):,}")
    print(f"  Sessions: {len(data['sessions']):,}")
    print(f"  Risk signals: {len(data['risk_signals']):,}")
    print(f"  Disputes: {len(data['disputes']):,}")
    print()

    return data


# ============================================================================
# STEP 2: Device Fingerprint Cross-Reference
# ============================================================================
# This is where we join payment session data (JSON) with customer
# profiles (CSV). We're looking for accounts that share device
# fingerprints — a strong signal of coordinated activity.
#
# WORKSHOP NOTE:
# "Device fingerprinting is a browser/app-based identification technique.
# Each device generates a near-unique hash from its hardware, software,
# and configuration. When two 'different' accounts use the same device
# fingerprint, it strongly suggests they're controlled by the same person."
# ============================================================================

def step_2_device_fingerprint_analysis(data: dict) -> dict:
    """Cross-reference device fingerprints across payment sessions."""
    print("─" * 70)
    print("STEP 2: Device Fingerprint Cross-Reference")
    print("─" * 70)
    print()

    sessions = data["sessions"]

    # Build a map: device_fingerprint → list of customer_ids
    with create_span("build_device_map") as span:
        device_to_customers = defaultdict(set)
        device_to_sessions = defaultdict(list)

        for session in sessions:
            fp = session.get("device", {}).get("fingerprint")
            cid = session.get("customer_id")
            if fp and cid:
                device_to_customers[fp].add(cid)
                device_to_sessions[fp].append({
                    "session_id": session["session_id"],
                    "customer_id": cid,
                    "timestamp": session["timestamp"],
                    "city": session.get("geolocation", {}).get("city"),
                    "country": session.get("geolocation", {}).get("country"),
                })

        # Find devices shared by multiple accounts
        shared_devices = {
            fp: {
                "customers": sorted(customers),
                "session_count": len(device_to_sessions[fp]),
                "sessions": device_to_sessions[fp][:10],  # Sample sessions
            }
            for fp, customers in device_to_customers.items()
            if len(customers) >= 2
        }

        span.set_attribute("total_devices", len(device_to_customers))
        span.set_attribute("shared_devices", len(shared_devices))

    print(f"  Total unique devices: {len(device_to_customers):,}")
    print(f"  Devices shared by 2+ accounts: {len(shared_devices)}")
    print()

    if not shared_devices:
        print("  No shared devices found. Skipping LLM analysis.")
        return {}

    # Send shared device data to LLM for analysis
    print(f"  Sending {len(shared_devices)} shared devices to LLM...")

    # Enrich with customer profile data
    enriched_devices = {}
    for fp, info in list(shared_devices.items())[:20]:  # Top 20
        enriched_customers = []
        for cid in info["customers"]:
            customer = data["customers"].get(cid, {})
            enriched_customers.append({
                "customer_id": cid,
                "name": f"{customer.get('first_name', '?')} {customer.get('last_name', '?')}",
                "email": customer.get("email", "?"),
                "city": customer.get("city", "?"),
                "registration_date": customer.get("registration_date", "?"),
            })
        enriched_devices[fp[:16] + "..."] = {
            "customers": enriched_customers,
            "session_count": info["session_count"],
            "sample_sessions": info["sessions"][:5],
        }

    with create_span("llm_device_analysis") as span:
        response = call_openai_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["cross_reference_analyst"]},
                {"role": "user", "content": (
                    "I've identified device fingerprints shared by multiple customer accounts. "
                    "This is a strong signal of account coordination or fraud.\n\n"
                    "For each shared device, I've included the customer profiles and sample sessions. "
                    "Analyze the following and output JSON:\n"
                    "{\n"
                    '  "device_sharing_analysis": [\n'
                    "    {\n"
                    '      "device_fingerprint": "...",\n'
                    '      "shared_accounts": ["CUS-..."],\n'
                    '      "risk_assessment": "low|medium|high|critical",\n'
                    '      "reasoning": "why this is or isn\'t suspicious",\n'
                    '      "cross_reference_findings": {\n'
                    '        "email_domain_match": true/false,\n'
                    '        "registration_date_proximity": "X days apart",\n'
                    '        "geographic_consistency": "same city / different cities",\n'
                    '        "name_similarity": "assessment"\n'
                    "      }\n"
                    "    }\n"
                    "  ],\n"
                    '  "high_risk_accounts": ["CUS-..."],\n'
                    '  "summary": "overall device sharing assessment"\n'
                    "}\n\n"
                    f"Shared Device Data:\n{json.dumps(enriched_devices, indent=2)}"
                )},
            ],
            module_name=MODULE_NAME,
        )

        span.set_attribute("tokens_used", response["usage"]["total_tokens"])
        record_metric("tokens_used", response["usage"]["total_tokens"], "tokens", {"step": "2"})
        record_metric("ttft_ms", response["ttft_ms"], "ms", {"step": "2"})

    result = response["parsed"]
    high_risk = result.get("high_risk_accounts", [])
    print(f"  ✓ Analysis complete: {len(high_risk)} high-risk accounts identified")
    print()

    return result


# ============================================================================
# STEP 3: Impossible Travel Detection
# ============================================================================
# Cross-reference payment session timestamps with geolocation data.
# If the same customer has sessions from Lagos and London within
# 20 minutes, that's physically impossible — either the account is
# compromised or it's being used by multiple people.
# ============================================================================

def step_3_impossible_travel(data: dict) -> dict:
    """Detect impossible travel patterns from session geolocation data."""
    print("─" * 70)
    print("STEP 3: Impossible Travel Detection")
    print("─" * 70)
    print()

    sessions = data["sessions"]

    # Group sessions by customer, sorted by time
    with create_span("detect_impossible_travel") as span:
        customer_sessions = defaultdict(list)
        for session in sessions:
            customer_sessions[session["customer_id"]].append(session)

        # Sort each customer's sessions by timestamp
        for cid in customer_sessions:
            customer_sessions[cid].sort(key=lambda s: s["timestamp"])

        # Check consecutive sessions for impossible travel
        # Rule: different cities + < 60 minutes apart + > 500km distance
        travel_anomalies = []

        for cid, cust_sessions in customer_sessions.items():
            for i in range(len(cust_sessions) - 1):
                s1 = cust_sessions[i]
                s2 = cust_sessions[i + 1]

                city1 = s1.get("geolocation", {}).get("city", "")
                city2 = s2.get("geolocation", {}).get("city", "")

                if city1 and city2 and city1 != city2:
                    t1 = datetime.strptime(s1["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
                    t2 = datetime.strptime(s2["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
                    gap_minutes = (t2 - t1).total_seconds() / 60

                    if 0 < gap_minutes < 60:  # Less than 1 hour between different cities
                        lat1 = s1.get("geolocation", {}).get("latitude", 0)
                        lon1 = s1.get("geolocation", {}).get("longitude", 0)
                        lat2 = s2.get("geolocation", {}).get("latitude", 0)
                        lon2 = s2.get("geolocation", {}).get("longitude", 0)

                        # Rough distance calculation (Haversine simplified)
                        lat_diff = abs(lat1 - lat2)
                        lon_diff = abs(lon1 - lon2)
                        rough_distance_km = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 111

                        if rough_distance_km > 200:  # > 200km in < 60 min = impossible
                            customer = data["customers"].get(cid, {})
                            travel_anomalies.append({
                                "customer_id": cid,
                                "customer_name": f"{customer.get('first_name', '?')} {customer.get('last_name', '?')}",
                                "customer_email": customer.get("email", "?"),
                                "session_1": {
                                    "session_id": s1["session_id"],
                                    "timestamp": s1["timestamp"],
                                    "city": city1,
                                    "country": s1.get("geolocation", {}).get("country", "?"),
                                },
                                "session_2": {
                                    "session_id": s2["session_id"],
                                    "timestamp": s2["timestamp"],
                                    "city": city2,
                                    "country": s2.get("geolocation", {}).get("country", "?"),
                                },
                                "gap_minutes": round(gap_minutes, 1),
                                "estimated_distance_km": round(rough_distance_km, 0),
                            })

        span.set_attribute("travel_anomalies", len(travel_anomalies))

    print(f"  Found {len(travel_anomalies)} impossible travel instances")

    if not travel_anomalies:
        print("  No impossible travel detected. Moving on.")
        return {}

    # Send to LLM for contextual analysis
    print(f"  Sending {min(len(travel_anomalies), 20)} anomalies to LLM...")
    print()

    with create_span("llm_travel_analysis") as span:
        response = call_openai_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["cross_reference_analyst"]},
                {"role": "user", "content": (
                    "I've detected instances where customers had sessions in different "
                    "cities within a timeframe that makes physical travel impossible. "
                    "This could indicate:\n"
                    "- Account compromise (credentials shared/stolen)\n"
                    "- Coordinated fraud (same person, multiple accounts)\n"
                    "- VPN/proxy usage that wasn't detected\n\n"
                    "Analyze these cases and output JSON:\n"
                    "{\n"
                    '  "travel_anomalies": [\n'
                    "    {\n"
                    '      "customer_id": "CUS-...",\n'
                    '      "cities": ["City1", "City2"],\n'
                    '      "gap_minutes": N,\n'
                    '      "risk_level": "low|medium|high|critical",\n'
                    '      "likely_explanation": "...",\n'
                    '      "connection_to_other_signals": "..."\n'
                    "    }\n"
                    "  ],\n"
                    '  "accounts_to_flag": ["CUS-..."],\n'
                    '  "summary": "..."\n'
                    "}\n\n"
                    f"Impossible Travel Data:\n{json.dumps(travel_anomalies[:20], indent=2)}"
                )},
            ],
            module_name=MODULE_NAME,
        )

        span.set_attribute("tokens_used", response["usage"]["total_tokens"])
        record_metric("tokens_used", response["usage"]["total_tokens"], "tokens", {"step": "3"})
        record_metric("ttft_ms", response["ttft_ms"], "ms", {"step": "3"})

    result = response["parsed"]
    flagged = result.get("accounts_to_flag", [])
    print(f"  ✓ Analysis complete: {len(flagged)} accounts flagged")
    print()

    return result


# ============================================================================
# STEP 4: Multi-Turn Investigation Conversation
# ============================================================================
# This is the culmination of Module 03: a multi-turn conversation where
# each turn builds on the previous one. The LLM acts as an investigation
# partner, progressively connecting dots across datasets.
#
# WORKSHOP KEY CONCEPT:
# "Notice how conversation context accumulates — each turn includes ALL
# previous messages. This means token usage grows with each turn.
# In production, you'd use summarization or context windowing to manage
# costs. For investigation, the accumulation is valuable because the
# model needs the full picture."
# ============================================================================

def step_4_multi_turn_investigation(
    data: dict,
    device_results: dict,
    travel_results: dict,
) -> dict:
    """Run a multi-turn investigation conversation combining all signals."""
    print("─" * 70)
    print("STEP 4: Multi-Turn Investigation Conversation")
    print("─" * 70)
    print()
    print("  Starting a multi-turn investigation. Each turn builds on the last,")
    print("  progressively connecting signals across datasets.")
    print()

    # Build the investigation conversation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS["cross_reference_analyst"]},
    ]

    # ----------------------------------------------------------------
    # TURN 1: Present Module 02 findings
    # ----------------------------------------------------------------
    print("  ── Turn 1: Presenting initial findings ──")

    m02_findings = data.get("module_02_findings", {})
    suspicious_accounts = m02_findings.get("suspicious_accounts", [])
    suspicious_merchants = m02_findings.get("suspicious_merchants", [])

    messages.append({
        "role": "user",
        "content": (
            "I'm conducting a fraud investigation on a payment platform. "
            "Module 02 pattern detection identified the following:\n\n"
            f"Suspicious accounts: {json.dumps(suspicious_accounts[:20])}\n"
            f"Suspicious merchants: {json.dumps(suspicious_merchants[:10])}\n\n"
            "Think step by step. What patterns do you see in these IDs? "
            "What would you investigate next? What data would you cross-reference?"
        ),
    })

    with create_span("investigation_turn_1") as span:
        response = call_openai(
            messages=messages,
            module_name=MODULE_NAME,
        )
        span.set_attribute("turn", 1)
        span.set_attribute("tokens_used", response["usage"]["total_tokens"])
        record_metric("conversation_turn_tokens", response["usage"]["total_tokens"], "tokens", {"turn": 1})
        record_metric("ttft_ms", response["ttft_ms"], "ms", {"step": "4_turn_1"})

    messages.append({"role": "assistant", "content": response["content"]})
    print(f"  ✓ Turn 1 complete ({response['usage']['total_tokens']} tokens)")
    print()

    # ----------------------------------------------------------------
    # TURN 2: Add device fingerprint findings
    # ----------------------------------------------------------------
    print("  ── Turn 2: Adding device fingerprint evidence ──")

    device_summary = {
        "high_risk_accounts": device_results.get("high_risk_accounts", []),
        "shared_device_count": len(device_results.get("device_sharing_analysis", [])),
        "analysis": device_results.get("device_sharing_analysis", [])[:5],
    }

    messages.append({
        "role": "user",
        "content": (
            "I've now cross-referenced payment session device fingerprints. "
            "Here's what I found:\n\n"
            f"{json.dumps(device_summary, indent=2)}\n\n"
            "Think step by step. How do these device sharing patterns connect "
            "to the transaction anomalies from the previous turn? "
            "Are any of the device-sharing accounts also in the suspicious "
            "transactions list? What does this overlap tell us?"
        ),
    })

    with create_span("investigation_turn_2") as span:
        response = call_openai(
            messages=messages,
            module_name=MODULE_NAME,
        )
        span.set_attribute("turn", 2)
        span.set_attribute("tokens_used", response["usage"]["total_tokens"])
        record_metric("conversation_turn_tokens", response["usage"]["total_tokens"], "tokens", {"turn": 2})
        record_metric("ttft_ms", response["ttft_ms"], "ms", {"step": "4_turn_2"})

    messages.append({"role": "assistant", "content": response["content"]})
    print(f"  ✓ Turn 2 complete ({response['usage']['total_tokens']} tokens)")
    print()

    # ----------------------------------------------------------------
    # TURN 3: Add impossible travel findings
    # ----------------------------------------------------------------
    print("  ── Turn 3: Adding geolocation/travel evidence ──")

    travel_summary = {
        "flagged_accounts": travel_results.get("accounts_to_flag", []),
        "anomaly_count": len(travel_results.get("travel_anomalies", [])),
        "details": travel_results.get("travel_anomalies", [])[:5],
    }

    messages.append({
        "role": "user",
        "content": (
            "Additional evidence: impossible travel detection from geolocation data.\n\n"
            f"{json.dumps(travel_summary, indent=2)}\n\n"
            "Think step by step. We now have three signal types:\n"
            "1. Transaction patterns (smurfing, temporal coordination)\n"
            "2. Device fingerprint sharing\n"
            "3. Impossible travel\n\n"
            "Which accounts appear in MULTIPLE signal types? "
            "These multi-signal accounts are the strongest fraud candidates. "
            "Map out the connections you see."
        ),
    })

    with create_span("investigation_turn_3") as span:
        response = call_openai(
            messages=messages,
            module_name=MODULE_NAME,
        )
        span.set_attribute("turn", 3)
        span.set_attribute("tokens_used", response["usage"]["total_tokens"])
        record_metric("conversation_turn_tokens", response["usage"]["total_tokens"], "tokens", {"turn": 3})
        record_metric("ttft_ms", response["ttft_ms"], "ms", {"step": "4_turn_3"})

    messages.append({"role": "assistant", "content": response["content"]})
    print(f"  ✓ Turn 3 complete ({response['usage']['total_tokens']} tokens)")
    print()

    # ----------------------------------------------------------------
    # TURN 4: Ask for structured hypothesis
    # ----------------------------------------------------------------
    print("  ── Turn 4: Requesting structured hypothesis ──")

    # Pull in customer profile data for the suspicious accounts
    all_flagged = set()
    all_flagged.update(suspicious_accounts[:20])
    all_flagged.update(device_results.get("high_risk_accounts", []))
    all_flagged.update(travel_results.get("accounts_to_flag", []))

    customer_profiles = {}
    for cid in list(all_flagged)[:25]:
        customer = data["customers"].get(cid, {})
        if customer:
            customer_profiles[cid] = {
                "name": f"{customer.get('first_name', '?')} {customer.get('last_name', '?')}",
                "email": customer.get("email", "?"),
                "city": customer.get("city", "?"),
                "state": customer.get("state", "?"),
                "registration_date": customer.get("registration_date", "?"),
                "account_tier": customer.get("account_tier", "?"),
                "risk_score": customer.get("risk_score", "?"),
            }

    messages.append({
        "role": "user",
        "content": (
            "Here are the profiles of the flagged accounts:\n\n"
            f"{json.dumps(customer_profiles, indent=2)}\n\n"
            "Now synthesize ALL the evidence from this conversation into a "
            "structured hypothesis. Think step by step:\n"
            "1. Which accounts are part of a coordinated fraud ring?\n"
            "2. What role does each account play?\n"
            "3. Which merchants are shell companies?\n"
            "4. What is the estimated financial exposure?\n"
            "5. What is your confidence level and what would increase it?\n\n"
            "Be specific with account IDs and merchant IDs."
        ),
    })

    with create_span("investigation_turn_4") as span:
        response = call_openai(
            messages=messages,
            module_name=MODULE_NAME,
        )
        span.set_attribute("turn", 4)
        span.set_attribute("tokens_used", response["usage"]["total_tokens"])
        record_metric("conversation_turn_tokens", response["usage"]["total_tokens"], "tokens", {"turn": 4})
        record_metric("ttft_ms", response["ttft_ms"], "ms", {"step": "4_turn_4"})

    messages.append({"role": "assistant", "content": response["content"]})
    print(f"  ✓ Turn 4 complete ({response['usage']['total_tokens']} tokens)")
    print()

    return {
        "conversation_turns": 4,
        "hypothesis": response["content"],
        "all_flagged_accounts": sorted(all_flagged),
        "total_messages": len(messages),
    }


# ============================================================================
# STEP 5: Summary and Export
# ============================================================================

def step_5_summary(
    device_results: dict,
    travel_results: dict,
    investigation_results: dict,
):
    """Print module summary with token usage analysis."""
    print("=" * 70)
    print("MODULE 03 — Summary")
    print("=" * 70)

    totals = get_total_tokens()

    # Export findings for Module 04
    findings = {
        "module": "03_cross_reference_analysis",
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device_sharing_results": device_results,
        "impossible_travel_results": travel_results,
        "investigation_hypothesis": investigation_results.get("hypothesis", ""),
        "all_flagged_accounts": investigation_results.get("all_flagged_accounts", []),
        "conversation_turns": investigation_results.get("conversation_turns", 0),
        "token_usage": totals,
    }

    output_path = DATA_DIR / "module_03_findings.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, default=str)

    print(f"""
  Investigation Summary:
    ├─ Conversation Turns:     {investigation_results.get('conversation_turns', 0)}
    ├─ Device-sharing signals: {len(device_results.get('device_sharing_analysis', []))}
    ├─ Travel anomalies:       {len(travel_results.get('travel_anomalies', []))}
    ├─ Total flagged accounts: {len(investigation_results.get('all_flagged_accounts', []))}

  API Usage (cumulative across all turns):
    ├─ API Calls:              {totals['api_calls']}
    ├─ Total Tokens:           {totals['total_tokens']:,}
    ├─ Prompt Tokens:          {totals['prompt_tokens']:,}
    └─ Completion Tokens:      {totals['completion_tokens']:,}

  OBSERVE: Token usage GROWS with each conversation turn because the
  full conversation history is sent every time. This is the cost of
  context accumulation — essential for investigation, expensive at scale.

  KEY TAKEAWAY:
  Multi-turn conversations let the LLM build a progressively deeper
  understanding of the fraud landscape. By combining transaction patterns,
  device fingerprints, and geolocation data, we've identified a network
  of suspicious accounts far more accurately than any single signal could.

  NEXT: Run 04_fraud_ring_investigation.py (full ring mapping)
""")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Execute Module 03: Cross-Reference Analysis."""
    data = step_1_load_all_data()
    device_results = step_2_device_fingerprint_analysis(data)
    travel_results = step_3_impossible_travel(data)
    investigation_results = step_4_multi_turn_investigation(data, device_results, travel_results)
    step_5_summary(device_results, travel_results, investigation_results)


if __name__ == "__main__":
    main()
