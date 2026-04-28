"""
============================================================================
MODULE 04 — Fraud Ring Investigation & Report Generation
============================================================================

WORKSHOP NARRATIVE:
    This is the climax of the investigation. Modules 01-03 identified
    individual signals and cross-referenced them. Now we ask the LLM to
    do what it does best: SYNTHESIZE everything into a comprehensive
    fraud ring map and generate a formal investigation report.

    This module showcases Azure AI Foundry's model deployment capabilities
    and demonstrates how LLMs can generate investigation-grade reports
    that would take a human analyst days to compile.

LEARNING OBJECTIVES:
    1. Use Azure AI Foundry model deployments
    2. Generate comprehensive investigation reports with LLMs
    3. Map fraud networks (nodes = accounts, edges = transactions)
    4. Understand responsible AI considerations in fraud detection
    5. Content safety filtering and its role in production AI

AZURE SERVICES USED:
    - Azure AI Foundry (model deployment)
    - Azure OpenAI (report generation)
    - OpenTelemetry (content safety logging, responsible AI)

KEY TECHNIQUE: NETWORK GRAPH REASONING
    We present the LLM with transaction flows as a directed graph
    (sender → receiver) and ask it to identify clusters, cycles,
    and hub nodes. LLMs are surprisingly good at graph reasoning
    when the data is presented clearly.

ESTIMATED TIME: 25-30 minutes

============================================================================
"""

import os
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np

# ============================================================================
# PATH SETUP
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.azure_client import (
    call_openai, call_openai_json, SYSTEM_PROMPTS,
    get_total_tokens, get_token_usage_log,
)
from utils.telemetry import (
    init_telemetry, create_span, record_metric,
    estimate_cost, export_telemetry_to_file,
)

DATA_DIR = PROJECT_ROOT / "data"
MODULE_NAME = "04_fraud_ring_investigation"


# ============================================================================
# STEP 1: Load All Evidence
# ============================================================================

def step_1_load_evidence() -> dict:
    """Load all datasets and previous module findings."""
    print("=" * 70)
    print("MODULE 04 — Fraud Ring Investigation")
    print("=" * 70)
    print()

    init_telemetry(service_name=MODULE_NAME)

    data = {}

    with create_span("load_evidence") as span:
        # Core datasets
        transactions = []
        with open(DATA_DIR / "transactions.csv", "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["amount"] = float(row["amount"])
                transactions.append(row)
        data["transactions"] = transactions

        customers = {}
        with open(DATA_DIR / "customers.csv", "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                customers[row["customer_id"]] = row
        data["customers"] = customers

        merchants = {}
        with open(DATA_DIR / "merchants.csv", "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                merchants[row["merchant_id"]] = row
        data["merchants"] = merchants

        with open(DATA_DIR / "payment_sessions.json", "r", encoding="utf-8") as f:
            data["sessions"] = json.load(f)

        with open(DATA_DIR / "dispute_cases.json", "r", encoding="utf-8") as f:
            data["disputes"] = json.load(f)

        with open(DATA_DIR / "risk_signals.json", "r", encoding="utf-8") as f:
            data["risk_signals"] = json.load(f)

        # Previous module findings
        for module_file in ["module_02_findings.json", "module_03_findings.json"]:
            path = DATA_DIR / module_file
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data[module_file.replace(".json", "")] = json.load(f)
                print(f"  ✓ {module_file} loaded")
            else:
                print(f"  ⚠ {module_file} not found — running with limited prior context")

        span.set_attribute("datasets_loaded", len(data))

    print(f"  Evidence loaded: {len(data)} datasets")
    print()
    return data


# ============================================================================
# STEP 2: Build Transaction Network Graph
# ============================================================================
# We construct a directed graph from transactions:
#   - Nodes = customer accounts
#   - Edges = transactions (with amount, timestamp, merchant)
#
# This graph representation is what we'll feed to the LLM for
# network analysis. The fraud ring should appear as a tightly
# connected subgraph with circular flows.
#
# WORKSHOP NOTE:
# "Graph databases (Neo4j, Neptune) are purpose-built for this kind
# of analysis. Here we're showing that an LLM can reason about graph
# structures from text representations — useful for prototyping and
# investigation before committing to a graph DB deployment."
# ============================================================================

def step_2_build_network(data: dict) -> dict:
    """Build a transaction network graph and identify dense subgraphs."""
    print("─" * 70)
    print("STEP 2: Building Transaction Network Graph")
    print("─" * 70)
    print()

    transactions = data["transactions"]

    with create_span("build_network") as span:
        # Build adjacency representation
        edges = defaultdict(lambda: {"count": 0, "total_amount": 0, "amounts": [], "merchants": set()})
        node_out_degree = Counter()
        node_in_degree = Counter()

        for t in transactions:
            sender = t["sender_id"]
            receiver = t["receiver_id"]
            key = (sender, receiver)

            edges[key]["count"] += 1
            edges[key]["total_amount"] += t["amount"]
            edges[key]["amounts"].append(t["amount"])
            edges[key]["merchants"].add(t["merchant_id"])
            node_out_degree[sender] += 1
            node_in_degree[receiver] += 1

        # Find nodes with high bidirectional connectivity
        # (A sends to B AND B sends to A — unusual in legitimate commerce)
        bidirectional_pairs = []
        for (s, r), info in edges.items():
            reverse = (r, s)
            if reverse in edges:
                bidirectional_pairs.append({
                    "pair": [s, r],
                    "forward_count": info["count"],
                    "forward_total": round(info["total_amount"], 2),
                    "reverse_count": edges[reverse]["count"],
                    "reverse_total": round(edges[reverse]["total_amount"], 2),
                    "shared_merchants": len(info["merchants"] & edges[reverse]["merchants"]),
                })

        # Deduplicate (A,B) and (B,A)
        seen = set()
        unique_bidirectional = []
        for pair in bidirectional_pairs:
            key = tuple(sorted(pair["pair"]))
            if key not in seen:
                seen.add(key)
                unique_bidirectional.append(pair)

        # Sort by total transaction volume
        unique_bidirectional.sort(
            key=lambda p: p["forward_total"] + p["reverse_total"],
            reverse=True,
        )

        # Find high-connectivity nodes (hub nodes in fraud networks)
        high_connectivity = []
        all_nodes = set(node_out_degree.keys()) | set(node_in_degree.keys())
        for node in all_nodes:
            total_degree = node_out_degree.get(node, 0) + node_in_degree.get(node, 0)
            if total_degree >= 15:  # 15+ connections is notably high
                high_connectivity.append({
                    "node": node,
                    "out_degree": node_out_degree.get(node, 0),
                    "in_degree": node_in_degree.get(node, 0),
                    "total_degree": total_degree,
                })

        high_connectivity.sort(key=lambda n: n["total_degree"], reverse=True)

        span.set_attribute("total_edges", len(edges))
        span.set_attribute("bidirectional_pairs", len(unique_bidirectional))
        span.set_attribute("high_connectivity_nodes", len(high_connectivity))

    print(f"  Total unique edges: {len(edges):,}")
    print(f"  Bidirectional pairs: {len(unique_bidirectional):,}")
    print(f"  High-connectivity nodes (15+): {len(high_connectivity)}")
    print()

    return {
        "bidirectional_pairs": unique_bidirectional[:30],
        "high_connectivity_nodes": high_connectivity[:30],
        "total_edges": len(edges),
        "total_nodes": len(all_nodes),
    }


# ============================================================================
# STEP 3: LLM Network Analysis
# ============================================================================

def step_3_network_analysis(data: dict, network: dict) -> dict:
    """Use the LLM to analyze the transaction network for fraud rings."""
    print("─" * 70)
    print("STEP 3: LLM Network Analysis")
    print("─" * 70)
    print()

    # Gather all prior findings
    prior_findings = {}
    m02 = data.get("module_02_findings", {})
    m03 = data.get("module_03_findings", {})

    if m02:
        prior_findings["module_02_suspicious_accounts"] = m02.get("suspicious_accounts", [])[:20]
        prior_findings["module_02_suspicious_merchants"] = m02.get("suspicious_merchants", [])[:10]

    if m03:
        prior_findings["module_03_flagged_accounts"] = m03.get("all_flagged_accounts", [])[:20]
        prior_findings["device_sharing"] = m03.get("device_sharing_results", {}).get("high_risk_accounts", [])
        prior_findings["impossible_travel"] = m03.get("impossible_travel_results", {}).get("accounts_to_flag", [])

    # Enrich network data with customer profiles
    all_network_accounts = set()
    for pair in network["bidirectional_pairs"]:
        all_network_accounts.update(pair["pair"])
    for node in network["high_connectivity_nodes"]:
        all_network_accounts.add(node["node"])

    account_profiles = {}
    for cid in list(all_network_accounts)[:40]:
        customer = data["customers"].get(cid, {})
        if customer:
            account_profiles[cid] = {
                "name": f"{customer.get('first_name', '?')} {customer.get('last_name', '?')}",
                "email": customer.get("email", "?"),
                "city": customer.get("city", "?"),
                "state": customer.get("state", "?"),
                "registration_date": customer.get("registration_date", "?"),
                "risk_score": customer.get("risk_score", "?"),
            }

    # Also include merchant profiles for suspicious merchants
    merchant_profiles = {}
    for mid in prior_findings.get("module_02_suspicious_merchants", [])[:10]:
        merchant = data["merchants"].get(mid, {})
        if merchant:
            merchant_profiles[mid] = {
                "business_name": merchant.get("business_name", "?"),
                "dba_name": merchant.get("dba_name", "?"),
                "mcc_description": merchant.get("mcc_description", "?"),
                "city": merchant.get("city", "?"),
                "state": merchant.get("state", "?"),
                "registration_date": merchant.get("registration_date", "?"),
            }

    print("  Sending network graph + prior findings to LLM...")
    print()

    with create_span("llm_network_analysis") as span:
        response = call_openai_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["fraud_investigator"]},
                {"role": "user", "content": (
                    "You are investigating a suspected fraud ring in a payment platform. "
                    "I'm providing you with:\n\n"
                    "1. TRANSACTION NETWORK: Bidirectional transaction pairs and high-connectivity nodes\n"
                    "2. PRIOR FINDINGS: Suspicious accounts from pattern detection and cross-referencing\n"
                    "3. ACCOUNT PROFILES: Customer details for network participants\n"
                    "4. MERCHANT PROFILES: Details on potentially suspicious merchants\n\n"
                    "Your task: Identify the COMPLETE fraud ring. Output JSON:\n"
                    "{\n"
                    '  "fraud_ring": {\n'
                    '    "identified_members": [\n'
                    "      {\n"
                    '        "customer_id": "CUS-...",\n'
                    '        "role": "ring_leader|money_mule|shell_operator|facilitator",\n'
                    '        "evidence_signals": ["smurfing", "device_sharing", ...],\n'
                    '        "confidence": 0.0 to 1.0,\n'
                    '        "connections": ["CUS-...", ...]\n'
                    "      }\n"
                    "    ],\n"
                    '    "shell_merchants": [\n'
                    "      {\n"
                    '        "merchant_id": "MER-...",\n'
                    '        "business_name": "...",\n'
                    '        "evidence": "why this is a shell company",\n'
                    '        "confidence": 0.0 to 1.0\n'
                    "      }\n"
                    "    ],\n"
                    '    "money_flow_pattern": "description of how money moves through the ring",\n'
                    '    "estimated_total_exposure_usd": 0,\n'
                    '    "detection_difficulty": "why traditional rules missed this",\n'
                    '    "recommended_actions": ["..."],\n'
                    '    "confidence_assessment": {\n'
                    '      "overall_confidence": 0.0 to 1.0,\n'
                    '      "strongest_evidence": "...",\n'
                    '      "weakest_link": "...",\n'
                    '      "additional_evidence_needed": ["..."]\n'
                    "    }\n"
                    "  }\n"
                    "}\n\n"
                    f"TRANSACTION NETWORK:\n{json.dumps(network, indent=2)}\n\n"
                    f"PRIOR FINDINGS:\n{json.dumps(prior_findings, indent=2)}\n\n"
                    f"ACCOUNT PROFILES:\n{json.dumps(account_profiles, indent=2)}\n\n"
                    f"MERCHANT PROFILES:\n{json.dumps(merchant_profiles, indent=2)}"
                )},
            ],
            module_name=MODULE_NAME,
        )

        span.set_attribute("tokens_used", response["usage"]["total_tokens"])
        record_metric("tokens_used", response["usage"]["total_tokens"], "tokens", {"step": "3"})
        record_metric("ttft_ms", response["ttft_ms"], "ms", {"step": "3"})

    result = response["parsed"]
    fraud_ring = result.get("fraud_ring", {})
    members = fraud_ring.get("identified_members", [])
    shells = fraud_ring.get("shell_merchants", [])

    print(f"  ✓ Network analysis complete")
    print(f"    Ring members identified: {len(members)}")
    print(f"    Shell merchants identified: {len(shells)}")
    print()

    return result


# ============================================================================
# STEP 4: Generate Investigation Report
# ============================================================================
# This is where the LLM generates a formal investigation report —
# the kind of document a fraud analyst would present to leadership
# or regulators. It synthesizes ALL evidence into a narrative.
#
# WORKSHOP DISCUSSION POINT:
# "Responsible AI matters here. This report could trigger real actions
# against real people (account freezes, law enforcement referrals).
# How do we ensure the LLM's conclusions are accurate? What human
# oversight is needed? This is why Module 05 (Evaluation) exists."
# ============================================================================

def step_4_generate_report(data: dict, fraud_ring_analysis: dict) -> str:
    """Generate a formal investigation report."""
    print("─" * 70)
    print("STEP 4: Investigation Report Generation")
    print("─" * 70)
    print()

    # Gather transaction statistics for the identified ring members
    fraud_ring = fraud_ring_analysis.get("fraud_ring", {})
    member_ids = [m.get("customer_id") for m in fraud_ring.get("identified_members", [])]
    shell_ids = [s.get("merchant_id") for s in fraud_ring.get("shell_merchants", [])]

    ring_transactions = [
        t for t in data["transactions"]
        if t["sender_id"] in member_ids or t["receiver_id"] in member_ids
    ]

    shell_transactions = [
        t for t in data["transactions"]
        if t["merchant_id"] in shell_ids
    ]

    ring_disputes = [
        d for d in data["disputes"]
        if d.get("merchant_id") in shell_ids or d.get("customer_id") in member_ids
    ]

    transaction_summary = {
        "ring_member_transactions": len(ring_transactions),
        "shell_merchant_transactions": len(shell_transactions),
        "total_ring_volume_usd": round(sum(t["amount"] for t in ring_transactions), 2),
        "total_shell_volume_usd": round(sum(t["amount"] for t in shell_transactions), 2),
        "related_disputes": len(ring_disputes),
        "date_range": {
            "earliest": min((t["timestamp"] for t in ring_transactions), default="N/A"),
            "latest": max((t["timestamp"] for t in ring_transactions), default="N/A"),
        },
    }

    print(f"  Ring transaction volume: ${transaction_summary['total_ring_volume_usd']:,.2f}")
    print(f"  Related disputes: {len(ring_disputes)}")
    print()
    print("  Generating formal investigation report...")

    with create_span("generate_report") as span:
        response = call_openai(
            messages=[
                {"role": "system", "content": (
                    "You are a senior fraud investigator preparing a formal investigation "
                    "report for leadership and the compliance team. Your report must be "
                    "thorough, evidence-based, and suitable for regulatory review. "
                    "Use professional language. Cite specific account IDs, transaction "
                    "amounts, and dates. Include confidence levels for each finding."
                )},
                {"role": "user", "content": (
                    "Generate a formal investigation report based on the following evidence. "
                    "The report should follow this structure:\n\n"
                    "# FRAUD RING INVESTIGATION REPORT\n\n"
                    "## Executive Summary\n"
                    "(2-3 paragraph overview for leadership)\n\n"
                    "## Scope of Investigation\n"
                    "(What was analyzed, time period, data sources)\n\n"
                    "## Findings\n"
                    "### 1. Ring Structure & Members\n"
                    "### 2. Shell Company Network\n"
                    "### 3. Money Flow Analysis\n"
                    "### 4. Evasion Techniques\n\n"
                    "## Evidence Summary Table\n"
                    "(Structured evidence for each member/merchant)\n\n"
                    "## Financial Impact\n"
                    "(Total exposure, recovered vs. at-risk amounts)\n\n"
                    "## Risk Assessment\n"
                    "(Ongoing risk, expansion potential)\n\n"
                    "## Recommended Actions\n"
                    "(Immediate, short-term, long-term)\n\n"
                    "## Confidence Assessment & Limitations\n\n"
                    f"FRAUD RING ANALYSIS:\n{json.dumps(fraud_ring_analysis, indent=2)}\n\n"
                    f"TRANSACTION SUMMARY:\n{json.dumps(transaction_summary, indent=2)}\n\n"
                    f"DISPUTE CASES (sample):\n{json.dumps(ring_disputes[:10], indent=2, default=str)}"
                )},
            ],
            module_name=MODULE_NAME,
        )

        span.set_attribute("tokens_used", response["usage"]["total_tokens"])
        span.set_attribute("report_length_chars", len(response["content"]))
        record_metric("tokens_used", response["usage"]["total_tokens"], "tokens", {"step": "4"})
        record_metric("report_length", len(response["content"]), "chars", {"step": "4"})
        record_metric("ttft_ms", response["ttft_ms"], "ms", {"step": "4"})

    report = response["content"]

    # Save the report
    report_path = DATA_DIR / "investigation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"  ✓ Report generated ({len(report):,} characters)")
    print(f"  ✓ Saved to {report_path}")
    print()

    return report


# ============================================================================
# STEP 5: Summary
# ============================================================================

def step_5_summary(fraud_ring_analysis: dict, report: str):
    """Print module summary."""
    print("=" * 70)
    print("MODULE 04 — Summary")
    print("=" * 70)

    totals = get_total_tokens()
    estimated_cost = estimate_cost(totals["prompt_tokens"], totals["completion_tokens"])

    fraud_ring = fraud_ring_analysis.get("fraud_ring", {})
    members = fraud_ring.get("identified_members", [])
    shells = fraud_ring.get("shell_merchants", [])

    # Export findings for Module 05
    findings = {
        "module": "04_fraud_ring_investigation",
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "identified_ring_members": [m.get("customer_id") for m in members],
        "identified_shell_merchants": [s.get("merchant_id") for s in shells],
        "fraud_ring_analysis": fraud_ring_analysis,
        "report_length_chars": len(report),
        "token_usage": totals,
        "estimated_cost_usd": estimated_cost,
    }

    output_path = DATA_DIR / "module_04_findings.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, default=str)

    # Export telemetry for Module 06
    export_telemetry_to_file(str(DATA_DIR / "telemetry_export.json"))

    print(f"""
  Investigation Results:
    ├─ Ring members identified:  {len(members)}
    ├─ Shell merchants found:    {len(shells)}
    ├─ Report generated:         {len(report):,} characters
    └─ Findings exported to:     {output_path}

  API Usage:
    ├─ API Calls:                {totals['api_calls']}
    ├─ Total Tokens:             {totals['total_tokens']:,}
    └─ Estimated Cost:           ${estimated_cost:.4f}

  RESPONSIBLE AI NOTE:
  This investigation report was generated by an AI model. Before taking
  any action (account freezes, regulatory filings, law enforcement
  referrals), the findings MUST be reviewed by a human investigator.
  Module 05 provides a formal evaluation framework to measure the
  accuracy of these AI-generated findings against ground truth.

  NEXT: Run 05_evaluation_framework.py
""")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Execute Module 04: Fraud Ring Investigation."""
    data = step_1_load_evidence()
    network = step_2_build_network(data)
    fraud_ring_analysis = step_3_network_analysis(data, network)
    report = step_4_generate_report(data, fraud_ring_analysis)
    step_5_summary(fraud_ring_analysis, report)


if __name__ == "__main__":
    main()
