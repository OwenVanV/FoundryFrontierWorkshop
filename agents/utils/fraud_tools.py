"""
============================================================================
Agents Deep-Dive — Custom Fraud Investigation Tools
============================================================================

PURPOSE:
    @tool-decorated functions for the Microsoft Agent Framework (MAF) agents.
    These tools give MAF agents the ability to query the synthetic fraud
    datasets directly — transactions, customers, merchants, sessions.

    Foundry Agent Service agents use code_interpreter to run pandas inside
    a sandbox. MAF agents use THESE tools to query data locally.

DESIGN:
    Each tool reads from the CSV/JSON files in ../data/ and returns a
    formatted string result. The @tool decorator from agent-framework
    automatically generates the JSON schema for the LLM.

============================================================================
"""

import csv
import json
import os
from typing import Annotated
from pathlib import Path
from collections import defaultdict

from agent_framework import tool
from pydantic import Field

# ============================================================================
# DATA DIRECTORY — resolve relative to this file
# ============================================================================
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _load_csv(filename: str) -> list[dict]:
    """Load a CSV file from the data directory."""
    filepath = DATA_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_json(filename: str) -> list[dict]:
    """Load a JSON file from the data directory."""
    filepath = DATA_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# TOOL: Query Transactions
# ============================================================================

@tool(approval_mode="never_require")
def query_transactions(
    sender_id: Annotated[str | None, Field(description="Filter by sender customer ID (e.g. CUS-XXXX). Leave empty for no filter.", default=None)] = None,
    merchant_id: Annotated[str | None, Field(description="Filter by merchant ID (e.g. MER-XXXX). Leave empty for no filter.", default=None)] = None,
    min_amount: Annotated[float | None, Field(description="Minimum transaction amount in USD. Leave empty for no filter.", default=None)] = None,
    max_amount: Annotated[float | None, Field(description="Maximum transaction amount in USD. Leave empty for no filter.", default=None)] = None,
    limit: Annotated[int, Field(description="Maximum number of results to return.", default=50)] = 50,
) -> str:
    """Query the transaction ledger with optional filters. Returns matching transactions as a formatted summary."""
    transactions = _load_csv("transactions.csv")
    results = []

    for t in transactions:
        amount = float(t["amount"])
        if sender_id and t["sender_id"] != sender_id:
            continue
        if merchant_id and t["merchant_id"] != merchant_id:
            continue
        if min_amount is not None and amount < min_amount:
            continue
        if max_amount is not None and amount > max_amount:
            continue
        results.append(t)
        if len(results) >= limit:
            break

    if not results:
        return "No transactions found matching the specified filters."

    # Format as a readable summary
    lines = [f"Found {len(results)} transactions (showing up to {limit}):"]
    for t in results[:limit]:
        lines.append(
            f"  {t['transaction_id']} | {t['timestamp']} | "
            f"${float(t['amount']):,.2f} {t['currency']} | "
            f"{t['sender_id']} → {t['receiver_id']} | "
            f"Merchant: {t['merchant_id']} | Status: {t['status']} | "
            f"Type: {t['transaction_type']} | Desc: {t['description']}"
        )
    return "\n".join(lines)


# ============================================================================
# TOOL: Lookup Customer Profile
# ============================================================================

@tool(approval_mode="never_require")
def lookup_customer(
    customer_id: Annotated[str, Field(description="The customer ID to look up (e.g. CUS-XXXX).")],
) -> str:
    """Look up a customer's profile including registration date, email domain, city, risk score, and account details."""
    customers = _load_csv("customers.csv")
    for c in customers:
        if c["customer_id"] == customer_id:
            return (
                f"Customer Profile: {c['customer_id']}\n"
                f"  Name: {c['first_name']} {c['last_name']}\n"
                f"  Email: {c['email']}\n"
                f"  Phone: {c['phone']}\n"
                f"  Location: {c['city']}, {c['state']}, {c['country']}\n"
                f"  Registered: {c['registration_date']}\n"
                f"  Account Tier: {c['account_tier']}\n"
                f"  Account Status: {c['account_status']}\n"
                f"  Verified Identity: {c['verified_identity']}\n"
                f"  Lifetime Transactions: {c['lifetime_transaction_count']}\n"
                f"  Lifetime Volume: ${float(c['lifetime_transaction_volume']):,.2f}\n"
                f"  Risk Score: {c['risk_score']}"
            )
    return f"Customer {customer_id} not found."


# ============================================================================
# TOOL: Check Merchant
# ============================================================================

@tool(approval_mode="never_require")
def check_merchant(
    merchant_id: Annotated[str, Field(description="The merchant ID to check (e.g. MER-XXXX).")],
) -> str:
    """Check a merchant's profile including business name, MCC code, registration date, chargeback rate, and risk tier."""
    merchants = _load_csv("merchants.csv")
    for m in merchants:
        if m["merchant_id"] == merchant_id:
            return (
                f"Merchant Profile: {m['merchant_id']}\n"
                f"  Business Name: {m['business_name']}\n"
                f"  DBA Name: {m['dba_name']}\n"
                f"  MCC: {m['mcc_code']} ({m['mcc_description']})\n"
                f"  Location: {m['city']}, {m['state']}, {m['country']}\n"
                f"  Registered: {m['registration_date']}\n"
                f"  Verified: {m['verified']}\n"
                f"  Avg Ticket: ${float(m['average_ticket_size']):,.2f}\n"
                f"  Monthly Volume: ${float(m['monthly_volume']):,.2f}\n"
                f"  Chargeback Rate: {float(m['chargeback_rate'])*100:.2f}%\n"
                f"  Account Age: {m['account_age_days']} days\n"
                f"  Risk Tier: {m['industry_risk_tier']}"
            )
    return f"Merchant {merchant_id} not found."


# ============================================================================
# TOOL: Find Similar Merchants
# ============================================================================

@tool(approval_mode="never_require")
def find_similar_merchants(
    name_fragment: Annotated[str, Field(description="A name fragment to search for in merchant business names (case-insensitive).")],
) -> str:
    """Search for merchants whose business name contains the given fragment. Useful for finding shell company clusters with similar names."""
    merchants = _load_csv("merchants.csv")
    fragment_lower = name_fragment.lower()
    matches = [m for m in merchants if fragment_lower in m["business_name"].lower()]

    if not matches:
        return f"No merchants found matching '{name_fragment}'."

    lines = [f"Found {len(matches)} merchants matching '{name_fragment}':"]
    for m in matches:
        lines.append(
            f"  {m['merchant_id']} | {m['business_name']} (DBA: {m['dba_name']}) | "
            f"MCC: {m['mcc_code']} | {m['city']}, {m['state']} | "
            f"Registered: {m['registration_date']} | Risk: {m['industry_risk_tier']}"
        )
    return "\n".join(lines)


# ============================================================================
# TOOL: Analyze Account Network
# ============================================================================

@tool(approval_mode="never_require")
def analyze_account_network(
    customer_id: Annotated[str, Field(description="The customer ID to analyze network connections for.")],
) -> str:
    """Analyze a customer's transaction network — who they send to, who sends to them, through which merchants. Returns connection summary."""
    transactions = _load_csv("transactions.csv")

    sent_to = defaultdict(lambda: {"count": 0, "total": 0.0, "merchants": set()})
    received_from = defaultdict(lambda: {"count": 0, "total": 0.0, "merchants": set()})

    for t in transactions:
        amount = float(t["amount"])
        if t["sender_id"] == customer_id:
            key = t["receiver_id"]
            sent_to[key]["count"] += 1
            sent_to[key]["total"] += amount
            sent_to[key]["merchants"].add(t["merchant_id"])
        elif t["receiver_id"] == customer_id:
            key = t["sender_id"]
            received_from[key]["count"] += 1
            received_from[key]["total"] += amount
            received_from[key]["merchants"].add(t["merchant_id"])

    lines = [f"Network Analysis for {customer_id}:"]
    lines.append(f"\n  OUTGOING ({len(sent_to)} recipients):")
    for cid, info in sorted(sent_to.items(), key=lambda x: x[1]["total"], reverse=True)[:15]:
        lines.append(
            f"    → {cid}: {info['count']} txns, ${info['total']:,.2f} total, "
            f"via merchants: {', '.join(sorted(info['merchants']))}"
        )

    lines.append(f"\n  INCOMING ({len(received_from)} senders):")
    for cid, info in sorted(received_from.items(), key=lambda x: x[1]["total"], reverse=True)[:15]:
        lines.append(
            f"    ← {cid}: {info['count']} txns, ${info['total']:,.2f} total, "
            f"via merchants: {', '.join(sorted(info['merchants']))}"
        )

    # Check for bidirectional flows (A sends to B AND B sends to A)
    bidirectional = set(sent_to.keys()) & set(received_from.keys())
    if bidirectional:
        lines.append(f"\n  ⚠ BIDIRECTIONAL FLOWS ({len(bidirectional)} accounts):")
        for cid in sorted(bidirectional):
            lines.append(
                f"    ↔ {cid}: sent {sent_to[cid]['count']} (${sent_to[cid]['total']:,.2f}) / "
                f"received {received_from[cid]['count']} (${received_from[cid]['total']:,.2f})"
            )

    return "\n".join(lines)


# ============================================================================
# TOOL: Check Device Fingerprints
# ============================================================================

@tool(approval_mode="never_require")
def check_device_fingerprints(
    customer_id: Annotated[str, Field(description="The customer ID to check device fingerprints for.")],
) -> str:
    """Check what device fingerprints a customer has used across payment sessions. Also shows if other customers share those devices."""
    sessions = _load_json("payment_sessions.json")

    # Find all fingerprints used by this customer
    customer_fps = set()
    for s in sessions:
        if s.get("customer_id") == customer_id:
            fp = s.get("device", {}).get("fingerprint")
            if fp:
                customer_fps.add(fp)

    if not customer_fps:
        return f"No device fingerprints found for {customer_id}."

    # Find other customers sharing these fingerprints
    fp_to_customers = defaultdict(set)
    for s in sessions:
        fp = s.get("device", {}).get("fingerprint")
        if fp and fp in customer_fps:
            fp_to_customers[fp].add(s.get("customer_id"))

    lines = [f"Device Fingerprint Analysis for {customer_id}:"]
    lines.append(f"  Unique devices used: {len(customer_fps)}")

    shared_found = False
    for fp, customers in fp_to_customers.items():
        other_customers = customers - {customer_id}
        if other_customers:
            shared_found = True
            lines.append(f"\n  ⚠ SHARED DEVICE: {fp[:24]}...")
            lines.append(f"    Also used by: {', '.join(sorted(other_customers))}")

    if not shared_found:
        lines.append("  No shared devices detected with other customers.")

    return "\n".join(lines)
