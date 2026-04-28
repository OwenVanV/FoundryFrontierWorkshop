"""
============================================================================
Search + RAG — Synthetic Fraud Document Generator
============================================================================

PURPOSE:
    Generates realistic internal payment processor documents that contain
    evidence of the fraud ring hidden in the synthetic transaction data.

    These documents simulate what a compliance team at PayPal would produce
    during normal operations — SARs, compliance memos, merchant due
    diligence files, audit logs, onboarding summaries, and security alerts.

    The fraud evidence is SCATTERED across documents — no single document
    reveals the full ring. An investigator (or agent) must search across
    all documents and connect the pieces.

DETERMINISM:
    Uses the same ground truth IDs from data/ground_truth.json so the
    evidence is consistent with the transaction/customer/merchant data.

OUTPUTS:
    search_rag/data/documents/  — 10 text documents with embedded evidence

============================================================================
"""

import json
import os
import random
import base64
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv()

random.seed(42)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).parent / "documents"


def load_ground_truth():
    """Load fraud ring ground truth for consistent ID references."""
    with open(DATA_DIR / "ground_truth.json", "r") as f:
        return json.load(f)


def load_sample_data():
    """Load sample merchant/customer names for realistic references."""
    import csv
    customers = []
    with open(DATA_DIR / "customers.csv", "r") as f:
        for row in csv.DictReader(f):
            customers.append(row)
    merchants = []
    with open(DATA_DIR / "merchants.csv", "r") as f:
        for row in csv.DictReader(f):
            merchants.append(row)
    return customers, merchants


def generate_documents():
    gt = load_ground_truth()
    ring = gt["fraud_ring"]
    member_ids = ring["member_customer_ids"]
    shell_ids = ring["shell_merchant_ids"]
    customers, merchants = load_sample_data()

    # Get actual names for ring members and shell merchants
    ring_customers = [c for c in customers if c["customer_id"] in member_ids]
    shell_merchants = [m for m in merchants if m["merchant_id"] in shell_ids]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = []

    # ================================================================
    # Document 1: Suspicious Activity Report — Apex Digital Solutions
    # ================================================================
    doc = {
        "filename": "SAR-2026-0142_apex_digital.txt",
        "type": "suspicious_activity_report",
        "content": f"""SUSPICIOUS ACTIVITY REPORT (SAR)
Filing Reference: SAR-2026-0142
Date Filed: 2026-03-28
Filed By: Maria Chen, Senior Fraud Analyst
Priority: HIGH

SUBJECT ENTITY:
    Merchant: {shell_merchants[0]['business_name']} ({shell_merchants[0]['merchant_id']})
    DBA: {shell_merchants[0]['dba_name']}
    Location: {shell_merchants[0]['city']}, {shell_merchants[0]['state']}
    Registration Date: {shell_merchants[0]['registration_date']}
    MCC Code: {shell_merchants[0]['mcc_code']} ({shell_merchants[0]['mcc_description']})

RELATED ENTITIES:
    The following merchants appear to be related based on naming patterns and
    registration timing:
    - {shell_merchants[1]['business_name']} ({shell_merchants[1]['merchant_id']})
      Registered: {shell_merchants[1]['registration_date']}
    - {shell_merchants[2]['business_name']} ({shell_merchants[2]['merchant_id']})
      Registered: {shell_merchants[2]['registration_date']}
    - {shell_merchants[3]['business_name']} ({shell_merchants[3]['merchant_id']})
      Registered: {shell_merchants[3]['registration_date']}

    All four entities registered within a 10-day window in Delaware. The
    business names share the "Apex" or "A.D.S." prefix. All are categorized
    under Business Services or Computer Network Services despite receiving
    payments that resemble structured transactions.

SUSPICIOUS ACTIVITY DESCRIPTION:
    Between January and April 2026, the subject merchant processed {random.randint(80, 120)}
    transactions with a highly unusual amount distribution. Approximately 85% of
    inbound payments fall within the $9,200-$9,800 range — consistently below
    the $10,000 Bank Secrecy Act reporting threshold.

    Transaction descriptions are generic ("Consulting services", "Software
    development milestone", "IT infrastructure consulting") and do not correspond
    to any verifiable service delivery.

    Several sending accounts ({member_ids[0]}, {member_ids[1]}, {member_ids[2]})
    appear across multiple related merchant accounts, suggesting coordinated
    activity.

RECOMMENDED ACTION:
    Escalate to Financial Crimes Investigation Unit. Consider filing with FinCEN.
    Recommend temporary hold on merchant payouts pending review.

CONFIDENCE LEVEL: HIGH
    Based on amount structuring pattern, merchant name clustering, and
    cross-merchant sender overlap.
"""
    }
    documents.append(doc)

    # ================================================================
    # Document 2: Compliance Review Memo — Q1 2026
    # ================================================================
    doc = {
        "filename": "MEMO-compliance-Q1-2026.txt",
        "type": "compliance_memo",
        "content": f"""INTERNAL MEMORANDUM — CONFIDENTIAL

TO: Payment Operations Leadership Team
FROM: Compliance Review Board
DATE: 2026-04-01
RE: Q1 2026 Quarterly Compliance Review — Merchant Onboarding

EXECUTIVE SUMMARY:
    Q1 2026 merchant onboarding volume was within normal parameters. We
    onboarded 47 new merchant accounts during the quarter. Standard KYC/AML
    checks were performed on all applicants.

ITEMS OF NOTE:

    1. Delaware Registrations (LOW PRIORITY)
       Four merchant accounts were registered in Delaware within a 10-day
       period (Jan 20-29, 2026). While Delaware incorporation is common for
       legitimate businesses, the clustering is noted for the record:
       - {shell_merchants[0]['business_name']}: Jan 20, {shell_merchants[0]['city']}, DE
       - {shell_merchants[1]['business_name']}: Jan 23, Wilmington, DE
       - {shell_merchants[2]['business_name']}: Jan 26, Dover, DE
       - {shell_merchants[3]['business_name']}: Jan 29, Newark, DE

       All passed automated verification. Website checks returned valid
       domains. Business category codes are consistent with stated services.
       No action recommended at this time.

    2. Chargeback Rate Trends
       Overall platform chargeback rate: 1.2% (within SLA).
       Three merchants flagged for elevated rates (>2%): [unrelated IDs].
       The four Delaware merchants noted above have chargeback rates between
       0.5% and 1.8% — within acceptable bounds.

    3. Customer Registration Velocity
       An unusual cluster of {len(member_ids)} customer accounts were registered
       between February 1-14, 2026. The registration pattern shows:
       - Accounts primarily located in Houston, Miami, Chicago, LA, SF
       - Email domains concentrated on two providers: quickinbox.io, dropmail.cc
       - Phone area codes (212, 718, 347, 929, 646) do not match stated cities

       This was flagged by automated monitoring but scored below the alert
       threshold (risk scores ranged 0.10-0.45). The accounts show standard
       transaction patterns and verified identities. Marked for continued
       monitoring but no escalation.

CONCLUSION:
    No material compliance issues identified in Q1 2026. The Delaware merchant
    cluster and February customer registration velocity merit continued passive
    monitoring per standard protocol.

    Approved by: J. Richardson, Chief Compliance Officer
"""
    }
    documents.append(doc)

    # ================================================================
    # Document 3: Merchant Due Diligence — Apex Digital Solutions LLC
    # ================================================================
    doc = {
        "filename": "DD-apex-digital-solutions.txt",
        "type": "due_diligence",
        "content": f"""MERCHANT DUE DILIGENCE REPORT

MERCHANT: {shell_merchants[0]['business_name']}
MERCHANT ID: {shell_merchants[0]['merchant_id']}
DBA: {shell_merchants[0]['dba_name']}
DATE OF REVIEW: 2026-01-22
REVIEWER: K. Patel, Merchant Risk Team

BUSINESS INFORMATION:
    Legal Name: {shell_merchants[0]['business_name']}
    State of Incorporation: Delaware
    Date of Incorporation: 2026-01-18 (2 days before platform registration)
    Registered Agent: National Registered Agents, Inc.
    Business Address: 1209 Orange Street, Wilmington, DE 19801
        NOTE: This is a registered agent address, not a physical office.
        This is standard practice for Delaware LLCs.
    Website: www.apexdigitalsolutions.com
        - Site is operational. Generic consulting services content.
        - No client testimonials or case studies.
        - Contact form only; no direct phone number listed.
    Stated Annual Revenue: $500,000 - $1,000,000
    Number of Employees: 5-10 (per application)

KYC VERIFICATION:
    [✓] Business registration verified via Delaware Division of Corporations
    [✓] EIN verified via IRS TIN matching
    [✓] Beneficial owner identified: [Name redacted per privacy policy]
    [✓] OFAC/SDN screening: No matches
    [✓] Adverse media screening: No results

RISK ASSESSMENT:
    Industry Risk: MEDIUM (Business Services — MCC {shell_merchants[0]['mcc_code']})
    Geographic Risk: LOW (Domestic — Delaware)
    Volume Risk: MEDIUM (Projected monthly volume $50K-$200K)
    Overall Risk Tier: MEDIUM

    The merchant's registered agent address and recently formed LLC are noted
    but not disqualifying factors. Delaware registration with a registered
    agent is standard industry practice for legitimate businesses.

APPROVAL:
    APPROVED for merchant account onboarding.
    Standard monitoring tier applied.
    Next review date: 2026-07-22 (6-month cycle)

RELATED NOTE:
    During the review period, three additional applications were received from
    entities with similar naming conventions:
    - {shell_merchants[1]['business_name']} (applied Jan 23)
    - {shell_merchants[2]['business_name']} (applied Jan 26)
    - {shell_merchants[3]['business_name']} (applied Jan 29)
    Each was processed independently per standard procedure. No formal
    linkage analysis was conducted as the entities have distinct EINs.
"""
    }
    documents.append(doc)

    # ================================================================
    # Document 4: Transaction Audit Log — Q1 High-Value Review
    # ================================================================
    doc = {
        "filename": "AUDIT-q1-highvalue-review.txt",
        "type": "audit_log",
        "content": f"""TRANSACTION AUDIT LOG — Q1 2026 HIGH-VALUE REVIEW

AUDIT PERIOD: January 15 - April 15, 2026
AUDITOR: R. Fernandez, Transaction Monitoring Team
DATE: 2026-04-10

SCOPE:
    Review of all transactions exceeding $5,000 for structuring indicators,
    unusual velocity, and merchant concentration patterns.

SUMMARY STATISTICS:
    Total transactions in period: ~50,000
    Transactions > $5,000: ~2,500 (5.0% of total)
    Transactions $9,000-$10,000: ~{random.randint(180, 250)} ({random.uniform(0.35, 0.50):.2f}% of total)
    Transactions > $10,000: ~{random.randint(400, 600)}

FINDINGS:

    Finding 1: $9,200-$9,800 Amount Clustering (INFO)
        A concentration of {random.randint(100, 160)} transactions was observed in the
        $9,200-$9,800 range. Statistical analysis shows this represents
        approximately 3x the expected frequency for this amount band.

        Top senders in this range:
        - {member_ids[0]}: {random.randint(8, 15)} transactions, avg ${random.uniform(9300, 9700):.2f}
        - {member_ids[3]}: {random.randint(8, 15)} transactions, avg ${random.uniform(9200, 9600):.2f}
        - {member_ids[4]}: {random.randint(6, 12)} transactions, avg ${random.uniform(9400, 9800):.2f}
        - {member_ids[7]}: {random.randint(6, 12)} transactions, avg ${random.uniform(9300, 9700):.2f}

        These transactions route through four Business Services merchants:
        {shell_ids[0]}, {shell_ids[1]}, {shell_ids[2]}, {shell_ids[3]}

        Transaction descriptions are generic consulting/IT service labels.
        The pattern is consistent with legitimate high-value B2B payments in
        the consulting sector. However, the tight amount clustering is noted.

        Risk engine scores for these transactions: 0.08-0.35 (all below
        the 0.50 review threshold).

        CLASSIFICATION: Informational. The amounts do not exceed the $10,000
        BSA threshold. The clustering may reflect standard pricing tiers.
        No escalation recommended.

    Finding 2: Temporal Clustering (INFO)
        Subsets of the high-value transactions described in Finding 1 occur
        in coordinated bursts — 4-8 transactions within 3-7 minute windows.
        This pattern was observed approximately {random.randint(55, 75)} times during the
        audit period.

        Example burst (2026-02-15 14:22-14:27 UTC):
        - {member_ids[0]} → {member_ids[1]}: $9,412.67 via {shell_ids[0]}
        - {member_ids[2]} → {member_ids[3]}: $9,587.23 via {shell_ids[1]}
        - {member_ids[4]} → {member_ids[5]}: $9,334.89 via {shell_ids[2]}
        - {member_ids[6]} → {member_ids[7]}: $9,701.45 via {shell_ids[3]}

        While temporal clustering can indicate coordinated activity, it is
        also consistent with batch payment processing by businesses. The
        transactions passed all automated velocity checks.

        CLASSIFICATION: Informational. Continue standard monitoring.

CONCLUSION:
    No material issues requiring escalation. Findings 1 and 2 are noted for
    the quarterly review record. Standard monitoring continues.
"""
    }
    documents.append(doc)

    # ================================================================
    # Document 5: Customer Onboarding Summary — February 2026 Batch
    # ================================================================
    sample_members = ring_customers[:8]
    doc = {
        "filename": "ONBOARD-feb-2026-batch.txt",
        "type": "onboarding_summary",
        "content": f"""CUSTOMER ONBOARDING SUMMARY — FEBRUARY 2026

PREPARED BY: Automated Onboarding System
DATE: 2026-02-28
PERIOD: February 1-28, 2026

OVERVIEW:
    Total new registrations in February: 142
    Registrations passing automated KYC: 138 (97.2%)
    Registrations requiring manual review: 4 (2.8%)
    Registrations rejected: 0

NOTABLE CLUSTER — February 1-14, 2026:
    {len(member_ids)} accounts registered within a 14-day window with the following
    shared characteristics:

    Account Details:
""" + "\n".join([
    f"    - {c['customer_id']}: {c['first_name']} {c['last_name']}, "
    f"{c['email']}, {c['city']}, {c['state']}, "
    f"registered {c['registration_date'][:10]}"
    for c in sample_members
]) + f"""

    Shared Characteristics:
    - Email domains: All accounts use either quickinbox.io or dropmail.cc
    - Phone area codes: NYC-area codes (212, 718, 347, 929, 646)
    - Stated locations: Houston, Miami, Chicago, LA, San Francisco
    - NOTE: Phone area codes do NOT match stated cities in several cases

    Risk Assessment:
    - All accounts scored between 0.10 and 0.45 on the automated risk model
    - All passed identity verification (some via document upload, some via
      database match)
    - Account tiers: predominantly "standard" with some "premier"

    This registration pattern was flagged by the velocity monitoring system
    but did not trigger escalation because:
    1. Individual risk scores were below the 0.50 threshold
    2. Identity verification passed for all accounts
    3. The registration volume (18 in 14 days) is within normal variance

    RECOMMENDATION: Standard monitoring. No manual review required.
"""
    }
    documents.append(doc)

    # ================================================================
    # Document 6: IT Security Alert — Shared Device Fingerprints
    # ================================================================
    device_pairs = ring["device_sharing_pairs"]
    doc = {
        "filename": "ALERT-device-fingerprint-anomaly.txt",
        "type": "security_alert",
        "content": f"""IT SECURITY ALERT — DEVICE FINGERPRINT ANOMALY

ALERT ID: SEC-2026-0089
DATE: 2026-03-15
SEVERITY: MEDIUM
SOURCE: Device Intelligence Platform
ANALYST: T. Nakamura, Platform Security

SUMMARY:
    Routine device fingerprint analysis has identified three device
    fingerprints that are shared across multiple customer accounts. While
    device sharing can occur legitimately (shared household devices, business
    devices), the pattern below warrants documentation.

SHARED DEVICES DETECTED:

    Device 1 (fingerprint: 9a3f...):
        Used by accounts: {device_pairs[0][0]}, {device_pairs[0][1]}
        Sessions detected: {random.randint(15, 30)} total
        Geographic locations: Houston TX, Miami FL
        Browser: Chrome / Safari (mixed)
        OS: Windows 11, macOS 14

    Device 2 (fingerprint: 7b2e...):
        Used by accounts: {device_pairs[1][0]}, {device_pairs[1][1]}
        Sessions detected: {random.randint(12, 25)} total
        Geographic locations: Chicago IL, Los Angeles CA
        Browser: Chrome
        OS: Windows 11, Android 14

    Device 3 (fingerprint: 4c8d...):
        Used by accounts: {device_pairs[2][0]}, {device_pairs[2][1]}
        Sessions detected: {random.randint(10, 20)} total
        Geographic locations: San Francisco CA, Miami FL
        Browser: Safari / Firefox
        OS: macOS 14, iOS 17

ANALYSIS:
    In each case, two distinct customer accounts with different names,
    addresses, and contact information are accessing the platform from the
    same physical device. The accounts in question were all registered in
    the February 1-14, 2026 window.

    The device intelligence platform assigns a confidence score of 0.92+
    for all three fingerprint matches, indicating high certainty that these
    are the same physical devices.

    Cross-referencing with transaction data: all six accounts identified
    above are active transactors with amounts primarily in the $9,000-$10,000
    range through Business Services merchants.

RISK ASSESSMENT:
    While shared devices are not conclusive evidence of fraud, the combination
    of: (a) recently registered accounts, (b) shared devices across different
    cities, (c) concentrated transaction patterns, suggests further review
    may be warranted.

    The automated risk engine has NOT flagged these accounts — all maintain
    risk scores below 0.45.

ACTION TAKEN:
    Alert logged for quarterly security review. No account restrictions
    applied. Shared with the Fraud Analytics team for awareness.
"""
    }
    documents.append(doc)

    # ================================================================
    # Document 7: SAR — Circular Money Flow Detection
    # ================================================================
    chains = ring["circular_flow_chains"]
    doc = {
        "filename": "SAR-2026-0198_circular_flow.txt",
        "type": "suspicious_activity_report",
        "content": f"""SUSPICIOUS ACTIVITY REPORT (SAR)
Filing Reference: SAR-2026-0198
Date Filed: 2026-04-12
Filed By: D. Okonkwo, Senior Financial Investigator
Priority: HIGH

SUBJECT: Suspected Circular Fund Flow Pattern

EXECUTIVE SUMMARY:
    Analysis of transaction records between January and April 2026 has
    identified a pattern consistent with circular fund flows among a network
    of customer accounts. Money appears to move in cycles with 48-72 hour
    delays between hops, suggesting coordinated layering activity.

IDENTIFIED FLOW CHAINS:

    Chain 1: {chains[0][0]} → {chains[0][1]} → {chains[0][2]} → {chains[0][3]} → {chains[0][0]}
        Cycle count: {random.randint(8, 12)} complete cycles observed
        Average hop amount: $9,200-$9,800 (with ±3% variance per hop)
        Average delay between hops: 48-72 hours
        Merchants used: {shell_ids[0]}, {shell_ids[1]}

    Chain 2: {chains[1][0]} → {chains[1][1]} → {chains[1][2]} → {chains[1][3]} → {chains[1][0]}
        Cycle count: {random.randint(8, 12)} complete cycles observed
        Average hop amount: $9,200-$9,800
        Average delay between hops: 48-72 hours
        Merchants used: {shell_ids[2]}, {shell_ids[3]}

    Chain 3: {chains[2][0]} → {chains[2][1]} → {chains[2][2]} → {chains[2][3]} → {chains[2][0]}
        Cycle count: {random.randint(6, 10)} complete cycles observed
        Average hop amount: $9,200-$9,800
        Average delay between hops: 48-72 hours
        Merchants used: All four Apex-related merchants (rotating)

LAYERING TECHNIQUE:
    Between each circular flow hop, the involved accounts conduct
    noise transactions — small legitimate-appearing transfers — to obscure
    the circular pattern. The delay of 48-72 hours between hops is
    carefully calibrated to avoid same-day velocity triggers while
    maintaining the flow.

    The amount variance (±3% per hop) ensures no two transactions in a
    chain are identical, defeating simple amount-matching detection rules.

ESTIMATED EXPOSURE:
    Total funds circulated: approximately ${random.randint(3, 5)},{random.randint(100, 999)},{random.randint(100, 999):.2f}
    over the 90-day observation period.

CONNECTION TO SAR-2026-0142:
    The merchants involved in these circular flows are the same entities
    identified in SAR-2026-0142 (Apex Digital Solutions and related entities).
    The sending accounts overlap with the customer cluster identified in
    the February 2026 onboarding analysis.

RECOMMENDED ACTIONS:
    1. Immediate: Place holds on all identified accounts pending investigation
    2. Short-term: File CTR supplements for all circular flow transactions
    3. Long-term: Enhance velocity rules to detect delayed circular patterns
    4. Notify FinCEN via SAR filing (this document)

ATTACHMENTS:
    - Transaction detail export (see attached CSV)
    - Network diagram (see attached PDF)
    - Account relationship matrix
"""
    }
    documents.append(doc)

    # ================================================================
    # Document 8: Risk Engine Performance Report
    # ================================================================
    doc = {
        "filename": "REPORT-risk-engine-q1-2026.txt",
        "type": "risk_report",
        "content": f"""RISK ENGINE PERFORMANCE REPORT — Q1 2026

PREPARED BY: Risk Modeling Team
DATE: 2026-04-15
DISTRIBUTION: Risk Committee, Fraud Operations

MODEL VERSION: risk_engine_v3.2.1

PERFORMANCE METRICS:
    Total transactions scored: ~50,000
    Transactions flagged (score > 0.50): ~1,500 (3.0%)
    Transactions held for review (score > 0.70): ~450 (0.9%)
    Confirmed fraud in flagged transactions: 62 (4.1% precision)
    Known fraud missed (false negatives): Under investigation

RULE-BY-RULE PERFORMANCE:

    velocity_check:
        Triggered: 892 times
        True positives: 45
        Analysis: Effective for burst fraud but misses delayed patterns

    amount_threshold:
        Triggered: 1,204 times
        True positives: 38
        Analysis: BSA threshold ($10K) is well-known. Sophisticated actors
        structure below it. The $9,000-$10,000 range shows elevated
        transaction density but individual transactions score LOW (0.20-0.45)
        because each is below the threshold.

    network_analysis:
        Triggered: 234 times
        True positives: 28
        Analysis: Detects direct peer-to-peer abuse but DOES NOT detect
        delayed circular flows (48-72hr hop delays exceed the rule's
        24-hour lookback window).

        ** KNOWN GAP: Accounts {member_ids[0]}, {member_ids[1]}, {member_ids[2]},
        {member_ids[3]} consistently score 0.30-0.48 on the network analysis
        rule — elevated but below the 0.50 trigger threshold. The rule's
        "minor network density elevation" classification keeps these accounts
        in the "within tolerance" band.

    device_reputation:
        Triggered: 156 times
        True positives: 12
        Analysis: Device fingerprint sharing is detected as a feature but
        weighted insufficiently. Accounts sharing devices scored only 0.15-0.38
        on this rule because the shared devices have "clean" history.

GAPS IDENTIFIED:

    Gap 1: Delayed Circular Flow Detection
        The network_analysis rule uses a 24-hour lookback window. The
        suspected circular flow pattern uses 48-72 hour delays specifically
        to evade this window. RECOMMENDATION: Extend lookback to 7 days.

    Gap 2: Cross-Merchant Sender Concentration
        The current rules evaluate merchant risk and sender risk independently.
        They do not flag when the SAME senders appear across MULTIPLE
        merchants. The four Apex-related merchants each appear clean
        individually but collectively process payments from the same sender
        pool. RECOMMENDATION: Add cross-merchant sender overlap rule.

    Gap 3: Registration Velocity + Transaction Pattern Correlation
        The onboarding velocity flag and the transaction pattern flags are
        evaluated in separate pipelines. The February registration cluster
        was noted but not correlated with the subsequent transaction
        structuring pattern. RECOMMENDATION: Cross-pipeline correlation.

MODEL UPDATE RECOMMENDATIONS:
    1. Extend network_analysis lookback to 7 days (addresses Gap 1)
    2. Add cross-merchant sender concentration rule (addresses Gap 2)
    3. Build correlation pipeline between onboarding and transaction signals
    4. Increase device_reputation weight for recently registered accounts
    5. Add amount-band concentration detection ($9K-$10K specifically)
"""
    }
    documents.append(doc)

    # ================================================================
    # Document 9: Impossible Travel Alert
    # ================================================================
    travel_accts = ring["impossible_travel_accounts"]
    doc = {
        "filename": "ALERT-impossible-travel.txt",
        "type": "security_alert",
        "content": f"""GEOLOCATION ANOMALY ALERT — IMPOSSIBLE TRAVEL

ALERT ID: GEO-2026-0034
DATE: 2026-03-20
SOURCE: Geolocation Intelligence System
SEVERITY: MEDIUM

SUMMARY:
    Three customer accounts have been flagged for geolocation anomalies
    indicating impossible physical travel between payment sessions.

FLAGGED ACCOUNTS:

    Account: {travel_accts[0]}
        Session 1: Lagos, Nigeria — 2026-03-05 09:14 UTC
        Session 2: London, England — 2026-03-05 09:29 UTC
        Time gap: 15 minutes
        Distance: ~5,600 km
        VPN detected: No
        IP geolocation match: Yes (both sessions geolocate correctly)

    Account: {travel_accts[1]}
        Session 1: Lagos, Nigeria — 2026-02-22 14:42 UTC
        Session 2: London, England — 2026-02-22 15:00 UTC
        Time gap: 18 minutes
        Distance: ~5,600 km
        VPN detected: No

    Account: {travel_accts[2]}
        Session 1: Mumbai, India — 2026-03-10 11:33 UTC
        Session 2: Singapore — 2026-03-10 11:45 UTC
        Time gap: 12 minutes
        Distance: ~4,400 km
        VPN detected: No

ANALYSIS:
    The absence of VPN detection and positive IP geolocation matching in
    both sessions makes these particularly noteworthy. Typical VPN-based
    evasion would show mismatched IP geolocations or known VPN exit nodes.

    These patterns suggest either:
    a) Multiple individuals using the same account credentials from
       different physical locations, or
    b) A sophisticated proxy/tunneling setup that evades standard VPN detection

    All three accounts were registered in the February 2026 onboarding
    window. Cross-referencing with the device fingerprint analysis
    (SEC-2026-0089), account {travel_accts[0]} is also flagged for device
    sharing with another account.

RISK SCORE IMPACT:
    Despite the geolocation anomaly, the automated risk engine did NOT
    elevate these accounts' risk scores because:
    - The geo anomaly check is weighted at only 0.15 in the composite score
    - The accounts' other signals (identity verification, account age,
      transaction history) pull the composite score down
    - Current composite scores: 0.25-0.40 (below 0.50 threshold)

ACTION:
    Logged for quarterly review. No account restrictions applied.
    Shared with SAR team for potential inclusion in ongoing investigations.
"""
    }
    documents.append(doc)

    # ================================================================
    # Document 10: Dispute Summary — Unauthorized Charges
    # ================================================================
    doc = {
        "filename": "DISPUTES-apex-merchants-summary.txt",
        "type": "dispute_summary",
        "content": f"""DISPUTE & CHARGEBACK SUMMARY — APEX-RELATED MERCHANTS

PREPARED BY: Dispute Resolution Team
DATE: 2026-04-18
PERIOD: January - April 2026

MERCHANTS COVERED:
    {shell_merchants[0]['merchant_id']}: {shell_merchants[0]['business_name']}
    {shell_merchants[1]['merchant_id']}: {shell_merchants[1]['business_name']}
    {shell_merchants[2]['merchant_id']}: {shell_merchants[2]['business_name']}
    {shell_merchants[3]['merchant_id']}: {shell_merchants[3]['business_name']}

DISPUTE STATISTICS:
    Total disputes filed against these merchants: {random.randint(12, 22)}
    Dispute categories:
    - Unauthorized transaction: {random.randint(8, 14)} ({random.uniform(60, 75):.0f}%)
    - Services not rendered: {random.randint(2, 5)}
    - Item not received: {random.randint(1, 3)}

    Overall chargeback rate: 1.2-1.8% (below 2% threshold)
    NOTE: The chargeback rate is kept below the platform's 2% threshold,
    which would trigger automatic merchant review.

NOTABLE DISPUTE PATTERNS:
    The majority of disputes are filed by customers who are NOT among the
    regular senders to these merchants. These appear to be THIRD-PARTY
    VICTIMS — customers whose payment credentials were used without
    authorization to send funds to the Apex-related merchants.

    Sample disputes:
    - "I did not authorize this transaction. I have never heard of this
       company." (filed {random.randint(3, 6)} times)
    - "This charge appeared on my account without my knowledge."
       (filed {random.randint(2, 4)} times)
    - "Unauthorized charge. I have never done business with this company."
       (filed {random.randint(2, 3)} times)

    In contrast, the accounts identified in the February 2026 onboarding
    cluster ({member_ids[0]}, {member_ids[1]}, etc.) have filed ZERO
    disputes — which is itself unusual for accounts with such high
    transaction volumes.

RESOLUTION STATUS:
    - Resolved in buyer's favor: {random.randint(5, 10)}
    - Resolved in seller's favor: {random.randint(3, 6)}
    - Pending: {random.randint(2, 5)}
    - Escalated: {random.randint(1, 3)}

RECOMMENDATION:
    The unauthorized transaction pattern from non-associated accounts
    combined with zero disputes from high-volume associated accounts
    supports the hypothesis of coordinated activity. Recommend including
    this dispute data in the ongoing SAR-2026-0142 investigation.
"""
    }
    documents.append(doc)

    # ================================================================
    # Write all documents
    # ================================================================
    print(f"Generating {len(documents)} synthetic fraud documents...")
    for doc in documents:
        filepath = OUTPUT_DIR / doc["filename"]
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(doc["content"])
        print(f"  ✓ {doc['filename']} ({doc['type']})")

    # Write a manifest for tracking
    manifest = [{"filename": d["filename"], "type": d["type"]} for d in documents]
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  {len(documents)} text documents written to {OUTPUT_DIR}")

    # ================================================================
    # Generate document IMAGES via gpt-image-2
    # ================================================================
    # These are AI-generated images of realistic-looking documents,
    # emails, and reports that contain fraud evidence. Content
    # Understanding will analyze these using its image analyzer.
    # ================================================================
    print()
    print("  Generating document images via gpt-image-2...")

    image_docs = generate_document_images(gt, ring_customers, shell_merchants)
    documents.extend(image_docs)

    # Update manifest with images
    manifest = [{"filename": d["filename"], "type": d["type"]} for d in documents]
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Total: {len(documents)} documents ({len(documents) - len(image_docs)} text + {len(image_docs)} images)")
    print(f"  Written to: {OUTPUT_DIR}")
    return documents


def generate_document_images(gt, ring_customers, shell_merchants):
    """Generate realistic document images using gpt-image-2.

    Creates images that look like scanned/photographed internal documents
    containing fraud evidence — emails, printed reports, handwritten notes.
    Content Understanding's image analyzer will extract text and fields.
    """
    ring = gt["fraud_ring"]
    member_ids = ring["member_customer_ids"]
    shell_ids = ring["shell_merchant_ids"]

    # Set up the Azure OpenAI client for image generation
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    use_mi = os.getenv("USE_MANAGED_IDENTITY", "false").lower() in ("true", "1")

    if not endpoint:
        print("    ⚠ AZURE_OPENAI_ENDPOINT not set — skipping image generation")
        return []

    from openai import AzureOpenAI
    if use_mi:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version="2025-03-01-preview",
        )
    else:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2025-03-01-preview",
        )

    image_prompts = [
        {
            "filename": "IMG-email-whistleblower-tip.png",
            "type": "email_screenshot",
            "prompt": (
                "A realistic screenshot of a corporate email inbox showing an email message. "
                "The email is from 'anonymous_tip@protonmail.com' to 'fraud.team@paymentprocessor.com', "
                "subject line: 'Suspicious Activity - Apex Digital Solutions'. "
                "The email body reads: 'I am a former contractor for Apex Digital Solutions LLC. "
                "I want to report that this company and three related entities (Apex Digitial Svcs, "
                "Apex D.S. Consulting, ADS Global Partners) are being used to process fraudulent "
                "payments. The amounts are always between $9,200 and $9,800 to stay below the "
                "$10,000 reporting threshold. The same group of about 18 people control all the "
                f"accounts. Key account IDs I observed: {member_ids[0]}, {member_ids[1]}, "
                f"{member_ids[2]}. Merchant IDs: {shell_ids[0]}, {shell_ids[1]}. "
                "Please investigate.' "
                "The email interface should look like Microsoft Outlook with a corporate theme. "
                "Make it look like a real email screenshot, photorealistic."
            ),
        },
        {
            "filename": "IMG-handwritten-investigation-notes.png",
            "type": "handwritten_notes",
            "prompt": (
                "A photograph of handwritten investigation notes on a yellow legal pad. "
                "The handwriting is neat and legible. The notes read: "
                "'FRAUD RING INVESTIGATION NOTES - March 2026\n"
                "- 4 shell companies, all Delaware LLCs, registered Jan 20-29\n"
                "- Apex Digital Solutions, Apex Digitial Svcs, Apex DS Consulting, ADS Global\n"
                "- 18 linked accounts, all registered Feb 1-14\n"
                "- Transaction pattern: $9,200-$9,800 range (BSA structuring)\n"
                "- Circular flow detected: A->B->C->D->A with 48-72hr delays\n"
                f"- Key suspects: {member_ids[0]}, {member_ids[3]}, {member_ids[7]}\n"
                "- Device fingerprints shared across 6 accounts (3 devices)\n"
                "- Impossible travel: Lagos to London in 15 min\n"
                "- Risk engine MISSED IT - all scores below 0.50 threshold\n"
                "- TOTAL EXPOSURE: ~$4.2M over 90 days\n"
                "- ACTION: File SAR, freeze accounts, notify FinCEN'\n"
                "The legal pad sits on a dark wooden desk with a pen beside it. Photorealistic photo."
            ),
        },
        {
            "filename": "IMG-printed-transaction-report.png",
            "type": "printed_report",
            "prompt": (
                "A photograph of a printed financial transaction report on white paper. "
                "The report is titled 'HIGH-VALUE TRANSACTION SUMMARY - Q1 2026'. "
                "It shows a table with columns: Date, Sender, Receiver, Merchant, Amount, Status. "
                "The visible rows show transactions like: "
                f"'2026-02-15, {member_ids[0]}, {member_ids[1]}, {shell_ids[0]}, $9,412.67, Completed', "
                f"'2026-02-15, {member_ids[2]}, {member_ids[3]}, {shell_ids[1]}, $9,587.23, Completed', "
                f"'2026-02-17, {member_ids[1]}, {member_ids[2]}, {shell_ids[2]}, $9,334.89, Completed', "
                f"'2026-02-19, {member_ids[3]}, {member_ids[0]}, {shell_ids[3]}, $9,701.45, Completed'. "
                "Some rows are highlighted in yellow with a handwritten annotation 'CIRCULAR FLOW!' "
                "in red ink. The paper looks like it came from a laser printer. Photorealistic."
            ),
        },
        {
            "filename": "IMG-slack-message-fraud-team.png",
            "type": "chat_screenshot",
            "prompt": (
                "A screenshot of a corporate Slack channel called #fraud-investigations. "
                "The conversation shows: "
                "Maria Chen (10:14 AM): 'Has anyone looked at the Apex Digital cluster? "
                "I'm seeing the same 18 accounts routing through all 4 merchants.' "
                "Raj Patel (10:16 AM): 'Yes, I filed SAR-2026-0142 last week. The structuring "
                "pattern is textbook - amounts between $9,200-$9,800.' "
                "Tanya Nakamura (10:18 AM): 'IT flagged device fingerprint sharing too. "
                "Three devices shared across six of those accounts. Alert SEC-2026-0089.' "
                "David Okonkwo (10:22 AM): 'I found circular flows. Money goes A→B→C→D→A "
                "with 48-72hr delays between hops. Filing SAR-2026-0198 today.' "
                "Maria Chen (10:25 AM): 'The risk engine scored them all below 0.50. "
                "We need to update the network analysis rule - 24hr lookback is too short.' "
                "The Slack interface should look realistic with profile pictures and timestamps."
            ),
        },
    ]

    image_docs = []

    import concurrent.futures

    def _generate_one(img_spec):
        """Generate a single image — called in parallel."""
        filename = img_spec["filename"]
        filepath = OUTPUT_DIR / filename
        try:
            response = client.images.generate(
                model="gpt-image-2",
                prompt=img_spec["prompt"],
                n=1,
                size="1024x1024",
                quality="medium",
            )
            image_b64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_b64)
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            return {
                "filename": filename,
                "type": img_spec["type"],
                "content": f"[Image: {img_spec['type']}]",
                "size_kb": len(image_bytes) // 1024,
                "error": None,
            }
        except Exception as e:
            return {"filename": filename, "error": str(e)}

    # Generate up to 5 images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_generate_one, spec): spec for spec in image_prompts}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result.get("error"):
                print(f"    ✗ {result['filename']}: {result['error']}")
            else:
                print(f"    ✓ {result['filename']} ({result['size_kb']} KB)")
                image_docs.append(result)

    return image_docs


if __name__ == "__main__":
    generate_documents()
