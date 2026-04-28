"""
============================================================================
PayPal x Azure AI Workshop — Synthetic Data Generator
============================================================================

PURPOSE:
    This script generates all synthetic datasets for the "Cracking the Ring"
    fraud investigation workshop. It creates a realistic payment ecosystem
    with ~50,000 transactions, ~500 merchants, ~2,000 customers, and embeds
    a hidden fraud ring that participants will uncover using Azure OpenAI.

THE HIDDEN FRAUD RING:
    Buried inside this data are 18 accounts operating through 4 shell
    merchants in a coordinated fraud ring. The ring exhibits 7 interlocking
    signals — each individually plausible, but collectively damning:

    1. Smurfing (structured transactions just below $10K threshold)
    2. Circular money flow (A→B→C→D→A with 48-72hr delays)
    3. Shell merchant clustering (similar names, recent registration)
    4. Device fingerprint sharing (6 accounts share 3 devices)
    5. Impossible travel (same user in Lagos and London in 20 min)
    6. Synthetic identity signals (clustered registration, mismatched data)
    7. Temporal coordination (ring transactions fire within 3-7 min windows)

DETERMINISM:
    All random operations are seeded (SEED=42) so every workshop
    participant generates identical datasets. The fraud ring is
    programmatically embedded — not hand-placed — ensuring consistency.

OUTPUTS:
    data/transactions.csv       — ~50,000 transaction records
    data/merchants.csv          — ~500 merchant profiles
    data/customers.csv          — ~2,000 customer profiles
    data/payment_sessions.json  — Nested session/device/geo data
    data/dispute_cases.json     — Chargeback cases with evidence threads
    data/risk_signals.json      — Risk engine scoring breakdowns
    data/velocity_metrics.csv   — Hourly time-series aggregations
    data/system_telemetry.csv   — Platform health metrics

USAGE:
    python data/generate_synthetic_data.py

============================================================================
"""

import csv
import json
import os
import random
import string
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ============================================================================
# CONFIGURATION — All randomness flows from this single seed.
# Change this and you get a different (but internally consistent) universe.
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Dataset sizes
NUM_CUSTOMERS = 2000
NUM_MERCHANTS = 500
NUM_TRANSACTIONS = 50000
NUM_PAYMENT_SESSIONS = 15000
NUM_DISPUTE_CASES = 350
NUM_RISK_SIGNALS = 8000

# Fraud ring dimensions
FRAUD_RING_MEMBERS = 18
FRAUD_SHELL_MERCHANTS = 4
FRAUD_RING_TRANSACTIONS = 420  # ~0.84% of all transactions — needle in haystack

# Time range for the dataset (90 days of activity)
START_DATE = datetime(2026, 1, 15)
END_DATE = datetime(2026, 4, 15)

# Output directory (same folder as this script)
OUTPUT_DIR = Path(__file__).parent

# ============================================================================
# HELPER: Deterministic ID generation
# ============================================================================

def generate_id(prefix: str, index: int) -> str:
    """Generate a deterministic, realistic-looking ID.

    We hash the prefix+index so IDs look random but are reproducible.
    PayPal-style IDs are alphanumeric, 13-17 characters.
    """
    raw = hashlib.sha256(f"{prefix}-{index}-{SEED}".encode()).hexdigest()
    return f"{prefix.upper()}-{raw[:12].upper()}"


def generate_device_fingerprint(index: int) -> str:
    """Generate a device fingerprint hash — looks like a real browser fingerprint."""
    raw = hashlib.sha256(f"device-{index}-{SEED}".encode()).hexdigest()
    return raw[:32]


def random_timestamp(start: datetime, end: datetime) -> datetime:
    """Generate a random timestamp between start and end, biased toward business hours."""
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    ts = start + timedelta(seconds=random_seconds)

    # 70% chance of being during business hours (8 AM - 8 PM)
    if random.random() < 0.7:
        ts = ts.replace(hour=random.randint(8, 19), minute=random.randint(0, 59))
    return ts


# ============================================================================
# REFERENCE DATA — Realistic categories, names, locations
# ============================================================================

# Merchant Category Codes (MCC) used in payment processing
MCC_CODES = {
    "5411": "Grocery Stores",
    "5812": "Eating Places & Restaurants",
    "5691": "Clothing Stores",
    "5944": "Jewelry Stores",
    "5999": "Miscellaneous Retail",
    "7011": "Hotels & Motels",
    "7230": "Beauty & Barber Shops",
    "7299": "Miscellaneous Recreation",
    "7372": "Computer Software Stores",
    "7399": "Business Services",
    "4816": "Computer Network Services",
    "5045": "Computers & Peripherals",
    "5065": "Electronic Parts & Equipment",
    "5732": "Electronics Stores",
    "5912": "Drug Stores & Pharmacies",
    "5942": "Book Stores",
    "5947": "Gift & Card Shops",
    "6012": "Financial Institutions",
    "6051": "Non-FI Money Orders",
    "8011": "Doctors",
    "8021": "Dentists",
    "8099": "Health Services",
    "4121": "Taxicabs & Rideshares",
    "5541": "Service Stations",
    "5814": "Fast Food Restaurants",
}

# Geographic distribution reflecting PayPal's global corridors
CITIES = [
    ("New York", "NY", "US", 40.7128, -74.0060, "America/New_York"),
    ("Los Angeles", "CA", "US", 34.0522, -118.2437, "America/Los_Angeles"),
    ("Chicago", "IL", "US", 41.8781, -87.6298, "America/Chicago"),
    ("Houston", "TX", "US", 29.7604, -95.3698, "America/Chicago"),
    ("Miami", "FL", "US", 25.7617, -80.1918, "America/New_York"),
    ("San Francisco", "CA", "US", 37.7749, -122.4194, "America/Los_Angeles"),
    ("London", "England", "GB", 51.5074, -0.1278, "Europe/London"),
    ("Berlin", "Germany", "DE", 52.5200, 13.4050, "Europe/Berlin"),
    ("Paris", "France", "FR", 48.8566, 2.3522, "Europe/Paris"),
    ("Tokyo", "Japan", "JP", 35.6762, 139.6503, "Asia/Tokyo"),
    ("Sydney", "Australia", "AU", -33.8688, 151.2093, "Australia/Sydney"),
    ("Toronto", "Canada", "CA", 43.6532, -79.3832, "America/Toronto"),
    ("Singapore", "Singapore", "SG", 1.3521, 103.8198, "Asia/Singapore"),
    ("Dubai", "UAE", "AE", 25.2048, 55.2708, "Asia/Dubai"),
    ("Mumbai", "India", "IN", 19.0760, 72.8777, "Asia/Kolkata"),
    ("São Paulo", "Brazil", "BR", -23.5505, -46.6333, "America/Sao_Paulo"),
    ("Lagos", "Nigeria", "NG", 6.5244, 3.3792, "Africa/Lagos"),
    ("Mexico City", "Mexico", "MX", 19.4326, -99.1332, "America/Mexico_City"),
    ("Amsterdam", "Netherlands", "NL", 52.3676, 4.9041, "Europe/Amsterdam"),
    ("Stockholm", "Sweden", "SE", 59.3293, 18.0686, "Europe/Stockholm"),
]

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "SGD", "AED", "INR", "BRL"]
CURRENCY_WEIGHTS = [0.45, 0.20, 0.10, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03]

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Christopher", "Karen", "Charles", "Lisa", "Daniel", "Nancy",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
    "Amir", "Fatima", "Wei", "Yuki", "Raj", "Priya", "Ahmed", "Mei",
    "Carlos", "Sofia", "Hans", "Ingrid", "Lars", "Astrid", "Oluwaseun", "Chioma",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Chen", "Wang", "Tanaka", "Patel", "Singh", "Kim", "Okonkwo", "Johansson",
    "Mueller", "Larsson", "Al-Rashid", "Ibrahim", "Santos", "Silva", "Andersson",
]

EMAIL_DOMAINS_LEGIT = [
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com",
    "protonmail.com", "aol.com", "mail.com", "zoho.com", "fastmail.com",
]

# --- Fraud ring specific: obscure email domains that cluster ---
EMAIL_DOMAINS_FRAUD = ["quickinbox.io", "dropmail.cc"]

MERCHANT_NAME_PARTS = [
    "Global", "Premier", "Elite", "Pacific", "Atlantic", "Metro", "Urban",
    "Summit", "Pinnacle", "Nexus", "Horizon", "Vertex", "Nova", "Prime",
    "Sterling", "Crown", "Royal", "Grand", "Liberty", "Heritage",
]

MERCHANT_NAME_TYPES = [
    "Trading Co", "Enterprises", "Solutions", "Group", "International",
    "Services", "Holdings", "Ventures", "Partners", "Associates",
    "Industries", "Technologies", "Consulting", "Supply", "Logistics",
]

PAYMENT_METHODS = ["paypal_balance", "credit_card", "debit_card", "bank_transfer", "crypto"]
PAYMENT_METHOD_WEIGHTS = [0.30, 0.35, 0.20, 0.10, 0.05]

TRANSACTION_STATUSES = ["completed", "pending", "failed", "reversed", "held_for_review"]
TRANSACTION_STATUS_WEIGHTS = [0.88, 0.04, 0.03, 0.02, 0.03]

DEVICE_TYPES = ["desktop", "mobile", "tablet"]
DEVICE_TYPE_WEIGHTS = [0.35, 0.55, 0.10]

BROWSERS = ["Chrome", "Safari", "Firefox", "Edge", "Samsung Internet"]
BROWSER_WEIGHTS = [0.55, 0.25, 0.10, 0.07, 0.03]

OS_LIST = ["Windows 11", "macOS 14", "iOS 17", "Android 14", "Linux", "iPadOS 17"]
OS_WEIGHTS = [0.30, 0.15, 0.25, 0.20, 0.05, 0.05]


# ============================================================================
# PHASE 1: Generate Customers
# ============================================================================

def generate_customers() -> list[dict]:
    """Generate ~2,000 customer profiles.

    The fraud ring members (indices 0-17) get special treatment:
    - They register within a tight 2-week window (synthetic identity signal)
    - Their email domains cluster on 2 obscure providers
    - Their phone area codes DON'T match their stated city
    - They all list US addresses but use non-standard patterns

    Legitimate customers are spread across the full 90-day window with
    natural registration patterns.
    """
    print("  [1/8] Generating customers...")
    customers = []

    # --- Fraud ring registration window (2 weeks in early Feb) ---
    fraud_reg_start = datetime(2026, 2, 1)
    fraud_reg_end = datetime(2026, 2, 14)

    # Mismatched area codes for fraud ring (NYC area codes assigned to non-NYC cities)
    fraud_area_codes = ["212", "718", "347", "929", "646"]
    # Fraud ring members get assigned to cities that DON'T match these area codes
    fraud_cities = [
        ("Houston", "TX", "US"),
        ("Miami", "FL", "US"),
        ("Chicago", "IL", "US"),
        ("Los Angeles", "CA", "US"),
        ("San Francisco", "CA", "US"),
    ]

    for i in range(NUM_CUSTOMERS):
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        customer_id = generate_id("CUS", i)

        if i < FRAUD_RING_MEMBERS:
            # ============================================================
            # FRAUD RING MEMBER — subtle anomalies embedded here
            # ============================================================
            reg_date = random_timestamp(fraud_reg_start, fraud_reg_end)

            # Email on one of 2 obscure domains (signal #6: synthetic identity)
            email_domain = random.choice(EMAIL_DOMAINS_FRAUD)
            email = f"{first_name.lower()}.{last_name.lower()}{random.randint(1,99)}@{email_domain}"

            # Phone area code doesn't match stated city (signal #6)
            fraud_city = fraud_cities[i % len(fraud_cities)]
            area_code = random.choice(fraud_area_codes)
            phone = f"+1-{area_code}-{random.randint(100,999)}-{random.randint(1000,9999)}"

            city, state, country = fraud_city
            account_tier = random.choice(["standard", "standard", "premier"])  # mostly standard

            customer = {
                "customer_id": customer_id,
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "phone": phone,
                "city": city,
                "state": state,
                "country": country,
                "registration_date": reg_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "account_tier": account_tier,
                "account_status": "active",
                "verified_identity": random.choice([True, True, False]),  # some verified!
                "lifetime_transaction_count": random.randint(15, 80),
                "lifetime_transaction_volume": round(random.uniform(5000, 95000), 2),
                "risk_score": round(random.uniform(0.10, 0.45), 4),  # kept below alert threshold
            }
        else:
            # ============================================================
            # LEGITIMATE CUSTOMER — natural distribution
            # ============================================================
            reg_date = random_timestamp(
                START_DATE - timedelta(days=365 * 3),  # accounts up to 3 years old
                END_DATE - timedelta(days=30)
            )
            email_domain = random.choice(EMAIL_DOMAINS_LEGIT)
            email = f"{first_name.lower()}.{last_name.lower()}{random.randint(1,999)}@{email_domain}"

            city_data = random.choice(CITIES)
            city, state, country = city_data[0], city_data[1], city_data[2]

            # Phone matches geographic area (no mismatch)
            phone = f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"

            account_tier = random.choices(
                ["standard", "premier", "business"],
                weights=[0.60, 0.25, 0.15]
            )[0]

            customer = {
                "customer_id": customer_id,
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "phone": phone,
                "city": city,
                "state": state,
                "country": country,
                "registration_date": reg_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "account_tier": account_tier,
                "account_status": random.choices(
                    ["active", "active", "active", "suspended", "limited"],
                    weights=[0.85, 0.05, 0.02, 0.04, 0.04]
                )[0],
                "verified_identity": random.choices([True, False], weights=[0.80, 0.20])[0],
                "lifetime_transaction_count": random.randint(1, 500),
                "lifetime_transaction_volume": round(random.uniform(50, 500000), 2),
                "risk_score": round(random.uniform(0.01, 0.85), 4),
            }

        customers.append(customer)

    return customers


# ============================================================================
# PHASE 2: Generate Merchants
# ============================================================================

def generate_merchants() -> list[dict]:
    """Generate ~500 merchant profiles.

    The first 4 merchants are SHELL COMPANIES for the fraud ring:
    - Registered within a 10-day window (unusual clustering)
    - Names are suspiciously similar (e.g., "Apex Digital Solutions" / "Apex Digitial Svcs")
    - MCC codes subtly mismatch their transaction patterns
    - All in "Business Services" or "Computer Network Services" categories

    The remaining ~496 merchants are legitimate businesses with realistic
    profiles, diverse categories, and natural registration patterns.
    """
    print("  [2/8] Generating merchants...")
    merchants = []

    # --- Fraud shell merchants: similar names, tight registration window ---
    shell_merchant_data = [
        {
            "name": "Apex Digital Solutions LLC",
            "dba": "Apex Digital Solutions",
            "mcc": "7399",  # Business Services — but they'll process electronics-like amounts
            "city": "Wilmington",
            "state": "DE",  # Delaware — classic shell company state
            "country": "US",
            "reg_date": datetime(2026, 1, 20),
        },
        {
            "name": "Apex Digitial Svcs Inc",  # NOTE: "Digitial" is an intentional typo — subtle signal
            "dba": "Apex Digitial Services",
            "mcc": "4816",  # Computer Network Services
            "city": "Wilmington",
            "state": "DE",
            "country": "US",
            "reg_date": datetime(2026, 1, 23),
        },
        {
            "name": "Apex D.S. Consulting Group",
            "dba": "Apex DS Consulting",
            "mcc": "7399",
            "city": "Dover",
            "state": "DE",
            "country": "US",
            "reg_date": datetime(2026, 1, 26),
        },
        {
            "name": "A.D.S. Global Partners LLC",
            "dba": "ADS Global",
            "mcc": "4816",
            "city": "Newark",
            "state": "DE",
            "country": "US",
            "reg_date": datetime(2026, 1, 29),
        },
    ]

    for i, shell in enumerate(shell_merchant_data):
        merchant_id = generate_id("MER", i)
        merchants.append({
            "merchant_id": merchant_id,
            "business_name": shell["name"],
            "dba_name": shell["dba"],
            "mcc_code": shell["mcc"],
            "mcc_description": MCC_CODES[shell["mcc"]],
            "city": shell["city"],
            "state": shell["state"],
            "country": shell["country"],
            "registration_date": shell["reg_date"].strftime("%Y-%m-%d"),
            "website": f"https://www.{shell['dba'].lower().replace(' ', '')}.com",
            "verified": True,  # They passed initial verification — that's the point
            "average_ticket_size": round(random.uniform(800, 3500), 2),
            "monthly_volume": round(random.uniform(50000, 200000), 2),
            "chargeback_rate": round(random.uniform(0.005, 0.018), 4),  # kept below 2% threshold
            "account_age_days": (END_DATE - shell["reg_date"]).days,
            "industry_risk_tier": "medium",  # not flagged as high-risk
        })

    # --- Legitimate merchants ---
    for i in range(FRAUD_SHELL_MERCHANTS, NUM_MERCHANTS):
        merchant_id = generate_id("MER", i)
        mcc = random.choice(list(MCC_CODES.keys()))
        city_data = random.choice(CITIES)

        name_part1 = random.choice(MERCHANT_NAME_PARTS)
        name_part2 = random.choice(MERCHANT_NAME_TYPES)
        business_name = f"{name_part1} {name_part2}"

        reg_date = random_timestamp(
            START_DATE - timedelta(days=365 * 5),
            END_DATE - timedelta(days=60)
        )

        merchants.append({
            "merchant_id": merchant_id,
            "business_name": business_name,
            "dba_name": business_name,
            "mcc_code": mcc,
            "mcc_description": MCC_CODES[mcc],
            "city": city_data[0],
            "state": city_data[1],
            "country": city_data[2],
            "registration_date": reg_date.strftime("%Y-%m-%d"),
            "website": f"https://www.{name_part1.lower()}{name_part2.lower().replace(' ', '')}.com",
            "verified": random.choices([True, False], weights=[0.92, 0.08])[0],
            "average_ticket_size": round(random.uniform(5, 5000), 2),
            "monthly_volume": round(random.uniform(1000, 2000000), 2),
            "chargeback_rate": round(random.uniform(0.001, 0.04), 4),
            "account_age_days": (END_DATE - reg_date.replace(tzinfo=None)).days,
            "industry_risk_tier": random.choices(
                ["low", "medium", "high"],
                weights=[0.50, 0.35, 0.15]
            )[0],
        })

    return merchants


# ============================================================================
# PHASE 3: Generate Transactions
# ============================================================================

def generate_transactions(customers: list[dict], merchants: list[dict]) -> list[dict]:
    """Generate ~50,000 transactions including ~420 fraud ring transactions.

    FRAUD RING TRANSACTION PATTERNS:
    1. SMURFING: Amounts cluster at $9,200-$9,800 (below $10K reporting threshold)
       but are not round numbers — they look like real invoices.

    2. CIRCULAR FLOW: Money moves A→B→C→D→A in cycles with 48-72hr delays.
       Between ring hops, there are noise transactions to obscure the pattern.

    3. TEMPORAL COORDINATION: Ring transactions fire within 3-7 minute windows
       of each other, but only when you isolate ring-connected accounts.

    LEGITIMATE TRANSACTION PATTERNS:
    - Power-law amount distribution (many small, few large)
    - Business-hours bias with weekend dip
    - Currency distribution matching PayPal corridor volumes
    - Realistic failure/reversal rates
    """
    print("  [3/8] Generating transactions...")
    transactions = []

    fraud_customer_ids = [c["customer_id"] for c in customers[:FRAUD_RING_MEMBERS]]
    fraud_merchant_ids = [m["merchant_id"] for m in merchants[:FRAUD_SHELL_MERCHANTS]]
    legit_customer_ids = [c["customer_id"] for c in customers[FRAUD_RING_MEMBERS:]]
    legit_merchant_ids = [m["merchant_id"] for m in merchants[FRAUD_SHELL_MERCHANTS:]]

    txn_index = 0

    # ================================================================
    # FRAUD RING TRANSACTIONS — carefully structured patterns
    # ================================================================

    # Build circular flow chains: groups of 4 accounts form a ring
    # A→B→C→D→A, with each hop being a separate transaction
    ring_chains = []
    for chain_start in range(0, FRAUD_RING_MEMBERS, 4):
        chain = fraud_customer_ids[chain_start:chain_start + 4]
        if len(chain) == 4:
            ring_chains.append(chain)
        else:
            # Remaining members join existing chains
            ring_chains[0].extend(chain)

    # Generate coordinated ring transactions across the 90-day window
    # Each "burst" is a coordinated set of transactions within a 3-7 minute window
    num_bursts = FRAUD_RING_TRANSACTIONS // 6  # ~70 bursts of ~6 transactions each
    burst_dates = sorted([
        random_timestamp(START_DATE + timedelta(days=10), END_DATE - timedelta(days=5))
        for _ in range(num_bursts)
    ])

    for burst_idx, burst_time in enumerate(burst_dates):
        # Each burst: 4-8 transactions fired within a 3-7 minute window
        num_in_burst = random.randint(4, 8)
        burst_window_minutes = random.randint(3, 7)

        for j in range(num_in_burst):
            # SMURFING: amounts cluster at $9,200-$9,800 (signal #1)
            # Add cents to look like real invoices, not round numbers
            amount = round(random.uniform(9200, 9800) + random.uniform(0.01, 0.99), 2)

            # Temporal coordination: each txn offset by seconds within the burst window
            offset_seconds = random.randint(0, burst_window_minutes * 60)
            txn_time = burst_time + timedelta(seconds=offset_seconds)

            # Pick sender/receiver from ring members
            sender = random.choice(fraud_customer_ids)
            # Circular flow: receiver is the next person in the chain
            for chain in ring_chains:
                if sender in chain:
                    sender_idx = chain.index(sender)
                    receiver = chain[(sender_idx + 1) % len(chain)]
                    break
            else:
                receiver = random.choice(fraud_customer_ids)

            # Route through shell merchants
            shell_merchant = random.choice(fraud_merchant_ids)

            txn_id = generate_id("TXN", txn_index)
            transactions.append({
                "transaction_id": txn_id,
                "timestamp": txn_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sender_id": sender,
                "receiver_id": receiver,
                "merchant_id": shell_merchant,
                "amount": amount,
                "currency": "USD",  # Ring operates in USD only
                "payment_method": random.choice(["paypal_balance", "bank_transfer"]),
                "status": "completed",  # Ring transactions always complete
                "transaction_type": random.choice(["payment", "payment", "invoice"]),
                "description": random.choice([
                    "Consulting services - Q1",
                    "Digital marketing services",
                    "Software development milestone",
                    "IT infrastructure consulting",
                    "Business analytics deliverable",
                    "Cloud migration services",
                    "Data integration project",
                    "Platform development phase",
                    "Technical advisory retainer",
                    "System optimization engagement",
                ]),
                "ip_address": f"198.{random.randint(50,70)}.{random.randint(100,200)}.{random.randint(1,254)}",
                "country_code": "US",
                "risk_flag": False,  # They're evading the risk engine
                "is_international": False,
            })
            txn_index += 1

    # ================================================================
    # CIRCULAR FLOW DELAY TRANSACTIONS — the 48-72hr hops (signal #2)
    # These are the "return" legs of the circular flow, spaced out
    # to avoid detection by simple velocity rules.
    # ================================================================
    for chain in ring_chains:
        # Each chain does 8-12 full cycles over the 90-day period
        num_cycles = random.randint(8, 12)
        cycle_start = START_DATE + timedelta(days=random.randint(5, 15))

        for cycle in range(num_cycles):
            base_time = cycle_start + timedelta(days=cycle * random.randint(7, 12))
            if base_time > END_DATE - timedelta(days=5):
                break

            # A→B (day 0), B→C (day 2-3), C→D (day 4-6), D→A (day 6-9)
            for hop in range(len(chain)):
                delay_hours = random.randint(48, 72) * (hop)  # cumulative delay
                hop_time = base_time + timedelta(hours=delay_hours + random.randint(0, 12))

                if hop_time > END_DATE:
                    break

                sender = chain[hop]
                receiver = chain[(hop + 1) % len(chain)]
                shell_merchant = random.choice(fraud_merchant_ids)

                # Amounts vary slightly per hop to obscure the circle
                base_amount = random.uniform(9200, 9800)
                hop_amount = round(base_amount * random.uniform(0.97, 1.03), 2)

                txn_id = generate_id("TXN", txn_index)
                transactions.append({
                    "transaction_id": txn_id,
                    "timestamp": hop_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "sender_id": sender,
                    "receiver_id": receiver,
                    "merchant_id": shell_merchant,
                    "amount": hop_amount,
                    "currency": "USD",
                    "payment_method": random.choice(["paypal_balance", "bank_transfer"]),
                    "status": "completed",
                    "transaction_type": "payment",
                    "description": random.choice([
                        "Professional services rendered",
                        "Consulting engagement payment",
                        "Project milestone completion",
                        "Advisory services - monthly",
                        "Technical services agreement",
                    ]),
                    "ip_address": f"198.{random.randint(50,70)}.{random.randint(100,200)}.{random.randint(1,254)}",
                    "country_code": "US",
                    "risk_flag": False,
                    "is_international": False,
                })
                txn_index += 1

    num_fraud_txns = txn_index  # Track how many fraud transactions we generated
    print(f"         → {num_fraud_txns} fraud ring transactions embedded")

    # ================================================================
    # LEGITIMATE TRANSACTIONS — the haystack
    # ================================================================
    num_legit = NUM_TRANSACTIONS - num_fraud_txns

    for i in range(num_legit):
        txn_id = generate_id("TXN", txn_index)
        txn_time = random_timestamp(START_DATE, END_DATE)

        sender = random.choice(legit_customer_ids)
        receiver = random.choice(legit_customer_ids + fraud_customer_ids[:3])  # some interact with ring edges
        merchant = random.choice(legit_merchant_ids)

        # Power-law distribution: many small transactions, few large ones
        # This makes the ring's $9,200-$9,800 range blend in with legitimate high-value txns
        amount_tier = random.random()
        if amount_tier < 0.40:
            amount = round(random.uniform(1.00, 50.00), 2)
        elif amount_tier < 0.70:
            amount = round(random.uniform(50.00, 200.00), 2)
        elif amount_tier < 0.85:
            amount = round(random.uniform(200.00, 1000.00), 2)
        elif amount_tier < 0.95:
            amount = round(random.uniform(1000.00, 5000.00), 2)
        elif amount_tier < 0.99:
            amount = round(random.uniform(5000.00, 15000.00), 2)
        else:
            amount = round(random.uniform(15000.00, 100000.00), 2)

        currency = random.choices(CURRENCIES, weights=CURRENCY_WEIGHTS)[0]
        country_data = random.choice(CITIES)

        is_international = random.random() < 0.25

        transactions.append({
            "transaction_id": txn_id,
            "timestamp": txn_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sender_id": sender,
            "receiver_id": receiver,
            "merchant_id": merchant,
            "amount": amount,
            "currency": currency,
            "payment_method": random.choices(PAYMENT_METHODS, weights=PAYMENT_METHOD_WEIGHTS)[0],
            "status": random.choices(TRANSACTION_STATUSES, weights=TRANSACTION_STATUS_WEIGHTS)[0],
            "transaction_type": random.choices(
                ["payment", "refund", "transfer", "invoice", "subscription"],
                weights=[0.55, 0.10, 0.15, 0.12, 0.08]
            )[0],
            "description": random.choice([
                "Online purchase", "Monthly subscription", "Freelance payment",
                "Invoice settlement", "Refund processed", "Peer transfer",
                "Marketplace purchase", "Service payment", "Bill payment",
                "Digital goods", "Physical goods shipment", "Donation",
                "Rental payment", "Insurance premium", "Utility payment",
            ]),
            "ip_address": f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
            "country_code": country_data[2],
            "risk_flag": random.random() < 0.03,  # 3% flagged by rules engine
            "is_international": is_international,
        })
        txn_index += 1

    # Shuffle so fraud transactions aren't at the top of the file
    random.shuffle(transactions)
    return transactions


# ============================================================================
# PHASE 4: Generate Payment Sessions (JSON — nested structure)
# ============================================================================

def generate_payment_sessions(customers: list[dict], transactions: list[dict]) -> list[dict]:
    """Generate ~15,000 payment session records with nested device/geo data.

    FRAUD RING DEVICE SIGNALS:
    - 6 of the 18 ring members share 3 device fingerprints (signal #4)
      This means 2 accounts per shared device — hidden in nested JSON.
    - Some ring members show "impossible travel" (signal #5):
      transactions from Lagos and London within 20 minutes.

    The nested JSON structure makes this data harder to analyze with
    simple tabular tools — an LLM excels at navigating nested structures.
    """
    print("  [4/8] Generating payment sessions...")
    sessions = []

    fraud_customer_ids = [c["customer_id"] for c in customers[:FRAUD_RING_MEMBERS]]

    # --- Shared device fingerprints for fraud ring (signal #4) ---
    # 3 devices shared across 6 ring members (2 members per device)
    shared_devices = {
        fraud_customer_ids[0]: generate_device_fingerprint(9000),
        fraud_customer_ids[1]: generate_device_fingerprint(9000),  # SAME device as [0]
        fraud_customer_ids[2]: generate_device_fingerprint(9001),
        fraud_customer_ids[3]: generate_device_fingerprint(9001),  # SAME device as [2]
        fraud_customer_ids[4]: generate_device_fingerprint(9002),
        fraud_customer_ids[5]: generate_device_fingerprint(9002),  # SAME device as [4]
    }

    # Impossible travel pairs (signal #5)
    impossible_travel_pairs = [
        (fraud_customer_ids[6], "Lagos", "London", 15),    # 15 min apart
        (fraud_customer_ids[7], "Lagos", "London", 18),    # 18 min apart
        (fraud_customer_ids[8], "Mumbai", "Singapore", 12), # 12 min apart
    ]

    # Generate impossible travel sessions first
    for customer_id, city1, city2, gap_minutes in impossible_travel_pairs:
        base_time = random_timestamp(START_DATE + timedelta(days=20), END_DATE - timedelta(days=20))

        city1_data = next(c for c in CITIES if c[0] == city1)
        city2_data = next(c for c in CITIES if c[0] == city2)

        # Session 1: City A
        session_id_1 = generate_id("SES", len(sessions))
        sessions.append({
            "session_id": session_id_1,
            "customer_id": customer_id,
            "timestamp": base_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "session_duration_seconds": random.randint(30, 180),
            "device": {
                "fingerprint": shared_devices.get(customer_id, generate_device_fingerprint(hash(customer_id) % 10000)),
                "type": random.choices(DEVICE_TYPES, weights=DEVICE_TYPE_WEIGHTS)[0],
                "browser": random.choices(BROWSERS, weights=BROWSER_WEIGHTS)[0],
                "os": random.choices(OS_LIST, weights=OS_WEIGHTS)[0],
                "screen_resolution": random.choice(["1920x1080", "2560x1440", "1366x768", "390x844"]),
                "language": "en-US",
                "timezone_offset": random.choice([-5, -6, -8, 0, 1, 5.5, 8, 9]),
            },
            "geolocation": {
                "city": city1,
                "country": city1_data[2],
                "latitude": round(city1_data[3] + random.uniform(-0.05, 0.05), 4),
                "longitude": round(city1_data[4] + random.uniform(-0.05, 0.05), 4),
                "ip_geolocation_match": True,
                "vpn_detected": False,
            },
            "actions": [
                {"type": "login", "timestamp_offset_ms": 0},
                {"type": "view_balance", "timestamp_offset_ms": random.randint(1000, 5000)},
                {"type": "initiate_payment", "timestamp_offset_ms": random.randint(5000, 15000)},
                {"type": "confirm_payment", "timestamp_offset_ms": random.randint(15000, 30000)},
            ],
            "risk_assessment": {
                "session_risk_score": round(random.uniform(0.15, 0.40), 4),
                "anomaly_flags": [],
                "velocity_check_passed": True,
            },
        })

        # Session 2: City B — impossibly soon after Session 1
        session_time_2 = base_time + timedelta(minutes=gap_minutes)
        session_id_2 = generate_id("SES", len(sessions))
        sessions.append({
            "session_id": session_id_2,
            "customer_id": customer_id,
            "timestamp": session_time_2.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "session_duration_seconds": random.randint(30, 180),
            "device": {
                "fingerprint": generate_device_fingerprint(hash(customer_id + "travel") % 10000),
                "type": random.choices(DEVICE_TYPES, weights=DEVICE_TYPE_WEIGHTS)[0],
                "browser": random.choices(BROWSERS, weights=BROWSER_WEIGHTS)[0],
                "os": random.choices(OS_LIST, weights=OS_WEIGHTS)[0],
                "screen_resolution": random.choice(["1920x1080", "2560x1440", "1366x768", "390x844"]),
                "language": "en-US",
                "timezone_offset": random.choice([-5, -6, -8, 0, 1, 5.5, 8, 9]),
            },
            "geolocation": {
                "city": city2,
                "country": city2_data[2],
                "latitude": round(city2_data[3] + random.uniform(-0.05, 0.05), 4),
                "longitude": round(city2_data[4] + random.uniform(-0.05, 0.05), 4),
                "ip_geolocation_match": True,  # IP also geolocates to city 2 — not a VPN
                "vpn_detected": False,
            },
            "actions": [
                {"type": "login", "timestamp_offset_ms": 0},
                {"type": "initiate_payment", "timestamp_offset_ms": random.randint(2000, 8000)},
                {"type": "confirm_payment", "timestamp_offset_ms": random.randint(8000, 20000)},
            ],
            "risk_assessment": {
                "session_risk_score": round(random.uniform(0.20, 0.45), 4),
                "anomaly_flags": [],  # The risk engine missed it
                "velocity_check_passed": True,
            },
        })

    # --- Regular sessions (both fraud ring members and legitimate customers) ---
    all_customer_ids = [c["customer_id"] for c in customers]

    for i in range(len(sessions), NUM_PAYMENT_SESSIONS):
        session_id = generate_id("SES", i)
        customer_id = random.choice(all_customer_ids)
        session_time = random_timestamp(START_DATE, END_DATE)
        city_data = random.choice(CITIES)

        # If this is a fraud ring member, sometimes use the shared device fingerprint
        if customer_id in shared_devices and random.random() < 0.6:
            device_fp = shared_devices[customer_id]
        else:
            device_fp = generate_device_fingerprint(hash(customer_id + str(i)) % 100000)

        num_actions = random.randint(1, 8)
        action_types = ["login", "view_balance", "view_history", "initiate_payment",
                        "confirm_payment", "cancel_payment", "update_profile",
                        "add_payment_method", "view_notifications"]
        actions = []
        offset = 0
        for _ in range(num_actions):
            offset += random.randint(500, 10000)
            actions.append({
                "type": random.choice(action_types),
                "timestamp_offset_ms": offset,
            })

        vpn_detected = random.random() < 0.08
        ip_match = not vpn_detected and random.random() > 0.05

        sessions.append({
            "session_id": session_id,
            "customer_id": customer_id,
            "timestamp": session_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "session_duration_seconds": random.randint(5, 1800),
            "device": {
                "fingerprint": device_fp,
                "type": random.choices(DEVICE_TYPES, weights=DEVICE_TYPE_WEIGHTS)[0],
                "browser": random.choices(BROWSERS, weights=BROWSER_WEIGHTS)[0],
                "os": random.choices(OS_LIST, weights=OS_WEIGHTS)[0],
                "screen_resolution": random.choice([
                    "1920x1080", "2560x1440", "3840x2160", "1366x768",
                    "390x844", "412x915", "360x800", "1024x768",
                ]),
                "language": random.choice(["en-US", "en-GB", "de-DE", "fr-FR", "ja-JP",
                                            "pt-BR", "es-MX", "zh-CN", "hi-IN", "ar-SA"]),
                "timezone_offset": random.choice([-8, -7, -6, -5, -4, 0, 1, 2, 3, 5.5, 8, 9, 10]),
            },
            "geolocation": {
                "city": city_data[0],
                "country": city_data[2],
                "latitude": round(city_data[3] + random.uniform(-0.1, 0.1), 4),
                "longitude": round(city_data[4] + random.uniform(-0.1, 0.1), 4),
                "ip_geolocation_match": ip_match,
                "vpn_detected": vpn_detected,
            },
            "actions": actions,
            "risk_assessment": {
                "session_risk_score": round(random.uniform(0.01, 0.95), 4),
                "anomaly_flags": random.sample(
                    ["unusual_time", "new_device", "high_velocity", "geo_mismatch",
                     "bot_pattern", "rapid_clicks"],
                    k=random.choices([0, 0, 0, 1, 2], weights=[0.6, 0.1, 0.1, 0.15, 0.05])[0]
                ),
                "velocity_check_passed": random.random() > 0.05,
            },
        })

    random.shuffle(sessions)
    return sessions


# ============================================================================
# PHASE 5: Generate Dispute Cases (JSON)
# ============================================================================

def generate_dispute_cases(customers: list[dict], transactions: list[dict]) -> list[dict]:
    """Generate ~350 dispute/chargeback cases with evidence threads.

    Some disputes involve fraud ring transactions — but the ring members
    file very FEW disputes (they don't want attention). The disputes that
    DO exist against shell merchants come from legitimate customers who
    noticed unauthorized charges.
    """
    print("  [5/8] Generating dispute cases...")
    disputes = []

    fraud_merchant_ids = set(generate_id("MER", i) for i in range(FRAUD_SHELL_MERCHANTS))

    dispute_reasons = [
        "unauthorized_transaction", "item_not_received", "item_significantly_not_as_described",
        "duplicate_charge", "subscription_cancelled", "credit_not_processed",
        "merchandise_defective", "services_not_rendered",
    ]
    dispute_reason_weights = [0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05]

    resolution_outcomes = ["resolved_buyer", "resolved_seller", "escalated", "pending", "withdrawn"]
    resolution_weights = [0.35, 0.25, 0.15, 0.15, 0.10]

    # Pick random transactions to be disputed
    completed_txns = [t for t in transactions if t["status"] == "completed"]
    disputed_txns = random.sample(completed_txns, min(NUM_DISPUTE_CASES, len(completed_txns)))

    for i, txn in enumerate(disputed_txns):
        dispute_id = generate_id("DSP", i)
        filed_date = datetime.strptime(txn["timestamp"], "%Y-%m-%dT%H:%M:%SZ") + timedelta(
            days=random.randint(1, 30)
        )

        is_against_shell = txn["merchant_id"] in fraud_merchant_ids

        if is_against_shell:
            # Disputes against shell merchants are from victims noticing unauthorized charges
            reason = "unauthorized_transaction"
            description = random.choice([
                "I did not authorize this transaction. I have never heard of this company.",
                "This charge appeared on my account without my knowledge. I did not make this purchase.",
                "I don't recognize this merchant or this transaction. Please investigate.",
                "Unauthorized charge. I have never done business with this company.",
            ])
        else:
            reason = random.choices(dispute_reasons, weights=dispute_reason_weights)[0]
            description = random.choice([
                "The item I received was completely different from what was advertised.",
                "I was charged twice for the same transaction.",
                "I cancelled my subscription but was still charged.",
                "The service was never delivered despite payment.",
                "The product arrived damaged and the seller won't respond.",
                "I returned the item but never received my refund.",
                "The quality was far below what was promised.",
                "Merchant is unresponsive to my refund request.",
            ])

        # Evidence thread — messages between buyer, seller, and PayPal
        evidence = []
        evidence.append({
            "type": "buyer_complaint",
            "timestamp": filed_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "author": "buyer",
            "content": description,
        })

        if random.random() < 0.7:
            evidence.append({
                "type": "seller_response",
                "timestamp": (filed_date + timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "author": "seller",
                "content": random.choice([
                    "The service was delivered as agreed. We have records of the engagement.",
                    "We are looking into this matter and will respond shortly.",
                    "This transaction was authorized. We can provide documentation.",
                    "We apologize for the inconvenience. A refund has been initiated.",
                    "We dispute this claim. The product was delivered on time.",
                ]),
            })

        if random.random() < 0.4:
            evidence.append({
                "type": "paypal_review",
                "timestamp": (filed_date + timedelta(days=random.randint(5, 15))).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "author": "paypal_agent",
                "content": random.choice([
                    "After reviewing the evidence, we have decided in favor of the buyer.",
                    "Additional documentation has been requested from both parties.",
                    "This case has been escalated for further review.",
                    "The seller has provided sufficient evidence. Case resolved in seller's favor.",
                ]),
            })

        disputes.append({
            "dispute_id": dispute_id,
            "transaction_id": txn["transaction_id"],
            "customer_id": txn["sender_id"],
            "merchant_id": txn["merchant_id"],
            "amount": txn["amount"],
            "currency": txn["currency"],
            "reason": reason,
            "description": description,
            "filed_date": filed_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "resolution": random.choices(resolution_outcomes, weights=resolution_weights)[0],
            "resolution_date": (filed_date + timedelta(days=random.randint(5, 45))).strftime("%Y-%m-%dT%H:%M:%SZ")
            if random.random() < 0.75 else None,
            "evidence_thread": evidence,
            "escalation_level": random.choices([1, 2, 3], weights=[0.65, 0.25, 0.10])[0],
        })

    return disputes


# ============================================================================
# PHASE 6: Generate Risk Signals (JSON)
# ============================================================================

def generate_risk_signals(transactions: list[dict]) -> list[dict]:
    """Generate ~8,000 risk engine scoring breakdowns.

    The risk engine produces detailed scoring for flagged or sampled
    transactions. CRITICALLY: fraud ring transactions have LOW risk scores
    because the ring has learned to evade the rules engine. This is the
    whole point — the rules-based system misses them, but an LLM can
    find them through cross-referencing.
    """
    print("  [6/8] Generating risk signals...")
    signals = []

    # Sample transactions for risk scoring
    sampled_txns = random.sample(transactions, min(NUM_RISK_SIGNALS, len(transactions)))

    fraud_customer_ids = set(generate_id("CUS", i) for i in range(FRAUD_RING_MEMBERS))

    risk_rules = [
        "velocity_check", "amount_threshold", "geo_anomaly", "device_reputation",
        "account_age_check", "behavioral_pattern", "blacklist_check", "ml_model_v3",
        "network_analysis", "time_pattern_check",
    ]

    for i, txn in enumerate(sampled_txns):
        signal_id = generate_id("RSK", i)
        is_fraud_ring = txn["sender_id"] in fraud_customer_ids

        if is_fraud_ring:
            # Fraud ring transactions score LOW — the ring evades the rules
            overall_score = round(random.uniform(0.08, 0.35), 4)
            rule_results = {}
            for rule in risk_rules:
                # Most rules pass; a couple are borderline but below threshold
                if rule == "amount_threshold":
                    rule_results[rule] = {
                        "score": round(random.uniform(0.20, 0.45), 4),  # elevated but not flagged
                        "triggered": False,
                        "details": "Amount within normal range for merchant category",
                    }
                elif rule == "velocity_check":
                    rule_results[rule] = {
                        "score": round(random.uniform(0.15, 0.38), 4),
                        "triggered": False,
                        "details": "Transaction velocity within acceptable limits",
                    }
                elif rule == "network_analysis":
                    # This rule ALMOST catches them — score is elevated
                    rule_results[rule] = {
                        "score": round(random.uniform(0.30, 0.48), 4),  # close to threshold
                        "triggered": False,
                        "details": "Minor network density elevation — within tolerance",
                    }
                else:
                    rule_results[rule] = {
                        "score": round(random.uniform(0.05, 0.25), 4),
                        "triggered": False,
                        "details": "No anomalies detected",
                    }
        else:
            # Legitimate transactions — natural distribution of risk scores
            overall_score = round(random.uniform(0.01, 0.95), 4)
            rule_results = {}
            for rule in risk_rules:
                rule_score = round(random.uniform(0.01, 0.90), 4)
                rule_results[rule] = {
                    "score": rule_score,
                    "triggered": rule_score > 0.7,
                    "details": "Anomaly detected" if rule_score > 0.7 else "No anomalies detected",
                }

        signals.append({
            "signal_id": signal_id,
            "transaction_id": txn["transaction_id"],
            "customer_id": txn["sender_id"],
            "merchant_id": txn["merchant_id"],
            "timestamp": txn["timestamp"],
            "overall_risk_score": overall_score,
            "decision": "approve" if overall_score < 0.5 else random.choice(["approve", "review", "decline"]),
            "rule_results": rule_results,
            "model_version": "risk_engine_v3.2.1",
            "processing_time_ms": random.randint(12, 250),
        })

    return signals


# ============================================================================
# PHASE 7: Generate Velocity Metrics (Time-Series CSV)
# ============================================================================

def generate_velocity_metrics(transactions: list[dict]) -> list[dict]:
    """Generate hourly time-series velocity metrics.

    This creates aggregated metrics per hour:
    - Transaction counts, total amounts, unique senders/receivers
    - Average transaction size, max single transaction
    - Cross-border transaction percentage

    FRAUD SIGNAL IN TIME SERIES:
    The fraud ring creates subtle spikes in the $9K-$10K band during
    their coordinated burst windows. These spikes are only visible when
    you segment by amount range — the overall volume looks normal.
    """
    print("  [7/8] Generating velocity metrics...")
    metrics = []

    # Aggregate transactions by hour
    hourly_buckets = {}
    for txn in transactions:
        ts = datetime.strptime(txn["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
        hour_key = ts.replace(minute=0, second=0)
        if hour_key not in hourly_buckets:
            hourly_buckets[hour_key] = []
        hourly_buckets[hour_key].append(txn)

    # Fill in empty hours
    current = START_DATE.replace(minute=0, second=0)
    while current <= END_DATE:
        if current not in hourly_buckets:
            hourly_buckets[current] = []
        current += timedelta(hours=1)

    for hour in sorted(hourly_buckets.keys()):
        txns = hourly_buckets[hour]
        if not txns:
            metrics.append({
                "timestamp": hour.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "transaction_count": 0,
                "total_amount_usd": 0.0,
                "avg_amount_usd": 0.0,
                "max_amount_usd": 0.0,
                "unique_senders": 0,
                "unique_receivers": 0,
                "unique_merchants": 0,
                "cross_border_pct": 0.0,
                "high_value_count": 0,
                "high_value_total_usd": 0.0,
                "failed_transaction_count": 0,
                "reversal_count": 0,
            })
        else:
            amounts = [t["amount"] for t in txns]
            high_value = [t for t in txns if t["amount"] > 9000]
            failed = [t for t in txns if t["status"] == "failed"]
            reversed_txns = [t for t in txns if t["status"] == "reversed"]
            international = [t for t in txns if t["is_international"]]

            metrics.append({
                "timestamp": hour.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "transaction_count": len(txns),
                "total_amount_usd": round(sum(amounts), 2),
                "avg_amount_usd": round(np.mean(amounts), 2),
                "max_amount_usd": round(max(amounts), 2),
                "unique_senders": len(set(t["sender_id"] for t in txns)),
                "unique_receivers": len(set(t["receiver_id"] for t in txns)),
                "unique_merchants": len(set(t["merchant_id"] for t in txns)),
                "cross_border_pct": round(len(international) / len(txns) * 100, 2),
                "high_value_count": len(high_value),
                "high_value_total_usd": round(sum(t["amount"] for t in high_value), 2),
                "failed_transaction_count": len(failed),
                "reversal_count": len(reversed_txns),
            })

    return metrics


# ============================================================================
# PHASE 8: Generate System Telemetry (Time-Series CSV)
# ============================================================================

def generate_system_telemetry() -> list[dict]:
    """Generate platform health metrics — latency, error rates, throughput.

    This data helps participants understand the operational context:
    - API response times for payment processing
    - Error rates by service
    - Throughput metrics

    This dataset supports Module 06 (Observability Dashboard) and helps
    participants correlate system behavior with fraud patterns.
    """
    print("  [8/8] Generating system telemetry...")
    telemetry = []

    services = [
        "payment_processor", "risk_engine", "auth_service",
        "merchant_api", "notification_service", "ledger_service",
    ]

    current = START_DATE.replace(minute=0, second=0)
    while current <= END_DATE:
        for service in services:
            # Base metrics with time-of-day variation
            hour = current.hour
            is_business_hours = 8 <= hour <= 20
            is_weekend = current.weekday() >= 5

            base_throughput = 1200 if is_business_hours and not is_weekend else 400
            throughput = max(0, int(base_throughput + np.random.normal(0, base_throughput * 0.15)))

            # Latency: payment_processor is slowest, auth_service is fastest
            latency_base = {
                "payment_processor": 145,
                "risk_engine": 89,
                "auth_service": 23,
                "merchant_api": 67,
                "notification_service": 34,
                "ledger_service": 112,
            }
            latency = max(1, round(latency_base[service] + np.random.normal(0, latency_base[service] * 0.2), 1))

            # Occasional latency spikes (simulating real production behavior)
            if random.random() < 0.02:
                latency *= random.uniform(3, 8)
                latency = round(latency, 1)

            error_rate = max(0, round(np.random.exponential(0.5), 3))
            if random.random() < 0.01:  # Occasional error spike
                error_rate = round(random.uniform(5, 15), 3)

            telemetry.append({
                "timestamp": current.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "service": service,
                "requests_per_minute": throughput,
                "avg_latency_ms": latency,
                "p99_latency_ms": round(latency * random.uniform(2.5, 5.0), 1),
                "error_rate_pct": error_rate,
                "error_count": int(throughput * error_rate / 100),
                "success_count": int(throughput * (1 - error_rate / 100)),
                "cpu_utilization_pct": round(min(100, max(5, 30 + throughput / 50 + np.random.normal(0, 10))), 1),
                "memory_utilization_pct": round(min(95, max(20, 55 + np.random.normal(0, 8))), 1),
                "active_connections": max(1, int(throughput * random.uniform(0.5, 1.5))),
            })

        current += timedelta(hours=1)

    return telemetry


# ============================================================================
# FILE WRITERS
# ============================================================================

def write_csv(data: list[dict], filename: str):
    """Write a list of dicts to CSV."""
    filepath = OUTPUT_DIR / filename
    if not data:
        print(f"    ⚠ No data to write for {filename}")
        return
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"    ✓ {filepath} ({len(data):,} records)")


def write_json(data: list[dict], filename: str):
    """Write a list of dicts to JSON with readable formatting."""
    filepath = OUTPUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"    ✓ {filepath} ({len(data):,} records)")


# ============================================================================
# GROUND TRUTH — Answer key for the evaluation framework (Module 05)
# ============================================================================

def write_ground_truth(customers: list[dict], merchants: list[dict]):
    """Write the ground truth file that Module 05 uses to evaluate LLM accuracy.

    This file is the 'answer key' — it contains the actual fraud ring member IDs
    and shell merchant IDs. In a real workshop, the facilitator can decide
    whether to share this with participants or keep it hidden.
    """
    fraud_customer_ids = [c["customer_id"] for c in customers[:FRAUD_RING_MEMBERS]]
    fraud_merchant_ids = [m["merchant_id"] for m in merchants[:FRAUD_SHELL_MERCHANTS]]

    ground_truth = {
        "_description": "FACILITATOR ONLY — Ground truth for fraud ring identification",
        "_warning": "Do not share with participants until after Module 05",
        "fraud_ring": {
            "member_customer_ids": fraud_customer_ids,
            "shell_merchant_ids": fraud_merchant_ids,
            "total_members": FRAUD_RING_MEMBERS,
            "total_shell_merchants": FRAUD_SHELL_MERCHANTS,
            "signals": [
                "smurfing_amount_range_9200_9800",
                "circular_money_flow_48_72hr_delays",
                "shell_merchant_name_clustering",
                "device_fingerprint_sharing_6_accounts_3_devices",
                "impossible_travel_lagos_london_mumbai_singapore",
                "synthetic_identity_registration_clustering",
                "temporal_coordination_3_7_minute_windows",
            ],
            "device_sharing_pairs": [
                [fraud_customer_ids[0], fraud_customer_ids[1]],
                [fraud_customer_ids[2], fraud_customer_ids[3]],
                [fraud_customer_ids[4], fraud_customer_ids[5]],
            ],
            "impossible_travel_accounts": [
                fraud_customer_ids[6],
                fraud_customer_ids[7],
                fraud_customer_ids[8],
            ],
            "circular_flow_chains": [
                fraud_customer_ids[0:4],
                fraud_customer_ids[4:8],
                fraud_customer_ids[8:12],
                fraud_customer_ids[12:16],
            ],
        },
    }

    filepath = OUTPUT_DIR / "ground_truth.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"    ✓ {filepath} (answer key)")


# ============================================================================
# MAIN — Orchestrate all data generation
# ============================================================================

def main():
    print("=" * 70)
    print("PayPal x Azure AI Workshop — Synthetic Data Generator")
    print("=" * 70)
    print(f"  Seed: {SEED}")
    print(f"  Date range: {START_DATE.date()} → {END_DATE.date()}")
    print(f"  Customers: {NUM_CUSTOMERS:,} ({FRAUD_RING_MEMBERS} fraud ring)")
    print(f"  Merchants: {NUM_MERCHANTS:,} ({FRAUD_SHELL_MERCHANTS} shell companies)")
    print(f"  Transactions: ~{NUM_TRANSACTIONS:,}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)
    print()

    # Phase 1-2: Generate reference data
    customers = generate_customers()
    merchants = generate_merchants()

    # Phase 3: Generate transactions (depends on customers + merchants)
    transactions = generate_transactions(customers, merchants)

    # Phase 4-6: Generate supporting datasets (depend on customers + transactions)
    sessions = generate_payment_sessions(customers, transactions)
    disputes = generate_dispute_cases(customers, transactions)
    risk_signals = generate_risk_signals(transactions)

    # Phase 7-8: Generate time-series data
    velocity = generate_velocity_metrics(transactions)
    telemetry = generate_system_telemetry()

    # Write all files
    print()
    print("Writing datasets:")
    write_csv(customers, "customers.csv")
    write_csv(merchants, "merchants.csv")
    write_csv(transactions, "transactions.csv")
    write_json(sessions, "payment_sessions.json")
    write_json(disputes, "dispute_cases.json")
    write_json(risk_signals, "risk_signals.json")
    write_csv(velocity, "velocity_metrics.csv")
    write_csv(telemetry, "system_telemetry.csv")
    write_ground_truth(customers, merchants)

    print()
    print("=" * 70)
    print("Data generation complete!")
    print(f"  Total fraud ring transactions: ~{len([t for t in transactions if t['sender_id'] in set(c['customer_id'] for c in customers[:FRAUD_RING_MEMBERS])])}")
    print(f"  Fraud ring percentage: ~{len([t for t in transactions if t['sender_id'] in set(c['customer_id'] for c in customers[:FRAUD_RING_MEMBERS])]) / len(transactions) * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
