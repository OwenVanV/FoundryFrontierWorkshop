"""
============================================================================
SEARCH+RAG MODULE 01 — Content Understanding: Create Custom Analyzer
============================================================================

WORKSHOP NARRATIVE:
    The fraud investigation team has a pile of internal documents — SARs,
    compliance memos, merchant due diligence files, audit logs, onboarding
    summaries, and security alerts. These are unstructured text, not the
    clean CSVs from the LLM track.

    Before we can search this evidence, we need to UNDERSTAND it. Azure
    Content Understanding creates a custom analyzer that extracts
    fraud-relevant fields from any document: merchant names, account IDs,
    transaction amounts, risk indicators, and a generated summary.

    This module creates the analyzer. Module 02 runs it on the documents.

LEARNING OBJECTIVES:
    1. Understand Azure Content Understanding's analyzer concept
    2. Create a custom analyzer with a field extraction schema
    3. Define extract, generate, and classify field methods
    4. Configure confidence scoring and source grounding

AZURE SERVICES:
    - Azure Content Understanding (GA, api-version 2025-11-01)

ESTIMATED TIME: 10-15 minutes

============================================================================
"""

import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
_dir = Path(__file__).parent
load_dotenv(_dir / ".env")
load_dotenv()

from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentunderstanding import ContentUnderstandingClient

# ============================================================================
# CONFIGURATION
# ============================================================================
ENDPOINT = os.getenv("CONTENT_UNDERSTANDING_ENDPOINT")
KEY = os.getenv("CONTENT_UNDERSTANDING_KEY")
USE_MI = os.getenv("USE_MANAGED_IDENTITY", "false").lower() in ("true", "1")
ANALYZER_ID = "fraud_evidence_analyzer"
IMAGE_ANALYZER_ID = "fraud_evidence_image_analyzer"

if not ENDPOINT:
    print("ERROR: Set CONTENT_UNDERSTANDING_ENDPOINT in search_rag/.env")
    sys.exit(1)


def get_client():
    """Create the Content Understanding client with appropriate auth."""
    if USE_MI:
        return ContentUnderstandingClient(endpoint=ENDPOINT, credential=DefaultAzureCredential())
    else:
        if not KEY:
            print("ERROR: Set CONTENT_UNDERSTANDING_KEY or USE_MANAGED_IDENTITY=true")
            sys.exit(1)
        return ContentUnderstandingClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))


# ============================================================================
# STEP 1: Define the Fraud Evidence Analyzer Schema
# ============================================================================
# The field schema tells Content Understanding WHAT to extract from each
# document. We use three methods:
#
#   extract — Pull values directly from the text (merchant names, IDs, amounts)
#   classify — Categorize the document type from a fixed set
#   generate — Use the LLM to synthesize (summaries, risk indicators)
#
# This is the key difference from raw OCR/text extraction: Content
# Understanding uses GPT-4.1 to reason about the content and extract
# semantically meaningful fields.
# ============================================================================

ANALYZER_SCHEMA = {
    "description": "Extracts fraud investigation evidence from internal payment processor documents",
    "baseAnalyzerId": "prebuilt-document",
    "models": {
        "completion": "gpt-4.1",
    },
    "config": {
        "returnDetails": True,
        "estimateFieldSourceAndConfidence": True,
    },
    "fieldSchema": {
        "fields": {
            "document_type": {
                "type": "string",
                "method": "classify",
                "description": "Document type classification",
                "enum": [
                    "suspicious_activity_report",
                    "compliance_memo",
                    "due_diligence",
                    "audit_log",
                    "onboarding_summary",
                    "security_alert",
                    "risk_report",
                    "dispute_summary",
                    "other",
                ],
            },
            "document_date": {
                "type": "date",
                "method": "extract",
                "description": "The primary date of the document (filing date, memo date, report date)",
            },
            "author": {
                "type": "string",
                "method": "extract",
                "description": "The person who authored or filed the document",
            },
            "priority_level": {
                "type": "string",
                "method": "classify",
                "description": "Priority or severity level",
                "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL", "INFO"],
            },
            "merchant_names": {
                "type": "array",
                "method": "extract",
                "items": {"type": "string"},
                "description": "All merchant or company names mentioned in the document",
            },
            "merchant_ids": {
                "type": "array",
                "method": "extract",
                "items": {"type": "string"},
                "description": "All merchant IDs (MER-XXXX format) mentioned",
            },
            "customer_ids": {
                "type": "array",
                "method": "extract",
                "items": {"type": "string"},
                "description": "All customer account IDs (CUS-XXXX format) mentioned",
            },
            "transaction_amounts": {
                "type": "array",
                "method": "extract",
                "items": {"type": "number"},
                "description": "Dollar amounts mentioned in connection with transactions",
            },
            "risk_indicators": {
                "type": "array",
                "method": "generate",
                "items": {"type": "string"},
                "description": "Specific fraud risk indicators identified in the document (e.g., 'structuring below BSA threshold', 'shared device fingerprints', 'impossible travel')",
            },
            "recommended_actions": {
                "type": "array",
                "method": "generate",
                "items": {"type": "string"},
                "description": "Actions recommended by the document author",
            },
            "summary": {
                "type": "string",
                "method": "generate",
                "description": "A concise one-paragraph summary of the document's key findings and their relevance to fraud investigation",
            },
        },
    },
}


def main():
    print("=" * 70)
    print("SEARCH+RAG MODULE 01 — Create Custom Analyzer")
    print("=" * 70)
    print()

    client = get_client()
    print(f"  ✓ Connected to: {ENDPOINT}")
    print()

    # ================================================================
    # Step 0: Configure default model deployments
    # ================================================================
    # Content Understanding requires default model mappings before you
    # can create analyzers. This is a one-time setup per resource.
    # ================================================================
    print("  Configuring default model deployments...")
    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    try:
        defaults = client.update_defaults(
            model_deployments={
                "gpt-4.1": "gpt-4.1",
                "gpt-4.1-mini": "gpt-4.1-mini",
                "text-embedding-3-large": embedding_deployment,
            }
        )
        print(f"  ✓ Default models configured (embedding: {embedding_deployment})")
    except Exception as e:
        print(f"  ⚠ Could not set defaults (may already be configured): {e}")
    print()

    # ================================================================
    # Create (or replace) the custom analyzer
    # ================================================================
    print("  Creating custom fraud evidence analyzer...")
    print(f"  Analyzer ID: {ANALYZER_ID}")
    print(f"  Base: prebuilt-document")
    print(f"  Fields: {len(ANALYZER_SCHEMA['fieldSchema']['fields'])}")
    for name, field in ANALYZER_SCHEMA["fieldSchema"]["fields"].items():
        print(f"    • {name} ({field['type']}, method={field['method']})")
    print()

    try:
        poller = client.begin_create_analyzer(
            analyzer_id=ANALYZER_ID,
            resource=ANALYZER_SCHEMA,
            allow_replace=True,
        )
        result = poller.result()
        print(f"  ✓ Text analyzer created: {ANALYZER_ID}")
    except Exception as e:
        print(f"  ✗ Error creating text analyzer: {e}")
        sys.exit(1)

    # ================================================================
    # Create the IMAGE analyzer (same fields, image base)
    # ================================================================
    # Content Understanding uses prebuilt-image as the base for image
    # documents. Image analyzers only support 'generate' and 'classify'
    # methods — NOT 'extract'. So we copy the schema and convert all
    # 'extract' fields to 'generate'.
    # ================================================================
    print()
    print("  Creating image fraud evidence analyzer...")
    print(f"  Analyzer ID: {IMAGE_ANALYZER_ID}")
    print(f"  Base: prebuilt-image")

    # Build image field schema: same fields but extract → generate
    import copy
    image_fields = copy.deepcopy(ANALYZER_SCHEMA["fieldSchema"]["fields"])
    for field_name, field_def in image_fields.items():
        if field_def.get("method") == "extract":
            field_def["method"] = "generate"

    IMAGE_ANALYZER_SCHEMA = {
        "description": "Extracts fraud evidence from document images — screenshots, scanned documents, photos of notes",
        "baseAnalyzerId": "prebuilt-image",
        "models": {
            "completion": "gpt-4.1",
        },
        "fieldSchema": {"fields": image_fields},
    }

    try:
        poller = client.begin_create_analyzer(
            analyzer_id=IMAGE_ANALYZER_ID,
            resource=IMAGE_ANALYZER_SCHEMA,
            allow_replace=True,
        )
        result = poller.result()
        print(f"  ✓ Image analyzer created: {IMAGE_ANALYZER_ID}")
    except Exception as e:
        print(f"  ✗ Error creating image analyzer: {e}")
        print("    Continuing — text analyzer is still available.")

    # ================================================================
    # Verify the analyzers
    # ================================================================
    print()
    print("  Verifying analyzer configurations...")
    analyzer = client.get_analyzer(analyzer_id=ANALYZER_ID)
    print(f"  ✓ Text analyzer verified: {analyzer.analyzer_id}")
    try:
        img_analyzer = client.get_analyzer(analyzer_id=IMAGE_ANALYZER_ID)
        print(f"  ✓ Image analyzer verified: {img_analyzer.analyzer_id}")
    except Exception:
        print(f"  ⚠ Image analyzer not available")
    print()

    print("=" * 70)
    print("MODULE 01 — Summary")
    print("=" * 70)
    print(f"""
  Created: {ANALYZER_ID}
  Base: prebuilt-document (OCR + layout + table extraction)
  Completion Model: gpt-4.1
  Fields: {len(ANALYZER_SCHEMA['fieldSchema']['fields'])}
    ├─ Extract: document_date, author, merchant_names, merchant_ids,
    │           customer_ids, transaction_amounts
    ├─ Classify: document_type, priority_level
    └─ Generate: risk_indicators, recommended_actions, summary

  KEY INSIGHT:
  Content Understanding combines OCR/layout extraction with LLM reasoning.
  The 'extract' fields pull values directly from text. The 'generate' fields
  use GPT-4.1 to synthesize insights (risk indicators, summaries) that
  aren't explicitly stated. The 'classify' fields categorize the document.

  NEXT: Run 02_analyze_documents.py
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
