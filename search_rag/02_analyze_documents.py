"""
============================================================================
SEARCH+RAG MODULE 02 — Content Understanding: Analyze Documents
============================================================================

WORKSHOP NARRATIVE:
    The analyzer is created. Now we run it against all 10 fraud evidence
    documents. For each document, Content Understanding:
    1. Extracts text (OCR/layout)
    2. Identifies fields using the schema we defined
    3. Assigns confidence scores to each extraction
    4. Generates summaries and risk indicators via GPT-4.1

    The output is structured JSON — ready for indexing in Azure AI Search.

LEARNING OBJECTIVES:
    1. Analyze documents with a custom analyzer
    2. Work with confidence scores and source grounding
    3. Understand the analyze → poll → result async pattern
    4. Prepare extracted data for search indexing

ESTIMATED TIME: 15-20 minutes

============================================================================
"""

import os
import sys
import json
import time
from pathlib import Path

from dotenv import load_dotenv
_dir = Path(__file__).parent
load_dotenv(_dir / ".env")
load_dotenv()

from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentunderstanding import ContentUnderstandingClient

ENDPOINT = os.getenv("CONTENT_UNDERSTANDING_ENDPOINT")
KEY = os.getenv("CONTENT_UNDERSTANDING_KEY")
USE_MI = os.getenv("USE_MANAGED_IDENTITY", "false").lower() in ("true", "1")
ANALYZER_ID = "fraud_evidence_analyzer"
IMAGE_ANALYZER_ID = "fraud_evidence_image_analyzer"
DOCS_DIR = Path(__file__).parent / "data" / "documents"
OUTPUT_DIR = Path(__file__).parent / "data" / "analyzed"

# File extensions that should use the image analyzer
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}


def get_client():
    if USE_MI:
        return ContentUnderstandingClient(endpoint=ENDPOINT, credential=DefaultAzureCredential())
    return ContentUnderstandingClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))


def main():
    print("=" * 70)
    print("SEARCH+RAG MODULE 02 — Analyze Documents")
    print("=" * 70)
    print()

    if not DOCS_DIR.exists():
        print("  ✗ Documents not found. Run: python search_rag/data/generate_documents.py")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = get_client()
    print(f"  ✓ Connected to: {ENDPOINT}")
    print(f"  Analyzer: {ANALYZER_ID}")
    print()

    # Load manifest
    manifest_path = DOCS_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        # No manifest — discover text and image files
        all_files = sorted(DOCS_DIR.glob("*.txt")) + sorted(DOCS_DIR.glob("*.png")) + sorted(DOCS_DIR.glob("*.jpg"))
        manifest = [{"filename": f.name, "type": "unknown"} for f in all_files]

    print(f"  Documents to analyze: {len(manifest)}")
    print()

    all_results = []

    for i, entry in enumerate(manifest, 1):
        filename = entry["filename"]
        filepath = DOCS_DIR / filename

        if not filepath.exists():
            print(f"  [{i}/{len(manifest)}] ✗ {filename} — file not found, skipping")
            continue

        print(f"  [{i}/{len(manifest)}] Analyzing: {filename}...")

        # Read the document as binary
        with open(filepath, "rb") as f:
            doc_bytes = f.read()

        # ============================================================
        # Determine analyzer and content type based on file extension
        # ============================================================
        ext = Path(filename).suffix.lower()
        is_image = ext in IMAGE_EXTENSIONS

        if is_image:
            analyzer_to_use = IMAGE_ANALYZER_ID
            # Map extension to MIME type
            mime_map = {
                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".bmp": "image/bmp", ".tiff": "image/tiff",
                ".webp": "image/webp",
            }
            content_type = mime_map.get(ext, "application/octet-stream")
            print(f"    Using IMAGE analyzer ({content_type})")
        else:
            analyzer_to_use = ANALYZER_ID
            content_type = "text/plain"

        # ============================================================
        # Analyze the document
        # ============================================================
        try:
            poller = client.begin_analyze_binary(
                analyzer_id=analyzer_to_use,
                binary_input=doc_bytes,
                content_type=content_type,
            )
            result = poller.result()
        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue

        # Extract the structured data
        if result.contents:
            content = result.contents[0]
            fields = {}

            if content.fields:
                for field_name, field_val in content.fields.items():
                    # Extract the value based on field type
                    if hasattr(field_val, "value_string") and field_val.value_string is not None:
                        fields[field_name] = {
                            "value": field_val.value_string,
                            "confidence": getattr(field_val, "confidence", None),
                        }
                    elif hasattr(field_val, "value_date") and field_val.value_date is not None:
                        fields[field_name] = {
                            "value": str(field_val.value_date),
                            "confidence": getattr(field_val, "confidence", None),
                        }
                    elif hasattr(field_val, "value_number") and field_val.value_number is not None:
                        fields[field_name] = {
                            "value": field_val.value_number,
                            "confidence": getattr(field_val, "confidence", None),
                        }
                    elif hasattr(field_val, "value_array") and field_val.value_array is not None:
                        arr_values = []
                        for item in field_val.value_array:
                            if hasattr(item, "value_string") and item.value_string is not None:
                                arr_values.append(item.value_string)
                            elif hasattr(item, "value_number") and item.value_number is not None:
                                arr_values.append(item.value_number)
                            else:
                                arr_values.append(str(item))
                        fields[field_name] = {
                            "value": arr_values,
                            "confidence": getattr(field_val, "confidence", None),
                        }
                    else:
                        # Fallback: try .value attribute
                        val = getattr(field_val, "value", None)
                        fields[field_name] = {
                            "value": str(val) if val else None,
                            "confidence": getattr(field_val, "confidence", None),
                        }

            doc_result = {
                "filename": filename,
                "document_type": entry.get("type", "unknown"),
                "markdown": getattr(content, "markdown", ""),
                "fields": fields,
            }
            all_results.append(doc_result)

            # Print summary
            doc_type = fields.get("document_type", {}).get("value") or "unknown"
            priority = fields.get("priority_level", {}).get("value") or "N/A"
            n_merchants = len(fields.get("merchant_names", {}).get("value") or [])
            n_customers = len(fields.get("customer_ids", {}).get("value") or [])
            n_risks = len(fields.get("risk_indicators", {}).get("value") or [])
            summary = fields.get("summary", {}).get("value") or ""

            print(f"    ✓ Type: {doc_type} | Priority: {priority}")
            print(f"      Merchants: {n_merchants} | Customers: {n_customers} | Risk indicators: {n_risks}")
            if summary:
                print(f"      Summary: {str(summary)[:100]}...")
            print()
        else:
            print(f"    ✗ No content returned")
            print()

    # ================================================================
    # Save all results
    # ================================================================
    output_path = OUTPUT_DIR / "analyzed_documents.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("─" * 70)
    print(f"  ✓ {len(all_results)} documents analyzed")
    print(f"  ✓ Results saved to: {output_path}")
    print()

    # Aggregate statistics
    all_merchants = set()
    all_customers = set()
    all_risks = set()
    for r in all_results:
        for mid in (r["fields"].get("merchant_ids", {}).get("value") or []):
            all_merchants.add(mid)
        for cid in (r["fields"].get("customer_ids", {}).get("value") or []):
            all_customers.add(cid)
        for risk in (r["fields"].get("risk_indicators", {}).get("value") or []):
            all_risks.add(risk)

    print("=" * 70)
    print("MODULE 02 — Summary")
    print("=" * 70)
    print(f"""
  Documents Analyzed:   {len(all_results)}
  Unique Merchant IDs:  {len(all_merchants)}
  Unique Customer IDs:  {len(all_customers)}
  Unique Risk Signals:  {len(all_risks)}

  KEY INSIGHT:
  Content Understanding extracted structured fields WITH confidence scores
  from unstructured text. The 'generate' fields (risk_indicators, summary)
  are especially valuable — they capture insights that aren't explicitly
  stated in the documents but are inferred by the LLM.

  NEXT: Run 03_build_search_index.py
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
