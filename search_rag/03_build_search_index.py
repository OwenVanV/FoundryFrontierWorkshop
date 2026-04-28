"""
============================================================================
SEARCH+RAG MODULE 03 — Azure AI Search: Build the Evidence Index
============================================================================

WORKSHOP NARRATIVE:
    Content Understanding extracted structured fields from our fraud
    documents. Now we load those results into Azure AI Search — creating
    a searchable index with:
    - Full-text search over the markdown content
    - Filterable facets (document_type, merchant_ids, customer_ids)
    - Vector embeddings for semantic search
    - Confidence scores as sortable fields

    This index becomes the data source for our investigation agent.

LEARNING OBJECTIVES:
    1. Create an Azure AI Search index with typed fields
    2. Generate embeddings for vector search
    3. Upload documents with structured metadata
    4. Test search queries (text, filter, vector)

AZURE SERVICES:
    - Azure AI Search
    - Azure OpenAI (for embeddings)

ESTIMATED TIME: 15-20 minutes

============================================================================
"""

import os
import sys
import json
import hashlib
from pathlib import Path

from dotenv import load_dotenv
_dir = Path(__file__).parent
load_dotenv(_dir / ".env")
load_dotenv()

# Add project root to path for utils.auth
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from openai import AzureOpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "fraud-evidence")
USE_MI = os.getenv("USE_MANAGED_IDENTITY", "false").lower() in ("true", "1")

OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

ANALYZED_DIR = Path(__file__).parent / "data" / "analyzed"

if not SEARCH_ENDPOINT:
    print("ERROR: Set AZURE_SEARCH_ENDPOINT in search_rag/.env")
    sys.exit(1)


def get_search_index_client():
    if USE_MI:
        return SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=DefaultAzureCredential())
    return SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=AzureKeyCredential(SEARCH_KEY))


def get_search_client():
    if USE_MI:
        return SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=DefaultAzureCredential())
    return SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=AzureKeyCredential(SEARCH_KEY))


def get_openai_client():
    if USE_MI:
        from utils.auth import get_token_provider
        return AzureOpenAI(azure_endpoint=OPENAI_ENDPOINT, azure_ad_token_provider=get_token_provider(), api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))
    return AzureOpenAI(azure_endpoint=OPENAI_ENDPOINT, api_key=OPENAI_KEY, api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))


def get_embedding(text: str, client: AzureOpenAI) -> list[float]:
    """Generate an embedding vector for the given text."""
    response = client.embeddings.create(input=[text[:8000]], model=EMBEDDING_DEPLOYMENT)
    return response.data[0].embedding


def main():
    print("=" * 70)
    print("SEARCH+RAG MODULE 03 — Build Search Index")
    print("=" * 70)
    print()

    # ================================================================
    # Step 1: Load analyzed documents
    # ================================================================
    analyzed_path = ANALYZED_DIR / "analyzed_documents.json"
    if not analyzed_path.exists():
        print("  ✗ Analyzed documents not found. Run 02_analyze_documents.py first.")
        sys.exit(1)

    with open(analyzed_path, "r") as f:
        analyzed_docs = json.load(f)
    print(f"  ✓ Loaded {len(analyzed_docs)} analyzed documents")

    # ================================================================
    # Step 2: Create the search index
    # ================================================================
    print()
    print("  Creating search index...")

    # Determine embedding dimensions by generating a test embedding
    openai_client = get_openai_client()
    test_embedding = get_embedding("test", openai_client)
    embedding_dims = len(test_embedding)
    print(f"  ✓ Embedding model: {EMBEDDING_DEPLOYMENT} ({embedding_dims} dimensions)")

    index_client = get_search_index_client()

    # Define the index schema
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchableField(name="filename", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="document_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="priority_level", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="author", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="summary", type=SearchFieldDataType.String),
        # Collection fields for multi-valued facets
        SimpleField(name="merchant_names", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True, facetable=True),
        SimpleField(name="merchant_ids", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True, facetable=True),
        SimpleField(name="customer_ids", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True, facetable=True),
        SimpleField(name="risk_indicators", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True, facetable=True),
        SimpleField(name="recommended_actions", type=SearchFieldDataType.Collection(SearchFieldDataType.String)),
        # Vector field for semantic search
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=embedding_dims,
            vector_search_profile_name="default-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="default-algo")],
        profiles=[VectorSearchProfile(name="default-profile", algorithm_configuration_name="default-algo")],
    )

    index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)

    try:
        index_client.delete_index(INDEX_NAME)
        print(f"  ✓ Deleted existing index: {INDEX_NAME}")
    except Exception:
        pass

    index_client.create_index(index)
    print(f"  ✓ Created index: {INDEX_NAME}")
    print(f"    Fields: {len(fields)}")
    print()

    # ================================================================
    # Step 3: Generate embeddings and upload documents
    # ================================================================
    print("  Generating embeddings and uploading documents...")

    search_client = get_search_client()
    upload_docs = []

    for i, doc in enumerate(analyzed_docs, 1):
        filename = doc["filename"]
        fields_data = doc.get("fields", {})
        markdown = doc.get("markdown", "")

        # Build the search document
        doc_id = hashlib.md5(filename.encode()).hexdigest()

        # Extract field values (handling the {value, confidence} structure)
        def get_val(field_name, default=None):
            fd = fields_data.get(field_name, {})
            if isinstance(fd, dict):
                val = fd.get("value")
                return val if val is not None else default
            return fd if fd is not None else default

        # Generate embedding from the full content
        embed_text = f"{get_val('summary', '')} {markdown[:4000]}"
        embedding = get_embedding(embed_text, openai_client)

        search_doc = {
            "id": doc_id,
            "content": markdown[:30000],  # AI Search field size limit
            "filename": filename,
            "document_type": get_val("document_type", "unknown"),
            "priority_level": get_val("priority_level", "INFO"),
            "author": get_val("author", ""),
            "summary": get_val("summary", ""),
            "merchant_names": get_val("merchant_names", []),
            "merchant_ids": get_val("merchant_ids", []),
            "customer_ids": get_val("customer_ids", []),
            "risk_indicators": get_val("risk_indicators", []),
            "recommended_actions": get_val("recommended_actions", []),
            "embedding": embedding,
        }
        upload_docs.append(search_doc)

        n_merchants = len(search_doc["merchant_ids"])
        n_customers = len(search_doc["customer_ids"])
        print(f"  [{i}/{len(analyzed_docs)}] {filename}: {search_doc['document_type']} | "
              f"{n_merchants} merchants, {n_customers} customers")

    # Upload to search index
    print()
    print("  Uploading to Azure AI Search...")
    result = search_client.upload_documents(documents=upload_docs)
    succeeded = sum(1 for r in result if r.succeeded)
    print(f"  ✓ Uploaded: {succeeded}/{len(upload_docs)} documents")

    # ================================================================
    # Step 4: Test queries
    # ================================================================
    print()
    print("─" * 70)
    print("  Test Queries")
    print("─" * 70)
    print()

    # Text search
    print("  [Test 1] Text search: 'structuring below BSA threshold'")
    results = search_client.search(search_text="structuring below BSA threshold", top=3)
    for r in results:
        print(f"    → {r['filename']} (type: {r['document_type']}, score: {r['@search.score']:.2f})")
    print()

    # Filter search
    print("  [Test 2] Filter: document_type eq 'suspicious_activity_report'")
    results = search_client.search(search_text="*", filter="document_type eq 'suspicious_activity_report'", top=5)
    for r in results:
        print(f"    → {r['filename']}")
    print()

    # Faceted search
    print("  [Test 3] Facets: document_type")
    results = search_client.search(search_text="*", facets=["document_type"], top=0)
    for facet in results.get_facets().get("document_type", []):
        print(f"    → {facet['value']}: {facet['count']} documents")
    print()

    print("=" * 70)
    print("MODULE 03 — Summary")
    print("=" * 70)
    print(f"""
  Index: {INDEX_NAME}
  Documents indexed: {succeeded}
  Embedding model: {EMBEDDING_DEPLOYMENT} ({embedding_dims}D)

  The index supports:
    ├─ Full-text search (BM25) over document content
    ├─ Vector search (HNSW) over embeddings
    ├─ Filtered search by document_type, merchant_ids, customer_ids
    └─ Faceted navigation across risk_indicators and document types

  NEXT: Run 04_agent_investigation.py
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
