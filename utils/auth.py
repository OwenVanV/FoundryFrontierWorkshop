"""
============================================================================
PayPal x Azure AI Workshop — Authentication Helper
============================================================================

PURPOSE:
    Centralized authentication logic used across ALL workshop scenarios
    (LLM modules, agents, Voice_CUA). Supports two modes:

    1. API Key authentication (default, simpler for workshops)
    2. Managed Identity / DefaultAzureCredential (production pattern)

    Controlled by the USE_MANAGED_IDENTITY environment variable.

USAGE:
    from utils.auth import get_azure_credential, get_openai_client_args, get_voice_live_headers

============================================================================
"""

import os
from typing import Optional

from dotenv import load_dotenv
load_dotenv()


def use_managed_identity() -> bool:
    """Check if managed identity is enabled via environment variable."""
    return os.getenv("USE_MANAGED_IDENTITY", "false").lower() in ("true", "1", "yes")


def get_azure_credential():
    """
    Get an Azure credential object for managed identity scenarios.

    Returns DefaultAzureCredential which tries (in order):
    1. Environment variables (AZURE_CLIENT_ID, etc.)
    2. Managed Identity (on Azure VMs, App Service, AKS, etc.)
    3. Azure CLI (az login)
    4. Azure PowerShell
    5. VS Code Azure extension

    In workshops, participants typically use `az login`.
    In production, Managed Identity is the recommended approach.
    """
    from azure.identity import DefaultAzureCredential
    return DefaultAzureCredential()


def get_token_provider(scope: str = "https://cognitiveservices.azure.com/.default"):
    """
    Get a bearer token provider function for managed identity.

    This returns a callable that produces fresh tokens — used by the
    OpenAI SDK's `azure_ad_token_provider` parameter.
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    credential = DefaultAzureCredential()
    return get_bearer_token_provider(credential, scope)


def get_openai_client_args() -> dict:
    """
    Get the kwargs for constructing an AzureOpenAI client.

    Returns a dict that can be unpacked into AzureOpenAI(**args).

    When USE_MANAGED_IDENTITY=true:
        Uses azure_ad_token_provider (no API key needed)
    When USE_MANAGED_IDENTITY=false (default):
        Uses api_key from AZURE_OPENAI_API_KEY
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not endpoint:
        raise EnvironmentError(
            "AZURE_OPENAI_ENDPOINT is required. Set it in your .env file."
        )

    args = {
        "azure_endpoint": endpoint,
        "api_version": api_version,
    }

    if use_managed_identity():
        args["azure_ad_token_provider"] = get_token_provider()
    else:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "AZURE_OPENAI_API_KEY is required when USE_MANAGED_IDENTITY is not enabled.\n"
                "Either set the API key or set USE_MANAGED_IDENTITY=true and run `az login`."
            )
        args["api_key"] = api_key

    return args


def get_voice_live_headers() -> dict:
    """
    Get WebSocket connection headers for Voice Live API.

    When USE_MANAGED_IDENTITY=true:
        Uses Bearer token from DefaultAzureCredential
    When USE_MANAGED_IDENTITY=false (default):
        Uses api-key header
    """
    if use_managed_identity():
        credential = get_azure_credential()
        # Voice Live accepts tokens scoped to cognitiveservices or ai.azure.com
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        return {"Authorization": f"Bearer {token.token}"}
    else:
        api_key = os.getenv("AZURE_AI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "AZURE_AI_API_KEY is required when USE_MANAGED_IDENTITY is not enabled."
            )
        return {"api-key": api_key}


def get_foundry_credential():
    """
    Get credential for Foundry Agent Service / MAF.

    When USE_MANAGED_IDENTITY=true:
        Returns DefaultAzureCredential (works with both
        AIProjectClient and FoundryChatClient)
    When USE_MANAGED_IDENTITY=false:
        Still returns DefaultAzureCredential — the Foundry SDK
        always uses credential objects, never raw API keys.
        Participants must run `az login` regardless.
    """
    # Foundry SDKs (azure-ai-projects, agent-framework-foundry) always
    # use credential objects. The difference with managed identity is
    # whether it picks up az login vs. a managed identity on Azure.
    return get_azure_credential()
