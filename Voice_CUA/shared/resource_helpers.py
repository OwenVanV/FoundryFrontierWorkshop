"""
Voice_CUA shared helpers for extracting resource names from env vars.
"""

import re


def parse_resource_name(value: str) -> str:
    """Extract the Azure AI resource name from a value that could be:
    - Just the name: "owenv-foundry-resource"
    - A full endpoint URL: "https://owenv-foundry-resource.services.ai.azure.com/..."
    - A full URL with path: "https://owenv-foundry-resource.services.ai.azure.com/api/projects/..."

    Returns just the resource name (e.g., "owenv-foundry-resource").
    """
    if not value:
        return value

    # Strip whitespace
    value = value.strip()

    # If it looks like a URL, extract the subdomain
    if "://" in value or value.startswith("wss://") or value.startswith("https://"):
        # Remove protocol
        without_protocol = re.sub(r"^(https?://|wss://)", "", value)
        # Take the hostname part (before the first /)
        hostname = without_protocol.split("/")[0]
        # Extract subdomain (before .services.ai.azure.com or .cognitiveservices.azure.com)
        for suffix in [".services.ai.azure.com", ".cognitiveservices.azure.com", ".openai.azure.com"]:
            if suffix in hostname:
                return hostname.replace(suffix, "")
        # If no known suffix, return the full hostname
        return hostname

    # Already just a name
    return value
