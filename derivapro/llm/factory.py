import os
from typing import Any, Optional

from .adapters import (
    AzureOpenAIProvider,
    GroqProvider,
    LMStudioProvider,
    OpenAIProvider,
    OllamaProvider,
    TogetherProvider,
)


def _get_env_value(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value.strip()
    return default


def get_llm_provider() -> Any:
    provider_name = _get_env_value(
        "LLM_PROVIDER", "LLM_MODE", "PROVIDER", "OpenAI_PROVIDER", default="azure"
    ).lower()
    api_key = _get_env_value("LLM_API_KEY", "OpenAI_API_Key", default="")
    base_url = _get_env_value("LLM_BASE_URL", "Base_URL", default="")
    api_version = _get_env_value("LLM_API_VERSION", "API_Version", default=None)
    model = _get_env_value("LLM_MODEL", "Model", default=None)
    auth_header_name = _get_env_value(
        "LLM_AUTH_HEADER_NAME", "Auth_headers", default=None
    )

    if provider_name in {"azure", "azure_openai"}:
        return AzureOpenAIProvider(
            api_key=api_key,
            endpoint=base_url,
            api_version=api_version,
            auth_header_name=auth_header_name,
        )
    if provider_name in {"openai"}:
        return OpenAIProvider(api_key=api_key, endpoint=base_url or None)
    if provider_name in {"ollama"}:
        return OllamaProvider(endpoint=base_url)
    if provider_name in {"groq"}:
        return GroqProvider(api_key=api_key, endpoint=base_url)
    if provider_name in {"together"}:
        return TogetherProvider(api_key=api_key, endpoint=base_url)
    if provider_name in {"lmstudio", "lm_studio"}:
        return LMStudioProvider(endpoint=base_url)

    raise ValueError(f"Unsupported LLM provider: {provider_name}")


llm_client = get_llm_provider()
