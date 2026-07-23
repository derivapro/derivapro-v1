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


class LazyLLMClient:
    """Lazily construct the configured LLM provider on first use.

    Route modules import ``llm_client`` during app startup. Creating the provider at
    import time makes the whole Flask app fail when optional LLM credentials are not
    configured, even for pages that do not use AI assessment. Lazy construction keeps
    startup healthy and lets existing ``ask_gpt`` handlers surface configuration
    errors only when an assessment is requested.
    """

    def __init__(self) -> None:
        self._provider: Optional[Any] = None

    def _get_provider(self) -> Any:
        if self._provider is None:
            self._provider = get_llm_provider()
        return self._provider

    def chat_completion(self, *args: Any, **kwargs: Any) -> Any:
        return self._get_provider().chat_completion(*args, **kwargs)

    def generate_response(self, *args: Any, **kwargs: Any) -> str:
        return self._get_provider().generate_response(*args, **kwargs)

    def get_model_info(self) -> dict[str, Any]:
        return self._get_provider().get_model_info()


llm_client = LazyLLMClient()
