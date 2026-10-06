import logging
from typing import Any, Optional

from .adapters import (
    AzureOpenAIProvider,
    GroqProvider,
    LMStudioProvider,
    OpenAIProvider,
    OllamaProvider,
    TogetherProvider,
)
from .atlas_auth import (
    atlas_default_headers,
    atlas_env_present,
    atlas_settings,
    build_token_provider,
    env_value as _get_env_value,
)

logger = logging.getLogger(__name__)


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

    if provider_name in {"azure", "azure_openai", "atlas"}:
        if atlas_env_present():
            # Atlas gateway: Entra ID bearer token + subscription-key header.
            settings = atlas_settings()
            logger.info("Using Entra ID authentication for Atlas endpoint %s", settings["endpoint"])
            return AzureOpenAIProvider(
                api_key=settings["subscription_key"] or api_key,
                endpoint=settings["endpoint"] or base_url,
                api_version=settings["api_version"] or api_version,
                auth_header_name=settings["subscription_header"] or auth_header_name,
                token_provider=build_token_provider(prime=True),
                default_headers=atlas_default_headers(),
            )
        if not api_key or not base_url:
            raise RuntimeError(
                "No Azure/Atlas LLM credentials found. Set LLM_API_KEY and LLM_BASE_URL "
                "(or OpenAI_API_Key / Base_URL), and for the Atlas gateway also "
                "ATLAS_TENANT_ID, ATLAS_CLIENT_ID and ATLAS_TOKEN_SCOPE. If you edited .env "
                "while the server was running, stop it and start it again; the auto-reloader "
                "does not re-read .env."
            )
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
