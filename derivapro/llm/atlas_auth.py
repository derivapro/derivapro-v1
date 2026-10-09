"""Entra ID (OAuth2) authentication for the Atlas Azure OpenAI gateway.

Atlas sits behind an Azure Application Gateway that rejects a bare
subscription key with ``403 Forbidden``. Requests must carry *both*:

* an Entra ID bearer token for the Atlas API scope, and
* the per-experiment subscription-key header (plus ``AtlasAuthType: EntraID``).

This mirrors the ``targets/atlas.py`` helper used by the RAG leakage demo: the
token comes from ``InteractiveBrowserCredential`` (a browser sign-in on the
machine running the Flask app) and is handed to the OpenAI SDK through
``azure_ad_token_provider`` so the SDK never sees a static key.

Environment variables (same names as the reference project):

    ATLAS_TENANT_ID            Entra tenant GUID
    ATLAS_CLIENT_ID            Entra app registration (client) GUID
    ATLAS_TOKEN_SCOPE          e.g. api://<app-id>/.default
    ATLAS_ENDPOINT             full gateway base URL (falls back to LLM_BASE_URL / Base_URL)
    ATLAS_API_VERSION          falls back to LLM_API_VERSION / API_Version
    ATLAS_SUBSCRIPTION_HEADER  falls back to LLM_AUTH_HEADER_NAME / Auth_headers
    ATLAS_SUBSCRIPTION_KEY     falls back to LLM_API_KEY / OpenAI_API_Key
    ATLAS_AUTH_TYPE_HEADER     optional, default "AtlasAuthType"
    ATLAS_AUTH_TYPE_VALUE      optional, default "EntraID"
    ATLAS_TOKEN_CACHE_PERSIST  optional, default "1": keep the token cache on disk so a
                               Flask reload does not force another browser sign-in

Prime the sign-in before using the app (optional but avoids a browser popup
in the middle of the first AI Assessment request)::

    python -m derivapro.llm.atlas_auth
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_PLACEHOLDER_MARKERS = ("xxx", "yyyymmdd", "replace_me", "changeme", "<")


def _dotenv_file_values() -> Dict[str, Optional[str]]:
    """Current contents of the ``.env`` file on disk.

    ``load_dotenv()`` never overrides variables that already exist in
    ``os.environ``. Werkzeug's auto-reloader spawns the child process with the
    parent's environment, so edits to ``.env`` made after the server was first
    started are invisible to ``os.getenv`` until a full restart. Reading the file
    directly lets LLM settings pick up those edits.
    """
    try:
        from dotenv import dotenv_values, find_dotenv

        path = find_dotenv(usecwd=True)
        return dict(dotenv_values(path)) if path else {}
    except Exception:  # pragma: no cover - dotenv missing or unreadable
        return {}


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    if not value or is_placeholder(value):
        return None
    return value


def _running_under_reloader() -> bool:
    """True inside the Werkzeug auto-reloader child process (``flask run`` /
    ``app.run(debug=True)``). Its environment is a snapshot taken when the
    parent was launched, so the ``.env`` file is the more current source."""
    return os.environ.get("WERKZEUG_RUN_MAIN") == "true"


def env_value(*names: str, default: Optional[str] = None) -> Optional[str]:
    """First non-empty, non-placeholder value among ``names``.

    Resolution order per name: the process environment, then the ``.env`` file
    on disk (see :func:`_dotenv_file_values`). Inside the Werkzeug reloader
    child the order is flipped so edits to ``.env`` take effect on the next
    automatic reload instead of requiring a full stop/start. Values such as
    ``"xxx"`` or ``"https://atlas.protiviti.com/xxx"`` left over from
    ``.env.example`` are treated as unset so that a real legacy variable
    (``Base_URL``) wins over a placeholder preferred one (``LLM_BASE_URL``).
    """
    file_values: Optional[Dict[str, Optional[str]]] = None
    prefer_file = _running_under_reloader()
    for name in names:
        if prefer_file:
            if file_values is None:
                file_values = _dotenv_file_values()
            value = _clean(file_values.get(name))
            if value is not None:
                return value
        value = _clean(os.getenv(name))
        if value is not None:
            return value
        if not prefer_file:
            if file_values is None:
                file_values = _dotenv_file_values()
            value = _clean(file_values.get(name))
            if value is not None:
                return value
    return default


def is_placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def atlas_settings() -> Dict[str, Optional[str]]:
    """Resolve every Atlas-related setting with its fallbacks."""
    return {
        "tenant_id": env_value("ATLAS_TENANT_ID"),
        "client_id": env_value("ATLAS_CLIENT_ID"),
        "token_scope": env_value("ATLAS_TOKEN_SCOPE"),
        "endpoint": env_value("ATLAS_ENDPOINT", "LLM_BASE_URL", "Base_URL"),
        "api_version": env_value("ATLAS_API_VERSION", "LLM_API_VERSION", "API_Version"),
        "subscription_header": env_value(
            "ATLAS_SUBSCRIPTION_HEADER", "LLM_AUTH_HEADER_NAME", "Auth_headers"
        ),
        "subscription_key": env_value("ATLAS_SUBSCRIPTION_KEY", "LLM_API_KEY", "OpenAI_API_Key"),
        "auth_type_header": env_value("ATLAS_AUTH_TYPE_HEADER", default="AtlasAuthType"),
        "auth_type_value": env_value("ATLAS_AUTH_TYPE_VALUE", default="EntraID"),
        "persist_cache": env_value("ATLAS_TOKEN_CACHE_PERSIST", default="1"),
    }


def atlas_env_present() -> bool:
    """True when the Entra ID variables are set, i.e. use bearer-token auth."""
    settings = atlas_settings()
    return bool(settings["tenant_id"] and settings["client_id"] and settings["token_scope"])


# One credential per (tenant, client) for the life of the process so the MSAL
# in-memory cache is reused between requests.
_credential_cache: Dict[Tuple[str, str], Any] = {}


def _import_identity():
    try:
        from azure.identity import InteractiveBrowserCredential, get_bearer_token_provider
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Entra ID authentication requires the 'azure-identity' package. "
            "Install it with: pip install azure-identity"
        ) from exc
    return InteractiveBrowserCredential, get_bearer_token_provider


def get_credential(tenant_id: str, client_id: str, persist_cache: bool = True):
    key = (tenant_id, client_id)
    credential = _credential_cache.get(key)
    if credential is not None:
        return credential

    InteractiveBrowserCredential, _ = _import_identity()
    kwargs: Dict[str, Any] = {"tenant_id": tenant_id, "client_id": client_id}
    if persist_cache:
        try:
            from azure.identity import TokenCachePersistenceOptions

            kwargs["cache_persistence_options"] = TokenCachePersistenceOptions(
                name="derivapro-atlas"
            )
        except Exception as exc:  # pragma: no cover - platform dependent
            logger.warning("Persistent token cache unavailable (%s); using in-memory cache.", exc)

    try:
        credential = InteractiveBrowserCredential(**kwargs)
    except Exception as exc:
        if "cache_persistence_options" not in kwargs:
            raise
        logger.warning("Persistent token cache failed (%s); retrying without it.", exc)
        kwargs.pop("cache_persistence_options", None)
        credential = InteractiveBrowserCredential(**kwargs)

    _credential_cache[key] = credential
    return credential


def build_token_provider(prime: bool = True) -> Callable[[], str]:
    """Return a callable the OpenAI SDK can use as ``azure_ad_token_provider``.

    With ``prime=True`` a token is acquired immediately so the browser sign-in
    happens while the client is being built rather than inside the first
    chat completion call.
    """
    settings = atlas_settings()
    missing = [k for k in ("tenant_id", "client_id", "token_scope") if not settings[k]]
    if missing:
        raise RuntimeError(
            "Entra ID auth is incomplete; set " + ", ".join(f"ATLAS_{k.upper()}" for k in missing)
        )

    _, get_bearer_token_provider = _import_identity()
    credential = get_credential(
        settings["tenant_id"],
        settings["client_id"],
        persist_cache=str(settings["persist_cache"]).lower() in {"1", "true", "yes", "on"},
    )
    scope = settings["token_scope"]
    if prime:
        credential.get_token(scope)
        logger.info("Atlas Entra ID sign-in ready for %s", settings["endpoint"])
    return get_bearer_token_provider(credential, scope)


def atlas_default_headers() -> Dict[str, str]:
    """``AtlasAuthType`` and subscription-key headers the gateway expects."""
    settings = atlas_settings()
    headers: Dict[str, str] = {}
    if settings["auth_type_header"] and settings["auth_type_value"]:
        headers[settings["auth_type_header"]] = settings["auth_type_value"]
    if settings["subscription_header"] and settings["subscription_key"]:
        headers[settings["subscription_header"]] = settings["subscription_key"]
    return headers


def prime_atlas_login() -> str:
    """Sign in interactively and return the acquired token (used by ``__main__``)."""
    settings = atlas_settings()
    credential = get_credential(
        settings["tenant_id"] or "",
        settings["client_id"] or "",
        persist_cache=str(settings["persist_cache"]).lower() in {"1", "true", "yes", "on"},
    )
    return credential.get_token(settings["token_scope"] or "").token


if __name__ == "__main__":  # pragma: no cover - manual utility
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())
    if not atlas_env_present():
        raise SystemExit(
            "ATLAS_TENANT_ID, ATLAS_CLIENT_ID and ATLAS_TOKEN_SCOPE must be set in .env"
        )
    token = prime_atlas_login()
    print(f"Signed in. Token acquired ({len(token)} chars). Endpoint: {atlas_settings()['endpoint']}")
