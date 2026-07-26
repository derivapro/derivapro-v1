"""Lightweight test harness for LLM provider selection.

Run this script to validate that `get_llm_provider()` returns an adapter
for various provider configurations without modifying application code.

Usage:
    python -m derivapro.llm.test_provider_selection

This script does not make network calls; it only constructs provider clients.
If a provider adapter requires external packages (e.g., `openai`) and they
are missing, the script will report the error.
"""

import os
from importlib import import_module

from derivapro.llm import factory


def test_provider(provider_name, extra_env=None):
    print("\n=== Testing provider:", provider_name, "===")
    env_backup = dict(os.environ)
    try:
        os.environ["LLM_PROVIDER"] = provider_name
        if extra_env:
            for k, v in extra_env.items():
                os.environ[k] = v

        # Call the factory to create a provider instance
        prov = factory.get_llm_provider()
        print("Provider instance type:", type(prov).__name__)
        try:
            info = prov.get_model_info()
            print("Model info:", info)
        except Exception as e:
            print("get_model_info() raised:", repr(e))
    except Exception as e:
        print("Failed to initialize provider:", repr(e))
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


if __name__ == "__main__":
    tests = [
        (
            "azure",
            {
                "LLM_API_KEY": "fake-key",
                "LLM_BASE_URL": "https://atlas.protiviti.com/xxx",
            },
        ),
        ("openai", {"LLM_API_KEY": "sk-fake", "LLM_BASE_URL": ""}),
        ("ollama", {"LLM_BASE_URL": "http://localhost:11434/v1"}),
        ("lmstudio", {"LLM_BASE_URL": "http://localhost:1234/v1"}),
        (
            "groq",
            {
                "LLM_API_KEY": "gsk-fake",
                "LLM_BASE_URL": "https://api.groq.com/openai/v1",
            },
        ),
        (
            "together",
            {"LLM_API_KEY": "tk-fake", "LLM_BASE_URL": "https://api.together.xyz/v1"},
        ),
    ]

    for provider, env in tests:
        test_provider(provider, extra_env=env)
