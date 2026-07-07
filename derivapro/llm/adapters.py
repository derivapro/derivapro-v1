import os
from typing import Any, Dict, List, Optional

from .llm_provider import LLMProvider

try:
    from openai import OpenAI, AzureOpenAI
except ImportError:  # pragma: no cover
    OpenAI = None
    AzureOpenAI = None


class AzureOpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: Optional[str] = None,
        auth_header_name: Optional[str] = None,
    ):
        if AzureOpenAI is None:
            raise RuntimeError("openai package is required for AzureOpenAIProvider")

        default_headers = None
        if auth_header_name and api_key:
            default_headers = {auth_header_name: api_key}

        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            default_headers=default_headers,
            azure_endpoint=endpoint,
        )
        self.endpoint = endpoint

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        return self.client.chat.completions.create(model=model, messages=messages, **kwargs)

    def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": "Assistant is a large language model hosted in Azure OpenAI.",
            },
            {"role": "user", "content": prompt},
        ]
        response = self.chat_completion(messages=messages, model=model, **kwargs)
        return response.choices[0].message.content

    def get_model_info(self) -> Dict[str, Any]:
        return {"provider": "azure", "endpoint": self.endpoint}


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, endpoint: Optional[str] = None):
        if OpenAI is None:
            raise RuntimeError("openai package is required for OpenAIProvider")

        self.client = OpenAI(api_key=api_key, base_url=endpoint)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        return self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

    def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {"role": "system", "content": "Assistant is a large language model."},
            {"role": "user", "content": prompt},
        ]
        response = self.chat_completion(messages=messages, model=model, **kwargs)
        return response.choices[0].message.content

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "openai",
            "base_url": getattr(self.client, "base_url", None),
        }


class OllamaProvider(LLMProvider):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        import requests

        url = f"{self.endpoint}/chat/completions"
        payload = {"model": model, "messages": messages}
        payload.update(kwargs)

        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {"role": "system", "content": "Assistant is a large language model."},
            {"role": "user", "content": prompt},
        ]
        result = self.chat_completion(messages=messages, model=model, **kwargs)
        if isinstance(result, dict) and result.get("choices"):
            return result["choices"][0]["message"]["content"]
        raise RuntimeError("Unexpected response from Ollama provider")

    def get_model_info(self) -> Dict[str, Any]:
        return {"provider": "ollama", "endpoint": self.endpoint}


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        import requests

        url = f"{self.endpoint}/chat/completions"
        payload = {"model": model, "messages": messages}
        payload.update(kwargs)
        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {"role": "system", "content": "Assistant is a large language model."},
            {"role": "user", "content": prompt},
        ]
        result = self.chat_completion(messages=messages, model=model, **kwargs)
        if isinstance(result, dict) and result.get("choices"):
            return result["choices"][0]["message"]["content"]
        raise RuntimeError("Unexpected response from Groq provider")

    def get_model_info(self) -> Dict[str, Any]:
        return {"provider": "groq", "endpoint": self.endpoint}


class TogetherProvider(LLMProvider):
    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        import requests

        url = f"{self.endpoint}/chat/completions"
        payload = {"model": model, "messages": messages}
        payload.update(kwargs)
        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {"role": "system", "content": "Assistant is a large language model."},
            {"role": "user", "content": prompt},
        ]
        result = self.chat_completion(messages=messages, model=model, **kwargs)
        if isinstance(result, dict) and result.get("choices"):
            return result["choices"][0]["message"]["content"]
        raise RuntimeError("Unexpected response from Together provider")

    def get_model_info(self) -> Dict[str, Any]:
        return {"provider": "together", "endpoint": self.endpoint}


class LMStudioProvider(LLMProvider):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        import requests

        url = f"{self.endpoint}/chat/completions"
        payload = {"model": model, "messages": messages}
        payload.update(kwargs)

        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {"role": "system", "content": "Assistant is a large language model."},
            {"role": "user", "content": prompt},
        ]
        result = self.chat_completion(messages=messages, model=model, **kwargs)
        if isinstance(result, dict) and result.get("choices"):
            return result["choices"][0]["message"]["content"]
        raise RuntimeError("Unexpected response from LM Studio provider")

    def get_model_info(self) -> Dict[str, Any]:
        return {"provider": "lmstudio", "endpoint": self.endpoint}
