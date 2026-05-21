"""Teacher API helpers.

The DeepSeek API is OpenAI-compatible for chat completions, so this module keeps
the dependency surface small and uses requests directly. Hidden-state extraction
is model/provider-specific; this reference client stores teacher text and leaves
latent targets optional for providers that expose them.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import requests


class TeacherClientError(RuntimeError):
    """Raised when the teacher API call fails or returns an unexpected payload."""


@dataclass(frozen=True)
class TeacherResponse:
    text: str
    raw: Dict[str, Any]


class DeepSeekTeacherClient:
    """Minimal OpenAI-compatible DeepSeek chat client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise TeacherClientError("DEEPSEEK_API_KEY is not set.")

        self.base_url = (
            base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        ).rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"
        self.timeout = timeout

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str = "You are a careful teacher. Improve the answer and explain the reasoning.",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        reasoning_effort: Optional[str] = None,
    ) -> TeacherResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise TeacherClientError(f"Teacher request failed: {exc}") from exc

        data = response.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TeacherClientError(f"Unexpected teacher response shape: {data}") from exc
        return TeacherResponse(text=text, raw=data)
