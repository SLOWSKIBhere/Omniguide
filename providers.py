"""Provider-independent model runtime for OmniGuide.

The pipeline talks only to ProviderRouter. Gemini is optional; any multimodal
OpenAI-compatible endpoint can be the primary provider.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Iterable, Optional

import httpx
from PIL import Image

logger = logging.getLogger("omniguide.providers")


def _google_genai_dependency_ready() -> bool:
    try:
        import_module("google.genai")
    except Exception:
        return False
    return True


class ProviderError(RuntimeError):
    """Base provider error safe to surface after sanitization."""


class ProviderUnavailableError(ProviderError):
    """Raised when no configured provider can handle a request."""


class ProviderCallError(ProviderError):
    """Raised when a configured provider rejects or fails a request."""


def sanitize_provider_error(value: object) -> str:
    """Remove likely secrets and cap provider error text."""
    text = str(value)
    patterns = [
        r"AIza[0-9A-Za-z_-]{20,}",
        r"sk-[0-9A-Za-z_-]{12,}",
        r"Bearer\s+[0-9A-Za-z._-]+",
        r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+",
        r"(?i)(authorization\s*[=:]\s*)[^\s,;]+",
    ]
    for pattern in patterns:
        text = re.sub(pattern, r"\1[redacted]" if "(" in pattern else "[redacted]", text)
    return " ".join(text.split())[:240]


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating markdown fences and surrounding prose."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ProviderCallError("Provider returned non-JSON output")
        value = json.loads(raw[start : end + 1])
    if not isinstance(value, dict):
        raise ProviderCallError("Provider returned JSON that was not an object")
    return value


@dataclass
class ProviderResult:
    text: str
    tokens: int
    provider: str
    model: str
    attempts: list[str] = field(default_factory=list)
    data: Optional[dict[str, Any]] = None


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_COMPAT_API_KEY", "").strip()
        self.base_url = os.getenv("OPENAI_COMPAT_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.text_model = os.getenv("OPENAI_COMPAT_TEXT_MODEL", "openai/gpt-4o-mini").strip()
        self.vision_model = os.getenv("OPENAI_COMPAT_VISION_MODEL", self.text_model).strip()
        self.timeout = float(os.getenv("MODEL_REQUEST_TIMEOUT_SECONDS", "35"))
        self.site_url = os.getenv("OPENAI_COMPAT_SITE_URL", "").strip()
        self.app_name = os.getenv("OPENAI_COMPAT_APP_NAME", "OmniGuide").strip()

    def available_for(self, *, vision: bool) -> bool:
        return bool(self.api_key and (self.vision_model if vision else self.text_model))

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "configured": bool(self.api_key),
            "vision_model": self.vision_model or None,
            "text_model": self.text_model or None,
        }

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderCallError("OpenAI-compatible response had no message content") from exc
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            pieces: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        pieces.append(text)
            return "\n".join(pieces).strip()
        raise ProviderCallError("OpenAI-compatible response content had an unsupported shape")

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_base64: Optional[str] = None,
        json_mode: bool = False,
    ) -> ProviderResult:
        vision = image_base64 is not None
        model = self.vision_model if vision else self.text_model
        if not self.available_for(vision=vision):
            raise ProviderUnavailableError(f"{self.name} is not configured for {'vision' if vision else 'text'}")

        user_content: Any = user_prompt
        if image_base64 is not None:
            user_content = [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ]

        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0 if json_mode else 0.3,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.endpoint, headers=headers, json=body)
            if response.status_code >= 400 and json_mode and response.status_code in {400, 404, 422}:
                body.pop("response_format", None)
                response = await client.post(self.endpoint, headers=headers, json=body)
            if response.status_code >= 400:
                safe_body = sanitize_provider_error(response.text)
                raise ProviderCallError(f"{self.name} HTTP {response.status_code}: {safe_body}")
            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderCallError(f"{self.name} returned invalid JSON") from exc

        text = self._extract_text(payload)
        usage = payload.get("usage") or {}
        tokens = int(usage.get("total_tokens") or 0)
        return ProviderResult(text=text, tokens=tokens, provider=self.name, model=model)


class GeminiProvider:
    name = "gemini"

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.text_model = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.0-flash").strip()
        self.vision_model = os.getenv("GEMINI_VISION_MODEL", self.text_model).strip()
        self.dependency_ready = _google_genai_dependency_ready()

    def available_for(self, *, vision: bool) -> bool:
        model = self.vision_model if vision else self.text_model
        return bool(self.dependency_ready and self.api_key and model)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "configured": bool(self.api_key),
            "dependency_ready": self.dependency_ready,
            "vision_model": self.vision_model or None,
            "text_model": self.text_model or None,
        }

    def _sync_generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_base64: Optional[str],
        json_mode: bool,
    ) -> ProviderResult:
        try:
            from google import genai
        except ImportError as exc:
            raise ProviderUnavailableError("google-genai is not installed") from exc

        vision = image_base64 is not None
        model = self.vision_model if vision else self.text_model
        client = genai.Client(api_key=self.api_key)
        contents: Any = f"{system_prompt}\n\n{user_prompt}"
        if image_base64 is not None:
            image_bytes = base64.b64decode(image_base64, validate=True)
            image = Image.open(io.BytesIO(image_bytes))
            contents = [f"{system_prompt}\n\n{user_prompt}", image]

        config: dict[str, Any] = {"temperature": 0.0 if json_mode else 0.3}
        if json_mode:
            config["response_mime_type"] = "application/json"
        response = client.models.generate_content(model=model, contents=contents, config=config)
        text = (getattr(response, "text", "") or "").strip()
        usage = getattr(response, "usage_metadata", None)
        tokens = int(getattr(usage, "total_token_count", 0) or 0)
        return ProviderResult(text=text, tokens=tokens, provider=self.name, model=model)

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_base64: Optional[str] = None,
        json_mode: bool = False,
    ) -> ProviderResult:
        if not self.available_for(vision=image_base64 is not None):
            raise ProviderUnavailableError(f"{self.name} is not configured")
        return await asyncio.to_thread(
            self._sync_generate,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_base64=image_base64,
            json_mode=json_mode,
        )


class ProviderRouter:
    """Failover router across provider adapters."""

    def __init__(self, providers: Optional[Iterable[Any]] = None) -> None:
        if providers is not None:
            self.providers = list(providers)
            return
        known = {
            "openai_compatible": OpenAICompatibleProvider,
            "gemini": GeminiProvider,
        }
        order = [
            item.strip().lower()
            for item in os.getenv("MODEL_PROVIDER_ORDER", "openai_compatible,gemini").split(",")
            if item.strip()
        ]
        self.providers = [known[name]() for name in order if name in known]

    def has_available_provider(self, *, vision: bool) -> bool:
        return any(provider.available_for(vision=vision) for provider in self.providers)

    def describe(self) -> list[dict[str, Any]]:
        return [provider.describe() for provider in self.providers]

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_base64: Optional[str] = None,
        json_mode: bool = False,
    ) -> ProviderResult:
        vision = image_base64 is not None
        attempts: list[str] = []
        configured = False
        for provider in self.providers:
            if not provider.available_for(vision=vision):
                continue
            configured = True
            try:
                result = await provider.generate_text(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_base64=image_base64,
                    json_mode=json_mode,
                )
                result.attempts = attempts
                return result
            except Exception as exc:
                safe = sanitize_provider_error(exc)
                attempts.append(f"{provider.name}: {safe}")
                logger.warning("Provider %s failed: %s", provider.name, safe)

        if not configured:
            raise ProviderUnavailableError(
                "No compatible model provider is configured. Set OPENAI_COMPAT_API_KEY or GEMINI_API_KEY."
            )
        raise ProviderCallError("All configured providers failed: " + " | ".join(attempts))

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_base64: Optional[str] = None,
    ) -> ProviderResult:
        vision = image_base64 is not None
        attempts: list[str] = []
        configured = False
        for provider in self.providers:
            if not provider.available_for(vision=vision):
                continue
            configured = True
            try:
                result = await provider.generate_text(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_base64=image_base64,
                    json_mode=True,
                )
                result.data = parse_json_object(result.text)
                result.attempts = attempts
                return result
            except Exception as exc:
                safe = sanitize_provider_error(exc)
                attempts.append(f"{provider.name}: {safe}")
                logger.warning("Provider %s failed: %s", provider.name, safe)

        if not configured:
            raise ProviderUnavailableError(
                "No compatible model provider is configured. Set OPENAI_COMPAT_API_KEY or GEMINI_API_KEY."
            )
        raise ProviderCallError("All configured providers failed: " + " | ".join(attempts))
