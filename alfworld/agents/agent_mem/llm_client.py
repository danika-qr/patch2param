"""OpenAI-compatible chat client for actor / reward-model separation."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ChatResponse:
    content: str
    raw: Dict[str, Any]


def _is_openai_reasoning_model(model: str) -> bool:
    """gpt-5 / o-series: max_completion_tokens, no custom temperature."""
    m = model.lower()
    return (
        m.startswith("gpt-5")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
    )


def _uses_max_completion_tokens(model: str) -> bool:
    return _is_openai_reasoning_model(model)


def _omits_temperature(model: str) -> bool:
    return _is_openai_reasoning_model(model)


class OpenAICompatibleClient:
    """
    Minimal chat-completions client (stdlib only).
    Works with OpenAI, vLLM, Ollama OpenAI shim, etc.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> ChatResponse:
        if not self.api_key:
            raise RuntimeError(
                "Missing API key. Set OPENAI_API_KEY or pass --api-key."
            )
        url = f"{self.base_url}/chat/completions"
        raw = self._post_chat(url, messages, temperature=temperature, max_tokens=max_tokens)

        content = _extract_message_content(raw)
        return ChatResponse(content=content.strip(), raw=raw)


def _extract_message_content(raw: Dict[str, Any]) -> str:
    choice = raw.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
            elif isinstance(part, str):
                parts.append(part)
        content = "".join(parts)
    if content is None:
        content = ""
    content = str(content)
    if not content.strip():
        refusal = msg.get("refusal")
        if refusal:
            content = str(refusal)
        elif choice.get("finish_reason") == "length":
            raise RuntimeError(
                "LLM returned empty content (finish_reason=length). "
                "Increase --max-tokens (e.g. 8192) for patch calls."
            )
    return content

    def _post_chat(
        self,
        url: str,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        body = self._build_body(messages, temperature=temperature, max_tokens=max_tokens)
        try:
            return self._request_json(url, body)
        except RuntimeError as exc:
            return self._retry_after_api_error(url, body, exc, max_tokens=max_tokens)

    def _build_body(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        include_temperature: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if include_temperature is None:
            include_temperature = not _omits_temperature(self.model)
        body: Dict[str, Any] = {"model": self.model, "messages": messages}
        if include_temperature:
            body["temperature"] = temperature
        if _uses_max_completion_tokens(self.model):
            body["max_completion_tokens"] = max_tokens
        else:
            body["max_tokens"] = max_tokens
        return body

    def _retry_after_api_error(
        self,
        url: str,
        body: Dict[str, Any],
        exc: RuntimeError,
        *,
        max_tokens: int,
    ) -> Dict[str, Any]:
        err = str(exc)
        changed = False
        if "temperature" in err and ("unsupported" in err.lower() or "default" in err.lower()):
            if "temperature" in body:
                body = {k: v for k, v in body.items() if k != "temperature"}
                changed = True
        if "max_completion_tokens" in err and "max_tokens" in body:
            body.pop("max_tokens", None)
            body["max_completion_tokens"] = max_tokens
            changed = True
        elif "max_tokens" in err and "max_completion_tokens" in body:
            body.pop("max_completion_tokens", None)
            body["max_tokens"] = max_tokens
            changed = True
        if changed:
            return self._request_json(url, body)
        raise exc

    def _request_json(self, url: str, body: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc


@dataclass
class ActorRMClients:
    """Actor (policy) and optional reward model as separate endpoints/models."""

    actor: OpenAICompatibleClient
    reward_model: Optional[OpenAICompatibleClient] = None

    @classmethod
    def from_cli(
        cls,
        model: str,
        *,
        rm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        rm_api_key: Optional[str] = None,
        rm_base_url: Optional[str] = None,
    ) -> "ActorRMClients":
        actor = OpenAICompatibleClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        rm = None
        if rm_model:
            rm = OpenAICompatibleClient(
                model=rm_model,
                api_key=rm_api_key or api_key,
                base_url=rm_base_url or base_url,
            )
        return cls(actor=actor, reward_model=rm)
