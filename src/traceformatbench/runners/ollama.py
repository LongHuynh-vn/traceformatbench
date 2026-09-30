"""Minimal Ollama HTTP client (standard library only)."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

DEFAULT_URL = "http://localhost:11434"


@dataclass(frozen=True)
class PromptTokenCount:
    """Token counts reported by Ollama for one prompt."""

    prompt_tokens: int
    cached_tokens: int
    total_duration_ns: int


@dataclass(frozen=True)
class ChatResult:
    """Unscored chat output and Ollama's token counts and nanosecond timings."""

    content: str
    thinking: str
    done_reason: str
    prompt_eval_count: int
    prompt_eval_cached_count: int
    eval_count: int
    load_duration_ns: int
    prompt_eval_duration_ns: int
    eval_duration_ns: int
    total_duration_ns: int


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout_s: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._capabilities: dict[str, list[str]] = {}

    def _request(self, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def version(self) -> str:
        return str(self._request("/api/version")["version"])

    def model_digest(self, model: str) -> str:
        """Digest of a locally pulled model, so every measurement names an exact model build."""
        for entry in self._request("/api/tags").get("models", []):
            if entry.get("name") == model or entry.get("model") == model:
                return str(entry["digest"])
        raise LookupError(f"Model {model!r} is not pulled. Run: ollama pull {model}")

    def model_capabilities(self, model: str) -> list[str]:
        """Read capabilities once, outside timed calls when preflighted by the runner."""
        if model not in self._capabilities:
            response = self._request("/api/show", {"model": model})
            self._capabilities[model] = list(response.get("capabilities", []))
        return list(self._capabilities[model])

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        num_ctx: int,
        num_predict: int,
        temperature: float,
        seed: int,
        think: bool,
        output_schema: dict | None = None,
    ) -> ChatResult:
        """Make one non-streaming chat request with explicit generation options.

        Ask Ollama to reject overlong input before generation rather than truncate it.
        The probe also checks returned counts for servers with legacy truncation behavior.
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "truncate": False,
            "options": {
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "temperature": temperature,
                "seed": seed,
            },
        }
        if output_schema is not None:
            payload["format"] = output_schema
        if "thinking" in self.model_capabilities(model):
            payload["think"] = think
        response = self._request("/api/chat", payload)
        message = response["message"]
        return ChatResult(
            content=message["content"],
            thinking=message.get("thinking") or "",
            done_reason=response["done_reason"],
            # Ollama omits prompt_eval_count when the whole prompt came from its cache.
            prompt_eval_count=int(response.get("prompt_eval_count", 0)),
            prompt_eval_cached_count=int(response.get("prompt_eval_cached_count", 0)),
            eval_count=int(response["eval_count"]),
            load_duration_ns=int(response["load_duration"]),
            prompt_eval_duration_ns=int(response["prompt_eval_duration"]),
            eval_duration_ns=int(response["eval_duration"]),
            total_duration_ns=int(response["total_duration"]),
        )

    def unload_model(self, model: str) -> None:
        """Release the current model before the probe loads the next one."""
        self._request("/api/generate", {"model": model, "keep_alive": 0})

    def count_prompt_tokens(
        self, model: str, text: str, num_ctx: int, seed: int
    ) -> PromptTokenCount:
        """Count prompt tokens exactly as the model tokenizes them.

        Uses raw mode (no chat template) and generates a single token, so the count
        covers only `text`. `num_ctx` is always sent explicitly, never left to defaults.
        """
        response = self._request(
            "/api/generate",
            {
                "model": model,
                "prompt": text,
                "raw": True,
                "stream": False,
                "keep_alive": "10m",
                "options": {"num_ctx": num_ctx, "num_predict": 1, "temperature": 0, "seed": seed},
            },
        )
        return PromptTokenCount(
            prompt_tokens=int(response["prompt_eval_count"]),
            cached_tokens=int(response.get("prompt_eval_cached_count", 0)),
            total_duration_ns=int(response.get("total_duration", 0)),
        )
