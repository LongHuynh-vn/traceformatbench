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


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout_s: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

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
