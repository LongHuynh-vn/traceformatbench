import pytest

from traceformatbench.runners.ollama import OllamaClient


def test_count_prompt_tokens_sends_raw_prompt_and_explicit_num_ctx(monkeypatch):
    sent = {}

    def fake_request(self, path, payload=None):
        sent["path"], sent["payload"] = path, payload
        return {"prompt_eval_count": 42, "prompt_eval_cached_count": 5, "total_duration": 9}

    monkeypatch.setattr(OllamaClient, "_request", fake_request)
    count = OllamaClient().count_prompt_tokens("m", "hello", num_ctx=4096, seed=3)

    assert (count.prompt_tokens, count.cached_tokens, count.total_duration_ns) == (42, 5, 9)
    assert sent["path"] == "/api/generate"
    assert sent["payload"]["raw"] is True
    assert sent["payload"]["options"]["num_ctx"] == 4096
    assert sent["payload"]["options"]["num_predict"] == 1


def test_model_digest_found_and_missing(monkeypatch):
    tags = {"models": [{"name": "qwen3.5:9b", "model": "qwen3.5:9b", "digest": "abc"}]}
    monkeypatch.setattr(OllamaClient, "_request", lambda self, path, payload=None: tags)
    client = OllamaClient()
    assert client.model_digest("qwen3.5:9b") == "abc"
    with pytest.raises(LookupError, match="ollama pull"):
        client.model_digest("missing:1b")
