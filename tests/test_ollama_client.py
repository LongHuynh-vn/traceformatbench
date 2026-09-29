import json
from io import BytesIO
from pathlib import Path

import pytest

from traceformatbench.runners.ollama import ChatResult, OllamaClient


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Tests must not connect to Ollama or the network.")

    monkeypatch.setattr("urllib.request.urlopen", fail)


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


@pytest.fixture
def chat_response():
    # Recorded example from Ollama's API documentation, not a benchmark output:
    # https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-structured-outputs
    path = Path(__file__).parent / "fixtures" / "ollama_chat_response.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("capabilities", [[], ["completion"], ["completion", "thinking"]])
def test_chat_http_payload_has_explicit_options_and_capability_gated_think(
    monkeypatch, chat_response, capabilities
):
    requests = []

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data)
        requests.append((request.full_url, request.method, payload, timeout))
        response = (
            {"capabilities": capabilities}
            if request.full_url.endswith("/api/show")
            else chat_response
        )
        return BytesIO(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = OllamaClient("http://mock", timeout_s=12).chat(
        "m:1b",
        "system text",
        "user text",
        num_ctx=4096,
        num_predict=256,
        temperature=0.25,
        seed=19,
        think=False,
    )
    assert requests[0] == ("http://mock/api/show", "POST", {"model": "m:1b"}, 12)
    url, method, payload, timeout = requests[1]
    assert (url, method, timeout) == ("http://mock/api/chat", "POST", 12)
    expected = {
        "model": "m:1b",
        "messages": [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "user text"},
        ],
        "stream": False,
        "truncate": False,
        "options": {"num_ctx": 4096, "num_predict": 256, "temperature": 0.25, "seed": 19},
    }
    if "thinking" in capabilities:
        expected["think"] = False
    assert payload == expected
    assert result.content == chat_response["message"]["content"]


@pytest.mark.parametrize("cached_count", [None, 8])
@pytest.mark.parametrize("thinking", [None, "private reasoning"])
def test_chat_parses_recorded_response_and_optional_fields(
    monkeypatch, chat_response, cached_count, thinking
):
    if cached_count is not None:
        chat_response["prompt_eval_cached_count"] = cached_count
    if thinking is not None:
        chat_response["message"]["thinking"] = thinking

    def fake_request(self, path, payload=None):
        return {"capabilities": ["thinking"]} if path == "/api/show" else chat_response

    monkeypatch.setattr(OllamaClient, "_request", fake_request)
    result = OllamaClient().chat("m", "system", "user", 32768, 256, 0, 7, False)
    assert result == ChatResult(
        content='{"age": 22, "available": false}',
        thinking=thinking or "",
        done_reason="stop",
        prompt_eval_count=34,
        prompt_eval_cached_count=cached_count or 0,
        eval_count=12,
        load_duration_ns=574751416,
        prompt_eval_duration_ns=1502000000,
        eval_duration_ns=175000000,
        total_duration_ns=2254970291,
    )


def test_capabilities_are_cached_and_callers_cannot_mutate_them(monkeypatch):
    calls = []

    def fake_request(self, path, payload=None):
        calls.append((path, payload))
        return {"capabilities": ["completion", "thinking"]}

    monkeypatch.setattr(OllamaClient, "_request", fake_request)
    client = OllamaClient()
    client.model_capabilities("m").clear()
    assert client.model_capabilities("m") == ["completion", "thinking"]
    assert calls == [("/api/show", {"model": "m"})]


def test_unload_model_does_not_generate_an_answer(monkeypatch):
    calls = []
    monkeypatch.setattr(
        OllamaClient, "_request", lambda self, path, payload: calls.append((path, payload))
    )
    OllamaClient().unload_model("m")
    assert calls == [("/api/generate", {"model": "m", "keep_alive": 0})]
