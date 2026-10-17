"""LLMClient: provider wiring, retries, cache, token accounting (SPEC §B4.4).

The network is replaced by ``httpx.MockTransport``; no request leaves the process.
"""

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from tenacity import wait_none

from ragantine.config import ModelDefaults, ModelEntry, ModelsConfig, ProviderConfig, Settings
from ragantine.generate.llm import LLMClient, MissingAPIKeyError

BASE = "https://llm.test/v1"
MSGS = [{"role": "user", "content": "hello"}]


def _models(**entry_kw: object) -> ModelsConfig:
    return ModelsConfig(
        providers={"p": ProviderConfig(base_url=BASE, api_key_env="TEST_LLM_KEY")},
        models={"m": ModelEntry(provider="p", model="model-x", **entry_kw)},  # type: ignore[arg-type]
        defaults=ModelDefaults(
            generator="m",
            judge="m",
            poison_writer="m",
            temperature=0.0,
            max_tokens=50,
            max_concurrency=2,
        ),
    )


def _completion(text: str, p: int = 11, c: int = 3) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "model-x",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c},
        },
    )


def _error(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message}})


class Recorder:
    """Scripted transport handler: returns the queued responses in order and records requests."""

    def __init__(self, *responses: httpx.Response) -> None:
        self._responses: Iterator[httpx.Response] = iter(responses)
        self.requests: list[httpx.Request] = []
        self.last: httpx.Response | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.last = next(self._responses, self.last)
        assert self.last is not None
        return self.last

    def body(self, i: int = 0) -> dict[str, object]:
        data: dict[str, object] = json.loads(self.requests[i].content)
        return data


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_LLM_KEY", "test-not-a-real-key")


Factory = Callable[..., LLMClient]


@pytest.fixture
def make(tmp_path: Path) -> Factory:
    def factory(rec: Recorder, cache: bool = True, **entry_kw: object) -> LLMClient:
        return LLMClient(
            _models(**entry_kw),
            Settings(_env_file=None),
            cache_dir=tmp_path / "cache" if cache else None,
            transport=httpx.MockTransport(rec),
            async_transport=httpx.MockTransport(rec),
            retry_wait=wait_none(),
        )

    return factory


def test_chat_returns_text_tokens_and_caches(make: Factory) -> None:
    rec = Recorder(_completion("hi"))
    client = make(rec)
    first = client.chat("m", MSGS)
    assert (first.text, first.prompt_tokens, first.completion_tokens, first.cached) == (
        "hi",
        11,
        3,
        False,
    )
    second = client.chat("m", MSGS)
    assert second.cached and second.text == "hi"
    assert len(rec.requests) == 1 and client.cache_hits == 1
    assert rec.requests[0].headers["authorization"] == "Bearer test-not-a-real-key"
    assert rec.body() | {"messages": None} == rec.body() | {
        "messages": None,
        "model": "model-x",
        "max_completion_tokens": 50,
        "temperature": 0.0,
    }


def test_nonzero_temperature_is_not_cached(make: Factory) -> None:
    rec = Recorder(_completion("hi"))
    client = make(rec)
    client.chat("m", MSGS, temperature=0.7)
    client.chat("m", MSGS, temperature=0.7)
    assert len(rec.requests) == 2


def test_retries_on_429_and_5xx_then_succeeds(make: Factory) -> None:
    rec = Recorder(_error(429, "slow down"), _error(503, "busy"), _completion("ok"))
    assert make(rec, cache=False).chat("m", MSGS).text == "ok"
    assert len(rec.requests) == 3


def test_gives_up_after_five_attempts(make: Factory) -> None:
    rec = Recorder(_error(500, "down"))
    with pytest.raises(Exception, match="down"):
        make(rec, cache=False).chat("m", MSGS)
    assert len(rec.requests) == 5


def test_does_not_retry_client_errors(make: Factory) -> None:
    rec = Recorder(_error(401, "bad key"))
    with pytest.raises(Exception, match="bad key"):
        make(rec).chat("m", MSGS)
    assert len(rec.requests) == 1


def test_drops_temperature_when_model_rejects_it(make: Factory) -> None:
    rec = Recorder(_error(400, "Unsupported parameter: 'temperature'"), _completion("ok"))
    client = make(rec, cache=False)
    assert client.chat("m", MSGS).text == "ok"
    assert "temperature" not in rec.body(1)
    client.chat("m", MSGS)  # remembered: no failing round-trip the second time
    assert len(rec.requests) == 3 and "temperature" not in rec.body(2)


def test_extra_params_and_json_mode_are_sent(make: Factory) -> None:
    rec = Recorder(_completion("{}"))
    make(rec, extra_params={"reasoning_effort": "low"}).chat("m", MSGS, json_mode=True)
    body = rec.body()
    assert body["reasoning_effort"] == "low"
    assert body["response_format"] == {"type": "json_object"}


async def test_async_chat(make: Factory) -> None:
    rec = Recorder(_completion("async"))
    assert (await make(rec).achat("m", MSGS, temperature=0.5)).text == "async"


async def test_stream_yields_deltas_then_result(make: Factory) -> None:
    def chunk(delta: str | None, usage: dict[str, int] | None = None) -> str:
        payload: dict[str, object] = {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "model-x",
        }
        payload["choices"] = [] if delta is None else [{"index": 0, "delta": {"content": delta}}]
        if usage:
            payload["usage"] = usage
        return f"data: {json.dumps(payload)}\n\n"

    sse = (
        chunk("Hel")
        + chunk("lo")
        + chunk(None, {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})
    )
    rec = Recorder(
        httpx.Response(
            200, text=sse + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"}
        )
    )
    items = [i async for i in make(rec, cache=False).astream("m", MSGS)]
    assert items[:-1] == ["Hel", "lo"]
    final = items[-1]
    assert not isinstance(final, str)
    assert (final.text, final.prompt_tokens, final.completion_tokens) == ("Hello", 5, 2)
    assert rec.body()["stream"] is True


def test_missing_or_placeholder_key_raises(monkeypatch: pytest.MonkeyPatch, make: Factory) -> None:
    monkeypatch.setenv("TEST_LLM_KEY", "<your-key>")
    with pytest.raises(MissingAPIKeyError, match="TEST_LLM_KEY"):
        make(Recorder(_completion("x"))).chat("m", MSGS)


def test_unknown_alias(make: Factory) -> None:
    with pytest.raises(KeyError, match="unknown model alias"):
        make(Recorder(_completion("x"))).chat("nope", MSGS)
