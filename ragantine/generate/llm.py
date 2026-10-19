"""``LLMClient``: OpenAI-compatible chat with retries, disk cache and token accounting (SPEC §B4.4).

Every LLM call in the project goes through this class. Providers are reached with the
``openai`` SDK pointed at each provider's ``base_url``. Responses at temperature 0 are
cached on disk, keyed on sha256(provider, model id, messages, params).
Subclasses (``ragantine.fakes.FakeLLM``) override only the raw ``_complete`` / ``_stream``
calls, so caching and accounting behave the same in tests.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, cast

import httpx
import openai
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    retry_if_exception,
    wait_exponential,
)

from ragantine.config import ModelEntry, ModelsConfig, Settings

Message = dict[str, str]
RETRY_ATTEMPTS = 5


class MissingAPIKeyError(RuntimeError):
    """Raised when a provider key is unset or still a ``<placeholder>`` in .env."""


class LLMResult(BaseModel):
    text: str
    model_alias: str
    model_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    cached: bool = False


class _Raw(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, openai.RateLimitError | openai.APIConnectionError | openai.APITimeoutError):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500


MAX_RETRY_AFTER_S = 120.0


def retry_after_seconds(exc: BaseException | None) -> float | None:
    """Seconds the provider asked us to wait (Retry-After header on a 429), if any."""
    if not isinstance(exc, openai.RateLimitError):
        return None
    value = exc.response.headers.get("retry-after") if exc.response is not None else None
    try:
        return min(MAX_RETRY_AFTER_S, max(0.0, float(value))) if value is not None else None
    except ValueError:
        return None


class RetryAfterWait:
    """Wait what the provider asked for on 429s; otherwise fall back to exponential backoff."""

    def __init__(self, fallback: Any) -> None:
        self.fallback = fallback

    def __call__(self, state: RetryCallState) -> float:
        exc = state.outcome.exception() if state.outcome else None
        seconds = retry_after_seconds(exc)
        return seconds if seconds is not None else float(self.fallback(state))


def _rejects_temperature(exc: BaseException) -> bool:
    return (
        isinstance(exc, openai.BadRequestError)
        and "temperature" in str(exc).lower()
        and ("unsupported" in str(exc).lower() or "not support" in str(exc).lower())
    )


class LLMClient:
    """Chat client over OpenAI-compatible providers with cache, retries and token accounting."""

    def __init__(
        self,
        models: ModelsConfig,
        settings: Settings | None = None,
        cache_dir: Path | None = None,
        use_cache: bool = True,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
        retry_wait: Any = None,
    ) -> None:
        """``transport``/``async_transport`` replace the network (tests); ``retry_wait`` the backoff."""
        self.models = models
        self._transport = transport
        self._async_transport = async_transport
        self._retry_wait = retry_wait or wait_exponential(multiplier=1, min=1, max=30)
        self.settings = settings or Settings()
        self.use_cache = use_cache and cache_dir is not None
        self._cache: Any = None
        if self.use_cache and cache_dir is not None:
            import diskcache

            self._cache = diskcache.Cache(str(cache_dir))
        self._sync_clients: dict[str, openai.OpenAI] = {}
        # Async clients and semaphores are bound to an event loop, so keep one per loop.
        self._async_clients: dict[tuple[int, str], openai.AsyncOpenAI] = {}
        self._semaphores: dict[int, asyncio.Semaphore] = {}
        self._no_temperature: set[str] = set()
        self.cache_hits = 0
        self.calls = 0

    # ------------------------------------------------------------------ public API

    def entry(self, model_alias: str) -> ModelEntry:
        try:
            return self.models.models[model_alias]
        except KeyError:
            known = ", ".join(sorted(self.models.models))
            raise KeyError(f"unknown model alias {model_alias!r} (known: {known})") from None

    def chat(
        self,
        model_alias: str,
        messages: Sequence[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        """Blocking chat completion (cached at temperature 0)."""
        params = self._params(model_alias, temperature, max_tokens, json_mode)
        key = self._cache_key(model_alias, messages, params)
        if (hit := self._cache_get(key, model_alias)) is not None:
            return hit
        start = time.perf_counter()
        raw = self._with_retries_sync(model_alias, messages, params)
        return self._finish(key, model_alias, raw, start)

    async def achat(
        self,
        model_alias: str,
        messages: Sequence[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        """Async chat completion, bounded by ``max_concurrency`` (cached at temperature 0)."""
        params = self._params(model_alias, temperature, max_tokens, json_mode)
        key = self._cache_key(model_alias, messages, params)
        if (hit := self._cache_get(key, model_alias)) is not None:
            return hit
        async with self._sem():
            start = time.perf_counter()
            raw = await self._with_retries_async(model_alias, messages, params)
        return self._finish(key, model_alias, raw, start)

    async def astream(
        self,
        model_alias: str,
        messages: Sequence[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str | LLMResult]:
        """Yield text deltas, then the final ``LLMResult``. A cache hit yields one delta."""
        params = self._params(model_alias, temperature, max_tokens, json_mode=False)
        key = self._cache_key(model_alias, messages, params)
        if (hit := self._cache_get(key, model_alias)) is not None:
            yield hit.text
            yield hit
            return
        async with self._sem():
            start = time.perf_counter()
            parts: list[str] = []
            usage = _Raw(text="")
            async for item in self._stream(model_alias, messages, params):
                if isinstance(item, str):
                    parts.append(item)
                    yield item
                else:
                    usage = item
        raw = _Raw(
            text="".join(parts),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        yield self._finish(key, model_alias, raw, start)

    # ------------------------------------------------------------------ raw provider calls

    def _complete(
        self, model_alias: str, messages: Sequence[Message], params: dict[str, Any]
    ) -> _Raw:
        entry = self.entry(model_alias)
        resp = self._sync_client(entry.provider).chat.completions.create(
            model=entry.model,
            messages=cast(Any, list(messages)),
            **params,
        )
        return self._raw_from_response(resp)

    async def _acomplete(
        self, model_alias: str, messages: Sequence[Message], params: dict[str, Any]
    ) -> _Raw:
        entry = self.entry(model_alias)
        resp = await self._async_client(entry.provider).chat.completions.create(
            model=entry.model,
            messages=cast(Any, list(messages)),
            **params,
        )
        return self._raw_from_response(resp)

    async def _stream(
        self, model_alias: str, messages: Sequence[Message], params: dict[str, Any]
    ) -> AsyncIterator[str | _Raw]:
        entry = self.entry(model_alias)
        stream = await self._async_client(entry.provider).chat.completions.create(
            model=entry.model,
            messages=cast(Any, list(messages)),
            stream=True,
            stream_options={"include_usage": True},
            **params,
        )
        usage = _Raw(text="")
        async for event in stream:
            if event.choices and event.choices[0].delta and event.choices[0].delta.content:
                yield event.choices[0].delta.content
            if getattr(event, "usage", None):
                usage = _Raw(
                    text="",
                    prompt_tokens=event.usage.prompt_tokens or 0,
                    completion_tokens=event.usage.completion_tokens or 0,
                )
        yield usage

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _raw_from_response(resp: Any) -> _Raw:
        text = resp.choices[0].message.content or "" if resp.choices else ""
        usage = getattr(resp, "usage", None)
        return _Raw(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    def _params(
        self, model_alias: str, temperature: float | None, max_tokens: int | None, json_mode: bool
    ) -> dict[str, Any]:
        entry = self.entry(model_alias)
        defaults = self.models.defaults
        params: dict[str, Any] = {
            "temperature": defaults.temperature if temperature is None else temperature,
            "max_completion_tokens": max_tokens or defaults.max_tokens,
        }
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        if entry.extra_params:
            # provider-specific options (e.g. Groq's include_reasoning) are not SDK keyword
            # arguments; extra_body merges them into the request JSON
            params["extra_body"] = dict(entry.extra_params)
        return params

    def _send_params(self, model_alias: str, params: dict[str, Any]) -> dict[str, Any]:
        """Params actually sent: drops temperature for models that reject it."""
        entry = self.entry(model_alias)
        if not entry.supports_temperature or model_alias in self._no_temperature:
            return {k: v for k, v in params.items() if k != "temperature"}
        return params

    def _cache_key(
        self, model_alias: str, messages: Sequence[Message], params: dict[str, Any]
    ) -> str | None:
        if not self.use_cache or params.get("temperature") != 0:
            return None
        entry = self.entry(model_alias)
        blob = json.dumps(
            {
                "provider": entry.provider,
                "model": entry.model,
                "messages": list(messages),
                **params,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_get(self, key: str | None, model_alias: str) -> LLMResult | None:
        if key is None or self._cache is None:
            return None
        hit = self._cache.get(key)
        if hit is None:
            return None
        self.cache_hits += 1
        result = LLMResult.model_validate_json(hit)
        return result.model_copy(
            update={"cached": True, "latency_ms": 0, "model_alias": model_alias}
        )

    def _finish(self, key: str | None, model_alias: str, raw: _Raw, start: float) -> LLMResult:
        self.calls += 1
        result = LLMResult(
            text=raw.text,
            model_alias=model_alias,
            model_id=self.entry(model_alias).model,
            prompt_tokens=raw.prompt_tokens,
            completion_tokens=raw.completion_tokens,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
        if key is not None and self._cache is not None:
            self._cache.set(key, result.model_dump_json())
        return result

    def _with_retries_sync(
        self, model_alias: str, messages: Sequence[Message], params: dict[str, Any]
    ) -> _Raw:
        for attempt in Retrying(**self._retry_kwargs()):
            with attempt:
                try:
                    return self._complete(
                        model_alias, messages, self._send_params(model_alias, params)
                    )
                except openai.BadRequestError as exc:
                    if not _rejects_temperature(exc):
                        raise
                    self._no_temperature.add(model_alias)
                    return self._complete(
                        model_alias, messages, self._send_params(model_alias, params)
                    )
        raise AssertionError("unreachable")  # pragma: no cover

    async def _with_retries_async(
        self, model_alias: str, messages: Sequence[Message], params: dict[str, Any]
    ) -> _Raw:
        async for attempt in AsyncRetrying(**self._retry_kwargs()):
            with attempt:
                try:
                    return await self._acomplete(
                        model_alias, messages, self._send_params(model_alias, params)
                    )
                except openai.BadRequestError as exc:
                    if not _rejects_temperature(exc):
                        raise
                    self._no_temperature.add(model_alias)
                    return await self._acomplete(
                        model_alias, messages, self._send_params(model_alias, params)
                    )
        raise AssertionError("unreachable")  # pragma: no cover

    def _retry_kwargs(self) -> dict[str, Any]:
        rate_limit_attempts = self.models.defaults.rate_limit_attempts

        def stop(state: RetryCallState) -> bool:
            exc = state.outcome.exception() if state.outcome else None
            limit = (
                rate_limit_attempts if isinstance(exc, openai.RateLimitError) else RETRY_ATTEMPTS
            )
            return state.attempt_number >= limit

        return {
            "retry": retry_if_exception(_retryable),
            "stop": stop,
            "wait": RetryAfterWait(self._retry_wait),
            "reraise": True,
        }

    def _sem(self) -> asyncio.Semaphore:
        loop_id = id(asyncio.get_running_loop())
        if loop_id not in self._semaphores:
            self._semaphores[loop_id] = asyncio.Semaphore(self.models.defaults.max_concurrency)
        return self._semaphores[loop_id]

    def _api_key(self, provider: str) -> str:
        env_name = self.models.providers[provider].api_key_env
        key = self.settings.api_key(env_name)
        if key is None:
            raise MissingAPIKeyError(
                f"{env_name} is not set (or is still a placeholder) in .env; "
                f"it is needed for provider {provider!r}"
            )
        return key

    def _sync_client(self, provider: str) -> openai.OpenAI:
        if provider not in self._sync_clients:
            self._sync_clients[provider] = openai.OpenAI(
                base_url=self.models.providers[provider].base_url,
                api_key=self._api_key(provider),
                max_retries=0,  # tenacity owns retries
                # the SDK accepts classic httpx clients at runtime; its stubs only name httpx2
                http_client=cast(Any, httpx.Client(transport=self._transport))
                if self._transport
                else None,
            )
        return self._sync_clients[provider]

    def _async_client(self, provider: str) -> openai.AsyncOpenAI:
        key = (id(asyncio.get_running_loop()), provider)
        if key not in self._async_clients:
            self._async_clients[key] = openai.AsyncOpenAI(
                base_url=self.models.providers[provider].base_url,
                api_key=self._api_key(provider),
                max_retries=0,
                http_client=(
                    cast(Any, httpx.AsyncClient(transport=self._async_transport))
                    if self._async_transport
                    else None
                ),
            )
        return self._async_clients[key]
