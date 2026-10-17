"""Deterministic offline stand-ins for the networked/heavy components (SPEC §B8).

``FakeLLM`` implements ``LLMClient`` with scripted responses keyed by a substring of the
prompt, so graph, defense and benchmark logic run without network. ``HashingEmbedder``
and ``OverlapReranker`` replace the bge models in unit/component tests and the CI smoke run.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

import numpy as np

from ragantine.config import ModelDefaults, ModelEntry, ModelsConfig, ProviderConfig
from ragantine.generate.llm import LLMClient, Message, _Raw

_WORD = re.compile(r"[a-z0-9]+")
Responder = str | Callable[[Sequence[Message]], str]


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def fake_models_config(aliases: Sequence[str] = ("fake",)) -> ModelsConfig:
    """A ModelsConfig whose aliases all point at an offline 'fake' provider."""
    return ModelsConfig(
        providers={"fake": ProviderConfig(base_url="http://fake.invalid", api_key_env="FAKE_KEY")},
        models={a: ModelEntry(provider="fake", model=f"fake/{a}") for a in aliases},
        defaults=ModelDefaults(
            generator=aliases[0],
            judge=aliases[-1],
            poison_writer=aliases[0],
            temperature=0.0,
            max_tokens=400,
            max_concurrency=4,
        ),
    )


class FakeLLM(LLMClient):
    """LLMClient returning scripted responses: first rule whose substring is in the prompt wins."""

    def __init__(
        self,
        rules: Sequence[tuple[str, Responder]] = (),
        default: Responder = "I don't know.",
        models: ModelsConfig | None = None,
    ) -> None:
        super().__init__(models or fake_models_config(), cache_dir=None, use_cache=False)
        self.rules = list(rules)
        self.default = default
        self.prompts: list[list[Message]] = []

    def respond(self, messages: Sequence[Message]) -> str:
        self.prompts.append(list(messages))
        prompt = "\n".join(m.get("content", "") for m in messages)
        for needle, responder in self.rules:
            if needle in prompt:
                return responder if isinstance(responder, str) else responder(messages)
        return self.default if isinstance(self.default, str) else self.default(messages)

    def _raw(self, messages: Sequence[Message]) -> _Raw:
        text = self.respond(messages)
        prompt_tokens = sum(len(tokenize(m.get("content", ""))) for m in messages)
        return _Raw(text=text, prompt_tokens=prompt_tokens, completion_tokens=len(tokenize(text)))

    def _complete(
        self, model_alias: str, messages: Sequence[Message], params: dict[str, Any]
    ) -> _Raw:
        return self._raw(messages)

    async def _acomplete(
        self, model_alias: str, messages: Sequence[Message], params: dict[str, Any]
    ) -> _Raw:
        return self._raw(messages)

    async def _stream(
        self, model_alias: str, messages: Sequence[Message], params: dict[str, Any]
    ) -> AsyncIterator[str | _Raw]:
        raw = self._raw(messages)
        for i, word in enumerate(raw.text.split(" ")):
            yield word if i == 0 else " " + word
        yield raw


class HashingEmbedder:
    """Signed feature-hashing bag-of-words embedder (L2-normalised, deterministic)."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dimension, dtype=np.float32)
        for tok in tokenize(text):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
            v[h % self.dimension] += 1.0 if (h >> 63) & 1 else -1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._vec(query)


class OverlapReranker:
    """Scores (query, passage) by weighted overlap of query words found in the passage.

    Capitalised query words (names, the question's subject) weigh more, a crude stand-in
    for how a cross-encoder focuses on the entity being asked about.
    """

    SUBJECT_WEIGHT = 5.0

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        weights: dict[str, float] = {}
        for i, word in enumerate(re.findall(r"[A-Za-z0-9]+", query)):
            w = self.SUBJECT_WEIGHT if i > 0 and word[0].isupper() else 1.0
            tok = word.lower()
            weights[tok] = max(weights.get(tok, 0.0), w)
        total = sum(weights.values())
        if not total:
            return [0.0 for _ in texts]
        return [
            sum(w for t, w in weights.items() if t in set(tokenize(text))) / total for text in texts
        ]


INJECTION_CUES = re.compile(
    r"\b(ai (assistant|system|model|readers?)|assistants?|chatbots?|language models?|automated (readers?|assistants?)|"
    r"instruction|directive|ignore (all|previous|the)|system note|append|include .{0,40}\bcode\b)",
    re.I,
)


class KeywordInjectionScorer:
    """Heuristic stand-in for the D2 classifier (offline tests and CI smoke only).

    Flags text addressed to AI systems. It is not the benchmarked D2 model.
    """

    def score(self, texts: Sequence[str]) -> list[float]:
        return [0.97 if INJECTION_CUES.search(t) else 0.03 for t in texts]


_STOP = frozenset(
    {
        "the", "a", "an", "of", "in", "on", "at", "to", "is", "was", "were", "be", "by", "for",
        "and", "or", "as", "with", "from", "that", "this", "it", "its", "when", "what", "where",
        "who", "whom", "which", "how", "why", "did", "does", "do", "he", "she", "they", "there",
    }
)  # fmt: skip


class LexicalNLI:
    """Heuristic stand-in for the D5 NLI cross-encoder (offline tests and CI smoke only).

    Entailment when every content word of the hypothesis occurs in the premise;
    contradiction when the premise covers the hypothesis' words but not its numbers.
    """

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        out = []
        for premise, hypothesis in pairs:
            prem = set(tokenize(premise))
            hyp = [t for t in tokenize(hypothesis) if t not in _STOP]
            nums = [t for t in hyp if t.isdigit()]
            words = [t for t in hyp if not t.isdigit()]
            if hyp and all(t in prem for t in hyp):
                out.append({"entailment": 0.95, "neutral": 0.04, "contradiction": 0.01})
            elif (
                nums
                and words
                and sum(t in prem for t in words) / len(words) >= 0.6
                and not all(n in prem for n in nums)
            ):
                out.append({"entailment": 0.02, "neutral": 0.08, "contradiction": 0.90})
            else:
                out.append({"entailment": 0.05, "neutral": 0.90, "contradiction": 0.05})
        return out
