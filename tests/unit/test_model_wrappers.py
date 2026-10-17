"""HF wrappers for D2/D5 with the heavy libraries replaced by tiny fakes (no downloads)."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from ragantine.defenses import d2_classifier, d5_consistency
from tests.defense_helpers import chunk, defense, hit


class FakeTok:
    def __call__(self, texts: list[str], **_: Any) -> Any:
        class Enc(dict[str, Any]):
            def to(self, _device: str) -> "Enc":
                return self

        return Enc(input_ids=torch.tensor([[len(t)] for t in texts]))


class FakeSeqModel(torch.nn.Module):
    config = SimpleNamespace(id2label={0: "SAFE", 1: "INJECTION"})

    def forward(self, input_ids: torch.Tensor) -> Any:
        # long texts look like injections in this fake
        logits = torch.stack(
            [torch.zeros(len(input_ids)), (input_ids[:, 0] > 20).float() * 6 - 3], dim=1
        )
        return SimpleNamespace(logits=logits)


def test_hf_injection_scorer_reads_label_and_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda _n: FakeTok())
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda _n: FakeSeqModel(),
    )
    scorer = d2_classifier.HFInjectionScorer("org/fake")
    scores = scorer.score(["short", "a much longer text that looks like an injection"])
    assert scorer.positive_index == 1
    assert scores[0] < 0.1 and scores[1] > 0.9


def test_d2_builds_hf_scorer_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[str] = []
    monkeypatch.setattr(
        d2_classifier,
        "HFInjectionScorer",
        lambda name: built.append(name) or SimpleNamespace(score=lambda t: [0.0] * len(t)),
    )
    d2 = defense("d2", variant="zeroshot")
    kept, _ = d2.on_chunks([chunk("a", "fine")])
    assert built == ["protectai/deberta-v3-base-prompt-injection-v2"] and len(kept) == 1
    assert d2.on_chunks([]) == ([], [])
    assert d2.on_retrieve("q", []) == ([], [])


class FakeCrossEncoder:
    def __init__(self, name: str, device: str) -> None:
        self.model = SimpleNamespace(
            config=SimpleNamespace(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})
        )

    def predict(self, pairs: list[tuple[str, str]], **_: Any) -> Any:
        import numpy as np

        return np.array(
            [[0.1, 0.8, 0.1] if h.split()[-1] in p else [0.7, 0.1, 0.2] for p, h in pairs]
        )


def test_cross_encoder_nli_uses_model_label_order(monkeypatch: pytest.MonkeyPatch) -> None:
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    nli = d5_consistency.CrossEncoderNLI("cross-encoder/fake")
    assert nli.predict([]) == []
    out = nli.predict([("built in 1889", "it was 1889"), ("built in 1889", "it was 1901")])
    assert out[0]["entailment"] == pytest.approx(0.8) and out[1]["contradiction"] == pytest.approx(
        0.7
    )


def test_d5_builds_nli_lazily_and_needs_ctx_to_regenerate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        d5_consistency,
        "CrossEncoderNLI",
        lambda name: SimpleNamespace(predict=lambda pairs: [{"neutral": 1.0} for _ in pairs]),
    )
    d5 = defense("d5")
    text, decisions = d5.on_verify(
        "q?", [hit(chunk("a", "x"), 1.0)], "Unsupported claim here.", ctx=None
    )
    assert (
        text == d5_consistency.UNSUPPORTED_ANSWER
        and "no regeneration context" in decisions[0].reason
    )
