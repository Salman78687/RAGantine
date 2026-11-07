"""D5 cross-source consistency check (SPEC §B6).

1. Split the draft answer into sentences.
2. Run the NLI cross-encoder for each sentence against every final chunk
   (premise = chunk, hypothesis = sentence); label order comes from the model config.
3. A sentence is *supported* if entailment >= ``support_threshold`` in chunks from
   >= ``min_supporting_docs`` distinct doc_ids, or in at least one ``curated`` chunk.
4. All supported -> allow. An unsupported sentence backed only by web/user_upload chunks
   -> regenerate once without them (``rewrite``). A higher-tier chunk contradicting the
   draft -> regenerate with a note that sources conflict (``hedge``). Still unsupported ->
   "I don't know based on the trusted sources."

Short sentences ("1953.") are checked as "<question> <sentence>" so the NLI model sees
what the sentence claims.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol

from ragantine.defenses.base import Defense, QueryContext, register
from ragantine.generate.prompting import load_prompt
from ragantine.ingest.chunker import split_sentences
from ragantine.local_models import serialized
from ragantine.schemas import DefenseDecision, RetrievedChunk, SourceTier

UNSUPPORTED_ANSWER = "I don't know based on the trusted sources."
LOW_TIERS = frozenset({SourceTier.WEB, SourceTier.USER_UPLOAD})
TIER_RANK = {
    SourceTier.CURATED: 3,
    SourceTier.PARTNER: 2,
    SourceTier.WEB: 1,
    SourceTier.USER_UPLOAD: 0,
}
SHORT_SENTENCE_WORDS = 6
_CITATION = re.compile(r"\s*\[[^\]]+\]")
_REFUSAL = re.compile(r"\bi (do not|don't) know\b", re.I)


class NLIModel(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        """(premise, hypothesis) -> probabilities keyed by label name (entailment/neutral/contradiction)."""
        ...


class CrossEncoderNLI:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._labels: list[str] = []

    def _load(self) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_name, device=self.device)
        id2label = self._model.model.config.id2label  # read the label order, never hard-code it
        self._labels = [str(id2label[i]).lower() for i in range(len(id2label))]

    @serialized
    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[dict[str, float]]:
        if not pairs:
            return []
        if self._model is None:
            self._load()
        probs = self._model.predict(list(pairs), apply_softmax=True, convert_to_numpy=True)
        return [{label: float(p[i]) for i, label in enumerate(self._labels)} for p in probs]


def claim_sentences(draft: str) -> list[str]:
    text = _CITATION.sub("", draft).strip()
    return [s for s in split_sentences(text) if re.search(r"\w", s)]


@register
class ConsistencyCheck(Defense):
    id = "d5"

    def __init__(self, cfg: dict[str, Any], llm: Any = None) -> None:
        super().__init__(cfg, llm)
        self.threshold = float(cfg["support_threshold"])
        self.min_docs = int(cfg["min_supporting_docs"])
        self.min_relevance = float(cfg.get("contradiction_min_relevance", 0.0))
        self.allow_uncontested = bool(cfg.get("allow_uncontested", False))
        self._nli: NLIModel | None = None

    @property
    def nli(self) -> NLIModel:
        if self._nli is None:
            self._nli = CrossEncoderNLI(str(self.cfg["nli_model"]))
        return self._nli

    @nli.setter
    def nli(self, value: NLIModel) -> None:
        self._nli = value

    # ------------------------------------------------------------------ analysis

    def analyse(self, q: str, hits: Sequence[RetrievedChunk], draft: str) -> list[dict[str, Any]]:
        """Per sentence: which chunks entail it and which contradict it."""
        sentences = claim_sentences(draft)
        hyps = [f"{q} {s}" if len(s.split()) < SHORT_SENTENCE_WORDS else s for s in sentences]
        pairs = [(h.chunk.text, hyp) for hyp in hyps for h in hits]
        probs = self.nli.predict(pairs)
        out = []
        for i, sentence in enumerate(sentences):
            row = probs[i * len(hits) : (i + 1) * len(hits)]
            entail = [
                h
                for h, p in zip(hits, row, strict=True)
                if p.get("entailment", 0.0) >= self.threshold
            ]
            contra = [
                h
                for h, p in zip(hits, row, strict=True)
                if p.get("contradiction", 0.0) >= self.threshold and self._relevant(h)
            ]
            docs = {h.chunk.doc_id for h in entail}
            supported = (
                bool(_REFUSAL.search(sentence))
                or len(docs) >= self.min_docs
                or any(h.chunk.source_tier is SourceTier.CURATED for h in entail)
                or (self.allow_uncontested and not entail and not contra)
            )
            out.append(
                {"sentence": sentence, "entail": entail, "contra": contra, "supported": supported}
            )
        return out

    def _relevant(self, hit: RetrievedChunk) -> bool:
        """Off-topic passages cannot contradict a claim (NLI calls unrelated facts contradictions)."""
        return hit.rerank_score is None or hit.rerank_score >= self.min_relevance

    def _all_supported(self, q: str, hits: Sequence[RetrievedChunk], text: str) -> bool:
        return all(a["supported"] for a in self.analyse(q, hits, text))

    # ------------------------------------------------------------------ hook

    def on_verify(
        self,
        q: str,
        hits: list[RetrievedChunk],
        draft: str,
        ctx: QueryContext | None = None,
    ) -> tuple[str, list[DefenseDecision]]:
        analysis = self.analyse(q, hits, draft)
        if all(a["supported"] for a in analysis):
            return draft, [
                self.decision(
                    stage="verification",
                    target_id="answer",
                    action="allow",
                    reason="all sentences supported",
                )
            ]
        if ctx is None:
            return UNSUPPORTED_ANSWER, [self._fallback("no regeneration context")]

        unsupported = [a for a in analysis if not a["supported"]]
        low_only = [
            a
            for a in unsupported
            if a["entail"] and all(h.chunk.source_tier in LOW_TIERS for h in a["entail"])
        ]
        draft_tiers = [TIER_RANK[h.chunk.source_tier] for a in unsupported for h in a["entail"]]
        top_support = max(draft_tiers, default=-1)
        contradicted = [
            a
            for a in unsupported
            if any(TIER_RANK[h.chunk.source_tier] > top_support for h in a["contra"])
        ]

        if low_only:
            trusted = [h for h in hits if h.chunk.source_tier not in LOW_TIERS]
            redo = ctx.regenerate(trusted, None)
            decision = self.decision(
                stage="verification", target_id="answer", action="rewrite",
                reason=f"unsupported sentence backed only by web/user_upload: {low_only[0]['sentence'][:80]!r}",
            )  # fmt: skip
            if self._all_supported(q, trusted, redo.text):
                return redo.text, [decision]
            return UNSUPPORTED_ANSWER, [decision, self._fallback("still unsupported after rewrite")]
        if contradicted:
            note = load_prompt("d5_conflict_note").strip()
            redo = ctx.regenerate(hits, note)
            decision = self.decision(
                stage="verification", target_id="answer", action="hedge",
                reason=f"higher-tier source contradicts: {contradicted[0]['sentence'][:80]!r}",
            )  # fmt: skip
            if self._all_supported(q, hits, redo.text):
                return redo.text, [decision]
            return UNSUPPORTED_ANSWER, [decision, self._fallback("still unsupported after hedge")]
        return UNSUPPORTED_ANSWER, [
            self._fallback(f"unsupported: {unsupported[0]['sentence'][:80]!r}")
        ]

    def _fallback(self, why: str) -> DefenseDecision:
        return self.decision(
            stage="verification",
            target_id="answer",
            action="hedge",
            reason=f"unsupported_answer: {why}",
        )
