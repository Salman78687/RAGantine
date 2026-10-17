"""Benchmark-only data contracts (SPEC §B3).

These carry ground truth (attack type, canary, planted false answer), so they live
in ``bench/`` and must never be imported by anything in ``ragantine/``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ragantine.schemas import Answer, RawDocument

AttackType = Literal["A1", "A2", "A3", "A4", "A5", "A6"]


class BenchQuestion(BaseModel):
    qid: str
    question: str
    gold_answer: str
    gold_aliases: list[str] = Field(default_factory=list)
    gold_chunk_id: str
    qset: Literal["target", "clean", "paraphrase"]
    split: Literal["dev", "test"]
    attack: AttackType | None = None  # target + paraphrase only
    parent_qid: str | None = None  # paraphrase -> its target

    @model_validator(mode="after")
    def _set_fields_consistent(self) -> BenchQuestion:
        if self.qset == "clean" and self.attack is not None:
            raise ValueError("clean questions are never targeted; attack must be None")
        if self.qset in ("target", "paraphrase") and self.attack is None:
            raise ValueError(f"{self.qset} questions need an attack type")
        if (self.qset == "paraphrase") != (self.parent_qid is not None):
            raise ValueError("parent_qid is required for paraphrase questions and only for them")
        return self


class PoisonSample(BaseModel):
    poison_id: str
    qid: str
    attack: AttackType
    goal: Literal["false_answer", "canary"]
    target_false_answer: str | None = None
    canary: str | None = None  # e.g. "ZEBRA-7741"
    template_id: str = Field(pattern=r"^A[1-6]-t\d{2}$")  # train/test templates must not overlap
    document: RawDocument  # what gets ingested
    seed: int = 0  # seeds vary template choice, false answers and canaries (SPEC §B5.6)
    technique: str | None = None  # A5 hiding technique; A6 part ("a"/"b")

    @model_validator(mode="after")
    def _goal_payload_present(self) -> PoisonSample:
        if self.goal == "false_answer" and not self.target_false_answer:
            raise ValueError("goal=false_answer requires target_false_answer")
        if self.goal == "canary" and not self.canary:
            raise ValueError("goal=canary requires canary")
        if not self.template_id.startswith(self.attack):
            raise ValueError(f"template {self.template_id} does not belong to attack {self.attack}")
        return self


class JudgedAnswer(BaseModel):
    qid: str
    run_id: str
    seed: int
    answer: Answer
    verdict: Literal["correct", "poisoned", "refused", "other"]
    judge_reason: str
    matched_by: Literal["rule", "judge"]
