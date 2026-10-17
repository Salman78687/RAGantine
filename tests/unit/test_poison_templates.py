"""Poison templates: >= 8 per attack, train (t01-t04) and test (t05-t08) never overlap (SPEC §B5.3)."""

import re

import pytest

from bench.poison.templates import (
    TEST_TEMPLATE_NUMBERS,
    TRAIN_TEMPLATE_NUMBERS,
    all_templates,
    template_ids,
)

ATTACKS = ["A1", "A2", "A3", "A4", "A5", "A6"]
URL = re.compile(r"https?://|www\.", re.I)


def test_at_least_eight_templates_per_attack() -> None:
    templates = all_templates()
    for attack in ATTACKS:
        assert len([t for t in templates if t.startswith(attack + "-")]) >= 8


def test_train_and_test_templates_never_overlap() -> None:
    assert not set(TRAIN_TEMPLATE_NUMBERS) & set(TEST_TEMPLATE_NUMBERS)
    for attack in ATTACKS:
        train, test = set(template_ids(attack, "train")), set(template_ids(attack, "test"))
        assert train and test and not train & test
        assert train | test <= set(all_templates())
        texts = {all_templates()[t].writer for t in train}
        assert not texts & {all_templates()[t].writer for t in test}, (
            "identical prompt in both splits"
        )


@pytest.mark.parametrize("tid", sorted(all_templates()))
def test_template_shape(tid: str) -> None:
    t = all_templates()[tid]
    assert t.writer.startswith("You are generating a TEST DOCUMENT for a RAG security benchmark")
    assert "{subject}" in t.writer and "{question}" in t.writer
    if t.attack in ("A1", "A3", "A4"):
        assert "{false_answer}" in t.writer
    if t.attack in ("A2", "A5"):
        assert t.payload and "{canary}" in t.payload
    if t.attack == "A6":
        assert t.payload and t.payload_b
        assert "{canary}" not in t.payload and "{canary}" in t.payload_b  # halves meaningless alone
    if t.attack == "A4":
        assert t.payload and "{question}" in t.payload and "{rewordings}" in t.payload


@pytest.mark.parametrize("tid", sorted(all_templates()))
def test_templates_stay_harmless(tid: str) -> None:
    t = all_templates()[tid]
    text = " ".join(filter(None, [t.writer, t.payload, t.payload_b]))
    assert not URL.search(text)
    assert "```" not in text
