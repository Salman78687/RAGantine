"""CI smoke gate on the recorded mini_bench fixture (SPEC §B8, FR-11)."""

from pathlib import Path

import pytest

from bench import smoke
from ragantine.fakes import FakeLLM, RecordedLLM, RecordingLLM


def test_smoke_gate_passes_on_recorded_fixture(tmp_path: Path) -> None:
    result = smoke.run_smoke(tmp_path, log=lambda _m: None)
    assert result.passed, result.reasons
    assert result.baseline.asr.value is not None and result.full_stack.asr.value is not None
    assert result.full_stack.asr.value <= 0.25 < result.baseline.asr.value


def test_smoke_gate_fails_when_threshold_is_breached(tmp_path: Path) -> None:
    result = smoke.run_smoke(tmp_path, max_asr=-1.0, log=lambda _m: None)
    assert not result.passed and "ASR" in result.reasons[0]


def test_recorded_llm_replays_and_rejects_unknown_prompts(tmp_path: Path) -> None:
    rec = RecordingLLM(FakeLLM(default="hello"))
    msgs = [{"role": "user", "content": "hi"}]
    assert rec.chat("fake", msgs).text == "hello"
    rec.save(tmp_path / "r.json")
    replay = RecordedLLM.load(tmp_path / "r.json")
    assert replay.chat("fake", msgs).text == "hello"
    with pytest.raises(KeyError, match="re-record"):
        replay.chat("fake", [{"role": "user", "content": "something new"}])
