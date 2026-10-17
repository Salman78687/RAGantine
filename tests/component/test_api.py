"""FastAPI service with TestClient over an in-memory pipeline (SPEC §B7.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragantine.api import create_app
from ragantine.fakes import FakeLLM
from ragantine.schemas import RawDocument
from tests.helpers import SAMPLE_DOCS, make_doc, memory_pipeline
from tests.unit.test_d1_sanitizer import PAYLOAD

HIDDEN = make_doc(
    "Danube notes",
    f"<body><p>The Danube is long.</p><!-- {PAYLOAD} --></body>",
    fmt="html",
    tier="web",
)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    from ragantine.defenses.base import build_defenses
    from tests.helpers import small_config

    llm = FakeLLM(rules=[("Eiffel", "It was completed in 1889."), ("CONTEXT", "I don't know.")])
    docs: list[RawDocument] = [*SAMPLE_DOCS, HIDDEN]
    pipeline, report = memory_pipeline(
        llm, docs=docs, defenses=build_defenses(["d1"], small_config())
    )
    assert report.n_quarantined > 0
    return TestClient(create_app(pipeline, results_dir=tmp_path))


def _sse(text: str) -> list[tuple[str, dict]]:  # type: ignore[type-arg]
    events, name = [], None
    for line in text.splitlines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and name:
            events.append((name, json.loads(line.split(":", 1)[1])))
    return events


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {
        "status": "ok",
        "opensearch": True,
        "models": ["fake"],
    }


def test_ask_json(client: TestClient) -> None:
    r = client.post(
        "/ask", json={"question": "When was the Eiffel Tower completed?", "stream": False}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "It was completed in 1889." and body["model_alias"] == "fake"


def test_ask_sse_event_order(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/ask",
        json={"question": "When was the Eiffel Tower completed?", "defenses": ["d3", "d4"]},
    ) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = _sse(r.read().decode())
    names = [n for n, _ in events]
    assert names[0] == "retrieval" and names[-1] == "done"
    body = names[1:-1]
    assert "token" in body
    first_token = body.index("token")
    assert all(
        n == "defense" for n in body[:first_token]
    )  # retrieval-stage decisions precede tokens
    assert events[-1][1]["text"] == "It was completed in 1889."


def test_ask_validation_and_limits(client: TestClient) -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 501}).status_code == 422
    assert client.post("/ask", json={"question": "q", "defenses": ["d9"]}).status_code == 422
    assert (
        client.post("/ask", json={"question": "q", "model": "nope", "stream": False}).status_code
        == 422
    )
    big = json.dumps({"question": "q", "pad": "x" * 9000})
    assert (
        client.post("/ask", content=big, headers={"content-type": "application/json"}).status_code
        == 413
    )


def test_quarantine_list_get_release(client: TestClient) -> None:
    page = client.get("/quarantine", params={"reason": "hidden_content"}).json()
    assert page["total"] >= 1
    cid = page["items"][0]["chunk"]["chunk_id"]
    assert client.get(f"/quarantine/{cid}").json()["status"] == "quarantined"
    assert client.get("/quarantine/missing:0000").status_code == 404
    assert client.get("/quarantine", params={"limit": 201}).status_code == 422
    r = client.post(f"/quarantine/{cid}/release", json={"released_by": "reviewer"})
    assert (
        r.status_code == 200
        and r.json()["released_by"] == "reviewer"
        and r.json()["status"] == "released"
    )
    assert (
        client.post("/quarantine/missing:0000/release", json={"released_by": "x"}).status_code
        == 404
    )
    assert client.post(f"/quarantine/{cid}/release", json={"released_by": ""}).status_code == 422


def test_runs_endpoints(client: TestClient, tmp_path: Path) -> None:
    run = tmp_path / "r1"
    run.mkdir()
    (run / "metrics.json").write_text(
        json.dumps({"asr": {"value": 0.5}, "clean_accuracy": {"value": 0.9}})
    )
    (run / "config.yaml").write_text(
        "spec: {label: D0 Baseline, defenses: [], model_alias: m, split: dev, seed: 1}\n"
    )
    listing = client.get("/runs").json()
    assert listing == [{"run_id": "r1", "label": "D0 Baseline", "defenses": [], "model": "m", "split": "dev",
                        "seed": 1, "asr": 0.5, "clean_accuracy": 0.9}]  # fmt: skip
    assert client.get("/runs/r1").json()["metrics"]["asr"]["value"] == 0.5
    assert client.get("/runs/..").status_code == 404
    assert client.get("/runs/nope").status_code == 404


def test_cors_is_localhost_only(client: TestClient) -> None:
    ok = client.options(
        "/health",
        headers={"origin": "http://localhost:7860", "access-control-request-method": "GET"},
    )
    bad = client.options(
        "/health",
        headers={"origin": "https://evil.example", "access-control-request-method": "GET"},
    )
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:7860"
    assert "access-control-allow-origin" not in bad.headers


def test_tracer_is_noop_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    from ragantine.config import Settings
    from ragantine.tracing import NoopTracer, make_tracer

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert isinstance(make_tracer(Settings(_env_file=None)), NoopTracer)
