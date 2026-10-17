"""FastAPI service (SPEC §B7.1): a thin layer over ``stream_query`` / ``arun_query``.

* ``GET  /health``                         {status, opensearch, models}; 503 if the index is down
* ``POST /ask``                            SSE stream (retrieval, defense*, token*, done | error)
                                           or ``Answer`` JSON when ``stream=false``
* ``GET  /quarantine``                     review queue: status, reason, limit (<= 200), offset
* ``GET  /quarantine/{chunk_id}``          one record or 404
* ``POST /quarantine/{chunk_id}/release``  moves the chunk back; logs ``released_by``
* ``GET  /runs`` / ``GET /runs/{run_id}``  benchmark results (metrics.json + config)

Local tool: no auth, CORS open for localhost only, request bodies capped at 8 KB.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ragantine.generate.llm import MissingAPIKeyError
from ragantine.graph import Pipeline, arun_query, stream_query
from ragantine.quarantine import QuarantineService
from ragantine.schemas import Answer, QuarantineRecord

MAX_BODY_BYTES = 8 * 1024
LOCALHOST_ORIGINS = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
RUN_ID = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
DefenseId = Literal["d1", "d2", "d3", "d4", "d5", "d6"]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    defenses: list[DefenseId] | None = None
    model: str | None = None
    stream: bool = True


class ReleaseRequest(BaseModel):
    released_by: str = Field(min_length=1, max_length=200)


class QuarantinePage(BaseModel):
    items: list[QuarantineRecord]
    total: int


def create_app(pipeline: Pipeline | None = None, results_dir: Path | None = None) -> FastAPI:
    """Build the app. Without a pipeline, one is wired from config on first use."""
    app = FastAPI(title="RAGantine", version="0.1.0", description="Quarantine for retrieval.")
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=LOCALHOST_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    state: dict[str, Any] = {"pipeline": pipeline}

    def get_pipeline() -> Pipeline:
        if state["pipeline"] is None:
            from ragantine.config import Settings, load_config, load_models_config

            state["pipeline"] = Pipeline.from_config(
                load_config(), load_models_config(), Settings()
            )
        p: Pipeline = state["pipeline"]
        return p

    def runs_dir() -> Path:
        return results_dir or get_pipeline().cfg.paths.results_dir

    @app.exception_handler(MissingAPIKeyError)
    async def missing_key(_request: Request, exc: MissingAPIKeyError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.middleware("http")
    async def limit_body(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        length = request.headers.get("content-length")
        if length is not None and length.isdigit() and int(length) > MAX_BODY_BYTES:
            return JSONResponse(
                {"detail": f"request body over {MAX_BODY_BYTES} bytes"}, status_code=413
            )
        return await call_next(request)

    @app.get("/health")
    def health() -> JSONResponse:
        p = get_pipeline()
        ok = p.backend.healthy()
        body = {
            "status": "ok" if ok else "degraded",
            "opensearch": ok,
            "models": sorted(p.llm.models.models),
        }
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.post("/ask", response_model=None)
    async def ask(req: AskRequest) -> Answer | EventSourceResponse:
        p = get_pipeline()
        if req.model is not None and req.model not in p.llm.models.models:
            raise HTTPException(422, f"unknown model alias {req.model!r}")
        defenses = req.defenses if req.defenses is not None else None
        if not req.stream:
            return await arun_query(req.question, p, defenses, req.model)

        async def events() -> AsyncIterator[dict[str, str]]:
            async for name, data in stream_query(req.question, p, defenses, req.model):
                yield {"event": name, "data": json.dumps(data)}

        return EventSourceResponse(events())

    @app.get("/quarantine")
    def quarantine_list(
        status: Literal["quarantined", "released"] | None = None,
        reason: str | None = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> QuarantinePage:
        items, total = QuarantineService(get_pipeline().backend).list(status, reason, limit, offset)
        return QuarantinePage(items=items, total=total)

    @app.get("/quarantine/{chunk_id}")
    def quarantine_get(chunk_id: str) -> QuarantineRecord:
        record = QuarantineService(get_pipeline().backend).get(chunk_id)
        if record is None:
            raise HTTPException(404, f"{chunk_id} is not in quarantine")
        return record

    @app.post("/quarantine/{chunk_id}/release")
    def quarantine_release(chunk_id: str, req: ReleaseRequest) -> QuarantineRecord:
        record = QuarantineService(get_pipeline().backend).release(chunk_id, req.released_by)
        if record is None:
            raise HTTPException(404, f"{chunk_id} is not in quarantine")
        return record

    @app.get("/runs")
    def runs() -> list[dict[str, Any]]:
        root = runs_dir()
        out = []
        for d in sorted(root.glob("*/metrics.json")) if root.exists() else []:
            m = json.loads(d.read_text())
            spec = yaml.safe_load((d.parent / "config.yaml").read_text()).get("spec", {})
            out.append(
                {
                    "run_id": d.parent.name, "label": spec.get("label"), "defenses": spec.get("defenses"),
                    "model": spec.get("model_alias"), "split": spec.get("split"), "seed": spec.get("seed"),
                    "asr": m["asr"]["value"], "clean_accuracy": m["clean_accuracy"]["value"],
                }
            )  # fmt: skip
        return out

    @app.get("/runs/{run_id}")
    def run(run_id: str) -> dict[str, Any]:
        if not RUN_ID.match(run_id):
            raise HTTPException(404, "unknown run")
        d = (runs_dir() / run_id).resolve()
        if d.parent != runs_dir().resolve() or not (d / "metrics.json").exists():
            raise HTTPException(404, "unknown run")
        return {
            "run_id": run_id,
            "metrics": json.loads((d / "metrics.json").read_text()),
            "config": yaml.safe_load((d / "config.yaml").read_text()),
        }

    return app
