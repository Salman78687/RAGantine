"""Ablation protocol on the TEST split (SPEC §10, Phase 6).

For every model and seed: D0 baseline, each defense alone, the full stack, and the full
stack minus each defense (leave-one-out). Run ids are recorded in
results/ablation-<stamp>.json so ``bench report --runs latest`` reports exactly this set.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from bench.report import config_label
from bench.run import QSet, RunSpec, make_run_id
from ragantine.defenses.base import FULL_STACK

SETS: list[QSet] = ["target", "clean", "paraphrase"]


class AblationPlan(BaseModel):
    ablation_id: str
    specs: list[RunSpec]


def configurations(include_d6: bool = False) -> list[list[str]]:
    singles = [[d] for d in FULL_STACK] + ([["d6"]] if include_d6 else [])
    full = list(FULL_STACK)
    leave_one_out = [[d for d in full if d != drop] for drop in full]
    return [[], *singles, full, *leave_one_out]


def plan(
    models: Sequence[str],
    seeds: Sequence[int],
    split: str = "test",
    include_d6: bool = False,
    limit: int | None = None,
    label_suffix: str = "",
) -> AblationPlan:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    specs = []
    for model in models:
        for seed in seeds:
            for ids in configurations(include_d6):
                specs.append(
                    RunSpec(
                        run_id=f"{make_run_id(ids, model, split, seed)}-{stamp[-6:]}",
                        defenses=ids, model_alias=model, sets=SETS, split=split,
                        seed=seed, limit=limit, label=config_label(ids) + label_suffix,
                    )
                )  # fmt: skip
    return AblationPlan(ablation_id=f"ablation-{stamp}", specs=specs)


def record(results_dir: Path, ablation: AblationPlan, done: Sequence[str]) -> Path:
    path = results_dir / f"{ablation.ablation_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"runs": list(done), "planned": [s.run_id for s in ablation.specs]}, indent=2)
    )
    return path


def log_progress(log: Callable[[str], None], i: int, n: int, spec: RunSpec) -> None:
    log(f"[{i}/{n}] {spec.label} · {spec.model_alias} · seed {spec.seed}")
