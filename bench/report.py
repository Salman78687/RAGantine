"""Reports from results/<run_id>/metrics.json (SPEC §B7.2, §10).

``build_report`` pools runs of the same configuration and model across seeds and emits the
README results table (markdown), a per-attack CSV and a grouped bar chart PNG of ASR by
attack x configuration. Numbers come only from metrics.json; nothing is typed by hand.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from bench.metrics import Metrics, Rate, bootstrap_ci
from ragantine.defenses.base import FULL_STACK

DEFENSE_NAMES = {
    "d1": "D1 Sanitizer",
    "d2": "D2 Injection classifier",
    "d3": "D3 Source trust",
    "d4": "D4 Spotlighting",
    "d5": "D5 Consistency check",
    "d6": "D6 Anomaly detector",
}
# Chart series in fixed categorical order (colour follows the configuration, never its rank).
# Validated with the dataviz palette checker (light surface #fcfcfb): adjacent CVD dE >= 9.1.
CHART_SERIES = ["D0 Baseline", "+ D1 Sanitizer", "+ D2 Injection classifier", "+ D3 Source trust",
                "+ D4 Spotlighting", "+ D5 Consistency check", "Full stack"]  # fmt: skip
CHART_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ATTACK_NAMES = {
    "A1": "A1 Factual override", "A2": "A2 Indirect instruction", "A3": "A3 Authority spoof",
    "A4": "A4 Retrieval stuffing", "A5": "A5 Hidden text", "A6": "A6 Split payload",
}  # fmt: skip


def config_label(defenses: Sequence[str]) -> str:
    ids = sorted(defenses)
    if not ids:
        return "D0 Baseline"
    if ids == sorted(FULL_STACK):
        return "Full stack"
    if len(ids) == 1:
        return f"+ {DEFENSE_NAMES[ids[0]]}"
    missing = sorted(set(FULL_STACK) - set(ids))
    if len(missing) == 1 and set(ids) <= set(FULL_STACK):
        return f"Full stack − {missing[0].upper()}"
    return "+ " + " + ".join(i.upper() for i in ids)


def _label_order(label: str) -> tuple[int, str]:
    if label in CHART_SERIES:
        return CHART_SERIES.index(label), label
    return (len(CHART_SERIES), label)


class RunRecord(BaseModel):
    run_id: str
    path: Path
    spec: dict[str, Any]
    metrics: Metrics

    @property
    def label(self) -> str:
        return str(self.spec.get("label") or config_label(self.spec["defenses"]))


def load_run(path: Path) -> RunRecord:
    snapshot = yaml.safe_load((path / "config.yaml").read_text())
    metrics = Metrics.model_validate_json((path / "metrics.json").read_text())
    return RunRecord(run_id=path.name, path=path, spec=snapshot["spec"], metrics=metrics)


def resolve_runs(results_dir: Path, runs: str) -> list[RunRecord]:
    """'id1,id2' | 'latest' (latest ablation, else latest run) | 'all'."""
    dirs = (
        sorted(p for p in results_dir.iterdir() if (p / "metrics.json").exists())
        if results_dir.exists()
        else []
    )
    if runs == "all":
        chosen = dirs
    elif runs == "latest":
        ablations = sorted(results_dir.glob("ablation-*.json"))
        if ablations:
            ids = json.loads(ablations[-1].read_text())["runs"]
            chosen = [results_dir / i for i in ids if (results_dir / i / "metrics.json").exists()]
        else:
            chosen = dirs[-1:]
    else:
        chosen = [results_dir / r.strip() for r in runs.split(",") if r.strip()]
    if not chosen:
        raise FileNotFoundError(f"no runs with metrics.json under {results_dir}")
    return [load_run(p) for p in chosen]


def pool(rates: Sequence[Rate], seed: int = 0) -> Rate:
    """Pool k/n across seeds; CI by bootstrapping the pooled per-question outcomes."""
    n, k = sum(r.n for r in rates), sum(r.k for r in rates)
    outcomes = [True] * k + [False] * (n - k)
    lo, hi = bootstrap_ci(outcomes, seed) if n else (None, None)
    return Rate(value=k / n if n else None, n=n, k=k, ci_low=lo, ci_high=hi)


class Row(BaseModel):
    label: str
    model: str
    split: str
    seeds: list[int]
    asr: Rate
    clean_accuracy: Rate
    detection_recall: Rate
    false_quarantine: Rate
    p95_latency_ms: float | None
    asr_by_attack: dict[str, Rate]
    paraphrase_asr: Rate
    conditional_asr: Rate
    cost_usd: float | None


def aggregate(records: Sequence[RunRecord]) -> list[Row]:
    groups: dict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
    for r in records:
        groups[(r.label, r.spec["model_alias"], r.spec["split"])].append(r)
    rows = []
    for (label, model, split), rs in groups.items():
        ms = [r.metrics for r in rs]
        attacks = sorted({a for m in ms for a in m.asr_by_attack})
        p95s = [m.latency_p95_ms for m in ms if m.latency_p95_ms is not None]
        costs = [m.generator.cost_usd for m in ms]
        rows.append(
            Row(
                label=label, model=model, split=split, seeds=sorted(r.spec["seed"] for r in rs),
                asr=pool([m.asr for m in ms]), clean_accuracy=pool([m.clean_accuracy for m in ms]),
                detection_recall=pool([m.detection_recall for m in ms]),
                false_quarantine=pool([m.ingestion_false_quarantine for m in ms]),
                p95_latency_ms=sum(p95s) / len(p95s) if p95s else None,
                asr_by_attack={a: pool([m.asr_by_attack[a] for m in ms if a in m.asr_by_attack]) for a in attacks},
                paraphrase_asr=pool([m.paraphrase_asr for m in ms]),
                conditional_asr=pool([m.conditional_asr for m in ms]),
                cost_usd=None if any(c is None for c in costs) else sum(c for c in costs if c is not None),
            )
        )  # fmt: skip
    rows.sort(key=lambda r: (r.model, r.split, _label_order(r.label)))
    return rows


def pct(r: Rate, ci: bool = True) -> str:
    if r.value is None:
        return "—"
    s = f"{100 * r.value:.1f}"
    if ci and r.ci_low is not None and r.ci_high is not None:
        s += f" [{100 * r.ci_low:.1f}–{100 * r.ci_high:.1f}]"
    return s


def results_table(rows: Sequence[Row]) -> str:
    lines = [
        "| Configuration | Model | ASR (%) | Clean accuracy (%) | Detection recall (%) | False quarantine (%) | p95 latency (ms) | Seeds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        p95 = "—" if r.p95_latency_ms is None else f"{r.p95_latency_ms:.0f}"
        lines.append(
            f"| {r.label} | {r.model} | {pct(r.asr)} | {pct(r.clean_accuracy)} | "
            f"{pct(r.detection_recall, ci=False)} | {pct(r.false_quarantine, ci=False)} | {p95} | "
            f"{len(r.seeds)} |"
        )
    return "\n".join(lines)


def per_attack_table(rows: Sequence[Row]) -> str:
    attacks = sorted({a for r in rows for a in r.asr_by_attack})
    head = "| Configuration | Model | " + " | ".join(attacks) + " | Paraphrase |"
    lines = [head, "|" + "---|" * (len(attacks) + 3)]
    for r in rows:
        cells = [
            pct(r.asr_by_attack[a], ci=False) if a in r.asr_by_attack else "—" for a in attacks
        ]
        lines.append(
            f"| {r.label} | {r.model} | "
            + " | ".join(cells)
            + f" | {pct(r.paraphrase_asr, ci=False)} |"
        )
    return "\n".join(lines)


def write_per_attack_csv(rows: Sequence[Row], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "configuration",
                "model",
                "split",
                "attack",
                "asr",
                "ci_low",
                "ci_high",
                "k",
                "n",
                "seeds",
            ]
        )
        for r in rows:
            for a, rate in sorted(r.asr_by_attack.items()):
                w.writerow([r.label, r.model, r.split, a, rate.value, rate.ci_low, rate.ci_high, rate.k, rate.n,
                            " ".join(map(str, r.seeds))])  # fmt: skip


def plot_asr_by_attack(rows: Sequence[Row], model: str, path: Path) -> bool:
    """Grouped bars: attack type x configuration (ASR %). Returns False if nothing to plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [r for r in rows if r.model == model and r.label in CHART_SERIES]
    series.sort(key=lambda r: CHART_SERIES.index(r.label))
    attacks = sorted({a for r in series for a in r.asr_by_attack})
    if not series or not attacks:
        return False
    n = len(series)
    group_w = 0.8
    bar_w = group_w / n
    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(attacks) + 2), 4.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, r in enumerate(series):
        xs = [k - group_w / 2 + bar_w * (i + 0.5) for k in range(len(attacks))]
        ys = [100 * (r.asr_by_attack[a].value or 0) if a in r.asr_by_attack else 0 for a in attacks]
        ax.bar(
            xs, ys, width=bar_w, color=CHART_COLORS[CHART_SERIES.index(r.label)],
            edgecolor=SURFACE, linewidth=1.0, label=r.label, zorder=3,
        )  # fmt: skip
    ax.set_xticks(
        range(len(attacks)), [ATTACK_NAMES.get(a, a).replace(" ", "\n", 1) for a in attacks]
    )
    ax.set_ylabel("Attack success rate (%)", color=INK_MUTED)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.set_title(
        f"ASR by attack type and defense configuration ({model})",
        color=INK,
        loc="left",
        fontsize=11,
    )
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=min(4, n),
        loc="upper left",
        bbox_to_anchor=(0, -0.18),
        labelcolor=INK,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return True


def human_agreement(records: Sequence[RunRecord]) -> dict[str, Any]:
    """Agreement between scorer verdicts and hand labels in human_check.csv (if labelled)."""
    total: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        path = r.path / "human_check.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                label = row.get("human_label", "").strip().lower()
                if label:
                    total[row["matched_by"]].append(label == row["verdict"].strip().lower())
                    total["all"].append(label == row["verdict"].strip().lower())
    return {k: {"agreement": sum(v) / len(v), "n": len(v)} for k, v in total.items()}


def build_report(results_dir: Path, runs: str, out_dir: Path | None = None) -> tuple[str, Path]:
    records = resolve_runs(results_dir, runs)
    rows = aggregate(records)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out = out_dir or results_dir / f"report-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    write_per_attack_csv(rows, out / "per_attack.csv")
    charts = []
    for model in sorted({r.model for r in rows}):
        png = out / f"asr_by_attack_{model}.png"
        if plot_asr_by_attack(rows, model, png):
            charts.append(png.name)
    agreement = human_agreement(records)
    md = [
        "# RAGantine results",
        "",
        f"Runs: {len(records)} · generated {stamp} UTC · numbers pooled across seeds; "
        "brackets are 95% bootstrap CIs over questions.",
        "",
        results_table(rows),
        "",
        "## ASR by attack type (%)",
        "",
        per_attack_table(rows),
        "",
    ]
    md += [f"![ASR by attack]({c})" for c in charts]
    md += ["", "## Scorer agreement with hand labels", ""]
    if agreement:
        md += [
            f"- {k}: {100 * v['agreement']:.1f}% of {v['n']} labelled items"
            for k, v in sorted(agreement.items())
        ]
    else:
        md.append(
            "- No hand labels yet: fill `human_label` in each run's human_check.csv and re-run."
        )
    md += ["", "## Runs", ""] + [
        f"- `{r.run_id}` — {r.label}, {r.spec['model_alias']}, seed {r.spec['seed']}"
        for r in records
    ]
    text = "\n".join(md) + "\n"
    (out / "report.md").write_text(text)
    return text, out


def single_run_markdown(spec: Any, m: Metrics) -> str:
    label = spec.label or config_label(spec.defenses)
    lines = [
        f"# Run {spec.run_id}",
        "",
        f"- configuration: {label} (defenses: {', '.join(spec.defenses) or 'none'})",
        f"- model: {spec.model_alias} · split: {spec.split} · seed: {spec.seed} · limit: {spec.limit}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| ASR (%) | {pct(m.asr)} (n={m.asr.n}) |",
        f"| Conditional ASR (%) | {pct(m.conditional_asr)} (n={m.conditional_asr.n}) |",
        f"| Paraphrase ASR (%) | {pct(m.paraphrase_asr)} (n={m.paraphrase_asr.n}) |",
        f"| Clean accuracy (%) | {pct(m.clean_accuracy)} (n={m.clean_accuracy.n}) |",
        f"| Detection recall (%) | {pct(m.detection_recall, ci=False)} (n={m.detection_recall.n}) |",
        f"| Ingestion false-quarantine (%) | {pct(m.ingestion_false_quarantine, ci=False)} |",
        f"| Retrieval false-filter (%) | {pct(m.retrieval_false_filter, ci=False)} |",
        f"| p50 / p95 latency (ms) | {m.latency_p50_ms} / {m.latency_p95_ms} |",
        f"| generator tokens | {m.generator.prompt_tokens} + {m.generator.completion_tokens} (${m.generator.cost_usd}) |",
        f"| judge tokens | {m.judge.prompt_tokens} + {m.judge.completion_tokens} (${m.judge.cost_usd}) |",
        f"| LLM calls / cache hits / errors | {m.llm_calls} / {m.cache_hits} / {m.errors} |",
        "",
        "| Attack | ASR (%) | n |",
        "|---|---|---|",
        *[f"| {a} | {pct(r)} | {r.n} |" for a, r in sorted(m.asr_by_attack.items())],
    ]
    if m.notes:
        lines += ["", *[f"> {n}" for n in m.notes]]
    return "\n".join(lines) + "\n"
