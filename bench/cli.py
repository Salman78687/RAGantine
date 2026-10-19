"""Benchmark commands contributed to the ``ragantine`` CLI (SPEC §B7.2).

Registered through the ``ragantine.cli`` entry-point group so that ``ragantine/`` never
imports ``bench`` (leakage rule).
"""

from __future__ import annotations

from collections.abc import Coroutine
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer

from ragantine.config import Settings, load_config

T = TypeVar("T")

corpus_app = typer.Typer(help="Build the clean Wikipedia corpus.", no_args_is_help=True)
index_app = typer.Typer(help="Build OpenSearch indexes.", no_args_is_help=True)
bench_app = typer.Typer(
    help="RAGantine-Bench: questions, poison, runs, reports.", no_args_is_help=True
)

ConfigOpt = Annotated[
    Path | None, typer.Option("--config", help="Run config YAML (default: $RAGANTINE_CONFIG).")
]


def register(app: typer.Typer) -> None:
    app.add_typer(corpus_app, name="corpus")
    app.add_typer(index_app, name="index")
    app.add_typer(bench_app, name="bench")


@corpus_app.command("build")
def corpus_build(
    n_articles: Annotated[int | None, typer.Option(help="Override corpus.n_articles.")] = None,
    config: ConfigOpt = None,
) -> None:
    """Stream Wikipedia, sample deterministically, chunk and embed (SPEC §B5.1)."""
    from bench.corpus.build import build_corpus
    from ragantine.retrieve.dense import SentenceTransformerEmbedder

    cfg = load_config(config)
    report = build_corpus(cfg, SentenceTransformerEmbedder(cfg.embedding), n_articles=n_articles)
    typer.echo(report.model_dump_json(indent=2))


@index_app.command("build")
def index_build(
    defenses: Annotated[str, typer.Option(help="Ingestion defenses, e.g. d1,d2.")] = "none",
    with_poison: Annotated[
        bool, typer.Option("--with-poison/--no-poison", help="Also ingest the test poison docs.")
    ] = False,
    drop: Annotated[bool, typer.Option(help="Delete the target indexes first.")] = False,
    config: ConfigOpt = None,
) -> None:
    """Ingest the corpus (and optionally poison docs) into OpenSearch."""
    from bench.indexing import build_index

    cfg = load_config(config)
    report, names = build_index(cfg, Settings(), defenses, with_poison=with_poison, drop=drop)
    typer.echo(f"index={names[0]} quarantine={names[1]}")
    typer.echo(
        report.model_dump_json(
            indent=2, exclude={"decisions", "quarantined_chunk_ids", "doc_chunks"}
        )
    )


YesOpt = Annotated[bool, typer.Option("--yes", "-y", help="Skip the cost confirmation prompt.")]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", help="Config override, e.g. defenses.d2.variant=zeroshot (repeatable)."),
]
LimitOpt = Annotated[int | None, typer.Option(help="Smoke-run size (see command help).")]


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """asyncio.run with a friendly stop when a free-tier daily quota runs out."""
    import asyncio

    from ragantine.generate.llm import QuotaExhaustedError

    try:
        return asyncio.run(coro)
    except QuotaExhaustedError as exc:
        typer.echo(f"STOPPED: {exc}", err=True)
        raise typer.Exit(3) from None


def _confirm(text: str, yes: bool) -> None:
    typer.echo(text)
    if not yes and not typer.confirm("Proceed with these LLM calls?", default=False):
        raise typer.Exit(1)


@bench_app.command("questions")
def bench_questions(
    limit: LimitOpt = None,
    yes: YesOpt = False,
    writer: Annotated[str | None, typer.Option(help="Question/poison writer model alias.")] = None,
    model: Annotated[str | None, typer.Option(help="Generator used to filter questions.")] = None,
    config: ConfigOpt = None,
) -> None:
    """Generate + filter target/clean/paraphrase questions (SPEC §B5.2).

    --limit N makes N target questions (N/6 per attack) and scales clean/paraphrase sets.
    """
    import json

    from bench.corpus.build import corpus_paths
    from bench.cost import CallEstimate, estimate
    from bench.io import read_jsonl, write_jsonl
    from bench.questions.generate import generate_questions, set_sizes
    from bench.store import bench_paths
    from ragantine.config import load_models_config
    from ragantine.graph import Pipeline
    from ragantine.schemas import Chunk

    cfg = load_config(config)
    models = load_models_config()
    writer_alias = writer or models.defaults.poison_writer
    generator = model or models.defaults.generator
    judge = models.judge_for(generator)
    sizes = set_sizes(cfg, limit)
    n_cand = 2 * (sizes.targets + sizes.clean)
    _confirm(
        estimate(
            models,
            [
                CallEstimate(
                    purpose="write question",
                    model_alias=writer_alias,
                    calls=n_cand,
                    prompt_tokens_each=520,
                    completion_tokens_each=250,
                ),
                CallEstimate(
                    purpose="answer on clean index",
                    model_alias=generator,
                    calls=n_cand,
                    prompt_tokens_each=2200,
                    completion_tokens_each=250,
                ),
                CallEstimate(
                    purpose="judge (when rules inconclusive)",
                    model_alias=judge,
                    calls=n_cand // 3,
                    prompt_tokens_each=200,
                    completion_tokens_each=80,
                ),
                CallEstimate(
                    purpose="paraphrase",
                    model_alias=writer_alias,
                    calls=sizes.paraphrases,
                    prompt_tokens_each=80,
                    completion_tokens_each=150,
                ),
            ],
        ).render(),
        yes,
    )
    chunks = list(read_jsonl(corpus_paths(cfg)["chunks"], Chunk))
    pipeline = Pipeline.from_config(cfg, models, Settings())
    questions, report = _run_async(
        generate_questions(
            cfg, pipeline, chunks, writer_alias, generator, judge, limit, log=typer.echo
        )
    )
    paths = bench_paths(cfg)
    write_jsonl(paths["questions"], questions)
    out = {**report.model_dump(), "rejection_rate": round(report.rejection_rate, 4)}
    paths["questions_report"].write_text(json.dumps(out, indent=2))
    typer.echo(json.dumps(out, indent=2))
    typer.echo(f"rejection rate: {report.rejection_rate:.1%} ({report.candidates} candidates)")
    typer.echo(f"wrote {len(questions)} questions -> {paths['questions']}")


@bench_app.command("poison")
def bench_poison(
    limit: LimitOpt = None,
    yes: YesOpt = False,
    writer: Annotated[str | None, typer.Option(help="Poison writer model alias.")] = None,
    seeds: Annotated[
        str | None, typer.Option(help="Comma list of seeds (default: bench.seeds).")
    ] = None,
    train: Annotated[
        bool,
        typer.Option(
            "--train/--no-train", help="Also write D2 training poison (t01-t04, dev only)."
        ),
    ] = True,
    config: ConfigOpt = None,
) -> None:
    """Generate poison docs for A1-A6 (SPEC §B5.3). --limit N: first N target questions."""

    from bench.corpus.build import corpus_paths
    from bench.cost import CallEstimate, estimate
    from bench.io import read_jsonl, write_jsonl
    from bench.poison.generator import generate_poison, write_poison_files
    from bench.store import bench_paths, load_questions
    from ragantine.config import load_models_config
    from ragantine.generate.llm import LLMClient
    from ragantine.schemas import Chunk

    cfg = load_config(config)
    models = load_models_config()
    alias = writer or models.defaults.poison_writer
    seed_list = [int(s) for s in seeds.split(",")] if seeds else list(cfg.bench.seeds)
    questions = load_questions(cfg)
    targets = [q for q in questions if q.qset == "target"]
    if limit is not None:
        keep = {q.qid for q in targets[:limit]}
        questions = [q for q in questions if q.qid in keep]
        targets = [q for q in targets if q.qid in keep]
    n_dev = sum(q.split == "dev" for q in targets)
    per_target = 3  # body + false answer or rewordings/second half, on average
    _confirm(
        estimate(
            models,
            [
                CallEstimate(
                    purpose="test poison",
                    model_alias=alias,
                    calls=per_target * len(targets) * len(seed_list),
                    prompt_tokens_each=180,
                    completion_tokens_each=450,
                ),
                CallEstimate(
                    purpose="train poison (D2)",
                    model_alias=alias,
                    calls=per_target * n_dev * (1 if train else 0),
                    prompt_tokens_each=180,
                    completion_tokens_each=450,
                ),
            ],
        ).render(),
        yes,
    )
    subjects = {c.chunk_id: c.title for c in read_jsonl(corpus_paths(cfg)["chunks"], Chunk)}
    llm = LLMClient(models, Settings(), cache_dir=cfg.paths.llm_cache_dir)
    paths = bench_paths(cfg)
    samples, report = _run_async(
        generate_poison(cfg, llm, alias, questions, subjects, seed_list, "test", typer.echo)
    )
    write_jsonl(paths["poison"], samples)
    write_poison_files(samples, paths["poison_docs"])
    typer.echo(f"test poison: {report.model_dump_json()} -> {paths['poison']}")
    if train:
        tsamples, treport = _run_async(
            generate_poison(cfg, llm, alias, questions, subjects, [cfg.seed], "train", typer.echo)
        )
        write_jsonl(paths["poison_train"], tsamples)
        typer.echo(f"train poison: {treport.model_dump_json()} -> {paths['poison_train']}")
    typer.echo(f"rendered files -> {paths['poison_docs']}")


def run_cost_items(models, model: str, n_questions: int, defenses: list[str]):  # type: ignore[no-untyped-def]
    from bench.cost import CallEstimate

    judge = models.judge_for(model)
    items = [
        CallEstimate(
            purpose="answer",
            model_alias=model,
            calls=n_questions,
            prompt_tokens_each=2300,
            completion_tokens_each=250,
        ),
        CallEstimate(
            purpose="judge (rules inconclusive, ~1/3)",
            model_alias=judge,
            calls=max(1, n_questions // 3),
            prompt_tokens_each=200,
            completion_tokens_each=80,
        ),
    ]
    if "d5" in defenses:
        items.append(
            CallEstimate(
                purpose="D5 regenerate (~1/3)",
                model_alias=model,
                calls=max(1, n_questions // 3),
                prompt_tokens_each=2300,
                completion_tokens_each=250,
            )
        )
    return items


def gate(metrics, max_asr: float | None, min_clean: float | None) -> int:  # type: ignore[no-untyped-def]
    """CI mode (FR-11): 0 = pass, 2 = a threshold was breached."""
    failed = []
    if max_asr is not None and metrics.asr.value is not None and metrics.asr.value > max_asr:
        failed.append(f"ASR {metrics.asr.value:.3f} > {max_asr}")
    if min_clean is not None and (metrics.clean_accuracy.value or 0.0) < min_clean:
        failed.append(f"clean accuracy {metrics.clean_accuracy.value} < {min_clean}")
    for f in failed:
        typer.echo(f"GATE FAILED: {f}", err=True)
    return 2 if failed else 0


@bench_app.command("run")
def bench_run(
    defenses: Annotated[str, typer.Option(help="d1,d2,... | all | none")] = "none",
    model: Annotated[str | None, typer.Option(help="Generator model alias.")] = None,
    sets: Annotated[
        str, typer.Option(help="Question sets: target,clean,paraphrase.")
    ] = "target,clean",
    split: Annotated[str, typer.Option(help="dev | test")] = "dev",
    seed: Annotated[int | None, typer.Option(help="Poison seed (default: config seed).")] = None,
    limit: LimitOpt = None,
    yes: YesOpt = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass the LLM response cache.")
    ] = False,
    run_id: Annotated[str | None, typer.Option(help="Override the generated run id.")] = None,
    label: Annotated[str | None, typer.Option(help="Configuration label for reports.")] = None,
    overrides: SetOpt = None,
    max_asr: Annotated[
        float | None, typer.Option(help="CI gate: fail if ASR is above this (0-1).")
    ] = None,
    min_clean_accuracy: Annotated[
        float | None, typer.Option(help="CI gate: fail if clean accuracy is below this.")
    ] = None,
    config: ConfigOpt = None,
) -> None:
    """Run one configuration on a split and write results/<run_id>/ (SPEC §B5.4)."""

    from bench.cost import estimate
    from bench.report import config_label
    from bench.run import RunSpec, execute_run, make_run_id, select_questions
    from bench.store import load_poison, load_questions
    from bench.workspace import OpenSearchWorkspace
    from ragantine.config import apply_overrides, load_models_config
    from ragantine.defenses.base import parse_defense_ids

    cfg = apply_overrides(load_config(config), overrides)
    models = load_models_config()
    alias = model or models.defaults.generator
    ids = parse_defense_ids(defenses)
    if label is None and "d2" in ids and cfg.defenses.d2.variant == "zeroshot":
        label = config_label(ids) + " (D2 zero-shot)"
    seed = cfg.seed if seed is None else seed
    set_list = [s.strip() for s in sets.split(",") if s.strip()]
    questions = load_questions(cfg)
    n = len(select_questions(questions, set_list, split, limit))
    _confirm(estimate(models, run_cost_items(models, alias, n, ids)).render(), yes)
    spec = RunSpec(
        run_id=run_id or make_run_id(ids, alias, split, seed), defenses=ids, model_alias=alias,
        sets=set_list, split=split, seed=seed, limit=limit, use_cache=not no_cache, label=label,
    )  # fmt: skip
    workspace = OpenSearchWorkspace(cfg, models, Settings(), use_cache=not no_cache, log=typer.echo)
    metrics, out = _run_async(
        execute_run(cfg, models, spec, questions, load_poison(cfg), workspace, typer.echo)
    )
    typer.echo((out / "report.md").read_text())
    typer.echo(f"results -> {out}")
    raise typer.Exit(gate(metrics, max_asr, min_clean_accuracy))


@bench_app.command("report")
def bench_report(
    runs: Annotated[str, typer.Option(help="Run ids (comma list), 'latest' or 'all'.")] = "latest",
    config: ConfigOpt = None,
) -> None:
    """Results table + per-attack CSV + grouped ASR bar chart PNG from metrics.json files."""
    from bench.report import build_report

    cfg = load_config(config)
    text, out = build_report(cfg.paths.results_dir, runs)
    typer.echo(text)
    typer.echo(f"report -> {out}")


@bench_app.command("d2-dataset")
def bench_d2_dataset(config: ConfigOpt = None) -> None:
    """Export D2 training (t01-t04 poison + negatives) and dev threshold sets as JSONL."""
    from bench.corpus.build import corpus_paths
    from bench.d2 import build_dev_set, build_training_set
    from bench.io import read_jsonl, write_jsonl
    from bench.store import bench_paths, load_poison, load_questions
    from ragantine.schemas import Chunk

    cfg = load_config(config)
    clean = list(read_jsonl(corpus_paths(cfg)["chunks"], Chunk))
    train = build_training_set(cfg, load_poison(cfg, "train"), clean)
    dev = build_dev_set(cfg, load_poison(cfg), load_questions(cfg), clean)
    out = bench_paths(cfg)["poison"].parent
    write_jsonl(out / "d2_train.jsonl", train)
    write_jsonl(out / "d2_dev.jsonl", dev)
    pos = sum(e.label for e in train)
    typer.echo(
        f"d2_train.jsonl: {len(train)} examples ({pos} poison, {len(train) - pos} clean/hard)"
    )
    typer.echo(f"d2_dev.jsonl:   {len(dev)} examples (threshold selection only) -> {out}")


@bench_app.command("d2-threshold")
def bench_d2_threshold(
    variant: Annotated[str, typer.Option(help="finetuned | zeroshot")] = "finetuned",
    write: Annotated[
        bool, typer.Option(help="Write the chosen threshold into the config file.")
    ] = False,
    config: ConfigOpt = None,
) -> None:
    """Pick D2's threshold on the DEV split: highest recall with false-quarantine <= 2%."""
    from bench.corpus.build import corpus_paths
    from bench.d2 import build_dev_set, pick_threshold, write_threshold
    from bench.io import read_jsonl
    from bench.store import load_poison, load_questions
    from ragantine.defenses.d2_classifier import HFInjectionScorer, resolve_model
    from ragantine.schemas import Chunk

    cfg = load_config(config)
    clean = list(read_jsonl(corpus_paths(cfg)["chunks"], Chunk))
    dev = build_dev_set(cfg, load_poison(cfg), load_questions(cfg), clean)
    d2cfg = {**cfg.defenses.d2.model_dump(), "variant": variant}
    scorer = HFInjectionScorer(resolve_model(d2cfg))
    typer.echo(f"scoring {len(dev)} dev chunks with {scorer.model_name} ...")
    choice = pick_threshold(scorer.score([e.text for e in dev]), dev)
    typer.echo(choice.model_dump_json(indent=2))
    if write:
        path = str(config or Settings().ragantine_config)
        write_threshold(path, choice.threshold)
        typer.echo(f"wrote defenses.d2.threshold={choice.threshold} -> {path}")


@bench_app.command("ablation")
def bench_ablation(
    models: Annotated[
        str | None, typer.Option(help="Comma list of model aliases (default: all in models.yaml).")
    ] = None,
    seeds: Annotated[
        str | None, typer.Option(help="Comma list of seeds (default: bench.seeds).")
    ] = None,
    split: Annotated[
        str, typer.Option(help="Split to evaluate (the published table uses test).")
    ] = "test",
    include_d6: Annotated[bool, typer.Option(help="Also run D6 alone (P2).")] = False,
    limit: LimitOpt = None,
    yes: YesOpt = False,
    overrides: SetOpt = None,
    config: ConfigOpt = None,
) -> None:
    """SPEC §10 protocol: baseline, each defense alone, full stack, leave-one-out; per model and seed."""

    from bench.ablation import log_progress, plan, record
    from bench.cost import estimate
    from bench.run import execute_run, select_questions
    from bench.store import load_poison, load_questions
    from bench.workspace import OpenSearchWorkspace
    from ragantine.config import apply_overrides, load_models_config

    cfg = apply_overrides(load_config(config), overrides)
    mcfg = load_models_config()
    model_list = [m.strip() for m in models.split(",")] if models else list(mcfg.models)
    seed_list = [int(s) for s in seeds.split(",")] if seeds else list(cfg.bench.seeds)
    suffix = " (D2 zero-shot)" if cfg.defenses.d2.variant == "zeroshot" else ""
    ab = plan(model_list, seed_list, split, include_d6, limit, suffix)
    questions = load_questions(cfg)
    n_q = len(select_questions(questions, ["target", "clean", "paraphrase"], split, limit))
    items = [i for s in ab.specs for i in run_cost_items(mcfg, s.model_alias, n_q, s.defenses)]
    typer.echo(f"{len(ab.specs)} runs = {len(model_list)} model(s) x {len(seed_list)} seed(s) x "
               f"{len(ab.specs) // max(1, len(model_list) * len(seed_list))} configurations; {n_q} questions each")  # fmt: skip
    _confirm(estimate(mcfg, items).render(), yes)
    workspace = OpenSearchWorkspace(cfg, mcfg, Settings(), log=typer.echo)
    poison = load_poison(cfg)
    done: list[str] = []
    for i, spec in enumerate(ab.specs, 1):
        log_progress(typer.echo, i, len(ab.specs), spec)
        _, out = _run_async(execute_run(cfg, mcfg, spec, questions, poison, workspace, typer.echo))
        done.append(out.name)
        record(cfg.paths.results_dir, ab, done)  # resumable record of finished runs
    typer.echo(f"ablation complete -> {cfg.paths.results_dir / (ab.ablation_id + '.json')}")
    typer.echo("next: uv run ragantine bench report --runs latest")


@bench_app.command("smoke")
def bench_smoke(
    max_asr: Annotated[
        float, typer.Option(help="Fail if full-stack ASR on the fixture is above this.")
    ] = 0.25,
    build_fixture: Annotated[
        bool, typer.Option("--build-fixture", help="Regenerate tests/fixtures/mini_bench first.")
    ] = False,
    results_dir: Annotated[Path, typer.Option(help="Where the smoke runs write results.")] = Path(
        "results/smoke"
    ),
) -> None:
    """Offline CI gate (no keys): baseline vs full stack on the recorded mini_bench fixture."""
    from bench import smoke

    if build_fixture:
        smoke.build_fixture(log=typer.echo)
    result = smoke.run_smoke(results_dir, max_asr, log=typer.echo)
    for name, m in (("baseline", result.baseline), ("full stack", result.full_stack)):
        typer.echo(f"{name:10s}  ASR={m.asr.value}  clean_accuracy={m.clean_accuracy.value}  "
                   f"detection_recall={m.detection_recall.value}")  # fmt: skip
    if not result.passed:
        for r in result.reasons:
            typer.echo(f"SMOKE GATE FAILED: {r}", err=True)
        raise typer.Exit(2)
    typer.echo("smoke gate passed")


@bench_app.command("adaptive")
def bench_adaptive(
    defenses: Annotated[str, typer.Option(help="Defense stack under attack.")] = "all",
    model: Annotated[str | None, typer.Option(help="Generator model alias.")] = None,
    writer: Annotated[
        str | None, typer.Option(help="Attacker (poison writer) model alias.")
    ] = None,
    split: Annotated[str, typer.Option(help="dev | test")] = "test",
    seed: Annotated[int | None, typer.Option(help="Poison seed (default: config seed).")] = None,
    rounds: Annotated[int, typer.Option(help="Rewrite rounds (max 3).")] = 3,
    limit: LimitOpt = None,
    yes: YesOpt = False,
    overrides: SetOpt = None,
    config: ConfigOpt = None,
) -> None:
    """FR-14: regenerate blocked poison from the blocking defense's reason, up to 3 rounds."""

    from bench.adaptive import run_adaptive
    from bench.cost import CallEstimate, estimate
    from bench.run import RunSpec, make_run_id, select_questions
    from bench.store import load_poison, load_questions
    from bench.workspace import OpenSearchWorkspace
    from ragantine.config import apply_overrides, load_models_config
    from ragantine.defenses.base import parse_defense_ids

    cfg = apply_overrides(load_config(config), overrides)
    mcfg = load_models_config()
    alias = model or mcfg.defaults.generator
    writer_alias = writer or mcfg.defaults.poison_writer
    ids = parse_defense_ids(defenses)
    seed = cfg.seed if seed is None else seed
    questions = load_questions(cfg)
    n = len(select_questions(questions, ["target"], split, limit))
    items = run_cost_items(mcfg, alias, n * (1 + rounds), ids)
    items.append(CallEstimate(purpose="attacker rewrites", model_alias=writer_alias, calls=2 * n * rounds,
                              prompt_tokens_each=250, completion_tokens_each=450))  # fmt: skip
    _confirm(estimate(mcfg, items).render(), yes)
    spec = RunSpec(
        run_id=make_run_id(ids, alias, split, seed) + "-adaptive",
        defenses=ids,
        model_alias=alias,
        sets=["target"],
        split=split,
        seed=seed,
        limit=limit,
    )
    workspace = OpenSearchWorkspace(cfg, mcfg, Settings(), log=typer.echo)
    report, _, out = _run_async(
        run_adaptive(
            cfg,
            mcfg,
            spec,
            questions,
            load_poison(cfg),
            workspace,
            writer_alias,
            rounds,
            typer.echo,
        )
    )
    for r in report.rounds:
        typer.echo(
            f"round {r.round}: ASR={r.asr} ({r.poisoned}/{r.n}), regenerated={r.regenerated}"
        )
    typer.echo(f"results -> {out}")


@bench_app.command("publish-dataset")
def bench_publish_dataset(
    repo_id: Annotated[
        str, typer.Option(help="Hugging Face dataset repo, e.g. <you>/ragantine-bench.")
    ],
    version: Annotated[str, typer.Option(help="Release tag.")] = "v0.1",
    out: Annotated[Path, typer.Option(help="Local release folder.")] = Path(
        "build/ragantine-bench"
    ),
    push_: Annotated[
        bool, typer.Option("--push", help="Upload to the Hub (needs HF_TOKEN).")
    ] = False,
    config: ConfigOpt = None,
) -> None:
    """Build the RAGantine-Bench release folder + dataset card; optionally push it with a version tag."""
    from bench.dataset.publish import build_release, push

    cfg = load_config(config)
    values = build_release(cfg, out, version)
    typer.echo(
        f"release folder ready: {out} ({values['n_questions']} questions, {values['n_poison']} poison docs)"
    )
    if push_:
        token = Settings().api_key("HF_TOKEN")
        if token is None:
            typer.echo("HF_TOKEN is not set (or is a placeholder) in .env", err=True)
            raise typer.Exit(1)
        typer.echo(f"pushed: {push(out, repo_id, token, version)}")


@bench_app.command("quality")
def bench_quality(
    runs: Annotated[str, typer.Option(help="Run ids (comma list), 'latest' or 'all'.")] = "latest",
    config: ConfigOpt = None,
) -> None:
    """Clean-set context recall + NLI faithfulness for finished runs (local models, no API cost)."""
    from bench.quality import quality_for_run
    from bench.report import resolve_runs
    from bench.store import load_questions
    from ragantine.defenses.d5_consistency import CrossEncoderNLI

    cfg = load_config(config)
    nli = CrossEncoderNLI(cfg.defenses.d5.nli_model)
    questions = load_questions(cfg)
    for rec in resolve_runs(cfg.paths.results_dir, runs):
        q = quality_for_run(rec.path, questions, nli, cfg.defenses.d5.support_threshold)
        typer.echo(
            f"{rec.run_id}: context_recall={q.context_recall} faithfulness={q.faithfulness} (n={q.n_answered})"
        )
