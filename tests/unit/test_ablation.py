"""Ablation plan follows SPEC §10: baseline, singles, full stack, leave-one-out per model x seed."""

from bench.ablation import configurations, plan


def test_configurations() -> None:
    confs = configurations()
    assert confs[0] == [] and len(confs) == 12  # D0 + 5 singles + full + 5 leave-one-out
    assert ["d1", "d2", "d3", "d4", "d5"] in confs
    assert ["d1", "d2", "d4", "d5"] in confs
    assert len(configurations(include_d6=True)) == 13


def test_plan_covers_models_and_seeds_with_unique_ids() -> None:
    p = plan(["m1", "m2"], [1, 2, 3])
    assert len(p.specs) == 2 * 3 * 12
    assert len({s.run_id for s in p.specs}) == len(p.specs)
    assert all(s.split == "test" and s.sets == ["target", "clean", "paraphrase"] for s in p.specs)
    labels = {s.label for s in p.specs}
    assert {"D0 Baseline", "Full stack", "Full stack − D3", "+ D2 Injection classifier"} <= labels
