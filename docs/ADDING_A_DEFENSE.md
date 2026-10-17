# Adding your own defense

A defense is a class with an id, a config section, and the hooks it needs. Each hook has a
pass-through default, so implement only what you use. Every block must explain itself
with a `DefenseDecision`.

## 1. Write the class

```python
# ragantine/defenses/d7_example.py
from ragantine.defenses.base import Defense, register
from ragantine.schemas import DefenseDecision, RetrievedChunk


@register
class LengthGuard(Defense):
    """Filters retrieved chunks that are suspiciously short (example only)."""

    id = "d7"

    def on_retrieve(self, q: str, hits: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], list[DefenseDecision]]:
        keep, decisions = [], []
        for h in hits:
            if h.chunk.n_tokens < int(self.cfg["min_tokens"]):
                decisions.append(self.decision(stage="retrieval", target_id=h.chunk.chunk_id, action="filter",
                                               score=float(h.chunk.n_tokens), reason="too_short: below min_tokens"))
            else:
                keep.append(h)
        return keep, decisions
```

The hooks run in id order:

| Hook | Stage | Return | Effect of a decision |
|---|---|---|---|
| `on_load(doc, text)` | ingestion | `(text, decisions)` | `quarantine` targeting `doc.doc_id` holds every chunk of the document |
| `on_chunks(chunks)` | ingestion | `(chunks, decisions)` | `quarantine` targeting a `chunk_id` routes it to the quarantine store |
| `on_retrieve(q, hits)` | retrieval | `(hits, decisions)` | removing a hit + `filter` drops it for this query only |
| `on_prompt(q, hits, messages)` | generation | `messages` | rewrite or replace the prompt (see D4) |
| `on_verify(q, hits, draft, ctx)` | verification | `(text, decisions)` | `ctx.regenerate(hits, note)` asks the model again (see D5) |

## 2. Register the id and config

- Add `"d7"` to `DefenseId` in `ragantine/schemas.py` and to `ALL_IDS` in `ragantine/defenses/base.py`.
- Add a `D7Config` model to `ragantine/config.py`, add it to `DefensesConfig`, and add its
  values to `configs/default.yaml`. No magic numbers in code.
- Import the module in `ragantine/defenses/__init__.py` so it registers itself.
- If the defense acts at ingestion, add its id to `ingestion_defenses()` in
  `ragantine/index/naming.py`, so each configuration gets its own index.

`self.cfg` holds your section plus shared values: `final_k`, `fused_k`, `models_dir`, `data_dir`.

## 3. Test it

- Unit tests next to the others (`tests/unit/test_d*.py`) using `tests/defense_helpers.py`.
  No network and no OpenSearch; use `FakeLLM` and stub models.
- The coverage gate requires ≥ 80% on `ragantine/defenses`.
- Check end to end offline: `uv run ragantine bench smoke`.

## 4. Measure it

```bash
uv run ragantine bench run --defenses d7 --split dev --limit 60      # tune on dev only
uv run ragantine bench ablation --include-d6 …                        # add it to the protocol if it earns a row
uv run ragantine bench report --runs all
```

Tune on the dev split only, and report where your defense fails, not only where it wins.
