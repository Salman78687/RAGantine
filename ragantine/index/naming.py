"""Index names per ingestion configuration (SPEC §B5.4).

Ingestion-stage defenses change what is indexed, so each distinct ingestion config gets
its own pair of indexes: ``<name>__<hash>`` and ``<quarantine_name>__<hash>``. The plain
clean index (no ingestion defenses, no extra documents) keeps the base names.
"""

from __future__ import annotations

import hashlib
import json

from ragantine.config import AppConfig

# Defenses whose hooks run at ingestion time (D2 only when configured to).
INGESTION_HOOK_IDS = ("d1", "d2")


def ingestion_defenses(cfg: AppConfig, enabled: list[str]) -> list[str]:
    ids = []
    for did in sorted(enabled):
        if did == "d1" or (did == "d2" and "ingestion" in cfg.defenses.d2.apply_at):
            ids.append(did)
    return ids


def ingestion_fingerprint(cfg: AppConfig, enabled: list[str], variant: str | None = None) -> str:
    ids = ingestion_defenses(cfg, enabled)
    blob = {
        "defenses": {d: getattr(cfg.defenses, d).model_dump(mode="json") for d in ids},
        "chunking": cfg.chunking.model_dump(mode="json"),
        "embedding": cfg.embedding.model,
        "variant": variant,
    }
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:10]


def index_names(cfg: AppConfig, enabled: list[str], variant: str | None = None) -> tuple[str, str]:
    """(main, quarantine) index names for this ingestion config and document set variant."""
    if not ingestion_defenses(cfg, enabled) and variant is None:
        return cfg.index.name, cfg.index.quarantine_name
    h = ingestion_fingerprint(cfg, enabled, variant)
    return f"{cfg.index.name}__{h}", f"{cfg.index.quarantine_name}__{h}"
