"""Toggleable defenses D1-D6 behind one ``Defense`` interface (SPEC §B6).

Importing this package registers every implemented defense in ``REGISTRY``.
"""

from ragantine.defenses import (  # noqa: F401  (imports register the defenses)
    d1_sanitizer,
    d2_classifier,
    d3_trust,
    d4_spotlight,
    d5_consistency,
    d6_anomaly,
)
from ragantine.defenses.base import (
    ALL_IDS,
    FULL_STACK,
    REGISTRY,
    Defense,
    QueryContext,
    build_defenses,
    parse_defense_ids,
    register,
)

__all__ = [
    "ALL_IDS",
    "FULL_STACK",
    "REGISTRY",
    "Defense",
    "QueryContext",
    "build_defenses",
    "parse_defense_ids",
    "register",
]
