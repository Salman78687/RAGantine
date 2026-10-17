"""Toggleable defenses D1-D6 behind one ``Defense`` interface (SPEC §B6).

Importing this package registers every implemented defense in ``REGISTRY``.
"""

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
