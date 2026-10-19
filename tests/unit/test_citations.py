"""Citation parsing, including the bracket style gpt-oss actually produces."""

from ragantine.generate.prompting import parse_citations
from tests.defense_helpers import chunk, hit

C = chunk("b", "The Market Place is an outdoor shopping center.", "web", title="The Market Place")


def test_square_brackets() -> None:
    assert [
        c.chunk_id for c in parse_citations(f"It is in Irvine [{C.chunk_id}].", [hit(C, 1.0)])
    ] == [C.chunk_id]


def test_full_width_brackets_from_gpt_oss() -> None:
    # regression: exact shape of a real open-large answer (2026-10-19)
    text = f"…straddling the borders of Irvine and Tustin【{C.chunk_id}】."
    cites = parse_citations(text, [hit(C, 1.0)])
    assert [c.chunk_id for c in cites] == [C.chunk_id] and cites[0].title == "The Market Place"


def test_unknown_ids_and_duplicates_are_ignored() -> None:
    text = f"[{C.chunk_id}] again [{C.chunk_id}] and [ffffffffffffffff:0001]"
    assert [c.chunk_id for c in parse_citations(text, [hit(C, 1.0)])] == [C.chunk_id]
