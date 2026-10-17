"""Naive baseline loaders keep hidden content (SPEC §B4.1); D1 is what strips it."""

import pymupdf

from ragantine.ingest.loaders import load_text
from tests.helpers import make_doc


def test_html_keeps_comments_alt_title_meta_and_hidden_css() -> None:
    html = (
        '<html><head><meta name="description" content="META-SECRET"><style>p{color:red}</style>'
        "</head><body><p>Visible paragraph.</p><!-- COMMENT-SECRET -->"
        '<span style="display:none">CSS-HIDDEN</span><img alt="ALT-SECRET" src="x.png">'
        '<a title="TITLE-SECRET">link</a><script>var SCRIPT = 1;</script></body></html>'
    )
    text = load_text(make_doc("H", html, fmt="html"))
    for needle in (
        "Visible paragraph.",
        "META-SECRET",
        "COMMENT-SECRET",
        "CSS-HIDDEN",
        "ALT-SECRET",
        "TITLE-SECRET",
    ):
        assert needle in text
    assert "SCRIPT" not in text and "color:red" not in text


def test_html_blocks_become_paragraphs() -> None:
    text = load_text(make_doc("H", "<p>First one.</p><p>Second one.</p>", fmt="html"))
    assert text == "First one.\n\nSecond one."


def test_markdown_renders_through_html_loader() -> None:
    md = "# Heading\n\nSome *text* here.\n\n<!-- MD-COMMENT -->\n"
    text = load_text(make_doc("M", md, fmt="markdown"))
    assert "Heading" in text and "Some text here." in text and "MD-COMMENT" in text


def test_pdf_keeps_white_and_tiny_text() -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Visible PDF text.", fontsize=12)
    page.insert_text((72, 100), "WHITE-SECRET", fontsize=1, color=(1, 1, 1))
    data = doc.tobytes()
    text = load_text(make_doc("P", data, fmt="pdf"))
    assert "Visible PDF text." in text and "WHITE-SECRET" in text


def test_text_passthrough_normalises_whitespace() -> None:
    assert load_text(make_doc("T", "a  b\n\n\n\nc")) == "a b\n\nc"
