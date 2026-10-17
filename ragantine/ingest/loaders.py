"""Per-format loaders returning plain text (SPEC §B4.1).

The baseline loaders are deliberately naive, like most real pipelines: HTML keeps every
text node including comments, ``alt``/``title`` attributes and ``<meta name=description>``,
and ignores CSS (so ``display:none`` text survives); PDF keeps white and tiny text.
D1 (``ragantine.defenses.d1_sanitizer``) re-parses the raw document to strip hidden text.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup
from bs4.element import Comment, NavigableString, Tag
from markdown_it import MarkdownIt

from ragantine.schemas import RawDocument

BLOCK_TAGS = frozenset(
    [
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "br",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "summary",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    ]
)
SKIP_TAGS = frozenset({"script", "style", "noscript", "template"})
ATTR_TEXT = ("alt", "title")
_SPACES = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n{3,}")

_md = MarkdownIt("commonmark")


def as_text(raw: str | bytes) -> str:
    return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw


def normalize_whitespace(text: str) -> str:
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANKS.sub("\n\n", "\n".join(lines)).strip()


def html_to_text(html: str, keep_hidden: bool = True) -> str:
    """Walk the DOM in document order; block elements become paragraph breaks."""
    soup = BeautifulSoup(html, "lxml")
    parts: list[str] = []

    if keep_hidden:
        for meta in soup.find_all("meta"):
            if isinstance(meta, Tag) and str(meta.get("name", "")).lower() == "description":
                parts.append(str(meta.get("content", "")) + "\n\n")

    def walk(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, Comment):
                if keep_hidden:
                    parts.append(f" {child} ")
            elif isinstance(child, NavigableString):
                parts.append(str(child))
            elif isinstance(child, Tag):
                if child.name in SKIP_TAGS or child.name == "head":
                    continue
                block = child.name in BLOCK_TAGS
                if block:
                    parts.append("\n\n")
                if keep_hidden:
                    for attr in ATTR_TEXT:
                        if child.get(attr):
                            parts.append(f" {child.get(attr)} ")
                walk(child)
                if block:
                    parts.append("\n\n")

    root = soup.body or soup
    walk(root)
    return normalize_whitespace("".join(parts))


def markdown_to_html(md: str) -> str:
    return str(_md.render(md))


def pdf_to_text(data: bytes) -> str:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        return normalize_whitespace("\n\n".join(page.get_text() for page in pdf))


def load_text(doc: RawDocument) -> str:
    """Naive baseline loader: every format to plain text, hidden content included."""
    if doc.fmt == "text":
        return normalize_whitespace(as_text(doc.raw))
    if doc.fmt == "html":
        return html_to_text(as_text(doc.raw))
    if doc.fmt == "markdown":
        return html_to_text(markdown_to_html(as_text(doc.raw)))
    if doc.fmt == "pdf":
        if not isinstance(doc.raw, bytes):
            raise TypeError(f"pdf document {doc.doc_id} must carry bytes")
        return pdf_to_text(doc.raw)
    raise ValueError(f"unsupported format {doc.fmt!r}")
