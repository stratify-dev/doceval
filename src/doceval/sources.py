"""Turning a path or a URL into one Document.

Both origins produce the same shape, so lint, evaluate, scoring, and report
never learn where a document came from.

The word-count floor matters more than the fetch. A JavaScript-rendered page
yields a couple hundred characters of navigation text, and the model would
score that fragment with high confidence. The floor stops a meaningless number
at the source, before any tokens are spent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import trafilatura

# Jev allows 32k tokens for the state plus the longest question. The margin
# leaves room for ten questions and their level descriptions.
MAX_DOCUMENT_TOKENS = 28_000
CHARS_PER_TOKEN = 4
USER_AGENT = "doceval/0.1 (+https://github.com/typesafe-ai)"

_FRONTMATTER = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n\s*", re.DOTALL)
_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


class SourceError(Exception):
    """One document could not be loaded. The run continues without it."""


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    origin: str
    fetched_at: datetime | None

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.text)


def is_url(arg: str) -> bool:
    return arg.startswith(("http://", "https://"))


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def strip_frontmatter(text: str) -> str:
    """Remove a leading YAML frontmatter block, leaving the body untouched."""
    return _FRONTMATTER.sub("", text, count=1)


def load_document(arg: str, *, timeout: float = 20.0, min_words: int = 150) -> Document:
    """Resolve one argument into a Document, or raise SourceError."""
    document = (
        _fetch_url(arg, timeout=timeout)
        if is_url(arg)
        else _read_file(arg)
    )
    _guard(document, min_words=min_words)
    return document


def _read_file(arg: str) -> Document:
    path = Path(arg)
    if not path.is_file():
        raise SourceError(f"{arg}: file not found")

    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise SourceError(f"{arg}: not UTF-8 text") from error

    body = strip_frontmatter(raw)
    heading = _H1.search(body)
    return Document(
        id=arg,
        title=heading.group(1) if heading else path.stem,
        text=body,
        origin="file",
        fetched_at=None,
    )


def _client(timeout: float) -> httpx.Client:
    """Isolated so tests substitute a MockTransport."""
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )


def _fetch_url(url: str, *, timeout: float) -> Document:
    try:
        with _client(timeout) as client:
            response = client.get(url)
    except httpx.TimeoutException as error:
        raise SourceError(f"{url}: timed out after {timeout}s") from error
    except httpx.HTTPError as error:
        raise SourceError(f"{url}: {error}") from error

    if response.status_code >= 400:
        raise SourceError(f"{url}: HTTP {response.status_code}")

    extracted = trafilatura.extract(
        response.text,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    if not extracted:
        extracted = ""

    return Document(
        id=url,
        title=_page_title(response.text) or url,
        text=extracted,
        origin="url",
        fetched_at=datetime.now(timezone.utc),
    )


def _page_title(html: str) -> str:
    metadata = trafilatura.extract_metadata(html)
    if metadata is not None and metadata.title:
        return str(metadata.title)
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _guard(document: Document, *, min_words: int) -> None:
    if document.word_count < min_words:
        raise SourceError(
            f"{document.id}: extracted {document.word_count} words, "
            f"below the {min_words} words floor. "
            f"The page may render its content with JavaScript."
        )
    tokens = document.estimated_tokens
    if tokens > MAX_DOCUMENT_TOKENS:
        raise SourceError(
            f"{document.id}: about {tokens:,} tokens, over the "
            f"{MAX_DOCUMENT_TOKENS:,} token budget for one state"
        )
