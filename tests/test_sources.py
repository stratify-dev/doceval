from datetime import datetime

import httpx
import pytest

from doceval import sources

PROSE = " ".join(["word"] * 200)

HTML = f"""<!doctype html>
<html><head><title>A Real Article</title></head>
<body>
  <nav>Home About Contact</nav>
  <article><h1>A Real Article</h1><p>{PROSE}</p></article>
  <footer>Copyright 2026</footer>
</body></html>"""


def test_is_url():
    assert sources.is_url("https://example.com/post")
    assert sources.is_url("http://example.com")
    assert not sources.is_url("post.md")
    assert not sources.is_url("/abs/path/post.md")


def test_strip_frontmatter_removes_block():
    text = "---\ntitle: X\ntags: [a]\n---\n\nBody text.\n"
    assert sources.strip_frontmatter(text) == "Body text.\n"


def test_strip_frontmatter_leaves_plain_text():
    assert sources.strip_frontmatter("Body only.\n") == "Body only.\n"


def test_strip_frontmatter_ignores_a_later_rule():
    text = "Body.\n\n---\n\nMore.\n"
    assert sources.strip_frontmatter(text) == text


def test_loads_a_file(tmp_path):
    path = tmp_path / "post.md"
    path.write_text(f"---\ntitle: T\n---\n\n# My Post\n\n{PROSE}\n")
    doc = sources.load_document(str(path), min_words=10)
    assert doc.origin == "file"
    assert doc.title == "My Post"
    assert "title: T" not in doc.text
    assert doc.fetched_at is None


def test_file_title_falls_back_to_stem(tmp_path):
    path = tmp_path / "my-post.md"
    path.write_text(PROSE)
    assert sources.load_document(str(path), min_words=10).title == "my-post"


def test_missing_file_raises():
    with pytest.raises(sources.SourceError, match="not found"):
        sources.load_document("nope.md")


def test_file_under_min_words_raises(tmp_path):
    path = tmp_path / "stub.md"
    path.write_text("Too short.")
    with pytest.raises(sources.SourceError, match="150 words"):
        sources.load_document(str(path), min_words=150)


def test_oversized_document_raises(tmp_path):
    path = tmp_path / "huge.md"
    path.write_text("word " * 200_000)
    with pytest.raises(sources.SourceError, match="token budget"):
        sources.load_document(str(path), min_words=10)


def test_loads_a_url(monkeypatch):
    def handler(request):
        return httpx.Response(200, html=HTML)

    monkeypatch.setattr(
        sources, "_client",
        lambda timeout: httpx.Client(transport=httpx.MockTransport(handler)),
    )
    doc = sources.load_document("https://example.com/post", min_words=10)
    assert doc.origin == "url"
    assert doc.id == "https://example.com/post"
    assert "word" in doc.text
    assert "Home About Contact" not in doc.text
    assert "Copyright 2026" not in doc.text
    assert isinstance(doc.fetched_at, datetime)


def test_url_http_error_raises(monkeypatch):
    def handler(request):
        return httpx.Response(404)

    monkeypatch.setattr(
        sources, "_client",
        lambda timeout: httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(sources.SourceError, match="404"):
        sources.load_document("https://example.com/gone")


def test_url_with_no_article_text_raises(monkeypatch):
    def handler(request):
        return httpx.Response(200, html="<html><body><nav>Home</nav></body></html>")

    monkeypatch.setattr(
        sources, "_client",
        lambda timeout: httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(sources.SourceError, match="150 words"):
        sources.load_document("https://example.com/spa", min_words=150)


def test_estimate_tokens_is_proportional():
    assert sources.estimate_tokens("a" * 400) == 100


def test_word_count_property(tmp_path):
    path = tmp_path / "p.md"
    path.write_text("one two three four five")
    assert sources.load_document(str(path), min_words=1).word_count == 5
