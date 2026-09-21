"""Answer persistence.

The key combines the document text with a fingerprint of everything sent to
the model. Editing a weight leaves both untouched, so retuning a rubric costs
nothing. Editing a document, an instruction, a level, or the audience moves the
key and the next run pays for a fresh judgment.

A republished URL re-evaluates on its own, because its extracted text changed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_CACHE_DIR = ".doceval-cache"


def cache_key(text: str, fingerprint: str) -> str:
    digest = hashlib.sha256()
    digest.update(text.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(fingerprint.encode("utf-8"))
    return digest.hexdigest()


def read(cache_dir: Path, key: str) -> dict | None:
    """Return a cached payload, or None when absent or unreadable.

    A corrupt entry is a cache miss, never an error. The run pays for a fresh
    request instead of failing on a half-written file.
    """
    path = Path(cache_dir) / f"{key}.json"
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def write(cache_dir: Path, key: str, payload: dict) -> None:
    """Persist a payload atomically, ignoring an unwritable cache directory."""
    directory = Path(cache_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        record = {**payload, "cached_at": datetime.now(UTC).isoformat()}
        temporary = directory / f"{key}.json.tmp"
        temporary.write_text(json.dumps(record), encoding="utf-8")
        temporary.replace(directory / f"{key}.json")
    except OSError:
        return  # a read-only cache directory degrades to no caching
