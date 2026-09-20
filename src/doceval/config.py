"""Environment and credential resolution.

The API key is read from the environment and nowhere else. It never appears
in source, in a profile, or in a committed file, and there is deliberately no
CLI flag for it: a key passed as an argument leaks into shell history and into
the process list.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

API_KEY_ENV = "TYPESAFE_API_KEY"
BASE_URL_ENV = "TYPESAFE_BASE_URL"
MODEL_ENV = "TYPESAFE_DEFAULT_MODEL"
DEFAULT_MODEL = "jev-latest"
CONSOLE_KEYS_URL = "https://console.typesafe.ai/keys"

_MISSING_KEY_MESSAGE = f"""{API_KEY_ENV} is not set

  export {API_KEY_ENV}=sk-...     create a key at {CONSOLE_KEYS_URL}
  or copy .env.example to .env and fill it in

  doceval lint needs no key and works now."""


class MissingAPIKey(Exception):
    """Raised before any work when no API key is available."""

    def __init__(self, message: str = _MISSING_KEY_MESSAGE) -> None:
        super().__init__(message)
        self.message = message


def load_env(start: Path | None = None) -> None:
    """Load a .env file from `start` or the current directory upward.

    A real environment variable always wins, so `override` stays False. A
    blank or whitespace-only exported value doesn't count as real, so it's
    cleared first, letting a `.env` value fill it in.
    """
    existing = os.environ.get(API_KEY_ENV)
    if existing is not None and not existing.strip():
        del os.environ[API_KEY_ENV]
    base = Path(start) if start is not None else Path.cwd()
    for directory in [base, *base.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


def require_api_key() -> str:
    """Return the API key, or raise MissingAPIKey before any work happens."""
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise MissingAPIKey()
    return key


def resolve_model() -> str:
    """Return the model id to send, honoring a pinned override."""
    return os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL
