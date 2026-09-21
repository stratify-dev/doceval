"""Async fan-out over documents.

One request per document, every question inside it. Documents run concurrently
under a semaphore, so a corpus finishes in about the time of its slowest
document rather than the sum of all of them.

A failure is isolated to its own document. One 404 or one timeout never ends a
run, because a partial report beats no report.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from typesafe_sdk import (
    AsyncTypeSafeClient,
    RetryPolicy,
    TypeSafeUnprocessableEntityError,
)

from . import cache as cache_mod
from .config import resolve_model
from .profile import Profile
from .questions import build_questions, build_state, questions_fingerprint
from .sources import Document

DEFAULT_CONCURRENCY = 8

# Per-attempt HTTP timeout for one system_one call. Without an explicit
# value, AsyncTypeSafeClient falls back to the SDK's own DEFAULT_TIMEOUT
# (10.0s, typesafe_sdk.constants.DEFAULT_TIMEOUT) -- far too short for the
# single request this module sends per document: up to MAX_DOCUMENT_TOKENS
# (28k, see sources.py) of state plus eleven questions, all in one call.
# This is distinct from the retry budget below, which is the TOTAL budget
# across every attempt and its backoff, not a per-attempt timeout. 60s is a
# practical default for that request shape; --api-timeout overrides it per
# run for a slower model or a corpus of unusually large documents.
API_TIMEOUT = 60.0

# 408, 429, and every 5xx retry by default, which covers 529 Overloaded.
# respect_retry_after honors the header when the response carries one.
RETRY_MAX_RETRIES = 3
RETRY_BACKOFF_INITIAL = 0.5
RETRY_BACKOFF_MAX = 8.0


def _retry_budget(api_timeout: float) -> float:
    """The total retry budget for a client built with this api_timeout.

    RetryPolicy.timeout is measured from the start of the FIRST attempt, not
    reset per attempt (typesafe_sdk._core.retry.RetryPolicy.timeout), and
    tenacity's stop_before_delay refuses the next retry once elapsed time
    plus the upcoming backoff would reach that budget
    (tenacity.stop.stop_before_delay). So THE INVARIANT THAT MUST HOLD: the
    budget must exceed one full-length attempt (api_timeout) plus its
    backoff delay, or the very first timeout consumes the whole budget and
    no retry can ever fire -- the bug this function exists to prevent.

    This used to be a fixed constant (180.0), which only satisfied that
    invariant for the default api_timeout of 60.0. Once --api-timeout
    became a user-settable flag, a caller passing --api-timeout above 180
    reintroduced the exact dead-retry bug in the flag added to fix it,
    silently, because RETRY.timeout no longer moved with it.

    Budgeting for RETRY_MAX_RETRIES full-length attempts (not
    RETRY_MAX_RETRIES + 1, the total attempt count including the initial
    one) leaves comfortable room for at least one retry after a
    full-length attempt, at any api_timeout, without letting one document
    hang for the full attempt count times its timeout.
    """
    return RETRY_MAX_RETRIES * api_timeout


def _retry_policy(api_timeout: float) -> RetryPolicy:
    """Build the retry policy for one client, budgeted for this api_timeout."""
    return RetryPolicy(
        max_retries=RETRY_MAX_RETRIES, backoff_initial=RETRY_BACKOFF_INITIAL,
        backoff_max=RETRY_BACKOFF_MAX, respect_retry_after=True,
        timeout=_retry_budget(api_timeout),
    )


@dataclass(frozen=True)
class Outcome:
    document: Document
    answers: dict[str, dict] | None
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    cached: bool = False
    error: str | None = None


def _new_client(**kwargs) -> AsyncTypeSafeClient:
    """Isolated so tests substitute a fake client.

    The retry policy is built here, from whatever timeout this call
    carries, rather than reused from a module-level constant -- see
    _retry_budget for why a fixed budget can't be correct for every
    api_timeout a caller might pass.
    """
    kwargs.setdefault("retry", _retry_policy(kwargs.get("timeout", API_TIMEOUT)))
    return AsyncTypeSafeClient(**kwargs)


def describe_invalid_request(error: Exception) -> str:
    """Turn a 422 into a message naming the question and the field.

    The body follows the usual validation shape, where `loc` walks the request
    path. `questions.active_voice.criteria` tells you which rubric to fix; a
    bare status code does not.
    """
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("error")
        if isinstance(detail, list) and detail and isinstance(detail[0], dict):
            first = detail[0]
            location = ".".join(str(part) for part in first.get("loc", []))
            message = first.get("msg", "failed validation")
            return f"invalid question definition at {location or 'request'}: {message}"
        if detail:
            return f"invalid request: {detail}"
    return f"invalid request: {error}"


async def evaluate_documents(
    documents: Sequence[Document],
    prof: Profile,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    cache_dir: Path | str | None = None,
    use_cache: bool = True,
    api_timeout: float = API_TIMEOUT,
    on_start: Callable[[Document], None] | None = None,
    on_done: Callable[[Outcome], None] | None = None,
) -> list[Outcome]:
    """Evaluate every document, returning outcomes in input order."""
    if not documents:
        return []

    questions = build_questions(prof)
    fingerprint = questions_fingerprint(prof)
    directory = Path(cache_dir or cache_mod.DEFAULT_CACHE_DIR)
    model = resolve_model()
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async with _new_client(timeout=api_timeout) as client:

        async def run(document: Document) -> Outcome:
            def finish(outcome: Outcome) -> Outcome:
                if on_done is not None:
                    try:
                        on_done(outcome)
                    except Exception:
                        # A broken progress callback must not cost this
                        # document its already-computed result.
                        pass
                return outcome

            try:
                if on_start is not None:
                    on_start(document)

                key = cache_mod.cache_key(document.text, fingerprint)
                if use_cache:
                    hit = cache_mod.read(directory, key)
                    if hit is not None:
                        return finish(Outcome(
                            document=document, answers=hit.get("answers", {}),
                            model=hit.get("model", model), usage={}, cached=True,
                        ))

                async with semaphore:
                    try:
                        response = await client.system_one(
                            build_state(document, prof), questions
                        )
                    except TypeSafeUnprocessableEntityError as error:
                        # A 422 is a profile bug, not a runtime one. Name the field.
                        return finish(Outcome(
                            document=document, answers=None, model=model,
                            error=describe_invalid_request(error),
                        ))
                    except Exception as error:  # isolate one document's failure
                        detail = getattr(error, "request_id", None)
                        suffix = f" (request {detail})" if detail else ""
                        return finish(Outcome(
                            document=document, answers=None, model=model,
                            error=f"{type(error).__name__}: {error}{suffix}",
                        ))

                payload = response.raw_http_response.json()
                answers = dict(payload.get("answers", {}))
                answered_model = str(payload.get("model", model))
                usage = dict(payload.get("usage", {}))

                if use_cache:
                    cache_mod.write(
                        directory, key,
                        {"model": answered_model, "answers": answers, "usage": usage},
                    )

                return finish(Outcome(
                    document=document, answers=answers, model=answered_model,
                    usage=usage,
                ))
            except Exception as error:
                # Nothing above this point may escape into gather: a raising
                # on_start, a corrupt cache read, or a malformed response must
                # land as this document's own error, not take the whole run
                # down with it. A raising on_done is already swallowed inside
                # finish(), so a document that was actually evaluated keeps
                # its real answers instead of being overwritten here.
                return finish(Outcome(
                    document=document, answers=None, model=model,
                    error=f"{type(error).__name__}: {error}",
                ))

        return list(await asyncio.gather(*(run(d) for d in documents)))
