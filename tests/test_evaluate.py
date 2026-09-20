import asyncio
from types import SimpleNamespace

import pytest
from tenacity.stop import stop_before_delay

from doceval import evaluate, sources
from doceval import profile as profile_mod

PROF = profile_mod.Profile(
    name="t", audience="Developers.",
    dimensions=(profile_mod.Dimension("a", "house_style", 1.0, "q", ("x", "y")),),
    gate=None,
)

ANSWERS = {"a": {"type": "score", "score": 1.0, "confidence": 0.9,
                 "legend": {"0": "x", "1": "y"}, "probabilities": {"0": 0.0, "1": 1.0}}}


def make_docs(count):
    return [
        sources.Document(f"doc{i}.md", f"Doc {i}", f"Body {i}.", "file", None)
        for i in range(count)
    ]


class FakeResponse:
    def __init__(self, answers, model="jev-1.13.0"):
        self.raw_http_response = self
        self._payload = {"answers": answers, "model": model,
                         "usage": {"input_tokens": 100, "output_tokens": 10}}
        self.model = model

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, *, fail_on=(), delay=0.0):
        self.fail_on = set(fail_on)
        self.delay = delay
        self.calls = []
        self.concurrent = 0
        self.peak = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def system_one(self, state, questions, **kwargs):
        path = state["document"]["path"]
        self.calls.append(path)
        self.concurrent += 1
        self.peak = max(self.peak, self.concurrent)
        try:
            await asyncio.sleep(self.delay)
            if path in self.fail_on:
                raise RuntimeError(f"boom for {path}")
            return FakeResponse(ANSWERS)
        finally:
            self.concurrent -= 1


@pytest.fixture
def fake(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(evaluate, "_new_client", lambda **kwargs: client)
    return client


async def test_evaluates_every_document(fake, tmp_path):
    outcomes = await evaluate.evaluate_documents(
        make_docs(3), PROF, cache_dir=tmp_path, use_cache=False)
    assert len(outcomes) == 3
    assert all(o.error is None for o in outcomes)
    assert outcomes[0].answers["a"]["score"] == 1.0


async def test_preserves_input_order(fake, tmp_path):
    docs = make_docs(5)
    outcomes = await evaluate.evaluate_documents(
        docs, PROF, cache_dir=tmp_path, use_cache=False)
    assert [o.document.id for o in outcomes] == [d.id for d in docs]


async def test_one_failure_does_not_stop_the_run(monkeypatch, tmp_path):
    client = FakeClient(fail_on={"doc1.md"})
    monkeypatch.setattr(evaluate, "_new_client", lambda **kwargs: client)
    outcomes = await evaluate.evaluate_documents(
        make_docs(3), PROF, cache_dir=tmp_path, use_cache=False)
    assert outcomes[1].error is not None
    assert "boom" in outcomes[1].error
    assert outcomes[0].error is None and outcomes[2].error is None


async def test_respects_the_concurrency_limit(monkeypatch, tmp_path):
    client = FakeClient(delay=0.01)
    monkeypatch.setattr(evaluate, "_new_client", lambda **kwargs: client)
    await evaluate.evaluate_documents(
        make_docs(8), PROF, concurrency=2, cache_dir=tmp_path, use_cache=False)
    # == pins the lower bound too: <= 2 stays true even if concurrency
    # silently serialized down to 1, which wouldn't actually be respecting
    # the limit, just never hitting it.
    assert client.peak == 2


async def test_cache_hit_skips_the_api(monkeypatch, tmp_path):
    client = FakeClient()
    monkeypatch.setattr(evaluate, "_new_client", lambda **kwargs: client)
    docs = make_docs(1)
    first = await evaluate.evaluate_documents(docs, PROF, cache_dir=tmp_path)
    second = await evaluate.evaluate_documents(docs, PROF, cache_dir=tmp_path)
    assert len(client.calls) == 1
    assert first[0].cached is False
    assert second[0].cached is True
    assert second[0].answers == first[0].answers


async def test_no_cache_flag_forces_a_request(monkeypatch, tmp_path):
    client = FakeClient()
    monkeypatch.setattr(evaluate, "_new_client", lambda **kwargs: client)
    docs = make_docs(1)
    await evaluate.evaluate_documents(docs, PROF, cache_dir=tmp_path)
    await evaluate.evaluate_documents(docs, PROF, cache_dir=tmp_path, use_cache=False)
    assert len(client.calls) == 2


async def test_edited_text_misses_the_cache(monkeypatch, tmp_path):
    client = FakeClient()
    monkeypatch.setattr(evaluate, "_new_client", lambda **kwargs: client)
    await evaluate.evaluate_documents(make_docs(1), PROF, cache_dir=tmp_path)
    edited = [sources.Document("doc0.md", "Doc 0", "CHANGED.", "file", None)]
    await evaluate.evaluate_documents(edited, PROF, cache_dir=tmp_path)
    assert len(client.calls) == 2


async def test_callbacks_fire_per_document(fake, tmp_path):
    started, finished = [], []
    await evaluate.evaluate_documents(
        make_docs(2), PROF, cache_dir=tmp_path, use_cache=False,
        on_start=started.append, on_done=lambda o: finished.append(o.document.id),
    )
    assert len(started) == 2
    assert sorted(finished) == ["doc0.md", "doc1.md"]


async def test_usage_is_reported(fake, tmp_path):
    outcomes = await evaluate.evaluate_documents(
        make_docs(1), PROF, cache_dir=tmp_path, use_cache=False)
    assert outcomes[0].usage["input_tokens"] == 100


def test_describe_invalid_request_names_the_field():
    error = RuntimeError("boom")
    error.body = {"detail": [{"loc": ["questions", "active_voice", "criteria"],
                              "msg": "must have at least 2 levels"}]}
    message = evaluate.describe_invalid_request(error)
    assert "questions.active_voice.criteria" in message
    assert "at least 2 levels" in message


def test_describe_invalid_request_falls_back_gracefully():
    error = RuntimeError("boom")
    error.body = None
    assert "boom" in evaluate.describe_invalid_request(error)


async def test_a_422_reports_the_offending_field(monkeypatch, tmp_path):
    from typesafe_sdk import TypeSafeUnprocessableEntityError

    class Failing(FakeClient):
        async def system_one(self, state, questions, **kwargs):
            # Built without calling __init__, so the test does not depend on
            # the SDK exception's constructor signature.
            error = TypeSafeUnprocessableEntityError.__new__(
                TypeSafeUnprocessableEntityError
            )
            error.args = ("unprocessable",)
            error.body = {"detail": [{"loc": ["questions", "concision", "criteria"],
                                      "msg": "too many levels"}]}
            raise error

    monkeypatch.setattr(evaluate, "_new_client", lambda **kwargs: Failing())
    [outcome] = await evaluate.evaluate_documents(
        make_docs(1), PROF, cache_dir=tmp_path, use_cache=False)
    assert "questions.concision.criteria" in outcome.error


def test_client_is_built_with_a_retry_policy():
    policy = evaluate._retry_policy(evaluate.API_TIMEOUT)
    assert policy.max_retries == 3
    assert policy.respect_retry_after is True
    assert 429 in policy.http_statuses
    assert 529 in policy.http_statuses


# --- Fix wave, finding 3 re-review -----------------------------------------
#
# The retry budget used to be a fixed constant (180.0), which cleared the
# invariant below only for the default api_timeout (60.0). --api-timeout is
# user-settable, so a caller raising it past 180 silently reintroduced the
# exact dead-retry bug finding 3 existed to fix, in the flag it added. The
# budget now has to be DERIVED from api_timeout, so it holds at every value,
# not just the default one a fixed-constant test would keep pinned to.
#
# A prior version of this test asserted `evaluate.RETRY.timeout >
# evaluate.API_TIMEOUT` -- comparing two constants to each other, which
# stays true no matter what value either constant holds, and so cannot
# fail no matter how wrong the derivation is for any api_timeout other than
# the one baked into the constants. This project has already found seven
# instances of that exact assertion shape (something that holds while its
# bug is present); this was an eighth. The test below instead drives the
# REAL mechanism tenacity uses to decide whether to retry
# (tenacity.stop.stop_before_delay, wired in via
# typesafe_sdk._core.retry.RetryPolicy._stop) at several api_timeout
# values, including several well above the old fixed 180.0, and checks it
# would actually let a retry fire.


@pytest.mark.parametrize("api_timeout", [1.0, 10.0, 60.0, 90.0, 181.0, 300.0, 600.0])
def test_retry_budget_leaves_room_for_a_retry_after_one_full_attempt(api_timeout):
    """Simulate the moment right after one full-length attempt has just
    timed out: seconds_since_start == api_timeout, with the resulting
    backoff delay about to be slept. stop_before_delay(budget) returning
    False there means tenacity proceeds with the retry; True means the
    budget was already exhausted and no retry ever fires -- the bug this
    test exists to catch, at every api_timeout, not only the default.
    """
    budget = evaluate._retry_budget(api_timeout)
    stop = stop_before_delay(budget)
    state = SimpleNamespace(
        seconds_since_start=api_timeout,
        upcoming_sleep=evaluate.RETRY_BACKOFF_INITIAL,
    )
    assert stop(state) is False


def test_retry_policy_timeout_matches_the_derived_budget():
    """_new_client must actually use _retry_budget's number, not merely
    have it exist unused somewhere."""
    policy = evaluate._retry_policy(250.0)
    assert policy.timeout == evaluate._retry_budget(250.0)
    assert policy.timeout == pytest.approx(750.0)


# --- Fix wave, finding 3 --------------------------------------------------
#
# _new_client() used to be called with no timeout at all, so
# AsyncTypeSafeClient fell back to the SDK's own 10s-per-attempt default --
# far too short for the one request this module sends per document, which
# can carry up to 28k tokens of state plus eleven questions.


async def test_client_is_built_with_an_explicit_timeout(monkeypatch, tmp_path):
    captured = {}

    def fake_new_client(**kwargs):
        captured.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(evaluate, "_new_client", fake_new_client)
    await evaluate.evaluate_documents(make_docs(1), PROF, cache_dir=tmp_path, use_cache=False)

    assert captured.get("timeout") == evaluate.API_TIMEOUT


async def test_api_timeout_override_reaches_the_client(monkeypatch, tmp_path):
    captured = {}

    def fake_new_client(**kwargs):
        captured.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(evaluate, "_new_client", fake_new_client)
    await evaluate.evaluate_documents(
        make_docs(1), PROF, cache_dir=tmp_path, use_cache=False, api_timeout=5.0)

    assert captured.get("timeout") == 5.0


def test_new_client_derives_the_retry_budget_from_its_own_timeout(monkeypatch):
    """Exercises the real _new_client (not monkeypatched away, unlike the
    two tests above), monkeypatching only the SDK's AsyncTypeSafeClient
    underneath it, to prove the retry policy that actually reaches the SDK
    carries a budget derived from the timeout _new_client was given --
    not a policy built once from the module default and reused regardless
    of what api_timeout the caller passed.
    """
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(evaluate, "AsyncTypeSafeClient", fake_client)
    evaluate._new_client(timeout=250.0)

    assert captured["retry"].timeout == evaluate._retry_budget(250.0)


# --- Properties beyond the brief -------------------------------------------
#
# Each of these pins a failure mode that would be silent: the run would still
# "complete", just with lost data or a corrupted result, and none of the
# tests above would catch it.


async def test_cache_hits_do_not_consume_a_concurrency_slot(monkeypatch, tmp_path):
    """A fully cached corpus makes zero client calls, even with concurrency=1.

    This pins that cache hits never reach client.system_one, which is a real
    property worth pinning on its own. It does NOT prove the cache check runs
    before the semaphore is acquired -- moving the check inside `async with
    semaphore:` would still make zero client calls and pass this same
    assertion, since a cache hit never needs the client either way. See
    test_cache_check_happens_before_the_semaphore_is_acquired below for the
    test that actually discriminates on ordering.
    """
    warm_client = FakeClient()
    monkeypatch.setattr(evaluate, "_new_client", lambda **kwargs: warm_client)
    docs = make_docs(5)
    await evaluate.evaluate_documents(docs, PROF, cache_dir=tmp_path)  # populate the cache

    cold_client = FakeClient()
    monkeypatch.setattr(evaluate, "_new_client", lambda **kwargs: cold_client)
    outcomes = await evaluate.evaluate_documents(
        docs, PROF, concurrency=1, cache_dir=tmp_path)

    assert len(outcomes) == 5
    assert all(o.cached for o in outcomes)
    assert cold_client.calls == []


async def test_input_order_survives_out_of_order_completion(monkeypatch, tmp_path):
    """Outcomes come back in input order even when the last document to be
    requested is the first to finish."""

    class ReverseDelayClient(FakeClient):
        def __init__(self, delays):
            super().__init__()
            self.delays = delays

        async def system_one(self, state, questions, **kwargs):
            path = state["document"]["path"]
            self.calls.append(path)
            await asyncio.sleep(self.delays[path])
            return FakeResponse(ANSWERS)

    docs = make_docs(4)
    # doc0 is requested first but sleeps longest; doc3 is requested last but
    # sleeps least, so completion order is the exact reverse of input order.
    delays = {doc.id: 0.03 * (len(docs) - i) for i, doc in enumerate(docs)}
    client = ReverseDelayClient(delays)
    monkeypatch.setattr(evaluate, "_new_client", lambda **kwargs: client)

    outcomes = await evaluate.evaluate_documents(
        docs, PROF, concurrency=len(docs), cache_dir=tmp_path, use_cache=False)

    assert [o.document.id for o in outcomes] == [d.id for d in docs]


async def test_a_failure_does_not_corrupt_survivors_answers(monkeypatch, tmp_path):
    """A mid-list failure leaves the other outcomes carrying their real
    answers, not merely error=None with empty or stale data."""
    client = FakeClient(fail_on={"doc1.md"})
    monkeypatch.setattr(evaluate, "_new_client", lambda **kwargs: client)
    outcomes = await evaluate.evaluate_documents(
        make_docs(3), PROF, cache_dir=tmp_path, use_cache=False)

    assert outcomes[1].answers is None
    assert outcomes[0].answers["a"]["score"] == 1.0
    assert outcomes[0].answers["a"]["confidence"] == 0.9
    assert outcomes[2].answers["a"]["score"] == 1.0
    assert outcomes[2].answers["a"]["confidence"] == 0.9


async def test_cached_and_live_outcomes_feed_scoring_identically(monkeypatch, tmp_path):
    """A cached outcome and a live outcome carry the same answers and score
    to the same composite, so a report can't tell the two apart."""
    from doceval import scoring

    client = FakeClient()
    monkeypatch.setattr(evaluate, "_new_client", lambda **kwargs: client)
    docs = make_docs(1)

    [live] = await evaluate.evaluate_documents(docs, PROF, cache_dir=tmp_path)
    [cached] = await evaluate.evaluate_documents(docs, PROF, cache_dir=tmp_path)

    assert live.cached is False
    assert cached.cached is True
    assert live.answers == cached.answers

    live_result = scoring.score_document(
        document=docs[0], prof=PROF, answers=live.answers, violations=(),
        min_confidence=0.5, model=live.model, cached=live.cached)
    cached_result = scoring.score_document(
        document=docs[0], prof=PROF, answers=cached.answers, violations=(),
        min_confidence=0.5, model=cached.model, cached=cached.cached)

    assert live_result.composite == cached_result.composite
    assert live_result.verdict == cached_result.verdict


async def test_a_422_message_never_degrades_to_a_bare_status(monkeypatch, tmp_path):
    """The 422 handler names the question and field, not just 'Unprocessable
    Entity' or a status code -- that's the whole point of catching it
    separately from the generic error path."""
    from typesafe_sdk import TypeSafeUnprocessableEntityError

    class Failing(FakeClient):
        async def system_one(self, state, questions, **kwargs):
            error = TypeSafeUnprocessableEntityError.__new__(
                TypeSafeUnprocessableEntityError
            )
            error.args = ("unprocessable",)
            error.body = {"detail": [{"loc": ["questions", "concision", "criteria"],
                                      "msg": "too many levels"}]}
            raise error

    monkeypatch.setattr(evaluate, "_new_client", lambda **kwargs: Failing())
    [outcome] = await evaluate.evaluate_documents(
        make_docs(1), PROF, cache_dir=tmp_path, use_cache=False)

    assert outcome.error is not None
    assert "questions.concision.criteria" in outcome.error
    assert "too many levels" in outcome.error
    assert outcome.error.strip() not in {"422", "unprocessable", "Unprocessable Entity"}


# --- Fix round 1 -------------------------------------------------------
#
# Findings from the first review pass: the isolation net had holes outside
# the client-call try/except (on_start, on_done, response parsing, the cache
# write), so a raising callback or a malformed response could escape
# asyncio.gather and take the whole run down. And an earlier property test's
# docstring overclaimed what its assertion actually pinned.


async def test_cache_check_happens_before_the_semaphore_is_acquired(monkeypatch, tmp_path):
    """Cached documents must not queue behind a slow live one.

    With concurrency=1, a live document first in input order, and several
    already-cached documents behind it, every cached document's on_done must
    fire before the live document's does. If the cache check sat inside
    `async with semaphore:`, the live document would hold the sole slot for
    its whole delay and the cached documents would have to wait behind it,
    so their on_done calls would fire only after the live one's.
    """
    warm_client = FakeClient()
    monkeypatch.setattr(evaluate, "_new_client", lambda **kwargs: warm_client)
    cached_docs = make_docs(3)
    await evaluate.evaluate_documents(cached_docs, PROF, cache_dir=tmp_path)  # populate cache

    live_doc = sources.Document("live.md", "Live", "Body live.", "file", None)
    docs = [live_doc, *cached_docs]

    slow_client = FakeClient(delay=0.1)
    monkeypatch.setattr(evaluate, "_new_client", lambda **kwargs: slow_client)

    order = []
    await evaluate.evaluate_documents(
        docs, PROF, concurrency=1, cache_dir=tmp_path,
        on_done=lambda outcome: order.append(outcome.document.id),
    )

    live_position = order.index("live.md")
    cached_positions = [order.index(doc.id) for doc in cached_docs]
    assert live_position > max(cached_positions)


async def test_a_raising_on_start_does_not_abort_the_run(monkeypatch, tmp_path):
    """A progress callback that raises for one document must not lose the
    other documents' outcomes or crash the run."""
    client = FakeClient()
    monkeypatch.setattr(evaluate, "_new_client", lambda **kwargs: client)

    def flaky_on_start(document):
        if document.id == "doc1.md":
            raise RuntimeError("progress display broke")

    outcomes = await evaluate.evaluate_documents(
        make_docs(3), PROF, cache_dir=tmp_path, use_cache=False,
        on_start=flaky_on_start,
    )

    assert len(outcomes) == 3
    assert outcomes[0].error is None
    assert outcomes[0].answers["a"]["score"] == 1.0
    assert outcomes[2].error is None
    assert outcomes[2].answers["a"]["score"] == 1.0
    assert outcomes[1].error is not None
    assert "progress display broke" in outcomes[1].error


async def test_a_raising_on_done_does_not_abort_the_run(monkeypatch, tmp_path):
    """A progress callback that raises after a document finishes must not
    lose the other documents' outcomes, crash the run, or even overwrite the
    raising document's own already-computed answers -- the document was
    evaluated correctly; only the display of that fact broke."""
    client = FakeClient()
    monkeypatch.setattr(evaluate, "_new_client", lambda **kwargs: client)

    def flaky_on_done(outcome):
        if outcome.document.id == "doc1.md":
            raise RuntimeError("progress display broke")

    outcomes = await evaluate.evaluate_documents(
        make_docs(3), PROF, cache_dir=tmp_path, use_cache=False,
        on_done=flaky_on_done,
    )

    assert len(outcomes) == 3
    assert all(o.error is None for o in outcomes)
    assert outcomes[0].answers["a"]["score"] == 1.0
    assert outcomes[1].answers["a"]["score"] == 1.0
    assert outcomes[2].answers["a"]["score"] == 1.0
