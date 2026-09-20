import asyncio

import pytest

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
    assert client.peak <= 2


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


def test_answers_to_dicts_reads_the_raw_response():
    assert evaluate.answers_to_dicts(FakeResponse(ANSWERS)) == ANSWERS


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
    assert evaluate.RETRY.max_retries == 3
    assert evaluate.RETRY.respect_retry_after is True
    assert 429 in evaluate.RETRY.http_statuses
    assert 529 in evaluate.RETRY.http_statuses


# --- Properties beyond the brief -------------------------------------------
#
# Each of these pins a failure mode that would be silent: the run would still
# "complete", just with lost data or a corrupted result, and none of the
# tests above would catch it.


async def test_cache_hits_do_not_consume_a_concurrency_slot(monkeypatch, tmp_path):
    """A fully cached corpus makes zero client calls, even with concurrency=1.

    If the cache check were moved inside the semaphore, hits would still all
    resolve (cache reads are fast and uncontended), so "all documents
    resolve" alone would not catch that regression. Asserting the client was
    never touched does: it only holds when the cache check happens before a
    slot is ever requested.
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
