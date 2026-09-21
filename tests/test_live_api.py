"""One live request against the real API.

Skipped unless DOCEVAL_LIVE_API=1. Mocks cannot catch API drift; this can.
"""

import os

import pytest

from doceval import evaluate, sources
from doceval import profile as profile_mod

# Two independent gates, deliberately separate:
#   skipif  decides WHETHER this test runs at all
#   mark.live decides WHETHER it may touch the network (conftest blocks it otherwise)
pytestmark = [
    pytest.mark.skipif(
        os.environ.get("DOCEVAL_LIVE_API") != "1",
        reason="set DOCEVAL_LIVE_API=1 to run against the real API",
    ),
    pytest.mark.live,
]


async def test_live_request_returns_the_documented_shape():
    prof = profile_mod.load_profile("house-style")
    document = sources.Document(
        id="live-test",
        title="Short Test",
        text=(
            "You read this once and you know what to do. Short sentences work. "
            "The linter finds the rest. Ship the change today."
        ),
        origin="file",
        fetched_at=None,
    )

    [outcome] = await evaluate.evaluate_documents([document], prof, use_cache=False)

    assert outcome.error is None, outcome.error
    assert outcome.model.startswith("jev-")
    assert set(outcome.answers) == {d.id for d in prof.dimensions} | {prof.gate.id}

    for dimension in prof.dimensions:
        answer = outcome.answers[dimension.id]
        assert answer["type"] == "score"
        assert 0.0 <= answer["confidence"] <= 1.0
        assert len(answer["probabilities"]) == len(dimension.levels)
        assert abs(sum(answer["probabilities"].values()) - 1.0) < 0.01

    gate = outcome.answers[prof.gate.id]
    assert gate["type"] == "noul"
    assert 0.0 <= gate["noul"] <= 1.0
    assert outcome.usage["input_tokens"] > 0
