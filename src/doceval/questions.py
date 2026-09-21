"""Turning a Profile into one TypeSafe request.

Every question for a document rides in a single request. Jev ingests the state
once and evaluates all of them in parallel, which the parallel-questions
cookbook measures at roughly 12x cheaper and 10x faster than one request per
question, with the same answers.

The gate travels with the dimensions rather than preceding them. Asking it
speculatively costs one round trip instead of two, and code discards the
dimension answers when the gate comes back negative.

The audience lives in the state, not inside every question. Questions point at
it with a backticked path, so it is ingested once per request.
"""

from __future__ import annotations

import hashlib
import json

from typesafe_sdk import Noul, Score

from .profile import Profile
from .sources import Document


def build_questions(prof: Profile) -> dict[str, object]:
    """One Score per dimension, plus the gate Noul when the profile has one."""
    built: dict[str, object] = {
        dimension.id: Score(
            instructions=dimension.instructions,
            criteria=list(dimension.levels),
        )
        for dimension in prof.dimensions
    }

    if prof.gate is not None:
        built[prof.gate.id] = Noul(
            instructions=prof.gate.instructions,
            criteria=prof.gate.criteria or None,
        )

    return built


def build_state(document: Document, prof: Profile) -> dict[str, object]:
    """One document plus the audience it is judged against."""
    return {
        "document": {
            "path": document.id,
            "title": document.title,
            "text": document.text,
        },
        "audience": prof.audience,
    }


def questions_fingerprint(prof: Profile) -> str:
    """Hash of everything sent to the model, excluding weights.

    Weights are policy applied in code at render time, so changing one leaves
    cached answers valid. Editing instructions, levels, or the audience changes
    what the model sees, so the fingerprint moves and the cache misses.
    """
    payload = {
        "audience": prof.audience,
        "dimensions": [
            {
                "id": dimension.id,
                "instructions": dimension.instructions,
                "levels": list(dimension.levels),
            }
            for dimension in prof.dimensions
        ],
        "gate": None
        if prof.gate is None
        else {
            "id": prof.gate.id,
            "instructions": prof.gate.instructions,
            "criteria": prof.gate.criteria,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
