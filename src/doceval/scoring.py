"""Composition and policy.

Raw judgments and policy stay separate. The model returns a distribution per
dimension; weights, thresholds, and bands are applied here, in code. Changing a
weight re-ranks a corpus from cache because nothing sent to the model moved.

This module is pure. Answers come in as plain dictionaries in the API's own
shape, which is also the shape the cache stores.
"""

from __future__ import annotations

from dataclasses import dataclass

from .lint import Violation
from .profile import Dimension, Profile
from .sources import Document

GOOD_THRESHOLD = 0.80
FAIR_THRESHOLD = 0.60
GATE_THRESHOLD = 0.5

VERDICT_NOT_PROSE = "NOT_PROSE"
VERDICT_UNSCORED = "UNSCORED"
VERDICT_ERROR = "ERROR"


@dataclass(frozen=True)
class DimensionResult:
    id: str
    label: str
    group: str
    weight: float
    raw: float
    normalized: float
    probabilities: dict[str, float]
    confidence: float
    needs_review: bool


@dataclass(frozen=True)
class GroupResult:
    name: str
    score: float | None
    dimensions: tuple[DimensionResult, ...]


@dataclass(frozen=True)
class DocumentResult:
    document: Document
    composite: float | None
    verdict: str
    groups: tuple[GroupResult, ...]
    violations: tuple[Violation, ...]
    gate_passed: bool
    model: str
    cached: bool
    error: str | None = None

    @property
    def errors(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity == "error")

    @property
    def warnings(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity == "warning")


def normalize(raw: float, level_count: int) -> float:
    """Map a raw Score onto 0 through 1."""
    if level_count < 2:
        return 0.0
    return max(0.0, min(1.0, raw / (level_count - 1)))


def verdict_for(composite: float) -> str:
    if composite >= GOOD_THRESHOLD:
        return "GOOD"
    if composite >= FAIR_THRESHOLD:
        return "FAIR"
    return "WEAK"


def score_document(
    *,
    document: Document,
    prof: Profile,
    answers: dict[str, dict],
    violations: tuple[Violation, ...] | list[Violation],
    min_confidence: float,
    model: str,
    cached: bool,
) -> DocumentResult:
    """Combine one document's answers and violations into a result."""
    gate_passed = _gate_passed(prof, answers)
    dimensions = tuple(
        _dimension_result(dimension, answers.get(dimension.id), min_confidence)
        for dimension in prof.dimensions
    )
    groups = _group_results(prof, dimensions)

    if not gate_passed:
        return DocumentResult(
            document=document, composite=None, verdict=VERDICT_NOT_PROSE,
            groups=groups, violations=tuple(violations), gate_passed=False,
            model=model, cached=cached,
        )

    composite = _weighted_mean(dimensions)
    verdict = VERDICT_UNSCORED if composite is None else verdict_for(composite)

    return DocumentResult(
        document=document, composite=composite, verdict=verdict, groups=groups,
        violations=tuple(violations), gate_passed=True, model=model, cached=cached,
    )


def error_result(document: Document, message: str) -> DocumentResult:
    """A document that never reached scoring. One failure never ends a run."""
    return DocumentResult(
        document=document, composite=None, verdict=VERDICT_ERROR, groups=(),
        violations=(), gate_passed=False, model="", cached=False, error=message,
    )


def corpus_group_averages(results: list[DocumentResult]) -> dict[str, float]:
    """Mean group score across every scored document."""
    totals: dict[str, list[float]] = {}
    for result in results:
        for group in result.groups:
            if group.score is not None:
                totals.setdefault(group.name, []).append(group.score)
    return {name: sum(values) / len(values) for name, values in totals.items()}


def weakest_dimensions(
    results: list[DocumentResult], limit: int = 2
) -> list[tuple[str, float]]:
    """Dimension labels with the lowest mean across the corpus."""
    totals: dict[str, list[float]] = {}
    for result in results:
        for group in result.groups:
            for dimension in group.dimensions:
                if not dimension.needs_review:
                    totals.setdefault(dimension.label, []).append(dimension.normalized)
    means = [(label, sum(v) / len(v)) for label, v in totals.items()]
    return sorted(means, key=lambda pair: pair[1])[:limit]


def _gate_passed(prof: Profile, answers: dict[str, dict]) -> bool:
    if prof.gate is None:
        return True
    answer = answers.get(prof.gate.id)
    if not answer:
        return True  # a missing gate answer never blocks a document
    return float(answer.get("noul", 1.0)) >= GATE_THRESHOLD


def _dimension_result(
    dimension: Dimension, answer: dict | None, min_confidence: float
) -> DimensionResult:
    if not answer:
        return DimensionResult(
            id=dimension.id, label=dimension.label, group=dimension.group,
            weight=dimension.weight, raw=0.0, normalized=0.0, probabilities={},
            confidence=0.0, needs_review=True,
        )

    raw = float(answer.get("score", 0.0))
    confidence = float(answer.get("confidence", 0.0))
    return DimensionResult(
        id=dimension.id,
        label=dimension.label,
        group=dimension.group,
        weight=dimension.weight,
        raw=raw,
        normalized=normalize(raw, len(dimension.levels)),
        probabilities=dict(answer.get("probabilities", {})),
        confidence=confidence,
        needs_review=confidence < min_confidence,
    )


def _group_results(prof: Profile, dimensions: tuple[DimensionResult, ...]) -> tuple[GroupResult, ...]:
    by_id = {d.id: d for d in dimensions}
    groups = []
    for name, members in prof.by_group().items():
        results = tuple(by_id[m.id] for m in members)
        groups.append(GroupResult(name=name, score=_weighted_mean(results), dimensions=results))
    return tuple(groups)


def _weighted_mean(dimensions: tuple[DimensionResult, ...]) -> float | None:
    """Weighted mean over dimensions passing the confidence gate.

    Excluded dimensions release their weight, and the rest rescale, so one
    uncertain judgment never silently drags a composite toward zero.
    """
    included = [d for d in dimensions if not d.needs_review]
    total_weight = sum(d.weight for d in included)
    if not included or total_weight <= 0:
        return None
    return sum(d.normalized * d.weight for d in included) / total_weight
