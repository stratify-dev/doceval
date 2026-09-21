import pytest

from doceval import lint, scoring, sources
from doceval import profile as profile_mod

DOC = sources.Document("post.md", "Post", "Body.", "file", None)

PROF = profile_mod.Profile(
    name="t",
    audience="Developers.",
    dimensions=(
        profile_mod.Dimension("active_voice", "house_style", 0.5, "q", ("a", "b", "c", "d", "e")),
        profile_mod.Dimension("concision", "editorial", 0.5, "q", ("a", "b", "c", "d", "e")),
    ),
    gate=profile_mod.Gate("is_prose", "Prose?", {}),
)


def score_answer(raw, confidence, probabilities=None):
    return {
        "type": "score",
        "score": raw,
        "confidence": confidence,
        "legend": {str(i): c for i, c in enumerate("abcde")},
        "probabilities": probabilities or {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 1.0},
    }


def build(answers, *, min_confidence=0.6, violations=()):
    return scoring.score_document(
        document=DOC,
        prof=PROF,
        answers=answers,
        violations=violations,
        min_confidence=min_confidence,
        model="jev-1.13.0",
        cached=False,
    )


def test_normalize_maps_top_level_to_one():
    assert scoring.normalize(4.0, 5) == 1.0


def test_normalize_maps_bottom_level_to_zero():
    assert scoring.normalize(0.0, 5) == 0.0


def test_normalize_handles_fractional_scores():
    assert scoring.normalize(2.0, 5) == 0.5


def test_composite_is_the_weighted_mean():
    result = build(
        {
            "active_voice": score_answer(4.0, 0.9),
            "concision": score_answer(2.0, 0.9),
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    assert result.composite == pytest.approx(0.75)
    assert result.verdict == "FAIR"


@pytest.mark.parametrize(
    "composite,expected",
    [
        (0.95, "GOOD"),
        (0.80, "GOOD"),
        (0.79, "FAIR"),
        (0.60, "FAIR"),
        (0.59, "WEAK"),
        (0.0, "WEAK"),
    ],
)
def test_verdict_bands(composite, expected):
    assert scoring.verdict_for(composite) == expected


def test_low_confidence_dimension_is_flagged_and_excluded():
    result = build(
        {
            "active_voice": score_answer(4.0, 0.9),
            "concision": score_answer(0.0, 0.3),
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    flagged = [d for g in result.groups for d in g.dimensions if d.needs_review]
    assert [d.id for d in flagged] == ["concision"]
    # remaining weight rescales, so the composite is active_voice alone
    assert result.composite == pytest.approx(1.0)


def test_all_dimensions_below_confidence_leaves_no_composite():
    result = build(
        {
            "active_voice": score_answer(4.0, 0.1),
            "concision": score_answer(2.0, 0.2),
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    assert result.composite is None
    assert result.verdict == "UNSCORED"


def test_failed_gate_skips_scoring():
    result = build(
        {
            "active_voice": score_answer(4.0, 0.9),
            "concision": score_answer(4.0, 0.9),
            "is_prose": {"type": "noul", "noul": 0.1},
        }
    )
    assert result.gate_passed is False
    assert result.verdict == "NOT_PROSE"
    assert result.composite is None


def test_gate_at_exactly_half_passes():
    result = build(
        {
            "active_voice": score_answer(4.0, 0.9),
            "concision": score_answer(4.0, 0.9),
            "is_prose": {"type": "noul", "noul": 0.5},
        }
    )
    assert result.gate_passed is True


def test_groups_carry_their_own_rollup():
    result = build(
        {
            "active_voice": score_answer(4.0, 0.9),
            "concision": score_answer(0.0, 0.9),
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    rollups = {g.name: g.score for g in result.groups}
    assert rollups["house_style"] == pytest.approx(1.0)
    assert rollups["editorial"] == pytest.approx(0.0)


def test_group_rollup_is_none_when_every_member_is_flagged():
    result = build(
        {
            "active_voice": score_answer(4.0, 0.9),
            "concision": score_answer(0.0, 0.1),
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    rollups = {g.name: g.score for g in result.groups}
    assert rollups["editorial"] is None


def test_probabilities_survive_onto_the_result():
    probs = {"0": 0.1, "1": 0.2, "2": 0.4, "3": 0.2, "4": 0.1}
    result = build(
        {
            "active_voice": score_answer(2.0, 0.9, probs),
            "concision": score_answer(2.0, 0.9),
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    dimension = result.groups[0].dimensions[0]
    assert dimension.probabilities == probs


def test_missing_answer_is_treated_as_needing_review():
    result = build(
        {
            "active_voice": score_answer(4.0, 0.9),
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    concision = [d for g in result.groups for d in g.dimensions if d.id == "concision"][0]
    assert concision.needs_review is True
    assert concision.confidence == 0.0


def test_violations_split_into_errors_and_warnings():
    violations = (
        lint.Violation("em_dash", "error", 1, 1, "—", "comma"),
        lint.Violation("common_word", "warning", 2, 1, "that", "remove"),
    )
    result = build(
        {
            "active_voice": score_answer(4.0, 0.9),
            "concision": score_answer(4.0, 0.9),
            "is_prose": {"type": "noul", "noul": 1.0},
        },
        violations=violations,
    )
    assert len(result.errors) == 1
    assert len(result.warnings) == 1


def test_lint_violations_do_not_change_the_composite():
    answers = {
        "active_voice": score_answer(4.0, 0.9),
        "concision": score_answer(4.0, 0.9),
        "is_prose": {"type": "noul", "noul": 1.0},
    }
    clean = build(answers)
    dirty = build(answers, violations=(lint.Violation("em_dash", "error", 1, 1, "—", "c"),))
    assert clean.composite == dirty.composite


def test_error_result_carries_the_message():
    result = scoring.error_result(DOC, "HTTP 404")
    assert result.verdict == "ERROR"
    assert result.error == "HTTP 404"
    assert result.composite is None


def test_corpus_group_averages():
    a = build(
        {
            "active_voice": score_answer(4.0, 0.9),
            "concision": score_answer(4.0, 0.9),
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    b = build(
        {
            "active_voice": score_answer(0.0, 0.9),
            "concision": score_answer(0.0, 0.9),
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    averages = scoring.corpus_group_averages([a, b])
    assert averages["house_style"] == pytest.approx(0.5)


def test_weakest_dimensions_ranks_ascending():
    a = build(
        {
            "active_voice": score_answer(4.0, 0.9),
            "concision": score_answer(1.0, 0.9),
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    weakest = scoring.weakest_dimensions([a], limit=1)
    assert weakest[0][0] == "concision"


# --- Property tests pinning the rescaling and exclusion arithmetic ---
#
# These go beyond the brief. They exist because the rescaling math is the
# kind of thing that can silently produce a plausible-looking wrong number
# if it regresses, with nothing else in the suite pinning it.


def test_property_rescaling_is_exact():
    """Two dimensions of weight 0.5 each, one flagged below confidence.

    Hand-computed: active_voice raw=3.0 over 5 levels normalizes to
    3/4 = 0.75. concision is excluded (confidence 0.2 < min_confidence
    0.6), so its weight leaves both the numerator and the denominator.
    What remains is active_voice's own weight over itself, which cancels:
    the composite is 0.75 exactly, not 0.75 scaled by 0.5 weight-share.
    """
    result = build(
        {
            "active_voice": score_answer(3.0, 0.9),
            "concision": score_answer(4.0, 0.2),
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    assert result.composite == pytest.approx(0.75)


def test_property_exclusion_never_drags_composite_down():
    """Excluding a low-scoring, low-confidence dimension must never score
    lower than including that same dimension would have. A document whose
    weak dimension gets flagged should outscore the identical document
    where that weak dimension was confident enough to count.
    """
    excluded = build(
        {
            "active_voice": score_answer(4.0, 0.9),
            "concision": score_answer(0.0, 0.2),  # below min_confidence: excluded
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    included = build(
        {
            "active_voice": score_answer(4.0, 0.9),
            "concision": score_answer(0.0, 0.9),  # confident: counted
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    assert excluded.composite == pytest.approx(1.0)
    assert included.composite == pytest.approx(0.5)
    assert excluded.composite > included.composite


def test_property_violations_cannot_move_composite():
    """Same answers, one call with violations and one without: identical
    composites, computed by hand rather than just compared to each other.

    active_voice raw=3.0 -> normalized 0.75, weight 0.5.
    concision raw=1.0 -> normalized 0.25, weight 0.5.
    composite = (0.75*0.5 + 0.25*0.5) / 1.0 = 0.5.
    """
    answers = {
        "active_voice": score_answer(3.0, 0.9),
        "concision": score_answer(1.0, 0.9),
        "is_prose": {"type": "noul", "noul": 1.0},
    }
    heavy_violations = (
        lint.Violation("em_dash", "error", 1, 1, "—", "comma"),
        lint.Violation("common_word", "warning", 2, 1, "that", "remove"),
        lint.Violation("semicolon", "error", 3, 1, ";", "period"),
    )
    clean = build(answers)
    dirty = build(answers, violations=heavy_violations)
    assert clean.composite == pytest.approx(0.5)
    assert clean.composite == pytest.approx(dirty.composite)


def test_property_missing_answer_produces_a_row_not_a_hole():
    """An absent dimension answer must still surface as a DimensionResult,
    flagged for review, rather than being silently dropped from the
    report's row count.
    """
    result = build(
        {
            "active_voice": score_answer(4.0, 0.9),
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    all_dimensions = [d for g in result.groups for d in g.dimensions]
    assert len(all_dimensions) == len(PROF.dimensions)
    concision = next(d for d in all_dimensions if d.id == "concision")
    assert concision.needs_review is True
    assert concision.confidence == 0.0
    assert concision.raw == 0.0
    assert concision.normalized == 0.0
    assert concision.probabilities == {}


# --- Fix wave: a malformed answer payload must degrade, never raise -------
#
# Whole-branch review finding 1: score_document ran outside evaluate.py's
# own per-document isolation, and float(answer.get(key, default)) never
# falls back for a key present with value null -- only for a missing key.
# A null score, a null confidence, a null gate noul, a non-numeric score, or
# null probabilities each used to raise (TypeError or ValueError) and take
# the whole corpus down with them.


def test_null_score_is_treated_like_a_missing_answer():
    result = build(
        {
            "active_voice": {**score_answer(4.0, 0.9), "score": None},
            "concision": score_answer(4.0, 0.9),
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    dims = {d.id: d for g in result.groups for d in g.dimensions}
    assert dims["active_voice"].needs_review is True
    assert dims["active_voice"].confidence == 0.0
    assert dims["active_voice"].raw == 0.0
    assert dims["active_voice"].probabilities == {}
    # the other, well-formed dimension still scores normally
    assert result.composite == pytest.approx(1.0)


def test_non_numeric_score_is_treated_like_a_missing_answer():
    result = build(
        {
            "active_voice": {**score_answer(4.0, 0.9), "score": "not-a-number"},
            "concision": score_answer(4.0, 0.9),
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    dims = {d.id: d for g in result.groups for d in g.dimensions}
    assert dims["active_voice"].needs_review is True


def test_null_confidence_falls_back_to_zero_rather_than_raising():
    result = build(
        {
            "active_voice": {**score_answer(4.0, 0.9), "confidence": None},
            "concision": score_answer(4.0, 0.9),
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    dims = {d.id: d for g in result.groups for d in g.dimensions}
    assert dims["active_voice"].confidence == 0.0
    assert dims["active_voice"].needs_review is True


def test_null_gate_noul_defaults_to_passing_rather_than_raising():
    result = build(
        {
            "active_voice": score_answer(4.0, 0.9),
            "concision": score_answer(4.0, 0.9),
            "is_prose": {"type": "noul", "noul": None},
        }
    )
    assert result.gate_passed is True


def test_null_probabilities_does_not_raise():
    result = build(
        {
            "active_voice": {**score_answer(4.0, 0.9), "probabilities": None},
            "concision": score_answer(4.0, 0.9),
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    dims = {d.id: d for g in result.groups for d in g.dimensions}
    assert dims["active_voice"].probabilities == {}
    # a null probabilities field doesn't imply an unreadable score
    assert dims["active_voice"].needs_review is False


def test_property_unscored_is_distinct_from_a_zero_composite():
    """UNSCORED (composite is None) means every judgment was too uncertain
    to combine. A WEAK verdict with composite == 0.0 means the model was
    confident the document sits at the floor. These are different claims
    and must never collapse into each other.
    """
    unscored = build(
        {
            "active_voice": score_answer(4.0, 0.1),
            "concision": score_answer(2.0, 0.2),
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    assert unscored.composite is None
    assert unscored.verdict == "UNSCORED"

    zeroed = build(
        {
            "active_voice": score_answer(0.0, 0.9),
            "concision": score_answer(0.0, 0.9),
            "is_prose": {"type": "noul", "noul": 1.0},
        }
    )
    assert zeroed.composite == pytest.approx(0.0)
    assert zeroed.verdict == "WEAK"
