import dataclasses

from doceval import profile as profile_mod
from doceval import questions, sources

PROF = profile_mod.Profile(
    name="t",
    audience="Working developers.",
    dimensions=(
        profile_mod.Dimension("active_voice", "house_style", 0.5, "How active?", ("bad", "ok", "good")),
        profile_mod.Dimension("concision", "editorial", 0.5, "How tight?", ("loose", "tight")),
    ),
    gate=profile_mod.Gate("is_prose", "Finished prose?", {"true": "yes", "false": "no"}),
)

DOC = sources.Document("post.md", "My Post", "Body text.", "file", None)


def test_builds_one_question_per_dimension_plus_gate():
    built = questions.build_questions(PROF)
    assert set(built) == {"active_voice", "concision", "is_prose"}


def test_score_question_carries_instructions_and_levels():
    built = questions.build_questions(PROF)
    assert built["active_voice"].instructions == "How active?"
    assert list(built["active_voice"].criteria) == ["bad", "ok", "good"]


def test_gate_question_carries_criteria():
    criteria = questions.build_questions(PROF)["is_prose"].criteria
    # The SDK may keep a plain dict or wrap it in a model; assert the value.
    value = criteria["true"] if isinstance(criteria, dict) else criteria.true
    assert value == "yes"


def test_profile_without_gate_builds_only_dimensions():
    bare = profile_mod.Profile("t", "a", PROF.dimensions, None)
    assert set(questions.build_questions(bare)) == {"active_voice", "concision"}


def test_state_carries_document_and_audience():
    state = questions.build_state(DOC, PROF)
    assert state["document"]["path"] == "post.md"
    assert state["document"]["title"] == "My Post"
    assert state["document"]["text"] == "Body text."
    assert state["audience"] == "Working developers."


def test_fingerprint_is_stable_across_calls():
    assert questions.questions_fingerprint(PROF) == questions.questions_fingerprint(PROF)


def test_fingerprint_changes_when_instructions_change():
    edited = profile_mod.Profile(
        "t", PROF.audience,
        (profile_mod.Dimension("active_voice", "house_style", 0.5, "DIFFERENT", ("bad", "ok", "good")),
         PROF.dimensions[1]),
        PROF.gate,
    )
    assert questions.questions_fingerprint(edited) != questions.questions_fingerprint(PROF)


def test_fingerprint_ignores_weight_changes():
    reweighted = profile_mod.Profile(
        "t", PROF.audience,
        (profile_mod.Dimension("active_voice", "house_style", 0.9, "How active?", ("bad", "ok", "good")),
         profile_mod.Dimension("concision", "editorial", 0.1, "How tight?", ("loose", "tight"))),
        PROF.gate,
    )
    assert questions.questions_fingerprint(reweighted) == questions.questions_fingerprint(PROF)


def test_fingerprint_changes_when_audience_changes():
    other = profile_mod.Profile("t", "Retired sailors.", PROF.dimensions, PROF.gate)
    assert questions.questions_fingerprint(other) != questions.questions_fingerprint(PROF)


def test_fingerprint_changes_when_a_level_text_changes():
    edited_dimension = dataclasses.replace(
        PROF.dimensions[0], levels=("DIFFERENT", "ok", "good")
    )
    edited = dataclasses.replace(PROF, dimensions=(edited_dimension, PROF.dimensions[1]))
    assert questions.questions_fingerprint(edited) != questions.questions_fingerprint(PROF)


def test_fingerprint_changes_when_levels_are_reordered():
    """Level order defines the scale, not just which strings appear.

    ("bad", "ok", "good") and ("good", "ok", "bad") are the same three
    strings, but their order is what makes a high score mean something
    different from a low one. Reversing it inverts the meaning of every
    answer the model gives against this dimension. A fingerprint that
    sorted levels before hashing, for determinism, would miss this change
    and keep serving cached answers scored against the inverted scale.
    """
    reordered_dimension = dataclasses.replace(
        PROF.dimensions[0], levels=tuple(reversed(PROF.dimensions[0].levels))
    )
    reordered = dataclasses.replace(PROF, dimensions=(reordered_dimension, PROF.dimensions[1]))
    assert questions.questions_fingerprint(reordered) != questions.questions_fingerprint(PROF)


def test_fingerprint_changes_when_gate_instructions_change():
    edited_gate = dataclasses.replace(PROF.gate, instructions="DIFFERENT")
    edited = dataclasses.replace(PROF, gate=edited_gate)
    assert questions.questions_fingerprint(edited) != questions.questions_fingerprint(PROF)


def test_fingerprint_changes_when_gate_criteria_change():
    edited_gate = dataclasses.replace(PROF.gate, criteria={"true": "DIFFERENT", "false": "no"})
    edited = dataclasses.replace(PROF, gate=edited_gate)
    assert questions.questions_fingerprint(edited) != questions.questions_fingerprint(PROF)


def test_fingerprint_changes_when_a_dimension_is_removed():
    edited = dataclasses.replace(PROF, dimensions=(PROF.dimensions[0],))
    assert questions.questions_fingerprint(edited) != questions.questions_fingerprint(PROF)


def test_fingerprint_changes_when_dimensions_are_reordered():
    reordered = dataclasses.replace(PROF, dimensions=(PROF.dimensions[1], PROF.dimensions[0]))
    assert questions.questions_fingerprint(reordered) != questions.questions_fingerprint(PROF)


def test_fingerprint_ignores_profile_name_change():
    renamed = dataclasses.replace(PROF, name="different-name")
    assert questions.questions_fingerprint(renamed) == questions.questions_fingerprint(PROF)


def test_fingerprint_ignores_dimension_group_change():
    regrouped_dimension = dataclasses.replace(PROF.dimensions[0], group="editorial")
    regrouped = dataclasses.replace(PROF, dimensions=(regrouped_dimension, PROF.dimensions[1]))
    assert questions.questions_fingerprint(regrouped) == questions.questions_fingerprint(PROF)


def test_fingerprint_pins_the_shipped_house_style_digest():
    """A hard pin, not just a stability check against itself.

    If a future change to questions_fingerprint alters what gets hashed or
    how, this fails loudly instead of silently invalidating every cached
    answer in the field. A human then decides whether that invalidation was
    intended, rather than it happening as a side effect of an unrelated
    change.
    """
    house_style = profile_mod.load_profile("house-style")
    assert (
        questions.questions_fingerprint(house_style)
        == "6d1f45202b325933097d393374ae0266d13fe0215e27f26287fa17969f0d08e7"
    )
