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
    value = criteria["true"] if isinstance(criteria, dict) else getattr(criteria, "true")
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
