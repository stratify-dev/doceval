import pytest

from doceval import profile as mod

VALID = {
    "name": "t",
    "audience": "Working developers.",
    "dimensions": {
        "a": {
            "group": "house_style",
            "weight": 0.6,
            "type": "score",
            "instructions": "How active?",
            "levels": ["bad", "ok", "good"],
        },
        "b": {
            "group": "editorial",
            "weight": 0.4,
            "type": "score",
            "instructions": "How clear?",
            "levels": ["bad", "good"],
        },
    },
}


def test_parses_valid_profile():
    p = mod.parse_profile(VALID, "test")
    assert p.name == "t"
    assert len(p.dimensions) == 2
    assert p.dimensions[0].id == "a"
    assert p.dimensions[0].levels == ("bad", "ok", "good")
    assert p.gate is None


def test_label_humanizes_the_id():
    p = mod.parse_profile(VALID, "test")
    assert p.dimensions[0].label == "a"
    d = mod.Dimension("active_voice", "house_style", 1.0, "x", ("a", "b"))
    assert d.label == "active voice"


def test_by_group_preserves_declaration_order():
    p = mod.parse_profile(VALID, "test")
    grouped = p.by_group()
    assert list(grouped) == ["house_style", "editorial"]
    assert grouped["house_style"][0].id == "a"


def test_parses_gate():
    data = {
        **VALID,
        "gate": {
            "is_prose": {
                "type": "noul",
                "instructions": "Finished prose?",
                "criteria": {"true": "yes it is", "false": "no it is not"},
            }
        },
    }
    p = mod.parse_profile(data, "test")
    assert p.gate is not None
    assert p.gate.id == "is_prose"
    assert p.gate.criteria["true"] == "yes it is"


def test_rejects_weights_not_summing_to_one():
    data = {**VALID, "dimensions": {**VALID["dimensions"]}}
    data["dimensions"] = {
        "a": {**VALID["dimensions"]["a"], "weight": 0.5},
        "b": {**VALID["dimensions"]["b"], "weight": 0.2},
    }
    with pytest.raises(mod.ProfileError) as exc:
        mod.parse_profile(data, "test")
    assert "0.7" in str(exc.value)


def test_accepts_weights_within_tolerance():
    data = {
        "name": "t",
        "audience": "x",
        "dimensions": {
            "a": {
                "group": "house_style",
                "weight": 0.3333,
                "type": "score",
                "instructions": "q",
                "levels": ["a", "b"],
            },
            "b": {
                "group": "editorial",
                "weight": 0.3333,
                "type": "score",
                "instructions": "q",
                "levels": ["a", "b"],
            },
            "c": {
                "group": "audience_fit",
                "weight": 0.3334,
                "type": "score",
                "instructions": "q",
                "levels": ["a", "b"],
            },
        },
    }
    assert len(mod.parse_profile(data, "test").dimensions) == 3


def test_rejects_unknown_group():
    data = {
        "name": "t",
        "audience": "x",
        "dimensions": {
            "a": {
                "group": "nonsense",
                "weight": 1.0,
                "type": "score",
                "instructions": "q",
                "levels": ["a", "b"],
            },
        },
    }
    with pytest.raises(mod.ProfileError, match="nonsense"):
        mod.parse_profile(data, "test")


def test_rejects_single_level_score():
    data = {
        "name": "t",
        "audience": "x",
        "dimensions": {
            "a": {
                "group": "house_style",
                "weight": 1.0,
                "type": "score",
                "instructions": "q",
                "levels": ["only"],
            },
        },
    }
    with pytest.raises(mod.ProfileError, match="at least 2"):
        mod.parse_profile(data, "test")


def test_rejects_more_than_ten_levels():
    data = {
        "name": "t",
        "audience": "x",
        "dimensions": {
            "a": {
                "group": "house_style",
                "weight": 1.0,
                "type": "score",
                "instructions": "q",
                "levels": [str(i) for i in range(11)],
            },
        },
    }
    with pytest.raises(mod.ProfileError, match="at most 10"):
        mod.parse_profile(data, "test")


def test_rejects_missing_instructions():
    data = {
        "name": "t",
        "audience": "x",
        "dimensions": {
            "a": {"group": "house_style", "weight": 1.0, "type": "score", "levels": ["a", "b"]},
        },
    }
    with pytest.raises(mod.ProfileError, match="instructions"):
        mod.parse_profile(data, "test")


def test_rejects_missing_audience():
    data = {"name": "t", "dimensions": VALID["dimensions"]}
    with pytest.raises(mod.ProfileError, match="audience"):
        mod.parse_profile(data, "test")


def test_rejects_empty_dimensions():
    with pytest.raises(mod.ProfileError, match="at least one dimension"):
        mod.parse_profile({"name": "t", "audience": "x", "dimensions": {}}, "test")


def test_rejects_duplicate_dimension_id(tmp_path):
    path = tmp_path / "dup.yaml"
    path.write_text(
        "name: t\naudience: x\ndimensions:\n"
        "  a:\n    group: house_style\n    weight: 0.5\n    type: score\n"
        "    instructions: q\n    levels: [a, b]\n"
        "  a:\n    group: editorial\n    weight: 0.5\n    type: score\n"
        "    instructions: q\n    levels: [a, b]\n"
    )
    with pytest.raises(mod.ProfileError, match="duplicate key"):
        mod.load_profile(path)


def test_rejects_unquoted_boolean_gate_criteria_keys(tmp_path):
    path = tmp_path / "badgate.yaml"
    path.write_text(
        "name: t\naudience: x\n"
        "gate:\n  is_finished_prose:\n    type: noul\n    instructions: q\n"
        "    criteria:\n      true: yes it is\n      false: no it is not\n"
        "dimensions:\n  a:\n    group: house_style\n    weight: 1.0\n    type: score\n"
        "    instructions: q\n    levels: [a, b]\n"
    )
    with pytest.raises(mod.ProfileError, match="quoted"):
        mod.load_profile(path)


# --- Fix wave, finding 2 --------------------------------------------------
#
# The old check only verified criteria keys were strings, not that they
# were "true"/"false" -- the SDK's NoulCriteria is a closed TypedDict
# accepting only those two keys. A "yes"/"no" gate used to pass profile
# validation here and then blow up as a raw pydantic traceback deep inside
# evaluate.py's build_questions -> Noul(criteria=...), outside every
# isolation net, and outside the documented "surfaces before the first
# request, not as a 422" contract.


def test_rejects_gate_criteria_keys_outside_true_false(tmp_path):
    path = tmp_path / "badkeys.yaml"
    path.write_text(
        "name: t\naudience: x\n"
        "gate:\n  is_finished_prose:\n    type: noul\n    instructions: q\n"
        '    criteria:\n      "yes": it is\n      "no": it is not\n'
        "dimensions:\n  a:\n    group: house_style\n    weight: 1.0\n    type: score\n"
        "    instructions: q\n    levels: [a, b]\n"
    )
    with pytest.raises(mod.ProfileError) as exc:
        mod.load_profile(path)
    message = str(exc.value)
    assert "not allowed" in message
    assert "yes" in message  # names the offending key
    assert '"true"' in message and '"false"' in message  # names the allowed set


def test_unquoted_boolean_gate_criteria_still_gets_its_own_quoting_message(tmp_path):
    """Control for the finding-2 fix: the pre-existing bool-key rejection
    (an unquoted `true:` parsing as a Python bool, not a string) must keep
    its own specific quoting guidance rather than falling into the new
    "not allowed" message, since that's still the most likely real mistake.
    """
    path = tmp_path / "badgate.yaml"
    path.write_text(
        "name: t\naudience: x\n"
        "gate:\n  is_finished_prose:\n    type: noul\n    instructions: q\n"
        "    criteria:\n      true: yes it is\n      false: no it is not\n"
        "dimensions:\n  a:\n    group: house_style\n    weight: 1.0\n    type: score\n"
        "    instructions: q\n    levels: [a, b]\n"
    )
    with pytest.raises(mod.ProfileError, match="quoted"):
        mod.load_profile(path)


def test_loads_bundled_profile_by_name():
    p = mod.load_profile("house-style")
    assert p.name == "house-style"
    assert len(p.dimensions) == 10


def test_bundled_profile_weights_sum_to_one():
    p = mod.load_profile("house-style")
    assert abs(sum(d.weight for d in p.dimensions) - 1.0) < mod.WEIGHT_TOLERANCE


def test_bundled_profile_has_all_three_groups():
    p = mod.load_profile("house-style")
    assert set(p.by_group()) == set(mod.GROUPS)


def test_bundled_profile_has_a_gate():
    p = mod.load_profile("house-style")
    assert p.gate is not None
    assert p.gate.id == "is_finished_prose"


def test_bundled_profiles_lists_house_style():
    assert "house-style" in mod.bundled_profiles()


def test_unknown_profile_name_raises():
    with pytest.raises(mod.ProfileError, match="nonexistent"):
        mod.load_profile("nonexistent")
