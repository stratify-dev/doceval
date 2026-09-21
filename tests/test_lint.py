import pytest

from doceval import lint


def rules(text):
    return [v.rule for v in lint.lint(text)]


def test_flags_em_dash():
    [v] = lint.lint("The results — and the cost — surprised us.")[:1]
    assert v.rule == "em_dash"
    assert v.severity == "error"
    assert v.line == 1


def test_flags_en_dash():
    assert "em_dash" in rules("A range – of values.")


def test_flags_semicolon():
    [v] = lint.lint("It runs fast; the cost is low.")
    assert v.rule == "semicolon"
    assert v.suggestion


def test_flags_asterisk():
    assert "asterisk" in rules("Use **bold** text.")


def test_flags_hashtag_mid_line():
    assert "hashtag" in rules("Ship it #buildinpublic")


def test_ignores_markdown_heading():
    assert rules("# Heading\n\nPlain prose here.\n") == []


def test_ignores_indented_heading():
    assert rules("  ## Heading\n") == []


def test_flags_banned_word_with_suggestion():
    [v] = lint.lint("We utilize the API.")
    assert v.rule == "banned_word"
    assert v.severity == "error"
    assert v.text == "utilize"
    assert v.suggestion == "use"


def test_banned_word_is_case_insensitive():
    [v] = lint.lint("Utilize the API.")
    assert v.text == "Utilize"


def test_banned_word_respects_word_boundaries():
    assert rules("The canonical form.") == []
    assert rules("Thatcher wrote it.") == []


def test_common_word_is_a_warning():
    [v] = lint.lint("This is the thing that matters.")
    assert v.rule == "common_word"
    assert v.severity == "warning"


def test_it_is_never_flagged():
    assert rules("It works. It is fine.") == []


def test_flags_setup_language():
    [v] = lint.lint("In conclusion, ship it.")
    assert v.rule == "setup_language"


def test_flags_not_just_construction():
    # "just" is also a common-word warning, so filter rather than unpack.
    found = [v for v in lint.lint("Not just faster, but also cheaper.") if v.rule == "not_just"]
    assert len(found) == 1
    assert found[0].severity == "error"


def test_reports_accurate_line_and_column():
    text = "First line.\nSecond line.\nThird — line.\n"
    [v] = [x for x in lint.lint(text) if x.rule == "em_dash"]
    assert v.line == 3
    assert v.column == 7


def test_ignores_fenced_code_blocks():
    text = "Prose here.\n\n```python\nx = 1; y = 2  # utilize\n```\n\nMore prose.\n"
    assert rules(text) == []


def test_ignores_inline_code_spans():
    assert rules("Run `a; b` and `utilize()` now.") == []


def test_finds_violations_after_a_code_block():
    text = "```\nx = 1;\n```\n\nWe utilize it.\n"
    [v] = lint.lint(text)
    assert v.rule == "banned_word"
    assert v.line == 5


def test_mask_code_preserves_offsets():
    text = "a `b` c"
    masked = lint.mask_code(text)
    assert len(masked) == len(text)
    assert masked == "a     c"


def test_mask_code_preserves_newlines():
    text = "a\n```\nx\n```\nb"
    masked = lint.mask_code(text)
    assert masked.count("\n") == text.count("\n")


def test_violations_are_sorted_by_position():
    text = "We utilize it; the results — matter.\n"
    positions = [(v.line, v.column) for v in lint.lint(text)]
    assert positions == sorted(positions)


def test_clean_prose_produces_nothing():
    assert lint.lint("Short sentences work. You read them once.\n") == []


@pytest.mark.parametrize(
    "word,suggestion",
    [
        ("delve", "dig, go into"),
        ("leverage", None),
        ("pivotal", "key, central"),
        ("tapestry", "delete"),
    ],
)
def test_banned_word_table(word, suggestion):
    if suggestion is None:
        assert word not in lint.BANNED_WORDS
    else:
        assert lint.BANNED_WORDS[word] == suggestion


def test_flags_previously_missing_banned_words():
    found = [
        v
        for v in lint.lint("Imagine the tapestry. Discover more. You are not alone.")
        if v.rule == "banned_word"
    ]
    assert len(found) == 4


def test_in_a_world_where_is_setup_language_not_banned_word():
    [v] = lint.lint("In a world where software ships daily, tests matter.")
    assert v.rule == "setup_language"


def test_bold_markdown_is_a_single_asterisk_violation():
    found = [v for v in lint.lint("Use **bold** text.") if v.rule == "asterisk"]
    assert len(found) == 1
    assert found[0].text == "**bold**"


def test_bullet_marker_is_a_single_asterisk_violation():
    found = [v for v in lint.lint("* item") if v.rule == "asterisk"]
    assert len(found) == 1


def test_lone_asterisk_is_a_single_violation():
    found = [v for v in lint.lint("Some text * more text.") if v.rule == "asterisk"]
    assert len(found) == 1


# --- Fix wave, finding 4 ---------------------------------------------------
#
# The hashtag rule matched any #word, so a markdown link's URL anchor
# (...#configuration) was flagged as a hashtag to remove, which fails CI on
# an ordinary documentation page under --fail-on-lint. mask_code covers
# fences and inline code, not URLs, so nothing upstream caught this either.


def test_ignores_hashtag_in_a_markdown_link_url():
    text = "See [the config guide](https://docs.example.com/guide#configuration)."
    assert "hashtag" not in rules(text)


def test_ignores_hashtag_at_the_end_of_a_bare_url():
    assert "hashtag" not in rules("Jump to https://example.com/page#top for details.")


def test_still_flags_a_genuine_mid_line_hashtag_after_the_url_fix():
    # Existing coverage (test_flags_hashtag_mid_line) must stay green; this
    # is the same property stated as a regression guard for this fix.
    assert "hashtag" in rules("Ship it #buildinpublic")


def test_still_ignores_a_heading_hashtag_after_the_url_fix():
    assert rules("# Heading\n\nPlain prose here.\n") == []


# The reviewer's second, lower-frequency instance of the same shape:
# &nbsp; and other HTML/XML character references trip the semicolon rule,
# since the semicolon that closes one is ordinary punctuation syntax to the
# regex, not prose punctuation.


def test_ignores_semicolon_closing_an_html_entity():
    assert "semicolon" not in rules("Line one&nbsp;line two.")


def test_ignores_semicolon_closing_a_named_html_entity_other_than_nbsp():
    assert "semicolon" not in rules("Cats&mdash;and dogs.")


def test_still_flags_a_real_semicolon_near_an_entity():
    found = [
        v for v in lint.lint("It runs fast&nbsp;now; the cost is low.") if v.rule == "semicolon"
    ]
    assert len(found) == 1
