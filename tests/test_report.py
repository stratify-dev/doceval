import io
import json

import pytest
from rich.console import Console

from doceval import lint, report, scoring, sources
from doceval import profile as profile_mod

DOC = sources.Document("post.md", "Post", "Body.", "file", None)

PROF = profile_mod.Profile(
    name="house-style",
    audience="Developers.",
    dimensions=(
        profile_mod.Dimension("active_voice", "house_style", 0.5, "q", ("a", "b", "c", "d", "e")),
        profile_mod.Dimension("concision", "editorial", 0.5, "q", ("a", "b", "c", "d", "e")),
    ),
    gate=None,
)


def make_result(**overrides):
    dims = {
        "active_voice": scoring.DimensionResult(
            "active_voice",
            "active voice",
            "house_style",
            0.5,
            4.0,
            1.0,
            {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 1.0},
            0.95,
            False,
        ),
        "concision": scoring.DimensionResult(
            "concision",
            "concision",
            "editorial",
            0.5,
            2.0,
            0.5,
            {"0": 0.1, "1": 0.2, "2": 0.4, "3": 0.2, "4": 0.1},
            0.44,
            True,
        ),
    }
    base = dict(
        document=DOC,
        composite=1.0,
        verdict="GOOD",
        groups=(
            scoring.GroupResult("house_style", 1.0, (dims["active_voice"],)),
            scoring.GroupResult("editorial", None, (dims["concision"],)),
        ),
        violations=(lint.Violation("em_dash", "error", 14, 7, "—", "comma"),),
        gate_passed=True,
        model="jev-1.13.0",
        cached=False,
        error=None,
    )
    return scoring.DocumentResult(**{**base, **overrides})


def render(fn, *args, width=100, **kwargs):
    console = Console(width=width, no_color=True, force_terminal=False, record=True)
    fn(console, *args, **kwargs)
    return console.export_text()


def test_sparkline_has_one_block_per_level():
    assert len(report.sparkline({"0": 0.0, "1": 0.5, "2": 1.0})) == 3


def test_sparkline_maps_certainty_to_the_tallest_block():
    assert report.sparkline({"0": 0.0, "1": 1.0})[1] == report.BLOCKS[-1]


def test_sparkline_maps_zero_to_the_shortest_block():
    assert report.sparkline({"0": 0.0, "1": 1.0})[0] == report.BLOCKS[0]


def test_sparkline_orders_levels_numerically():
    line = report.sparkline({"10": 1.0, "2": 0.0, "1": 0.0})
    assert line[-1] == report.BLOCKS[-1]


def test_sparkline_handles_empty_probabilities():
    assert report.sparkline({}) == ""


def test_bar_is_full_at_one():
    assert report.bar(1.0, width=10) == "█" * 10


def test_bar_is_empty_at_zero():
    assert report.bar(0.0, width=10) == "░" * 10


def test_bar_handles_none():
    assert report.bar(None, width=10) == "░" * 10


def test_bar_keeps_a_constant_width():
    assert all(len(report.bar(v / 10, width=20)) == 20 for v in range(11))


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.9, "green"),
        (0.7, "yellow"),
        (0.3, "red"),
        (None, "dim"),
    ],
)
def test_style_bands(value, expected):
    assert report.style_for(value) == expected


def test_tree_shows_document_verdict_and_composite():
    out = render(report.render_documents, [make_result()])
    assert "post.md" in out
    assert "GOOD" in out
    assert "1.00" in out


def test_tree_shows_every_group_and_dimension():
    out = render(report.render_documents, [make_result()])
    assert "house style" in out
    assert "editorial" in out
    assert "active voice" in out
    assert "concision" in out


def test_tree_marks_low_confidence_for_review():
    out = render(report.render_documents, [make_result()])
    assert "review" in out


def test_tree_lists_lint_violations_with_positions():
    out = render(report.render_documents, [make_result()])
    assert "line 14" in out
    assert "em dash" in out or "em_dash" in out
    assert "comma" in out


def test_compact_hides_dimension_rows():
    out = render(report.render_documents, [make_result()], compact=True)
    assert "house style" in out
    assert "active voice" not in out


def test_tree_renders_an_errored_document():
    result = scoring.error_result(DOC, "HTTP 404")
    out = render(report.render_documents, [result])
    assert "ERROR" in out
    assert "HTTP 404" in out


def test_tree_renders_a_gate_failure():
    result = make_result(verdict="NOT_PROSE", composite=None, gate_passed=False)
    out = render(report.render_documents, [result])
    assert "NOT_PROSE" in out


def test_corpus_lists_documents_and_group_averages():
    out = render(
        report.render_corpus,
        [make_result()],
        PROF,
        {"input_tokens": 14_200, "output_tokens": 300},
        2.1,
        "jev-1.13.0",
    )
    assert "post.md" in out
    assert "house style" in out
    assert "jev-1.13.0" in out


def test_corpus_reports_the_cached_count():
    out = render(
        report.render_corpus,
        [make_result(cached=True)],
        PROF,
        {"input_tokens": 0, "output_tokens": 0},
        0.2,
        "jev-1.13.0",
    )
    assert "1 cached" in out


def test_json_round_trips():
    payload = json.loads(report.to_json([make_result()], PROF))
    assert payload["profile"] == "house-style"
    document = payload["documents"][0]
    assert document["id"] == "post.md"
    assert document["composite"] == 1.0
    assert document["groups"][0]["dimensions"][0]["id"] == "active_voice"
    assert document["lint"]["errors"][0]["line"] == 14


def test_json_marks_dimensions_needing_review():
    payload = json.loads(report.to_json([make_result()], PROF))
    flagged = payload["documents"][0]["groups"][1]["dimensions"][0]
    assert flagged["needs_review"] is True


def test_json_includes_summary():
    payload = json.loads(report.to_json([make_result()], PROF))
    assert "group_averages" in payload["summary"]


def test_json_handles_an_errored_document():
    payload = json.loads(report.to_json([scoring.error_result(DOC, "boom")], PROF))
    assert payload["documents"][0]["error"] == "boom"


def test_markdown_has_a_table_header():
    out = report.to_markdown([make_result()], PROF)
    assert "| document |" in out
    assert "post.md" in out


# --- Property tests: sparkline height mapping is exact and stable -----------
#
# A silent mis-mapping (e.g. reversed order, or an off-by-one landing the
# tallest block on the wrong level) would still "return a string" and pass a
# looser test. These pin the actual characters for a known distribution.


def test_sparkline_exact_characters_for_bimodal_distribution():
    assert report.sparkline({"0": 0.0, "1": 1.0}) == report.BLOCKS[0] + report.BLOCKS[-1]


def test_sparkline_exact_characters_are_stable_across_calls():
    first = report.sparkline({"0": 0.0, "1": 1.0})
    second = report.sparkline({"0": 0.0, "1": 1.0})
    assert first == second == "▁█"  # ▁ then █


# --- Property tests: bar width is constant, including None ------------------


def test_bar_width_is_constant_across_full_range_including_none():
    values = [v / 10 for v in range(11)] + [None]
    assert all(len(report.bar(v, width=20)) == 20 for v in values)


# --- Property tests: every degenerate DocumentResult renders ----------------


def test_tree_renders_an_unscored_document():
    result = make_result(verdict="UNSCORED", composite=None, gate_passed=True)
    out = render(report.render_documents, [result])
    assert "UNSCORED" in out


def test_tree_renders_a_dimension_with_empty_probabilities():
    dim = scoring.DimensionResult(
        "active_voice", "active voice", "house_style", 0.5, 0.0, 0.0, {}, 0.0, True
    )
    result = make_result(groups=(scoring.GroupResult("house_style", None, (dim,)),))
    out = render(report.render_documents, [result])
    assert "active voice" in out


def test_tree_renders_all_four_degenerate_documents_without_raising():
    empty_prob_dim = scoring.DimensionResult(
        "active_voice", "active voice", "house_style", 0.5, 0.0, 0.0, {}, 0.0, True
    )
    results = [
        scoring.error_result(DOC, "boom"),
        make_result(verdict="NOT_PROSE", composite=None, gate_passed=False),
        make_result(verdict="UNSCORED", composite=None, gate_passed=True),
        make_result(groups=(scoring.GroupResult("house_style", None, (empty_prob_dim,)),)),
    ]
    out = render(report.render_documents, results)
    assert "ERROR" in out
    assert "NOT_PROSE" in out
    assert "UNSCORED" in out
    assert "active voice" in out


# --- Property tests: JSON round-trips for a mixed corpus --------------------


def test_json_round_trips_a_mixed_corpus():
    results = [
        make_result(),
        make_result(verdict="NOT_PROSE", composite=None, gate_passed=False),
        scoring.error_result(DOC, "boom"),
    ]
    payload = json.loads(report.to_json(results, PROF))
    assert len(payload["documents"]) == 3
    assert payload["documents"][0]["verdict"] == "GOOD"
    assert payload["documents"][1]["verdict"] == "NOT_PROSE"
    assert payload["documents"][2]["error"] == "boom"


# --- Property tests: piped output carries no ANSI escapes -------------------


def test_piped_output_has_no_ansi_escapes():
    buffer = io.StringIO()
    console = Console(file=buffer, width=100)
    report.render_documents(console, [make_result()])
    assert "\x1b[" not in buffer.getvalue()


def test_piped_corpus_output_has_no_ansi_escapes():
    buffer = io.StringIO()
    console = Console(file=buffer, width=100)
    report.render_corpus(
        console, [make_result()], PROF, {"input_tokens": 0, "output_tokens": 0}, 0.1, "jev-1.13.0"
    )
    assert "\x1b[" not in buffer.getvalue()


# --- Requirement A: violation text is whitespace-collapsed and truncated ----
#
# Task 3 widened the asterisk rule from a single character to a span, so
# Violation.text is now unbounded (an entire italic paragraph can be one
# violation). Code inside a span is masked to spaces of equal length by
# lint.mask_code, so a span crossing inline code carries a run of blanks.
# Both must be cleaned before the report's fixed-width columns touch them.


def test_clean_violation_text_collapses_blanked_code_whitespace():
    text = "*span across        here*"
    assert report._clean_violation_text(text) == "*span across here*"


def test_clean_violation_text_truncates_long_spans():
    cleaned = report._clean_violation_text("x" * 500)
    assert len(cleaned) <= report.VIOLATION_TEXT_WIDTH
    assert cleaned.endswith("…")  # ellipsis


def test_clean_violation_text_leaves_short_text_untouched():
    assert report._clean_violation_text("—") == "—"


def test_render_documents_survives_a_500_char_violation_span():
    violation = lint.Violation("asterisk", "error", 3, 1, "x" * 500, "rewrite for emphasis")
    result = make_result(violations=(violation,))
    out = render(report.render_documents, [result])
    assert "x" * 100 not in out


def test_render_documents_shows_blanked_code_span_without_a_run_of_spaces():
    text = "*span across        here*"
    violation = lint.Violation("asterisk", "error", 1, 1, text, "rewrite for emphasis")
    result = make_result(violations=(violation,))
    out = render(report.render_documents, [result])
    assert '"*span across here*"' in out
    detail_line = next(line for line in out.splitlines() if "asterisk" in line)
    assert "across  " not in detail_line  # no leftover run from the masked code


def _render_lines(result, width=80):
    buffer = io.StringIO()
    console = Console(file=buffer, width=width)
    report.render_documents(console, [result])
    return buffer.getvalue().splitlines()


def test_violation_detail_rows_align_and_never_wrap_at_a_piped_width_80():
    """Fix round 1, findings 1 and 2 (round 2): every fixed-width piece of
    the detail row -- the text cap, and then the suggestion after it -- has
    to agree with the columns it renders into, or a maximal row overruns
    them and Rich wraps it onto a second line at a piped console's default
    width of 80.

    This is built from lint.py's own real data rather than hand-picked
    fixtures, because a fixture with short suggestion strings cannot catch
    a cap that's wrong for the long ones: the fixed ERROR-severity
    suggestions (COMMON_WORDS is warning-severity and never reaches this
    block) plus every BANNED_WORDS and SETUP_PHRASES suggestion, each paired
    with a maximal-length violation text, is the actual worst case in
    practice -- and asterisk, em_dash, and semicolon, whose suggestions are
    the longest here, are also the three most common violations in real
    prose.

    Neither "every line <= 80" nor "one line contains this violation's rule
    name" proves nothing wrapped: Rich splits an overlong row into two
    segments each under 80, and only the first segment carries the rule
    name the second is a bare continuation, so both a naive length check and
    a naive per-violation substring count silently pass while the row is
    broken across two lines regardless. The property that actually holds a
    wrap accountable is the total output line count: with none of these
    violations wrapped, the total must be exactly the no-violation baseline
    plus one line per violation. Verified this catches a real regression by
    simulating the un-capped suggestion this round fixes: it added 42
    violations across the baseline document and produced 56 lines instead
    of the expected 52 -- 4 rows had wrapped.
    """
    fixed_error_suggestions = [
        "comma, period, or parentheses",  # em_dash
        "period, or split the sentence",  # semicolon
        "'-' for bullets, rewrite for emphasis",  # asterisk
        "remove",  # hashtag
        "state the point directly",  # not_just
    ]
    real_suggestions = sorted(
        {*fixed_error_suggestions, *lint.SETUP_PHRASES.values(), *lint.BANNED_WORDS.values()}
    )
    violations = tuple(
        lint.Violation("worst_case", "error", index + 1, 1, "x" * 500, suggestion)
        for index, suggestion in enumerate(real_suggestions)
    )
    baseline_lines = _render_lines(make_result(violations=()))
    lines = _render_lines(make_result(violations=violations))

    assert all(len(line) <= 80 for line in lines)
    # The real "did anything wrap" check: one extra output line per
    # violation over the no-violation baseline, no more.
    assert len(lines) == len(baseline_lines) + len(violations)

    detail_lines = [line for line in lines if "worst case" in line]
    arrow_columns = {line.index("→") for line in detail_lines}
    assert len(arrow_columns) == 1  # the arrow lands in the same column every time


def test_suggestion_width_is_derived_to_exactly_fill_an_80_column_row():
    total = (
        report.DETAIL_LINE_WIDTH
        + report.DETAIL_RULE_WIDTH
        + report.DETAIL_TEXT_WIDTH
        + len(report.DETAIL_ARROW)
        + report.SUGGESTION_WIDTH
    )
    assert total == report.PIPED_CONSOLE_WIDTH


def test_longest_real_suggestion_is_truncated_to_the_suggestion_width():
    cleaned = report._clean_violation_text(
        "'-' for bullets, rewrite for emphasis", width=report.SUGGESTION_WIDTH
    )
    assert len(cleaned) <= report.SUGGESTION_WIDTH
    assert cleaned.endswith("…")


# --- Requirement B: corpus view shows the sample count per group ------------
#
# corpus_group_averages collapses each group to a single float and discards
# how many documents fed it. A document whose group score is None drops out
# of that group's denominator only, so two groups on the same corpus line can
# rest on very different sample sizes with nothing showing it.


def test_corpus_shows_document_count_per_group():
    doc_b = sources.Document("other.md", "Other", "Body.", "file", None)
    doc_c = sources.Document("third.md", "Third", "Body.", "file", None)
    results = [
        make_result(
            groups=(
                scoring.GroupResult("house_style", 1.0, ()),
                scoring.GroupResult("editorial", 0.4, ()),
            )
        ),
        make_result(
            document=doc_b,
            groups=(
                scoring.GroupResult("house_style", 0.5, ()),
                scoring.GroupResult("editorial", None, ()),
            ),
        ),
        make_result(
            document=doc_c,
            groups=(
                scoring.GroupResult("house_style", 0.9, ()),
                scoring.GroupResult("editorial", None, ()),
            ),
        ),
    ]
    out = render(
        report.render_corpus,
        results,
        PROF,
        {"input_tokens": 0, "output_tokens": 0},
        0.1,
        "jev-1.13.0",
    )
    assert "(n=3)" in out  # house_style scored in all three documents
    assert "(n=1)" in out  # editorial scored in only one


def test_json_summary_group_averages_unaffected_by_report_count_tracking():
    # Requirement B must not touch corpus_group_averages's own signature or
    # its values in the JSON summary.
    payload = json.loads(report.to_json([make_result()], PROF))
    assert payload["summary"]["group_averages"]["house_style"] == 1.0


# --- Global constraint: low confidence dims the bar -------------------------
#
# The brief colours every dimension bar purely by its score band. A
# needs_review dimension (confidence below the gate) is excluded from the
# composite entirely, but style_for(normalized) alone would still colour a
# high-scoring, low-confidence bar the same as a real, trustworthy one. Each
# row is now built as a plain Text() with "dim" applied only to its
# tree-branch prefix span (see _render_one), so a bare style_for(value) bar
# already reads solid; _dimension_bar_style adds "dim" on top only for the
# needs_review case, so a shaky bar reads visibly fainter than a solid one
# instead of both reading the same.


def test_high_score_low_confidence_dimension_bar_is_dimmed():
    dim = scoring.DimensionResult(
        "active_voice",
        "active voice",
        "house_style",
        0.5,
        4.0,
        0.95,
        {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.05, "4": 0.95},
        0.10,
        True,
    )
    assert report._dimension_bar_style(dim) == "dim green"


def test_high_score_high_confidence_dimension_bar_is_not_dimmed():
    dim = scoring.DimensionResult(
        "active_voice",
        "active voice",
        "house_style",
        0.5,
        4.0,
        0.95,
        {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.05, "4": 0.95},
        0.95,
        False,
    )
    assert report._dimension_bar_style(dim) == "green"


def test_low_confidence_bar_carries_the_dim_ansi_code_on_a_real_terminal():
    dim = scoring.DimensionResult(
        "active_voice",
        "active voice",
        "house_style",
        0.5,
        4.0,
        0.95,
        {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.05, "4": 0.95},
        0.10,
        True,
    )
    group = scoring.GroupResult("house_style", None, (dim,))
    result = make_result(groups=(group,))
    console = Console(width=100, force_terminal=True, color_system="standard", record=True)
    report.render_documents(console, [result])
    assert "\x1b[2;32m" in console.export_text(styles=True)


def test_high_confidence_bar_reads_solid_despite_the_dim_tree_prefix():
    """The tree row's own base style is "dim"; a trusted bar must still
    escape it and render plain green (SGR 32 with no 2), or a reader can
    never tell a solid score from a shaky one at a glance."""
    dim = scoring.DimensionResult(
        "active_voice",
        "active voice",
        "house_style",
        0.5,
        4.0,
        0.95,
        {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.05, "4": 0.95},
        0.95,
        False,
    )
    group = scoring.GroupResult("house_style", 0.95, (dim,))
    result = make_result(groups=(group,))
    console = Console(width=100, force_terminal=True, color_system="standard", record=True)
    report.render_documents(console, [result])
    rendered = console.export_text(styles=True)
    bar_run = report.bar(0.95, width=20)
    assert f"\x1b[32m{bar_run}\x1b[0m" in rendered  # the bar itself, solid and undimmed


def test_trusted_group_rollup_number_is_not_dim():
    """Fix round 1, finding 2: the group row used to be built as a single
    Text(prefix, style="dim"), so its own rollup number inherited dim too
    (rendering \\x1b[2;32m). Under --compact the group number is the only
    colour left on screen, so it must read solid, not faded."""
    dim = scoring.DimensionResult(
        "active_voice",
        "active voice",
        "house_style",
        0.5,
        4.0,
        0.95,
        {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.05, "4": 0.95},
        0.95,
        False,
    )
    group = scoring.GroupResult("house_style", 0.95, (dim,))
    result = make_result(groups=(group,))
    console = Console(width=100, force_terminal=True, color_system="standard", record=True)
    report.render_documents(console, [result])
    rendered = console.export_text(styles=True)
    assert "\x1b[32m0.95\x1b[0m" in rendered
    assert "\x1b[2;32m0.95\x1b[0m" not in rendered


# --- Fix wave: sparkline must degrade on a malformed payload, never raise --
#
# Whole-branch review finding 1: sorted(probabilities, key=lambda k: int(k))
# raised ValueError the moment any probability key off the wire wasn't
# int-parseable, and float(probabilities[level]) raised the same way for an
# unparseable value -- either one used to take the whole report down.


def test_sparkline_skips_a_key_that_does_not_parse_as_an_int():
    assert report.sparkline({"bad": 0.5, "1": 1.0}) == report.BLOCKS[-1]


def test_sparkline_skips_a_value_that_does_not_parse_as_a_float():
    assert report.sparkline({"0": "not-a-number", "1": 1.0}) == report.BLOCKS[-1]


def test_sparkline_returns_empty_when_nothing_parses():
    assert report.sparkline({"bad": 0.5, "worse": "also-bad"}) == ""


# --- Fix wave: a huge line number must not overrun its fixed column --------
#
# report.py:208's "line N" prefix used to be a bare .ljust(), which only
# pads a short string -- it does nothing once the string already reaches or
# exceeds the column width, so a six-digit line number silently overran the
# 15-column budget and wrapped the row, the same failure mode
# _clean_violation_text exists to prevent for the columns next to it.


def test_huge_line_number_does_not_wrap_the_detail_row():
    """A short text/suggestion pairing can't expose this: the row falls
    well short of 80 columns regardless of the line-number column, so the
    extra character from an unbounded 6-digit line number is lost in the
    slack. test_suggestion_width_is_derived_to_exactly_fill_an_80_column_row
    already proves the worst-case real text+suggestion pairing (a maximal
    span, asterisk's suggestion -- the longest fixed one) lands at exactly
    80 columns for an ordinary line number; reusing that same pairing here
    means the one variable left is the line-number column, so a wrap can
    only be caused by it.
    """
    text = "x" * 500
    suggestion = "'-' for bullets, rewrite for emphasis"  # the longest fixed suggestion
    normal = lint.Violation("asterisk", "error", 1, 1, text, suggestion)
    huge = lint.Violation("asterisk", "error", 100_000, 1, text, suggestion)
    baseline_lines = _render_lines(make_result(violations=(normal,)))
    huge_lines = _render_lines(make_result(violations=(huge,)))
    assert len(huge_lines) == len(baseline_lines)


def test_needs_review_marker_is_not_dim():
    """The ⚠ review marker is itself the warning; muting it with an
    inherited dim (\\x1b[2;33m) defeats the point of it being coloured
    yellow at all. It must render at full intensity."""
    dim = scoring.DimensionResult(
        "concision",
        "concision",
        "editorial",
        0.5,
        2.0,
        0.5,
        {"0": 0.1, "1": 0.2, "2": 0.4, "3": 0.2, "4": 0.1},
        0.44,
        True,
    )
    group = scoring.GroupResult("editorial", None, (dim,))
    result = make_result(groups=(group,))
    console = Console(width=100, force_terminal=True, color_system="standard", record=True)
    report.render_documents(console, [result])
    rendered = console.export_text(styles=True)
    assert "\x1b[33m  ⚠ review\x1b[0m" in rendered
    assert "\x1b[2;33m" not in rendered
