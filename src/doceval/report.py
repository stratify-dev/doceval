"""Rendering.

Three numbers per dimension: the normalized score as a bar, the distribution
across levels as a sparkline, and the confidence. The distribution is the
reason to use a System One model at all. A 0.35 spread across the bottom three
levels differs from a 0.35 concentrated on one, and a mean hides the difference.

Rich detects a missing terminal and drops color, animation, and box drawing on
its own, so piped output stays readable.
"""

from __future__ import annotations

import json
import re

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .profile import Profile
from .scoring import (
    DimensionResult,
    DocumentResult,
    corpus_group_averages,
    weakest_dimensions,
)

BLOCKS = "▁▂▃▄▅▆▇█"
BAR_FILLED = "█"
BAR_EMPTY = "░"
LABEL_WIDTH = 20

# The violation detail row's fixed columns, left to right: a padded
# "line N" prefix, the rule name, the quoted violation text, then "→ " and
# the suggestion. Named here, and used by both the row and the arithmetic
# below, so the two can never drift back out of sync the way the text cap
# and its column already did once.
DETAIL_LINE_WIDTH = 15
DETAIL_RULE_WIDTH = 16
DETAIL_ARROW = "→ "

# A piped console has no TTY, and Rich's own default width without one is
# 80 columns -- the width real usage (CI logs, `doceval | less`, redirected
# output) actually renders at, regardless of the terminal this was written
# in.
PIPED_CONSOLE_WIDTH = 80

# A violation's text used to be exactly one character (the old asterisk rule
# matched single characters). It is now an unbounded span, so it needs a cap
# before it reaches a fixed-width report column. The cap must agree with that
# column: the violation detail row quotes the cleaned text and ljust()s it to
# DETAIL_TEXT_WIDTH (18 + 2 quote characters = 20). Widen one and the other
# must widen with it, or a maximal violation still overruns the row.
VIOLATION_TEXT_WIDTH = 18
DETAIL_TEXT_WIDTH = VIOLATION_TEXT_WIDTH + 2  # the two quote characters

# lint.py's fixed rule suggestions were never capped either, and several are
# long enough on their own to overrun a piped console: asterisk's is 37
# characters, and em_dash's and semicolon's are 29 -- the three most common
# violations in practice. Capping the text column (above) was necessary but
# not sufficient; the suggestion after it needs the same treatment, sized to
# whatever the other fixed columns and the arrow leave inside 80 columns.
SUGGESTION_WIDTH = (
    PIPED_CONSOLE_WIDTH - DETAIL_LINE_WIDTH - DETAIL_RULE_WIDTH
    - DETAIL_TEXT_WIDTH - len(DETAIL_ARROW)
)

_WHITESPACE_RUN = re.compile(r"\s+")


def sparkline(probabilities: dict[str, float]) -> str:
    """One block per level, height proportional to that level's probability.

    A key has to parse as an int to sort and place, and a value has to parse
    as a float to size a block. A malformed answer payload can carry either
    kind of bad entry; each is skipped rather than raised on, so one bad key
    never drops the whole row. No usable entry at all renders the same ""
    an empty dict already does.
    """
    parsed: list[tuple[int, float]] = []
    for key, value in probabilities.items():
        try:
            level = int(key)
        except (TypeError, ValueError):
            continue
        try:
            probability = float(value)
        except (TypeError, ValueError):
            continue
        parsed.append((level, probability))

    if not parsed:
        return ""

    top = len(BLOCKS) - 1
    return "".join(
        BLOCKS[round(max(0.0, min(1.0, probability)) * top)]
        for _, probability in sorted(parsed)
    )


def bar(value: float | None, width: int = LABEL_WIDTH) -> str:
    if value is None:
        return BAR_EMPTY * width
    filled = round(max(0.0, min(1.0, value)) * width)
    return BAR_FILLED * filled + BAR_EMPTY * (width - filled)


def style_for(value: float | None) -> str:
    if value is None:
        return "dim"
    if value >= 0.80:
        return "green"
    if value >= 0.60:
        return "yellow"
    return "red"


def _dimension_bar_style(dimension: DimensionResult) -> str:
    """The score band's colour, dimmed when confidence is too low to trust it.

    A dimension flagged needs_review is excluded from the composite by
    scoring._weighted_mean, but its bar still renders in the tree. Rows are
    built with a "dim" style on the tree-branch prefix only (see _render_one),
    not on the row as a whole, so a plain style_for(value) here already reads
    solid by default; "dim" is added on top only for the shaky case, so a
    low-confidence bar visibly fades against every neighbouring solid one.
    """
    band = style_for(dimension.normalized)
    return f"dim {band}" if dimension.needs_review else band


def _clean_violation_text(text: str, width: int = VIOLATION_TEXT_WIDTH) -> str:
    """Collapse whitespace runs and cap length before text hits a fixed column.

    Used for both a violation's own text and, at a different width, its
    suggestion (see SUGGESTION_WIDTH) -- both are strings a fixed-width
    report column has to absorb without wrapping the row.

    Two defects reach violation text unless it is cleaned first. The
    asterisk rule now matches a whole span rather than one character, so an
    entire italic paragraph can arrive as a single violation's text;
    rendered verbatim it would blow out the fixed-width column and wrap the
    row. Second, lint.mask_code blanks code with spaces of equal length to
    keep offsets exact, so a span crossing inline code carries a run of
    blanks in its text (e.g. "*span across        here*"). Collapsing
    whitespace runs to one space fixes both the visible gap and, combined
    with the length cap, the column width.

    A suggestion string has neither defect (lint.py's suggestions are
    static, clean text), but several are long enough on their own to
    overrun what's left of an 80-column row after the other fixed columns,
    so the same cap applies to it too.
    """
    collapsed = _WHITESPACE_RUN.sub(" ", text).strip()
    if len(collapsed) <= width:
        return collapsed
    return collapsed[: width - 1].rstrip() + "…"


def _line_prefix(line: int, width: int = DETAIL_LINE_WIDTH) -> str:
    """The detail row's "line N" column, padded like the others -- and,
    unlike the plain `.ljust()` this replaces, capped too. `ljust` only
    pads a short string; it does nothing once the string is already at or
    past `width`, so an implausibly large line number (a very long
    document) silently overran the fixed column and wrapped the row, the
    same failure mode `_clean_violation_text` exists to prevent for the
    text and suggestion columns next to it.
    """
    text = f"     line {line}"
    if len(text) <= width:
        return text.ljust(width)
    return text[: width - 1] + "…"


def render_documents(console: Console, results: list[DocumentResult], compact: bool = False) -> None:
    for result in results:
        _render_one(console, result, compact)
        console.print()


def _render_one(console: Console, result: DocumentResult, compact: bool) -> None:
    header = Text(f" {result.document.id}")
    header.pad_right(max(1, 56 - len(result.document.id)))
    if result.error is not None:
        header.append("ERROR", style="bold red")
        console.print(header)
        console.print(Text(f"   {result.error}", style="red"))
        return

    header.append(_number(result.composite), style=style_for(result.composite))
    header.append("  ")
    header.append(result.verdict, style=f"bold {style_for(result.composite)}")
    console.print(header)

    if not result.gate_passed:
        console.print(Text("   gate: not finished prose, scoring skipped", style="dim"))
        return

    console.print(Text(" │", style="dim"))
    for group in result.groups:
        # A Text's own style= constructor argument is a base that Rich
        # inherits into every span appended after it, even one carrying its
        # own explicit style. Building each row as a plain Text() and
        # appending the tree-branch prefix with style="dim" keeps the dim
        # confined to that prefix; every later span then carries only the
        # style it is given, instead of a color band getting muddied dim.
        line = Text()
        line.append(" ├ ", style="dim")
        line.append(group.name.replace("_", " ").ljust(LABEL_WIDTH + 24))
        line.append(_number(group.score), style=style_for(group.score))
        console.print(line)

        if compact:
            continue

        for index, dimension in enumerate(group.dimensions):
            last = index == len(group.dimensions) - 1
            row = Text()
            row.append(" │ " + ("└ " if last else "├ "), style="dim")
            row.append(dimension.label.ljust(LABEL_WIDTH))
            row.append(bar(dimension.normalized), style=_dimension_bar_style(dimension))
            row.append(f"  {dimension.normalized:.2f}  ")
            row.append(sparkline(dimension.probabilities), style="cyan")
            row.append(f"  {dimension.confidence:.2f}")
            if dimension.needs_review:
                row.append("  ⚠ review", style="yellow")
            console.print(row)
        console.print(Text(" │", style="dim"))

    counts = Text()
    counts.append(" └ ", style="dim")
    counts.append("lint".ljust(LABEL_WIDTH))
    counts.append(f"{len(result.errors)} errors · {len(result.warnings)} warnings")
    console.print(counts)

    for violation in result.errors:
        cleaned = _clean_violation_text(violation.text)
        suggestion = _clean_violation_text(violation.suggestion, width=SUGGESTION_WIDTH)
        detail = Text()
        detail.append(_line_prefix(violation.line), style="dim")
        detail.append(violation.rule.replace("_", " ").ljust(DETAIL_RULE_WIDTH))
        detail.append(f'"{cleaned}"'.ljust(DETAIL_TEXT_WIDTH))
        detail.append(f"{DETAIL_ARROW}{suggestion}", style="dim")
        console.print(detail)


def _group_counts(results: list[DocumentResult]) -> dict[str, int]:
    """Documents contributing a non-None score to each group.

    corpus_group_averages folds each group down to one float and drops the
    denominator behind it. A document whose group scored None drops out of
    that group's average only, so two groups on the same corpus line can
    rest on very different sample sizes with nothing able to show it. This
    mirrors corpus_group_averages's own None-filtering rather than changing
    its signature.
    """
    counts: dict[str, int] = {}
    for result in results:
        for group in result.groups:
            if group.score is not None:
                counts[group.name] = counts.get(group.name, 0) + 1
    return counts


def render_corpus(
    console: Console,
    results: list[DocumentResult],
    prof: Profile,
    usage: dict[str, int],
    elapsed: float,
    model: str,
) -> None:
    body = Text()

    for result in results:
        body.append("  ")
        body.append(result.document.id[:28].ljust(30))
        body.append(bar(result.composite, width=18), style=style_for(result.composite))
        body.append(f"  {_number(result.composite)}  ")
        body.append(_marker(result), style=style_for(result.composite))
        body.append("\n")

    averages = corpus_group_averages(results)
    if averages:
        counts = _group_counts(results)
        body.append("\n  group averages\n", style="bold")
        for name, value in averages.items():
            body.append(f"  {name.replace('_', ' ').ljust(16)}")
            body.append(bar(value, width=10), style=style_for(value))
            body.append(f"  {value:.2f}  (n={counts.get(name, 0)})\n")

    weakest = weakest_dimensions(results)
    if weakest:
        rendered = " · ".join(f"{label} {value:.2f}" for label, value in weakest)
        body.append(f"\n  weakest dimensions  {rendered}\n", style="dim")

    cached = sum(1 for r in results if r.cached)
    tokens = usage.get("input_tokens", 0)
    footer = (
        f"\n  {model} · {tokens:,} tokens · {elapsed:.1f}s · {cached} cached"
    )
    body.append(footer, style="dim")

    console.print(Panel(body, title=f"{len(results)} documents · {prof.name}",
                        title_align="left", border_style="dim"))


def to_json(results: list[DocumentResult], prof: Profile) -> str:
    payload = {
        "profile": prof.name,
        "audience": prof.audience,
        "documents": [_document_payload(r) for r in results],
        "summary": {
            "group_averages": corpus_group_averages(results),
            "weakest_dimensions": [
                {"label": label, "score": value}
                for label, value in weakest_dimensions(results)
            ],
            "cached": sum(1 for r in results if r.cached),
            "errored": sum(1 for r in results if r.error is not None),
        },
    }
    return json.dumps(payload, indent=2)


def to_markdown(results: list[DocumentResult], prof: Profile) -> str:
    lines = [
        f"# doceval · {prof.name}",
        "",
        "| document | composite | verdict | lint errors |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result.document.id} | {_number(result.composite)} | "
            f"{result.verdict} | {len(result.errors)} |"
        )
    for result in results:
        if result.error is not None or not result.groups:
            continue
        lines += ["", f"## {result.document.id}", "",
                  "| dimension | score | confidence | review |", "| --- | --- | --- | --- |"]
        for group in result.groups:
            for dimension in group.dimensions:
                mark = "yes" if dimension.needs_review else ""
                lines.append(
                    f"| {dimension.label} | {dimension.normalized:.2f} | "
                    f"{dimension.confidence:.2f} | {mark} |"
                )
    return "\n".join(lines) + "\n"


def _document_payload(result: DocumentResult) -> dict:
    return {
        "id": result.document.id,
        "title": result.document.title,
        "origin": result.document.origin,
        "verdict": result.verdict,
        "composite": result.composite,
        "gate_passed": result.gate_passed,
        "cached": result.cached,
        "model": result.model,
        "error": result.error,
        "groups": [
            {
                "name": group.name,
                "score": group.score,
                "dimensions": [
                    {
                        "id": d.id,
                        "label": d.label,
                        "weight": d.weight,
                        "raw": d.raw,
                        "normalized": d.normalized,
                        "confidence": d.confidence,
                        "needs_review": d.needs_review,
                        "probabilities": d.probabilities,
                    }
                    for d in group.dimensions
                ],
            }
            for group in result.groups
        ],
        "lint": {
            "errors": [_violation_payload(v) for v in result.errors],
            "warnings": [_violation_payload(v) for v in result.warnings],
        },
    }


def _violation_payload(violation) -> dict:
    return {
        "rule": violation.rule,
        "line": violation.line,
        "column": violation.column,
        "text": violation.text,
        "suggestion": violation.suggestion,
    }


def _number(value: float | None) -> str:
    return "  --" if value is None else f"{value:.2f}"


def _marker(result: DocumentResult) -> str:
    if result.error is not None:
        return "✗ error"
    if not result.gate_passed:
        return "· not prose"
    review = sum(
        1 for g in result.groups for d in g.dimensions if d.needs_review
    )
    if review:
        return f"⚠ {review} review"
    return "✓"
