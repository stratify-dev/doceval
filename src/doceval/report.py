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

# A violation's text used to be exactly one character (the old asterisk rule
# matched single characters). It is now an unbounded span, so it needs a cap
# before it reaches a fixed-width report column.
VIOLATION_TEXT_WIDTH = 40
_WHITESPACE_RUN = re.compile(r"\s+")


def sparkline(probabilities: dict[str, float]) -> str:
    """One block per level, height proportional to that level's probability."""
    if not probabilities:
        return ""
    levels = sorted(probabilities, key=lambda key: int(key))
    top = len(BLOCKS) - 1
    return "".join(
        BLOCKS[round(max(0.0, min(1.0, float(probabilities[level]))) * top)]
        for level in levels
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
    scoring._weighted_mean, but its bar still renders in the tree. Each
    dimension row's tree-branch prefix (" │ ├ ") carries a "dim" base style
    for the connector characters, and Rich Text inherits that base into
    every appended span whose own style doesn't say otherwise, so a plain
    style_for(value) bar reads dim either way. Both branches here are
    explicit for that reason: "not dim" overrides the inherited dim so a
    trustworthy score reads solid, and "dim" holds a shaky one faded, so
    the two are visibly different rather than uniformly muted.
    """
    band = style_for(dimension.normalized)
    return f"dim {band}" if dimension.needs_review else f"not dim {band}"


def _clean_violation_text(text: str, width: int = VIOLATION_TEXT_WIDTH) -> str:
    """Collapse whitespace runs and cap length before a violation hits a row.

    Two defects reach this text unless it is cleaned first. The asterisk
    rule now matches a whole span rather than one character, so an entire
    italic paragraph can arrive as a single violation's text; rendered
    verbatim it would blow out the fixed-width column and wrap the row.
    Second, lint.mask_code blanks code with spaces of equal length to keep
    offsets exact, so a span crossing inline code carries a run of blanks
    in its text (e.g. "*span across        here*"). Collapsing whitespace
    runs to one space fixes both the visible gap and, combined with the
    length cap, the column width.
    """
    collapsed = _WHITESPACE_RUN.sub(" ", text).strip()
    if len(collapsed) <= width:
        return collapsed
    return collapsed[: width - 1].rstrip() + "…"


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
        line = Text(" ├ ", style="dim")
        line.append(group.name.replace("_", " ").ljust(LABEL_WIDTH + 24))
        line.append(_number(group.score), style=style_for(group.score))
        console.print(line)

        if compact:
            continue

        for index, dimension in enumerate(group.dimensions):
            last = index == len(group.dimensions) - 1
            row = Text(" │ " + ("└ " if last else "├ "), style="dim")
            row.append(dimension.label.ljust(LABEL_WIDTH))
            row.append(bar(dimension.normalized), style=_dimension_bar_style(dimension))
            row.append(f"  {dimension.normalized:.2f}  ")
            row.append(sparkline(dimension.probabilities), style="cyan")
            row.append(f"  {dimension.confidence:.2f}")
            if dimension.needs_review:
                row.append("  ⚠ review", style="yellow")
            console.print(row)
        console.print(Text(" │", style="dim"))

    counts = Text(" └ ", style="dim")
    counts.append("lint".ljust(LABEL_WIDTH))
    counts.append(f"{len(result.errors)} errors · {len(result.warnings)} warnings")
    console.print(counts)

    for violation in result.errors:
        cleaned = _clean_violation_text(violation.text)
        detail = Text(f"     line {violation.line}".ljust(15), style="dim")
        detail.append(violation.rule.replace("_", " ").ljust(16))
        detail.append(f'"{cleaned}"'.ljust(20))
        detail.append(f"→ {violation.suggestion}", style="dim")
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
