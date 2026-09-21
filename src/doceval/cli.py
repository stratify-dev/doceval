"""Command wiring.

Thresholds are opt-in. A plain run reports and exits 0 whatever the scores,
which is what you want at a terminal. `--fail-under` and `--fail-on-lint` turn
the same command into a CI gate.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import click
from rich.console import Console

from . import evaluate, report, scoring
from . import lint as lint_mod
from . import profile as profile_mod
from .cache import DEFAULT_CACHE_DIR
from .config import MissingAPIKey, load_env, require_api_key, resolve_model
from .sources import SourceError, load_document

EXIT_OK = 0
EXIT_THRESHOLD = 1
EXIT_ERROR = 2


@click.group()
@click.version_option(package_name="doceval")
def main() -> None:
    """Score written content on house style, editorial quality, and audience fit."""
    load_env()


@main.command(name="profiles")
def profiles_command() -> None:
    """List the bundled profiles."""
    console = Console()
    for name in profile_mod.bundled_profiles():
        prof = profile_mod.load_profile(name)
        console.print(f"  [bold]{name}[/bold]  {len(prof.dimensions)} dimensions")
        console.print(f"    [dim]{prof.audience}[/dim]")


@main.command(name="lint")
@click.argument("paths", nargs=-1, required=True)
@click.option("--min-words", default=1, show_default=True, help="Reject shorter documents.")
@click.option("--timeout", default=20.0, show_default=True, help="URL fetch timeout in seconds.")
@click.option("--fail-on-lint", is_flag=True, help="Exit 1 when any lint error is found.")
@click.option("--no-color", is_flag=True, help="Plain output.")
def lint_command(paths, min_words, timeout, fail_on_lint, no_color) -> None:
    """Run the mechanical style rules. No API key, no network for local files."""
    console = Console(no_color=no_color)
    documents, failures = _load_all(paths, timeout=timeout, min_words=min_words, console=console)

    total_errors = 0
    for document in documents:
        violations = lint_mod.lint(document.text)
        errors = [v for v in violations if v.severity == "error"]
        warnings = [v for v in violations if v.severity == "warning"]
        total_errors += len(errors)

        console.print(
            f"\n [bold]{document.id}[/bold]  "
            f"{len(errors)} errors · {len(warnings)} warnings"
        )
        for violation in violations:
            style = "red" if violation.severity == "error" else "yellow"
            console.print(
                f"   [dim]line {violation.line}:{violation.column}[/dim]  "
                f"[{style}]{violation.rule.replace('_', ' ')}[/{style}]  "
                f'"{violation.text}"  [dim]→ {violation.suggestion}[/dim]'
            )

    # A source that fails to load is an operational error even when other
    # paths succeeded and even when --fail-on-lint would otherwise pass;
    # this matches eval, where any failure exits 2 regardless of the rest
    # of the run. The per-file results above still print, so a partial
    # report survives even though the exit code reports the failure.
    if failures:
        sys.exit(EXIT_ERROR)
    if fail_on_lint and total_errors:
        sys.exit(EXIT_THRESHOLD)


@main.command(name="eval")
@click.argument("paths", nargs=-1, required=True)
@click.option("--profile", "profile_name", default="house-style", show_default=True)
@click.option("--format", "output_format",
              type=click.Choice(["table", "json", "markdown"]), default="table", show_default=True)
@click.option("--min-confidence", default=0.6, show_default=True,
              help="Dimensions below this are flagged and excluded from the composite.")
@click.option("--fail-under", type=float, default=None,
              help="Exit 1 if any document scores below this, or couldn't be scored.")
@click.option("--fail-on-lint", is_flag=True, help="Exit 1 when any lint error is found.")
@click.option("--concurrency", default=8, show_default=True)
@click.option("--no-cache", is_flag=True, help="Ignore cached answers.")
@click.option("--cache-dir", default=DEFAULT_CACHE_DIR, show_default=True)
@click.option("--timeout", default=20.0, show_default=True, help="URL fetch timeout in seconds.")
@click.option("--api-timeout", default=evaluate.API_TIMEOUT, show_default=True,
              help="TypeSafe API request timeout in seconds, per HTTP attempt. "
                   "Distinct from --timeout, which only bounds fetching a URL source.")
@click.option("--min-words", default=100, show_default=True,
              help="Reject documents extracting fewer words than this.")
@click.option("--dump-text", type=click.Path(file_okay=False), default=None,
              help="Write each document's extracted prose to this directory.")
@click.option("--compact", is_flag=True, help="Show group rollups only.")
@click.option("--no-color", is_flag=True, help="Plain output.")
def eval_command(paths, profile_name, output_format, min_confidence, fail_under,
                 fail_on_lint, concurrency, no_cache, cache_dir, timeout, api_timeout,
                 min_words, dump_text, compact, no_color) -> None:
    """Evaluate documents and URLs against a profile."""
    quiet = output_format != "table"
    console = Console(no_color=no_color, stderr=quiet)

    try:
        require_api_key()
    except MissingAPIKey as error:
        console.print(f"[red]✗[/red] {error.message}")
        sys.exit(EXIT_ERROR)

    try:
        prof = profile_mod.load_profile(profile_name)
    except profile_mod.ProfileError as error:
        console.print(f"[red]✗[/red] {error}")
        sys.exit(EXIT_ERROR)

    documents, failures = _load_all(paths, timeout=timeout, min_words=min_words, console=console)
    if not documents:
        sys.exit(EXIT_ERROR)

    if dump_text:
        _dump(documents, Path(dump_text))

    started = time.monotonic()
    with console.status("[dim]evaluating…[/dim]", spinner="dots") as status:
        def on_start(document):
            status.update(f"[dim]evaluating[/dim] {document.id}")

        outcomes = asyncio.run(
            evaluate.evaluate_documents(
                documents, prof, concurrency=concurrency,
                cache_dir=cache_dir, use_cache=not no_cache, api_timeout=api_timeout,
                on_start=on_start,
            )
        )
    elapsed = time.monotonic() - started

    results = [_score_outcome(outcome, prof, min_confidence) for outcome in outcomes]
    results += [scoring.error_result(document, message) for document, message in failures]

    _render(results, prof, outcomes, elapsed, output_format, compact, no_color)
    sys.exit(_exit_code(results, fail_under, fail_on_lint))


def _score_outcome(outcome, prof, min_confidence):
    """Score one outcome, or turn a scoring failure into that document's
    own error_result rather than letting it take the whole run down.

    evaluate.py isolates every per-document failure during the network call,
    but scoring and rendering happen afterward, outside that net entirely.
    scoring.py's own coercion (see _as_float) already keeps a malformed
    answer payload from raising in the first place, but this is the second,
    outer layer: whatever scoring bug slips past that -- known or not --
    must still cost only its own document, the same guarantee evaluate.py
    already gives the request itself.
    """
    if outcome.error is not None:
        return scoring.error_result(outcome.document, outcome.error)
    try:
        return scoring.score_document(
            document=outcome.document, prof=prof, answers=outcome.answers or {},
            violations=lint_mod.lint(outcome.document.text),
            min_confidence=min_confidence, model=outcome.model, cached=outcome.cached,
        )
    except Exception as error:
        return scoring.error_result(
            outcome.document, f"{type(error).__name__}: {error}"
        )


def _render(results, prof, outcomes, elapsed, output_format, compact, no_color) -> None:
    if output_format == "json":
        click.echo(report.to_json(results, prof))
        return
    if output_format == "markdown":
        click.echo(report.to_markdown(results, prof))
        return

    console = Console(no_color=no_color)
    usage = {
        "input_tokens": sum(o.usage.get("input_tokens", 0) for o in outcomes),
        "output_tokens": sum(o.usage.get("output_tokens", 0) for o in outcomes),
    }
    model = next((o.model for o in outcomes if o.model), resolve_model())
    console.print()
    report.render_documents(console, results, compact=compact)
    report.render_corpus(console, results, prof, usage, elapsed, model)


def _load_all(paths, *, timeout, min_words, console):
    documents, failures = [], []
    for path in paths:
        try:
            documents.append(load_document(path, timeout=timeout, min_words=min_words))
        except SourceError as error:
            console.print(f"[red]✗[/red] {error}")
            failures.append((_placeholder(path), str(error)))
    return documents, failures


def _placeholder(path: str):
    from .sources import Document, is_url

    return Document(id=path, title=path, text="",
                    origin="url" if is_url(path) else "file", fetched_at=None)


def _dump(documents, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for document in documents:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in document.id)
        (directory / f"{safe}.txt").write_text(document.text, encoding="utf-8")


def _exit_code(results, fail_under: float | None, fail_on_lint: bool) -> int:
    if any(r.error is not None for r in results):
        return EXIT_ERROR
    # A document with no composite (gate failure -> NOT_PROSE, or every
    # dimension flagged needs_review -> UNSCORED) has not met a threshold
    # the caller explicitly asked for. Treating a missing composite as a
    # pass would let exactly the documents the model could not judge sail
    # through a CI gate; report.py already renders NOT_PROSE/UNSCORED
    # distinctly from a low score, so this doesn't get lumped in as "0.0".
    if fail_under is not None and any(
        r.composite is None or r.composite < fail_under for r in results
    ):
        return EXIT_THRESHOLD
    if fail_on_lint and any(r.errors for r in results):
        return EXIT_THRESHOLD
    return EXIT_OK
