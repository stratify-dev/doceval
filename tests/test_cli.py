import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from doceval import cli, config, evaluate, sources
from doceval import profile as profile_mod

PROSE = " ".join(["word"] * 200)

ANSWERS = {
    dim: {"type": "score", "score": 4.0, "confidence": 0.95,
          "legend": {str(i): c for i, c in enumerate("abcde")},
          "probabilities": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 1.0}}
    for dim in ("active_voice", "sentence_impact", "vague_referents",
                "adjective_restraint", "cliche_free", "opening_strength",
                "structure_flow", "concision", "technical_level", "takeaway_clarity")
}
ANSWERS["is_finished_prose"] = {"type": "noul", "noul": 1.0}


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    """Keep a developer's real .env out of the CLI tests.

    cli.main calls load_env(), which walks up from the working directory. A
    real .env would re-inject TYPESAFE_API_KEY after monkeypatch.delenv and
    quietly break every missing-key assertion.
    """
    monkeypatch.setattr(cli, "load_env", lambda *args, **kwargs: None)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def post(tmp_path):
    path = tmp_path / "post.md"
    path.write_text(f"# Post\n\n{PROSE}\n")
    return str(path)


@pytest.fixture
def stub_api(monkeypatch):
    async def fake(documents, prof, **kwargs):
        return [
            evaluate.Outcome(document=d, answers=ANSWERS, model="jev-1.13.0",
                             usage={"input_tokens": 100, "output_tokens": 10})
            for d in documents
        ]

    monkeypatch.setattr(cli.evaluate, "evaluate_documents", fake)
    monkeypatch.setenv(config.API_KEY_ENV, "sk-test")


def test_lint_needs_no_api_key(runner, post, monkeypatch, tmp_path):
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    result = runner.invoke(cli.main, ["lint", post])
    assert result.exit_code == 0


def test_lint_reports_violations(runner, tmp_path, monkeypatch):
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    path = tmp_path / "bad.md"
    path.write_text("We utilize the API — daily.\n")
    result = runner.invoke(cli.main, ["lint", str(path)])
    assert "utilize" in result.output
    assert result.exit_code == 0


def test_lint_fails_on_errors_when_asked(runner, tmp_path, monkeypatch):
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    path = tmp_path / "bad.md"
    path.write_text("We utilize the API.\n")
    result = runner.invoke(cli.main, ["lint", "--fail-on-lint", str(path)])
    assert result.exit_code == 1


def test_profiles_lists_house_style(runner, monkeypatch):
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    result = runner.invoke(cli.main, ["profiles"])
    assert "house-style" in result.output
    assert result.exit_code == 0


def test_eval_without_a_key_exits_two(runner, post, monkeypatch):
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    result = runner.invoke(cli.main, ["eval", post])
    assert result.exit_code == 2
    assert config.API_KEY_ENV in result.output
    assert "console.typesafe.ai/keys" in result.output


def test_eval_renders_a_report(runner, post, stub_api):
    result = runner.invoke(cli.main, ["eval", post, "--no-cache"])
    assert result.exit_code == 0
    assert "active voice" in result.output
    assert "GOOD" in result.output


def test_eval_json_output_is_valid(runner, post, stub_api):
    result = runner.invoke(cli.main, ["eval", post, "--format", "json", "--no-cache"])
    payload = json.loads(result.output)
    assert payload["documents"][0]["composite"] == 1.0


def test_eval_markdown_output(runner, post, stub_api):
    result = runner.invoke(cli.main, ["eval", post, "--format", "markdown", "--no-cache"])
    assert "| document |" in result.output


def test_eval_exits_zero_without_a_threshold(runner, post, stub_api, monkeypatch):
    async def weak(documents, prof, **kwargs):
        answers = {**ANSWERS}
        for key in answers:
            if key != "is_finished_prose":
                answers[key] = {**answers[key], "score": 0.0}
        return [evaluate.Outcome(document=d, answers=answers, model="jev-1.13.0")
                for d in documents]

    monkeypatch.setattr(cli.evaluate, "evaluate_documents", weak)
    result = runner.invoke(cli.main, ["eval", post, "--no-cache"])
    assert result.exit_code == 0


def test_eval_fails_under_threshold(runner, post, stub_api, monkeypatch):
    async def weak(documents, prof, **kwargs):
        answers = {k: ({**v, "score": 0.0} if k != "is_finished_prose" else v)
                   for k, v in ANSWERS.items()}
        return [evaluate.Outcome(document=d, answers=answers, model="jev-1.13.0")
                for d in documents]

    monkeypatch.setattr(cli.evaluate, "evaluate_documents", weak)
    result = runner.invoke(cli.main, ["eval", post, "--fail-under", "0.7", "--no-cache"])
    assert result.exit_code == 1


def test_eval_reports_an_unreadable_source(runner, stub_api):
    result = runner.invoke(cli.main, ["eval", "missing.md", "--no-cache"])
    assert "not found" in result.output
    assert result.exit_code == 2


def test_eval_continues_past_one_bad_source(runner, post, stub_api):
    result = runner.invoke(cli.main, ["eval", post, "missing.md", "--no-cache"])
    assert "not found" in result.output
    assert "active voice" in result.output


def test_dump_text_writes_extracted_prose(runner, post, stub_api, tmp_path):
    target = tmp_path / "dump"
    runner.invoke(cli.main, ["eval", post, "--dump-text", str(target), "--no-cache"])
    assert list(target.glob("*.txt"))


def test_bad_profile_exits_two(runner, post, stub_api, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: b\naudience: x\ndimensions:\n  a:\n    group: nope\n"
                   "    weight: 1.0\n    type: score\n    instructions: q\n    levels: [a, b]\n")
    result = runner.invoke(cli.main, ["eval", post, "--profile", str(bad)])
    assert result.exit_code == 2
    assert "nope" in result.output


# --- Property tests beyond the brief -----------------------------------
#
# Each of these pins a property that could silently regress: a leaked key
# requirement, progress text bleeding into machine-readable output, an exit
# code drifting from its documented meaning, one bad document swallowing the
# rest of a run, or a security property (no --api-key flag) that is easy to
# reintroduce by accident later.


def test_lint_and_profiles_work_without_key_or_network(runner, post, monkeypatch):
    """lint and profiles must run with no API key set at all.

    _block_network (conftest, autouse) already guarantees no test in this
    file can reach the network unless marked @pytest.mark.live; asserting
    exit_code == 0 here pins the other half: that neither command even
    tries to require a key first.
    """
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    lint_result = runner.invoke(cli.main, ["lint", post])
    assert lint_result.exit_code == 0

    profiles_result = runner.invoke(cli.main, ["profiles"])
    assert profiles_result.exit_code == 0


def test_json_format_stdout_is_only_json(runner, post, stub_api):
    """Nothing but the JSON payload may reach stdout in --format json.

    Two things were wrong with the first version of this test, both found
    in review:

    1. It asserted against `result.output`, but in click 8.2+ that's stdout
       and stderr MIXED in the order they were written (`Result.output`'s
       own docstring says so; `mix_stderr` was removed). Swapping
       `_load_all`'s console for a plain `Console()` -- i.e. reintroducing
       the exact bug this test claims to pin -- would leave it green,
       because the contaminating text would just get folded into the same
       string being parsed. `result.stdout` and `result.stderr` are the
       independent streams that can actually tell them apart.
    2. Its corpus was one clean document, so `_load_all` never printed an
       error line at all -- there was nothing that *could* contaminate
       stdout even with broken stream routing. Adding a source that fails
       to load gives the test something real to fail to keep off stdout.
    """
    result = runner.invoke(
        cli.main, ["eval", post, "missing.md", "--format", "json", "--no-cache"]
    )
    payload = json.loads(result.stdout)
    assert payload["documents"][0]["id"] == post
    assert "missing.md" in result.stderr
    assert "not found" in result.stderr


@pytest.mark.parametrize(
    "args,expected_exit",
    [
        pytest.param(["eval", "{post}", "--no-cache"], 0, id="clean-run"),
        pytest.param(["eval", "{post}", "--fail-under", "0.7", "--no-cache"], 1,
                     id="threshold-breach"),
    ],
)
def test_eval_exit_code_is_exact(runner, post, stub_api, monkeypatch, args, expected_exit):
    """Exit codes are exact integers, not merely falsy/truthy.

    A prior bug could return any nonzero code on failure and any assertion
    using `!= 0` or `bool(exit_code)` would miss a threshold breach reported
    as exit 2 (an operational error) instead of exit 1. Pinning the literal
    integer catches that class of bug.
    """
    if expected_exit == 1:
        async def weak(documents, prof, **kwargs):
            answers = {k: ({**v, "score": 0.0} if k != "is_finished_prose" else v)
                       for k, v in ANSWERS.items()}
            return [evaluate.Outcome(document=d, answers=answers, model="jev-1.13.0")
                    for d in documents]

        monkeypatch.setattr(cli.evaluate, "evaluate_documents", weak)

    resolved = [a.format(post=post) for a in args]
    result = runner.invoke(cli.main, resolved)
    assert result.exit_code == expected_exit


def test_missing_key_exit_code_is_exactly_two(runner, post, monkeypatch):
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    result = runner.invoke(cli.main, ["eval", post])
    assert result.exit_code == 2


def test_bad_source_does_not_prevent_good_document_reporting(runner, post, stub_api):
    """One missing file must not blank out a document that loaded fine.

    Checking only "not found" appears (as test_eval_continues_past_one_bad_source
    does) would still pass if the good document's own SCORES never rendered --
    e.g. if it silently dropped out of the run instead of reporting alongside
    the error. Asserting the good document's filename sits right next to its
    verdict band pins that its actual result, not just some report boilerplate,
    survived. (The full tmp_path is not asserted verbatim: report.py wraps a
    long path across lines at its fixed 80-column width, so only the short
    filename suffix is guaranteed to stay intact on one line.)
    """
    result = runner.invoke(cli.main, ["eval", post, "missing.md", "--no-cache"])
    assert "missing.md" in result.output
    assert "not found" in result.output
    assert f"{Path(post).name} 1.00  GOOD" in result.output


def test_eval_rejects_api_key_flag(runner, post):
    """--api-key must not exist: a key argument leaks into shell history
    and the process list. Asserting rejection pins the property instead of
    just assuming nobody adds the flag back later.
    """
    result = runner.invoke(cli.main, ["eval", post, "--api-key", "sk-whatever"])
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


# --- Fix round 1 -------------------------------------------------------
#
# Finding 1: --fail-under silently passed documents nobody could score
# (NOT_PROSE from a failed gate, UNSCORED from every dimension falling
# below --min-confidence), because the old check only compared a composite
# that existed. Those are exactly the documents a CI gate most needs to
# catch, since "the model couldn't judge this" is not the same claim as
# "the model judged this and it passed."


def test_fail_under_treats_a_gate_failure_as_a_breach(runner, post, monkeypatch):
    async def not_prose(documents, prof, **kwargs):
        answers = {**ANSWERS, "is_finished_prose": {"type": "noul", "noul": 0.0}}
        return [evaluate.Outcome(document=d, answers=answers, model="jev-1.13.0")
                for d in documents]

    monkeypatch.setenv(config.API_KEY_ENV, "sk-test")
    monkeypatch.setattr(cli.evaluate, "evaluate_documents", not_prose)
    result = runner.invoke(cli.main, ["eval", post, "--fail-under", "0.9", "--no-cache"])
    assert result.exit_code == 1
    assert "NOT_PROSE" in result.output


def test_fail_under_treats_unscored_as_a_breach(runner, post, monkeypatch):
    async def unscored(documents, prof, **kwargs):
        # Gate passes, but every dimension's confidence sits below
        # --min-confidence's default of 0.6, so none of them are included
        # in the weighted mean and the composite comes back None.
        answers = {
            k: ({**v, "confidence": 0.0} if k != "is_finished_prose" else v)
            for k, v in ANSWERS.items()
        }
        return [evaluate.Outcome(document=d, answers=answers, model="jev-1.13.0")
                for d in documents]

    monkeypatch.setenv(config.API_KEY_ENV, "sk-test")
    monkeypatch.setattr(cli.evaluate, "evaluate_documents", unscored)
    result = runner.invoke(cli.main, ["eval", post, "--fail-under", "0.9", "--no-cache"])
    assert result.exit_code == 1
    assert "UNSCORED" in result.output


def test_fail_under_still_passes_a_document_above_threshold(runner, post, stub_api):
    """Control for the two tests above: a real, passing composite must
    still exit 0. Proves the NOT_PROSE/UNSCORED fix didn't turn
    --fail-under into an unconditional failure.
    """
    result = runner.invoke(cli.main, ["eval", post, "--fail-under", "0.5", "--no-cache"])
    assert result.exit_code == 0


# Without --fail-under, thresholds are opt-in entirely: a document that
# couldn't be scored must still exit 0, exactly like a low-scoring one does
# (test_eval_exits_zero_without_a_threshold). Nothing pinned this direction
# after the round-1 fix -- drop the `fail_under is not None` guard in
# _exit_code later and these two would start failing a plain run on an
# UNSCORED/NOT_PROSE document while the rest of the suite stayed green.


def test_unscored_exits_zero_without_a_threshold(runner, post, stub_api, monkeypatch):
    async def unscored(documents, prof, **kwargs):
        answers = {
            k: ({**v, "confidence": 0.0} if k != "is_finished_prose" else v)
            for k, v in ANSWERS.items()
        }
        return [evaluate.Outcome(document=d, answers=answers, model="jev-1.13.0")
                for d in documents]

    monkeypatch.setattr(cli.evaluate, "evaluate_documents", unscored)
    result = runner.invoke(cli.main, ["eval", post, "--no-cache"])
    assert result.exit_code == 0
    assert "UNSCORED" in result.output


def test_gate_failure_exits_zero_without_a_threshold(runner, post, stub_api, monkeypatch):
    async def not_prose(documents, prof, **kwargs):
        answers = {**ANSWERS, "is_finished_prose": {"type": "noul", "noul": 0.0}}
        return [evaluate.Outcome(document=d, answers=answers, model="jev-1.13.0")
                for d in documents]

    monkeypatch.setattr(cli.evaluate, "evaluate_documents", not_prose)
    result = runner.invoke(cli.main, ["eval", post, "--no-cache"])
    assert result.exit_code == 0
    assert "NOT_PROSE" in result.output


# Finding 2: `lint` only exited 2 when EVERY path failed to load
# (`if failures and not documents`), so one typo'd path alongside a good
# one reported success. `eval` was already stricter -- any failure exits
# 2 regardless of the rest of the run -- so the same mistake was caught in
# one command and silently swallowed in the other.


def test_lint_exits_two_on_an_unreadable_source_even_with_a_good_one(
    runner, post, monkeypatch
):
    # No filename check here: Rich folds a long tmp_path at its fixed
    # 80-column width, and where the fold lands shifts with pytest's tmpdir
    # slug (which changes with e.g. the test's own name). The exit code and
    # the error text below already carry what this test claims to pin.
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    result = runner.invoke(cli.main, ["lint", post, "missing.md"])
    assert result.exit_code == 2
    assert "not found" in result.output


# --- Fix wave, finding 1 -------------------------------------------------
#
# Scoring and rendering run after asyncio.run, at cli.py:147-159, with no
# isolation of their own -- evaluate.py's per-document try/except only
# covers the network call. A malformed answer payload (a null score, a null
# gate noul, a non-numeric probability key) used to raise uncaught, lose
# every other document's already-computed result, and exit 1 -- the code
# README.md documents as "a threshold was breached," not a crash.


def test_a_null_bearing_answer_does_not_crash_the_whole_run(runner, tmp_path, monkeypatch):
    """End to end: a null score in one document's payload must not lose the
    other document's result or disguise a crash as exit 1. With the
    composed fix (scoring.py's own coercion handles the null gracefully),
    the affected dimension is flagged needs_review, the document still
    scores from its other nine dimensions, and a plain run with no
    --fail-under exits 0 exactly like any other clean run -- not 1 (the old
    crash) and not 2 (which is what it would exit if only cli.py's outer
    guard, and not scoring.py's own fix, were catching this).
    """
    good_path = tmp_path / "good.md"
    good_path.write_text(f"# Good\n\n{PROSE}\n")
    bad_path = tmp_path / "bad.md"
    bad_path.write_text(f"# Bad\n\n{PROSE}\n")

    async def fake(documents, prof, **kwargs):
        outcomes = []
        for d in documents:
            answers = ANSWERS
            if d.id == str(bad_path):
                answers = {**ANSWERS,
                           "active_voice": {**ANSWERS["active_voice"], "score": None}}
            outcomes.append(evaluate.Outcome(
                document=d, answers=answers, model="jev-1.13.0",
                usage={"input_tokens": 100, "output_tokens": 10}))
        return outcomes

    monkeypatch.setattr(cli.evaluate, "evaluate_documents", fake)
    monkeypatch.setenv(config.API_KEY_ENV, "sk-test")

    result = runner.invoke(cli.main, ["eval", str(good_path), str(bad_path), "--no-cache"])

    assert result.exception is None
    assert result.exit_code == 0
    assert good_path.name in result.output
    assert bad_path.name in result.output
    assert "review" in result.output  # the null-scored dimension is flagged


def test_a_scoring_exception_for_one_document_does_not_kill_the_run(
    runner, post, tmp_path, monkeypatch
):
    """Layer test for cli.py's own outer guard, independent of scoring.py's
    coercion fix: force scoring.score_document itself to raise for one
    document (something scoring.py's own fix can't be expected to
    anticipate every case of) and confirm the other document still reports
    and the run doesn't exit 1 as if a threshold had been breached.
    """
    other_path = tmp_path / "other.md"
    other_path.write_text(f"# Other\n\n{PROSE}\n")

    async def fake(documents, prof, **kwargs):
        return [evaluate.Outcome(document=d, answers=ANSWERS, model="jev-1.13.0",
                                  usage={"input_tokens": 100, "output_tokens": 10})
                for d in documents]

    monkeypatch.setattr(cli.evaluate, "evaluate_documents", fake)
    monkeypatch.setenv(config.API_KEY_ENV, "sk-test")

    real_score_document = cli.scoring.score_document

    def flaky(*, document, **kwargs):
        if document.id == post:
            raise RuntimeError("boom during scoring")
        return real_score_document(document=document, **kwargs)

    monkeypatch.setattr(cli.scoring, "score_document", flaky)

    result = runner.invoke(cli.main, ["eval", post, str(other_path), "--no-cache"])

    # exit 2 (operational error) is the correct, honest code here -- one
    # document genuinely failed to score. exit 1 would be the bug: a crash
    # disguised as "a threshold was breached" (README.md's own words for 1),
    # which is what happens pre-fix when the exception isn't caught at all
    # and CliRunner's default handling reports it as exit 1.
    assert result.exit_code == 2
    assert not isinstance(result.exception, RuntimeError)
    assert "boom during scoring" in result.output
    assert other_path.name in result.output
    assert "GOOD" in result.output

