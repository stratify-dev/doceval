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

    Asserting "valid JSON" alone (as in test_eval_json_output_is_valid)
    would still pass if a stray progress line landed on stdout AFTER a
    complete, separately-parseable JSON blob never printed at all being the
    bug -- json.loads on the *entire* captured output is the assertion that
    actually fails the moment anything else is mixed in, because a stray
    line anywhere makes the whole stream fail to parse as one JSON value.
    """
    result = runner.invoke(cli.main, ["eval", post, "--format", "json", "--no-cache"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["documents"][0]["id"] == post


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
