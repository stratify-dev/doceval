import os
import stat
import sys

import pytest

from doceval import cache

PAYLOAD = {
    "model": "jev-1.13.0",
    "answers": {"a": {"type": "noul", "noul": 1.0}},
    "usage": {"input_tokens": 100, "output_tokens": 10},
}


def test_key_is_stable():
    assert cache.cache_key("text", "fp") == cache.cache_key("text", "fp")


def test_key_changes_with_text():
    assert cache.cache_key("a", "fp") != cache.cache_key("b", "fp")


def test_key_changes_with_fingerprint():
    assert cache.cache_key("text", "fp1") != cache.cache_key("text", "fp2")


def test_round_trip(tmp_path):
    key = cache.cache_key("text", "fp")
    cache.write(tmp_path, key, PAYLOAD)
    assert cache.read(tmp_path, key)["answers"] == PAYLOAD["answers"]


def test_read_missing_returns_none(tmp_path):
    assert cache.read(tmp_path, "nope") is None


def test_read_corrupt_entry_returns_none(tmp_path):
    key = cache.cache_key("text", "fp")
    cache.write(tmp_path, key, PAYLOAD)
    (tmp_path / f"{key}.json").write_text("{ not json")
    assert cache.read(tmp_path, key) is None


def test_write_creates_the_directory(tmp_path):
    target = tmp_path / "nested" / "cache"
    cache.write(target, "k", PAYLOAD)
    assert (target / "k.json").is_file()


def test_write_records_a_timestamp(tmp_path):
    cache.write(tmp_path, "k", PAYLOAD)
    assert "cached_at" in cache.read(tmp_path, "k")


# --- Extra property tests: each pins behavior that would otherwise fail
# silently, per the task brief's warning about the previous six tasks. ---


def test_null_separator_prevents_collisions():
    """Without the 0x00 separator, ("ab", "c") and ("a", "bc") would collide
    because sha256("ab" + "c") == sha256("a" + "bc") as plain concatenation.
    """
    assert cache.cache_key("ab", "c") != cache.cache_key("a", "bc")


def test_write_leaves_no_tmp_file(tmp_path):
    key = cache.cache_key("text", "fp")
    cache.write(tmp_path, key, PAYLOAD)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []
    assert (tmp_path / f"{key}.json").is_file()


def test_read_ignores_stray_tmp_file_from_a_simulated_crash(tmp_path):
    key = cache.cache_key("text", "fp")
    cache.write(tmp_path, key, PAYLOAD)

    # Simulate a crash mid-write: a leftover .tmp file with different,
    # possibly incomplete content must never be mistaken for the real entry.
    stray = tmp_path / f"{key}.json.tmp"
    stray.write_text('{"incomplete')

    result = cache.read(tmp_path, key)
    assert result is not None
    assert result["answers"] == PAYLOAD["answers"]
    assert stray.is_file()  # read() does not touch or clean up the stray file


@pytest.mark.parametrize(
    "contents",
    [
        pytest.param("{ not json, truncated mid", id="truncated_json"),
        pytest.param("", id="empty_file"),
        pytest.param("[1, 2, 3]", id="valid_json_not_an_object"),
        pytest.param('"just a string"', id="valid_json_scalar"),
    ],
)
def test_corrupt_or_non_object_entries_are_misses(tmp_path, contents):
    key = cache.cache_key("text", "fp")
    (tmp_path / f"{key}.json").write_text(contents)
    assert cache.read(tmp_path, key) is None


def test_invalid_utf8_bytes_are_a_miss(tmp_path):
    """A cache file with undecodable bytes must miss, not raise.

    Without UnicodeDecodeError in read()'s except clause, one corrupt file
    aborts the whole run instead of costing a single fresh request. This
    needs a raw bytes write; a text-mode write cannot reproduce the failure.
    """
    key = cache.cache_key("text", "fp")
    (tmp_path / f"{key}.json").write_bytes(b"\xff\xfe\x00not utf-8")
    assert cache.read(tmp_path, key) is None


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="permission bits are not enforced for root or reliably on this platform",
)
def test_unwritable_directory_degrades_to_no_caching(tmp_path):
    target = tmp_path / "readonly"
    target.mkdir()
    original_mode = target.stat().st_mode
    target.chmod(stat.S_IRUSR | stat.S_IXUSR)  # read + execute, no write
    try:
        key = cache.cache_key("text", "fp")
        cache.write(target, key, PAYLOAD)  # must not raise
        assert cache.read(target, key) is None
    finally:
        target.chmod(original_mode)


def test_round_trip_preserves_floats_nested_dicts_unicode_and_empty_dict(tmp_path):
    payload = {
        "model": "jev-1.13.0",
        "answers": {
            "clarity": {"type": "noul", "noul": 0.8333333333333334},
            "empty": {},
            "unicode": {"note": "café — 日本語 — emoji 🎯"},
            "nested": {"a": {"b": {"c": [1, 2.5, "x", None, True]}}},
        },
        "usage": {"input_tokens": 100, "output_tokens": 10, "cost": 0.0001234},
    }
    key = cache.cache_key("text", "fp")
    cache.write(tmp_path, key, payload)
    result = cache.read(tmp_path, key)
    assert result is not None
    for field in ("model", "answers", "usage"):
        assert result[field] == payload[field]
