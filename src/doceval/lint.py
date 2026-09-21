"""Mechanical style rules.

Regex owns what regex does well: an em dash is an em dash, and the match
carries a line, a column, and a replacement. Judgments live in the profile
dimensions instead.

The word `it` is deliberately absent from every table here. A regex on `it`
flags every legitimate use and drowns the report. The real target is the vague
referent, which is a judgment, so it belongs to the `vague_referents` dimension.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

# Distinctive words. A match is unambiguous, so these are errors.
BANNED_WORDS: dict[str, str] = {
    "literally": "delete",
    "delve": "dig, go into",
    "embark": "start",
    "enlightening": "useful",
    "esteemed": "delete",
    "shed light": "explain, show",
    "craft": "build, write",
    "crafting": "building, writing",
    "realm": "area, field",
    "game-changer": "name the actual change",
    "unlock": "open, enable",
    "skyrocket": "rise sharply",
    "skyrocketing": "rising sharply",
    "abyss": "delete",
    "revolutionize": "change",
    "disruptive": "delete",
    "utilize": "use",
    "utilizing": "using",
    "dive deep": "examine",
    "tapestry": "delete",
    "illuminate": "show",
    "unveil": "show, release",
    "pivotal": "key, central",
    "intricate": "detailed, complex",
    "elucidate": "explain",
    "hence": "so",
    "furthermore": "and, also",
    "however": "but, or start a new sentence",
    "harness": "use",
    "exciting": "delete",
    "groundbreaking": "delete",
    "cutting-edge": "delete",
    "remarkable": "delete",
    "remains to be seen": "delete",
    "glimpse into": "look at",
    "navigating": "working with",
    "landscape": "field, market",
    "stark": "sharp, plain",
    "testament": "proof, evidence",
    "moreover": "also",
    "boost": "raise, increase",
    "opened up": "opened",
    "powerful": "name the capability",
    "inquiries": "questions",
    "ever-evolving": "changing",
    "imagine": "delete",
    "discover": "find",
    "not alone": "delete",
}

# Common words needing a human read. Usually removable, sometimes correct.
COMMON_WORDS: dict[str, str] = {
    "can": "usually removable",
    "may": "usually removable",
    "just": "usually removable",
    "that": "usually removable",
    "very": "delete",
    "really": "delete",
    "could": "usually removable",
    "maybe": "usually removable",
    "actually": "delete",
    "certainly": "delete",
    "probably": "delete",
    "basically": "delete",
}

SETUP_PHRASES: dict[str, str] = {
    "in conclusion": "delete the phrase",
    "in summary": "delete the phrase",
    "in closing": "delete the phrase",
    "in a world where": "delete",
}

_FENCED_BLOCK = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_EM_DASH = re.compile(r"[—–]")
_SEMICOLON = re.compile(r";")
_ASTERISK = re.compile(r"\*{1,3}[^*\n]+\*{1,3}|\*+")
# Not preceded by a word character, /, :, ., #, or -, so a URL fragment
# anchor (...#configuration) or a second heading marker (##) is left alone;
# a real hashtag is normally preceded by whitespace or line start.
_HASHTAG = re.compile(r"(?<![\w/:.#-])#\w+")
# A named or numeric HTML/XML character reference (&nbsp; &amp; &#8212;).
# Used to exempt the semicolon that closes one from the semicolon rule,
# below -- it's punctuation syntax, not prose punctuation.
_HTML_ENTITY = re.compile(r"&#?[0-9a-zA-Z]+;")
_NOT_JUST = re.compile(r"\bnot just\b[^.!?]*?\bbut also\b", re.IGNORECASE)


@dataclass(frozen=True)
class Violation:
    rule: str
    severity: str
    line: int
    column: int
    text: str
    suggestion: str


def mask_code(text: str) -> str:
    """Blank out code while preserving every offset and newline.

    Replacing code with spaces of equal length keeps line and column numbers
    exact, so a violation found after a code block still points at the right
    place in the original file.
    """
    buffer = list(text)

    def blank(start: int, end: int) -> None:
        for index in range(start, end):
            if buffer[index] != "\n":
                buffer[index] = " "

    for match in _FENCED_BLOCK.finditer(text):
        blank(match.start(), match.end())

    for match in _INLINE_CODE.finditer("".join(buffer)):
        blank(match.start(), match.end())

    return "".join(buffer)


def lint(text: str) -> list[Violation]:
    """Return every mechanical style violation, ordered by position."""
    masked = mask_code(text)
    line_starts = _line_starts(masked)
    violations: list[Violation] = []

    def add(rule: str, severity: str, start: int, matched: str, suggestion: str) -> None:
        line, column = _position(line_starts, start)
        violations.append(Violation(rule, severity, line, column, matched, suggestion))

    for match in _EM_DASH.finditer(masked):
        add(
            "em_dash", SEVERITY_ERROR, match.start(), match.group(), "comma, period, or parentheses"
        )

    entity_semicolons = {match.end() - 1 for match in _HTML_ENTITY.finditer(masked)}
    for match in _SEMICOLON.finditer(masked):
        if match.start() in entity_semicolons:
            continue  # closes an HTML entity (&nbsp;), not prose punctuation
        add(
            "semicolon",
            SEVERITY_ERROR,
            match.start(),
            match.group(),
            "period, or split the sentence",
        )

    for match in _ASTERISK.finditer(masked):
        add(
            "asterisk",
            SEVERITY_ERROR,
            match.start(),
            match.group(),
            "'-' for bullets, rewrite for emphasis",
        )

    for match in _HASHTAG.finditer(masked):
        if _at_line_start(masked, line_starts, match.start()):
            continue  # a Markdown heading, not a hashtag
        add("hashtag", SEVERITY_ERROR, match.start(), match.group(), "remove")

    for phrase, suggestion in SETUP_PHRASES.items():
        for match in _phrase_pattern(phrase).finditer(masked):
            add("setup_language", SEVERITY_ERROR, match.start(), match.group(), suggestion)

    for match in _NOT_JUST.finditer(masked):
        add("not_just", SEVERITY_ERROR, match.start(), match.group(), "state the point directly")

    for phrase, suggestion in BANNED_WORDS.items():
        for match in _phrase_pattern(phrase).finditer(masked):
            add("banned_word", SEVERITY_ERROR, match.start(), match.group(), suggestion)

    for phrase, suggestion in COMMON_WORDS.items():
        for match in _phrase_pattern(phrase).finditer(masked):
            add("common_word", SEVERITY_WARNING, match.start(), match.group(), suggestion)

    return sorted(violations, key=lambda v: (v.line, v.column, v.rule))


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return starts


def _position(line_starts: list[int], offset: int) -> tuple[int, int]:
    low, high = 0, len(line_starts) - 1
    while low < high:
        mid = (low + high + 1) // 2
        if line_starts[mid] <= offset:
            low = mid
        else:
            high = mid - 1
    return low + 1, offset - line_starts[low] + 1


def _at_line_start(text: str, line_starts: list[int], offset: int) -> bool:
    """True when only whitespace precedes the offset on its line."""
    line_index, _ = _position(line_starts, offset)
    start = line_starts[line_index - 1]
    return text[start:offset].strip() == ""
