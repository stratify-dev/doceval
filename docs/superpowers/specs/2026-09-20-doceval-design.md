# doceval design

Date: 2026-09-20
Status: approved for planning

## Purpose

A CLI scoring written content on house style, editorial quality, and audience fit.
It accepts local files and web URLs, evaluates each document against a YAML profile,
and reports per-dimension scores with the probability distribution behind each one.

Target content: blog posts, documentation, marketing copy, READMEs.

## Approach

Two evaluation paths, merged at report time.

```
sources ──> lint   (regex, local, free)    ──┐
        └─> judge  (Jev, one request/doc)  ──┴─> combine ──> report
```

Mechanical rules stay in code. A regex finds an em dash exactly, names the line,
costs nothing, and never contradicts itself. Semantic dimensions go to Jev, which
supplies the reading no regex reaches.

The word `it` sits in the house banned list and demonstrates the split. A regex on
`it` flags every legitimate use and drowns the report. The real target is the vague
referent pointing at nothing definite, which is a judgment. It becomes the
`vague_referents` dimension rather than a lint rule.

## Module layout

```
doceval/
  pyproject.toml
  README.md
  profiles/house-style.yaml
  src/doceval/
    sources.py    # path or URL -> Document
    profile.py    # YAML -> validated Profile
    lint.py       # text -> Violation[]
    questions.py  # Profile -> typesafe_sdk question map
    evaluate.py   # async fan-out, one request per document
    cache.py      # answers keyed by content hash + profile hash
    scoring.py    # normalize, weight, composite, confidence gate
    report.py     # table | json | markdown renderers
    cli.py        # wiring only
  tests/
```

Every module below `cli.py` is importable and testable on its own. `lint.py` and
`scoring.py` are pure functions with no I/O.

## Sources

Both origins resolve to one shape, so nothing downstream knows the difference.

```python
@dataclass(frozen=True)
class Document:
    id: str                      # path or URL, for display
    title: str
    text: str                    # clean prose, ready to be state
    origin: Literal["file", "url"]
    fetched_at: datetime | None
```

A path reads from disk and strips YAML frontmatter. A URL fetches with `httpx` and
extracts article prose with `trafilatura`, which removes navigation, advertising,
cookie banners, and footers, and recovers the title. Mixed arguments work in one run.

## State shape

One document becomes one state object. The audience appears once per request rather
than once per dimension.

```python
state = {
    "document": {"path": "posts/typesafe.md", "title": "...", "text": "..."},
    "audience": "Working developers who skim before they read.",
}
```

Questions reference it by backticked path, the documented way to point at nested
state: ``"Does the document's technical level match `audience`?"``

## Profile format

A profile is YAML. It carries the audience, one optional gate, and the weighted
dimensions.

```yaml
name: house-style
audience: >
  Working developers who skim before they read. Impatient,
  allergic to filler, looking for something to apply today.

gate:
  is_finished_prose:
    type: noul
    instructions: >
      The document is finished prose written for a reader, rather than
      an outline, a stub, a changelog, or configuration notes.
    criteria:
      true: Complete sentences and paragraphs meant to be read start to finish
      false: Fragments, bare bullet lists, boilerplate, or placeholder text

dimensions:
  <id>:
    group: house_style | editorial | audience_fit
    weight: <float>
    type: score
    instructions: <string>
    levels: [<string>, ...]     # 2 to 10, ordered worst to best
```

Weights sum to 1.00. Validation rejects a profile failing any of: weights summing
outside 1.00 +/- 0.001, a Score with fewer than two or more than ten levels, a
missing `instructions`, an unknown `group`, or a duplicate dimension id.

The gate rides in the same request as the dimensions rather than preceding it. All
questions evaluate in parallel against one state, so asking the gate speculatively
alongside the scores costs one round trip instead of two. When the gate returns
below 0.5, code discards the dimension answers and reports `not prose`. This is the
documented speculative fan-out pattern: state the premise, ask everything at once,
and let code decide which answers apply.

## Default profile: ten dimensions

Group weights: house style 0.40, editorial 0.35, audience fit 0.25.

### House style

**active_voice** (0.10) — How consistently does the document use active voice?

1. Passive constructions dominate; actors are missing from most sentences
2. Passive voice appears throughout, including in the main claims
3. A mix; active in places, passive whenever the subject is awkward
4. Nearly all active, with isolated passive sentences
5. Active voice throughout; every sentence names who does what

**sentence_impact** (0.10) — How short and direct are the sentences? Long sentences
carrying several clauses score low. Short sentences landing one idea score high.

1. Long compound sentences with stacked clauses; hard to follow once
2. Mostly long sentences; the reader re-reads to find the point
3. Mixed lengths with no pattern; some sentences run past their point
4. Mostly short and direct, with a few overlong sentences
5. Short sentences, one idea each, varied only for rhythm

**vague_referents** (0.08) — Do pronouns and demonstratives point at something
definite? Flag sentences opening with a pronoun whose referent is unclear.

1. Referents are unclear throughout; the reader guesses constantly
2. Several sentences open with a pronoun pointing at nothing definite
3. Occasional vague reference, recoverable from context
4. Referents are clear except in one or two places
5. Every pronoun points at a named thing in the previous clause

**adjective_restraint** (0.07) — Do adjectives and adverbs carry information, or pad
the sentence? Intensifiers adding no meaning score low.

1. Stacked intensifiers and decorative adjectives in most sentences
2. Frequent padding words; removing them would change nothing
3. Some unnecessary modifiers, mostly in transitions and openings
4. Modifiers nearly always carry information
5. Every adjective and adverb changes the meaning of its sentence

**cliche_free** (0.05) — Does the writing avoid metaphors, stock phrases, and
borrowed imagery in favor of plain description?

1. Built on extended metaphor and stock phrases throughout
2. Several clichés and figurative flourishes carrying the main points
3. Occasional stock phrasing, mostly in openings and closings
4. Plain description with one or two borrowed phrases
5. Plain description start to finish; no metaphor, no stock phrasing

### Editorial quality

**opening_strength** (0.10) — Does the opening give a skimming reader a reason to
continue, or warm up before saying anything?

1. Several sentences of throat-clearing before the subject appears
2. Opens with background the reader does not need yet
3. States the subject, without a reason to keep reading
4. States the subject and the stake in the first two sentences
5. First sentence lands the point; the reader knows immediately what they get

**structure_flow** (0.15) — Does the document move in a deliberate order, with each
section following from the one before?

1. Sections appear in arbitrary order; ideas repeat and double back
2. Loose grouping; the reader reorders mentally to follow
3. Reasonable order with one or two jumps or orphaned sections
4. Clear progression; transitions carry the reader between sections
5. Each section earns its position; removing one would break the argument

**concision** (0.10) — How much of the document survives editing? Repetition,
restatement, and filler paragraphs score low.

1. Half the text restates points already made
2. Frequent repetition and paragraphs adding nothing new
3. Some redundancy, mainly in summaries and section openings
4. Tight, with isolated sentences worth cutting
5. Nothing removable without losing meaning

### Audience fit

**technical_level** (0.13) — Does the assumed knowledge match `audience`? Both
over-explaining and under-explaining score low.

1. Badly mismatched; either explains the obvious or assumes unstated expertise
2. Frequently off target, forcing the reader to skip or look things up
3. Roughly right, with sections pitched too high or too low
4. Well matched, with isolated lapses
5. Pitched exactly at `audience` throughout

**takeaway_clarity** (0.12) — After one read, would a reader matching `audience`
know what to do differently?

1. No takeaway; the reader finishes with nothing to apply
2. An implied takeaway the reader must reconstruct
3. A takeaway stated once, buried in the middle
4. A clear takeaway, stated and supported
5. The takeaway is unmistakable and the reader knows the first step

## Extending the profile

Adding a dimension takes one block. Adding `direct_address`, for instance:

```yaml
  direct_address:
    group: house_style
    weight: 0.05
    type: score
    instructions: >
      Does the document address the reader directly as "you", rather than
      describing an abstract third party or hiding behind passive phrasing?
    levels:
      - Never addresses the reader; abstract third person throughout
      - Occasional second person, mostly abstract
      - Mixed; addresses the reader in some sections only
      - Addresses the reader directly in most sections
      - Speaks to the reader throughout
```

Remaining weights rescale so the total returns to 1.00. Validation reports the
required adjustment when a profile misses the target.

## Lint rules

Rules run over prose only. Fenced code blocks, inline code spans, and YAML
frontmatter are excluded before matching, since code legitimately contains
semicolons and asterisks.

| Rule | Severity | Match | Suggestion |
|---|---|---|---|
| `em_dash` | error | `—` or `–` | comma, period, or parentheses |
| `semicolon` | error | `;` in prose | period, or split the sentence |
| `asterisk` | error | `*` as emphasis or bullet | `-` for bullets, rewrite for emphasis |
| `hashtag` | error | `#word` not at line start | remove |
| `banned_word` | error | distinctive entries: `utilize`, `delve`, `tapestry`, `realm`, `pivotal`, `groundbreaking`, and the rest of the list | per-word replacement |
| `common_word` | warning | `that`, `can`, `may`, `just`, `could`, `very`, `really` | usually removable; needs a human read |
| `setup_language` | error | `in conclusion`, `in summary`, `in closing`, `in a world where` | delete the phrase |
| `not_just` | error | `not just ... but also` | state the point directly |

Matching is case-insensitive on word boundaries. `it` is deliberately absent and
handled by the `vague_referents` dimension.

Severity splits the list because distinctive words are unambiguous while common
words need judgment. Errors count against the report; warnings list separately.

## Scoring

1. Each Score normalizes to 0 through 1 as `score / (len(levels) - 1)`
2. The composite is the weighted sum of normalized scores
3. A dimension whose confidence falls below `--min-confidence` is marked
   `needs review` and excluded from the composite; remaining weights rescale
4. The composite band sets the verdict: 0.80 and up `GOOD`, 0.60 and up `FAIR`,
   below 0.60 `WEAK`
5. Lint errors report alongside the composite without entering it, since one banned
   word is a fix rather than a quality signal

Raw answers and policy stay separate. Weights apply at render time, so retuning a
weight re-ranks the corpus from cache with no API calls.

## Caching

Answers persist in `.doceval-cache/`, keyed on `sha256(document text) +
sha256(profile questions)`. Changing a weight leaves the key untouched and costs
nothing. Editing a document or a question's wording invalidates its entry. A
republished URL re-evaluates automatically because its extracted text changed.
`--no-cache` forces fresh requests.

## Report

Rich renders everything. Sparklines come from `▁▂▃▄▅▆▇█`, mapping each level's
probability to a height.

### Per document

Dimensions nest under their group, and each group shows its own rollup.

```
 posts/typesafe-cli.md                                       0.81  GOOD
 │
 ├ house style                                               0.79
 │ ├ active voice        ███████████████████░  0.88  ▁▁▂▇█  0.92
 │ ├ sentence impact     ████████████░░░░░░░░  0.61  ▁▃█▄▁  0.71
 │ ├ vague referents     █████████████████░░░  0.85  ▁▁▂▆█  0.88
 │ ├ adjective restraint ███████████████░░░░░  0.75  ▁▂▇█▃  0.74
 │ └ cliche free         ████████████████████  1.00  ▁▁▁▁█  0.97
 │
 ├ editorial                                                 0.84
 │ ├ opening strength    ██████████████░░░░░░  0.70  ▁▂█▆▁  0.69
 │ ├ structure & flow    ██████████████████░░  0.90  ▁▁▁▂█  0.95
 │ └ concision           █████████████████░░░  0.85  ▁▁▃▇█  0.91
 │
 ├ audience fit                                              0.78
 │ ├ technical level     ███████░░░░░░░░░░░░░  0.35  ▃█▃▁▁  0.44  ⚠ review
 │ └ takeaway clarity    ███████████████████░  0.95  ▁▁▁▁█  0.96
 │
 └ lint                  3 errors · 2 warnings
     line 14  banned word   "utilize"        → use
     line 27  em dash       "results — and"  → comma or period
     line 31  semicolon     "fast; the"      → period
```

Three numbers per dimension row: normalized score, distribution across levels,
confidence. A group rollup is the weighted mean of its dimensions, rescaled within
the group. `--compact` collapses the tree to the three group rows and the lint count.
The distribution is the reason to use a System One model. A 0.35 spread across the
bottom three levels differs from a 0.35 concentrated on one, and the mean hides it.

### Corpus

```
╭─ 3 documents · house-style ──────────────────────────────────────────╮
│                                                                      │
│  posts/typesafe-cli.md    ████████████████▏      0.81   ✓            │
│  example.com/article      ███████████▍           0.57   ⚠ 2 review   │
│  README.md                ████████▊              0.44   ✗ below 0.70 │
│                                                                      │
│  group averages                                                      │
│  house style  ▇▇▇▇▇▇▇▆  0.74   editorial  ▇▇▇▇▇▇▅  0.71             │
│  audience fit ▇▇▇▃      0.39                                        │
│                                                                      │
│  weakest dimensions  technical level 0.39 · sentence impact 0.52     │
│                                                                      │
│  jev-1.13.0 · 14.2k tokens · $0.0006 · 2.1s · 1 cached               │
╰──────────────────────────────────────────────────────────────────────╯
```

Group averages point at the fix. A weak group across every document indicts the
writing target rather than one draft. The weakest-dimension line names the two
worst offenders by mean across the corpus, so the next edit has somewhere to start.

### Progress

```
 ⠹ evaluating   ━━━━━━━━━━━━━━━━╺━━━━━━━━   2/3   fetching example.com…
 ✓ posts/typesafe-cli.md   0.81    1.4s
 ⠴ README.md               judging…
```

Color follows the band: green at 0.80, amber at 0.60, red below. Low confidence dims
the bar so a shaky number never reads as solid.

Output degrades correctly. Rich detects a missing TTY and drops animation, color,
and box drawing when piped. `--format json` stays machine-clean. `--no-color` forces
plain text.

## Configuration

Nothing about the tool is specific to one machine or one account. Clone it, export
a key, run it.

### Credentials

The API key comes from the environment and only from the environment. The SDK reads
`TYPESAFE_API_KEY` on its own, so no key ever appears in source, in a profile, or in
a committed file.

```bash
export TYPESAFE_API_KEY=sk-...        # get one at console.typesafe.ai/keys
doceval eval post.md
```

A `.env` file in the project root loads at startup through `python-dotenv` for local
convenience. A real environment variable always wins over a `.env` entry. `.env` is
gitignored; `.env.example` is committed and lists every variable with empty values,
so a new user copies it and fills one line.

There is deliberately no `--api-key` flag. A key passed as an argument leaks into
shell history and into the process list, where any other user on the machine reads
it with `ps`.

### Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `TYPESAFE_API_KEY` | yes | — | Authenticates every request. Read by the SDK |
| `TYPESAFE_BASE_URL` | no | `https://api.typesafe.ai` | Point at a proxy or a mock server |
| `TYPESAFE_DEFAULT_MODEL` | no | `jev-latest` | Pin a version such as `jev-1.13.0` |
| `TYPESAFE_LOG_LEVEL` | no | unset | `debug` logs request and response bodies |

Pinning matters for thresholds. An alias moves when a release ships, so a team
tuning `--min-confidence` against a specific version sets `TYPESAFE_DEFAULT_MODEL`
to that versioned id and upgrades deliberately. Every report records the model id
the API returned, so a shifted score is traceable to a model change.

### Missing key

A missing key fails before any work, with a message naming the variable and the URL
to create one. It never surfaces as a 401 after a fetch already ran.

```
✗ TYPESAFE_API_KEY is not set

  export TYPESAFE_API_KEY=sk-...     create a key at console.typesafe.ai/keys
  or copy .env.example to .env and fill it in

  doceval lint needs no key and works now.
```

`doceval lint` and `doceval profiles` run with no key at all, so someone evaluating
the tool sees it work before signing up for anything.

### Secret hygiene

The key never reaches disk or output. `TYPESAFE_LOG_LEVEL=debug` logs request bodies
but the SDK redacts authorization headers. `--dump-text` writes extracted prose
only. The JSON report carries documents, answers, and the model id, and never the
environment.

## CLI surface

```bash
doceval eval content/**/*.md --profile house-style
doceval eval post.md https://example.com/article
doceval eval post.md --format json | jq '.documents[0].dimensions'
doceval lint content/**/*.md          # no API key, no network
doceval profiles                      # list bundled profiles
```

`eval` flags: `--profile`, `--format table|json|markdown`, `--min-confidence`
(default 0.6), `--fail-under` (no default), `--fail-on-lint` (off by default),
`--concurrency` (default 8), `--no-cache`, `--timeout` (default 20),
`--min-words` (default 150), `--dump-text DIR`, `--compact`, `--no-color`.

Exit codes: `0` success, `1` a threshold was breached, `2` operational error.

Thresholds are opt-in. Without `--fail-under` or `--fail-on-lint`, a run reports and
exits `0` whatever the scores, which is what you want interactively. Passing
`--fail-under 0.7` exits `1` when any document scores below it. Passing
`--fail-on-lint` exits `1` on any lint error, ignoring warnings. Together they make
the tool a CI gate on a content repo.

## Error handling

| Failure | Behavior |
|---|---|
| Missing `TYPESAFE_API_KEY` | Fail before any work, point at the console URL. `lint` still runs |
| 429 or 529 | SDK `RetryPolicy` with backoff, honoring `retry-after` |
| 422 | Print the offending question id and field. A profile bug, not a runtime one |
| Document over the 32k state budget | Reject with the measured size and the limit |
| URL fetch failure | Mark one document errored, continue the run |
| Extraction under `--min-words` | Reject before spending tokens |
| Gate returns false | Skip scoring, report `not prose`, spend nothing further |

Profile validation runs before the first request, so a bad weight fails immediately
rather than surfacing as a 422 after nine other documents were paid for.

The empty-extraction guard matters most. A JavaScript-rendered page yields a couple
hundred characters of navigation text, and Jev would score the fragment with high
confidence. The word-count floor stops meaningless numbers at the source.
`--dump-text` writes extracted prose beside the report so a suspicious score is
checked against the exact state sent.

## Testing

Unit tests need no network.

- `lint.py` — pure. Table-driven cases per rule, including code-block and
  frontmatter exclusion
- `scoring.py` — pure. Recorded answer fixtures in, composite out. Covers
  normalization, weighting, confidence gating with rescale, and flat distributions
- `profile.py` — valid and deliberately broken YAML fixtures
- `sources.py` — saved HTML fixtures and a temp directory; no live fetching
- `report.py` — renders fixtures to a fixed-width Rich console and asserts on text,
  pinning sparkline mapping and bar widths
- `cache.py` — key stability across weight edits, invalidation on text edits

One integration test hits the live API behind an env guard, skipped by default. It
evaluates a short known document and asserts the response shape, catching real API
drift rather than mocking it away.

## Out of scope

- Crawling. One URL is one document. A sitemap is a different tool
- Chunking documents over the state budget. Oversized documents are rejected
- Rewriting or suggesting replacement prose. This tool scores and reports
- Line-level evidence for dimension scores. A later version could score line ids
  the way the semantic-find cookbook does

## Dependencies

`typesafe-sdk`, `httpx`, `trafilatura`, `pyyaml`, `click`, `rich`, `python-dotenv`.
Python 3.11 and up, managed with uv.

Setup for a new user is three commands, with no account needed for the first one:

```bash
uv sync
doceval lint README.md                 # works immediately, no key
cp .env.example .env && $EDITOR .env   # then doceval eval
```
