# doceval

Score written content on house style, editorial quality, and audience fit.

Mechanical rules run as regex, locally and free, and report an exact line. Ten
semantic dimensions go to [TypeSafe](https://typesafe.ai)'s Jev model in one
request per document. Your code owns the weights.

## Install

```bash
uv sync
```

## Run

`lint` needs no account and no network for local files:

```bash
doceval lint README.md
```

`eval` needs an API key, read from the environment:

```bash
export TYPESAFE_API_KEY=sk-...        # create one at console.typesafe.ai/keys
doceval eval posts/*.md
doceval eval https://example.com/article
doceval eval post.md --format json | jq '.documents[0].groups'
```

Or copy `.env.example` to `.env` and fill in the key.

## What you get

```
 posts/typesafe-cli.md                                       0.81  GOOD
 │
 ├ house style                                               0.79
 │ ├ active voice        ███████████████████░  0.88  ▁▁▂▇█  0.92
 │ ├ sentence impact     ████████████░░░░░░░░  0.61  ▁▃█▄▁  0.71
 │ └ cliche free         ████████████████████  1.00  ▁▁▁▁█  0.97
 │
 └ lint                  3 errors · 2 warnings
     line 14  banned word   "utilize"        → use
```

Three numbers per dimension: the normalized score, the probability
distribution across levels, and the confidence. The distribution is the point.
A 0.61 spread across three levels means something different from a 0.61 sitting
on one, and a mean hides it. A dimension below `--min-confidence` is flagged
`⚠ review` and leaves the composite rather than dragging it.

## Profiles

A profile is YAML: the audience, a gate, and weighted dimensions.

```yaml
name: house-style
audience: Working developers who skim before they read.
dimensions:
  active_voice:
    group: house_style
    weight: 0.10
    type: score
    instructions: How consistently does the document use active voice?
    levels:
      - Passive constructions dominate; actors are missing from most sentences
      - ...
```

Copy the bundled rubric (`doceval profiles` lists what ships), edit it, and pass
`--profile my-profile.yaml`.
Weights sum to 1.00. Retuning a weight re-ranks a corpus from cache with no API
calls, because nothing sent to the model changed.

## CI

Thresholds are opt-in. A plain run always exits 0.

```bash
doceval eval content/**/*.md --fail-under 0.7 --fail-on-lint
```

Exit codes: `0` success, `1` a threshold was breached (a low score, or a
document `--fail-under` couldn't score at all), `2` operational error.

## Timeouts

`eval` has two separate timeout flags for two separate HTTP calls:

| Flag | Bounds | Default |
|---|---|---|
| `--timeout` | Fetching a URL source | 20s |
| `--api-timeout` | The TypeSafe API request per document, per attempt | 60s |

They don't share a value on purpose. A single `eval` request can carry up to
28,000 tokens of document state plus eleven questions, which needs longer than
a page fetch does. Raise `--api-timeout` for very large documents or a slower
model. `RetryPolicy`'s own `timeout` (not a flag) is the separate total retry
budget across every attempt and its backoff, not a per-attempt timeout.

## Configuration

| Variable | Required | Default |
|---|---|---|
| `TYPESAFE_API_KEY` | yes | — |
| `TYPESAFE_BASE_URL` | no | `https://api.typesafe.ai` |
| `TYPESAFE_DEFAULT_MODEL` | no | `jev-latest` |
| `TYPESAFE_LOG_LEVEL` | no | unset |

There is no `--api-key` flag on purpose. A key passed as an argument lands in
shell history and in the process list.

## Tests

```bash
uv run pytest                      # offline, no key needed
DOCEVAL_LIVE_API=1 uv run pytest   # adds one real API request
```
