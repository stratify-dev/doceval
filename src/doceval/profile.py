"""Profile loading and validation.

A profile is data, not code. Weights, levels, and the audience live in YAML so
tuning a rubric never touches Python. Validation runs before the first request,
so a bad weight fails immediately instead of surfacing as a 422 after other
documents were already paid for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

GROUPS = ("house_style", "editorial", "audience_fit")
WEIGHT_TOLERANCE = 0.001
MIN_LEVELS = 2
MAX_LEVELS = 10
PROFILE_DIR = Path(__file__).resolve().parents[2] / "profiles"


class ProfileError(ValueError):
    """A profile failed validation. The message names the offending field."""


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader rejecting duplicate mapping keys.

    PyYAML silently keeps the last value for a repeated key, which would turn a
    duplicated dimension id into a silently dropped dimension.
    """

    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        seen: set[str] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ProfileError(f"duplicate key {key!r} in profile")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


@dataclass(frozen=True)
class Dimension:
    id: str
    group: str
    weight: float
    instructions: str
    levels: tuple[str, ...]

    @property
    def label(self) -> str:
        """Human-facing name for reports."""
        return self.id.replace("_", " ")

    @property
    def max_index(self) -> int:
        return len(self.levels) - 1


@dataclass(frozen=True)
class Gate:
    id: str
    instructions: str
    criteria: dict[str, str]


@dataclass(frozen=True)
class Profile:
    name: str
    audience: str
    dimensions: tuple[Dimension, ...]
    gate: Gate | None

    def by_group(self) -> dict[str, tuple[Dimension, ...]]:
        """Dimensions grouped, in declaration order of first appearance."""
        grouped: dict[str, list[Dimension]] = {}
        for dimension in self.dimensions:
            grouped.setdefault(dimension.group, []).append(dimension)
        return {name: tuple(items) for name, items in grouped.items()}


def bundled_profiles() -> list[str]:
    """Names of the profiles shipped with the package."""
    if not PROFILE_DIR.is_dir():
        return []
    return sorted(path.stem for path in PROFILE_DIR.glob("*.yaml"))


def load_profile(name_or_path: str | Path) -> Profile:
    """Load a bundled profile by name, or any profile by path."""
    candidate = Path(name_or_path)
    if candidate.suffix in {".yaml", ".yml"} or candidate.is_file():
        path = candidate
    else:
        path = PROFILE_DIR / f"{name_or_path}.yaml"

    if not path.is_file():
        available = ", ".join(bundled_profiles()) or "none"
        raise ProfileError(
            f"profile {name_or_path!r} not found. Bundled profiles: {available}"
        )

    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)
    except yaml.YAMLError as error:
        raise ProfileError(f"{path}: invalid YAML: {error}") from error

    return parse_profile(data, str(path))


def parse_profile(data: dict, source: str) -> Profile:
    """Validate a parsed profile mapping and return a Profile."""
    if not isinstance(data, dict):
        raise ProfileError(f"{source}: profile must be a mapping")

    name = str(data.get("name") or Path(source).stem)

    audience = data.get("audience")
    if not audience or not str(audience).strip():
        raise ProfileError(f"{source}: 'audience' is required and must be non-empty")

    raw_dimensions = data.get("dimensions")
    if not isinstance(raw_dimensions, dict) or not raw_dimensions:
        raise ProfileError(f"{source}: needs at least one dimension")

    dimensions = tuple(
        _parse_dimension(dim_id, body, source)
        for dim_id, body in raw_dimensions.items()
    )

    total = sum(d.weight for d in dimensions)
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        raise ProfileError(
            f"{source}: dimension weights sum to {total:.4g}, expected 1.0. "
            f"Adjust by {1.0 - total:+.4g}"
        )

    return Profile(
        name=name,
        audience=str(audience).strip(),
        dimensions=dimensions,
        gate=_parse_gate(data.get("gate"), source),
    )


def _parse_dimension(dim_id: str, body: object, source: str) -> Dimension:
    where = f"{source}: dimension {dim_id!r}"
    if not isinstance(body, dict):
        raise ProfileError(f"{where} must be a mapping")

    kind = body.get("type", "score")
    if kind != "score":
        raise ProfileError(f"{where}: only type 'score' is supported, got {kind!r}")

    group = body.get("group")
    if group not in GROUPS:
        raise ProfileError(
            f"{where}: unknown group {group!r}. Valid groups: {', '.join(GROUPS)}"
        )

    instructions = body.get("instructions")
    if not instructions or not str(instructions).strip():
        raise ProfileError(f"{where}: 'instructions' is required and must be non-empty")

    levels = body.get("levels")
    if not isinstance(levels, list):
        raise ProfileError(f"{where}: 'levels' must be a list")
    if len(levels) < MIN_LEVELS:
        raise ProfileError(f"{where}: needs at least {MIN_LEVELS} levels, got {len(levels)}")
    if len(levels) > MAX_LEVELS:
        raise ProfileError(f"{where}: accepts at most {MAX_LEVELS} levels, got {len(levels)}")

    try:
        weight = float(body["weight"])
    except (KeyError, TypeError, ValueError) as error:
        raise ProfileError(f"{where}: 'weight' must be a number") from error
    if weight <= 0:
        raise ProfileError(f"{where}: 'weight' must be greater than 0, got {weight}")

    return Dimension(
        id=str(dim_id),
        group=str(group),
        weight=weight,
        instructions=str(instructions).strip(),
        levels=tuple(str(level).strip() for level in levels),
    )


def _parse_gate(raw: object, source: str) -> Gate | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or len(raw) != 1:
        raise ProfileError(f"{source}: 'gate' must hold exactly one question")

    gate_id, body = next(iter(raw.items()))
    where = f"{source}: gate {gate_id!r}"
    if not isinstance(body, dict):
        raise ProfileError(f"{where} must be a mapping")
    if body.get("type", "noul") != "noul":
        raise ProfileError(f"{where}: gate must be type 'noul'")

    instructions = body.get("instructions")
    if not instructions or not str(instructions).strip():
        raise ProfileError(f"{where}: 'instructions' is required")

    criteria = body.get("criteria") or {}
    if not isinstance(criteria, dict):
        raise ProfileError(f"{where}: 'criteria' must be a mapping")

    return Gate(
        id=str(gate_id),
        instructions=str(instructions).strip(),
        criteria={str(k): str(v).strip() for k, v in criteria.items()},
    )
