"""Canonical category transforms and deterministic IDs."""

from __future__ import annotations

import hashlib
import math
from typing import Any


MISSING_VALUES = {"", "nan", "none", "null", "na", "n/a", "-9", "-8", "-7", "-1"}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in MISSING_VALUES


def clean_string(value: Any) -> str:
    if is_missing(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def normalize_mapping(value: Any, mapping: dict[str, Any], default: str = "unknown") -> Any:
    key = clean_string(value)
    if key in mapping:
        return mapping[key]
    lowered = key.lower()
    if lowered in mapping:
        return mapping[lowered]
    return mapping.get("default", default)


def age_to_group(value: Any) -> str:
    if is_missing(value):
        return "unknown"
    try:
        age = int(float(value))
    except (TypeError, ValueError):
        return "unknown"
    if age < 18:
        return "unknown"
    if age <= 29:
        return "18_29"
    if age <= 44:
        return "30_44"
    if age <= 64:
        return "45_64"
    return "65_plus"


def education_to_binary(value: Any) -> str:
    key = clean_string(value).lower()
    if key in {"college_plus", "college", "ba", "bachelor", "bachelors", "postgrad", "postgraduate"}:
        return "college_plus"
    if key in {"non_college", "hs", "high_school", "some_college", "less_than_hs"}:
        return "non_college"
    if key in {"1", "2", "3", "4"}:
        return "non_college"
    if key in {"5", "6", "7"}:
        return "college_plus"
    return "unknown"


def party7_to_party3(value: Any) -> str:
    key = clean_string(value).lower()
    if key in {"democrat", "strong_democrat", "weak_democrat", "lean_democrat", "1", "2", "3"}:
        return "democrat"
    if key in {"republican", "strong_republican", "weak_republican", "lean_republican", "5", "6", "7"}:
        return "republican"
    if key in {"independent", "independent_or_other", "other", "4"}:
        return "independent_or_other"
    return "unknown"


def ideology7_to_ideology3(value: Any) -> str:
    key = clean_string(value).lower()
    if key in {"liberal", "very_liberal", "slightly_liberal", "1", "2", "3"}:
        return "liberal"
    if key in {"moderate", "middle", "4"}:
        return "moderate"
    if key in {"conservative", "very_conservative", "slightly_conservative", "5", "6", "7"}:
        return "conservative"
    return "unknown"


def normalize_vote(value: Any) -> str:
    key = clean_string(value).lower()
    if key in {"democrat", "democratic", "harris", "biden", "1"}:
        return "democrat"
    if key in {"republican", "trump", "gop", "2"}:
        return "republican"
    if key in {"other", "third_party", "3"}:
        return "other"
    return "not_vote_or_unknown"


TRANSFORMS = {
    "age_to_group": age_to_group,
    "education_to_binary": education_to_binary,
    "party7_to_party3": party7_to_party3,
    "ideology7_to_ideology3": ideology7_to_ideology3,
    "normalize_vote": normalize_vote,
    "int": lambda value: None if is_missing(value) else int(float(value)),
    "float": lambda value: None if is_missing(value) else float(value),
    "string": lambda value: None if is_missing(value) else clean_string(value),
}


def apply_transform(value: Any, spec: dict[str, Any]) -> Any:
    if "mapping" in spec:
        return normalize_mapping(value, {str(k): v for k, v in spec["mapping"].items()})
    name = spec.get("transform", "string")
    if name not in TRANSFORMS:
        raise ValueError(f"Unknown transform: {name}")
    return TRANSFORMS[name](value)


def stable_hash(*parts: Any, length: int = 16) -> str:
    raw = "|".join(clean_string(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def make_cell_id(row: dict[str, Any], columns: list[str]) -> str:
    return "|".join(clean_string(row.get(col, "unknown")) or "unknown" for col in columns)
