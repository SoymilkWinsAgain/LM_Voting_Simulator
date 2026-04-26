"""Canonical category transforms and deterministic IDs."""

from __future__ import annotations

import hashlib
import math
from typing import Any


MISSING_VALUES = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "-9",
    "-8",
    "-7",
    "-6",
    "-5",
    "-4",
    "-3",
    "-2",
    "-1",
}

STATE_FIPS_TO_PO = {
    "1": "AL",
    "2": "AK",
    "4": "AZ",
    "5": "AR",
    "6": "CA",
    "8": "CO",
    "9": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}


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


def state_fips_to_po(value: Any) -> str | None:
    return STATE_FIPS_TO_PO.get(clean_string(value))


def birthyr_to_age(value: Any, year: int = 2024) -> int | None:
    if is_missing(value):
        return None
    try:
        birthyr = int(float(value))
    except (TypeError, ValueError):
        return None
    if birthyr < 1900 or birthyr > year:
        return None
    return year - birthyr


def birthyr_to_age_group(value: Any) -> str:
    age = birthyr_to_age(value)
    return age_to_group(age)


def ces_gender4(value: Any) -> str:
    key = clean_string(value)
    if key == "1":
        return "male"
    if key == "2":
        return "female"
    if key == "3":
        return "non_binary"
    if key == "4":
        return "other"
    return "other_or_unknown"


def ces_education_detail(value: Any) -> str | None:
    return {
        "1": "no_high_school",
        "2": "high_school_graduate",
        "3": "some_college",
        "4": "two_year_degree",
        "5": "four_year_degree",
        "6": "postgraduate",
    }.get(clean_string(value))


def ces_education_binary(value: Any) -> str:
    key = clean_string(value)
    if key in {"1", "2", "3", "4"}:
        return "non_college"
    if key in {"5", "6"}:
        return "college_plus"
    return "unknown"


def ces_race_ethnicity(value: Any, hispanic: Any = None) -> str:
    if clean_string(hispanic) == "1":
        return "hispanic"
    key = clean_string(value)
    if key == "1":
        return "white"
    if key == "2":
        return "black"
    if key == "3":
        return "hispanic"
    if key == "4":
        return "asian"
    return "other_or_unknown"


def ces_pid3_to_party3(value: Any) -> str:
    key = clean_string(value)
    if key == "1":
        return "democrat"
    if key == "2":
        return "republican"
    if key in {"3", "4", "5"}:
        return "independent_or_other"
    return "unknown"


def ces_pid7_to_party3(value: Any) -> str:
    key = clean_string(value)
    if key in {"1", "2", "3"}:
        return "democrat"
    if key in {"5", "6", "7"}:
        return "republican"
    if key in {"4", "8"}:
        return "independent_or_other"
    return "unknown"


def ces_ideo5_to_ideology3(value: Any) -> str:
    key = clean_string(value)
    if key in {"1", "2"}:
        return "liberal"
    if key == "3":
        return "moderate"
    if key in {"4", "5"}:
        return "conservative"
    return "unknown"


def ces_ideo5_to_ideology7(value: Any) -> str | None:
    return {
        "1": "very_liberal",
        "2": "liberal",
        "3": "moderate",
        "4": "conservative",
        "5": "very_conservative",
        "6": "not_sure",
    }.get(clean_string(value))


def ces_turnout_self_report(value: Any) -> str:
    key = clean_string(value)
    if key == "5":
        return "voted"
    if key in {"1", "2", "3", "4"}:
        return "not_voted"
    return "unknown"


def ces_president_vote_choice(value: Any) -> str:
    key = clean_string(value)
    if key == "1":
        return "democrat"
    if key == "2":
        return "republican"
    if key in {"3", "4", "5", "6", "8"}:
        return "other"
    if key == "9":
        return "not_vote"
    return "unknown"


def ces_president_nonvoter_preference(value: Any) -> str:
    key = clean_string(value)
    if key == "1":
        return "democrat"
    if key == "2":
        return "republican"
    if key in {"3", "4", "5", "6", "8"}:
        return "other"
    if key == "9":
        return "undecided"
    return "unknown"


def ces_validated_turnout(value: Any) -> str:
    key = clean_string(value)
    if key in {"1", "2", "3", "4", "5", "6"}:
        return "voted"
    if key == "7":
        return "not_voted"
    return "unknown"


def anes_2024_education_to_binary(value: Any) -> str:
    key = clean_string(value)
    if key in {str(code) for code in range(1, 13)}:
        return "non_college"
    if key in {str(code) for code in range(13, 17)}:
        return "college_plus"
    return "unknown"


def anes_2024_hispanic_to_race_ethnicity(value: Any) -> str:
    key = clean_string(value)
    if key == "1":
        return "hispanic"
    return "other_or_unknown"


def anes_2024_party_id_to_party3(value: Any) -> str:
    key = clean_string(value)
    if key == "1":
        return "democrat"
    if key == "2":
        return "republican"
    if key in {"0", "3", "5"}:
        return "independent_or_other"
    return "unknown"


def anes_vote_choice_president(value: Any) -> str:
    key = clean_string(value)
    if key == "1":
        return "democrat"
    if key == "2":
        return "republican"
    if key in {str(code) for code in range(3, 13)}:
        return "other"
    return "not_vote_or_unknown"


TRANSFORMS = {
    "age_to_group": age_to_group,
    "education_to_binary": education_to_binary,
    "party7_to_party3": party7_to_party3,
    "ideology7_to_ideology3": ideology7_to_ideology3,
    "normalize_vote": normalize_vote,
    "state_fips_to_po": state_fips_to_po,
    "birthyr_to_age": birthyr_to_age,
    "birthyr_to_age_group": birthyr_to_age_group,
    "ces_gender4": ces_gender4,
    "ces_education_detail": ces_education_detail,
    "ces_education_binary": ces_education_binary,
    "ces_pid3_to_party3": ces_pid3_to_party3,
    "ces_pid7_to_party3": ces_pid7_to_party3,
    "ces_ideo5_to_ideology3": ces_ideo5_to_ideology3,
    "ces_ideo5_to_ideology7": ces_ideo5_to_ideology7,
    "ces_turnout_self_report": ces_turnout_self_report,
    "ces_president_vote_choice": ces_president_vote_choice,
    "ces_president_nonvoter_preference": ces_president_nonvoter_preference,
    "ces_validated_turnout": ces_validated_turnout,
    "anes_2024_education_to_binary": anes_2024_education_to_binary,
    "anes_2024_hispanic_to_race_ethnicity": anes_2024_hispanic_to_race_ethnicity,
    "anes_2024_party_id_to_party3": anes_2024_party_id_to_party3,
    "anes_vote_choice_president": anes_vote_choice_president,
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
