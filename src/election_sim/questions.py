"""Question bank loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .io import load_yaml, read_table
from .validation import require_columns


QUESTION_COLUMNS = [
    "question_id",
    "source",
    "source_year",
    "source_variable",
    "topic",
    "subtopic",
    "question_text",
    "options_json",
    "canonical_target_type",
    "is_vote_choice",
    "is_candidate_eval",
    "is_party_eval",
    "is_issue_position",
    "allowed_answer_codes",
    "missing_answer_codes",
    "excluded_memory_variables",
    "excluded_memory_topics",
    "created_at",
]


def _normalize_question(raw: dict[str, Any]) -> dict[str, Any]:
    options = raw.get("options", raw.get("options_json", {})) or {}
    if isinstance(options, str):
        options_dict = json.loads(options)
    else:
        options_dict = dict(options)
    allowed = list(raw.get("allowed_answer_codes") or options_dict.keys())
    normalized = {
        "question_id": raw["question_id"],
        "source": raw.get("source", "custom"),
        "source_year": raw.get("source_year"),
        "source_variable": raw.get("source_variable"),
        "topic": raw.get("topic", "unknown"),
        "subtopic": raw.get("subtopic"),
        "question_text": raw["question_text"],
        "options_json": json.dumps(options_dict, sort_keys=True),
        "canonical_target_type": raw.get("canonical_target_type", "categorical"),
        "is_vote_choice": bool(raw.get("is_vote_choice", False)),
        "is_candidate_eval": bool(raw.get("is_candidate_eval", False)),
        "is_party_eval": bool(raw.get("is_party_eval", False)),
        "is_issue_position": bool(raw.get("is_issue_position", False)),
        "allowed_answer_codes": allowed,
        "missing_answer_codes": list(raw.get("missing_answer_codes", [])),
        "excluded_memory_variables": list(raw.get("excluded_memory_variables", [])),
        "excluded_memory_topics": list(raw.get("excluded_memory_topics", [])),
        "created_at": raw.get("created_at", pd.Timestamp.now(tz="UTC")),
    }
    for key, value in raw.items():
        if key not in normalized and key != "options":
            if key == "value_mapping" and isinstance(value, dict):
                value = {str(k): v for k, v in value.items()}
            normalized[key] = value
    return normalized


def load_question_config(path: str | Path) -> pd.DataFrame:
    if Path(path).suffix == ".parquet":
        df = read_table(path)
        validate_question_bank(df)
        return df
    raw = load_yaml(path)
    if isinstance(raw, list):
        questions = raw
    else:
        questions = raw.get("questions", [raw])
    df = pd.DataFrame([_normalize_question(q) for q in questions])
    validate_question_bank(df)
    return df


def validate_question_bank(df: pd.DataFrame) -> None:
    require_columns(df, QUESTION_COLUMNS, "question_bank")
    for _, row in df.iterrows():
        options = json.loads(row["options_json"])
        allowed = set(row["allowed_answer_codes"])
        missing = allowed - set(options)
        if missing:
            raise ValueError(f"Question {row['question_id']} has allowed codes missing from options: {missing}")
