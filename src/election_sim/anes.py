"""ANES fixture/CSV ingestion, fact rendering, and leakage guard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io import ensure_dir, load_yaml, read_table, write_table
from .questions import load_question_config
from .survey_memory import LeakageGuard, build_survey_memory_cards
from .transforms import TRANSFORMS, apply_transform, clean_string, is_missing
from .validation import require_columns, validate_categories


ANES_RESPONDENT_COLUMNS = [
    "anes_id",
    "source_year",
    "sample_component",
    "wave_available_pre",
    "wave_available_post",
    "state_po",
    "region",
    "age",
    "age_group",
    "gender",
    "race_ethnicity",
    "education_binary",
    "education_detail",
    "income_bin",
    "party_id_3",
    "party_id_7",
    "ideology_3",
    "ideology_7",
    "political_interest",
    "religion",
    "urbanicity",
    "weight_pre",
    "weight_post",
    "weight_full",
    "schema_version",
]

ANES_ANSWER_COLUMNS = [
    "anes_id",
    "source_year",
    "wave",
    "source_variable",
    "question_id",
    "topic",
    "raw_value",
    "raw_label",
    "canonical_value",
    "canonical_label",
    "is_missing",
    "is_refusal",
    "is_dont_know",
    "is_valid_for_memory",
    "is_valid_for_target",
]


def _field_value(raw_row: pd.Series, spec: dict[str, Any]) -> Any:
    var = spec.get("variable")
    if var is None or var not in raw_row:
        return spec.get("default")
    return apply_transform(raw_row[var], spec)


def build_anes(
    config_path: str | Path,
    profile_crosswalk_path: str | Path,
    question_crosswalk_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Path]:
    cfg = load_yaml(config_path)
    crosswalk = load_yaml(profile_crosswalk_path)
    questions = load_question_config(question_crosswalk_path)
    raw_path = Path(cfg["path"])
    if raw_path.suffix.lower() == ".csv" and cfg.get("usecols"):
        raw = pd.read_csv(raw_path, usecols=cfg["usecols"])
    else:
        raw = read_table(raw_path)
    year = int(crosswalk.get("source_year", cfg.get("year")))
    respondent_id_col = crosswalk["respondent_id"]

    respondent_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    fields = crosswalk.get("fields", {})
    weights = crosswalk.get("weights", {})

    for _, raw_row in raw.iterrows():
        anes_id = clean_string(raw_row[respondent_id_col])
        respondent = {
            "anes_id": anes_id,
            "source_year": year,
            "sample_component": cfg.get("sample_component", "fixture"),
            "wave_available_pre": True,
            "wave_available_post": True,
            "state_po": _field_value(raw_row, fields.get("state_po", {"default": None})),
            "region": _field_value(raw_row, fields.get("region", {"default": None})),
            "age": _field_value(raw_row, fields.get("age", {"default": None})),
            "age_group": _field_value(raw_row, fields.get("age_group", {"default": "unknown"})),
            "gender": _field_value(raw_row, fields.get("gender", {"default": "other_or_unknown"})),
            "race_ethnicity": _field_value(raw_row, fields.get("race_ethnicity", {"default": "other_or_unknown"})),
            "education_binary": _field_value(raw_row, fields.get("education_binary", {"default": "unknown"})),
            "education_detail": _field_value(raw_row, fields.get("education_detail", {"default": None})),
            "income_bin": _field_value(raw_row, fields.get("income_bin", {"default": None})),
            "party_id_3": _field_value(raw_row, fields.get("party_id_3", {"default": "unknown"})),
            "party_id_7": _field_value(raw_row, fields.get("party_id_7", {"default": None})),
            "ideology_3": _field_value(raw_row, fields.get("ideology_3", {"default": "unknown"})),
            "ideology_7": _field_value(raw_row, fields.get("ideology_7", {"default": None})),
            "political_interest": _field_value(raw_row, fields.get("political_interest", {"default": None})),
            "religion": _field_value(raw_row, fields.get("religion", {"default": None})),
            "urbanicity": _field_value(raw_row, fields.get("urbanicity", {"default": None})),
            "weight_pre": float(raw_row[weights["pre"]]) if weights.get("pre") in raw_row else None,
            "weight_post": float(raw_row[weights["post"]]) if weights.get("post") in raw_row else None,
            "weight_full": float(raw_row[weights["full"]]) if weights.get("full") in raw_row else None,
            "schema_version": "anes_respondents_v1",
        }
        respondent_rows.append(respondent)

        for _, question in questions.iterrows():
            source_var = question["source_variable"]
            if not source_var or source_var not in raw_row:
                continue
            raw_value = raw_row[source_var]
            mapping = question.get("value_mapping")
            canonical = raw_value
            if isinstance(mapping, dict):
                mapping = {str(k): v for k, v in mapping.items()}
                canonical = mapping.get(str(raw_value), mapping.get(clean_string(raw_value), raw_value))
            elif isinstance(question.get("canonical_transform"), str):
                transform_name = question["canonical_transform"]
                if transform_name not in TRANSFORMS:
                    raise ValueError(f"Unknown canonical_transform: {transform_name}")
                canonical = TRANSFORMS[transform_name](raw_value)
            elif question["question_id"].startswith("vote_choice"):
                from .transforms import normalize_vote

                canonical = normalize_vote(raw_value)
            answer_rows.append(
                {
                    "anes_id": anes_id,
                    "source_year": year,
                    "wave": question.get("wave", "unknown"),
                    "source_variable": source_var,
                    "question_id": question["question_id"],
                    "topic": question["topic"],
                    "raw_value": raw_value,
                    "raw_label": None,
                    "canonical_value": canonical,
                    "canonical_label": canonical,
                    "is_missing": is_missing(raw_value),
                    "is_refusal": False,
                    "is_dont_know": clean_string(raw_value).lower() in {"dk", "dont_know", "don't know"},
                    "is_valid_for_memory": not bool(question["is_vote_choice"]),
                    "is_valid_for_target": True,
                }
            )

    respondents = pd.DataFrame(respondent_rows, columns=ANES_RESPONDENT_COLUMNS)
    answers = pd.DataFrame(answer_rows, columns=ANES_ANSWER_COLUMNS)
    validate_anes_respondents(respondents)

    out = ensure_dir(out_dir)
    paths = {
        "respondents": out / "anes_respondents.parquet",
        "answers": out / "anes_answers.parquet",
        "question_bank": out / "question_bank.parquet",
    }
    write_table(respondents, paths["respondents"])
    write_table(answers, paths["answers"])
    write_table(questions, paths["question_bank"])
    return paths


def validate_anes_respondents(df: pd.DataFrame) -> None:
    require_columns(df, ANES_RESPONDENT_COLUMNS, "anes_respondents")
    validate_categories(df, "anes_respondents")


def build_memory_cards(
    respondents_path: str | Path,
    answers_path: str | Path,
    fact_templates_path: str | Path,
    policy: str,
    out_path: str | Path,
    *,
    max_facts: int = 24,
) -> dict[str, Path]:
    respondents = pd.read_parquet(respondents_path)
    answers = pd.read_parquet(answers_path)
    return build_survey_memory_cards(
        respondents,
        answers,
        fact_templates_path,
        policy,
        out_path,
        id_col="anes_id",
        output_prefix="anes",
        max_facts=max_facts,
        include_profile_facts=True,
    )
