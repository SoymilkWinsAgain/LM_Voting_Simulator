"""ANES fixture/CSV ingestion, fact rendering, and leakage guard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io import ensure_dir, load_yaml, read_table, write_table
from .questions import load_question_config
from .transforms import apply_transform, clean_string, is_missing, stable_hash
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
    raw = read_table(cfg["path"])
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
                canonical = mapping.get(str(raw_value), mapping.get(clean_string(raw_value), raw_value))
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


def _profile_facts(row: pd.Series) -> list[str]:
    return [
        f"The respondent lives in {row.get('state_po') or 'an unknown state'}.",
        f"The respondent is in the {row['age_group']} age group.",
        f"The respondent identifies as {row['gender']}.",
        f"The respondent's race/ethnicity category is {row['race_ethnicity']}.",
        f"The respondent's education category is {row['education_binary']}.",
        f"The respondent's party identification is {row['party_id_3']}.",
        f"The respondent's ideology is {row['ideology_3']}.",
    ]


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
    templates = load_yaml(fact_templates_path)
    if isinstance(templates, dict):
        templates = templates.get("templates", [])

    template_by_var = {tpl["source_variable"]: tpl for tpl in templates}
    fact_rows: list[dict[str, Any]] = []
    for _, ans in answers.iterrows():
        tpl = template_by_var.get(ans["source_variable"])
        if not tpl:
            continue
        if ans["is_missing"] and tpl.get("missing_policy", "skip") == "skip":
            continue
        value_templates = {str(k): v for k, v in tpl.get("value_templates", {}).items()}
        value_key = clean_string(ans["canonical_value"])
        if value_key not in value_templates:
            value_key = clean_string(ans["raw_value"])
        if value_key not in value_templates:
            continue
        fact_id = stable_hash(ans["anes_id"], ans["source_variable"], value_key, policy)
        fact_rows.append(
            {
                "memory_fact_id": fact_id,
                "anes_id": ans["anes_id"],
                "source_year": int(ans["source_year"]),
                "source_variable": ans["source_variable"],
                "topic": tpl["topic"],
                "subtopic": tpl.get("subtopic"),
                "fact_text": value_templates[value_key],
                "fact_strength": tpl.get("fact_strength"),
                "safe_as_memory": bool(tpl.get("safe_as_memory", False)),
                "allowed_memory_policies": list(tpl.get("allowed_memory_policies", [])),
                "excluded_target_question_ids": list(tpl.get("excluded_target_question_ids", [])),
                "excluded_target_topics": list(tpl.get("excluded_target_topics", [])),
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )

    facts = pd.DataFrame(fact_rows)
    if facts.empty:
        facts = pd.DataFrame(
            columns=[
                "memory_fact_id",
                "anes_id",
                "source_year",
                "source_variable",
                "topic",
                "subtopic",
                "fact_text",
                "fact_strength",
                "safe_as_memory",
                "allowed_memory_policies",
                "excluded_target_question_ids",
                "excluded_target_topics",
                "created_at",
            ]
        )

    card_rows: list[dict[str, Any]] = []
    for _, respondent in respondents.iterrows():
        person_facts = facts[facts["anes_id"] == respondent["anes_id"]]
        allowed = person_facts[
            person_facts["allowed_memory_policies"].apply(lambda policies: policy in set(policies))
        ]
        allowed = allowed[allowed["safe_as_memory"]].head(max_facts)
        by_topic = {
            topic: allowed[allowed["topic"] == topic]["fact_text"].tolist()
            for topic in sorted(set(allowed["topic"].tolist()))
        }
        card_rows.append(
            {
                "memory_card_id": f"{respondent['anes_id']}_{policy}",
                "anes_id": respondent["anes_id"],
                "source_year": int(respondent["source_year"]),
                "memory_policy": policy,
                "profile_facts": _profile_facts(respondent),
                "political_facts": by_topic.get("political_engagement", []),
                "media_facts": by_topic.get("media_use", []),
                "issue_facts": [
                    text
                    for topic, texts in by_topic.items()
                    if topic not in {"political_engagement", "media_use", "party_affect"}
                    for text in texts
                ],
                "affect_facts": by_topic.get("party_affect", []),
                "open_end_facts": [],
                "all_fact_ids": allowed["memory_fact_id"].tolist(),
                "max_facts": max_facts,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )

    out = Path(out_path)
    if out.suffix:
        card_path = out
        facts_path = out.with_name("anes_memory_facts.parquet")
    else:
        out.mkdir(parents=True, exist_ok=True)
        card_path = out / "anes_memory_cards.parquet"
        facts_path = out / "anes_memory_facts.parquet"
    write_table(facts, facts_path)
    cards = pd.DataFrame(card_rows)
    write_table(cards, card_path)
    return {"facts": facts_path, "cards": card_path}


class LeakageGuard:
    """Filter memory facts that leak target answers or violate policy."""

    def filter_facts(
        self,
        facts: pd.DataFrame,
        target_question: pd.Series | dict[str, Any],
        memory_policy: str,
    ) -> pd.DataFrame:
        if facts.empty:
            return facts.copy()
        question = dict(target_question)
        out = facts.copy()
        out = out[out["safe_as_memory"].astype(bool)]
        out = out[out["allowed_memory_policies"].apply(lambda policies: memory_policy in set(policies))]
        out = out[
            ~out["excluded_target_question_ids"].apply(
                lambda ids: question.get("question_id") in set(ids or [])
            )
        ]
        out = out[
            ~out["excluded_target_topics"].apply(lambda topics: question.get("topic") in set(topics or []))
        ]
        excluded_vars = set(question.get("excluded_memory_variables") or [])
        if excluded_vars:
            out = out[~out["source_variable"].isin(excluded_vars)]
        excluded_topics = set(question.get("excluded_memory_topics") or [])
        if excluded_topics:
            out = out[~out["topic"].isin(excluded_topics)]
        return out
