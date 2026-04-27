"""CES/CCES respondent ingest, memory inputs, and weighted cell distributions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_cell_schema
from .io import ensure_dir, load_yaml, read_table, write_table
from .survey_memory import build_leakage_audit, build_survey_memory_cards
from .transforms import (
    apply_transform,
    birthyr_to_age,
    ces_education_binary,
    ces_education_detail,
    ces_gender4,
    ces_ideo5_to_ideology3,
    ces_ideo5_to_ideology7,
    ces_pid3_to_party3,
    ces_pid7_to_party3,
    ces_president_nonvoter_preference,
    ces_president_vote_choice,
    ces_race_ethnicity,
    ces_turnout_self_report,
    ces_validated_turnout,
    clean_string,
    is_missing,
    make_cell_id,
    normalize_vote,
    state_fips_to_po,
)
from .validation import require_columns, validate_categories, validate_probability_simplex


CES_INGEST_RESPONDENT_COLUMNS = [
    "ces_id",
    "source_year",
    "tookpost",
    "state_po",
    "state_fips",
    "county_fips",
    "county_name",
    "cdid",
    "region",
    "birthyr",
    "age",
    "age_group",
    "gender",
    "race_ethnicity",
    "hispanic",
    "education_detail",
    "education_binary",
    "income_bin",
    "employment",
    "religion",
    "bornagain",
    "marital_status",
    "party_id_3_pre",
    "party_id_7_pre",
    "party_id_3_post",
    "party_id_7_post",
    "ideology_self_7",
    "ideology_3",
    "registered_self_pre",
    "registered_self_post",
    "party_registration_self",
    "party_registration_validated",
    "citizenship",
    "weight_common",
    "weight_common_post",
    "weight_vv",
    "weight_vv_post",
    "validated_registration",
    "validated_turnout_2024",
    "validated_vote_mode_2024",
    "schema_version",
]

CES_ANSWER_COLUMNS = [
    "ces_id",
    "source_year",
    "wave",
    "question_id",
    "source_variable",
    "question_text",
    "answer_code",
    "answer_label",
    "canonical_value",
    "topic",
    "is_multiselect",
    "is_grid_item",
    "is_pre_election",
    "allowed_for_memory_strict",
    "leakage_group",
    "fact_role",
    "is_missing",
    "schema_version",
]

CES_TARGET_COLUMNS = [
    "ces_id",
    "source_year",
    "target_id",
    "source_variable",
    "target_type",
    "answer_code",
    "answer_label",
    "canonical_value",
    "truth_source",
    "weight_column_recommended",
    "schema_version",
]

CES_CONTEXT_COLUMNS = [
    "ces_id",
    "year",
    "office",
    "state_po",
    "district",
    "candidate_slot",
    "candidate_name",
    "candidate_party",
    "candidate_incumbent",
    "context_source_variable",
    "schema_version",
]


CES_RESPONDENT_COLUMNS = [
    "ces_id",
    "source_year",
    "state_po",
    "state_fips",
    "age_group",
    "gender",
    "race_ethnicity",
    "education_binary",
    "income_bin",
    "party_id_3",
    "ideology_3",
    "registered_voter",
    "validated_vote",
    "vote_choice_president",
    "common_weight",
    "schema_version",
]


def _load_codebook(config: dict[str, Any]) -> dict[str, Any]:
    path = config.get("codebook")
    if not path:
        return {}
    return load_yaml(path).get("variables", {})


def _required_usecols(*configs: dict[str, Any]) -> list[str] | None:
    cols: set[str] = set()
    for config in configs:
        if not config:
            continue
        respondent_id = config.get("respondent_id")
        if respondent_id:
            cols.add(respondent_id)
        for spec in config.get("fields", {}).values():
            if isinstance(spec, dict) and spec.get("variable"):
                cols.add(spec["variable"])
        for item in config.get("questions", []):
            if item.get("source_variable"):
                cols.add(item["source_variable"])
        for item in config.get("targets", []):
            if item.get("source_variable"):
                cols.add(item["source_variable"])
            if item.get("variable"):
                cols.add(item["variable"])
    return sorted(cols) if cols else None


def _read_ces_raw(dataset_cfg: dict[str, Any], usecols: list[str] | None) -> pd.DataFrame:
    path = Path(dataset_cfg["path"])
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=usecols)
    return read_table(path)


def _value(row: pd.Series, variable: str | None) -> Any:
    if not variable or variable not in row:
        return None
    return row[variable]


def _float_or_none(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if is_missing(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _label(codebook: dict[str, Any], variable: str, value: Any) -> str | None:
    labels = codebook.get(variable, {}).get("labels", {})
    return labels.get(clean_string(value))


def _canonical_from_mapping(mapping: dict[str, Any] | None, value: Any, default: str = "unknown") -> str:
    if not mapping:
        return default
    return str(mapping.get(clean_string(value), default))


def _is_answer_missing(value: Any, label: str | None) -> bool:
    if is_missing(value):
        return True
    return clean_string(label).lower() in {"skipped", "not asked"}


def _yes_no_unknown(value: Any, yes_code: str = "1", no_code: str = "2") -> str:
    key = clean_string(value)
    if key == yes_code:
        return "yes"
    if key == no_code:
        return "no"
    return "unknown"


def _party_registration(value: Any) -> str:
    key = clean_string(value)
    if key in {"2"}:
        return "democrat"
    if key in {"3"}:
        return "republican"
    if key in {"1", "4"}:
        return "independent_or_other"
    return "unknown"


def _validated_party_registration(value: Any) -> str:
    key = clean_string(value)
    if key == "2":
        return "democrat"
    if key == "8":
        return "republican"
    if key in {"1", "3", "4", "5", "6", "7", "9"}:
        return "independent_or_other"
    return "unknown"


def _respondent_from_row(
    row: pd.Series,
    *,
    dataset_cfg: dict[str, Any],
    profile: dict[str, Any],
    codebook: dict[str, Any],
) -> dict[str, Any]:
    fields = profile.get("fields", {})
    rid_col = profile.get("respondent_id", dataset_cfg.get("respondent_id", "caseid"))
    birthyr = _int_or_none(_value(row, fields.get("birthyr", {}).get("variable", "birthyr")))
    age = birthyr_to_age(birthyr, int(dataset_cfg.get("year", 2024))) if birthyr else None
    state_fips = clean_string(_value(row, fields.get("state_po", {}).get("variable", "inputstate")))
    gender_value = _value(row, fields.get("gender", {}).get("variable", "gender4"))
    race_value = _value(row, fields.get("race_ethnicity", {}).get("variable", "race"))
    hispanic_value = _value(row, fields.get("hispanic", {}).get("variable", "hispanic"))
    educ_value = _value(row, fields.get("education_detail", {}).get("variable", "educ"))
    pid3_value = _value(row, fields.get("party_id_3_pre", {}).get("variable", "pid3"))
    pid7_value = _value(row, fields.get("party_id_7_pre", {}).get("variable", "pid7"))
    post_pid7_value = _value(row, fields.get("party_id_7_post", {}).get("variable", "CC24_pid7"))
    ideo_value = _value(row, fields.get("ideology_self_7", {}).get("variable", "ideo5"))
    ts_g2024 = _value(row, fields.get("validated_turnout_2024", {}).get("variable", "TS_g2024"))
    tookpost_value = _value(row, fields.get("tookpost", {}).get("variable", "tookpost"))
    county_fips = _value(row, fields.get("county_fips", {}).get("variable", "countyfips"))
    return {
        "ces_id": clean_string(row[rid_col]),
        "source_year": int(profile.get("source_year", dataset_cfg.get("year", 2024))),
        "tookpost": clean_string(tookpost_value) in {"2", "yes", "true"},
        "state_po": state_fips_to_po(state_fips),
        "state_fips": state_fips,
        "county_fips": clean_string(county_fips) or None,
        "county_name": _value(row, fields.get("county_name", {}).get("variable", "countyname")),
        "cdid": _value(row, fields.get("cdid", {}).get("variable", "cdid119")),
        "region": _value(row, fields.get("region", {}).get("variable", "region")),
        "birthyr": birthyr,
        "age": age,
        "age_group": apply_transform(age, {"transform": "age_to_group"}),
        "gender": ces_gender4(gender_value),
        "race_ethnicity": ces_race_ethnicity(race_value, hispanic_value),
        "hispanic": _yes_no_unknown(hispanic_value),
        "education_detail": ces_education_detail(educ_value),
        "education_binary": ces_education_binary(educ_value),
        "income_bin": _value(row, fields.get("income_bin", {}).get("variable", "faminc_new")),
        "employment": _label(codebook, "employ", _value(row, fields.get("employment", {}).get("variable", "employ"))),
        "religion": _label(codebook, "religpew", _value(row, fields.get("religion", {}).get("variable", "religpew"))),
        "bornagain": _yes_no_unknown(_value(row, fields.get("bornagain", {}).get("variable", "pew_bornagain"))),
        "marital_status": _label(codebook, "marstat", _value(row, fields.get("marital_status", {}).get("variable", "marstat"))),
        "party_id_3_pre": ces_pid3_to_party3(pid3_value),
        "party_id_7_pre": _label(codebook, "pid7", pid7_value),
        "party_id_3_post": ces_pid7_to_party3(post_pid7_value),
        "party_id_7_post": _label(codebook, "pid7", post_pid7_value),
        "ideology_self_7": ces_ideo5_to_ideology7(ideo_value),
        "ideology_3": ces_ideo5_to_ideology3(ideo_value),
        "registered_self_pre": _yes_no_unknown(_value(row, fields.get("registered_self_pre", {}).get("variable", "votereg"))),
        "registered_self_post": _yes_no_unknown(
            _value(row, fields.get("registered_self_post", {}).get("variable", "votereg_post"))
        ),
        "party_registration_self": _party_registration(
            _value(row, fields.get("party_registration_self", {}).get("variable", "CC24_361b"))
        ),
        "party_registration_validated": _validated_party_registration(
            _value(row, fields.get("party_registration_validated", {}).get("variable", "TS_partyreg"))
        ),
        "citizenship": _yes_no_unknown(_value(row, fields.get("citizenship", {}).get("variable", "cit1"))),
        "weight_common": _float_or_none(_value(row, fields.get("weight_common", {}).get("variable", "commonweight"))),
        "weight_common_post": _float_or_none(
            _value(row, fields.get("weight_common_post", {}).get("variable", "commonpostweight"))
        ),
        "weight_vv": _float_or_none(_value(row, fields.get("weight_vv", {}).get("variable", "vvweight"))),
        "weight_vv_post": _float_or_none(_value(row, fields.get("weight_vv_post", {}).get("variable", "vvweight_post"))),
        "validated_registration": clean_string(
            _value(row, fields.get("validated_registration", {}).get("variable", "TS_voterstatus"))
        )
        == "1",
        "validated_turnout_2024": ces_validated_turnout(ts_g2024),
        "validated_vote_mode_2024": _label(codebook, "TS_g2024", ts_g2024),
        "schema_version": "ces_respondents_2024_v1",
    }


def normalize_ces_respondents(
    raw: pd.DataFrame,
    dataset_cfg: dict[str, Any],
    profile_crosswalk: dict[str, Any],
    codebook: dict[str, Any],
) -> pd.DataFrame:
    rows = [
        _respondent_from_row(row, dataset_cfg=dataset_cfg, profile=profile_crosswalk, codebook=codebook)
        for _, row in raw.iterrows()
    ]
    respondents = pd.DataFrame(rows, columns=CES_INGEST_RESPONDENT_COLUMNS)
    validate_categories(
        respondents.rename(
            columns={
                "party_id_3_pre": "party_id_3",
                "ideology_3": "ideology_3",
            }
        ),
        "ces_respondents",
    )
    return respondents


def build_ces_answers(
    raw: pd.DataFrame,
    respondents: pd.DataFrame,
    dataset_cfg: dict[str, Any],
    question_crosswalk: dict[str, Any],
    codebook: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    respondent_ids = respondents["ces_id"].tolist()
    year = int(dataset_cfg.get("year", 2024))
    answer_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    for question in question_crosswalk.get("questions", []):
        source_var = question["source_variable"]
        labels = codebook.get(source_var, {}).get("labels", question.get("labels", {}))
        canonical_mapping = {str(k): v for k, v in question.get("canonical_mapping", {}).items()}
        question_rows.append(
            {
                "question_id": question["question_id"],
                "source_variable": source_var,
                "question_text": question["question_text"],
                "topic": question.get("topic", "unknown"),
                "wave": question.get("wave", "pre"),
                "is_grid_item": bool(question.get("is_grid_item", False)),
                "allowed_for_memory_strict": bool(question.get("allowed_for_memory_strict", True)),
                "leakage_group": question.get("leakage_group", "safe_pre"),
                "fact_role": question.get("fact_role", "safe_pre"),
                "schema_version": "ces_question_bank_2024_v1",
            }
        )
        if source_var not in raw.columns:
            continue
        for ces_id, raw_value in zip(respondent_ids, raw[source_var], strict=False):
            answer_code = clean_string(raw_value)
            answer_label = labels.get(answer_code)
            answer_rows.append(
                {
                    "ces_id": ces_id,
                    "source_year": year,
                    "wave": question.get("wave", "pre"),
                    "question_id": question["question_id"],
                    "source_variable": source_var,
                    "question_text": question["question_text"],
                    "answer_code": answer_code,
                    "answer_label": answer_label,
                    "canonical_value": _canonical_from_mapping(canonical_mapping, raw_value),
                    "topic": question.get("topic", "unknown"),
                    "is_multiselect": bool(question.get("is_multiselect", False)),
                    "is_grid_item": bool(question.get("is_grid_item", False)),
                    "is_pre_election": question.get("wave", "pre") == "pre",
                    "allowed_for_memory_strict": bool(question.get("allowed_for_memory_strict", True)),
                    "leakage_group": question.get("leakage_group", "safe_pre"),
                    "fact_role": question.get("fact_role", "safe_pre"),
                    "is_missing": _is_answer_missing(raw_value, answer_label),
                    "schema_version": "ces_answers_2024_v1",
                }
            )
    return (
        pd.DataFrame(answer_rows, columns=CES_ANSWER_COLUMNS),
        pd.DataFrame(question_rows),
    )


def _target_canonical(transform: str, value: Any) -> str:
    if transform == "ces_turnout_self_report":
        return ces_turnout_self_report(value)
    if transform == "ces_president_vote_choice":
        return ces_president_vote_choice(value)
    if transform == "ces_president_nonvoter_preference":
        return ces_president_nonvoter_preference(value)
    if transform == "ces_validated_turnout":
        return ces_validated_turnout(value)
    return clean_string(value) or "unknown"


def build_ces_targets(
    raw: pd.DataFrame,
    respondents: pd.DataFrame,
    dataset_cfg: dict[str, Any],
    target_crosswalk: dict[str, Any],
    codebook: dict[str, Any],
) -> pd.DataFrame:
    respondent_ids = respondents["ces_id"].tolist()
    year = int(dataset_cfg.get("year", 2024))
    rows: list[dict[str, Any]] = []
    for target in target_crosswalk.get("targets", []):
        source_var = target.get("source_variable", target.get("variable"))
        if source_var not in raw.columns:
            continue
        labels = codebook.get(source_var, {}).get("labels", target.get("labels", {}))
        for ces_id, raw_value in zip(respondent_ids, raw[source_var], strict=False):
            answer_code = clean_string(raw_value)
            answer_label = labels.get(answer_code)
            rows.append(
                {
                    "ces_id": ces_id,
                    "source_year": year,
                    "target_id": target["target_id"],
                    "source_variable": source_var,
                    "target_type": target["target_type"],
                    "answer_code": answer_code,
                    "answer_label": answer_label,
                    "canonical_value": _target_canonical(target.get("canonical_transform", ""), raw_value),
                    "truth_source": target["truth_source"],
                    "weight_column_recommended": target.get("weight_column_recommended"),
                    "schema_version": "ces_targets_2024_v1",
                }
            )
    return pd.DataFrame(rows, columns=CES_TARGET_COLUMNS)


def build_ces_context(
    respondents: pd.DataFrame,
    dataset_cfg: dict[str, Any],
    context_crosswalk: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    year = int(dataset_cfg.get("year", 2024))
    candidates = context_crosswalk.get("president", {}).get("candidates", [])
    for _, respondent in respondents.iterrows():
        for candidate in candidates:
            rows.append(
                {
                    "ces_id": respondent["ces_id"],
                    "year": year,
                    "office": "president",
                    "state_po": respondent["state_po"],
                    "district": None,
                    "candidate_slot": candidate["slot"],
                    "candidate_name": candidate["name"],
                    "candidate_party": candidate["party"],
                    "candidate_incumbent": bool(candidate.get("incumbent", False)),
                    "context_source_variable": "static_2024_president_context",
                    "schema_version": "ces_context_2024_v1",
                }
            )
    return pd.DataFrame(rows, columns=CES_CONTEXT_COLUMNS)


def build_ces(
    config_path: str | Path,
    profile_crosswalk_path: str | Path,
    question_crosswalk_path: str | Path,
    target_crosswalk_path: str | Path,
    context_crosswalk_path: str | Path,
    out_path: str | Path,
) -> dict[str, Path]:
    dataset_cfg = load_yaml(config_path)
    profile = load_yaml(profile_crosswalk_path)
    questions = load_yaml(question_crosswalk_path)
    targets = load_yaml(target_crosswalk_path)
    context = load_yaml(context_crosswalk_path)
    codebook = _load_codebook(dataset_cfg)
    usecols = _required_usecols(profile, questions, targets)
    raw = _read_ces_raw(dataset_cfg, usecols)

    respondents = normalize_ces_respondents(raw, dataset_cfg, profile, codebook)
    answers, question_bank = build_ces_answers(raw, respondents, dataset_cfg, questions, codebook)
    target_df = build_ces_targets(raw, respondents, dataset_cfg, targets, codebook)
    context_df = build_ces_context(respondents, dataset_cfg, context)

    out = ensure_dir(out_path)
    paths = {
        "respondents": out / "ces_respondents.parquet",
        "answers": out / "ces_answers.parquet",
        "targets": out / "ces_targets.parquet",
        "context": out / "ces_context.parquet",
        "question_bank": out / "ces_question_bank.parquet",
        "ingest_report": out / "ces_ingest_report.md",
    }
    write_table(respondents, paths["respondents"])
    write_table(answers, paths["answers"])
    write_table(target_df, paths["targets"])
    write_table(context_df, paths["context"])
    write_table(question_bank, paths["question_bank"])
    report = [
        f"# CES Ingest Report: {dataset_cfg.get('name', 'ces_2024')}",
        "",
        f"- Raw rows read: {len(raw)}",
        f"- Respondents: {len(respondents)}",
        f"- Pre-election answers: {len(answers)}",
        f"- Targets: {len(target_df)}",
        f"- Context rows: {len(context_df)}",
        f"- Source path: `{dataset_cfg['path']}`",
        f"- Schema version: `{dataset_cfg.get('schema_version', 'ces_2024_common_vv_v1')}`",
        "",
    ]
    paths["ingest_report"].write_text("\n".join(report), encoding="utf-8")
    return paths


def build_ces_memory_cards(
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
    memory_paths = build_survey_memory_cards(
        respondents,
        answers,
        fact_templates_path,
        policy,
        out_path,
        id_col="ces_id",
        output_prefix="ces",
        max_facts=max_facts,
        include_profile_facts=False,
    )
    audit_path = build_leakage_audit(answers, fact_templates_path, policy, out_path)
    return {**memory_paths, "audit": audit_path}


def normalize_ces(config_path: str | Path, crosswalk_path: str | Path) -> pd.DataFrame:
    cfg = load_yaml(config_path)
    crosswalk = load_yaml(crosswalk_path)
    raw = read_table(cfg["path"])
    fields = crosswalk.get("fields", {})
    rows: list[dict[str, Any]] = []
    for _, raw_row in raw.iterrows():
        vote_spec = fields.get("vote_choice_president", {})
        vote_value = raw_row[vote_spec["variable"]] if vote_spec.get("variable") in raw_row else None
        rows.append(
            {
                "ces_id": str(raw_row[crosswalk["respondent_id"]]),
                "source_year": int(crosswalk.get("source_year", cfg.get("year"))),
                "state_po": apply_transform(raw_row[fields["state_po"]["variable"]], fields["state_po"]),
                "state_fips": apply_transform(raw_row[fields["state_fips"]["variable"]], fields["state_fips"])
                if fields.get("state_fips")
                else None,
                "age_group": apply_transform(raw_row[fields["age_group"]["variable"]], fields["age_group"]),
                "gender": apply_transform(raw_row[fields["gender"]["variable"]], fields["gender"]),
                "race_ethnicity": apply_transform(
                    raw_row[fields["race_ethnicity"]["variable"]], fields["race_ethnicity"]
                ),
                "education_binary": apply_transform(
                    raw_row[fields["education_binary"]["variable"]], fields["education_binary"]
                ),
                "income_bin": apply_transform(raw_row[fields["income_bin"]["variable"]], fields["income_bin"])
                if fields.get("income_bin")
                else None,
                "party_id_3": apply_transform(raw_row[fields["party_id_3"]["variable"]], fields["party_id_3"]),
                "ideology_3": apply_transform(raw_row[fields["ideology_3"]["variable"]], fields["ideology_3"]),
                "registered_voter": None,
                "validated_vote": None,
                "vote_choice_president": normalize_vote(vote_value),
                "common_weight": float(raw_row[crosswalk["weight"]]),
                "schema_version": "ces_respondents_v1",
            }
        )
    df = pd.DataFrame(rows, columns=CES_RESPONDENT_COLUMNS)
    validate_ces_respondents(df)
    return df


def validate_ces_respondents(df: pd.DataFrame) -> None:
    require_columns(df, CES_RESPONDENT_COLUMNS, "ces_respondents")
    validate_categories(df, "ces_respondents")


def build_cell_distribution(
    ces_df: pd.DataFrame,
    cell_schema: dict[str, Any],
    *,
    tau: float = 500.0,
) -> pd.DataFrame:
    cell_cols = list(cell_schema["columns"])
    cell_feature_cols = [col for col in cell_cols if col != "state_po"]
    work = ces_df.copy()
    work["cell_id"] = work.apply(lambda row: make_cell_id(row.to_dict(), cell_cols), axis=1)

    grouped = (
        work.groupby(["source_year", "state_po", "cell_id", *cell_feature_cols], dropna=False)
        .agg(weighted_n=("common_weight", "sum"), raw_n=("common_weight", "size"))
        .reset_index()
    )
    totals = grouped.groupby("state_po")["weighted_n"].transform("sum")
    grouped["weighted_share_raw"] = grouped["weighted_n"] / totals

    national = (
        work.groupby(["cell_id", *cell_feature_cols], dropna=False)
        .agg(national_weighted_n=("common_weight", "sum"))
        .reset_index()
    )
    national["national_prior_share"] = national["national_weighted_n"] / national["national_weighted_n"].sum()

    rows: list[pd.DataFrame] = []
    state_raw_n = work.groupby("state_po").size().to_dict()
    year = int(work["source_year"].iloc[0])
    all_states = sorted(work["state_po"].unique())
    for state in all_states:
        state_part = grouped[grouped["state_po"] == state].copy()
        merged = national.merge(
            state_part[["cell_id", "weighted_n", "raw_n", "weighted_share_raw"]],
            on="cell_id",
            how="left",
        )
        merged["weighted_n"] = merged["weighted_n"].fillna(0.0)
        merged["raw_n"] = merged["raw_n"].fillna(0).astype(int)
        merged["weighted_share_raw"] = merged["weighted_share_raw"].fillna(0.0)
        lam = state_raw_n[state] / (state_raw_n[state] + tau)
        merged["weighted_share_smoothed"] = (
            lam * merged["weighted_share_raw"] + (1.0 - lam) * merged["national_prior_share"]
        )
        merged["weighted_share_smoothed"] = merged["weighted_share_smoothed"] / merged[
            "weighted_share_smoothed"
        ].sum()
        merged["smoothing_lambda"] = lam
        merged["state_prior_share"] = merged["weighted_share_raw"]
        merged["year"] = year
        merged["state_po"] = state
        merged["cell_schema"] = cell_schema["name"]
        rows.append(merged)

    result = pd.concat(rows, ignore_index=True)
    ordered = [
        "year",
        "state_po",
        "cell_schema",
        "cell_id",
        *cell_feature_cols,
        "weighted_n",
        "raw_n",
        "weighted_share_raw",
        "weighted_share_smoothed",
        "smoothing_lambda",
        "national_prior_share",
        "state_prior_share",
    ]
    result = result[ordered]
    validate_probability_simplex(result, "state_po", "weighted_share_smoothed")
    return result


def build_ces_cells(
    config_path: str | Path,
    crosswalk_path: str | Path,
    cell_schema_path: str | Path,
    out_path: str | Path,
) -> dict[str, Path]:
    cfg = load_yaml(config_path)
    respondents = normalize_ces(config_path, crosswalk_path)
    cell_schema = load_cell_schema(cell_schema_path)
    smoothing = cfg.get("smoothing", {})
    dist = build_cell_distribution(respondents, cell_schema, tau=float(smoothing.get("tau", 500)))

    out = Path(out_path)
    if out.suffix:
        cell_path = out
        respondent_path = out.with_name("ces_respondents.parquet")
    else:
        ensure_dir(out)
        cell_path = out / "ces_cell_distribution.parquet"
        respondent_path = out / "ces_respondents.parquet"
    write_table(respondents, respondent_path)
    write_table(dist, cell_path)
    return {"respondents": respondent_path, "cell_distribution": cell_path}


def empirical_vote_probabilities(ces_df: pd.DataFrame, cell_cols: list[str]) -> pd.DataFrame:
    work = ces_df.copy()
    work["cell_id"] = work.apply(lambda row: make_cell_id(row.to_dict(), cell_cols), axis=1)
    grouped = (
        work.groupby(["state_po", "cell_id", "vote_choice_president"], dropna=False)
        .agg(weighted_n=("common_weight", "sum"))
        .reset_index()
    )
    totals = grouped.groupby(["state_po", "cell_id"])["weighted_n"].transform("sum")
    grouped["probability"] = np.where(totals > 0, grouped["weighted_n"] / totals, 0.0)
    return grouped
