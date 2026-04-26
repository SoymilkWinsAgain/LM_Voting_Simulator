"""CES/CCES normalization and weighted cell distributions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_cell_schema
from .io import ensure_dir, load_yaml, read_table, write_table
from .transforms import apply_transform, make_cell_id, normalize_vote
from .validation import require_columns, validate_categories, validate_probability_simplex


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
