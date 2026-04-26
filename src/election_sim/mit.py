"""MIT Election Lab result normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io import load_yaml, read_table, write_table
from .validation import require_columns


MIT_COLUMNS = [
    "year",
    "office",
    "level",
    "state_po",
    "state_fips",
    "county_name",
    "county_fips",
    "candidate",
    "party_detailed",
    "party_simplified",
    "candidatevotes",
    "totalvotes",
    "two_party_votes",
    "two_party_share_dem",
    "two_party_share_rep",
    "source_file",
]


def simplify_party(value: Any, candidate: Any = None) -> str:
    key = str(value or "").strip().lower()
    cand = str(candidate or "").strip().lower()
    if key in {"democrat", "democratic", "dem"} or cand in {"kamala harris", "joe biden"}:
        return "democrat"
    if key in {"republican", "gop", "rep"} or cand == "donald trump":
        return "republican"
    return "other"


def normalize_mit_results(config_path: str | Path, year: int) -> pd.DataFrame:
    cfg = load_yaml(config_path)
    raw = read_table(cfg["path"])
    columns = cfg.get("columns", {})
    level = cfg.get("level", "county")
    office = cfg.get("office", "president")
    source_file = str(cfg["path"])

    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        candidate = row[columns.get("candidate", "candidate")]
        party_detailed = row[columns.get("party_detailed", "party_detailed")]
        totalvotes = int(row[columns.get("totalvotes", "totalvotes")])
        candidatevotes = int(row[columns.get("candidatevotes", "candidatevotes")])
        rows.append(
            {
                "year": year,
                "office": office,
                "level": level,
                "state_po": row[columns.get("state_po", "state_po")],
                "state_fips": row[columns.get("state_fips", "state_fips")]
                if columns.get("state_fips", "state_fips") in raw.columns
                else None,
                "county_name": row[columns.get("county_name", "county_name")]
                if columns.get("county_name", "county_name") in raw.columns
                else None,
                "county_fips": row[columns.get("county_fips", "county_fips")]
                if columns.get("county_fips", "county_fips") in raw.columns
                else None,
                "candidate": candidate,
                "party_detailed": party_detailed,
                "party_simplified": simplify_party(party_detailed, candidate),
                "candidatevotes": candidatevotes,
                "totalvotes": totalvotes,
                "two_party_votes": None,
                "two_party_share_dem": None,
                "two_party_share_rep": None,
                "source_file": source_file,
            }
        )
    df = pd.DataFrame(rows, columns=MIT_COLUMNS)

    major = df[df["party_simplified"].isin(["democrat", "republican"])]
    state_major = (
        major.groupby(["state_po", "party_simplified"], dropna=False)["candidatevotes"].sum().unstack(fill_value=0)
    )
    for state, shares in state_major.iterrows():
        dem = float(shares.get("democrat", 0))
        rep = float(shares.get("republican", 0))
        denom = dem + rep
        mask = df["state_po"] == state
        df.loc[mask, "two_party_votes"] = int(denom)
        df.loc[mask, "two_party_share_dem"] = dem / denom if denom else None
        df.loc[mask, "two_party_share_rep"] = rep / denom if denom else None
    validate_mit_results(df)
    return df


def validate_mit_results(df: pd.DataFrame) -> None:
    require_columns(df, MIT_COLUMNS, "mit_election_results")
    invalid = set(df["party_simplified"]) - {"democrat", "republican", "other"}
    if invalid:
        raise ValueError(f"Invalid MIT party values: {sorted(invalid)}")


def build_mit_results(config_path: str | Path, year: int, out_path: str | Path) -> Path:
    df = normalize_mit_results(config_path, year)
    write_table(df, out_path)
    return Path(out_path)


def state_truth_table(results: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results.groupby(["year", "state_po", "party_simplified"], dropna=False)["candidatevotes"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["democrat", "republican", "other"]:
        if col not in grouped.columns:
            grouped[col] = 0
    grouped["two_party_total"] = grouped["democrat"] + grouped["republican"]
    grouped["true_dem_2p"] = grouped["democrat"] / grouped["two_party_total"]
    grouped["true_rep_2p"] = grouped["republican"] / grouped["two_party_total"]
    grouped["true_margin"] = grouped["true_dem_2p"] - grouped["true_rep_2p"]
    grouped["true_winner"] = grouped["true_margin"].map(
        lambda margin: "democrat" if margin > 0 else "republican" if margin < 0 else "tie"
    )
    return grouped
