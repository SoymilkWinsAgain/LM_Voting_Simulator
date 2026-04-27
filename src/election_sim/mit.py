"""MIT Election Lab presidential returns processing."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .io import ensure_dir, load_yaml, read_table, write_table
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

RETURN_COLUMNS = [
    "year",
    "office",
    "geo_level",
    "state",
    "state_po",
    "state_fips",
    "county_name",
    "county_fips",
    "geo_id",
    "candidate_raw",
    "candidate_norm",
    "party_raw",
    "party_norm",
    "major_choice",
    "candidatevotes",
    "totalvotes",
    "mode_policy_used",
    "source_modes_used",
    "source_file",
    "source_version",
    "audit_flags",
    "schema_version",
    "created_at",
]

TRUTH_COLUMNS = [
    "year",
    "office",
    "geo_level",
    "state_po",
    "county_fips",
    "county_name",
    "geo_id",
    "dem_votes",
    "rep_votes",
    "other_votes",
    "candidate_total_votes",
    "two_party_total_votes",
    "totalvotes",
    "dem_share_raw",
    "rep_share_raw",
    "other_share_raw",
    "dem_share_2p",
    "rep_share_2p",
    "margin_2p",
    "winner",
    "truth_source",
    "source_version",
    "audit_flags",
    "schema_version",
    "created_at",
]

AUDIT_COLUMNS = [
    "audit_type",
    "severity",
    "year",
    "state_po",
    "county_fips",
    "count",
    "details",
    "source_file",
    "created_at",
]

ADMIN_CANDIDATE_PATTERNS = {
    "TOTAL VOTES CAST",
    "OVERVOTES",
    "UNDERVOTES",
    "SPOILED",
    "BLANK VOTES",
}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _upper(value: Any) -> str:
    return re.sub(r"\s+", " ", _clean(value).upper()).strip()


def _candidate_key(value: Any) -> str:
    text = re.sub(r"[^A-Z0-9]+", " ", _upper(value))
    return re.sub(r"\s+", " ", text).strip()


def _state_fips(value: Any) -> str | None:
    text = _clean(value)
    if not text:
        return None
    try:
        text = str(int(float(text)))
    except ValueError:
        pass
    return text.zfill(2)


def _county_fips(value: Any) -> str | None:
    text = _clean(value)
    if not text:
        return None
    try:
        text = str(int(float(text)))
    except ValueError:
        pass
    return text.zfill(5)


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _stable_join(values: Any) -> str:
    if isinstance(values, pd.Series):
        values = values.tolist()
    cleaned = sorted({_clean(value) for value in values if _clean(value)})
    return "|".join(cleaned)


def simplify_party(value: Any, candidate: Any = None) -> str:
    key = _upper(value)
    cand = _candidate_key(candidate)
    if key in {"DEMOCRAT", "DEMOCRATIC", "DEM"} or cand in {
        "KAMALA D HARRIS",
        "KAMALA HARRIS",
        "BIDEN JOSEPH R JR",
        "JOSEPH R BIDEN JR",
        "JOE BIDEN",
    }:
        return "democrat"
    if key in {"REPUBLICAN", "GOP", "REP"} or cand in {"DONALD J TRUMP", "DONALD TRUMP"}:
        return "republican"
    if key == "LIBERTARIAN":
        return "libertarian"
    return "other"


def _major_choice_from_party(party_norm: str) -> str:
    if party_norm == "democrat":
        return "democrat"
    if party_norm == "republican":
        return "republican"
    return "other"


def _load_candidate_crosswalk(path: str | Path | None) -> dict[tuple[int, str], dict[str, str]]:
    if path is None:
        return {}
    cfg = load_yaml(path)
    rows = cfg.get("candidate_crosswalk", [])
    out: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        year = int(row["year"])
        for pattern in row.get("candidate_patterns", []):
            out[(year, _candidate_key(pattern))] = {
                "candidate_norm": _upper(row["candidate_norm"]),
                "major_choice": str(row["major_choice"]).lower(),
            }
    return out


def _candidate_mapping(
    *,
    year: int,
    candidate: Any,
    party: Any,
    crosswalk: dict[tuple[int, str], dict[str, str]],
) -> tuple[str, str, str]:
    key = _candidate_key(candidate)
    party_norm = simplify_party(party, candidate)
    mapped = crosswalk.get((year, key))
    if mapped:
        return mapped["candidate_norm"], party_norm, mapped["major_choice"]
    return _upper(candidate) or "UNKNOWN", party_norm, _major_choice_from_party(party_norm)


def _is_admin_candidate(candidate: Any, configured_patterns: list[str] | None = None) -> bool:
    key = _candidate_key(candidate)
    patterns = {_candidate_key(value) for value in (configured_patterns or [])} | ADMIN_CANDIDATE_PATTERNS
    return key in patterns


def _required_columns(raw: pd.DataFrame, columns: dict[str, str], logical: list[str], name: str) -> None:
    physical = [columns.get(col, col) for col in logical]
    require_columns(raw, physical, name)


def _audit_row(
    audit_type: str,
    *,
    severity: str = "info",
    year: int | None = None,
    state_po: str | None = None,
    county_fips: str | None = None,
    count: int = 1,
    details: str = "",
    source_file: str | None = None,
) -> dict[str, Any]:
    return {
        "audit_type": audit_type,
        "severity": severity,
        "year": year,
        "state_po": state_po,
        "county_fips": county_fips,
        "count": count,
        "details": details,
        "source_file": source_file,
        "created_at": pd.Timestamp.now(tz="UTC"),
    }


def _normalize_mit_county_president_with_audit(
    config_path: str | Path,
    candidate_crosswalk_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_yaml(config_path)
    raw = read_table(cfg["path"])
    columns = cfg.get("columns", {})
    _required_columns(
        raw,
        columns,
        ["state", "county_name", "year", "state_po", "county_fips", "office", "candidate", "party", "candidatevotes", "totalvotes", "version", "mode"],
        "mit_county_president_raw",
    )
    crosswalk = _load_candidate_crosswalk(candidate_crosswalk_path)
    source_file = str(cfg["path"])
    exclude_patterns = cfg.get("exclude_candidate_patterns", [])
    audit_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    valid_state_fips: dict[str, str] = {}
    for _, row in raw.iterrows():
        state_po = _upper(row[columns.get("state_po", "state_po")])
        county_fips = _county_fips(row[columns.get("county_fips", "county_fips")])
        if county_fips and len(county_fips) == 5:
            valid_state_fips.setdefault(state_po, county_fips[:2])
    synthetic_fips: dict[tuple[str, str, str], str] = {}
    synthetic_counts: dict[str, int] = {}

    for raw_idx, row in raw.iterrows():
        year = int(row[columns.get("year", "year")])
        candidate = row[columns.get("candidate", "candidate")]
        state_po = _upper(row[columns.get("state_po", "state_po")])
        raw_county_fips = _clean(row[columns.get("county_fips", "county_fips")])
        county_fips = _county_fips(raw_county_fips)
        county_name = _upper(row[columns.get("county_name", "county_name")])
        row_audit_flags: list[str] = []
        if not county_fips or len(county_fips) != 5:
            key = (state_po, county_name, raw_county_fips)
            if key not in synthetic_fips:
                state_fips = valid_state_fips.get(state_po)
                if not state_fips:
                    raise ValueError(f"Cannot assign synthetic county_fips for {state_po} {county_name}")
                idx = synthetic_counts.get(state_po, 0)
                synthetic_counts[state_po] = idx + 1
                synthetic_fips[key] = f"{state_fips}{900 + idx:03d}"
            county_fips = synthetic_fips[key]
            row_audit_flags.append("county_fips_synthetic")
            audit_rows.append(
                _audit_row(
                    "county_fips_synthetic",
                    severity="warning",
                    year=year,
                    state_po=state_po,
                    county_fips=county_fips,
                    details=f"county_name={county_name};raw_county_fips={raw_county_fips or '<missing>'}",
                    source_file=source_file,
                )
            )
        if _is_admin_candidate(candidate, exclude_patterns):
            audit_rows.append(
                _audit_row(
                    "administrative_row_excluded",
                    year=year,
                    state_po=state_po,
                    county_fips=county_fips,
                    details=f"candidate={candidate}",
                    source_file=source_file,
                )
            )
            continue
        candidate_norm, party_norm, major_choice = _candidate_mapping(
            year=year,
            candidate=candidate,
            party=row[columns.get("party", "party")],
            crosswalk=crosswalk,
        )
        candidatevotes = _numeric(row[columns.get("candidatevotes", "candidatevotes")])
        audit_flags: list[str] = list(row_audit_flags)
        if candidatevotes is None:
            if major_choice in {"democrat", "republican"}:
                raise ValueError(
                    f"Missing major-party candidatevotes in MIT county file at row {raw_idx}: "
                    f"{year} {state_po} {county_fips} {candidate}"
                )
            candidatevotes = 0.0
            audit_flags.append("candidatevotes_missing_filled_zero")
            audit_rows.append(
                _audit_row(
                    "candidatevotes_missing_filled_zero",
                    severity="warning",
                    year=year,
                    state_po=state_po,
                    county_fips=county_fips,
                    details=f"candidate={candidate}",
                    source_file=source_file,
                )
            )
        totalvotes = _numeric(row[columns.get("totalvotes", "totalvotes")]) or 0.0
        mode = _upper(row[columns.get("mode", "mode")]) or "UNSPECIFIED"
        normalized_rows.append(
            {
                "raw_row_id": raw_idx,
                "year": year,
                "office": "president",
                "geo_level": "county",
                "state": _upper(row[columns.get("state", "state")]),
                "state_po": state_po,
                "state_fips": county_fips[:2] if county_fips else None,
                "county_name": county_name,
                "county_fips": county_fips,
                "geo_id": f"county:{state_po}:{county_fips}",
                "candidate_raw": _clean(candidate),
                "candidate_norm": candidate_norm,
                "party_raw": _clean(row[columns.get("party", "party")]),
                "party_norm": party_norm,
                "major_choice": major_choice,
                "candidatevotes": float(candidatevotes),
                "totalvotes": float(totalvotes),
                "mode_raw": mode,
                "source_file": source_file,
                "source_version": _clean(row[columns.get("version", "version")]),
                "audit_flags": "|".join(audit_flags),
            }
        )
    normalized = pd.DataFrame(normalized_rows)
    if normalized.empty:
        return pd.DataFrame(columns=RETURN_COLUMNS), pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)

    unit_cols = ["year", "state_po", "county_fips", "candidate_norm", "major_choice"]
    return_rows: list[dict[str, Any]] = []
    for _, group in normalized.groupby(unit_cols, dropna=False):
        total_rows = group[group["mode_raw"] == "TOTAL"]
        if not total_rows.empty:
            selected = total_rows
            mode_policy = "total_row"
        elif len(group) == 1:
            selected = group
            mode_policy = "single_mode"
        else:
            selected = group
            mode_policy = "summed_modes"
        first = selected.iloc[0]
        return_rows.append(
            {
                "year": int(first["year"]),
                "office": "president",
                "geo_level": "county",
                "state": first["state"],
                "state_po": first["state_po"],
                "state_fips": first["state_fips"],
                "county_name": first["county_name"],
                "county_fips": first["county_fips"],
                "geo_id": first["geo_id"],
                "candidate_raw": _stable_join(selected["candidate_raw"]),
                "candidate_norm": first["candidate_norm"],
                "party_raw": _stable_join(selected["party_raw"]),
                "party_norm": first["party_norm"],
                "major_choice": first["major_choice"],
                "candidatevotes": float(selected["candidatevotes"].sum()),
                "totalvotes": float(selected["totalvotes"].max()),
                "mode_policy_used": mode_policy,
                "source_modes_used": _stable_join(selected["mode_raw"]),
                "source_file": source_file,
                "source_version": _stable_join(selected["source_version"]),
                "audit_flags": _stable_join(selected["audit_flags"]),
                "schema_version": "mit_election_returns_county_v1",
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    returns = pd.DataFrame(return_rows, columns=RETURN_COLUMNS)
    mode_summary = (
        returns.groupby(["year", "state_po", "mode_policy_used"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    for _, row in mode_summary.iterrows():
        audit_rows.append(
            _audit_row(
                "mode_policy_summary",
                year=int(row["year"]),
                state_po=row["state_po"],
                count=int(row["count"]),
                details=f"mode_policy={row['mode_policy_used']}",
                source_file=source_file,
            )
        )
    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    return returns, audit


def normalize_mit_county_president(
    config_path: str | Path,
    candidate_crosswalk_path: str | Path | None = None,
) -> pd.DataFrame:
    returns, _ = _normalize_mit_county_president_with_audit(config_path, candidate_crosswalk_path)
    return returns


def _normalize_mit_state_president_with_audit(
    config_path: str | Path,
    candidate_crosswalk_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_yaml(config_path)
    raw = read_table(cfg["path"])
    columns = cfg.get("columns", {})
    _required_columns(
        raw,
        columns,
        ["year", "state", "state_po", "state_fips", "office", "candidate", "party_detailed", "party_simplified", "candidatevotes", "totalvotes", "version"],
        "mit_state_president_raw",
    )
    crosswalk = _load_candidate_crosswalk(candidate_crosswalk_path)
    source_file = str(cfg["path"])
    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        year = int(row[columns.get("year", "year")])
        candidate = row[columns.get("candidate", "candidate")]
        party_raw = row[columns.get("party_simplified", "party_simplified")]
        candidate_norm, party_norm, major_choice = _candidate_mapping(
            year=year,
            candidate=candidate,
            party=party_raw,
            crosswalk=crosswalk,
        )
        candidatevotes = _numeric(row[columns.get("candidatevotes", "candidatevotes")])
        if candidatevotes is None:
            if major_choice in {"democrat", "republican"}:
                raise ValueError(f"Missing major-party state candidatevotes: {year} {row['state_po']} {candidate}")
            candidatevotes = 0.0
            audit_flags = "candidatevotes_missing_filled_zero"
        else:
            audit_flags = ""
        state_po = _upper(row[columns.get("state_po", "state_po")])
        rows.append(
            {
                "year": year,
                "office": "president",
                "geo_level": "state",
                "state": _upper(row[columns.get("state", "state")]),
                "state_po": state_po,
                "state_fips": _state_fips(row[columns.get("state_fips", "state_fips")]),
                "county_name": None,
                "county_fips": None,
                "geo_id": f"state:{state_po}",
                "candidate_raw": _clean(candidate),
                "candidate_norm": candidate_norm,
                "party_raw": _clean(row[columns.get("party_detailed", "party_detailed")]),
                "party_norm": party_norm,
                "major_choice": major_choice,
                "candidatevotes": float(candidatevotes),
                "totalvotes": float(_numeric(row[columns.get("totalvotes", "totalvotes")]) or 0.0),
                "mode_policy_used": "state_file",
                "source_modes_used": "state_file",
                "source_file": source_file,
                "source_version": _clean(row[columns.get("version", "version")]),
                "audit_flags": audit_flags,
                "schema_version": "mit_election_returns_state_v1",
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    returns = pd.DataFrame(rows, columns=RETURN_COLUMNS)
    audit = pd.DataFrame(
        [
            _audit_row(
                "state_file_summary",
                year=int(year),
                count=int(len(group)),
                details=f"states={group['state_po'].nunique()}",
                source_file=source_file,
            )
            for year, group in returns.groupby("year")
        ],
        columns=AUDIT_COLUMNS,
    )
    return returns, audit


def normalize_mit_state_president(
    config_path: str | Path,
    candidate_crosswalk_path: str | Path | None = None,
) -> pd.DataFrame:
    returns, _ = _normalize_mit_state_president_with_audit(config_path, candidate_crosswalk_path)
    return returns


def _totalvotes_for_truth(returns: pd.DataFrame, group_cols: list[str], geo_level: str) -> pd.DataFrame:
    if geo_level == "state" and returns["geo_level"].eq("county").all():
        totals = (
            returns.drop_duplicates(["year", "office", "state_po", "county_fips"])[
                ["year", "office", "state_po", "totalvotes"]
            ]
            .groupby(["year", "office", "state_po"], dropna=False)["totalvotes"]
            .sum()
            .reset_index()
        )
        return totals
    return returns.groupby(group_cols, dropna=False)["totalvotes"].max().reset_index()


def build_president_truth(
    returns: pd.DataFrame,
    geo_level: str,
    years: list[int] | set[int] | None = None,
) -> pd.DataFrame:
    work = returns.copy()
    if years is not None:
        work = work[work["year"].isin({int(year) for year in years})].copy()
    if work.empty:
        return pd.DataFrame(columns=TRUTH_COLUMNS)
    if geo_level == "county":
        group_cols = ["year", "office", "state_po", "county_fips", "county_name", "geo_id"]
        truth_source = "mit_county_file"
    elif geo_level == "state":
        group_cols = ["year", "office", "state_po"]
        truth_source = "mit_county_rollup" if work["geo_level"].eq("county").all() else "mit_state_file"
    else:
        raise ValueError(f"Unsupported MIT truth geo_level: {geo_level}")

    votes = (
        work.groupby([*group_cols, "major_choice"], dropna=False)["candidatevotes"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["democrat", "republican", "other"]:
        if col not in votes.columns:
            votes[col] = 0.0
    totals = _totalvotes_for_truth(work, group_cols, geo_level)
    meta = (
        work.groupby(group_cols, dropna=False)
        .agg(source_version=("source_version", _stable_join), audit_flags=("audit_flags", _stable_join))
        .reset_index()
    )
    out = votes.merge(totals, on=group_cols, how="left").merge(meta, on=group_cols, how="left")
    if geo_level == "state":
        out["county_fips"] = None
        out["county_name"] = None
        out["geo_id"] = "state:" + out["state_po"].astype(str)
    out = out.rename(columns={"democrat": "dem_votes", "republican": "rep_votes", "other": "other_votes"})
    out["candidate_total_votes"] = out["dem_votes"] + out["rep_votes"] + out["other_votes"]
    out["two_party_total_votes"] = out["dem_votes"] + out["rep_votes"]
    out["dem_share_raw"] = out["dem_votes"] / out["candidate_total_votes"].replace(0, pd.NA)
    out["rep_share_raw"] = out["rep_votes"] / out["candidate_total_votes"].replace(0, pd.NA)
    out["other_share_raw"] = out["other_votes"] / out["candidate_total_votes"].replace(0, pd.NA)
    out["dem_share_2p"] = out["dem_votes"] / out["two_party_total_votes"].replace(0, pd.NA)
    out["rep_share_2p"] = out["rep_votes"] / out["two_party_total_votes"].replace(0, pd.NA)
    out["margin_2p"] = out["dem_share_2p"] - out["rep_share_2p"]
    out["winner"] = out["margin_2p"].map(
        lambda m: "unknown" if pd.isna(m) else "democrat" if m > 0 else "republican" if m < 0 else "tie"
    )
    out["geo_level"] = geo_level
    out["truth_source"] = truth_source
    out["schema_version"] = f"mit_president_{geo_level}_truth_v1"
    out["created_at"] = pd.Timestamp.now(tz="UTC")
    return out[TRUTH_COLUMNS]


def build_historical_features(truth: pd.DataFrame, *, max_year: int = 2020) -> pd.DataFrame:
    work = truth[truth["year"] <= max_year].copy()
    if work.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for _, group in work.sort_values(["geo_level", "geo_id", "year"]).groupby(["geo_level", "geo_id"], dropna=False):
        part = group.copy()
        margin = pd.to_numeric(part["margin_2p"], errors="coerce")
        part["margin_2p"] = margin
        part["swing_from_previous"] = margin - margin.shift(1)
        part["avg_margin_last_2"] = margin.rolling(2, min_periods=1).mean()
        part["avg_margin_last_3"] = margin.rolling(3, min_periods=1).mean()
        part["partisan_baseline"] = part["avg_margin_last_3"]
        rows.append(part)
    features = pd.concat(rows, ignore_index=True)
    features = features.rename(columns={"totalvotes": "turnout_total"})
    features["source_truth_table"] = features["truth_source"]
    features["schema_version"] = "mit_president_historical_features_v1"
    return features[
        [
            "year",
            "state_po",
            "county_fips",
            "geo_level",
            "geo_id",
            "dem_share_2p",
            "rep_share_2p",
            "margin_2p",
            "winner",
            "turnout_total",
            "swing_from_previous",
            "avg_margin_last_2",
            "avg_margin_last_3",
            "partisan_baseline",
            "source_truth_table",
            "schema_version",
        ]
    ]


def _compare_2024_reference(state_truth: pd.DataFrame, reference_path: str | Path | None) -> pd.DataFrame:
    if not reference_path:
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    ref = read_table(reference_path)
    truth_2024 = state_truth[state_truth["year"] == 2024]
    merged = truth_2024.merge(ref, on="state_po", suffixes=("_truth", "_reference"))
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        dem_diff = abs(float(row["dem_votes_truth"]) - float(row["dem_votes_reference"]))
        rep_diff = abs(float(row["rep_votes_truth"]) - float(row["rep_votes_reference"]))
        share_diff = abs(float(row["dem_share_2p_truth"]) - float(row["dem_share_2p_reference"]))
        if dem_diff > 0 or rep_diff > 0 or share_diff > 1e-10:
            rows.append(
                _audit_row(
                    "state_truth_reference_mismatch",
                    severity="error",
                    year=2024,
                    state_po=row["state_po"],
                    details=f"dem_diff={dem_diff};rep_diff={rep_diff};dem_share_2p_diff={share_diff}",
                    source_file=str(reference_path),
                )
            )
    rows.append(
        _audit_row(
            "state_truth_reference_comparison",
            year=2024,
            count=int(len(merged)),
            details="2024 county-derived state truth compared against validation reference",
            source_file=str(reference_path),
        )
    )
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def _write_mit_report(
    *,
    out_path: Path,
    county_returns: pd.DataFrame,
    state_returns: pd.DataFrame,
    state_truth: pd.DataFrame,
    county_truth: pd.DataFrame,
    historical_features: pd.DataFrame,
    audit: pd.DataFrame,
    config_path: str | Path,
) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        display = df.copy()
        for col in display.columns:
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else str(value))
        headers = list(display.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for _, row in display.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)

    mode_summary = (
        county_returns.groupby(["year", "state_po", "mode_policy_used"], dropna=False)
        .size()
        .reset_index(name="n_returns")
        .head(80)
    )
    audit_summary = audit.groupby(["audit_type", "severity"], dropna=False)["count"].sum().reset_index()
    lines = [
        "# MIT Election Lab Ingest Report",
        "",
        f"- Config: `{config_path}`",
        f"- County returns: {len(county_returns)} rows",
        f"- State returns: {len(state_returns)} rows",
        f"- State truth: {len(state_truth)} rows; 2024 states: {state_truth[state_truth['year'] == 2024]['state_po'].nunique()}",
        f"- County truth: {len(county_truth)} rows",
        f"- Historical features: {len(historical_features)} rows; max year: {historical_features['year'].max() if not historical_features.empty else 'none'}",
        "",
        "## Truth policy",
        "- 2024 state truth uses `mit_county_rollup` from `countypres_2000-2024.csv`.",
        "- 1976-2020 state truth uses `mit_state_file` from `1976-2024-president.csv`.",
        "- `2024-better-evaluation.csv` is validation reference only.",
        "",
        "## Mode policy summary",
        markdown_table(mode_summary),
        "",
        "## Audit summary",
        markdown_table(audit_summary),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_mit_processed_artifacts(config_path: str | Path, out_dir: str | Path | None = None) -> dict[str, Path]:
    cfg = load_yaml(config_path)
    outputs = cfg.get("outputs", {})
    out = ensure_dir(out_dir or outputs.get("out_dir", "data/processed/mit"))
    candidate_crosswalk = cfg.get("crosswalks", {}).get("candidate")
    county_config = cfg.get("inputs", {}).get("county", {}).get("config")
    state_config = cfg.get("inputs", {}).get("state", {}).get("config")
    if not county_config or not state_config:
        raise ValueError("MIT president returns config requires inputs.county.config and inputs.state.config")

    county_returns, county_audit = _normalize_mit_county_president_with_audit(county_config, candidate_crosswalk)
    state_returns, state_audit = _normalize_mit_state_president_with_audit(state_config, candidate_crosswalk)
    county_truth = build_president_truth(county_returns, "county")
    state_truth_from_state = build_president_truth(state_returns, "state")
    state_truth_from_county = build_president_truth(county_returns, "state", years=[2024])
    state_truth = pd.concat(
        [state_truth_from_state[state_truth_from_state["year"] != 2024], state_truth_from_county],
        ignore_index=True,
    ).sort_values(["year", "state_po"], ignore_index=True)
    historical_features = build_historical_features(pd.concat([state_truth, county_truth], ignore_index=True))
    reference_audit = _compare_2024_reference(state_truth, cfg.get("validation_reference"))
    audit = pd.concat([county_audit, state_audit, reference_audit], ignore_index=True)

    paths = {
        "county_returns": out / outputs.get("county_returns", "election_returns_county_2000_2024.parquet"),
        "state_returns": out / outputs.get("state_returns", "election_returns_state_1976_2024.parquet"),
        "state_truth": out / outputs.get("state_truth", "president_state_truth.parquet"),
        "county_truth": out / outputs.get("county_truth", "president_county_truth.parquet"),
        "historical_features": out / outputs.get("historical_features", "president_historical_features.parquet"),
        "audit": out / outputs.get("audit", "mit_ingest_audit.parquet"),
        "report": out / outputs.get("report", "mit_ingest_report.md"),
    }
    write_table(county_returns, paths["county_returns"])
    write_table(state_returns, paths["state_returns"])
    write_table(state_truth, paths["state_truth"])
    write_table(county_truth, paths["county_truth"])
    write_table(historical_features, paths["historical_features"])
    write_table(audit, paths["audit"])
    _write_mit_report(
        out_path=paths["report"],
        county_returns=county_returns,
        state_returns=state_returns,
        state_truth=state_truth,
        county_truth=county_truth,
        historical_features=historical_features,
        audit=audit,
        config_path=config_path,
    )
    return paths


def _legacy_normalize_mit_results(config_path: str | Path, year: int) -> pd.DataFrame:
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


def _returns_to_legacy(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in results.iterrows():
        rows.append(
            {
                "year": int(row["year"]),
                "office": row["office"],
                "level": row["geo_level"],
                "state_po": row["state_po"],
                "state_fips": row["state_fips"],
                "county_name": row["county_name"],
                "county_fips": row["county_fips"],
                "candidate": row["candidate_norm"],
                "party_detailed": row["party_raw"],
                "party_simplified": row["major_choice"],
                "candidatevotes": int(row["candidatevotes"]),
                "totalvotes": int(row["totalvotes"]),
                "two_party_votes": None,
                "two_party_share_dem": None,
                "two_party_share_rep": None,
                "source_file": row["source_file"],
            }
        )
    return pd.DataFrame(rows, columns=MIT_COLUMNS)


def normalize_mit_results(config_path: str | Path, year: int) -> pd.DataFrame:
    cfg = load_yaml(config_path)
    if cfg.get("geo_level") == "county" and "mode_policy" in cfg:
        returns = normalize_mit_county_president(config_path)
        return _returns_to_legacy(returns[returns["year"] == int(year)])
    if cfg.get("geo_level") == "state":
        returns = normalize_mit_state_president(config_path)
        return _returns_to_legacy(returns[returns["year"] == int(year)])
    return _legacy_normalize_mit_results(config_path, year)


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
    if {"dem_votes", "rep_votes", "other_votes", "dem_share_2p", "margin_2p", "winner"} <= set(results.columns):
        grouped = results.copy()
        if "geo_level" in grouped.columns:
            grouped = grouped[grouped["geo_level"] == "state"].copy()
        out = grouped.rename(
            columns={
                "dem_votes": "true_dem_votes",
                "rep_votes": "true_rep_votes",
                "other_votes": "true_other_votes",
                "dem_share_2p": "true_dem_2p",
                "rep_share_2p": "true_rep_2p",
                "margin_2p": "true_margin",
                "winner": "true_winner",
            }
        )
        if "two_party_total_votes" not in out.columns:
            out["two_party_total_votes"] = out["true_dem_votes"] + out["true_rep_votes"]
        return out[
            [
                "year",
                "state_po",
                "true_dem_votes",
                "true_rep_votes",
                "true_other_votes",
                "two_party_total_votes",
                "true_dem_2p",
                "true_rep_2p",
                "true_margin",
                "true_winner",
                *[col for col in ["truth_source", "source_version", "audit_flags"] if col in out.columns],
            ]
        ]

    grouped = (
        results.groupby(["year", "state_po", "party_simplified"], dropna=False)["candidatevotes"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["democrat", "republican", "other"]:
        if col not in grouped.columns:
            grouped[col] = 0
    grouped["true_dem_votes"] = grouped["democrat"]
    grouped["true_rep_votes"] = grouped["republican"]
    grouped["true_other_votes"] = grouped["other"]
    grouped["two_party_total_votes"] = grouped["true_dem_votes"] + grouped["true_rep_votes"]
    grouped["true_dem_2p"] = grouped["true_dem_votes"] / grouped["two_party_total_votes"]
    grouped["true_rep_2p"] = grouped["true_rep_votes"] / grouped["two_party_total_votes"]
    grouped["true_margin"] = grouped["true_dem_2p"] - grouped["true_rep_2p"]
    grouped["true_winner"] = grouped["true_margin"].map(
        lambda margin: "democrat" if margin > 0 else "republican" if margin < 0 else "tie"
    )
    return grouped[
        [
            "year",
            "state_po",
            "true_dem_votes",
            "true_rep_votes",
            "true_other_votes",
            "two_party_total_votes",
            "true_dem_2p",
            "true_rep_2p",
            "true_margin",
            "true_winner",
        ]
    ]
