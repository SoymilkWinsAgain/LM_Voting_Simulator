"""Response aggregation into state-level vote shares."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .baselines import VOTE_CODES
from .io import write_table


def _probability(row: pd.Series, code: str) -> float:
    probs = row.get("probabilities_json")
    if isinstance(probs, str) and probs:
        parsed = json.loads(probs)
        return float(parsed.get(code, 0.0))
    return 1.0 if row.get("parsed_answer_code") == code else 0.0


def aggregate_state_results(responses: pd.DataFrame, agents: pd.DataFrame, run_id: str, year: int) -> pd.DataFrame:
    merged = responses.merge(
        agents[["agent_id", "state_po", "sample_weight"]],
        on="agent_id",
        how="left",
        validate="many_to_one",
    )
    rows: list[dict[str, object]] = []
    for (state, baseline), group in merged.groupby(["state_po", "baseline"], dropna=False):
        weights = group["sample_weight"].astype(float)
        denom = float(weights.sum()) or 1.0
        shares = {
            code: float((group.apply(lambda row: _probability(row, code), axis=1) * weights).sum() / denom)
            for code in VOTE_CODES
        }
        two_party = shares["democrat"] + shares["republican"]
        if two_party:
            dem_2p = shares["democrat"] / two_party
            rep_2p = shares["republican"] / two_party
            margin = dem_2p - rep_2p
            winner = "democrat" if margin > 0 else "republican" if margin < 0 else "tie"
        else:
            margin = 0.0
            winner = "tie"
        rows.append(
            {
                "run_id": run_id,
                "year": year,
                "state_po": state,
                "baseline": baseline,
                "n_agents": int(group["agent_id"].nunique()),
                "dem_share_raw": shares["democrat"],
                "rep_share_raw": shares["republican"],
                "other_share_raw": shares["other"],
                "not_vote_or_unknown_share_raw": shares["not_vote_or_unknown"],
                "dem_share_calibrated": None,
                "rep_share_calibrated": None,
                "winner_raw": winner,
                "winner_calibrated": None,
                "margin_raw": margin,
                "margin_calibrated": None,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    return pd.DataFrame(rows)


def write_aggregate_state_results(
    responses: pd.DataFrame,
    agents: pd.DataFrame,
    run_id: str,
    year: int,
    out_path: str | Path,
) -> pd.DataFrame:
    df = aggregate_state_results(responses, agents, run_id, year)
    write_table(df, out_path)
    return df


def aggregate_turnout_vote_state_results(
    responses: pd.DataFrame,
    agents: pd.DataFrame,
    run_id: str,
    year: int,
    *,
    office: str = "president",
) -> pd.DataFrame:
    required = {"turnout_probability", "vote_prob_democrat", "vote_prob_republican", "vote_prob_other"}
    if not required <= set(responses.columns):
        legacy = aggregate_state_results(responses, agents, run_id, year)
        legacy["office"] = office
        return legacy
    agent_cols = ["agent_id", "state_po", "sample_weight"]
    if "weight_column" in agents.columns:
        agent_cols.append("weight_column")
    merged = responses.merge(agents[agent_cols], on="agent_id", how="left", validate="many_to_one")
    rows: list[dict[str, object]] = []
    for (state, baseline, model_name), group in merged.groupby(["state_po", "baseline", "model_name"], dropna=False):
        weights = group["sample_weight"].fillna(1.0).astype(float)
        turnout = group["turnout_probability"].fillna(0.0).astype(float).clip(0.0, 1.0)
        expected_turnout = float((weights * turnout).sum())
        expected_dem = float((weights * turnout * group["vote_prob_democrat"].fillna(0.0).astype(float)).sum())
        expected_rep = float((weights * turnout * group["vote_prob_republican"].fillna(0.0).astype(float)).sum())
        expected_other = float((weights * turnout * group["vote_prob_other"].fillna(0.0).astype(float)).sum())
        undecided = (
            group["vote_prob_undecided"].fillna(0.0).astype(float)
            if "vote_prob_undecided" in group.columns
            else pd.Series(0.0, index=group.index)
        )
        expected_undecided = float((weights * (1.0 - turnout)).sum() + (weights * turnout * undecided).sum())
        candidate_total = expected_dem + expected_rep + expected_other
        two_party = expected_dem + expected_rep
        dem_share_raw = expected_dem / candidate_total if candidate_total else 0.0
        rep_share_raw = expected_rep / candidate_total if candidate_total else 0.0
        other_share_raw = expected_other / candidate_total if candidate_total else 0.0
        dem_2p = expected_dem / two_party if two_party else 0.0
        rep_2p = expected_rep / two_party if two_party else 0.0
        margin = dem_2p - rep_2p
        winner = "democrat" if margin > 0 else "republican" if margin < 0 else "tie"
        weight_column = None
        if "weight_column" in group.columns:
            values = group["weight_column"].dropna().astype(str).unique()
            weight_column = values[0] if len(values) else None
        rows.append(
            {
                "run_id": run_id,
                "year": year,
                "state_po": state,
                "office": office,
                "baseline": baseline,
                "model_name": model_name,
                "n_agents": int(group["agent_id"].nunique()),
                "weight_column": weight_column,
                "expected_turnout": expected_turnout,
                "expected_dem_votes": expected_dem,
                "expected_rep_votes": expected_rep,
                "expected_other_votes": expected_other,
                "expected_undecided_or_not_vote": expected_undecided,
                "dem_share_raw": dem_share_raw,
                "rep_share_raw": rep_share_raw,
                "other_share_raw": other_share_raw,
                "dem_share_2p": dem_2p,
                "rep_share_2p": rep_2p,
                "margin_2p": margin,
                "winner": winner,
                "winner_raw": winner,
                "margin_raw": margin,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    return pd.DataFrame(rows)


def write_turnout_vote_state_results(
    responses: pd.DataFrame,
    agents: pd.DataFrame,
    run_id: str,
    year: int,
    out_path: str | Path,
    *,
    office: str = "president",
) -> pd.DataFrame:
    df = aggregate_turnout_vote_state_results(responses, agents, run_id, year, office=office)
    write_table(df, out_path)
    return df
