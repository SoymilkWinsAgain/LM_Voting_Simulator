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
