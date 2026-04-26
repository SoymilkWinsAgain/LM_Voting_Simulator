"""Election metrics and eval table generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import write_table
from .mit import state_truth_table


def election_metrics(aggregate: pd.DataFrame, mit_results: pd.DataFrame, run_id: str) -> pd.DataFrame:
    truth = state_truth_table(mit_results)
    merged = aggregate.merge(truth[["state_po", "true_dem_2p", "true_margin", "true_winner"]], on="state_po")
    rows: list[dict[str, object]] = []
    for baseline, group in merged.groupby("baseline"):
        pred_dem_2p = group["dem_share_raw"] / (group["dem_share_raw"] + group["rep_share_raw"])
        pred_margin = group["margin_raw"]
        winner_correct = group["winner_raw"] == group["true_winner"]
        metrics = {
            "winner_accuracy": float(winner_correct.mean()),
            "vote_share_rmse": float(np.sqrt(((pred_dem_2p - group["true_dem_2p"]) ** 2).mean())),
            "margin_mae": float((pred_margin - group["true_margin"]).abs().mean()),
        }
        for name, value in metrics.items():
            rows.append(
                {
                    "run_id": run_id,
                    "metric_scope": "election",
                    "baseline": baseline,
                    "model_name": None,
                    "metric_name": name,
                    "metric_value": value,
                    "state_po": None,
                    "group_key": None,
                    "question_id": None,
                    "confidence_low": None,
                    "confidence_high": None,
                    "created_at": pd.Timestamp.now(tz="UTC"),
                }
            )
        for _, row in group.iterrows():
            rows.append(
                {
                    "run_id": run_id,
                    "metric_scope": "election",
                    "baseline": baseline,
                    "model_name": None,
                    "metric_name": "state_margin_error",
                    "metric_value": float(row["margin_raw"] - row["true_margin"]),
                    "state_po": row["state_po"],
                    "group_key": None,
                    "question_id": None,
                    "confidence_low": None,
                    "confidence_high": None,
                    "created_at": pd.Timestamp.now(tz="UTC"),
                }
            )
    return pd.DataFrame(rows)


def write_eval_metrics(
    aggregate: pd.DataFrame,
    mit_results: pd.DataFrame,
    run_id: str,
    out_path: str | Path,
) -> pd.DataFrame:
    df = election_metrics(aggregate, mit_results, run_id)
    write_table(df, out_path)
    return df
