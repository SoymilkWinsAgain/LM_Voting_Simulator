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


def _metric_row(
    *,
    run_id: str,
    baseline: str,
    metric_scope: str,
    metric_name: str,
    metric_value: float,
    model_name: str | None = None,
    state_po: str | None = None,
    group_key: str | None = None,
    question_id: str | None = None,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "metric_scope": metric_scope,
        "baseline": baseline,
        "model_name": model_name,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "state_po": state_po,
        "group_key": group_key,
        "question_id": question_id,
        "confidence_low": None,
        "confidence_high": None,
        "created_at": pd.Timestamp.now(tz="UTC"),
    }


def individual_turnout_vote_metrics(
    responses: pd.DataFrame,
    targets: pd.DataFrame,
    run_id: str,
    *,
    turnout_target_id: str = "turnout_2024_self_report",
    vote_target_id: str = "president_vote_2024",
) -> pd.DataFrame:
    target_wide = targets.pivot_table(
        index="ces_id",
        columns="target_id",
        values="canonical_value",
        aggfunc="first",
    ).reset_index()
    merged = responses.merge(target_wide, left_on="base_ces_id", right_on="ces_id", how="left")
    rows: list[dict[str, object]] = []
    for (baseline, model_name), group in merged.groupby(["baseline", "model_name"], dropna=False):
        parse_ok = group["parse_status"] == "ok"
        rows.append(
            _metric_row(
                run_id=run_id,
                baseline=baseline,
                model_name=model_name,
                metric_scope="individual",
                metric_name="parse_ok_rate",
                metric_value=float(parse_ok.mean()) if len(group) else 0.0,
            )
        )
        if turnout_target_id in group.columns:
            known = group[group[turnout_target_id].isin(["voted", "not_voted"])].copy()
            if not known.empty:
                y = (known[turnout_target_id] == "voted").astype(float)
                p = known["turnout_probability"].fillna(0.0).astype(float).clip(0.0, 1.0)
                rows.extend(
                    [
                        _metric_row(
                            run_id=run_id,
                            baseline=baseline,
                            model_name=model_name,
                            metric_scope="individual",
                            metric_name="turnout_brier",
                            metric_value=float(((p - y) ** 2).mean()),
                            question_id=turnout_target_id,
                        ),
                        _metric_row(
                            run_id=run_id,
                            baseline=baseline,
                            model_name=model_name,
                            metric_scope="individual",
                            metric_name="turnout_accuracy_at_0_5",
                            metric_value=float(((p >= 0.5) == (y == 1.0)).mean()),
                            question_id=turnout_target_id,
                        ),
                    ]
                )
        if vote_target_id in group.columns:
            known_vote = group[group[vote_target_id].isin(["democrat", "republican", "other", "not_vote"])].copy()
            if not known_vote.empty:
                pred = known_vote["most_likely_choice"].replace({"not_vote": "not_vote"})
                rows.append(
                    _metric_row(
                        run_id=run_id,
                        baseline=baseline,
                        model_name=model_name,
                        metric_scope="individual",
                        metric_name="vote_accuracy",
                        metric_value=float((pred == known_vote[vote_target_id]).mean()),
                        question_id=vote_target_id,
                    )
                )
    return pd.DataFrame(rows)


def write_individual_turnout_vote_metrics(
    responses: pd.DataFrame,
    targets: pd.DataFrame,
    run_id: str,
    out_path: str | Path,
    *,
    turnout_target_id: str = "turnout_2024_self_report",
    vote_target_id: str = "president_vote_2024",
) -> pd.DataFrame:
    df = individual_turnout_vote_metrics(
        responses,
        targets,
        run_id,
        turnout_target_id=turnout_target_id,
        vote_target_id=vote_target_id,
    )
    write_table(df, out_path)
    return df


def empty_aggregate_metrics(run_id: str) -> pd.DataFrame:
    _ = run_id
    return pd.DataFrame(
        columns=[
            "run_id",
            "metric_scope",
            "baseline",
            "model_name",
            "metric_name",
            "metric_value",
            "state_po",
            "group_key",
            "question_id",
            "confidence_low",
            "confidence_high",
            "created_at",
        ]
    )
