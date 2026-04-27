"""Election metrics and eval table generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, roc_auc_score

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
    n: int | None = None,
    small_n: bool | None = None,
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
        "n": n,
        "small_n": small_n,
        "confidence_low": None,
        "confidence_high": None,
        "created_at": pd.Timestamp.now(tz="UTC"),
    }


def _safe_auc(y: pd.Series, p: pd.Series) -> float | None:
    if y.nunique() < 2:
        return None
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return None


VOTE_EVAL_CLASSES = ["democrat", "not_vote", "other", "republican"]


def _vote_probability_matrix(group: pd.DataFrame) -> np.ndarray:
    turnout = group["turnout_probability"].fillna(0.0).astype(float).clip(0.0, 1.0)
    probs = pd.DataFrame(
        {
            "democrat": turnout * group["vote_prob_democrat"].fillna(0.0).astype(float).clip(0.0, 1.0),
            "republican": turnout * group["vote_prob_republican"].fillna(0.0).astype(float).clip(0.0, 1.0),
            "other": turnout * group["vote_prob_other"].fillna(0.0).astype(float).clip(0.0, 1.0),
            "not_vote": 1.0 - turnout,
        },
        index=group.index,
    )
    denom = probs.sum(axis=1).replace(0.0, 1.0)
    return probs.div(denom, axis=0)[VOTE_EVAL_CLASSES].to_numpy()


def _turnout_vote_metric_rows(
    group: pd.DataFrame,
    *,
    run_id: str,
    baseline: str,
    model_name: str | None,
    metric_scope: str,
    group_key: str | None,
    turnout_target_id: str,
    vote_target_id: str,
    small_n_threshold: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_group = int(len(group))
    small_n = n_group < small_n_threshold if metric_scope == "subgroup" else None
    parse_ok = group["parse_status"] == "ok"
    rows.append(
        _metric_row(
            run_id=run_id,
            baseline=baseline,
            model_name=model_name,
            metric_scope=metric_scope,
            metric_name="parse_ok_rate",
            metric_value=float(parse_ok.mean()) if n_group else 0.0,
            group_key=group_key,
            n=n_group,
            small_n=small_n,
        )
    )
    if turnout_target_id in group.columns:
        known = group[group[turnout_target_id].isin(["voted", "not_voted"])].copy()
        if not known.empty:
            y = (known[turnout_target_id] == "voted").astype(int)
            p = known["turnout_probability"].fillna(0.0).astype(float).clip(0.0, 1.0)
            rows.extend(
                [
                    _metric_row(
                        run_id=run_id,
                        baseline=baseline,
                        model_name=model_name,
                        metric_scope=metric_scope,
                        metric_name="turnout_brier",
                        metric_value=float(((p - y) ** 2).mean()),
                        group_key=group_key,
                        question_id=turnout_target_id,
                        n=int(len(known)),
                        small_n=small_n,
                    ),
                    _metric_row(
                        run_id=run_id,
                        baseline=baseline,
                        model_name=model_name,
                        metric_scope=metric_scope,
                        metric_name="turnout_accuracy_at_0_5",
                        metric_value=float(((p >= 0.5) == (y == 1)).mean()),
                        group_key=group_key,
                        question_id=turnout_target_id,
                        n=int(len(known)),
                        small_n=small_n,
                    ),
                    _metric_row(
                        run_id=run_id,
                        baseline=baseline,
                        model_name=model_name,
                        metric_scope=metric_scope,
                        metric_name="turnout_auc",
                        metric_value=_safe_auc(y, p),
                        group_key=group_key,
                        question_id=turnout_target_id,
                        n=int(len(known)),
                        small_n=small_n,
                    ),
                ]
            )
    if vote_target_id in group.columns:
        known_vote = group[group[vote_target_id].isin(VOTE_EVAL_CLASSES)].copy()
        if not known_vote.empty:
            y_true = known_vote[vote_target_id].astype(str)
            y_pred = known_vote["most_likely_choice"].fillna("not_vote").astype(str)
            y_pred = y_pred.where(y_pred.isin(VOTE_EVAL_CLASSES), "not_vote")
            prob_matrix = _vote_probability_matrix(known_vote)
            rows.extend(
                [
                    _metric_row(
                        run_id=run_id,
                        baseline=baseline,
                        model_name=model_name,
                        metric_scope=metric_scope,
                        metric_name="vote_accuracy",
                        metric_value=float((y_pred == y_true).mean()),
                        group_key=group_key,
                        question_id=vote_target_id,
                        n=int(len(known_vote)),
                        small_n=small_n,
                    ),
                    _metric_row(
                        run_id=run_id,
                        baseline=baseline,
                        model_name=model_name,
                        metric_scope=metric_scope,
                        metric_name="vote_macro_f1",
                        metric_value=float(
                            f1_score(y_true, y_pred, labels=VOTE_EVAL_CLASSES, average="macro", zero_division=0)
                        ),
                        group_key=group_key,
                        question_id=vote_target_id,
                        n=int(len(known_vote)),
                        small_n=small_n,
                    ),
                    _metric_row(
                        run_id=run_id,
                        baseline=baseline,
                        model_name=model_name,
                        metric_scope=metric_scope,
                        metric_name="vote_log_loss",
                        metric_value=float(log_loss(y_true, prob_matrix, labels=VOTE_EVAL_CLASSES)),
                        group_key=group_key,
                        question_id=vote_target_id,
                        n=int(len(known_vote)),
                        small_n=small_n,
                    ),
                ]
            )
            confusion = pd.crosstab(y_true, y_pred)
            for truth in VOTE_EVAL_CLASSES:
                for pred in VOTE_EVAL_CLASSES:
                    rows.append(
                        _metric_row(
                            run_id=run_id,
                            baseline=baseline,
                            model_name=model_name,
                            metric_scope=metric_scope,
                            metric_name="vote_confusion_count",
                            metric_value=float(confusion.get(pred, pd.Series(dtype=float)).get(truth, 0.0)),
                            group_key=f"{group_key + ';' if group_key else ''}truth={truth};pred={pred}",
                            question_id=vote_target_id,
                            n=int(len(known_vote)),
                            small_n=small_n,
                        )
                    )
    return rows


SUBGROUP_COLUMNS = {
    "party_id": "party_id_3",
    "ideology": "ideology_3",
    "race_ethnicity": "race_ethnicity",
    "education": "education_binary",
    "age_group": "age_group",
    "gender": "gender",
    "state": "state_po",
}


def individual_turnout_vote_metrics(
    responses: pd.DataFrame,
    targets: pd.DataFrame,
    run_id: str,
    *,
    turnout_target_id: str = "turnout_2024_self_report",
    vote_target_id: str = "president_vote_2024",
    agents: pd.DataFrame | None = None,
    subgroup_columns: list[str] | None = None,
    small_n_threshold: int = 30,
) -> pd.DataFrame:
    target_wide = targets.pivot_table(
        index="ces_id",
        columns="target_id",
        values="canonical_value",
        aggfunc="first",
    ).reset_index()
    merged = responses.merge(target_wide, left_on="base_ces_id", right_on="ces_id", how="left")
    if agents is not None and not agents.empty:
        agent_cols = [
            "agent_id",
            *[col for col in set(SUBGROUP_COLUMNS.values()) if col in agents.columns],
        ]
        merged = merged.merge(agents[agent_cols], on="agent_id", how="left", suffixes=("", "_agent"))
    rows: list[dict[str, object]] = []
    for (baseline, model_name), group in merged.groupby(["baseline", "model_name"], dropna=False):
        rows.extend(
            _turnout_vote_metric_rows(
                group,
                run_id=run_id,
                baseline=baseline,
                model_name=model_name,
                metric_scope="individual",
                group_key=None,
                turnout_target_id=turnout_target_id,
                vote_target_id=vote_target_id,
                small_n_threshold=small_n_threshold,
            )
        )
        for subgroup_name in subgroup_columns or []:
            col = SUBGROUP_COLUMNS.get(subgroup_name, subgroup_name)
            if col not in group.columns:
                continue
            for value, subgroup in group.groupby(col, dropna=False):
                rows.extend(
                    _turnout_vote_metric_rows(
                        subgroup,
                        run_id=run_id,
                        baseline=baseline,
                        model_name=model_name,
                        metric_scope="subgroup",
                        group_key=f"{subgroup_name}={value}",
                        turnout_target_id=turnout_target_id,
                        vote_target_id=vote_target_id,
                        small_n_threshold=small_n_threshold,
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
    agents: pd.DataFrame | None = None,
    subgroup_columns: list[str] | None = None,
    small_n_threshold: int = 30,
) -> pd.DataFrame:
    df = individual_turnout_vote_metrics(
        responses,
        targets,
        run_id,
        turnout_target_id=turnout_target_id,
        vote_target_id=vote_target_id,
        agents=agents,
        subgroup_columns=subgroup_columns,
        small_n_threshold=small_n_threshold,
    )
    write_table(df, out_path)
    return df


def turnout_vote_election_metrics(aggregate: pd.DataFrame, mit_results: pd.DataFrame, run_id: str) -> pd.DataFrame:
    truth = state_truth_table(mit_results)
    merged = aggregate.merge(
        truth[
            [
                "state_po",
                "true_dem_votes",
                "true_rep_votes",
                "true_dem_2p",
                "true_margin",
                "true_winner",
            ]
        ],
        on="state_po",
    )
    rows: list[dict[str, object]] = []
    group_cols = ["baseline"] + (["model_name"] if "model_name" in merged.columns else [])
    for keys, group in merged.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        baseline = keys[0]
        model_name = keys[1] if len(keys) > 1 else None
        dem_error = group["dem_share_2p"] - group["true_dem_2p"]
        margin_error = group["margin_2p"] - group["true_margin"]
        winner_correct = group["winner"] == group["true_winner"]
        pred_dem = group["expected_dem_votes"].sum() if "expected_dem_votes" in group.columns else 0.0
        pred_rep = group["expected_rep_votes"].sum() if "expected_rep_votes" in group.columns else 0.0
        true_dem = group["true_dem_votes"].sum()
        true_rep = group["true_rep_votes"].sum()
        pred_national_dem_2p = pred_dem / (pred_dem + pred_rep) if (pred_dem + pred_rep) else np.nan
        true_national_dem_2p = true_dem / (true_dem + true_rep) if (true_dem + true_rep) else np.nan
        for metric_name, metric_value in {
            "dem_2p_rmse": float(np.sqrt((dem_error**2).mean())),
            "state_dem_2p_rmse": float(np.sqrt((dem_error**2).mean())),
            "margin_mae": float(margin_error.abs().mean()),
            "state_margin_mae": float(margin_error.abs().mean()),
            "winner_accuracy": float(winner_correct.mean()),
        }.items():
            rows.append(
                _metric_row(
                    run_id=run_id,
                    baseline=baseline,
                    model_name=model_name,
                    metric_scope="election",
                    metric_name=metric_name,
                    metric_value=metric_value,
                    n=int(len(group)),
                )
            )
        rows.append(
            _metric_row(
                run_id=run_id,
                baseline=baseline,
                model_name=model_name,
                metric_scope="election_national",
                metric_name="national_dem_2p_error",
                metric_value=float(pred_national_dem_2p - true_national_dem_2p),
                n=int(len(group)),
            )
        )
        for _, row in group.iterrows():
            rows.extend(
                [
                    _metric_row(
                        run_id=run_id,
                        baseline=baseline,
                        model_name=model_name,
                        metric_scope="election_state",
                        metric_name="state_dem_2p_error",
                        metric_value=float(row["dem_share_2p"] - row["true_dem_2p"]),
                        state_po=row["state_po"],
                    ),
                    _metric_row(
                        run_id=run_id,
                        baseline=baseline,
                        model_name=model_name,
                        metric_scope="election_state",
                        metric_name="state_margin_error",
                        metric_value=float(row["margin_2p"] - row["true_margin"]),
                        state_po=row["state_po"],
                    ),
                ]
            )
    return pd.DataFrame(rows)


def write_turnout_vote_election_metrics(
    aggregate: pd.DataFrame,
    mit_results: pd.DataFrame,
    run_id: str,
    out_path: str | Path,
) -> pd.DataFrame:
    df = turnout_vote_election_metrics(aggregate, mit_results, run_id)
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
            "n",
            "small_n",
            "confidence_low",
            "confidence_high",
            "created_at",
        ]
    )
