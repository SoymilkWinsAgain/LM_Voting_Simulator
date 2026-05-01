"""E06 subgroup and hard-choice calibration reliability runner."""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .ces_benchmark import VOTE_CLASSES, _target_wide
from .eval_suite import git_commit
from .io import ensure_dir, load_yaml, write_json, write_table, write_yaml


SUBGROUP_DIMENSIONS: list[tuple[str, list[str]]] = [
    ("party_id_3", ["party_id_3"]),
    ("party_id_7", ["party_id_7"]),
    ("ideology_3", ["ideology_3"]),
    ("race_ethnicity", ["race_ethnicity"]),
    ("education_binary", ["education_binary"]),
    ("age_group", ["age_group"]),
    ("gender", ["gender"]),
    ("state_po", ["state_po"]),
    ("state_po_x_party_id_3", ["state_po", "party_id_3"]),
    ("state_po_x_race_ethnicity", ["state_po", "race_ethnicity"]),
]
AGENT_COLUMNS = sorted({col for _, cols in SUBGROUP_DIMENSIONS for col in cols})
CHOICE_COLUMNS = {
    "democrat": "vote_prob_democrat",
    "republican": "vote_prob_republican",
    "not_vote": "turnout_probability",
}
FOCUS_BASELINES = [
    "ces_party_ideology_llm",
    "ces_survey_memory_llm_strict",
    "L3_party_ideology_llm",
    "L4_party_ideology_context_llm",
    "L5_strict_memory_llm",
    "L6_strict_memory_context_llm",
    "L7_poll_informed_memory_context_llm",
    "L8_post_hoc_oracle_memory_context_llm",
    "P1_memory_shuffled_within_state_llm",
    "P2_memory_shuffled_within_party_llm",
]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _weights(group: pd.DataFrame) -> pd.Series:
    weights = pd.to_numeric(group.get("analysis_weight", 1.0), errors="coerce").fillna(1.0).astype(float)
    weights = weights.replace([np.inf, -np.inf], 1.0)
    weights.loc[weights <= 0] = 1.0
    return weights


def _weighted_mean(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray | None = None) -> float | None:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return None
    if weights is None:
        return _safe_float(np.nanmean(arr))
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(arr) & np.isfinite(w)
    if not mask.any():
        return None
    denom = float(np.sum(w[mask]))
    if denom <= 0:
        return None
    return _safe_float(np.sum(arr[mask] * w[mask]) / denom)


def _weighted_sum(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(arr) & np.isfinite(w)
    return float(np.sum(arr[mask] * w[mask])) if mask.any() else 0.0


def _weighted_variance(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray) -> float | None:
    mean = _weighted_mean(values, weights)
    if mean is None:
        return None
    arr = np.asarray(values, dtype=float)
    return _weighted_mean((arr - mean) ** 2, weights)


def _entropy(shares: list[float]) -> float | None:
    clean = [float(value) for value in shares if value is not None and value > 0 and math.isfinite(float(value))]
    if not clean:
        return None
    return float(-sum(value * math.log(value) for value in clean))


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return _safe_float(numerator / denominator)


def _fmt_group_value(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    return str(value)


def _markdown_table(df: pd.DataFrame, *, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    display = df.head(max_rows).copy()
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


def _read_required_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required E06 input: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Required E06 input is empty: {path}")
    return df


def _discover_targets_path(cfg: dict[str, Any], source_dirs: dict[str, Path]) -> Path:
    paths = cfg.get("paths", {})
    if paths.get("ces_targets"):
        return Path(paths["ces_targets"])
    for run_dir in source_dirs.values():
        snapshot = run_dir / "config_snapshot.yaml"
        if not snapshot.exists():
            continue
        source_cfg = load_yaml(snapshot)
        target_path = source_cfg.get("paths", {}).get("ces_targets")
        if target_path:
            return Path(target_path)
    raise KeyError("E06 config must provide paths.ces_targets or source config snapshots with paths.ces_targets")


def _source_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = cfg.get("paths", {})
    root = Path(paths.get("root_dir", "data/runs/eval_suite_local"))
    return {
        "E01_individual_persona": Path(paths.get("e01_run_dir", root / "01_individual_persona")),
        "E05_ablation_placebo": Path(paths.get("e05_run_dir", root / "05_ablation_placebo")),
    }


def _prepare_source_frame(source: str, run_dir: Path, targets_wide: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    responses = _read_required_table(run_dir / "responses.parquet").copy()
    agents = _read_required_table(run_dir / "agents.parquet").copy()
    responses["base_ces_id"] = responses["base_ces_id"].astype(str)
    agents["base_ces_id"] = agents["base_ces_id"].astype(str)
    targets_wide = targets_wide.copy()
    targets_wide["ces_id"] = targets_wide["ces_id"].astype(str)

    available_agent_cols = ["base_ces_id", "sample_weight", *[col for col in AGENT_COLUMNS if col in agents.columns]]
    agent_meta = agents[available_agent_cols].drop_duplicates("base_ces_id")
    merged = responses.merge(agent_meta, on="base_ces_id", how="left", suffixes=("", "_agent"))
    for col in AGENT_COLUMNS:
        agent_col = f"{col}_agent"
        if col not in merged.columns and agent_col in merged.columns:
            merged[col] = merged[agent_col]
        elif agent_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[agent_col])

    if "sample_weight" in merged.columns:
        response_weight = pd.to_numeric(merged["sample_weight"], errors="coerce")
    else:
        response_weight = pd.Series(np.nan, index=merged.index)
    if "sample_weight_agent" in merged.columns:
        agent_weight = pd.to_numeric(merged["sample_weight_agent"], errors="coerce")
        response_weight = response_weight.where(response_weight.notna(), agent_weight)
    merged["analysis_weight"] = response_weight.fillna(1.0).astype(float)
    merged.loc[merged["analysis_weight"] <= 0, "analysis_weight"] = 1.0

    merged = merged.merge(targets_wide, left_on="base_ces_id", right_on="ces_id", how="left")
    merged["source"] = source
    merged["source_run_dir"] = str(run_dir)
    if "model_name" not in merged.columns:
        merged["model_name"] = ""
    if "headline_sample" not in merged.columns:
        merged["headline_sample"] = True
    if "most_likely_choice" not in merged.columns:
        merged["most_likely_choice"] = np.nan

    parse_status = merged.get("parse_status", pd.Series("", index=merged.index)).fillna("").astype(str)
    metadata = {
        "source": source,
        "run_dir": str(run_dir),
        "n_responses": int(len(merged)),
        "n_agents": int(len(agents)),
        "parse_ok_rate": float((parse_status == "ok").mean()) if len(merged) else 0.0,
        "invalid_choice_rate": float(merged.get("invalid_choice", pd.Series(False, index=merged.index)).fillna(False).astype(bool).mean()),
        "forbidden_choice_rate": float(merged.get("forbidden_choice", pd.Series(False, index=merged.index)).fillna(False).astype(bool).mean()),
        "legacy_probability_schema_rate": float(
            merged.get("legacy_probability_schema", pd.Series(False, index=merged.index)).fillna(False).astype(bool).mean()
        ),
        "transport_error_rate": float(merged.get("transport_error", pd.Series(False, index=merged.index)).fillna(False).astype(bool).mean()),
    }
    return merged, metadata


def _choice_probabilities(group: pd.DataFrame) -> pd.DataFrame:
    turnout = pd.to_numeric(group.get("turnout_probability", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out = pd.DataFrame(index=group.index)
    out["democrat"] = turnout * pd.to_numeric(group.get("vote_prob_democrat", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["republican"] = turnout * pd.to_numeric(group.get("vote_prob_republican", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    undecided = pd.to_numeric(group.get("vote_prob_undecided", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["not_vote"] = (1.0 - turnout) + turnout * undecided
    denom = out.sum(axis=1).replace(0.0, 1.0)
    return out.div(denom, axis=0)[VOTE_CLASSES]


def _dem_2p_from_probabilities(probs: pd.DataFrame, weights: pd.Series) -> float | None:
    dem = _weighted_sum(probs["democrat"], weights)
    rep = _weighted_sum(probs["republican"], weights)
    denom = dem + rep
    return _safe_float(dem / denom) if denom > 0 else None


def _true_dem_2p(true_vote: pd.Series, weights: pd.Series) -> float | None:
    mask = true_vote.isin(["democrat", "republican"])
    if not mask.any():
        return None
    dem = float(weights[mask & (true_vote == "democrat")].sum())
    rep = float(weights[mask & (true_vote == "republican")].sum())
    denom = dem + rep
    return _safe_float(dem / denom) if denom > 0 else None


def _multiclass_brier_hard_choice(true_vote: pd.Series, probs: pd.DataFrame, weights: pd.Series) -> float | None:
    mask = true_vote.isin(VOTE_CLASSES)
    if not mask.any():
        return None
    y = true_vote[mask].astype(str).to_numpy()
    p = probs.loc[mask, VOTE_CLASSES].to_numpy(dtype=float)
    onehot = np.zeros_like(p)
    index = {name: idx for idx, name in enumerate(VOTE_CLASSES)}
    for row_idx, value in enumerate(y):
        onehot[row_idx, index[value]] = 1.0
    losses = ((p - onehot) ** 2).sum(axis=1)
    return _weighted_mean(losses, weights[mask])


def _metric_row(
    *,
    source: str,
    baseline: str,
    model_name: str,
    dimension: str,
    subgroup_value: str,
    group: pd.DataFrame,
    small_n_threshold: int,
) -> dict[str, Any]:
    weights = _weights(group)
    parse_ok = group.get("parse_status", pd.Series("", index=group.index)).fillna("").astype(str).eq("ok")
    turnout_true = group.get("turnout_2024_self_report", pd.Series(index=group.index, dtype=str)).fillna("").astype(str)
    vote_true = group.get("president_vote_2024", pd.Series(index=group.index, dtype=str)).fillna("").astype(str)
    pred_choice = group.get("most_likely_choice", pd.Series(index=group.index, dtype=str)).fillna("").astype(str)
    turnout_pred = pd.to_numeric(group.get("turnout_probability", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    probs = _choice_probabilities(group)

    turnout_mask = turnout_true.isin(["voted", "not_voted"])
    vote_mask = vote_true.isin(VOTE_CLASSES) & pred_choice.isin(VOTE_CLASSES)
    turnout_accuracy = None
    turnout_brier = None
    mean_predicted_turnout = None
    mean_true_turnout = None
    if turnout_mask.any():
        y_turnout = turnout_true[turnout_mask].eq("voted").astype(float)
        p_turnout = turnout_pred[turnout_mask]
        w_turnout = weights[turnout_mask]
        turnout_accuracy = _weighted_mean(p_turnout.ge(0.5).eq(y_turnout.eq(1.0)).astype(float), w_turnout)
        turnout_brier = _weighted_mean((p_turnout - y_turnout) ** 2, w_turnout)
        mean_predicted_turnout = _weighted_mean(p_turnout, w_turnout)
        mean_true_turnout = _weighted_mean(y_turnout, w_turnout)

    vote_accuracy = None
    vote_macro_f1 = None
    if vote_mask.any():
        y_vote = vote_true[vote_mask].astype(str)
        p_vote = pred_choice[vote_mask].astype(str)
        w_vote = weights[vote_mask]
        vote_accuracy = _weighted_mean(p_vote.eq(y_vote).astype(float), w_vote)
        try:
            vote_macro_f1 = _safe_float(
                f1_score(y_vote, p_vote, labels=VOTE_CLASSES, average="macro", zero_division=0, sample_weight=w_vote)
            )
        except ValueError:
            vote_macro_f1 = None

    predicted_dem_2p = _dem_2p_from_probabilities(probs, weights)
    true_dem_2p = _true_dem_2p(vote_true, weights)
    dem_2p_error = None if predicted_dem_2p is None or true_dem_2p is None else _safe_float(predicted_dem_2p - true_dem_2p)
    return {
        "source": source,
        "baseline": baseline,
        "model_name": model_name,
        "dimension": dimension,
        "subgroup_value": subgroup_value,
        "n": int(len(group)),
        "weighted_n": float(weights.sum()),
        "n_turnout": int(turnout_mask.sum()),
        "n_vote": int(vote_mask.sum()),
        "small_n": bool(len(group) < small_n_threshold),
        "parse_ok_rate": _weighted_mean(parse_ok.astype(float), weights),
        "vote_accuracy": vote_accuracy,
        "vote_macro_f1": vote_macro_f1,
        "turnout_accuracy": turnout_accuracy,
        "turnout_brier": turnout_brier,
        "multiclass_brier_hard_choice": _multiclass_brier_hard_choice(vote_true, probs, weights),
        "mean_predicted_turnout": mean_predicted_turnout,
        "mean_true_turnout": mean_true_turnout,
        "mean_predicted_dem_2p": predicted_dem_2p,
        "mean_true_dem_2p": true_dem_2p,
        "dem_2p_error": dem_2p_error,
        "dem_2p_abs_error": abs(dem_2p_error) if dem_2p_error is not None else None,
        "created_at": pd.Timestamp.now(tz="UTC"),
    }


def _distribution_row(
    *,
    source: str,
    baseline: str,
    model_name: str,
    dimension: str,
    subgroup_value: str,
    group: pd.DataFrame,
    small_n_threshold: int,
) -> dict[str, Any]:
    weights = _weights(group)
    pred_choice = group.get("most_likely_choice", pd.Series(index=group.index, dtype=str)).fillna("").astype(str)
    true_vote = group.get("president_vote_2024", pd.Series(index=group.index, dtype=str)).fillna("").astype(str)
    pred_mask = pred_choice.isin(VOTE_CLASSES)
    true_mask = true_vote.isin(VOTE_CLASSES)
    pred_total = float(weights[pred_mask].sum())
    true_total = float(weights[true_mask].sum())
    pred_shares = {
        choice: float(weights[pred_mask & (pred_choice == choice)].sum() / pred_total) if pred_total > 0 else None
        for choice in VOTE_CLASSES
    }
    true_shares = {
        choice: float(weights[true_mask & (true_vote == choice)].sum() / true_total) if true_total > 0 else None
        for choice in VOTE_CLASSES
    }
    pred_entropy = _entropy([value for value in pred_shares.values() if value is not None])
    true_entropy = _entropy([value for value in true_shares.values() if value is not None])
    pred_dem_indicator = pred_choice[pred_mask].eq("democrat").astype(float)
    true_dem_indicator = true_vote[true_mask].eq("democrat").astype(float)
    pred_variance = _weighted_variance(pred_dem_indicator, weights[pred_mask]) if pred_mask.any() else None
    true_variance = _weighted_variance(true_dem_indicator, weights[true_mask]) if true_mask.any() else None
    entropy_ratio = _ratio(pred_entropy, true_entropy)
    variance_ratio = _ratio(pred_variance, true_variance)
    return {
        "source": source,
        "baseline": baseline,
        "model_name": model_name,
        "dimension": dimension,
        "subgroup_value": subgroup_value,
        "n": int(len(group)),
        "weighted_n": float(weights.sum()),
        "small_n": bool(len(group) < small_n_threshold),
        "pred_share_democrat": pred_shares["democrat"],
        "pred_share_republican": pred_shares["republican"],
        "pred_share_not_vote": pred_shares["not_vote"],
        "true_share_democrat": true_shares["democrat"],
        "true_share_republican": true_shares["republican"],
        "true_share_not_vote": true_shares["not_vote"],
        "predicted_vote_entropy": pred_entropy,
        "true_vote_entropy": true_entropy,
        "entropy_ratio": entropy_ratio,
        "entropy_ratio_distance": abs(entropy_ratio - 1.0) if entropy_ratio is not None else None,
        "predicted_dem_indicator_variance": pred_variance,
        "true_dem_indicator_variance": true_variance,
        "variance_ratio": variance_ratio,
        "variance_ratio_distance": abs(variance_ratio - 1.0) if variance_ratio is not None else None,
        "created_at": pd.Timestamp.now(tz="UTC"),
    }


def _iter_dimension_groups(frame: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", frame)]
    for dimension, columns in SUBGROUP_DIMENSIONS:
        if not all(col in frame.columns for col in columns):
            continue
        for key, group in frame.groupby(columns, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            value = " x ".join(_fmt_group_value(part) for part in key)
            groups.append((dimension, value, group.copy()))
    return groups


def _compute_baseline_tables(args: tuple[str, str, str, pd.DataFrame, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source, baseline, model_name, frame, small_n_threshold = args
    metric_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for dimension, subgroup_value, group in _iter_dimension_groups(frame):
        metric_rows.append(
            _metric_row(
                source=source,
                baseline=baseline,
                model_name=model_name,
                dimension=dimension,
                subgroup_value=subgroup_value,
                group=group,
                small_n_threshold=small_n_threshold,
            )
        )
        distribution_rows.append(
            _distribution_row(
                source=source,
                baseline=baseline,
                model_name=model_name,
                dimension=dimension,
                subgroup_value=subgroup_value,
                group=group,
                small_n_threshold=small_n_threshold,
            )
        )
    calibration_rows = _calibration_rows(source=source, baseline=baseline, model_name=model_name, frame=frame)
    return metric_rows, distribution_rows, calibration_rows


def _calibration_rows(*, source: str, baseline: str, model_name: str, frame: pd.DataFrame, n_bins: int = 10) -> list[dict[str, Any]]:
    turnout_true = frame.get("turnout_2024_self_report", pd.Series(index=frame.index, dtype=str)).fillna("").astype(str)
    turnout_mask = turnout_true.isin(["voted", "not_voted"])
    if not turnout_mask.any():
        return []
    work = frame[turnout_mask].copy()
    y = turnout_true[turnout_mask].eq("voted").astype(float)
    p = pd.to_numeric(work.get("turnout_probability", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    weights = _weights(work)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    errors: list[tuple[float, float]] = []
    total_weight = float(weights.sum())
    for idx, (low, high) in enumerate(zip(bins[:-1], bins[1:], strict=False)):
        upper = p <= high if high == 1.0 else p < high
        mask = (p >= low) & upper
        if not mask.any():
            continue
        weighted_n = float(weights[mask].sum())
        mean_predicted = _weighted_mean(p[mask], weights[mask])
        observed_rate = _weighted_mean(y[mask], weights[mask])
        absolute_error = None if mean_predicted is None or observed_rate is None else abs(mean_predicted - observed_rate)
        if absolute_error is not None and total_weight > 0:
            errors.append((weighted_n / total_weight, absolute_error))
        rows.append(
            {
                "source": source,
                "baseline": baseline,
                "model_name": model_name,
                "calibration_type": "turnout_hard_choice",
                "bin": int(idx),
                "bin_low": float(low),
                "bin_high": float(high),
                "n": int(mask.sum()),
                "weighted_n": weighted_n,
                "mean_predicted": mean_predicted,
                "observed_rate": observed_rate,
                "absolute_error": absolute_error,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    ece = sum(weight * error for weight, error in errors) if errors else None
    mce = max((error for _, error in errors), default=None)
    for row in rows:
        row["expected_calibration_error"] = ece
        row["maximum_calibration_error"] = mce
    return rows


def compute_e06_tables(
    source_frames: dict[str, pd.DataFrame],
    *,
    small_n_threshold: int = 30,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tasks: list[tuple[str, str, str, pd.DataFrame, int]] = []
    for source, frame in source_frames.items():
        for (baseline, model_name), group in frame.groupby(["baseline", "model_name"], dropna=False):
            tasks.append((source, str(baseline), str(model_name), group.copy(), small_n_threshold))
    metric_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    if workers <= 1 or len(tasks) <= 1:
        results = [_compute_baseline_tables(task) for task in tasks]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(_compute_baseline_tables, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
    for metrics, distributions, calibrations in results:
        metric_rows.extend(metrics)
        distribution_rows.extend(distributions)
        calibration_rows.extend(calibrations)
    return pd.DataFrame(metric_rows), pd.DataFrame(distribution_rows), pd.DataFrame(calibration_rows)


def quality_gate_rows(source_metadata: list[dict[str, Any]], reliability: pd.DataFrame, run_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    created_at = pd.Timestamp.now(tz="UTC")
    for meta in source_metadata:
        source = meta["source"]
        gates = [
            ("parse_ok_rate", ">=", 0.95, meta["parse_ok_rate"]),
            ("invalid_choice_rate", "<=", 0.02, meta["invalid_choice_rate"]),
            ("forbidden_choice_rate", "==", 0.0, meta["forbidden_choice_rate"]),
            ("legacy_probability_schema_rate", "==", 0.0, meta["legacy_probability_schema_rate"]),
            ("transport_error_rate", "==", 0.0, meta["transport_error_rate"]),
        ]
        for name, op, threshold, observed in gates:
            if op == ">=":
                passed = observed >= threshold
            elif op == "<=":
                passed = observed <= threshold
            else:
                passed = observed == threshold
            rows.append(
                {
                    "run_id": run_id,
                    "source": source,
                    "gate_name": name,
                    "operator": op,
                    "threshold": threshold,
                    "observed": observed,
                    "passed": bool(passed),
                    "created_at": created_at,
                }
            )
    required = {dimension for dimension, _ in SUBGROUP_DIMENSIONS}
    observed_dims = set(reliability.get("dimension", pd.Series(dtype=str)).dropna().astype(str))
    missing = sorted(required - observed_dims)
    rows.append(
        {
            "run_id": run_id,
            "source": "combined",
            "gate_name": "required_subgroup_dimensions_present",
            "operator": "==",
            "threshold": 0.0,
            "observed": float(len(missing)),
            "passed": not missing,
            "details": ",".join(missing),
            "created_at": created_at,
        }
    )
    return pd.DataFrame(rows)


def worst_subgroup_rows(reliability: pd.DataFrame, distribution: pd.DataFrame, *, max_rows_per_type: int = 40) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    core = reliability[reliability["dimension"] != "overall"].copy()
    if not core.empty:
        acc = core[core["vote_accuracy"].notna()].sort_values(["small_n", "vote_accuracy", "n_vote"], ascending=[True, True, False]).head(max_rows_per_type).copy()
        acc["failure_type"] = "lowest_vote_accuracy"
        acc["failure_score"] = 1.0 - acc["vote_accuracy"]
        rows.append(acc)
        dem = core[core["dem_2p_abs_error"].notna()].sort_values(["small_n", "dem_2p_abs_error"], ascending=[True, False]).head(max_rows_per_type).copy()
        dem["failure_type"] = "largest_dem_2p_abs_error"
        dem["failure_score"] = dem["dem_2p_abs_error"]
        rows.append(dem)
        turnout = core[core["turnout_brier"].notna()].sort_values(["small_n", "turnout_brier"], ascending=[True, False]).head(max_rows_per_type).copy()
        turnout["failure_type"] = "largest_turnout_brier"
        turnout["failure_score"] = turnout["turnout_brier"]
        rows.append(turnout)
    dist = distribution[distribution["dimension"] != "overall"].copy()
    if not dist.empty:
        for metric, failure_type in [
            ("entropy_ratio_distance", "largest_entropy_ratio_distance"),
            ("variance_ratio_distance", "largest_variance_ratio_distance"),
        ]:
            part = dist[dist[metric].notna()].sort_values(["small_n", metric], ascending=[True, False]).head(max_rows_per_type).copy()
            part["failure_type"] = failure_type
            part["failure_score"] = part[metric]
            rows.append(part)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    cols = [
        "failure_type",
        "failure_score",
        "source",
        "baseline",
        "model_name",
        "dimension",
        "subgroup_value",
        "n",
        "weighted_n",
        "small_n",
        "vote_accuracy",
        "turnout_brier",
        "dem_2p_error",
        "dem_2p_abs_error",
        "entropy_ratio",
        "variance_ratio",
    ]
    return out[[col for col in cols if col in out.columns]]


def _save_figure(fig: Any, out_base: Path) -> Path:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    return out_base.with_suffix(".png")


def _plot_label(df: pd.DataFrame) -> pd.Series:
    return df["source"].astype(str).str.replace("_", " ", regex=False) + " | " + df["baseline"].astype(str)


def write_e06_figures(run_dir: Path, reliability: pd.DataFrame, distribution: pd.DataFrame, calibration: pd.DataFrame) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir = ensure_dir(run_dir / "figures")
    written: list[Path] = []
    plot = reliability[(reliability["dimension"] != "overall") & reliability["vote_accuracy"].notna()].copy()
    if not plot.empty:
        plot["baseline_label"] = _plot_label(plot)
        focus = plot[plot["baseline"].isin(FOCUS_BASELINES)].copy()
        if focus.empty:
            focus = plot.copy()
        focus["subgroup_label"] = focus["dimension"] + "=" + focus["subgroup_value"].astype(str)
        focus = focus.sort_values(["small_n", "vote_accuracy"], ascending=[True, True]).head(60)
        pivot = focus.pivot_table(index="subgroup_label", columns="baseline_label", values="vote_accuracy", aggfunc="first")
        fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(pivot.columns)), max(6, 0.22 * len(pivot))))
        sns.heatmap(pivot, annot=False, cmap="viridis", vmin=0, vmax=1, ax=ax)
        ax.set_title("E06 Subgroup Vote Accuracy")
        written.append(_save_figure(fig, fig_dir / "e06_subgroup_vote_accuracy_heatmap"))
        plt.close(fig)

    plot = reliability[(reliability["dimension"] != "overall") & reliability["turnout_brier"].notna()].copy()
    if not plot.empty:
        plot["baseline_label"] = _plot_label(plot)
        focus = plot[plot["baseline"].isin(FOCUS_BASELINES)].copy()
        if focus.empty:
            focus = plot.copy()
        focus["subgroup_label"] = focus["dimension"] + "=" + focus["subgroup_value"].astype(str)
        focus = focus.sort_values(["small_n", "turnout_brier"], ascending=[True, False]).head(60)
        pivot = focus.pivot_table(index="subgroup_label", columns="baseline_label", values="turnout_brier", aggfunc="first")
        fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(pivot.columns)), max(6, 0.22 * len(pivot))))
        sns.heatmap(pivot, annot=False, cmap="magma", vmin=0, vmax=1, ax=ax)
        ax.set_title("E06 Subgroup Turnout Brier")
        written.append(_save_figure(fig, fig_dir / "e06_subgroup_turnout_brier_heatmap"))
        plt.close(fig)

    plot = reliability[(reliability["dimension"] != "overall") & reliability["dem_2p_abs_error"].notna()].copy()
    if not plot.empty:
        plot["label"] = _plot_label(plot) + " | " + plot["dimension"] + "=" + plot["subgroup_value"].astype(str)
        focus = plot.sort_values(["small_n", "dem_2p_abs_error"], ascending=[True, False]).head(35).sort_values("dem_2p_error")
        fig, ax = plt.subplots(figsize=(12, max(6, 0.28 * len(focus))))
        colors = np.where(focus["dem_2p_error"] >= 0, "#2563eb", "#dc2626")
        ax.barh(focus["label"], focus["dem_2p_error"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Predicted Democratic two-party share minus true share")
        ax.set_title("E06 Largest Subgroup Democratic Two-Party Errors")
        written.append(_save_figure(fig, fig_dir / "e06_subgroup_dem_2p_error"))
        plt.close(fig)

    if not calibration.empty:
        plot = calibration[calibration["calibration_type"] == "turnout_hard_choice"].copy()
        plot["baseline_label"] = _plot_label(plot)
        focus = plot[plot["baseline"].isin(FOCUS_BASELINES)].copy()
        if focus.empty:
            focus = plot.copy()
        fig, ax = plt.subplots(figsize=(10, 7))
        for label, group in focus.groupby("baseline_label", sort=False):
            ax.plot(group["mean_predicted"], group["observed_rate"], marker="o", label=label)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("Mean predicted turnout hard choice")
        ax.set_ylabel("Observed turnout")
        ax.set_title("E06 Deterministic Turnout Reliability")
        ax.legend(fontsize=6)
        written.append(_save_figure(fig, fig_dir / "e06_turnout_reliability"))
        plt.close(fig)

    for metric, title, file_name in [
        ("entropy_ratio", "E06 Vote Distribution Entropy Ratio", "e06_entropy_ratio_by_subgroup"),
        ("variance_ratio", "E06 Democratic Indicator Variance Ratio", "e06_variance_ratio_by_subgroup"),
    ]:
        if metric not in distribution.columns:
            continue
        plot = distribution[(distribution["dimension"] != "overall") & distribution[metric].notna()].copy()
        if plot.empty:
            continue
        distance = (plot[metric] - 1.0).abs()
        plot = plot.assign(distance=distance, label=_plot_label(plot) + " | " + plot["dimension"] + "=" + plot["subgroup_value"].astype(str))
        focus = plot.sort_values(["small_n", "distance"], ascending=[True, False]).head(35).sort_values(metric)
        fig, ax = plt.subplots(figsize=(12, max(6, 0.28 * len(focus))))
        ax.barh(focus["label"], focus[metric], color="#0f766e")
        ax.axvline(1, color="black", linewidth=0.8)
        ax.set_xlabel(metric)
        ax.set_title(title)
        written.append(_save_figure(fig, fig_dir / file_name))
        plt.close(fig)
    return written


def write_e06_report(
    *,
    run_dir: Path,
    cfg: dict[str, Any],
    source_metadata: list[dict[str, Any]],
    reliability: pd.DataFrame,
    distribution: pd.DataFrame,
    calibration: pd.DataFrame,
    worst: pd.DataFrame,
    gates: pd.DataFrame,
    figures: list[Path],
    runtime: dict[str, Any],
) -> Path:
    overall = reliability[reliability["dimension"] == "overall"].copy()
    overall_cols = [
        "source",
        "baseline",
        "n",
        "vote_accuracy",
        "vote_macro_f1",
        "turnout_accuracy",
        "turnout_brier",
        "dem_2p_error",
    ]
    calibration_summary = calibration[
        ["source", "baseline", "expected_calibration_error", "maximum_calibration_error"]
    ].drop_duplicates() if not calibration.empty else pd.DataFrame()
    dist_overall = distribution[distribution["dimension"] == "overall"].copy()
    dist_cols = [
        "source",
        "baseline",
        "pred_share_democrat",
        "pred_share_republican",
        "pred_share_not_vote",
        "entropy_ratio",
        "variance_ratio",
    ]
    gate_cols = [col for col in ["source", "gate_name", "observed", "threshold", "passed", "details"] if col in gates.columns]
    lines = [
        f"# E06 Subgroup and Calibration Reliability: {cfg.get('run_id', 'e06_subgroup_calibration')}",
        "",
        "## Summary",
        "- This experiment reuses E01 and E05 outputs; it made zero LLM calls.",
        "- Probability-like response columns are treated as system-derived hard-choice one-hot values, not subjective LLM probabilities.",
        "- Turnout calibration is a deterministic 0/1 reliability diagnostic.",
        "",
        "## Runtime",
        _markdown_table(pd.DataFrame([runtime]).T.reset_index().rename(columns={"index": "field", 0: "value"}), max_rows=80),
        "",
        "## Input Quality Gates",
        _markdown_table(gates[gate_cols], max_rows=80),
        "",
        "## Source Inputs",
        _markdown_table(pd.DataFrame(source_metadata), max_rows=20),
        "",
        "## Overall Metrics",
        _markdown_table(overall[[col for col in overall_cols if col in overall.columns]].sort_values(["source", "baseline"]), max_rows=80),
        "",
        "## Calibration Summary",
        _markdown_table(calibration_summary.sort_values(["source", "baseline"]), max_rows=80),
        "",
        "## Distribution Summary",
        _markdown_table(dist_overall[[col for col in dist_cols if col in dist_overall.columns]].sort_values(["source", "baseline"]), max_rows=80),
        "",
        "## Worst Subgroups",
        _markdown_table(worst, max_rows=80),
        "",
        "## Figures",
        "\n".join(f"- `{path.relative_to(run_dir)}`" for path in figures) if figures else "- None.",
        "",
        "## Output Files",
        "- `subgroup_reliability_metrics.parquet`",
        "- `distribution_diagnostics.parquet`",
        "- `calibration_bins.parquet`",
        "- `worst_subgroups.parquet`",
        "- `quality_gates.parquet`",
        "- `runtime.json`",
        "",
    ]
    out = run_dir / "benchmark_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run_ces_subgroup_calibration_benchmark(config_path: str | Path) -> dict[str, Path]:
    cfg = load_yaml(config_path)
    run_id = cfg.get("run_id", "e06_subgroup_calibration")
    run_dir = ensure_dir(cfg.get("paths", {}).get("run_dir", "data/runs/eval_suite_local/06_subgroup_calibration"))
    write_yaml(cfg, run_dir / "config_snapshot.yaml")
    started = time.time()
    workers = max(1, int(cfg.get("workers", cfg.get("analysis", {}).get("workers", 1))))
    small_n_threshold = int(cfg.get("small_n_threshold", cfg.get("analysis", {}).get("small_n_threshold", 30)))
    source_dirs = _source_dirs(cfg)
    targets_path = _discover_targets_path(cfg, source_dirs)
    targets = _read_required_table(targets_path)
    targets_wide = _target_wide(targets)

    source_frames: dict[str, pd.DataFrame] = {}
    source_metadata: list[dict[str, Any]] = []
    for source, run_source_dir in source_dirs.items():
        frame, metadata = _prepare_source_frame(source, run_source_dir, targets_wide)
        source_frames[source] = frame
        source_metadata.append(metadata)

    reliability, distribution, calibration = compute_e06_tables(
        source_frames,
        small_n_threshold=small_n_threshold,
        workers=workers,
    )
    worst = worst_subgroup_rows(reliability, distribution)
    gates = quality_gate_rows(source_metadata, reliability, run_id)
    figures = write_e06_figures(run_dir, reliability, distribution, calibration)
    runtime = {
        "run_id": run_id,
        "git_commit": git_commit(),
        "workers": workers,
        "n_llm_tasks": 0,
        "llm_calls_made": 0,
        "runtime_seconds": time.time() - started,
        "n_sources": len(source_frames),
        "n_responses": int(sum(len(frame) for frame in source_frames.values())),
        "n_subgroup_rows": int(len(reliability)),
        "n_distribution_rows": int(len(distribution)),
        "n_calibration_rows": int(len(calibration)),
        "small_n_threshold": small_n_threshold,
        "source_dirs": {source: str(path) for source, path in source_dirs.items()},
        "targets_path": str(targets_path),
        "all_gates_passed": bool(gates["passed"].fillna(False).astype(bool).all()) if not gates.empty else False,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }

    write_table(reliability, run_dir / "subgroup_reliability_metrics.parquet")
    write_table(distribution, run_dir / "distribution_diagnostics.parquet")
    write_table(calibration, run_dir / "calibration_bins.parquet")
    write_table(worst, run_dir / "worst_subgroups.parquet")
    write_table(gates, run_dir / "quality_gates.parquet")
    write_json(runtime, run_dir / "runtime.json")
    report = write_e06_report(
        run_dir=run_dir,
        cfg=cfg,
        source_metadata=source_metadata,
        reliability=reliability,
        distribution=distribution,
        calibration=calibration,
        worst=worst,
        gates=gates,
        figures=figures,
        runtime=runtime,
    )
    return {
        "subgroup_reliability_metrics": run_dir / "subgroup_reliability_metrics.parquet",
        "distribution_diagnostics": run_dir / "distribution_diagnostics.parquet",
        "calibration_bins": run_dir / "calibration_bins.parquet",
        "worst_subgroups": run_dir / "worst_subgroups.parquet",
        "quality_gates": run_dir / "quality_gates.parquet",
        "runtime": run_dir / "runtime.json",
        "config_snapshot": run_dir / "config_snapshot.yaml",
        "report": report,
    }
