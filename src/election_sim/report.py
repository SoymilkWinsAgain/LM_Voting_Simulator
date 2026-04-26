"""Markdown report generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .mit import state_truth_table


def _markdown_table(df: pd.DataFrame) -> str:
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


def write_eval_report(
    *,
    run_id: str,
    run_dir: str | Path,
    aggregate: pd.DataFrame,
    metrics: pd.DataFrame,
    mit_results: pd.DataFrame,
    agents: pd.DataFrame,
    responses: pd.DataFrame,
) -> Path:
    run_dir = Path(run_dir)
    truth = state_truth_table(mit_results)
    state_table = aggregate.merge(
        truth[["state_po", "true_dem_2p", "true_margin", "true_winner"]],
        on="state_po",
        how="left",
    )
    state_table["pred_dem_2p"] = state_table["dem_share_raw"] / (
        state_table["dem_share_raw"] + state_table["rep_share_raw"]
    )
    state_table["margin_error"] = state_table["margin_raw"] - state_table["true_margin"]
    state_table["correct"] = state_table["winner_raw"] == state_table["true_winner"]

    parse_summary = responses.groupby("parse_status").size().reset_index(name="n")
    match_summary = agents.groupby("match_level").size().reset_index(name="n_agents")
    metric_summary = metrics[metrics["state_po"].isna()][
        ["baseline", "metric_name", "metric_value"]
    ].sort_values(["baseline", "metric_name"])

    lines = [
        f"# Evaluation Report: {run_id}",
        "",
        "## 1. Run metadata",
        f"- Run ID: `{run_id}`",
        f"- Agents: {len(agents)}",
        f"- Responses: {len(responses)}",
        "",
        "## 2. Dataset versions",
        "- Fixture data uses `source=fixture`; real ANES/CES/MIT source files are not bundled.",
        "",
        "## 3. Population summary",
        _markdown_table(match_summary),
        "",
        "## 4. Baseline list",
        ", ".join(sorted(responses["baseline"].unique())),
        "",
        "## 5. Individual benchmark table",
        "Not implemented in v0 fixture smoke run.",
        "",
        "## 6. Distribution benchmark table",
        "Not implemented in v0 fixture smoke run.",
        "",
        "## 7. Election result table",
        state_table[
            [
                "state_po",
                "baseline",
                "pred_dem_2p",
                "true_dem_2p",
                "margin_error",
                "winner_raw",
                "true_winner",
                "correct",
            ]
        ].pipe(_markdown_table),
        "",
        "## 8. State-by-state error table",
        _markdown_table(state_table[["state_po", "baseline", "margin_error"]]),
        "",
        "## 9. Subgroup diagnostics",
        "Not implemented in v0 fixture smoke run.",
        "",
        "## 10. Calibration effect",
        "Calibration is not enabled in v0.",
        "",
        "## 11. Leakage diagnostics",
        "Leakage guard is applied before survey-memory prompts.",
        "",
        "## 12. Robustness summary",
        "Single-seed fixture smoke run only.",
        "",
        "## 13. Known failures",
        _markdown_table(parse_summary),
        "",
        "## Metrics",
        _markdown_table(metric_summary),
        "",
    ]
    out = run_dir / "eval_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
