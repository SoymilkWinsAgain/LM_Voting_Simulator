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


def write_individual_report(
    *,
    run_id: str,
    run_dir: str | Path,
    agents: pd.DataFrame,
    prompts: pd.DataFrame,
    responses: pd.DataFrame,
    metrics: pd.DataFrame,
) -> Path:
    run_dir = Path(run_dir)
    agent_cols = [
        "agent_id",
        "base_anes_id",
        "gender",
        "race_ethnicity",
        "education_binary",
        "party_id_3",
        "ideology_3",
    ]
    response_cols = [
        "baseline",
        "model_name",
        "parsed_answer_code",
        "confidence",
        "parse_status",
        "target_answer_code",
        "correct",
    ]
    lines = [
        f"# Individual Smoke Report: {run_id}",
        "",
        "## Run metadata",
        f"- Run ID: `{run_id}`",
        f"- Agents: {len(agents)}",
        f"- Prompts: {len(prompts)}",
        f"- Responses: {len(responses)}",
        "",
        "## Agent",
        _markdown_table(agents[[col for col in agent_cols if col in agents.columns]]),
        "",
        "## Response",
        _markdown_table(responses[[col for col in response_cols if col in responses.columns]]),
        "",
        "## Metrics",
        _markdown_table(metrics[["metric_scope", "baseline", "metric_name", "metric_value"]]),
        "",
        "## Prompt preview",
        "See `prompt_preview.md` in this run directory.",
        "",
    ]
    out = run_dir / "eval_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_ces_eval_report(
    *,
    run_id: str,
    run_dir: str | Path,
    agents: pd.DataFrame,
    prompts: pd.DataFrame,
    responses: pd.DataFrame,
    individual_metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    memory_policy: str,
    weight_column: str,
    leakage_audit: pd.DataFrame | None = None,
) -> Path:
    run_dir = Path(run_dir)
    parse_summary = responses.groupby("parse_status").size().reset_index(name="n")
    state_summary = agents.groupby("state_po").size().reset_index(name="n_agents")
    fact_counts = prompts["memory_fact_ids_used"].map(lambda value: len(value) if isinstance(value, list) else 0)
    metric_cols = ["metric_scope", "baseline", "model_name", "metric_name", "metric_value"]
    leakage_table = pd.DataFrame()
    if leakage_audit is not None and not leakage_audit.empty:
        leakage_table = leakage_audit[leakage_audit["excluded"].astype(bool)].head(20)
    lines = [
        f"# CES 2024 Smoke Report: {run_id}",
        "",
        "## Run metadata",
        f"- Run ID: `{run_id}`",
        f"- Agents: {len(agents)}",
        f"- Prompts: {len(prompts)}",
        f"- Responses: {len(responses)}",
        f"- Memory policy: `{memory_policy}`",
        f"- Weight column: `{weight_column}`",
        "",
        "## Population summary",
        _markdown_table(state_summary),
        "",
        "## Prompt fact coverage",
        f"- Mean facts per prompt: {fact_counts.mean():.2f}" if len(fact_counts) else "- Mean facts per prompt: 0.00",
        f"- Max facts in a prompt: {int(fact_counts.max())}" if len(fact_counts) else "- Max facts in a prompt: 0",
        "",
        "## Parse status",
        _markdown_table(parse_summary),
        "",
        "## Individual metrics",
        _markdown_table(individual_metrics[[col for col in metric_cols if col in individual_metrics.columns]]),
        "",
        "## Aggregate state results",
        _markdown_table(
            aggregate[
                [
                    col
                    for col in [
                        "state_po",
                        "baseline",
                        "n_agents",
                        "expected_turnout",
                        "dem_share_2p",
                        "rep_share_2p",
                        "margin_2p",
                        "winner",
                    ]
                    if col in aggregate.columns
                ]
            ]
        ),
        "",
        "## Aggregate metrics",
        _markdown_table(aggregate_metrics[[col for col in metric_cols if col in aggregate_metrics.columns]]),
        "",
        "## Leakage diagnostics",
        _markdown_table(
            leakage_table[["source_variable", "policy", "excluded", "reason"]]
            if not leakage_table.empty
            else leakage_table
        ),
        "",
        "## Prompt preview",
        "See `prompt_preview.md` in this run directory.",
        "",
        "## Known limitations",
        "Phase 1 smoke uses a small respondent sample and mock provider by default; aggregate metrics against MIT are not required unless configured separately.",
        "",
    ]
    out = run_dir / "eval_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
