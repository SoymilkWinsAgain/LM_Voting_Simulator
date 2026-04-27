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
    mit_results: pd.DataFrame | None = None,
    dataset_artifacts: dict[str, str] | None = None,
    population_source: str = "ces_rows",
) -> Path:
    run_dir = Path(run_dir)
    parse_summary = responses.groupby("parse_status").size().reset_index(name="n")
    state_summary = agents.groupby("state_po").size().reset_index(name="n_agents")
    fact_counts = prompts["memory_fact_ids_used"].map(lambda value: len(value) if isinstance(value, list) else 0)
    metric_cols = ["metric_scope", "baseline", "model_name", "metric_name", "metric_value"]
    comparison_cols = ["baseline", "model_name", "metric_name", "metric_value"]
    leakage_table = pd.DataFrame()
    if leakage_audit is not None and not leakage_audit.empty:
        leakage_table = leakage_audit[leakage_audit["excluded"].astype(bool)].head(20)
    leakage_summary = pd.DataFrame()
    poll_prior_summary = pd.DataFrame()
    if leakage_audit is not None and not leakage_audit.empty:
        leakage_summary = (
            leakage_audit.groupby(["policy", "reason", "excluded"], dropna=False)
            .size()
            .reset_index(name="n_variables")
        )
        if "fact_role" in leakage_audit.columns:
            poll_prior_summary = leakage_audit[
                (leakage_audit["fact_role"] == "poll_prior") & (~leakage_audit["excluded"].astype(bool))
            ][["source_variable", "policy", "fact_role", "potential_leakage_warning"]].head(30)
    subgroup_metrics = individual_metrics[individual_metrics["metric_scope"] == "subgroup"].copy()
    individual_core = individual_metrics[individual_metrics["metric_scope"] == "individual"].copy()
    baseline_comparison = pd.concat(
        [
            individual_core[individual_core["metric_name"].isin(["turnout_brier", "vote_accuracy", "vote_log_loss"])],
            aggregate_metrics[
                aggregate_metrics["metric_name"].isin(["state_dem_2p_rmse", "state_margin_mae", "winner_accuracy"])
            ],
        ],
        ignore_index=True,
    )
    state_comparison = aggregate.copy()
    if mit_results is not None and not mit_results.empty and not aggregate.empty:
        truth = state_truth_table(mit_results)
        state_comparison = aggregate.merge(
            truth[["state_po", "true_dem_2p", "true_margin", "true_winner"]],
            on="state_po",
            how="left",
        )
        state_comparison["dem_2p_error"] = state_comparison["dem_share_2p"] - state_comparison["true_dem_2p"]
        state_comparison["margin_error"] = state_comparison["margin_2p"] - state_comparison["true_margin"]
    parse_failures = responses[responses["parse_status"] != "ok"][
        ["agent_id", "baseline", "model_name", "parse_status", "raw_response"]
    ].head(20)
    artifacts = pd.DataFrame(
        [{"artifact": key, "path": value} for key, value in (dataset_artifacts or {}).items()]
    )
    lines = [
        f"# CES 2024 Smoke Report: {run_id}",
        "",
        "## Run metadata",
        f"- Run ID: `{run_id}`",
        f"- Population source: `{population_source}`",
        f"- Agents: {len(agents)}",
        f"- Prompts: {len(prompts)}",
        f"- Responses: {len(responses)}",
        f"- Memory policy: `{memory_policy}`",
        f"- Weight column: `{weight_column}`",
        f"- Explanatory-only policy: `{memory_policy == 'post_hoc_explanation_v1'}`",
        "",
        "## Dataset artifacts",
        _markdown_table(artifacts),
        "",
        "## Population summary",
        _markdown_table(state_summary),
        "",
        "## Model / baseline list",
        _markdown_table(responses[["baseline", "model_name"]].drop_duplicates().sort_values(["baseline", "model_name"])),
        "",
        "## Prompt fact coverage",
        f"- Mean facts per prompt: {fact_counts.mean():.2f}" if len(fact_counts) else "- Mean facts per prompt: 0.00",
        f"- Max facts in a prompt: {int(fact_counts.max())}" if len(fact_counts) else "- Max facts in a prompt: 0",
        "",
        "## Parse status",
        _markdown_table(parse_summary),
        "",
        "## Known failures",
        _markdown_table(parse_failures),
        "",
        "## Individual metrics",
        _markdown_table(individual_core[[col for col in metric_cols if col in individual_core.columns]]),
        "",
        "## Subgroup metrics",
        _markdown_table(
            subgroup_metrics[
                [
                    col
                    for col in [*metric_cols, "group_key", "n", "small_n"]
                    if col in subgroup_metrics.columns
                ]
            ].head(120)
        ),
        "",
        "## Baseline comparison",
        _markdown_table(baseline_comparison[[col for col in comparison_cols if col in baseline_comparison.columns]]),
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
        "## MIT official comparison",
        _markdown_table(
            state_comparison[
                [
                    col
                    for col in [
                        "state_po",
                        "baseline",
                        "model_name",
                        "dem_share_2p",
                        "true_dem_2p",
                        "dem_2p_error",
                        "margin_2p",
                        "true_margin",
                        "margin_error",
                        "winner",
                        "true_winner",
                    ]
                    if col in state_comparison.columns
                ]
            ]
        ),
        "",
        "## Leakage audit summary",
        _markdown_table(leakage_summary),
        "",
        "## Poll prior facts",
        _markdown_table(poll_prior_summary),
        "",
        "## Leakage diagnostics",
        _markdown_table(
            leakage_table[
                [
                    col
                    for col in [
                        "source_variable",
                        "policy",
                        "excluded",
                        "reason",
                        "fact_role",
                        "potential_leakage_warning",
                    ]
                    if col in leakage_table.columns
                ]
            ]
            if not leakage_table.empty
            else leakage_table
        ),
        "",
        "## TargetSmart / validation use",
        "TargetSmart-derived fields are treated as evaluation-only under strict and poll-informed policies. "
        "Strict prompts must not include validated turnout, validated vote mode, turnout history, or validated party registration.",
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
