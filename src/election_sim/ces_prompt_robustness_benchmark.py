"""Prompt robustness benchmark for CES turnout + vote simulation."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ces_aggregate_benchmark import AggregateLlmCache, complete_llm_task_with_cache, llm_cache_key
from .ces_schema import parse_turnout_vote_json
from .config import ModelConfig
from .eval_suite import (
    cfg_path,
    git_commit,
    gpu_peak_summary,
    gpu_snapshot,
    group_context_for_ids,
    group_memory_facts_for_ids,
    parse_quality_summary,
    raw_choice_diagnostics,
    sample_agents_by_state,
)
from .io import ensure_dir, load_yaml, write_json, write_table, write_yaml
from .llm import build_llm_client
from .prompts import build_ces_prompt
from .questions import load_question_config
from .transforms import stable_hash


PROMPT_VARIANTS = [
    "base_json",
    "json_strict_nonzero",
    "candidate_order_reversed",
    "interviewer_style",
    "analyst_style",
]
VOTE_EVAL_CLASSES = ["democrat", "not_vote", "republican"]


def _replace_task_line(prompt: str, new_line: str) -> str:
    return prompt.replace("Choose the single election behavior this voter would most likely take.", new_line).replace(
        "Estimate this voter's turnout probability and presidential vote choice.",
        new_line,
    )


def _strict_nonzero_contract(prompt: str) -> str:
    marker = "Return JSON only with this schema:"
    if marker not in prompt:
        return prompt
    prefix = prompt.split(marker, 1)[0].rstrip()
    return "\n".join(
        [
            prefix,
            "",
            "Return JSON only.",
            "Output rules:",
            "- choice must be exactly one of: not_vote, democrat, republican.",
            "- Do not include probabilities, confidence, markdown, or explanatory text.",
            "",
            '{"choice": "not_vote|democrat|republican"}',
        ]
    )


def render_prompt_variant(
    *,
    variant: str,
    agent: pd.Series,
    question: pd.Series,
    memory_facts: dict[str, pd.DataFrame],
    context: dict[str, list[dict[str, Any]]],
    max_memory_facts: int,
) -> tuple[str, list[str]]:
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt robustness variant: {variant}")
    context_for_prompt = context
    if variant == "candidate_order_reversed":
        base_ces_id = str(agent.get("base_ces_id") or agent.get("source_respondent_id"))
        context_for_prompt = dict(context)
        context_for_prompt[base_ces_id] = list(reversed(context_for_prompt.get(base_ces_id, [])))
    prompt, fact_ids = build_ces_prompt(
        agent,
        question,
        memory_facts=memory_facts,
        context=context_for_prompt,
        memory_policy="strict_pre_no_vote_v1",
        max_memory_facts=max_memory_facts,
        prompt_mode="ces_survey_memory",
    )
    if variant == "json_strict_nonzero":
        prompt = _strict_nonzero_contract(prompt)
    elif variant == "interviewer_style":
        prompt = prompt.replace(
            "Answer as this voter would behave, not as a political analyst.",
            "Please answer as this voter would behave in the election.",
        )
        prompt = _replace_task_line(prompt, "Please answer as this voter would behave in the election.")
    elif variant == "analyst_style":
        prompt = _replace_task_line(
            prompt,
            "Select this specific voter's single most likely election behavior.",
        )
    return prompt, fact_ids


def _combined_probability_frame(responses: pd.DataFrame) -> pd.DataFrame:
    turnout = responses["turnout_probability"].fillna(0.0).astype(float).clip(0.0, 1.0)
    probs = pd.DataFrame(
        {
            "democrat": turnout * responses["vote_prob_democrat"].fillna(0.0).astype(float).clip(0.0, 1.0),
            "republican": turnout * responses["vote_prob_republican"].fillna(0.0).astype(float).clip(0.0, 1.0),
            "not_vote": 1.0
            - turnout
            + turnout * responses["vote_prob_undecided"].fillna(0.0).astype(float).clip(0.0, 1.0),
        },
        index=responses.index,
    )
    denom = probs.sum(axis=1).replace(0.0, 1.0)
    return probs.div(denom, axis=0)[VOTE_EVAL_CLASSES]


def _target_wide(targets: pd.DataFrame) -> pd.DataFrame:
    return targets.pivot_table(index="ces_id", columns="target_id", values="canonical_value", aggfunc="first").reset_index()


def _weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float | None:
    if values.empty:
        return None
    vals = pd.to_numeric(values, errors="coerce")
    mask = vals.notna()
    if not mask.any():
        return None
    if weights is None:
        return float(vals[mask].mean())
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    w = w.where(w > 0, 0.0)
    if float(w[mask].sum()) <= 0:
        return float(vals[mask].mean())
    return float(np.average(vals[mask], weights=w[mask]))


def robustness_metric_rows(
    responses: pd.DataFrame,
    targets: pd.DataFrame,
    run_id: str,
) -> pd.DataFrame:
    quality = parse_quality_summary(responses, run_id=run_id, group_cols=["prompt_variant"])
    quality["baseline"] = quality["prompt_variant"]
    quality_rows = quality[["run_id", "baseline", "prompt_variant", "metric_name", "metric_value", "n", "created_at"]]
    target_wide = _target_wide(targets)
    merged = responses.merge(target_wide, left_on="base_ces_id", right_on="ces_id", how="left")
    rows: list[dict[str, Any]] = []
    for variant, group in merged.groupby("prompt_variant", sort=False):
        weights = group.get("sample_weight")
        ok = group[group["parse_status"] == "ok"].copy()
        if not ok.empty and "turnout_2024_self_report" in ok.columns:
            known_turnout = ok[ok["turnout_2024_self_report"].isin(["voted", "not_voted"])].copy()
            if not known_turnout.empty:
                y = (known_turnout["turnout_2024_self_report"] == "voted").astype(float)
                p = known_turnout["turnout_probability"].astype(float).clip(0.0, 1.0)
                brier = (p - y) ** 2
                rows.append(
                    {
                        "run_id": run_id,
                        "baseline": variant,
                        "prompt_variant": variant,
                        "metric_name": "turnout_brier",
                        "metric_value": _weighted_mean(brier, known_turnout.get("sample_weight")),
                        "n": int(len(known_turnout)),
                        "created_at": pd.Timestamp.now(tz="UTC"),
                    }
                )
        if not ok.empty and "president_vote_2024" in ok.columns:
            known_vote = ok[ok["president_vote_2024"].isin(VOTE_EVAL_CLASSES)].copy()
            if not known_vote.empty:
                y_true = known_vote["president_vote_2024"].astype(str)
                y_pred = known_vote["most_likely_choice"].fillna("not_vote").astype(str)
                y_pred = y_pred.where(y_pred.isin(VOTE_EVAL_CLASSES), "not_vote")
                correct = (y_true == y_pred).astype(float)
                rows.append(
                    {
                        "run_id": run_id,
                        "baseline": variant,
                        "prompt_variant": variant,
                        "metric_name": "vote_accuracy",
                        "metric_value": _weighted_mean(correct, known_vote.get("sample_weight")),
                        "n": int(len(known_vote)),
                        "created_at": pd.Timestamp.now(tz="UTC"),
                    }
                )
    metric_df = pd.DataFrame(rows)
    return pd.concat([quality_rows, metric_df], ignore_index=True) if not metric_df.empty else quality_rows


def _state_margins(responses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = responses[responses["parse_status"] == "ok"].copy()
    for (variant, state), group in ok.groupby(["prompt_variant", "state_po"], dropna=False):
        weights = group["sample_weight"].fillna(1.0).astype(float)
        turnout = group["turnout_probability"].fillna(0.0).astype(float).clip(0.0, 1.0)
        dem = float((weights * turnout * group["vote_prob_democrat"].fillna(0.0).astype(float)).sum())
        rep = float((weights * turnout * group["vote_prob_republican"].fillna(0.0).astype(float)).sum())
        margin = (dem - rep) / (dem + rep) if dem + rep else 0.0
        rows.append({"prompt_variant": variant, "state_po": state, "margin_2p": margin})
    return pd.DataFrame(rows)


def pairwise_variant_metric_rows(responses: pd.DataFrame, run_id: str, base_variant: str = "base_json") -> pd.DataFrame:
    if responses.empty or base_variant not in set(responses["prompt_variant"]):
        return pd.DataFrame()
    base = responses[responses["prompt_variant"] == base_variant].copy()
    rows: list[dict[str, Any]] = []
    for variant, comp in responses[responses["prompt_variant"] != base_variant].groupby("prompt_variant", sort=False):
        merged = base.merge(comp, on="base_ces_id", suffixes=("_base", "_variant"))
        both_ok = merged[(merged["parse_status_base"] == "ok") & (merged["parse_status_variant"] == "ok")].copy()
        n = int(len(both_ok))
        if n:
            metrics = {
                "choice_flip_rate": float((both_ok["most_likely_choice_base"] != both_ok["most_likely_choice_variant"]).mean()),
                "turnout_choice_flip_rate": float(
                    ((both_ok["most_likely_choice_base"] == "not_vote") != (both_ok["most_likely_choice_variant"] == "not_vote")).mean()
                ),
            }
        else:
            metrics = {
                "choice_flip_rate": None,
                "turnout_choice_flip_rate": None,
            }
        for metric_name, metric_value in metrics.items():
            rows.append(
                {
                    "run_id": run_id,
                    "base_variant": base_variant,
                    "prompt_variant": variant,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "n": n,
                    "state_po": None,
                    "created_at": pd.Timestamp.now(tz="UTC"),
                }
            )
    margins = _state_margins(responses)
    if not margins.empty:
        base_margins = margins[margins["prompt_variant"] == base_variant][["state_po", "margin_2p"]].rename(
            columns={"margin_2p": "base_margin_2p"}
        )
        for variant, group in margins[margins["prompt_variant"] != base_variant].groupby("prompt_variant", sort=False):
            joined = group.merge(base_margins, on="state_po", how="inner")
            for _, row in joined.iterrows():
                rows.append(
                    {
                        "run_id": run_id,
                        "base_variant": base_variant,
                        "prompt_variant": variant,
                        "metric_name": "state_margin_shift",
                        "metric_value": float(row["margin_2p"] - row["base_margin_2p"]),
                        "n": None,
                        "state_po": row["state_po"],
                        "created_at": pd.Timestamp.now(tz="UTC"),
                    }
                )
    return pd.DataFrame(rows)


def _write_figures(run_dir: Path, robustness: pd.DataFrame, pairwise: pd.DataFrame) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = ensure_dir(run_dir / "figures")
    written: list[Path] = []
    if not robustness.empty:
        parse = robustness[robustness["metric_name"] == "parse_ok_rate"].copy()
        if not parse.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(parse["prompt_variant"], parse["metric_value"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("parse_ok_rate")
            ax.tick_params(axis="x", rotation=30)
            out = fig_dir / "e03_parse_ok_by_variant.png"
            fig.savefig(out, dpi=160, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    if not pairwise.empty:
        for metric, filename in [
            ("choice_flip_rate", "e03_choice_flip_rate_vs_base.png"),
            ("turnout_choice_flip_rate", "e03_turnout_choice_flip_rate_vs_base.png"),
            ("state_margin_shift", "e03_state_margin_shift_by_variant.png"),
        ]:
            part = pairwise[pairwise["metric_name"] == metric].copy()
            if part.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            labels = part["prompt_variant"] if metric != "state_margin_shift" else part["prompt_variant"] + ":" + part["state_po"].astype(str)
            ax.bar(labels, part["metric_value"])
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=30)
            out = fig_dir / filename
            fig.savefig(out, dpi=160, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    return written


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        display[col] = display[col].map(lambda value: "" if pd.isna(value) else str(value))
    headers = list(display.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def run_ces_prompt_robustness_benchmark(config_path: str | Path) -> dict[str, Path]:
    cfg = load_yaml(config_path)
    run_id = cfg["run_id"]
    run_dir = ensure_dir(cfg.get("paths", {}).get("run_dir", f"data/runs/{run_id}"))
    write_yaml(cfg, run_dir / "config_snapshot.yaml")
    started = time.time()

    respondents = pd.read_parquet(cfg_path(cfg, "ces_respondents"))
    targets = pd.read_parquet(cfg_path(cfg, "ces_targets"))
    context = pd.read_parquet(cfg_path(cfg, "ces_context"))
    strict_memory_facts = pd.read_parquet(cfg_path(cfg, "strict_memory_facts", "ces_memory_facts_strict"))
    question = load_question_config(cfg_path(cfg, "question_set")).iloc[0]
    states = list(cfg.get("states", ["PA", "GA"]))
    sampled, agents = sample_agents_by_state(
        respondents=respondents,
        run_id=run_id,
        states=states,
        agents_per_state=int(cfg.get("agents_per_state", 5)),
        seed=int(cfg.get("seed", 20260426)),
        split=str(cfg.get("sampling", {}).get("split", "test")),
    )
    ces_ids = set(agents["base_ces_id"].dropna().astype(str))
    memory_by_id = group_memory_facts_for_ids(strict_memory_facts, ces_ids)
    context_by_id = group_context_for_ids(context, ces_ids)
    variants = list(cfg.get("prompt_variants", PROMPT_VARIANTS))
    max_memory_facts = int(cfg.get("memory", {}).get("max_memory_facts", 24))

    model_cfg = ModelConfig.model_validate(cfg.get("model", {"provider": "mock"}))
    client = build_llm_client(model_cfg)
    model_name = getattr(client, "model_name", model_cfg.model_name)
    cache = AggregateLlmCache(run_dir / "llm_cache.jsonl")
    llm_cfg = cfg.get("llm", {})
    workers = max(1, int(llm_cfg.get("workers", 1)))
    checkpoint_every = max(1, int(llm_cfg.get("checkpoint_every", 25)))
    gpu_sample_every = max(1, int(llm_cfg.get("gpu_sample_every", checkpoint_every)))
    gpu_snapshots = [gpu_snapshot("run_start")]
    tasks: list[dict[str, Any]] = []
    for _, agent in agents.iterrows():
        for variant in variants:
            prompt_text, fact_ids = render_prompt_variant(
                variant=variant,
                agent=agent,
                question=question,
                memory_facts=memory_by_id,
                context=context_by_id,
                max_memory_facts=max_memory_facts,
            )
            prompt_hash = stable_hash(prompt_text, length=32)
            prompt_id = stable_hash(run_id, variant, agent["base_ces_id"], prompt_hash, length=20)
            cache_key = llm_cache_key(
                model_name=model_name,
                baseline=variant,
                prompt_hash=prompt_hash,
                temperature=float(model_cfg.temperature),
                max_tokens=int(model_cfg.max_tokens),
                response_format=str(model_cfg.response_format),
            )
            tasks.append(
                {
                    "prompt_id": prompt_id,
                    "agent_id": agent["agent_id"],
                    "base_ces_id": str(agent["base_ces_id"]),
                    "state_po": agent["state_po"],
                    "sample_weight": agent["sample_weight"],
                    "prompt_variant": variant,
                    "prompt_hash": prompt_hash,
                    "prompt_text": prompt_text,
                    "memory_fact_ids_used": fact_ids,
                    "cache_key": cache_key,
                }
            )

    def finish(task: dict[str, Any]) -> tuple[dict[str, Any], str, bool, int | None, str | None]:
        started_call = time.time()
        try:
            raw, cache_hit, latency_ms = complete_llm_task_with_cache(
                client=client,
                cache=cache,
                cache_key=task["cache_key"],
                run_id=run_id,
                model_name=model_name,
                baseline=task["prompt_variant"],
                prompt_hash=task["prompt_hash"],
                prompt_text=task["prompt_text"],
            )
            return task, raw, cache_hit, latency_ms, None
        except Exception as exc:
            latency_ms = int((time.time() - started_call) * 1000)
            return task, "", False, latency_ms, f"{type(exc).__name__}: {exc}"

    prompt_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    llm_started = time.time()

    def write_checkpoint() -> None:
        if prompt_rows:
            write_table(pd.DataFrame(prompt_rows), run_dir / "prompts.partial.parquet")
        if response_rows:
            write_table(pd.DataFrame(response_rows), run_dir / "responses.partial.parquet")
        if runtime_rows:
            write_table(pd.DataFrame(runtime_rows), run_dir / "runtime_log.partial.parquet")

    def record_result(
        task: dict[str, Any],
        raw: str,
        cache_hit: bool,
        latency_ms: int | None,
        transport_error: str | None,
    ) -> None:
        parsed = parse_turnout_vote_json(raw)
        parse_status = "transport_error" if transport_error else parsed["parse_status"]
        raw_diag = raw_choice_diagnostics(raw)
        prompt_rows.append(
            {
                "run_id": run_id,
                "prompt_id": task["prompt_id"],
                "agent_id": task["agent_id"],
                "base_ces_id": task["base_ces_id"],
                "baseline": task["prompt_variant"],
                "condition": task["prompt_variant"],
                "prompt_variant": task["prompt_variant"],
                "model_name": model_name,
                "prompt_hash": task["prompt_hash"],
                "prompt_text": task["prompt_text"],
                "memory_fact_ids_used": task["memory_fact_ids_used"],
                "cache_hit": cache_hit,
                "latency_ms": latency_ms,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
        response_rows.append(
            {
                "run_id": run_id,
                "response_id": stable_hash(run_id, task["agent_id"], task["prompt_variant"], length=20),
                "prompt_id": task["prompt_id"],
                "agent_id": task["agent_id"],
                "base_ces_id": task["base_ces_id"],
                "baseline": task["prompt_variant"],
                "condition": task["prompt_variant"],
                "prompt_variant": task["prompt_variant"],
                "model_name": model_name,
                "raw_response": raw,
                "parse_status": parse_status,
                "turnout_probability": parsed["turnout_probability"],
                "vote_prob_democrat": parsed["vote_prob_democrat"],
                "vote_prob_republican": parsed["vote_prob_republican"],
                "vote_prob_other": parsed["vote_prob_other"],
                "vote_prob_undecided": parsed["vote_prob_undecided"],
                "most_likely_choice": parsed["most_likely_choice"],
                "confidence": parsed["confidence"],
                "sample_weight": task["sample_weight"],
                "state_po": task["state_po"],
                "cache_hit": cache_hit,
                "latency_ms": latency_ms,
                "transport_error": transport_error,
                **raw_diag,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
        runtime_rows.append(
            {
                "run_id": run_id,
                "event": "llm_response",
                "prompt_id": task["prompt_id"],
                "agent_id": task["agent_id"],
                "prompt_variant": task["prompt_variant"],
                "base_ces_id": task["base_ces_id"],
                "state_po": task["state_po"],
                "cache_hit": cache_hit,
                "latency_ms": latency_ms,
                "parse_status": parse_status,
                "raw_choice": raw_diag["raw_choice"],
                "invalid_choice": bool(raw_diag["invalid_choice"] or parse_status == "invalid_choice"),
                "forbidden_choice": bool(raw_diag["forbidden_choice"]),
                "legacy_probability_schema": bool(raw_diag["legacy_probability_schema"] or parsed.get("legacy_probability_schema")),
                "transport_error": transport_error,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
        completed_count = len(response_rows)
        if completed_count % gpu_sample_every == 0:
            gpu_snapshots.append(gpu_snapshot(f"after_response_{completed_count}"))
        if completed_count % checkpoint_every == 0:
            write_checkpoint()

    if workers == 1:
        for task in tasks:
            record_result(*finish(task))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(finish, task) for task in tasks]
            for future in as_completed(futures):
                record_result(*future.result())
    gpu_snapshots.append(gpu_snapshot("run_end"))
    write_checkpoint()
    variant_order = {variant: idx for idx, variant in enumerate(variants)}
    prompt_rows.sort(key=lambda row: (str(row["base_ces_id"]), variant_order.get(row["prompt_variant"], 999)))
    response_rows.sort(key=lambda row: (str(row["base_ces_id"]), variant_order.get(row["prompt_variant"], 999)))
    runtime_rows.sort(key=lambda row: (str(row["base_ces_id"]), variant_order.get(row["prompt_variant"], 999)))
    llm_runtime_seconds = time.time() - llm_started
    prompts = pd.DataFrame(prompt_rows)
    responses = pd.DataFrame(response_rows)
    robustness = robustness_metric_rows(responses, targets, run_id)
    pairwise = pairwise_variant_metric_rows(responses, run_id)
    variant_meta = pd.DataFrame(
        [
            {
                "prompt_variant": variant,
                "description": {
                    "base_json": "Current strict-memory hard-choice JSON prompt.",
                    "json_strict_nonzero": "Same information with stricter hard-choice JSON-only rules.",
                    "candidate_order_reversed": "Same information with candidate lines reversed.",
                    "interviewer_style": "Same information with survey-interviewer task wording.",
                    "analyst_style": "Same information with concise analyst-style hard-choice wording.",
                }.get(variant, variant),
            }
            for variant in variants
        ]
    )
    write_table(agents, run_dir / "agents.parquet")
    write_table(prompts, run_dir / "prompts.parquet")
    write_table(responses, run_dir / "responses.parquet")
    write_table(variant_meta, run_dir / "prompt_variant_metadata.parquet")
    write_table(robustness, run_dir / "robustness_metrics.parquet")
    write_table(pairwise, run_dir / "pairwise_variant_metrics.parquet")
    write_table(pd.DataFrame(runtime_rows), run_dir / "runtime_log.parquet")
    figures = _write_figures(run_dir, robustness, pairwise)
    runtime_log = pd.DataFrame(runtime_rows)
    latency = pd.to_numeric(runtime_log.get("latency_ms", pd.Series(dtype=float)), errors="coerce").dropna()
    parse_status = runtime_log.get("parse_status", pd.Series(dtype=str)).fillna("").astype(str) if not runtime_log.empty else pd.Series(dtype=str)
    invalid_choice = runtime_log.get("invalid_choice", pd.Series(dtype=bool)).fillna(False).astype(bool) if not runtime_log.empty else pd.Series(dtype=bool)
    forbidden_choice = runtime_log.get("forbidden_choice", pd.Series(dtype=bool)).fillna(False).astype(bool) if not runtime_log.empty else pd.Series(dtype=bool)
    legacy_schema = runtime_log.get("legacy_probability_schema", pd.Series(dtype=bool)).fillna(False).astype(bool) if not runtime_log.empty else pd.Series(dtype=bool)
    transport_errors = runtime_log.get("transport_error", pd.Series(dtype=object)).notna() if not runtime_log.empty else pd.Series(dtype=bool)
    cache_hits = responses["cache_hit"].fillna(False).astype(bool) if "cache_hit" in responses.columns and not responses.empty else pd.Series(dtype=bool)
    parse_ok_rate = float((parse_status == "ok").mean()) if len(parse_status) else None
    invalid_choice_rate = float(invalid_choice.mean()) if len(invalid_choice) else None
    forbidden_choice_rate = float(forbidden_choice.mean()) if len(forbidden_choice) else None
    legacy_schema_rate = float(legacy_schema.mean()) if len(legacy_schema) else None
    transport_error_rate = float(transport_errors.mean()) if len(transport_errors) else None
    runtime = {
        "run_id": run_id,
        "git_commit": git_commit(),
        "model_name": model_name,
        "provider": model_cfg.provider,
        "temperature": model_cfg.temperature,
        "max_tokens": model_cfg.max_tokens,
        "workers": workers,
        "prompt_variants": variants,
        "n_agents": int(len(agents)),
        "n_prompts": int(len(prompts)),
        "runtime_seconds": time.time() - started,
        "llm_runtime_seconds": llm_runtime_seconds,
        "median_latency_seconds": float(latency.median() / 1000.0) if not latency.empty else None,
        "p90_latency_seconds": float(latency.quantile(0.9) / 1000.0) if not latency.empty else None,
        "throughput_responses_per_second": float(len(responses) / llm_runtime_seconds) if llm_runtime_seconds > 0 else None,
        "cache_hit_rate": float(cache_hits.mean()) if len(cache_hits) else None,
        "ollama_calls": int((~cache_hits).sum()) if len(cache_hits) else 0,
        "parse_ok_rate": parse_ok_rate,
        "invalid_choice_rate": invalid_choice_rate,
        "forbidden_choice_rate": forbidden_choice_rate,
        "legacy_probability_schema_rate": legacy_schema_rate,
        "transport_error_rate": transport_error_rate,
        "all_gates_passed": bool(
            (parse_ok_rate is None or parse_ok_rate >= 0.95)
            and (invalid_choice_rate is None or invalid_choice_rate <= 0.02)
            and (forbidden_choice_rate is None or forbidden_choice_rate == 0.0)
            and (legacy_schema_rate is None or legacy_schema_rate == 0.0)
            and (transport_error_rate is None or transport_error_rate == 0.0)
        ),
        **gpu_peak_summary(gpu_snapshots),
        "gpu_snapshots": gpu_snapshots,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    write_json(runtime, run_dir / "runtime.json")
    report = run_dir / "report.md"
    report.write_text(
        "\n".join(
            [
                f"# Prompt Robustness Report: {run_id}",
                "",
                "## Run Metadata",
                f"- Model: `{model_name}`",
                f"- Agents: {len(agents)}",
                f"- Prompts: {len(prompts)}",
                f"- Variants: {', '.join(variants)}",
                "",
                "## Robustness Metrics",
                _markdown_table(robustness.head(120)),
                "",
                "## Pairwise Metrics",
                _markdown_table(pairwise.head(120)),
                "",
                "## Figures",
                "\n".join(f"- `{path.relative_to(run_dir)}`" for path in figures) if figures else "_No figures._",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "agents": run_dir / "agents.parquet",
        "prompts": run_dir / "prompts.parquet",
        "responses": run_dir / "responses.parquet",
        "prompt_variant_metadata": run_dir / "prompt_variant_metadata.parquet",
        "robustness_metrics": run_dir / "robustness_metrics.parquet",
        "pairwise_variant_metrics": run_dir / "pairwise_variant_metrics.parquet",
        "runtime_log": run_dir / "runtime_log.parquet",
        "runtime": run_dir / "runtime.json",
        "report": report,
    }
