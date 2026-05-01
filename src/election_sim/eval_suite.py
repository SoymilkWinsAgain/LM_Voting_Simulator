"""Evaluation-suite preflight and summary helpers."""

from __future__ import annotations

import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from .ces_aggregate_benchmark import AggregateLlmCache, complete_llm_task_with_cache, llm_cache_key
from .ces_benchmark import _agents_from_cohort, build_benchmark_cohort
from .ces_schema import CES_TURNOUT_VOTE_CHOICES, parse_turnout_vote_json
from .config import ModelConfig
from .io import ensure_dir, load_yaml, write_json, write_table, write_yaml
from .llm import build_llm_client
from .prompts import build_ces_prompt
from .questions import load_question_config
from .transforms import stable_hash


POST_OR_DIRECT_BLOCKED_VARIABLES = {
    "CC24_401",
    "CC24_410",
    "CC24_410_nv",
    "CC24_363",
    "CC24_364a",
    "CC24_365",
    "CC24_366",
    "CC24_367",
}
POLL_PRIOR_VARIABLES = {"CC24_363", "CC24_364a"}
POST_TARGET_VARIABLES = {"CC24_401", "CC24_410", "CC24_410_nv"}


def cfg_path(cfg: dict[str, Any], *keys: str) -> Path:
    paths = cfg.get("paths", {})
    for key in keys:
        if key in paths:
            return Path(paths[key])
    raise KeyError(f"Config missing paths key; tried {list(keys)}")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=5).strip()
    except Exception:
        return "unknown"


def gpu_snapshot(label: str) -> dict[str, Any]:
    """Return a best-effort nvidia-smi snapshot without making GPU monitoring fatal."""

    base: dict[str, Any] = {"label": label, "created_at": pd.Timestamp.now(tz="UTC").isoformat()}
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
    except Exception as exc:
        return {**base, "gpu_available": False, "gpu_error": f"{type(exc).__name__}: {exc}"}

    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mib": int(parts[2]),
                    "memory_used_mib": int(parts[3]),
                    "memory_free_mib": int(parts[4]),
                    "utilization_gpu_pct": int(parts[5]),
                }
            )
        except ValueError:
            continue
    if not gpus:
        return {**base, "gpu_available": False, "gpu_error": f"Could not parse nvidia-smi output: {output}"}
    used = [gpu["memory_used_mib"] for gpu in gpus]
    free = [gpu["memory_free_mib"] for gpu in gpus]
    util = [gpu["utilization_gpu_pct"] for gpu in gpus]
    return {
        **base,
        "gpu_available": True,
        "gpu_count": len(gpus),
        "gpu_memory_total_mib": int(sum(gpu["memory_total_mib"] for gpu in gpus)),
        "gpu_memory_used_mib": int(sum(used)),
        "gpu_memory_free_mib": int(sum(free)),
        "gpu_memory_used_mib_max": int(max(used)),
        "gpu_memory_free_mib_min": int(min(free)),
        "gpu_utilization_pct_max": int(max(util)),
        "gpu_utilization_pct_mean": float(sum(util) / len(util)),
        "gpus": gpus,
    }


def gpu_peak_summary(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    available = [snapshot for snapshot in snapshots if snapshot.get("gpu_available")]
    if not available:
        errors = [snapshot.get("gpu_error") for snapshot in snapshots if snapshot.get("gpu_error")]
        return {
            "gpu_available": False,
            "gpu_error": "; ".join(str(error) for error in errors if error) or "No GPU snapshots available",
        }
    peak_used = max(int(snapshot.get("gpu_memory_used_mib", 0)) for snapshot in available)
    min_free = min(int(snapshot.get("gpu_memory_free_mib", 0)) for snapshot in available)
    peak_util = max(int(snapshot.get("gpu_utilization_pct_max", 0)) for snapshot in available)
    return {
        "gpu_available": True,
        "gpu_peak_memory_used_mib": peak_used,
        "gpu_min_memory_free_mib": min_free,
        "gpu_peak_utilization_pct": peak_util,
    }


def extract_json_payload(raw_response: str) -> dict[str, Any] | None:
    text = str(raw_response or "").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def raw_choice_diagnostics(raw_response: str) -> dict[str, Any]:
    payload = extract_json_payload(raw_response)
    if payload is None:
        return {
            "raw_json_ok": False,
            "raw_choice": None,
            "invalid_choice": False,
            "forbidden_choice": False,
            "legacy_probability_schema": False,
        }
    choice = payload.get("choice")
    choice = choice if isinstance(choice, str) else None
    legacy = "vote_probabilities" in payload or "turnout_probability" in payload
    return {
        "raw_json_ok": True,
        "raw_choice": choice,
        "invalid_choice": choice is not None and choice not in CES_TURNOUT_VOTE_CHOICES,
        "forbidden_choice": choice in {"other", "undecided"},
        "legacy_probability_schema": legacy,
    }


def parse_quality_summary(
    responses: pd.DataFrame,
    *,
    run_id: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    if responses.empty:
        return pd.DataFrame()
    group_cols = group_cols or []
    groups = [((), responses)] if not group_cols else responses.groupby(group_cols, dropna=False)
    rows: list[dict[str, Any]] = []
    for keys, group in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {"run_id": run_id, "n": int(len(group)), "created_at": pd.Timestamp.now(tz="UTC")}
        for col, value in zip(group_cols, keys, strict=False):
            base[col] = value
        parse_status = group.get("parse_status", pd.Series("", index=group.index)).fillna("").astype(str)
        forbidden_choice = group.get("forbidden_choice", pd.Series(False, index=group.index)).fillna(False)
        legacy_schema = group.get("legacy_probability_schema", pd.Series(False, index=group.index)).fillna(False)
        cache_hit = group.get("cache_hit", pd.Series(dtype=bool))
        latency = pd.to_numeric(group.get("latency_ms", pd.Series(dtype=float)), errors="coerce")
        metrics = {
            "parse_ok_rate": float((parse_status == "ok").mean()) if len(group) else 0.0,
            "transport_error_rate": float((parse_status == "transport_error").mean()) if len(group) else 0.0,
            "invalid_json_rate": float((parse_status == "failed").mean()) if len(group) else 0.0,
            "invalid_schema_rate": float((parse_status == "invalid_schema").mean()) if len(group) else 0.0,
            "invalid_choice_rate": float((parse_status == "invalid_choice").mean()) if len(group) else 0.0,
            "legacy_probability_schema_rate": float(legacy_schema.astype(bool).mean()) if len(group) else 0.0,
            "forbidden_choice_rate": float(forbidden_choice.astype(bool).mean()) if len(group) else 0.0,
            "cache_hit_rate": float(cache_hit.fillna(False).astype(bool).mean()) if len(cache_hit) else None,
            "median_latency_seconds": float(latency.dropna().median() / 1000.0) if latency.notna().any() else None,
        }
        for metric_name, metric_value in metrics.items():
            rows.append({**base, "metric_name": metric_name, "metric_value": metric_value})
    return pd.DataFrame(rows)


def sample_agents_by_state(
    *,
    respondents: pd.DataFrame,
    run_id: str,
    states: list[str],
    agents_per_state: int,
    seed: int,
    split: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = build_benchmark_cohort(respondents, seed, states)
    if split != "all" and "split" in cohort.columns:
        cohort = cohort[cohort["split"] == split].copy()
    parts = []
    for state in states:
        group = cohort[cohort["state_po"] == state].copy()
        if group.empty:
            continue
        n = min(int(agents_per_state), len(group))
        weights = group["sample_weight"].fillna(0).astype(float) if "sample_weight" in group.columns else None
        if weights is not None and float(weights.sum()) <= 0:
            weights = None
        random_state = int(stable_hash("eval-suite-sample", run_id, seed, state, length=8), 16)
        try:
            sampled = group.sample(n=n, weights=weights, replace=False, random_state=random_state).copy()
        except ValueError as exc:
            if weights is None or "Weighted sampling cannot be achieved" not in str(exc):
                raise
            sampled = group.sample(n=n, replace=False, random_state=random_state).copy()
        sampled["sample_rank"] = range(1, len(sampled) + 1)
        parts.append(sampled)
    sampled = pd.concat(parts, ignore_index=True) if parts else cohort.head(0).copy()
    agents = _agents_from_cohort(run_id, sampled)
    if not sampled.empty:
        agents = agents.merge(
            sampled[["ces_id", "sample_rank"]],
            left_on="base_ces_id",
            right_on="ces_id",
            how="left",
        ).drop(columns=["ces_id"])
    return sampled, agents


def group_memory_facts_for_ids(memory_facts: pd.DataFrame, ces_ids: set[str]) -> dict[str, pd.DataFrame]:
    if memory_facts.empty or not ces_ids:
        return {}
    work = memory_facts[memory_facts["ces_id"].astype(str).isin(ces_ids)].copy()
    return {str(ces_id): group.copy() for ces_id, group in work.groupby("ces_id", sort=False)}


def group_context_for_ids(context: pd.DataFrame, ces_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    if context.empty or not ces_ids:
        return {}
    work = context[context["ces_id"].astype(str).isin(ces_ids)].copy()
    return {str(ces_id): group.to_dict("records") for ces_id, group in work.groupby("ces_id", sort=False)}


def _leakage_check_rows(
    *,
    strict_facts: pd.DataFrame,
    poll_facts: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    strict_vars = set(strict_facts.get("source_variable", pd.Series(dtype=str)).astype(str))
    poll_vars = set(poll_facts.get("source_variable", pd.Series(dtype=str)).astype(str))
    rows.append(
        {
            "check_name": "strict_blocks_post_direct_and_targetsmart",
            "passed": not bool(POST_OR_DIRECT_BLOCKED_VARIABLES & strict_vars)
            and not any(var.upper().startswith("TS_") for var in strict_vars),
            "details": ",".join(sorted(POST_OR_DIRECT_BLOCKED_VARIABLES & strict_vars)),
        }
    )
    poll_prior = poll_facts[poll_facts.get("source_variable", pd.Series(dtype=str)).astype(str).isin(POLL_PRIOR_VARIABLES)]
    rows.append(
        {
            "check_name": "poll_informed_has_poll_prior_role",
            "passed": not poll_prior.empty and set(poll_prior["fact_role"].fillna("").astype(str)) <= {"poll_prior"},
            "details": ",".join(sorted(set(poll_prior.get("source_variable", pd.Series(dtype=str)).astype(str)))),
        }
    )
    rows.append(
        {
            "check_name": "poll_informed_blocks_post_and_targetsmart",
            "passed": not bool(POST_TARGET_VARIABLES & poll_vars)
            and not any(var.upper().startswith("TS_") for var in poll_vars),
            "details": ",".join(sorted(POST_TARGET_VARIABLES & poll_vars)),
        }
    )
    return pd.DataFrame(rows)


def run_eval_preflight(config_path: str | Path) -> dict[str, Path]:
    cfg = load_yaml(config_path)
    run_id = cfg.get("run_id", "eval_suite_preflight")
    run_dir = ensure_dir(cfg.get("paths", {}).get("run_dir", "data/runs/eval_suite_local/00_preflight"))
    write_yaml(cfg, run_dir / "config_snapshot.yaml")

    started = time.time()
    respondents = pd.read_parquet(cfg_path(cfg, "ces_respondents"))
    targets = pd.read_parquet(cfg_path(cfg, "ces_targets"))
    context = pd.read_parquet(cfg_path(cfg, "ces_context"))
    strict_facts = pd.read_parquet(cfg_path(cfg, "strict_memory_facts", "ces_memory_facts_strict"))
    poll_facts = pd.read_parquet(cfg_path(cfg, "poll_memory_facts", "ces_memory_facts_poll"))
    mit_truth = pd.read_parquet(cfg_path(cfg, "mit_state_truth"))
    question = load_question_config(cfg_path(cfg, "question_set")).iloc[0]

    leakage_checks = _leakage_check_rows(strict_facts=strict_facts, poll_facts=poll_facts)
    leakage_summary = pd.concat(
        [
            strict_facts.assign(policy="strict_pre_no_vote_v1"),
            poll_facts.assign(policy="poll_informed_pre_v1"),
        ],
        ignore_index=True,
    ).groupby(["policy", "fact_role", "source_variable"], dropna=False).size().reset_index(name="n_facts")
    leakage_summary.to_csv(run_dir / "leakage_audit_summary.csv", index=False)
    write_table(leakage_checks, run_dir / "leakage_checks.parquet")

    sample_cfg = cfg.get("sample", {})
    states = list(sample_cfg.get("states", cfg.get("states", ["PA", "GA"])))
    sampled, agents = sample_agents_by_state(
        respondents=respondents,
        run_id=run_id,
        states=states,
        agents_per_state=int(sample_cfg.get("agents_per_state", 1)),
        seed=int(cfg.get("seed", 20260426)),
        split=str(sample_cfg.get("split", "test")),
    )
    write_table(agents, run_dir / "agents.parquet")

    ces_ids = set(agents["base_ces_id"].dropna().astype(str))
    memory_by_id = group_memory_facts_for_ids(strict_facts, ces_ids)
    context_by_id = group_context_for_ids(context, ces_ids)
    model_cfg = ModelConfig.model_validate(cfg.get("model", {"provider": "mock"}))
    client = build_llm_client(model_cfg)
    model_name = getattr(client, "model_name", model_cfg.model_name)
    cache = AggregateLlmCache(run_dir / "llm_cache.jsonl")
    cache.path.parent.mkdir(parents=True, exist_ok=True)
    cache.path.touch(exist_ok=True)
    max_facts = int(cfg.get("memory", {}).get("max_memory_facts", 24))
    llm_cfg = cfg.get("llm", {})
    workers = max(1, int(llm_cfg.get("workers", 1)))
    gpu_snapshots = [gpu_snapshot("start")]

    tasks: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for _, agent in agents.iterrows():
        prompt_text, fact_ids = build_ces_prompt(
            agent,
            question,
            memory_facts=memory_by_id,
            context=context_by_id,
            memory_policy="strict_pre_no_vote_v1",
            max_memory_facts=max_facts,
            prompt_mode="ces_survey_memory",
        )
        prompt_hash = stable_hash(prompt_text, length=32)
        cache_key = llm_cache_key(
            model_name=model_name,
            baseline="preflight_strict_memory",
            prompt_hash=prompt_hash,
            temperature=float(model_cfg.temperature),
            max_tokens=int(model_cfg.max_tokens),
            response_format=str(model_cfg.response_format),
        )
        prompt_id = stable_hash(run_id, agent["agent_id"], prompt_hash, length=20)
        tasks.append(
            {
                "run_id": run_id,
                "prompt_id": prompt_id,
                "agent_id": agent["agent_id"],
                "base_ces_id": agent["base_ces_id"],
                "state_po": agent["state_po"],
                "sample_rank": int(agent.get("sample_rank", 0) or 0),
                "sample_weight": agent["sample_weight"],
                "baseline": "preflight_strict_memory",
                "model_name": model_name,
                "prompt_hash": prompt_hash,
                "prompt_text": prompt_text,
                "memory_fact_ids_used": fact_ids,
                "cache_key": cache_key,
            }
        )

    def finish_task(task: dict[str, Any]) -> tuple[dict[str, Any], str, bool, int | None, str | None]:
        call_started = time.time()
        try:
            raw, cache_hit, latency_ms = complete_llm_task_with_cache(
                client=client,
                cache=cache,
                cache_key=task["cache_key"],
                run_id=run_id,
                model_name=model_name,
                baseline=task["baseline"],
                prompt_hash=task["prompt_hash"],
                prompt_text=task["prompt_text"],
            )
            return task, raw, bool(cache_hit), latency_ms, None
        except Exception as exc:
            latency_ms = int((time.time() - call_started) * 1000)
            return task, "", False, latency_ms, f"{type(exc).__name__}: {exc}"

    completed: list[tuple[dict[str, Any], str, bool, int | None, str | None]] = []
    if workers == 1:
        for task in tasks:
            completed.append(finish_task(task))
            gpu_snapshots.append(gpu_snapshot(f"after_response_{len(completed)}"))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(finish_task, task) for task in tasks]
            for future in as_completed(futures):
                completed.append(future.result())
                gpu_snapshots.append(gpu_snapshot(f"after_response_{len(completed)}"))
    gpu_snapshots.append(gpu_snapshot("end"))
    completed.sort(key=lambda item: (item[0].get("sample_rank", 0), item[0].get("state_po", ""), item[0].get("base_ces_id", "")))

    for task, raw, cache_hit, latency_ms, transport_error in completed:
        parsed = parse_turnout_vote_json(raw)
        parse_status = "transport_error" if transport_error else parsed["parse_status"]
        raw_diag = raw_choice_diagnostics(raw)
        prompt_rows.append(
            {
                "run_id": run_id,
                "prompt_id": task["prompt_id"],
                "agent_id": task["agent_id"],
                "base_ces_id": task["base_ces_id"],
                "baseline": task["baseline"],
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
                "response_id": stable_hash(run_id, task["agent_id"], task["prompt_hash"], "preflight", length=20),
                "prompt_id": task["prompt_id"],
                "agent_id": task["agent_id"],
                "base_ces_id": task["base_ces_id"],
                "baseline": task["baseline"],
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
                "raw_choice": raw_diag["raw_choice"],
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
                "base_ces_id": task["base_ces_id"],
                "baseline": task["baseline"],
                "model_name": model_name,
                "cache_hit": cache_hit,
                "latency_ms": latency_ms,
                "parse_status": parse_status,
                "raw_choice": raw_diag["raw_choice"],
                "invalid_choice": raw_diag["invalid_choice"],
                "legacy_probability_schema": raw_diag["legacy_probability_schema"],
                "forbidden_choice": raw_diag["forbidden_choice"],
                "transport_error": transport_error,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    prompts = pd.DataFrame(prompt_rows)
    responses = pd.DataFrame(response_rows)
    runtime_log = pd.DataFrame(runtime_rows)
    write_table(prompts, run_dir / "smoke_prompts.parquet")
    write_table(responses, run_dir / "smoke_responses.parquet")
    write_table(runtime_log, run_dir / "runtime_log.parquet")

    quality = parse_quality_summary(responses, run_id=run_id)
    write_table(quality, run_dir / "preflight_quality_metrics.parquet")
    gates = cfg.get("quality_gates", {})
    gate_values = {row["metric_name"]: row["metric_value"] for _, row in quality.iterrows()}
    gate_rows = [
        {
            "gate_name": "parse_ok_rate",
            "threshold": float(gates.get("parse_ok_rate_min", 0.95)),
            "observed": gate_values.get("parse_ok_rate"),
            "passed": (gate_values.get("parse_ok_rate") or 0.0) >= float(gates.get("parse_ok_rate_min", 0.95)),
        },
        {
            "gate_name": "invalid_choice_rate",
            "threshold": float(gates.get("invalid_choice_rate_max", 0.02)),
            "observed": gate_values.get("invalid_choice_rate"),
            "passed": (gate_values.get("invalid_choice_rate") or 0.0)
            <= float(gates.get("invalid_choice_rate_max", 0.02)),
        },
        {
            "gate_name": "forbidden_choice_rate",
            "threshold": float(gates.get("forbidden_choice_rate_max", 0.0)),
            "observed": gate_values.get("forbidden_choice_rate"),
            "passed": (gate_values.get("forbidden_choice_rate") or 0.0)
            <= float(gates.get("forbidden_choice_rate_max", 0.0)),
        },
        {
            "gate_name": "legacy_probability_schema_rate",
            "threshold": float(gates.get("legacy_probability_schema_rate_max", 0.0)),
            "observed": gate_values.get("legacy_probability_schema_rate"),
            "passed": (gate_values.get("legacy_probability_schema_rate") or 0.0)
            <= float(gates.get("legacy_probability_schema_rate_max", 0.0)),
        },
    ]
    gate_df = pd.DataFrame(gate_rows)
    write_table(gate_df, run_dir / "quality_gates.parquet")

    preview = run_dir / "smoke_prompt_preview.md"
    preview.write_text(
        "\n".join(
            [
                f"# Preflight Prompt Preview: {run_id}",
                "",
                "## Prompt",
                "```text",
                prompts.iloc[0]["prompt_text"] if not prompts.empty else "",
                "```",
                "",
                "## Raw response",
                "```text",
                responses.iloc[0]["raw_response"] if not responses.empty else "",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    runtime_seconds = time.time() - started
    latency = pd.to_numeric(responses.get("latency_ms", pd.Series(dtype=float)), errors="coerce").dropna()
    gpu_summary = gpu_peak_summary(gpu_snapshots)
    cache_hit_rate = (
        float(responses["cache_hit"].fillna(False).astype(bool).mean())
        if "cache_hit" in responses.columns and not responses.empty
        else None
    )
    runtime = {
        "run_id": run_id,
        "git_commit": git_commit(),
        "model_name": model_name,
        "provider": model_cfg.provider,
        "temperature": model_cfg.temperature,
        "max_tokens": model_cfg.max_tokens,
        "workers": workers,
        "n_smoke_agents": int(len(agents)),
        "n_llm_tasks": int(len(tasks)),
        "runtime_seconds": runtime_seconds,
        "median_latency_seconds": float(latency.median() / 1000.0) if not latency.empty else None,
        "p90_latency_seconds": float(latency.quantile(0.9) / 1000.0) if not latency.empty else None,
        "throughput_responses_per_second": float(len(responses) / runtime_seconds) if runtime_seconds > 0 else None,
        "cache_hit_rate": cache_hit_rate,
        "ollama_calls": int((~responses["cache_hit"].fillna(False).astype(bool)).sum()) if "cache_hit" in responses.columns else 0,
        "all_gates_passed": bool(leakage_checks["passed"].all() and gate_df["passed"].all()),
        **gpu_summary,
        "gpu_snapshots": gpu_snapshots,
        "input_artifact_paths": {key: str(value) for key, value in cfg.get("paths", {}).items()},
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    write_json(runtime, run_dir / "runtime.json")
    status = "PASS" if runtime["all_gates_passed"] else "FAIL"
    lines = [
        f"# Preflight Checks: {run_id}",
        "",
        f"- Status: `{status}`",
        f"- Model: `{model_name}`",
        f"- Workers: {workers}",
        f"- Smoke responses: {len(responses)}",
        f"- Runtime seconds: {runtime_seconds:.2f}",
        f"- Median latency seconds: {runtime['median_latency_seconds']}",
        f"- P90 latency seconds: {runtime['p90_latency_seconds']}",
        f"- GPU peak memory used MiB: {runtime.get('gpu_peak_memory_used_mib')}",
        f"- GPU min memory free MiB: {runtime.get('gpu_min_memory_free_mib')}",
        f"- MIT 2024 state rows: {int((mit_truth.get('year', pd.Series(dtype=int)) == 2024).sum()) if 'year' in mit_truth.columns else len(mit_truth)}",
        "",
        "## Leakage Checks",
        _markdown_table(leakage_checks),
        "",
        "## Quality Gates",
        _markdown_table(gate_df),
        "",
        "## Parse Metrics",
        _markdown_table(quality),
        "",
    ]
    report = run_dir / "preflight_checks.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return {
        "report": report,
        "agents": run_dir / "agents.parquet",
        "prompts": run_dir / "smoke_prompts.parquet",
        "responses": run_dir / "smoke_responses.parquet",
        "quality_metrics": run_dir / "preflight_quality_metrics.parquet",
        "quality_gates": run_dir / "quality_gates.parquet",
        "leakage_checks": run_dir / "leakage_checks.parquet",
        "leakage_summary": run_dir / "leakage_audit_summary.csv",
        "runtime_log": run_dir / "runtime_log.parquet",
        "runtime": run_dir / "runtime.json",
        "config_snapshot": run_dir / "config_snapshot.yaml",
        "prompt_preview": preview,
        "llm_cache": run_dir / "llm_cache.jsonl",
    }


def _read_metric_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


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


def write_eval_suite_summary(config_path: str | Path) -> dict[str, Path]:
    cfg = load_yaml(config_path)
    root = ensure_dir(cfg.get("root_dir", cfg.get("paths", {}).get("root_dir", "data/runs/eval_suite_local")))
    ensure_dir(root / "tables")
    experiments = cfg.get(
        "experiments",
        {
            "00_preflight": "00_preflight",
            "01_individual_persona": "01_individual_persona",
            "02_aggregate_accuracy": "02_aggregate_accuracy",
            "03_prompt_robustness": "03_prompt_robustness",
            "04_leakage": "04_leakage",
            "05_ablation_placebo": "05_ablation_placebo",
            "06_subgroup_calibration": "06_subgroup_calibration",
        },
    )
    inventory_rows = []
    for name, rel_path in experiments.items():
        exp_dir = root / rel_path
        files = sorted(path.name for path in exp_dir.glob("*")) if exp_dir.exists() else []
        report = next((exp_dir / candidate for candidate in ["report.md", "benchmark_report.md", "preflight_checks.md"] if (exp_dir / candidate).exists()), None)
        inventory_rows.append(
            {
                "experiment": name,
                "path": str(exp_dir),
                "exists": exp_dir.exists(),
                "report": str(report) if report else None,
                "n_files": len(files),
                "files": ",".join(files[:30]),
            }
        )
    inventory = pd.DataFrame(inventory_rows)
    inventory_csv = root / "tables" / "eval_suite_inventory.csv"
    inventory.to_csv(inventory_csv, index=False)

    def metric_lines(exp_dir: Path, path_name: str, wanted: list[str]) -> list[str]:
        table = _read_metric_table(exp_dir / path_name)
        if table.empty or "metric_name" not in table.columns:
            return ["_No metrics available._"]
        subset = table[table["metric_name"].isin(wanted)].copy()
        if subset.empty:
            return ["_No requested metrics available._"]
        cols = [
            col
            for col in [
                "sample_size",
                "baseline",
                "model_name",
                "prompt_variant",
                "condition",
                "contrast_name",
                "comparison_condition",
                "delta_name",
                "baseline_from",
                "baseline_to",
                "state_po",
                "displayed_state_po",
                "metric_name",
                "metric_value",
                "from_value",
                "to_value",
                "metric_delta",
                "named_value",
                "comparison_value",
                "named_improvement",
                "pred_shift",
                "truth_shift",
                "party_following_score",
                "name_following_score",
                "candidate_name_following_index",
                "n",
                "n_states",
            ]
            if col in subset.columns
        ]
        return [_markdown_table(subset[cols].head(80))]

    def gate_lines(exp_dir: Path) -> list[str]:
        table = _read_metric_table(exp_dir / "quality_gates.parquet")
        if table.empty:
            return ["_No quality gates available._"]
        cols = [col for col in ["gate_name", "threshold", "observed", "passed"] if col in table.columns]
        return [_markdown_table(table[cols])]

    def runtime_lines(exp_dir: Path) -> list[str]:
        path = exp_dir / "runtime.json"
        if not path.exists():
            return ["_No runtime metadata available._"]
        try:
            runtime = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return ["_Could not read runtime metadata._"]
        wanted = [
            "model_name",
            "workers",
            "n_llm_tasks",
            "llm_calls_made",
            "n_sources",
            "n_responses",
            "n_subgroup_rows",
            "runtime_seconds",
            "median_latency_seconds",
            "p90_latency_seconds",
            "throughput_responses_per_second",
            "gpu_peak_memory_used_mib",
            "gpu_min_memory_free_mib",
            "all_gates_passed",
        ]
        rows = [{"field": key, "value": runtime.get(key)} for key in wanted if key in runtime]
        return [_markdown_table(pd.DataFrame(rows))]

    def table_lines(exp_dir: Path, path_name: str, wanted_cols: list[str], head: int = 40) -> list[str]:
        table = _read_metric_table(exp_dir / path_name)
        if table.empty:
            return ["_No table available._"]
        cols = [col for col in wanted_cols if col in table.columns]
        if not cols:
            return ["_No requested columns available._"]
        return [_markdown_table(table[cols].head(head))]

    preflight_dir = root / experiments.get("00_preflight", "00_preflight")
    lines = [
        "# Evaluation Suite Summary",
        "",
        "## 1. Run Metadata",
        f"- Git commit: `{git_commit()}`",
        f"- Root directory: `{root}`",
        f"- Config: `{config_path}`",
        "",
        "## 2. Data Contract and Leakage Audit",
        *(metric_lines(preflight_dir, "preflight_quality_metrics.parquet", ["parse_ok_rate", "invalid_choice_rate", "forbidden_choice_rate", "legacy_probability_schema_rate"])),
        "",
        "### Preflight Quality Gates",
        *(gate_lines(preflight_dir)),
        "",
        "### Preflight Runtime",
        *(runtime_lines(preflight_dir)),
        "",
        "## Experiment Inventory",
        _markdown_table(inventory),
        "",
        "## 3. Individual Persona Fidelity",
        *(metric_lines(root / experiments.get("01_individual_persona", "01_individual_persona"), "individual_metrics.parquet", ["vote_accuracy", "vote_macro_f1", "turnout_accuracy_at_0_5"])),
        "",
        "### E01 Runtime",
        *(runtime_lines(root / experiments.get("01_individual_persona", "01_individual_persona"))),
        "",
        "## 4. Aggregate Election Accuracy",
        *(metric_lines(root / experiments.get("02_aggregate_accuracy", "02_aggregate_accuracy"), "aggregate_metrics.parquet", ["dem_2p_rmse", "margin_mae", "margin_bias", "winner_accuracy", "winner_flip_count"])),
        "",
        "### E02 Runtime",
        *(runtime_lines(root / experiments.get("02_aggregate_accuracy", "02_aggregate_accuracy"))),
        "",
        "## 5. Prompt Robustness",
        *(metric_lines(root / experiments.get("03_prompt_robustness", "03_prompt_robustness"), "robustness_metrics.parquet", ["parse_ok_rate", "choice_flip_rate", "vote_accuracy", "turnout_brier"])),
        "",
        "### E03 Runtime",
        *(runtime_lines(root / experiments.get("03_prompt_robustness", "03_prompt_robustness"))),
        "",
        "### Prompt Robustness Pairwise Shifts",
        *(metric_lines(root / experiments.get("03_prompt_robustness", "03_prompt_robustness"), "pairwise_variant_metrics.parquet", ["choice_flip_rate", "turnout_choice_flip_rate", "state_margin_shift"])),
        "",
        "## 6. Leakage Diagnostics",
        *(metric_lines(root / experiments.get("04_leakage", "04_leakage"), "leakage_contrasts.parquet", ["margin_mae", "dem_2p_rmse", "pred_shift_vs_truth_shift", "mean_name_minus_party_following"])),
        "",
        "### E04 Runtime",
        *(runtime_lines(root / experiments.get("04_leakage", "04_leakage"))),
        "",
        "### E04 State Swap",
        *(table_lines(root / experiments.get("04_leakage", "04_leakage"), "state_swap_diagnostics.parquet", ["state_po", "displayed_state_po", "pred_shift", "truth_shift"], head=20)),
        "",
        "### E04 Candidate Swap",
        *(table_lines(root / experiments.get("04_leakage", "04_leakage"), "candidate_swap_diagnostics.parquet", ["state_po", "party_following_score", "name_following_score", "candidate_name_following_index"], head=20)),
        "",
        "## 7. Ablation and Placebo Memory",
        *(metric_lines(root / experiments.get("05_ablation_placebo", "05_ablation_placebo"), "ablation_deltas.parquet", ["vote_accuracy", "vote_macro_f1", "turnout_accuracy_at_0_5"])),
        "",
        "### E05 Runtime",
        *(runtime_lines(root / experiments.get("05_ablation_placebo", "05_ablation_placebo"))),
        "",
        "## 8. Subgroup and Calibration Reliability",
        *(table_lines(
            root / experiments.get("06_subgroup_calibration", "06_subgroup_calibration"),
            "worst_subgroups.parquet",
            [
                "failure_type",
                "failure_score",
                "source",
                "baseline",
                "dimension",
                "subgroup_value",
                "n",
                "small_n",
                "vote_accuracy",
                "turnout_brier",
                "dem_2p_error",
                "entropy_ratio",
                "variance_ratio",
            ],
            head=40,
        )),
        "",
        "### E06 Calibration Summary",
        *(table_lines(
            root / experiments.get("06_subgroup_calibration", "06_subgroup_calibration"),
            "calibration_bins.parquet",
            [
                "source",
                "baseline",
                "calibration_type",
                "bin",
                "n",
                "mean_predicted",
                "observed_rate",
                "absolute_error",
                "expected_calibration_error",
            ],
            head=40,
        )),
        "",
        "### E06 Runtime",
        *(runtime_lines(root / experiments.get("06_subgroup_calibration", "06_subgroup_calibration"))),
        "",
        "## 9. Overall Verdict",
        "_Fill this section after the full local suite has completed._",
        "",
    ]
    out = root / "eval_suite_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return {"summary": out, "inventory": inventory_csv}
