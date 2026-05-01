"""CES world-knowledge leakage / post-2024 knowledge stress benchmark."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .ces_aggregate_benchmark import (
    AggregateBenchmarkPaths,
    AggregateLlmCache,
    _save_figure,
    _truth_year,
    complete_llm_task_with_cache,
    llm_cache_key,
    swing_aggregate_metric_rows,
)
from .ces_benchmark import (
    VOTE_CLASSES,
    _agents_from_cohort,
    _combined_probabilities,
    _group_memory_facts,
    _response_rows,
    _target_wide,
    benchmark_metric_rows,
    build_benchmark_cohort,
    expected_calibration_error,
    turnout_calibration_bins,
)
from .ces_schema import parse_turnout_vote_json
from .config import ModelConfig
from .eval_suite import git_commit, gpu_peak_summary, gpu_snapshot, raw_choice_diagnostics
from .io import ensure_dir, load_yaml, stable_json, write_json, write_table, write_yaml
from .llm import build_llm_client
from .transforms import stable_hash


DEFAULT_STATES = ["PA", "GA", "AZ", "MN", "VA", "CO"]
LEAKAGE_CONDITIONS = [
    "named_candidates",
    "party_only_candidates",
    "anonymous_candidates",
    "masked_year",
    "masked_state",
    "state_swap_placebo",
    "candidate_swap_placebo",
]
CONDITION_ORDER = {name: idx for idx, name in enumerate(LEAKAGE_CONDITIONS)}
FICTITIOUS_STATE_CODES = {
    "PA": "F01",
    "GA": "F02",
    "AZ": "F03",
    "MN": "F04",
    "VA": "F05",
    "CO": "F06",
    "MI": "F07",
    "WI": "F08",
    "NV": "F09",
    "NC": "F10",
}
STATE_SWAP_MAP = {
    "PA": "MN",
    "MN": "PA",
    "GA": "VA",
    "VA": "GA",
    "AZ": "CO",
    "CO": "AZ",
    "MI": "NC",
    "NC": "MI",
    "WI": "NV",
    "NV": "WI",
}


@dataclass
class LeakageTask:
    condition: str
    agent_id: str
    base_ces_id: str
    original_state_po: str
    displayed_state_po: str
    prompt_id: str
    prompt_hash: str
    prompt_text: str
    fact_ids: list[str]
    cache_key: str
    year_label: str
    candidate_mode: str
    democratic_display: str
    republican_display: str


def choose_effective_agents_per_state(
    *,
    requested_agents_per_state: int,
    n_states: int,
    n_conditions: int,
    observed_throughput_per_second: float | None,
    max_runtime_minutes: float,
) -> tuple[int, str | None, float | None]:
    requested_calls = requested_agents_per_state * n_states * n_conditions
    projected = (
        requested_calls / observed_throughput_per_second / 60.0
        if observed_throughput_per_second and observed_throughput_per_second > 0
        else None
    )
    if projected is None or projected <= max_runtime_minutes:
        return requested_agents_per_state, None, projected
    if 6 < requested_agents_per_state <= 10:
        candidate = 6
        reason = "runtime_reduced_to_6"
    else:
        candidate = min(30, requested_agents_per_state)
        reason = "runtime_reduced_to_30"
    if candidate >= requested_agents_per_state:
        return requested_agents_per_state, None, projected
    candidate_projected = candidate * n_states * n_conditions / observed_throughput_per_second / 60.0
    return candidate, reason, candidate_projected


def _weighted_sample(group: pd.DataFrame, *, n: int, seed_key: str, weight_col: str = "sample_weight") -> pd.DataFrame:
    if n <= 0 or group.empty:
        return group.head(0).copy()
    n = min(n, len(group))
    weights = group[weight_col].fillna(0).astype(float) if weight_col in group.columns else None
    rng = np.random.default_rng(int(stable_hash(seed_key, length=8), 16))
    if weights is not None and float(weights.sum()) > 0:
        prob = weights.to_numpy(dtype=float)
        prob = prob / prob.sum()
        try:
            return group.iloc[rng.choice(len(group), size=n, replace=False, p=prob)].copy()
        except ValueError:
            return group.iloc[rng.choice(len(group), size=n, replace=False)].copy()
    return group.iloc[rng.choice(len(group), size=n, replace=False)].copy()


def sample_leakage_agents(
    cohort: pd.DataFrame,
    *,
    states: list[str],
    agents_per_state: int,
    seed: int,
) -> pd.DataFrame:
    test = cohort[(cohort["split"] == "test") & (cohort["state_po"].isin(states))].copy()
    parts = []
    for state in states:
        group = test[test["state_po"] == state].copy()
        sampled = _weighted_sample(group, n=agents_per_state, seed_key=f"leakage-{seed}-{state}")
        sampled["sample_rank"] = np.arange(1, len(sampled) + 1)
        parts.append(sampled)
    if not parts:
        return test.head(0).copy()
    return pd.concat(parts, ignore_index=True)


def displayed_state_for_condition(original_state: str, condition: str) -> str:
    if condition == "masked_state":
        return FICTITIOUS_STATE_CODES.get(original_state, f"F{int(stable_hash(original_state, length=4), 16) % 90 + 10}")
    if condition == "state_swap_placebo":
        return STATE_SWAP_MAP.get(original_state, original_state)
    return original_state


def _select_strict_facts(memory_by_id: Mapping[str, pd.DataFrame], ces_id: str, max_facts: int) -> tuple[list[str], list[str]]:
    facts = memory_by_id.get(str(ces_id))
    if facts is None or facts.empty:
        return [], []
    selected = facts.copy()
    if "fact_role" in selected.columns:
        selected = selected[selected["fact_role"].fillna("safe_pre").astype(str) == "safe_pre"]
    if "fact_priority" in selected.columns:
        selected = selected.sort_values(["fact_priority", "source_variable"], ascending=[False, True])
    selected = selected.head(max_facts)
    if selected.empty:
        return [], []
    return selected["fact_text"].astype(str).tolist(), selected["memory_fact_id"].astype(str).tolist()


def scrub_memory_text(text: str, condition: str) -> str:
    out = str(text)
    if condition not in {"party_only_candidates", "anonymous_candidates", "masked_year"}:
        return out
    if condition == "anonymous_candidates":
        replacements = [
            (r"\bKamala Harris\b", "Candidate A"),
            (r"\bHarris\b", "Candidate A"),
            (r"\bDonald Trump\b", "Candidate B"),
            (r"\bTrump\b", "Candidate B"),
        ]
    else:
        replacements = [
            (r"\bKamala Harris\b", "Democratic nominee"),
            (r"\bHarris\b", "Democratic nominee"),
            (r"\bDonald Trump\b", "Republican nominee"),
            (r"\bTrump\b", "Republican nominee"),
        ]
    replacements.extend(
        [
            (r"\bPresident Biden\b", "incumbent president"),
            (r"\bJoe Biden\b", "incumbent president"),
            (r"\bBiden\b", "incumbent president"),
            (r"\b2024\b", "recent"),
        ]
    )
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


def candidate_lines_for_condition(condition: str) -> tuple[list[str], str, str, str]:
    if condition in {"named_candidates", "masked_state", "state_swap_placebo"}:
        return (
            ["- Democratic candidate: Kamala Harris", "- Republican candidate: Donald Trump"],
            "named_candidates",
            "Kamala Harris",
            "Donald Trump",
        )
    if condition in {"party_only_candidates", "masked_year"}:
        return (
            ["- Democratic nominee", "- Republican nominee"],
            "party_only",
            "Democratic nominee",
            "Republican nominee",
        )
    if condition == "anonymous_candidates":
        return (
            [
                "- Candidate A policy summary: supports abortion access, climate action, expanded health insurance, middle-class tax credits, and voting rights.",
                "- Candidate B policy summary: supports stricter border enforcement, lower taxes, deregulation, expanded fossil-fuel production, tariffs, and conservative judges.",
            ],
            "anonymous",
            "Candidate A",
            "Candidate B",
        )
    if condition == "candidate_swap_placebo":
        return (
            ["- Democratic candidate: Donald Trump", "- Republican candidate: Kamala Harris"],
            "candidate_swap",
            "Donald Trump",
            "Kamala Harris",
        )
    raise ValueError(f"Unknown leakage condition: {condition}")


def render_leakage_prompt(
    *,
    condition: str,
    agent: pd.Series,
    strict_memory: Mapping[str, pd.DataFrame],
    max_memory_facts: int,
) -> tuple[str, list[str], dict[str, Any]]:
    if condition not in LEAKAGE_CONDITIONS:
        raise ValueError(f"Unknown leakage condition: {condition}")
    original_state = str(agent["state_po"])
    displayed_state = displayed_state_for_condition(original_state, condition)
    year_label = "a recent presidential election" if condition == "masked_year" else "the 2024 general election"
    facts, fact_ids = _select_strict_facts(strict_memory, str(agent["base_ces_id"]), max_memory_facts)
    facts = [scrub_memory_text(fact, condition) for fact in facts]
    candidate_lines, candidate_mode, dem_display, rep_display = candidate_lines_for_condition(condition)
    schema = (
        [
            '{"choice": "not_vote|candidate_a|candidate_b"}',
        ]
        if condition == "anonymous_candidates"
        else [
            '{"choice": "not_vote|democrat|republican"}',
        ]
    )
    lines = [
        f"You are simulating how a specific U.S. eligible voter would behave in {year_label}.",
        "Answer as this voter would behave, not as a political analyst.",
        "",
        "Voter profile:",
        f"- State: {displayed_state}",
        f"- Age group: {agent.get('age_group') or 'unknown'}",
        f"- Gender: {agent.get('gender') or 'unknown'}",
        f"- Race/ethnicity: {agent.get('race_ethnicity') or 'unknown'}",
        f"- Education: {agent.get('education_binary') or 'unknown'}",
        f"- Party identification: {agent.get('party_id_3') or 'unknown'}",
        f"- 7-point party ID: {agent.get('party_id_7') or 'unknown'}",
        f"- Ideology: {agent.get('ideology_3') or 'unknown'}",
    ]
    if facts:
        lines.extend(["", "Strict pre-election survey-derived background facts:"])
        lines.extend(f"- {fact}" for fact in facts)
    lines.extend(["", "Election context:", "- Office: President", *candidate_lines])
    lines.extend(
        [
            "",
            "Task:",
            "Choose the single election behavior this voter would most likely take.",
            "",
            "Important output rules:",
            "- Return exactly one choice.",
            "- Do not include probabilities, confidence, markdown, or explanatory text.",
            "",
            "Return JSON only with this schema:",
            *schema,
        ]
    )
    text = "\n".join(lines)
    return text, fact_ids, {
        "original_state_po": original_state,
        "displayed_state_po": displayed_state,
        "year_label": year_label,
        "candidate_mode": candidate_mode,
        "democratic_display": dem_display,
        "republican_display": rep_display,
    }


def normalize_leakage_response(raw_response: str, condition: str) -> str:
    if condition != "anonymous_candidates":
        return raw_response
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
        if not match:
            return raw_response
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return raw_response
    choice_map = {"candidate_a": "democrat", "candidate_b": "republican"}
    if payload.get("choice") not in choice_map:
        return raw_response
    payload["choice"] = choice_map[payload["choice"]]
    return json.dumps(payload)


def build_leakage_tasks(
    *,
    cfg: dict[str, Any],
    agents: pd.DataFrame,
    strict_memory_facts: pd.DataFrame,
    model_name: str,
    model_cfg: ModelConfig,
) -> list[LeakageTask]:
    strict_memory = _group_memory_facts(strict_memory_facts)
    conditions = [name for name in cfg.get("conditions", LEAKAGE_CONDITIONS) if name in LEAKAGE_CONDITIONS]
    max_memory_facts = int(cfg.get("memory", {}).get("max_memory_facts", 24))
    tasks: list[LeakageTask] = []
    for _, agent in agents.sort_values(["state_po", "sample_rank", "base_ces_id"]).iterrows():
        for condition in conditions:
            prompt_text, fact_ids, meta = render_leakage_prompt(
                condition=condition,
                agent=agent,
                strict_memory=strict_memory,
                max_memory_facts=max_memory_facts,
            )
            prompt_hash = stable_hash(prompt_text, length=32)
            cache_key = llm_cache_key(
                model_name=model_name,
                baseline=condition,
                prompt_hash=prompt_hash,
                temperature=float(model_cfg.temperature),
                max_tokens=int(model_cfg.max_tokens),
                response_format=str(model_cfg.response_format),
            )
            tasks.append(
                LeakageTask(
                    condition=condition,
                    agent_id=str(agent["agent_id"]),
                    base_ces_id=str(agent["base_ces_id"]),
                    original_state_po=meta["original_state_po"],
                    displayed_state_po=meta["displayed_state_po"],
                    prompt_id=stable_hash(cfg["run_id"], condition, agent["base_ces_id"], prompt_hash, length=20),
                    prompt_hash=prompt_hash,
                    prompt_text=prompt_text,
                    fact_ids=fact_ids,
                    cache_key=cache_key,
                    year_label=meta["year_label"],
                    candidate_mode=meta["candidate_mode"],
                    democratic_display=meta["democratic_display"],
                    republican_display=meta["republican_display"],
                )
            )
    return tasks


def _finish_tasks(
    *,
    client: Any,
    cache: AggregateLlmCache,
    tasks: list[LeakageTask],
    cfg: dict[str, Any],
    model_name: str,
    workers: int,
    existing_results: dict[str, dict[str, Any]] | None = None,
    on_result: Callable[[tuple[LeakageTask, str, bool, int | None, str | None]], None] | None = None,
) -> list[tuple[LeakageTask, str, bool, int | None, str | None]]:
    existing_results = existing_results or {}

    def finish(task: LeakageTask) -> tuple[LeakageTask, str, bool, int | None, str | None]:
        started_call = time.time()
        if task.cache_key in existing_results:
            row = existing_results[task.cache_key]
            return task, str(row["raw"]), bool(row["cache_hit"]), row.get("latency_ms"), row.get("transport_error")
        try:
            raw, cache_hit, latency_ms = complete_llm_task_with_cache(
                client=client,
                cache=cache,
                cache_key=task.cache_key,
                run_id=cfg["run_id"],
                model_name=model_name,
                baseline=task.condition,
                prompt_hash=task.prompt_hash,
                prompt_text=task.prompt_text,
            )
            return task, raw, cache_hit, latency_ms, None
        except Exception as exc:
            latency_ms = int((time.time() - started_call) * 1000)
            return task, "", False, latency_ms, f"{type(exc).__name__}: {exc}"

    if workers <= 1:
        completed = []
        for task in tasks:
            result = finish(task)
            completed.append(result)
            if on_result is not None:
                on_result(result)
    else:
        completed = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task = {executor.submit(finish, task): task for task in tasks}
            for future in as_completed(future_to_task):
                result = future.result()
                completed.append(result)
                if on_result is not None:
                    on_result(result)
    completed.sort(key=lambda row: (row[0].original_state_po, row[0].base_ces_id, CONDITION_ORDER[row[0].condition]))
    return completed


def run_llm_leakage(
    *,
    cfg: dict[str, Any],
    paths: AggregateBenchmarkPaths,
    sampled: pd.DataFrame,
    agents: pd.DataFrame,
    strict_memory_facts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    model_cfg = ModelConfig.model_validate(cfg.get("model", {}))
    client = build_llm_client(model_cfg)
    model_name = getattr(client, "model_name", model_cfg.model_name)
    cache = AggregateLlmCache(paths.cache_path)
    workers = max(1, int(cfg.get("llm", {}).get("workers", 8)))
    timing_responses = int(cfg.get("llm", {}).get("timing_responses", 40))
    max_runtime_minutes = float(cfg.get("llm", {}).get("max_runtime_minutes", 45.0))
    checkpoint_every = max(1, int(cfg.get("llm", {}).get("checkpoint_every", 25)))
    gpu_sample_every = max(1, int(cfg.get("llm", {}).get("gpu_sample_every", checkpoint_every)))
    requested_agents = int(cfg.get("agents_per_state", 40))
    states = list(cfg.get("states", DEFAULT_STATES))
    conditions = [name for name in cfg.get("conditions", LEAKAGE_CONDITIONS) if name in LEAKAGE_CONDITIONS]
    gpu_snapshots = [gpu_snapshot("run_start")]

    tasks = build_leakage_tasks(
        cfg=cfg,
        agents=agents,
        strict_memory_facts=strict_memory_facts,
        model_name=model_name,
        model_cfg=model_cfg,
    )
    timing_tasks = [task for task in tasks if cache.get(task.cache_key) is None][:timing_responses]
    timing_results: dict[str, dict[str, Any]] = {}
    timing_latencies: list[float] = []
    timing_wall = 0.0
    if timing_tasks:
        started = time.time()
        for task, raw, cache_hit, latency_ms, transport_error in _finish_tasks(
            client=client,
            cache=cache,
            tasks=timing_tasks,
            cfg=cfg,
            model_name=model_name,
            workers=workers,
        ):
            timing_results[task.cache_key] = {
                "raw": raw,
                "cache_hit": cache_hit,
                "latency_ms": latency_ms,
                "transport_error": transport_error,
            }
            if latency_ms is not None:
                timing_latencies.append(float(latency_ms) / 1000.0)
        timing_wall = time.time() - started
    throughput = len(timing_tasks) / timing_wall if timing_tasks and timing_wall > 0 else None
    effective_agents, limit_reason, projected_minutes = choose_effective_agents_per_state(
        requested_agents_per_state=requested_agents,
        n_states=len(states),
        n_conditions=len(conditions),
        observed_throughput_per_second=throughput,
        max_runtime_minutes=max_runtime_minutes,
    )
    if effective_agents < requested_agents:
        keep = set(agents[agents["sample_rank"] <= effective_agents]["base_ces_id"].astype(str))
        tasks = [task for task in tasks if task.base_ces_id in keep]

    prompt_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    cache_hits = 0
    ollama_calls = 0

    def response_frame_from_predictions(rows: list[dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        pred_df = pd.DataFrame(rows)
        response_rows: list[dict[str, Any]] = []
        for condition, group in pred_df.groupby("baseline", sort=False):
            prompt_by_id = {
                str(row["ces_id"]): {
                    "prompt_id": row["prompt_id"],
                    "cache_hit": row["cache_hit"],
                    "latency_ms": row["latency_ms"],
                }
                for _, row in group.iterrows()
            }
            response_rows.extend(
                _response_rows(
                    run_id=cfg["run_id"],
                    baseline=condition,
                    predictions=group[["ces_id", "raw_response", "model_name"]],
                    cohort=sampled,
                    prediction_scope="leakage_stress",
                    is_llm=True,
                    prompt_by_id=prompt_by_id,
                )
            )
        out = pd.DataFrame(response_rows)
        if out.empty:
            return out
        diagnostics = pred_df[
            [
                "prompt_id",
                "raw_response_original",
                "raw_choice",
                "invalid_choice",
                "forbidden_choice",
                "legacy_probability_schema",
                "transport_error",
            ]
        ].copy()
        out = out.merge(diagnostics, on="prompt_id", how="left")
        out.loc[out["transport_error"].notna(), "parse_status"] = "transport_error"
        out = out.sort_values(["baseline", "base_ces_id"]).reset_index(drop=True)
        return out

    def write_checkpoint() -> None:
        if prompt_rows:
            write_table(pd.DataFrame(prompt_rows), paths.run_dir / "prompts.partial.parquet")
        if condition_rows:
            write_table(pd.DataFrame(condition_rows), paths.run_dir / "condition_metadata.partial.parquet")
        if pred_rows:
            write_table(response_frame_from_predictions(pred_rows), paths.run_dir / "responses.partial.parquet")
        if runtime_rows:
            write_table(pd.DataFrame(runtime_rows), paths.run_dir / "runtime_log.partial.parquet")

    def record_result(result: tuple[LeakageTask, str, bool, int | None, str | None]) -> None:
        nonlocal cache_hits, ollama_calls
        task, raw, cache_hit, latency_ms, transport_error = result
        cache_hits += int(cache_hit)
        ollama_calls += int(not cache_hit)
        normalized = normalize_leakage_response(raw, task.condition)
        parsed = parse_turnout_vote_json(normalized)
        parse_status = "transport_error" if transport_error else parsed["parse_status"]
        raw_diag = raw_choice_diagnostics(normalized)
        invalid_choice = bool(raw_diag["invalid_choice"] or parse_status == "invalid_choice")
        legacy_schema = bool(raw_diag["legacy_probability_schema"] or parsed.get("legacy_probability_schema"))
        prompt_rows.append(
            {
                "run_id": cfg["run_id"],
                "prompt_id": task.prompt_id,
                "agent_id": task.agent_id,
                "base_ces_id": task.base_ces_id,
                "condition": task.condition,
                "baseline": task.condition,
                "model_name": model_name,
                "prompt_hash": task.prompt_hash,
                "prompt_text": task.prompt_text,
                "memory_fact_ids_used": task.fact_ids,
                "cache_hit": cache_hit,
                "latency_ms": latency_ms,
                "transport_error": transport_error,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
        condition_rows.append(
            {
                "run_id": cfg["run_id"],
                "prompt_id": task.prompt_id,
                "agent_id": task.agent_id,
                "base_ces_id": task.base_ces_id,
                "condition": task.condition,
                "original_state_po": task.original_state_po,
                "displayed_state_po": task.displayed_state_po,
                "year_label": task.year_label,
                "candidate_mode": task.candidate_mode,
                "democratic_display": task.democratic_display,
                "republican_display": task.republican_display,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
        pred_rows.append(
            {
                "ces_id": task.base_ces_id,
                "baseline": task.condition,
                "raw_response": normalized,
                "raw_response_original": raw,
                "model_name": model_name,
                "prompt_id": task.prompt_id,
                "cache_hit": cache_hit,
                "latency_ms": latency_ms,
                "raw_choice": raw_diag["raw_choice"],
                "invalid_choice": invalid_choice,
                "forbidden_choice": bool(raw_diag["forbidden_choice"]),
                "legacy_probability_schema": legacy_schema,
                "transport_error": transport_error,
            }
        )
        runtime_rows.append(
            {
                "run_id": cfg["run_id"],
                "event": "llm_response",
                "condition": task.condition,
                "prompt_id": task.prompt_id,
                "agent_id": task.agent_id,
                "base_ces_id": task.base_ces_id,
                "original_state_po": task.original_state_po,
                "displayed_state_po": task.displayed_state_po,
                "cache_hit": cache_hit,
                "latency_ms": latency_ms,
                "parse_status": parse_status,
                "raw_choice": raw_diag["raw_choice"],
                "invalid_choice": invalid_choice,
                "forbidden_choice": bool(raw_diag["forbidden_choice"]),
                "legacy_probability_schema": legacy_schema,
                "transport_error": transport_error,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
        completed_count = len(runtime_rows)
        if completed_count % gpu_sample_every == 0:
            gpu_snapshots.append(gpu_snapshot(f"after_response_{completed_count}"))
        if completed_count % checkpoint_every == 0:
            write_checkpoint()

    started = time.time()
    _finish_tasks(
        client=client,
        cache=cache,
        tasks=tasks,
        cfg=cfg,
        model_name=model_name,
        workers=workers,
        existing_results=timing_results,
        on_result=record_result,
    )
    llm_wall = time.time() - started
    gpu_snapshots.append(gpu_snapshot("run_end"))
    write_checkpoint()
    prompt_rows.sort(key=lambda row: (str(row["base_ces_id"]), CONDITION_ORDER.get(row["condition"], 999)))
    condition_rows.sort(key=lambda row: (str(row["base_ces_id"]), CONDITION_ORDER.get(row["condition"], 999)))
    pred_rows.sort(key=lambda row: (str(row["ces_id"]), CONDITION_ORDER.get(row["baseline"], 999)))
    runtime_rows.sort(key=lambda row: (str(row["base_ces_id"]), CONDITION_ORDER.get(row["condition"], 999)))
    responses = response_frame_from_predictions(pred_rows)
    runtime_frame = pd.DataFrame(runtime_rows)
    latency = pd.to_numeric(runtime_frame.get("latency_ms", pd.Series(dtype=float)), errors="coerce").dropna()
    llm_runtime_seconds = timing_wall + llm_wall
    parse_status = runtime_frame.get("parse_status", pd.Series(dtype=str)).fillna("").astype(str) if not runtime_frame.empty else pd.Series(dtype=str)
    invalid_choice = runtime_frame.get("invalid_choice", pd.Series(dtype=bool)).fillna(False).astype(bool) if not runtime_frame.empty else pd.Series(dtype=bool)
    forbidden_choice = runtime_frame.get("forbidden_choice", pd.Series(dtype=bool)).fillna(False).astype(bool) if not runtime_frame.empty else pd.Series(dtype=bool)
    legacy_schema = (
        runtime_frame.get("legacy_probability_schema", pd.Series(dtype=bool)).fillna(False).astype(bool)
        if not runtime_frame.empty
        else pd.Series(dtype=bool)
    )
    transport_errors = runtime_frame.get("transport_error", pd.Series(dtype=object)).notna() if not runtime_frame.empty else pd.Series(dtype=bool)
    metadata = {
        "llm_enabled": True,
        "model_name": model_name,
        "provider": model_cfg.provider,
        "temperature": model_cfg.temperature,
        "max_tokens": model_cfg.max_tokens,
        "workers": workers,
        "requested_agents_per_state": requested_agents,
        "effective_agents_per_state": effective_agents,
        "limit_reason": limit_reason,
        "timing_responses": len(timing_tasks),
        "timing_wall_seconds": timing_wall,
        "timing_throughput_per_second": throughput,
        "projected_runtime_minutes": projected_minutes,
        "max_runtime_minutes": max_runtime_minutes,
        "n_selected_tasks": len(tasks),
        "llm_wall_seconds_after_timing": llm_wall,
        "llm_runtime_seconds": llm_runtime_seconds,
        "median_latency_seconds": float(latency.median() / 1000.0) if not latency.empty else None,
        "p90_latency_seconds": float(latency.quantile(0.9) / 1000.0) if not latency.empty else None,
        "throughput_responses_per_second": float(len(runtime_rows) / llm_runtime_seconds) if llm_runtime_seconds > 0 else None,
        "cache_hits": cache_hits,
        "ollama_calls": ollama_calls,
        "cache_hit_rate": cache_hits / len(tasks) if tasks else None,
        "parse_ok_rate": float((parse_status == "ok").mean()) if len(parse_status) else None,
        "invalid_choice_rate": float(invalid_choice.mean()) if len(invalid_choice) else None,
        "forbidden_choice_rate": float(forbidden_choice.mean()) if len(forbidden_choice) else None,
        "legacy_probability_schema_rate": float(legacy_schema.mean()) if len(legacy_schema) else None,
        "transport_error_rate": float(transport_errors.mean()) if len(transport_errors) else None,
        **gpu_peak_summary(gpu_snapshots),
        "gpu_snapshots": gpu_snapshots,
    }
    metadata["all_gates_passed"] = bool(
        (metadata["parse_ok_rate"] is None or metadata["parse_ok_rate"] >= 0.95)
        and (metadata["invalid_choice_rate"] is None or metadata["invalid_choice_rate"] <= 0.02)
        and (metadata["forbidden_choice_rate"] is None or metadata["forbidden_choice_rate"] == 0.0)
        and (metadata["legacy_probability_schema_rate"] is None or metadata["legacy_probability_schema_rate"] == 0.0)
        and (metadata["transport_error_rate"] is None or metadata["transport_error_rate"] == 0.0)
    )
    return (
        responses,
        pd.DataFrame(prompt_rows),
        pd.DataFrame(condition_rows),
        runtime_frame,
        metadata,
    )


def add_condition_metadata(responses: pd.DataFrame, condition_metadata: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    meta = condition_metadata[
        [
            "prompt_id",
            "condition",
            "original_state_po",
            "displayed_state_po",
            "year_label",
            "candidate_mode",
            "democratic_display",
            "republican_display",
        ]
    ].copy()
    out = responses.merge(meta, on="prompt_id", how="left", suffixes=("", "_condition"))
    agent_meta = agents[["base_ces_id", "sample_rank"]].copy()
    out = out.merge(agent_meta, on="base_ces_id", how="left")
    out["condition"] = out["condition"].fillna(out["baseline"])
    if "state_po" in out.columns:
        out["state_po"] = out["original_state_po"].fillna(out["state_po"])
    else:
        out["state_po"] = out["original_state_po"]
    return out


def state_prediction_rows(
    *,
    responses: pd.DataFrame,
    mit_truth: pd.DataFrame,
    run_id: str,
    states: list[str],
    sample_size: int,
) -> pd.DataFrame:
    truth = _truth_year(mit_truth, 2024, states).set_index("state_po")
    rows: list[dict[str, Any]] = []
    for (condition, model_name, state), group in responses.groupby(["condition", "model_name", "original_state_po"], dropna=False):
        if state not in truth.index:
            continue
        weights = group["sample_weight"].fillna(1.0).astype(float)
        turnout = group["turnout_probability"].fillna(0.0).astype(float).clip(0.0, 1.0)
        dem = float((weights * turnout * group["vote_prob_democrat"].fillna(0.0).astype(float)).sum())
        rep = float((weights * turnout * group["vote_prob_republican"].fillna(0.0).astype(float)).sum())
        pred_dem_2p = dem / (dem + rep) if dem + rep else 0.5
        pred_margin = 2.0 * pred_dem_2p - 1.0
        truth_row = truth.loc[state]
        true_margin = float(truth_row["margin_2p"])
        pred_winner = "democrat" if pred_margin > 0 else "republican" if pred_margin < 0 else "tie"
        true_winner = str(truth_row["winner"])
        rows.append(
            {
                "run_id": run_id,
                "sample_size": int(sample_size),
                "state_po": state,
                "condition": condition,
                "baseline": condition,
                "model_name": model_name,
                "displayed_state_po": str(group["displayed_state_po"].iloc[0]),
                "pred_dem_2p": pred_dem_2p,
                "true_dem_2p": float(truth_row["dem_share_2p"]),
                "dem_2p_error": pred_dem_2p - float(truth_row["dem_share_2p"]),
                "pred_margin": pred_margin,
                "true_margin": true_margin,
                "error": pred_margin - true_margin,
                "pred_winner": pred_winner,
                "true_winner": true_winner,
                "winner_correct": pred_winner == true_winner,
                "effective_n_agents": int(group["base_ces_id"].nunique()),
                "response_count": int(len(group)),
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    return pd.DataFrame(rows)


def parse_diagnostics(responses: pd.DataFrame, run_id: str) -> pd.DataFrame:
    rows = []
    for (condition, model_name), group in responses.groupby(["condition", "model_name"], dropna=False):
        parse_status = group["parse_status"].fillna("").astype(str)
        invalid_choice = group.get("invalid_choice", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        forbidden_choice = group.get("forbidden_choice", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        legacy_schema = group.get("legacy_probability_schema", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        transport_error = group.get("transport_error", pd.Series(index=group.index, dtype=object)).notna()
        rows.append(
            {
                "run_id": run_id,
                "condition": condition,
                "baseline": condition,
                "model_name": model_name,
                "n": int(len(group)),
                "parse_ok_rate": float((parse_status == "ok").mean()),
                "invalid_choice_rate": float(invalid_choice.mean()),
                "forbidden_choice_rate": float(forbidden_choice.mean()),
                "legacy_probability_schema_rate": float(legacy_schema.mean()),
                "transport_error_rate": float(transport_error.mean()),
                "cache_hit_rate": float(group["cache_hit"].fillna(False).astype(bool).mean())
                if "cache_hit" in group.columns and group["cache_hit"].notna().any()
                else None,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    return pd.DataFrame(rows)


def state_swap_diagnostics(state_predictions: pd.DataFrame, mit_truth: pd.DataFrame, run_id: str) -> pd.DataFrame:
    truth = _truth_year(mit_truth, 2024, None).set_index("state_po")
    named = state_predictions[state_predictions["condition"] == "named_candidates"].set_index("state_po")
    swapped = state_predictions[state_predictions["condition"] == "state_swap_placebo"].set_index("state_po")
    rows = []
    for state, row in swapped.iterrows():
        if state not in named.index:
            continue
        displayed = row["displayed_state_po"]
        displayed_truth = float(truth.loc[displayed, "margin_2p"]) if displayed in truth.index else np.nan
        original_truth = float(truth.loc[state, "margin_2p"]) if state in truth.index else np.nan
        rows.append(
            {
                "run_id": run_id,
                "state_po": state,
                "displayed_state_po": displayed,
                "named_pred_margin": float(named.loc[state, "pred_margin"]),
                "swapped_pred_margin": float(row["pred_margin"]),
                "pred_shift": float(row["pred_margin"]) - float(named.loc[state, "pred_margin"]),
                "original_true_margin": original_truth,
                "displayed_true_margin": displayed_truth,
                "truth_shift": displayed_truth - original_truth if np.isfinite(displayed_truth) and np.isfinite(original_truth) else np.nan,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    return pd.DataFrame(rows)


def candidate_swap_diagnostics(state_predictions: pd.DataFrame, run_id: str) -> pd.DataFrame:
    named = state_predictions[state_predictions["condition"] == "named_candidates"].set_index("state_po")
    swapped = state_predictions[state_predictions["condition"] == "candidate_swap_placebo"].set_index("state_po")
    rows = []
    for state, row in swapped.iterrows():
        if state not in named.index:
            continue
        named_dem = float(named.loc[state, "pred_dem_2p"])
        swapped_dem = float(row["pred_dem_2p"])
        party_following_score = 1.0 - abs(swapped_dem - named_dem)
        name_following_score = 1.0 - abs(swapped_dem - (1.0 - named_dem))
        rows.append(
            {
                "run_id": run_id,
                "state_po": state,
                "named_pred_dem_2p": named_dem,
                "candidate_swap_pred_dem_2p": swapped_dem,
                "party_following_score": party_following_score,
                "name_following_score": name_following_score,
                "candidate_name_following_index": name_following_score - party_following_score,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    return pd.DataFrame(rows)


def leakage_contrast_rows(
    *,
    individual_metrics: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    state_swap: pd.DataFrame,
    candidate_swap: pd.DataFrame,
    run_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    weighted = individual_metrics[(individual_metrics["metric_scope"] == "individual") & (individual_metrics["weighted"].astype(bool))]
    ind_lookup = weighted.groupby(["baseline", "metric_name"])["metric_value"].first()
    agg_lookup = aggregate_metrics.groupby(["baseline", "metric_name"])["metric_value"].first()
    comparisons = [c for c in LEAKAGE_CONDITIONS if c != "named_candidates"]
    for comp in comparisons:
        for metric_name in ["margin_mae", "dem_2p_rmse"]:
            if ("named_candidates", metric_name) in agg_lookup.index and (comp, metric_name) in agg_lookup.index:
                named_value = float(agg_lookup.loc[("named_candidates", metric_name)])
                comp_value = float(agg_lookup.loc[(comp, metric_name)])
                rows.append(
                    {
                        "run_id": run_id,
                        "contrast_name": "named_aggregate_advantage",
                        "comparison_condition": comp,
                        "metric_name": metric_name,
                        "named_value": named_value,
                        "comparison_value": comp_value,
                        "named_improvement": comp_value - named_value,
                        "created_at": pd.Timestamp.now(tz="UTC"),
                    }
                )
        if (
            ("named_candidates", "vote_accuracy") in ind_lookup.index
            and (comp, "vote_accuracy") in ind_lookup.index
            and ("named_candidates", "margin_mae") in agg_lookup.index
            and (comp, "margin_mae") in agg_lookup.index
        ):
            agg_named = float(agg_lookup.loc[("named_candidates", "margin_mae")])
            agg_comp = float(agg_lookup.loc[(comp, "margin_mae")])
            individual_improvement = float(ind_lookup.loc[("named_candidates", "vote_accuracy")]) - float(
                ind_lookup.loc[(comp, "vote_accuracy")]
            )
            aggregate_improvement = agg_comp - agg_named
            rows.append(
                {
                    "run_id": run_id,
                    "contrast_name": "individual_vs_aggregate_gap",
                    "comparison_condition": comp,
                    "metric_name": "vote_accuracy_vs_margin_mae",
                    "named_value": aggregate_improvement,
                    "comparison_value": individual_improvement,
                    "named_improvement": aggregate_improvement - individual_improvement,
                    "created_at": pd.Timestamp.now(tz="UTC"),
                }
            )
    masked = aggregate_metrics[
        (aggregate_metrics["baseline"] == "masked_state")
        & (aggregate_metrics["metric_name"].isin(["margin_mae", "dem_2p_rmse"]))
    ]
    for _, row in masked.iterrows():
        rows.append(
            {
                "run_id": run_id,
                "contrast_name": "masked_state_retention",
                "comparison_condition": "masked_state",
                "metric_name": row["metric_name"],
                "named_value": None,
                "comparison_value": float(row["metric_value"]),
                "named_improvement": None,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    if not state_swap.empty and state_swap["truth_shift"].notna().sum() >= 2:
        corr = float(state_swap["pred_shift"].corr(state_swap["truth_shift"]))
        slope = float(np.polyfit(state_swap["truth_shift"], state_swap["pred_shift"], 1)[0])
        rows.append(
            {
                "run_id": run_id,
                "contrast_name": "state_prior_shift",
                "comparison_condition": "state_swap_placebo",
                "metric_name": "pred_shift_vs_truth_shift",
                "named_value": corr,
                "comparison_value": slope,
                "named_improvement": None,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    if not candidate_swap.empty:
        rows.append(
            {
                "run_id": run_id,
                "contrast_name": "candidate_name_following_index",
                "comparison_condition": "candidate_swap_placebo",
                "metric_name": "mean_name_minus_party_following",
                "named_value": None,
                "comparison_value": float(candidate_swap["candidate_name_following_index"].mean()),
                "named_improvement": None,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    return pd.DataFrame(rows)


def write_leakage_figures(
    *,
    run_dir: Path,
    individual_metrics: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    state_predictions: pd.DataFrame,
    leakage_contrasts: pd.DataFrame,
    state_swap: pd.DataFrame,
    candidate_swap: pd.DataFrame,
    parse_diag: pd.DataFrame,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir = ensure_dir(run_dir / "figures")
    written: list[Path] = []
    core = individual_metrics[
        (individual_metrics["metric_scope"] == "individual")
        & (individual_metrics["weighted"].astype(bool))
        & individual_metrics["metric_name"].isin(["turnout_accuracy_at_0_5", "turnout_ece", "vote_accuracy", "vote_macro_f1"])
    ].copy()
    agg_core = aggregate_metrics[aggregate_metrics["metric_name"].isin(["dem_2p_rmse", "margin_mae", "margin_bias", "winner_accuracy"])].copy()
    if not core.empty:
        pivot = core.pivot_table(index="baseline", columns="metric_name", values="metric_value", aggfunc="first")
        fig, ax = plt.subplots(figsize=(11, max(4.5, 0.4 * len(pivot))))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title("World-Knowledge Stress: Individual Metrics")
        written.append(_save_figure(fig, fig_dir / "stress_condition_metric_heatmap"))
        plt.close(fig)
    if not state_predictions.empty:
        plot = state_predictions.sort_values(["state_po", "error"]).copy()
        fig, ax = plt.subplots(figsize=(12, max(6, 0.26 * len(plot))))
        y = np.arange(len(plot))
        ax.hlines(y=y, xmin=0, xmax=plot["error"], color="#9ca3af", linewidth=1)
        ax.scatter(plot["error"], y, c=np.where(plot["error"] >= 0, "#2563eb", "#dc2626"), s=24)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(plot["state_po"] + " | " + plot["condition"])
        ax.set_xlabel("Margin error vs original-state MIT truth")
        ax.set_title("Aggregate Margin Error by Stress Condition")
        written.append(_save_figure(fig, fig_dir / "aggregate_margin_error_lollipop"))
        plt.close(fig)
    deltas = leakage_contrasts[leakage_contrasts["contrast_name"] == "named_aggregate_advantage"].copy()
    if not deltas.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=deltas, x="comparison_condition", y="named_improvement", hue="metric_name", ax=ax)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.tick_params(axis="x", rotation=25)
        ax.set_title("Named Candidate Aggregate Advantage")
        written.append(_save_figure(fig, fig_dir / "named_vs_masked_delta_bars"))
        plt.close(fig)
    gap = leakage_contrasts[leakage_contrasts["contrast_name"] == "individual_vs_aggregate_gap"].copy()
    if not gap.empty:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(gap["comparison_value"], gap["named_value"])
        for _, row in gap.iterrows():
            ax.text(row["comparison_value"], row["named_value"], row["comparison_condition"], fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Named individual vote-accuracy improvement")
        ax.set_ylabel("Named aggregate margin-MAE improvement")
        ax.set_title("Individual vs Aggregate Leakage Gap")
        written.append(_save_figure(fig, fig_dir / "individual_vs_aggregate_leakage_quadrant"))
        plt.close(fig)
    if not state_swap.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(state_swap["truth_shift"], state_swap["pred_shift"])
        for _, row in state_swap.iterrows():
            ax.text(row["truth_shift"], row["pred_shift"], f"{row['state_po']}->{row['displayed_state_po']}", fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Displayed-state truth margin - original truth margin")
        ax.set_ylabel("State-swap predicted margin shift")
        ax.set_title("State-Swap Prior Diagnostic")
        written.append(_save_figure(fig, fig_dir / "state_swap_vector_plot"))
        plt.close(fig)
    masked = state_predictions[state_predictions["condition"] == "masked_state"].copy()
    if not masked.empty:
        masked["abs_error"] = masked["error"].abs()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=masked, x="state_po", y="abs_error", ax=ax)
        ax.set_ylabel("Absolute margin error vs original MIT truth")
        ax.set_title("Masked-State Retention")
        written.append(_save_figure(fig, fig_dir / "masked_state_retention_bar"))
        plt.close(fig)
    if not candidate_swap.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot = candidate_swap.melt(
            id_vars=["state_po"],
            value_vars=["party_following_score", "name_following_score", "candidate_name_following_index"],
            var_name="diagnostic",
            value_name="value",
        )
        sns.barplot(data=plot, x="state_po", y="value", hue="diagnostic", ax=ax)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Candidate-Swap Name Following")
        written.append(_save_figure(fig, fig_dir / "candidate_swap_name_following"))
        plt.close(fig)
    if not parse_diag.empty:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        plot = parse_diag.melt(id_vars=["condition"], value_vars=["parse_ok_rate", "cache_hit_rate"], var_name="diagnostic", value_name="rate")
        sns.barplot(data=plot, x="condition", y="rate", hue="diagnostic", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=25)
        ax.set_title("Parse and Runtime Cache Diagnostics")
        written.append(_save_figure(fig, fig_dir / "parse_runtime_diagnostics"))
        plt.close(fig)
    return written


def write_leakage_report(
    *,
    run_dir: Path,
    cfg: dict[str, Any],
    sampled_agents: pd.DataFrame,
    individual_metrics: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    leakage_contrasts: pd.DataFrame,
    state_swap: pd.DataFrame,
    candidate_swap: pd.DataFrame,
    parse_diag: pd.DataFrame,
    figures: list[Path],
    llm_metadata: dict[str, Any],
) -> Path:
    def table(df: pd.DataFrame, cols: list[str], head: int = 80) -> str:
        if df.empty:
            return "_No rows._"
        show = df[[col for col in cols if col in df.columns]].head(head).copy()
        for col in show.columns:
            show[col] = show[col].map(lambda value: "" if pd.isna(value) else str(value))
        lines = ["| " + " | ".join(show.columns) + " |", "| " + " | ".join("---" for _ in show.columns) + " |"]
        for _, row in show.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in show.columns) + " |")
        return "\n".join(lines)

    def metric_lookup(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        return df.groupby(["baseline", "metric_name"])["metric_value"].first()

    ind_lookup = metric_lookup(
        individual_metrics[
            (individual_metrics["metric_scope"] == "individual")
            & (individual_metrics["weighted"].astype(bool))
        ]
    )
    agg_lookup = metric_lookup(aggregate_metrics)

    def value(baseline: str, metric: str) -> float | None:
        key = (baseline, metric)
        if key not in agg_lookup.index and key not in ind_lookup.index:
            return None
        source = agg_lookup if key in agg_lookup.index else ind_lookup
        return float(source.loc[key])

    named_mae = value("named_candidates", "margin_mae")
    party_mae = value("party_only_candidates", "margin_mae")
    anon_mae = value("anonymous_candidates", "margin_mae")
    masked_year_mae = value("masked_year", "margin_mae")
    masked_state_mae = value("masked_state", "margin_mae")
    named_vote = value("named_candidates", "vote_accuracy")
    party_vote = value("party_only_candidates", "vote_accuracy")
    effective_per_state = llm_metadata.get("effective_agents_per_state") or cfg.get("agents_per_state", 40)
    state_prior = leakage_contrasts[
        (leakage_contrasts["contrast_name"] == "state_prior_shift")
        & (leakage_contrasts["metric_name"] == "pred_shift_vs_truth_shift")
    ]
    candidate_follow = leakage_contrasts[
        (leakage_contrasts["contrast_name"] == "candidate_name_following_index")
        & (leakage_contrasts["metric_name"] == "mean_name_minus_party_following")
    ]
    key_findings = [
        (
            f"Named candidates margin MAE = {named_mae:.3f}; party-only = {party_mae:.3f}; "
            f"anonymous = {anon_mae:.3f}; masked-year = {masked_year_mae:.3f}."
            if None not in {named_mae, party_mae, anon_mae, masked_year_mae}
            else "Named-vs-masked aggregate comparison was unavailable."
        ),
        (
            "In this pilot, explicit candidate names did not improve aggregate accuracy over party-only or masked-year prompts; "
            "that weakens the simplest world-knowledge leakage explanation for the named condition."
            if named_mae is not None and party_mae is not None and masked_year_mae is not None and named_mae > min(party_mae, masked_year_mae)
            else "Named candidates improved aggregate accuracy over at least one masked comparison; inspect individual-vs-aggregate gaps for leakage risk."
        ),
        (
            f"Weighted individual vote accuracy: named = {named_vote:.3f}, party-only = {party_vote:.3f}; "
            "aggregate gains should be interpreted against this individual-level behavior."
            if named_vote is not None and party_vote is not None
            else "Individual vote-accuracy comparison was unavailable."
        ),
        (
            f"Masked-state margin MAE = {masked_state_mae:.3f}; state labels still matter, but all conditions should be read with the small {effective_per_state}/state sample limit."
            if masked_state_mae is not None
            else "Masked-state retention metric was unavailable."
        ),
    ]
    if not state_prior.empty:
        key_findings.append(
            f"State-swap pred-vs-truth shift correlation = {float(state_prior['named_value'].iloc[0]):.3f}, "
            f"slope = {float(state_prior['comparison_value'].iloc[0]):.3f}; low values indicate weak state-result-prior following in this run."
        )
    if not candidate_follow.empty:
        idx = float(candidate_follow["comparison_value"].iloc[0])
        key_findings.append(
            f"Candidate name-following index = {idx:.3f}; negative values mean predictions stayed closer to party/voter facts than swapped candidate names."
        )

    metric_summary = individual_metrics[
        (individual_metrics["metric_scope"] == "individual")
        & (individual_metrics["weighted"].astype(bool))
        & individual_metrics["metric_name"].isin(["turnout_accuracy_at_0_5", "turnout_ece", "vote_accuracy", "vote_macro_f1"])
    ].sort_values(["baseline", "metric_name"])
    agg_summary = aggregate_metrics[
        aggregate_metrics["metric_name"].isin(["dem_2p_rmse", "margin_mae", "margin_bias", "winner_accuracy", "winner_flip_count"])
    ].sort_values(["baseline", "metric_name"])
    sample_summary = sampled_agents.groupby("state_po").size().reset_index(name="n")
    figure_rows = pd.DataFrame({"figure": [str(path.relative_to(run_dir)) for path in figures]})
    lines = [
        f"# CES World-Knowledge Leakage Stress Test: {cfg['run_id']}",
        "",
        "## Run Summary",
        f"- States: {', '.join(cfg.get('states', DEFAULT_STATES))}",
        f"- Sampled agents: {len(sampled_agents):,}",
        f"- LLM model: `{cfg.get('model', {}).get('model_name', 'qwen3.5:0.8b')}`",
        f"- LLM metadata: `{stable_json(llm_metadata)}`",
        "- Aggregate metrics use original CES respondent state, even when prompts mask or swap displayed state.",
        "",
        "## Key Findings",
        "\n".join(f"- {finding}" for finding in key_findings),
        "",
        "## Sample Summary",
        table(sample_summary, ["state_po", "n"], head=40),
        "",
        "## Weighted Individual Metrics",
        table(metric_summary, ["baseline", "model_name", "metric_name", "metric_value", "n"], head=120),
        "",
        "## Aggregate Metrics",
        table(agg_summary, ["baseline", "metric_name", "metric_value", "n_states"], head=120),
        "",
        "## Leakage Contrasts",
        table(leakage_contrasts, ["contrast_name", "comparison_condition", "metric_name", "named_value", "comparison_value", "named_improvement"], head=120),
        "",
        "## State Swap Diagnostics",
        table(state_swap, ["state_po", "displayed_state_po", "pred_shift", "truth_shift"], head=80),
        "",
        "## Candidate Swap Diagnostics",
        table(candidate_swap, ["state_po", "party_following_score", "name_following_score", "candidate_name_following_index"], head=80),
        "",
        "## Parse Diagnostics",
        table(parse_diag, ["condition", "model_name", "n", "parse_ok_rate", "cache_hit_rate"], head=80),
        "",
        "## Interpretation Rule",
        "- If named candidates improve aggregate margin without comparable individual gains, treat it as likely world-knowledge leakage.",
        "- If state-swap predictions move toward displayed-state truth, state-result priors are too strong.",
        "- If candidate-swap predictions follow names more than parties, candidate-name knowledge is contaminating voter simulation.",
        "",
        "## Figures",
        table(figure_rows, ["figure"], head=80),
        "",
    ]
    out = run_dir / "benchmark_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run_ces_leakage_benchmark(config_path: str | Path) -> dict[str, Path]:
    cfg = load_yaml(config_path)
    run_id = cfg["run_id"]
    run_dir = ensure_dir(cfg.get("paths", {}).get("run_dir", f"data/runs/{run_id}"))
    write_yaml(cfg, run_dir / "config_snapshot.yaml")
    started = time.time()
    paths = AggregateBenchmarkPaths(run_dir=run_dir, figures_dir=ensure_dir(run_dir / "figures"), cache_path=run_dir / "llm_cache.jsonl")
    states = list(cfg.get("states", DEFAULT_STATES))
    seed = int(cfg.get("seed", 20260426))
    respondents = pd.read_parquet(cfg["paths"]["ces_respondents"])
    targets = pd.read_parquet(cfg["paths"]["ces_targets"])
    strict_memory_facts = pd.read_parquet(cfg["paths"]["ces_memory_facts_strict"])
    mit_truth = pd.read_parquet(cfg["paths"]["mit_state_truth"])
    cohort = build_benchmark_cohort(respondents, seed, states)
    sampled = sample_leakage_agents(
        cohort,
        states=states,
        agents_per_state=int(cfg.get("agents_per_state", 40)),
        seed=seed,
    )
    agents = _agents_from_cohort(run_id, sampled)
    agents = agents.merge(sampled[["ces_id", "sample_rank"]], left_on="base_ces_id", right_on="ces_id", how="left").drop(columns=["ces_id"])
    responses, prompts, condition_metadata, runtime_log, llm_metadata = run_llm_leakage(
        cfg=cfg,
        paths=paths,
        sampled=sampled,
        agents=agents,
        strict_memory_facts=strict_memory_facts,
    )
    responses = add_condition_metadata(responses, condition_metadata, agents)
    active_ids = set(responses["base_ces_id"].astype(str))
    sampled_agents = sampled[sampled["ces_id"].astype(str).isin(active_ids)].copy()
    individual_metrics = pd.DataFrame(benchmark_metric_rows(responses, sampled_agents, targets, run_id, metric_scope="individual"))
    calibration_bins = turnout_calibration_bins(responses, targets, run_id)
    state_predictions = state_prediction_rows(
        responses=responses,
        mit_truth=mit_truth,
        run_id=run_id,
        states=states,
        sample_size=int(llm_metadata.get("effective_agents_per_state") or cfg.get("agents_per_state", 40)),
    )
    aggregate_metrics = pd.DataFrame(swing_aggregate_metric_rows(state_predictions, run_id))
    state_swap = state_swap_diagnostics(state_predictions, mit_truth, run_id)
    candidate_swap = candidate_swap_diagnostics(state_predictions, run_id)
    leakage_contrasts = leakage_contrast_rows(
        individual_metrics=individual_metrics,
        aggregate_metrics=aggregate_metrics,
        state_swap=state_swap,
        candidate_swap=candidate_swap,
        run_id=run_id,
    )
    parse_diag = parse_diagnostics(responses, run_id)

    write_table(sampled_agents, run_dir / "sampled_agents.parquet")
    write_table(condition_metadata, run_dir / "condition_metadata.parquet")
    write_table(prompts, run_dir / "prompts.parquet")
    write_table(responses, run_dir / "responses.parquet")
    write_table(individual_metrics, run_dir / "individual_metrics.parquet")
    write_table(calibration_bins, run_dir / "calibration_bins.parquet")
    write_table(state_predictions, run_dir / "state_predictions.parquet")
    write_table(aggregate_metrics, run_dir / "aggregate_metrics.parquet")
    write_table(leakage_contrasts, run_dir / "leakage_contrasts.parquet")
    write_table(state_swap, run_dir / "state_swap_diagnostics.parquet")
    write_table(candidate_swap, run_dir / "candidate_swap_diagnostics.parquet")
    write_table(parse_diag, run_dir / "parse_diagnostics.parquet")
    write_table(runtime_log, run_dir / "runtime_log.parquet")
    figures = write_leakage_figures(
        run_dir=run_dir,
        individual_metrics=individual_metrics,
        aggregate_metrics=aggregate_metrics,
        state_predictions=state_predictions,
        leakage_contrasts=leakage_contrasts,
        state_swap=state_swap,
        candidate_swap=candidate_swap,
        parse_diag=parse_diag,
    )
    report = write_leakage_report(
        run_dir=run_dir,
        cfg=cfg,
        sampled_agents=sampled_agents,
        individual_metrics=individual_metrics,
        aggregate_metrics=aggregate_metrics,
        leakage_contrasts=leakage_contrasts,
        state_swap=state_swap,
        candidate_swap=candidate_swap,
        parse_diag=parse_diag,
        figures=figures,
        llm_metadata=llm_metadata,
    )
    runtime = {
        "run_id": run_id,
        "git_commit": git_commit(),
        "states": states,
        "conditions": [name for name in cfg.get("conditions", LEAKAGE_CONDITIONS) if name in LEAKAGE_CONDITIONS],
        "n_agents": int(len(sampled_agents)),
        "n_prompts": int(len(prompts)),
        "n_llm_tasks": int(len(runtime_log)),
        "runtime_seconds": time.time() - started,
        **llm_metadata,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    write_json(runtime, run_dir / "runtime.json")
    return {
        "sampled_agents": run_dir / "sampled_agents.parquet",
        "condition_metadata": run_dir / "condition_metadata.parquet",
        "prompts": run_dir / "prompts.parquet",
        "responses": run_dir / "responses.parquet",
        "individual_metrics": run_dir / "individual_metrics.parquet",
        "calibration_bins": run_dir / "calibration_bins.parquet",
        "state_predictions": run_dir / "state_predictions.parquet",
        "aggregate_metrics": run_dir / "aggregate_metrics.parquet",
        "leakage_contrasts": run_dir / "leakage_contrasts.parquet",
        "state_swap_diagnostics": run_dir / "state_swap_diagnostics.parquet",
        "candidate_swap_diagnostics": run_dir / "candidate_swap_diagnostics.parquet",
        "parse_diagnostics": run_dir / "parse_diagnostics.parquet",
        "runtime_log": run_dir / "runtime_log.parquet",
        "runtime": run_dir / "runtime.json",
        "report": report,
    }
