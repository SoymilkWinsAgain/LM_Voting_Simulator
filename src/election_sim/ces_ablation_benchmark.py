"""CES baseline ladder / ablation benchmark runner."""

from __future__ import annotations

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
    DEFAULT_SUBGROUPS,
    VOTE_CLASSES,
    _agents_from_cohort,
    _combined_probabilities,
    _group_context,
    _group_memory_facts,
    _response_rows,
    _target_wide,
    benchmark_metric_rows,
    build_benchmark_cohort,
    compute_subgroup_metrics,
    expected_calibration_error,
    turnout_calibration_bins,
)
from .ces_schema import parse_turnout_vote_json
from .config import ModelConfig
from .eval_suite import git_commit, gpu_peak_summary, gpu_snapshot, raw_choice_diagnostics
from .io import ensure_dir, load_yaml, stable_json, write_json, write_table, write_yaml
from .llm import build_llm_client
from .transforms import stable_hash


DEFAULT_STATES = ["PA", "GA", "AZ", "WI"]
LADDER_BASELINES = [
    "L1_demographic_only_llm",
    "L2_demographic_state_llm",
    "L3_party_ideology_llm",
    "L4_party_ideology_context_llm",
    "L5_strict_memory_llm",
    "L6_strict_memory_context_llm",
    "L7_poll_informed_memory_context_llm",
    "L8_post_hoc_oracle_memory_context_llm",
    "P1_memory_shuffled_within_state_llm",
    "P2_memory_shuffled_within_party_llm",
]
BASELINE_ORDER = {name: idx + 1 for idx, name in enumerate(LADDER_BASELINES)}
CONTEXT_BASELINES = {
    "L4_party_ideology_context_llm",
    "L6_strict_memory_context_llm",
    "L7_poll_informed_memory_context_llm",
    "L8_post_hoc_oracle_memory_context_llm",
    "P1_memory_shuffled_within_state_llm",
    "P2_memory_shuffled_within_party_llm",
}
PARTY_BASELINES = set(LADDER_BASELINES) - {"L1_demographic_only_llm", "L2_demographic_state_llm"}
STATE_BASELINES = set(LADDER_BASELINES) - {"L1_demographic_only_llm"}
STRICT_MEMORY_BASELINES = {
    "L5_strict_memory_llm",
    "L6_strict_memory_context_llm",
    "P1_memory_shuffled_within_state_llm",
    "P2_memory_shuffled_within_party_llm",
}
POLL_MEMORY_BASELINES = {"L7_poll_informed_memory_context_llm"}
ORACLE_BASELINES = {"L8_post_hoc_oracle_memory_context_llm"}
PLACEBO_BASELINES = {"P1_memory_shuffled_within_state_llm", "P2_memory_shuffled_within_party_llm"}

PROMPT_COLUMNS = [
    "run_id",
    "prompt_id",
    "agent_id",
    "base_ces_id",
    "baseline",
    "model_name",
    "prompt_hash",
    "prompt_text",
    "memory_fact_ids_used",
    "memory_donor_ces_id",
    "memory_shuffle_scope",
    "memory_shuffle_fallback_reason",
    "cache_hit",
    "latency_ms",
    "transport_error",
    "created_at",
]


@dataclass
class AblationTask:
    baseline: str
    agent_id: str
    base_ces_id: str
    state_po: str
    sample_flag: str
    headline_sample: bool
    prompt_id: str
    prompt_hash: str
    prompt_text: str
    fact_ids: list[str]
    cache_key: str
    memory_donor_ces_id: str | None = None
    memory_shuffle_scope: str | None = None
    memory_shuffle_fallback_reason: str | None = None


def choose_effective_agents_per_state(
    *,
    requested_main_per_state: int,
    diagnostic_boost_per_state: int,
    n_states: int,
    n_baselines: int,
    observed_throughput_per_second: float | None,
    max_runtime_minutes: float,
) -> tuple[int, str | None, float | None]:
    """Return a conservative main sample size from measured LLM throughput."""

    requested_total = (requested_main_per_state + diagnostic_boost_per_state) * n_states * n_baselines
    projected_minutes = (
        requested_total / observed_throughput_per_second / 60.0
        if observed_throughput_per_second and observed_throughput_per_second > 0
        else None
    )
    if projected_minutes is None or projected_minutes <= max_runtime_minutes:
        return requested_main_per_state, None, projected_minutes
    for candidate in [40, 30]:
        candidate_total = (candidate + diagnostic_boost_per_state) * n_states * n_baselines
        candidate_minutes = candidate_total / observed_throughput_per_second / 60.0
        if candidate_minutes <= max_runtime_minutes:
            return min(candidate, requested_main_per_state), f"runtime_reduced_to_{candidate}", candidate_minutes
    return min(30, requested_main_per_state), "runtime_reduced_to_30_projected_over_budget", (
        (min(30, requested_main_per_state) + diagnostic_boost_per_state)
        * n_states
        * n_baselines
        / observed_throughput_per_second
        / 60.0
    )


def _weighted_sample(group: pd.DataFrame, *, n: int, seed_key: str, weight_col: str = "sample_weight") -> pd.DataFrame:
    if n <= 0 or group.empty:
        return group.head(0).copy()
    n = min(n, len(group))
    weights = group[weight_col].fillna(0).astype(float) if weight_col in group.columns else None
    if weights is not None and float(weights.sum()) <= 0:
        weights = None
    if weights is not None:
        prob = weights.to_numpy(dtype=float)
        prob = prob / prob.sum()
        rng = np.random.default_rng(int(stable_hash(seed_key, length=8), 16))
        try:
            return group.iloc[rng.choice(len(group), size=n, replace=False, p=prob)].copy()
        except ValueError:
            return group.iloc[rng.choice(len(group), size=n, replace=False)].copy()
    rng = np.random.default_rng(int(stable_hash(seed_key, length=8), 16))
    return group.iloc[rng.choice(len(group), size=n, replace=False)].copy()


def sample_ablation_agents(
    cohort: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    states: list[str],
    main_agents_per_state: int,
    diagnostic_boost_per_state: int,
    seed: int,
) -> pd.DataFrame:
    """Draw fixed test-split agents for headline and diagnostic ablation rows."""

    targets_wide = _target_wide(targets)
    test = cohort[(cohort["split"] == "test") & (cohort["state_po"].isin(states))].copy()
    test = test.merge(targets_wide, on="ces_id", how="left")
    parts: list[pd.DataFrame] = []
    for state in states:
        state_group = test[test["state_po"] == state].copy()
        if state_group.empty:
            continue
        main = _weighted_sample(
            state_group,
            n=main_agents_per_state,
            seed_key=f"ablation-main-{seed}-{state}",
        )
        main = main.copy()
        main["sample_flag"] = "main"
        main["headline_sample"] = True
        remaining = state_group[~state_group["ces_id"].isin(set(main["ces_id"].astype(str)))].copy()
        hard = remaining[
            remaining["president_vote_2024"].isin(["not_vote", "other"])
            | remaining["turnout_2024_self_report"].isin(["not_voted"])
            | remaining["party_id_3_pre"].isin(["unknown", "independent_or_other"])
        ].copy()
        boost = _weighted_sample(
            hard,
            n=diagnostic_boost_per_state,
            seed_key=f"ablation-boost-hard-{seed}-{state}",
        )
        if len(boost) < diagnostic_boost_per_state:
            filler = remaining[~remaining["ces_id"].isin(set(boost["ces_id"].astype(str)))].copy()
            extra = _weighted_sample(
                filler,
                n=diagnostic_boost_per_state - len(boost),
                seed_key=f"ablation-boost-fill-{seed}-{state}",
            )
            boost = pd.concat([boost, extra], ignore_index=False)
        boost = boost.copy()
        boost["sample_flag"] = "diagnostic_boost"
        boost["headline_sample"] = False
        out = pd.concat([main, boost], ignore_index=False).drop_duplicates("ces_id", keep="first").copy()
        out["sample_rank"] = np.arange(1, len(out) + 1)
        parts.append(out)
    if not parts:
        return test.head(0).copy()
    sampled = pd.concat(parts, ignore_index=True)
    keep_target_cols = [col for col in targets_wide.columns if col != "ces_id"]
    return sampled.drop(columns=keep_target_cols, errors="ignore").reset_index(drop=True)


def _select_facts(
    memory_by_id: Mapping[str, pd.DataFrame],
    ces_id: str,
    *,
    fact_roles: set[str],
    max_facts: int,
) -> tuple[list[str], list[str]]:
    facts = memory_by_id.get(str(ces_id))
    if facts is None or facts.empty:
        return [], []
    selected = facts.copy()
    if "fact_role" in selected.columns:
        selected = selected[selected["fact_role"].fillna("safe_pre").astype(str).isin(fact_roles)]
    if "fact_priority" in selected.columns:
        selected = selected.sort_values(["fact_priority", "source_variable"], ascending=[False, True])
    selected = selected.head(max_facts)
    if selected.empty:
        return [], []
    return selected["fact_text"].astype(str).tolist(), selected["memory_fact_id"].astype(str).tolist()


def _oracle_memory(agent: pd.Series, targets_wide: pd.DataFrame) -> tuple[list[str], list[str]]:
    target = targets_wide.set_index("ces_id").loc[str(agent["base_ces_id"])]
    turnout = str(target.get("turnout_2024_self_report", "unknown"))
    vote = str(target.get("president_vote_2024", "unknown"))
    facts = [
        f"Leakage upper bound: the respondent's post-election turnout label is {turnout}.",
        f"Leakage upper bound: the respondent's post-election presidential vote label is {vote}.",
    ]
    return facts, [f"oracle_turnout:{agent['base_ces_id']}", f"oracle_vote:{agent['base_ces_id']}"]


def build_memory_donor_map(sampled_agents: pd.DataFrame, *, scope: str, seed: int) -> dict[str, dict[str, Any]]:
    """Return deterministic memory donors for state or party placebo conditions."""

    rows: dict[str, dict[str, Any]] = {}
    for _, agent in sampled_agents.iterrows():
        ces_id = str(agent["base_ces_id"])
        if scope == "state":
            candidates = sampled_agents[
                (sampled_agents["state_po"].astype(str) == str(agent["state_po"]))
                & (sampled_agents["base_ces_id"].astype(str) != ces_id)
            ].copy()
        elif scope == "party":
            candidates = sampled_agents[
                (sampled_agents["party_id_3"].astype(str) == str(agent["party_id_3"]))
                & (sampled_agents["base_ces_id"].astype(str) != ces_id)
            ].copy()
        else:
            raise ValueError(f"Unsupported donor scope: {scope}")
        fallback = None
        if candidates.empty:
            fallback = "no_same_scope_donor_used_state_fallback"
            candidates = sampled_agents[
                (sampled_agents["state_po"].astype(str) == str(agent["state_po"]))
                & (sampled_agents["base_ces_id"].astype(str) != ces_id)
            ].copy()
        if candidates.empty:
            fallback = "no_donor_available"
            rows[ces_id] = {"memory_donor_ces_id": None, "memory_shuffle_fallback_reason": fallback}
            continue
        candidates = candidates.assign(
            _donor_rank=candidates["base_ces_id"].astype(str).map(
                lambda value: stable_hash("ablation-donor", seed, scope, ces_id, value, length=16)
            )
        )
        donor = candidates.sort_values("_donor_rank").iloc[0]
        rows[ces_id] = {
            "memory_donor_ces_id": str(donor["base_ces_id"]),
            "memory_shuffle_fallback_reason": fallback,
        }
    return rows


def render_ablation_prompt(
    *,
    baseline: str,
    agent: pd.Series,
    question: pd.Series | dict[str, Any],
    strict_memory: Mapping[str, pd.DataFrame],
    poll_memory: Mapping[str, pd.DataFrame],
    context: Mapping[str, list[dict[str, Any]]],
    targets_wide: pd.DataFrame,
    donor_maps: Mapping[str, Mapping[str, dict[str, Any]]],
    max_memory_facts: int,
) -> tuple[str, list[str], dict[str, Any]]:
    """Render one ablation prompt and return prompt facts metadata."""

    if baseline not in LADDER_BASELINES:
        raise ValueError(f"Unknown ablation baseline: {baseline}")
    base_ces_id = str(agent["base_ces_id"])
    memory_source_id = base_ces_id
    memory_donor_ces_id = None
    memory_shuffle_scope = None
    memory_shuffle_fallback_reason = None
    fact_texts: list[str] = []
    fact_ids: list[str] = []
    if baseline in PLACEBO_BASELINES:
        memory_shuffle_scope = "state" if baseline == "P1_memory_shuffled_within_state_llm" else "party"
        donor_meta = donor_maps.get(memory_shuffle_scope, {}).get(base_ces_id, {})
        memory_donor_ces_id = donor_meta.get("memory_donor_ces_id")
        memory_shuffle_fallback_reason = donor_meta.get("memory_shuffle_fallback_reason")
        memory_source_id = str(memory_donor_ces_id or base_ces_id)
    if baseline in STRICT_MEMORY_BASELINES:
        fact_texts, fact_ids = _select_facts(
            strict_memory,
            memory_source_id,
            fact_roles={"safe_pre"},
            max_facts=max_memory_facts,
        )
    elif baseline in POLL_MEMORY_BASELINES:
        fact_texts, fact_ids = _select_facts(
            poll_memory,
            memory_source_id,
            fact_roles={"safe_pre", "poll_prior"},
            max_facts=max_memory_facts,
        )
    elif baseline in ORACLE_BASELINES:
        fact_texts, fact_ids = _oracle_memory(agent, targets_wide)

    include_state = baseline in STATE_BASELINES
    include_party = baseline in PARTY_BASELINES
    include_context = baseline in CONTEXT_BASELINES
    include_memory = bool(fact_texts)

    lines = [
        "You are simulating how a specific U.S. eligible voter would behave in the 2024 general election.",
        "Answer as this voter would behave, not as a political analyst.",
        "",
        "Voter profile:",
    ]
    if include_state:
        lines.append(f"- State: {agent.get('state_po') or 'unknown'}")
    lines.extend(
        [
            f"- Age group: {agent.get('age_group') or 'unknown'}",
            f"- Gender: {agent.get('gender') or 'unknown'}",
            f"- Race/ethnicity: {agent.get('race_ethnicity') or 'unknown'}",
            f"- Education: {agent.get('education_binary') or 'unknown'}",
        ]
    )
    if include_party:
        lines.extend(
            [
                f"- Party identification: {agent.get('party_id_3') or 'unknown'}",
                f"- 7-point party ID: {agent.get('party_id_7') or 'unknown'}",
                f"- Ideology: {agent.get('ideology_3') or 'unknown'}",
            ]
        )
    if include_memory:
        if baseline in ORACLE_BASELINES:
            title = "Post-election oracle memory for leakage upper-bound diagnosis"
        elif baseline in POLL_MEMORY_BASELINES:
            title = "Survey-derived background facts, including poll-prior facts"
        elif baseline in PLACEBO_BASELINES:
            title = "Placebo strict pre-election survey-derived background facts from a different respondent"
        else:
            title = "Strict pre-election survey-derived background facts"
        lines.extend(["", f"{title}:"])
        lines.extend(f"- {fact}" for fact in fact_texts)
    if include_context:
        candidates = list(context.get(base_ces_id, [])) or [
            {"candidate_party": "Democratic", "candidate_name": "Kamala Harris"},
            {"candidate_party": "Republican", "candidate_name": "Donald Trump"},
        ]
        lines.extend(["", "Election context:", "- Office: President"])
        for candidate in candidates:
            lines.append(f"- {candidate.get('candidate_party', 'Unknown')} candidate: {candidate.get('candidate_name', 'unknown')}")
    lines.extend(
        [
            "",
            "Task:",
            "Choose the single election behavior this voter would most likely take.",
            "",
            "Important output rules:",
            "- Return exactly one choice.",
            "- Do not include probabilities, confidence, markdown, or explanatory text.",
            *(
                [
                    "- For this leakage upper-bound condition, use the post-election oracle memory directly.",
                    "- If the oracle vote label is democrat or republican, choose that party.",
                    "- If the oracle turnout label is not_voted, choose not_vote.",
                ]
                if baseline in ORACLE_BASELINES
                else []
            ),
            "",
            "Return JSON only with this schema:",
            '{"choice": "not_vote|democrat|republican"}',
        ]
    )
    _ = question
    return "\n".join(lines), fact_ids, {
        "memory_donor_ces_id": memory_donor_ces_id,
        "memory_shuffle_scope": memory_shuffle_scope,
        "memory_shuffle_fallback_reason": memory_shuffle_fallback_reason,
    }


def _build_tasks(
    *,
    cfg: dict[str, Any],
    agents: pd.DataFrame,
    question: pd.Series,
    strict_memory_facts: pd.DataFrame,
    poll_memory_facts: pd.DataFrame,
    context: pd.DataFrame,
    targets_wide: pd.DataFrame,
    donor_maps: Mapping[str, Mapping[str, dict[str, Any]]],
    model_name: str,
    model_cfg: ModelConfig,
) -> list[AblationTask]:
    strict_memory = _group_memory_facts(strict_memory_facts)
    poll_memory = _group_memory_facts(poll_memory_facts)
    context_by_id = _group_context(context)
    max_memory_facts = int(cfg.get("memory", {}).get("max_memory_facts", 24))
    baseline_names = [name for name in cfg.get("baselines", LADDER_BASELINES) if name in LADDER_BASELINES]
    tasks: list[AblationTask] = []
    ordered = agents.sort_values(["state_po", "sample_rank", "base_ces_id"]).copy()
    for _, agent in ordered.iterrows():
        for baseline in baseline_names:
            prompt_text, fact_ids, meta = render_ablation_prompt(
                baseline=baseline,
                agent=agent,
                question=question,
                strict_memory=strict_memory,
                poll_memory=poll_memory,
                context=context_by_id,
                targets_wide=targets_wide,
                donor_maps=donor_maps,
                max_memory_facts=max_memory_facts,
            )
            prompt_hash = stable_hash(prompt_text, length=32)
            cache_key = llm_cache_key(
                model_name=model_name,
                baseline=baseline,
                prompt_hash=prompt_hash,
                temperature=float(model_cfg.temperature),
                max_tokens=int(model_cfg.max_tokens),
                response_format=str(model_cfg.response_format),
            )
            prompt_id = stable_hash(cfg["run_id"], baseline, agent["base_ces_id"], prompt_hash, length=20)
            tasks.append(
                AblationTask(
                    baseline=baseline,
                    agent_id=str(agent["agent_id"]),
                    base_ces_id=str(agent["base_ces_id"]),
                    state_po=str(agent["state_po"]),
                    sample_flag=str(agent["sample_flag"]),
                    headline_sample=bool(agent["headline_sample"]),
                    prompt_id=prompt_id,
                    prompt_hash=prompt_hash,
                    prompt_text=prompt_text,
                    fact_ids=fact_ids,
                    cache_key=cache_key,
                    **meta,
                )
            )
    return tasks


def _finish_tasks(
    *,
    client: Any,
    cache: AggregateLlmCache,
    tasks: list[AblationTask],
    cfg: dict[str, Any],
    model_name: str,
    workers: int,
    existing_results: dict[str, dict[str, Any]] | None = None,
    on_result: Callable[[tuple[AblationTask, str, bool, int | None, str | None]], None] | None = None,
) -> list[tuple[AblationTask, str, bool, int | None, str | None]]:
    existing_results = existing_results or {}

    def finish(task: AblationTask) -> tuple[AblationTask, str, bool, int | None, str | None]:
        started_call = time.time()
        if task.cache_key in existing_results:
            result = existing_results[task.cache_key]
            return task, str(result["raw"]), bool(result["cache_hit"]), result.get("latency_ms"), result.get("transport_error")
        try:
            raw, cache_hit, latency_ms = complete_llm_task_with_cache(
                client=client,
                cache=cache,
                cache_key=task.cache_key,
                run_id=cfg["run_id"],
                model_name=model_name,
                baseline=task.baseline,
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
    completed.sort(key=lambda row: (row[0].state_po, row[0].base_ces_id, BASELINE_ORDER[row[0].baseline]))
    return completed


def _run_llm_ablation(
    *,
    cfg: dict[str, Any],
    paths: AggregateBenchmarkPaths,
    sampled: pd.DataFrame,
    agents: pd.DataFrame,
    question: pd.Series,
    strict_memory_facts: pd.DataFrame,
    poll_memory_facts: pd.DataFrame,
    context: pd.DataFrame,
    targets_wide: pd.DataFrame,
    donor_maps: Mapping[str, Mapping[str, dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    model_cfg = ModelConfig.model_validate(cfg.get("model", {}))
    client = build_llm_client(model_cfg)
    model_name = getattr(client, "model_name", model_cfg.model_name)
    cache = AggregateLlmCache(paths.cache_path)
    llm_cfg = cfg.get("llm", {})
    workers = max(1, int(llm_cfg.get("workers", 8)))
    timing_responses = int(llm_cfg.get("timing_responses", 40))
    max_runtime_minutes = float(llm_cfg.get("max_runtime_minutes", 45.0))
    checkpoint_every = max(1, int(llm_cfg.get("checkpoint_every", 25)))
    gpu_sample_every = max(1, int(llm_cfg.get("gpu_sample_every", checkpoint_every)))
    gpu_snapshots = [gpu_snapshot("run_start")]

    tasks = _build_tasks(
        cfg=cfg,
        agents=agents,
        question=question,
        strict_memory_facts=strict_memory_facts,
        poll_memory_facts=poll_memory_facts,
        context=context,
        targets_wide=targets_wide,
        donor_maps=donor_maps,
        model_name=model_name,
        model_cfg=model_cfg,
    )

    timing_tasks = [task for task in tasks if cache.get(task.cache_key) is None][:timing_responses]
    timing_results: dict[str, dict[str, Any]] = {}
    timing_wall = 0.0
    timing_latencies: list[float] = []
    if timing_tasks:
        started = time.time()
        completed_timing = _finish_tasks(
            client=client,
            cache=cache,
            tasks=timing_tasks,
            cfg=cfg,
            model_name=model_name,
            workers=workers,
        )
        timing_wall = time.time() - started
        for task, raw, cache_hit, latency_ms, transport_error in completed_timing:
            timing_results[task.cache_key] = {
                "raw": raw,
                "cache_hit": cache_hit,
                "latency_ms": latency_ms,
                "transport_error": transport_error,
            }
            if latency_ms is not None:
                timing_latencies.append(float(latency_ms) / 1000.0)
    observed_throughput = len(timing_tasks) / timing_wall if timing_wall > 0 and timing_tasks else None
    requested_main = int(cfg.get("main_agents_per_state", 50))
    diagnostic_boost = int(cfg.get("diagnostic_boost_per_state", 10))
    states = list(cfg.get("states", DEFAULT_STATES))
    effective_main, limit_reason, projected_minutes = choose_effective_agents_per_state(
        requested_main_per_state=requested_main,
        diagnostic_boost_per_state=diagnostic_boost,
        n_states=len(states),
        n_baselines=len([name for name in cfg.get("baselines", LADDER_BASELINES) if name in LADDER_BASELINES]),
        observed_throughput_per_second=observed_throughput,
        max_runtime_minutes=max_runtime_minutes,
    )
    if effective_main < requested_main:
        keep_ids = set(
            agents[
                (agents["sample_flag"] == "diagnostic_boost")
                | ((agents["sample_flag"] == "main") & (agents["sample_rank"] <= effective_main))
            ]["base_ces_id"].astype(str)
        )
        tasks = [task for task in tasks if task.base_ces_id in keep_ids]
    prompt_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    cache_hits = 0
    ollama_calls = 0

    def response_frame_from_predictions(rows: list[dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        pred_df = pd.DataFrame(rows)
        prompt_by_id_baseline = {
            (str(row["ces_id"]), row["baseline"]): {
                "prompt_id": row["prompt_id"],
                "cache_hit": row["cache_hit"],
                "latency_ms": row["latency_ms"],
            }
            for row in rows
        }
        response_rows: list[dict[str, Any]] = []
        for baseline, group in pred_df.groupby("baseline", sort=False):
            prompt_by_id = {
                str(row["ces_id"]): prompt_by_id_baseline[(str(row["ces_id"]), baseline)]
                for _, row in group.iterrows()
            }
            response_rows.extend(
                _response_rows(
                    run_id=cfg["run_id"],
                    baseline=baseline,
                    predictions=group[["ces_id", "raw_response", "model_name"]],
                    cohort=sampled,
                    prediction_scope="ablation_sample",
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
        return out.sort_values(["baseline", "base_ces_id"]).reset_index(drop=True)

    def write_checkpoint() -> None:
        if prompt_rows:
            write_table(pd.DataFrame(prompt_rows, columns=PROMPT_COLUMNS), paths.run_dir / "prompts.partial.parquet")
        if pred_rows:
            write_table(response_frame_from_predictions(pred_rows), paths.run_dir / "responses.partial.parquet")
        if runtime_rows:
            write_table(pd.DataFrame(runtime_rows), paths.run_dir / "runtime_log.partial.parquet")

    def record_result(result: tuple[AblationTask, str, bool, int | None, str | None]) -> None:
        nonlocal cache_hits, ollama_calls
        task, raw, cache_hit, latency_ms, transport_error = result
        cache_hits += int(cache_hit)
        ollama_calls += int(not cache_hit)
        parsed = parse_turnout_vote_json(raw)
        parse_status = "transport_error" if transport_error else parsed["parse_status"]
        raw_diag = raw_choice_diagnostics(raw)
        invalid_choice = bool(raw_diag["invalid_choice"] or parse_status == "invalid_choice")
        legacy_schema = bool(raw_diag["legacy_probability_schema"] or parsed.get("legacy_probability_schema"))
        prompt_rows.append(
            {
                "run_id": cfg["run_id"],
                "prompt_id": task.prompt_id,
                "agent_id": task.agent_id,
                "base_ces_id": task.base_ces_id,
                "baseline": task.baseline,
                "model_name": model_name,
                "prompt_hash": task.prompt_hash,
                "prompt_text": task.prompt_text,
                "memory_fact_ids_used": task.fact_ids,
                "memory_donor_ces_id": task.memory_donor_ces_id,
                "memory_shuffle_scope": task.memory_shuffle_scope,
                "memory_shuffle_fallback_reason": task.memory_shuffle_fallback_reason,
                "cache_hit": cache_hit,
                "latency_ms": latency_ms,
                "transport_error": transport_error,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
        pred_rows.append(
            {
                "ces_id": task.base_ces_id,
                "baseline": task.baseline,
                "raw_response": raw,
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
                "baseline": task.baseline,
                "prompt_id": task.prompt_id,
                "agent_id": task.agent_id,
                "base_ces_id": task.base_ces_id,
                "state_po": task.state_po,
                "sample_flag": task.sample_flag,
                "headline_sample": task.headline_sample,
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
    prompt_rows.sort(key=lambda row: (str(row["base_ces_id"]), BASELINE_ORDER.get(row["baseline"], 999)))
    pred_rows.sort(key=lambda row: (str(row["ces_id"]), BASELINE_ORDER.get(row["baseline"], 999)))
    runtime_rows.sort(key=lambda row: (str(row["base_ces_id"]), BASELINE_ORDER.get(row["baseline"], 999)))
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
        "requested_main_agents_per_state": requested_main,
        "effective_main_agents_per_state": effective_main,
        "diagnostic_boost_per_state": diagnostic_boost,
        "limit_reason": limit_reason,
        "timing_responses": len(timing_tasks),
        "timing_wall_seconds": timing_wall,
        "timing_throughput_per_second": observed_throughput,
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
    return responses, pd.DataFrame(prompt_rows, columns=PROMPT_COLUMNS), runtime_frame, metadata


def _add_sample_metadata(responses: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["base_ces_id", "state_po", "sample_rank", "sample_flag", "headline_sample"]
    meta = agents[meta_cols].copy()
    out = responses.merge(meta, on="base_ces_id", how="left", suffixes=("", "_sample"))
    if "state_po_sample" in out.columns:
        out["state_po"] = out["state_po_sample"]
        out = out.drop(columns=["state_po_sample"])
    return out


def vote_confidence_bins(responses: pd.DataFrame, targets: pd.DataFrame, run_id: str, n_bins: int = 10) -> pd.DataFrame:
    targets_wide = _target_wide(targets)
    merged = responses.merge(targets_wide, left_on="base_ces_id", right_on="ces_id", how="left")
    merged = merged[merged["president_vote_2024"].isin(VOTE_CLASSES)].copy()
    rows = []
    if merged.empty:
        return pd.DataFrame()
    probs = _combined_probabilities(merged)[VOTE_CLASSES]
    merged["pred_vote"] = probs.idxmax(axis=1)
    merged["pred_vote_confidence"] = probs.max(axis=1)
    merged["vote_correct"] = (merged["pred_vote"] == merged["president_vote_2024"]).astype(float)
    for (baseline, model_name), group in merged.groupby(["baseline", "model_name"], dropna=False):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        confidence = group["pred_vote_confidence"].fillna(0).astype(float)
        weights = group["sample_weight"].fillna(1.0).astype(float)
        for idx, (low, high) in enumerate(zip(bins[:-1], bins[1:], strict=False)):
            upper = confidence <= high if high == 1.0 else confidence < high
            mask = (confidence >= low) & upper
            if not mask.any():
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "baseline": baseline,
                    "model_name": model_name,
                    "bin": idx,
                    "bin_low": low,
                    "bin_high": high,
                    "n": int(mask.sum()),
                    "mean_confidence": float(np.average(confidence[mask], weights=weights[mask])),
                    "observed_accuracy": float(np.average(group.loc[mask, "vote_correct"], weights=weights[mask])),
                    "created_at": pd.Timestamp.now(tz="UTC"),
                }
            )
    return pd.DataFrame(rows)


def _state_prediction_rows(
    *,
    responses: pd.DataFrame,
    mit_truth: pd.DataFrame,
    run_id: str,
    states: list[str],
    sample_size: int,
) -> pd.DataFrame:
    truth = _truth_year(mit_truth, 2024, states).set_index("state_po")
    rows: list[dict[str, Any]] = []
    for (baseline, model_name, state), group in responses.groupby(["baseline", "model_name", "state_po"], dropna=False):
        if state not in truth.index:
            continue
        weights = group["sample_weight"].fillna(1.0).astype(float)
        turnout = group["turnout_probability"].fillna(0.0).astype(float).clip(0.0, 1.0)
        dem = float((weights * turnout * group["vote_prob_democrat"].fillna(0.0).astype(float)).sum())
        rep = float((weights * turnout * group["vote_prob_republican"].fillna(0.0).astype(float)).sum())
        pred_dem_2p = dem / (dem + rep) if dem + rep else 0.5
        pred_margin = 2 * pred_dem_2p - 1
        truth_row = truth.loc[state]
        true_margin = float(truth_row["margin_2p"])
        pred_winner = "democrat" if pred_margin > 0 else "republican" if pred_margin < 0 else "tie"
        true_winner = str(truth_row["winner"])
        rows.append(
            {
                "run_id": run_id,
                "sample_size": int(sample_size),
                "state_po": state,
                "baseline": baseline,
                "model_name": model_name,
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
    for (baseline, model_name), group in responses.groupby(["baseline", "model_name"], dropna=False):
        parse_status = group["parse_status"].fillna("").astype(str)
        invalid_choice = group.get("invalid_choice", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        forbidden_choice = group.get("forbidden_choice", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        legacy_schema = group.get("legacy_probability_schema", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        transport_error = group.get("transport_error", pd.Series(index=group.index, dtype=object)).notna()
        rows.append(
            {
                "run_id": run_id,
                "baseline": baseline,
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


def memory_placebo_diagnostics(prompts: pd.DataFrame, run_id: str) -> pd.DataFrame:
    placebo = prompts[prompts["baseline"].isin(PLACEBO_BASELINES)].copy()
    if placebo.empty:
        return pd.DataFrame()
    rows = []
    for _, row in placebo.iterrows():
        rows.append(
            {
                "run_id": run_id,
                "baseline": row["baseline"],
                "base_ces_id": row["base_ces_id"],
                "memory_donor_ces_id": row["memory_donor_ces_id"],
                "memory_shuffle_scope": row["memory_shuffle_scope"],
                "memory_shuffle_fallback_reason": row["memory_shuffle_fallback_reason"],
                "n_memory_facts": len(row["memory_fact_ids_used"] or []),
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    return pd.DataFrame(rows)


ABLATION_DELTA_COMPARISONS = [
    ("state_increment", "L1_demographic_only_llm", "L2_demographic_state_llm"),
    ("party_ideology_increment", "L2_demographic_state_llm", "L3_party_ideology_llm"),
    ("candidate_context_increment", "L3_party_ideology_llm", "L4_party_ideology_context_llm"),
    ("strict_memory_increment", "L4_party_ideology_context_llm", "L6_strict_memory_context_llm"),
    ("strict_context_increment", "L5_strict_memory_llm", "L6_strict_memory_context_llm"),
    ("poll_increment", "L6_strict_memory_context_llm", "L7_poll_informed_memory_context_llm"),
    ("oracle_gap_from_strict", "L6_strict_memory_context_llm", "L8_post_hoc_oracle_memory_context_llm"),
    ("oracle_gap_from_poll", "L7_poll_informed_memory_context_llm", "L8_post_hoc_oracle_memory_context_llm"),
    ("state_placebo_gap", "P1_memory_shuffled_within_state_llm", "L6_strict_memory_context_llm"),
    ("party_placebo_gap", "P2_memory_shuffled_within_party_llm", "L6_strict_memory_context_llm"),
]


def _metric_delta_rows(
    *,
    metric_frame: pd.DataFrame,
    metric_scope: str,
    run_id: str,
    metric_names: list[str],
) -> list[dict[str, Any]]:
    if metric_frame.empty:
        return []
    frame = metric_frame[metric_frame["metric_name"].isin(metric_names)].copy()
    if frame.empty:
        return []
    metric_lookup = frame.set_index(["baseline", "metric_name"])["metric_value"]
    rows: list[dict[str, Any]] = []
    for name, base, comp in ABLATION_DELTA_COMPARISONS:
        for metric_name in frame["metric_name"].dropna().unique():
            if (base, metric_name) not in metric_lookup.index or (comp, metric_name) not in metric_lookup.index:
                continue
            base_value = float(metric_lookup.loc[(base, metric_name)])
            comp_value = float(metric_lookup.loc[(comp, metric_name)])
            rows.append(
                {
                    "run_id": run_id,
                    "delta_name": name,
                    "metric_scope": metric_scope,
                    "baseline_from": base,
                    "baseline_to": comp,
                    "metric_name": metric_name,
                    "from_value": base_value,
                    "to_value": comp_value,
                    "metric_delta": comp_value - base_value,
                    "created_at": pd.Timestamp.now(tz="UTC"),
                }
            )
    return rows


def ablation_delta_rows(
    individual_metrics: pd.DataFrame,
    run_id: str,
    aggregate_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    weighted = individual_metrics[
        (individual_metrics["metric_scope"] == "individual_main")
        & (individual_metrics["weighted"].astype(bool))
        & individual_metrics["metric_name"].isin(["vote_accuracy", "vote_macro_f1", "turnout_accuracy_at_0_5", "turnout_ece"])
    ].copy()
    rows = _metric_delta_rows(
        metric_frame=weighted,
        metric_scope="individual_main",
        run_id=run_id,
        metric_names=["vote_accuracy", "vote_macro_f1", "turnout_accuracy_at_0_5", "turnout_ece"],
    )
    if aggregate_metrics is not None:
        rows.extend(
            _metric_delta_rows(
                metric_frame=aggregate_metrics,
                metric_scope="aggregate",
                run_id=run_id,
                metric_names=["dem_2p_rmse", "margin_mae", "margin_bias", "winner_accuracy", "winner_flip_count"],
            )
        )
    return pd.DataFrame(rows)


def _diagnostic_group_metrics(responses: pd.DataFrame, cohort: pd.DataFrame, targets: pd.DataFrame, run_id: str) -> pd.DataFrame:
    targets_wide = _target_wide(targets)
    merged = responses.merge(targets_wide, left_on="base_ces_id", right_on="ces_id", how="left")
    cohort_cols = ["ces_id", "race_ethnicity"]
    merged = merged.merge(cohort[cohort_cols], left_on="base_ces_id", right_on="ces_id", how="left", suffixes=("", "_cohort"))
    probs = _combined_probabilities(merged)[VOTE_CLASSES]
    merged["pred_vote"] = probs.idxmax(axis=1)
    rows = []
    groups = {
        "truth_other": merged["president_vote_2024"] == "other",
        "truth_not_vote": merged["president_vote_2024"] == "not_vote",
        "turnout_not_voted": merged["turnout_2024_self_report"] == "not_voted",
        "race_non_white_or_unknown": ~merged["race_ethnicity"].fillna("unknown").astype(str).isin(["white"]),
    }
    for group_name, mask in groups.items():
        part = merged[mask & merged["president_vote_2024"].isin(VOTE_CLASSES)].copy()
        if part.empty:
            continue
        for (baseline, model_name), group in part.groupby(["baseline", "model_name"], dropna=False):
            rows.append(
                {
                    "run_id": run_id,
                    "baseline": baseline,
                    "model_name": model_name,
                    "metric_scope": "diagnostic_group",
                    "metric_name": "vote_accuracy",
                    "metric_value": float((group["pred_vote"] == group["president_vote_2024"]).mean()),
                    "weighted": False,
                    "group_key": f"diagnostic={group_name}",
                    "n": int(len(group)),
                    "created_at": pd.Timestamp.now(tz="UTC"),
                }
            )
    return pd.DataFrame(rows)


def write_ablation_figures(
    *,
    run_dir: Path,
    individual_metrics: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    calibration_bins: pd.DataFrame,
    confidence_bins: pd.DataFrame,
    responses: pd.DataFrame,
    targets: pd.DataFrame,
    state_predictions: pd.DataFrame,
    deltas: pd.DataFrame,
    parse_diag: pd.DataFrame,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir = ensure_dir(run_dir / "figures")
    written: list[Path] = []
    metric_names = ["turnout_accuracy_at_0_5", "turnout_ece", "vote_accuracy", "vote_macro_f1"]
    core = individual_metrics[
        (individual_metrics["metric_scope"] == "individual_main")
        & (individual_metrics["weighted"].astype(bool))
        & individual_metrics["metric_name"].isin(metric_names)
    ].copy()
    if not core.empty:
        core["baseline"] = pd.Categorical(core["baseline"], categories=LADDER_BASELINES, ordered=True)
        pivot = core.pivot_table(index="baseline", columns="metric_name", values="metric_value", aggfunc="first", observed=False)
        fig, ax = plt.subplots(figsize=(11, max(5, 0.38 * len(pivot))))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title("Baseline Ladder: Weighted Main-Sample Metrics")
        written.append(_save_figure(fig, fig_dir / "ladder_metric_heatmap"))
        plt.close(fig)

        line = core[core["metric_name"].isin(["vote_accuracy", "vote_macro_f1"])].copy()
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=line, x="baseline", y="metric_value", hue="metric_name", marker="o", ax=ax)
        ax.tick_params(axis="x", rotation=35)
        ax.set_title("Vote Metrics Across Ladder")
        written.append(_save_figure(fig, fig_dir / "vote_metric_ladder_lines"))
        plt.close(fig)

    delta_plot = deltas[
        (deltas.get("metric_scope", pd.Series("individual_main", index=deltas.index)) == "individual_main")
        & deltas["metric_name"].isin(["vote_accuracy", "vote_macro_f1"])
    ].copy()
    if not delta_plot.empty:
        fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(delta_plot))))
        delta_plot["label"] = delta_plot["delta_name"] + " | " + delta_plot["metric_name"]
        delta_plot.sort_values("metric_delta").plot(kind="barh", x="label", y="metric_delta", ax=ax, legend=False)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Ablation Deltas")
        written.append(_save_figure(fig, fig_dir / "ablation_delta_waterfall"))
        plt.close(fig)

    placebo = core[core["baseline"].isin(["L6_strict_memory_context_llm", "P1_memory_shuffled_within_state_llm", "P2_memory_shuffled_within_party_llm"])].copy()
    if not placebo.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=placebo, x="metric_name", y="metric_value", hue="baseline", ax=ax)
        ax.tick_params(axis="x", rotation=20)
        ax.set_title("Strict Memory vs Shuffled Memory Placebos")
        written.append(_save_figure(fig, fig_dir / "memory_placebo_bars"))
        plt.close(fig)

    if not calibration_bins.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        for baseline, group in calibration_bins.groupby("baseline"):
            ax.plot(group["mean_predicted"], group["observed_rate"], marker="o", label=baseline)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("Mean predicted turnout")
        ax.set_ylabel("Observed turnout")
        ax.set_title("Turnout Calibration")
        ax.legend(fontsize=6)
        written.append(_save_figure(fig, fig_dir / "turnout_calibration_curves"))
        plt.close(fig)

    ece = core[core["metric_name"] == "turnout_ece"].copy()
    if not ece.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ece.sort_values("metric_value").plot(kind="bar", x="baseline", y="metric_value", ax=ax, legend=False)
        ax.tick_params(axis="x", rotation=35)
        ax.set_title("Turnout ECE by Ladder Condition")
        written.append(_save_figure(fig, fig_dir / "turnout_ece_bars"))
        plt.close(fig)

    if not confidence_bins.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        for baseline, group in confidence_bins.groupby("baseline"):
            ax.plot(group["mean_confidence"], group["observed_accuracy"], marker="o", label=baseline)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("Mean vote confidence")
        ax.set_ylabel("Observed vote accuracy")
        ax.set_title("Vote Confidence Calibration")
        ax.legend(fontsize=6)
        written.append(_save_figure(fig, fig_dir / "vote_confidence_calibration_curves"))
        plt.close(fig)

    targets_wide = _target_wide(targets)
    merged = responses.merge(targets_wide, left_on="base_ces_id", right_on="ces_id", how="left")
    for baseline in [
        "L1_demographic_only_llm",
        "L3_party_ideology_llm",
        "L6_strict_memory_context_llm",
        "L7_poll_informed_memory_context_llm",
        "L8_post_hoc_oracle_memory_context_llm",
        "P1_memory_shuffled_within_state_llm",
        "P2_memory_shuffled_within_party_llm",
    ]:
        part = merged[(merged["baseline"] == baseline) & (merged["headline_sample"].astype(bool)) & (merged["president_vote_2024"].isin(VOTE_CLASSES))].copy()
        if part.empty:
            continue
        probs = _combined_probabilities(part)[VOTE_CLASSES]
        confusion = pd.crosstab(part["president_vote_2024"], probs.idxmax(axis=1)).reindex(index=VOTE_CLASSES, columns=VOTE_CLASSES, fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion, annot=True, fmt=".0f", cmap="Blues", ax=ax)
        ax.set_title(f"Vote Confusion: {baseline}")
        written.append(_save_figure(fig, fig_dir / f"confusion_{baseline}"))
        plt.close(fig)

    if not state_predictions.empty:
        plot = state_predictions.sort_values(["state_po", "error"]).copy()
        fig, ax = plt.subplots(figsize=(12, max(6, 0.28 * len(plot))))
        y = np.arange(len(plot))
        ax.hlines(y=y, xmin=0, xmax=plot["error"], color="#9ca3af", linewidth=1)
        ax.scatter(plot["error"], y, c=np.where(plot["error"] >= 0, "#2563eb", "#dc2626"), s=24)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(plot["state_po"] + " | " + plot["baseline"])
        ax.set_xlabel("Margin error (predicted - true)")
        ax.set_title("Headline Main-Sample State Margin Error")
        written.append(_save_figure(fig, fig_dir / "aggregate_state_margin_error_lollipop"))
        plt.close(fig)

    if not parse_diag.empty:
        melt = parse_diag.melt(id_vars=["baseline"], value_vars=["parse_ok_rate", "cache_hit_rate"], var_name="diagnostic", value_name="rate")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.barplot(data=melt, x="baseline", y="rate", hue="diagnostic", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=35)
        ax.set_title("Parse and Cache Diagnostics")
        written.append(_save_figure(fig, fig_dir / "parse_fallback_diagnostics"))
        plt.close(fig)

    diag = subgroup_metrics[subgroup_metrics["metric_scope"] == "diagnostic_group"].copy()
    if not diag.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=diag, x="group_key", y="metric_value", hue="baseline", ax=ax)
        ax.tick_params(axis="x", rotation=25)
        ax.set_title("Minority, Nonvoter, and Third-Party Diagnostics")
        written.append(_save_figure(fig, fig_dir / "minority_nonvoter_third_party_panel"))
        plt.close(fig)

    return written


def write_ablation_report(
    *,
    run_dir: Path,
    cfg: dict[str, Any],
    sampled_agents: pd.DataFrame,
    responses: pd.DataFrame,
    individual_metrics: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    placebo_diag: pd.DataFrame,
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

    headline_metrics = individual_metrics[
        (individual_metrics["metric_scope"] == "individual_main")
        & (individual_metrics["weighted"].astype(bool))
        & individual_metrics["metric_name"].isin(["turnout_accuracy_at_0_5", "turnout_ece", "vote_accuracy", "vote_macro_f1"])
    ].sort_values(["baseline", "metric_name"])
    key_agg = aggregate_metrics[
        aggregate_metrics["metric_name"].isin(["dem_2p_rmse", "margin_mae", "margin_bias", "winner_accuracy", "winner_flip_count"])
    ].sort_values(["baseline", "metric_name"])
    figure_rows = pd.DataFrame({"figure": [str(path.relative_to(run_dir)) for path in figures]})
    sample_summary = sampled_agents.groupby(["state_po", "sample_flag"], dropna=False).size().reset_index(name="n")
    lines = [
        f"# CES Baseline Ladder / Ablation Benchmark: {cfg['run_id']}",
        "",
        "## Run Summary",
        f"- States: {', '.join(cfg.get('states', DEFAULT_STATES))}",
        f"- Sampled agents: {len(sampled_agents):,}; headline main agents: {int(sampled_agents['headline_sample'].sum()):,}",
        f"- LLM model: `{cfg.get('model', {}).get('model_name', 'qwen3.5:0.8b')}`",
        f"- LLM metadata: `{stable_json(llm_metadata)}`",
        "- This is a functionality ablation, not the main aggregate benchmark.",
        "- Headline individual and aggregate metrics use only `main` rows; diagnostic boost rows are for diagnostic panels.",
        "",
        "## Sample Summary",
        table(sample_summary, ["state_po", "sample_flag", "n"], head=40),
        "",
        "## Weighted Main-Sample Individual Metrics",
        table(headline_metrics, ["baseline", "model_name", "metric_name", "metric_value", "n"], head=160),
        "",
        "## Aggregate Metrics",
        table(key_agg, ["baseline", "model_name", "metric_name", "metric_value", "n_states"], head=120),
        "",
        "## Ablation Deltas",
        table(deltas, ["delta_name", "metric_scope", "metric_name", "baseline_from", "baseline_to", "metric_delta"], head=160),
        "",
        "## Memory Placebo Diagnostics",
        table(placebo_diag, ["baseline", "base_ces_id", "memory_donor_ces_id", "memory_shuffle_scope", "memory_shuffle_fallback_reason", "n_memory_facts"], head=80),
        "",
        "## Parse Diagnostics",
        table(parse_diag, ["baseline", "model_name", "n", "parse_ok_rate", "cache_hit_rate"], head=80),
        "",
        "## Interpretation Checks",
        "- If strict memory and shuffled memory are similar, respondent-specific memory is likely weakly used.",
        "- If shuffled-within-party remains strong, party identity may explain most individual vote choice differences.",
        "- If demographic-only aggregate looks strong, inspect world-knowledge/state-prior leakage and sample composition.",
        "- If strict and oracle are close, inspect leakage.",
        "",
        "## Figures",
        table(figure_rows, ["figure"], head=120),
        "",
    ]
    out = run_dir / "benchmark_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run_ces_ablation_benchmark(config_path: str | Path) -> dict[str, Path]:
    cfg = load_yaml(config_path)
    run_id = cfg["run_id"]
    run_dir = ensure_dir(cfg.get("paths", {}).get("run_dir", f"data/runs/{run_id}"))
    write_yaml(cfg, run_dir / "config_snapshot.yaml")
    started = time.time()
    paths = AggregateBenchmarkPaths(
        run_dir=run_dir,
        figures_dir=ensure_dir(run_dir / "figures"),
        cache_path=run_dir / "llm_cache.jsonl",
    )
    states = list(cfg.get("states", DEFAULT_STATES))
    seed = int(cfg.get("seed", 20260426))
    respondents = pd.read_parquet(cfg["paths"]["ces_respondents"])
    targets = pd.read_parquet(cfg["paths"]["ces_targets"])
    context = pd.read_parquet(cfg["paths"]["ces_context"])
    strict_memory_facts = pd.read_parquet(cfg["paths"]["ces_memory_facts_strict"])
    poll_memory_facts = pd.read_parquet(cfg["paths"]["ces_memory_facts_poll"])
    mit_truth = pd.read_parquet(cfg["paths"]["mit_state_truth"])
    question_path = cfg["paths"].get("question_set_parquet")
    if question_path:
        question = pd.read_parquet(question_path).iloc[0]
    else:
        from .questions import load_question_config

        question = load_question_config(cfg["paths"]["question_set"]).iloc[0]

    cohort = build_benchmark_cohort(respondents, seed, states)
    sampled = sample_ablation_agents(
        cohort,
        targets,
        states=states,
        main_agents_per_state=int(cfg.get("main_agents_per_state", 50)),
        diagnostic_boost_per_state=int(cfg.get("diagnostic_boost_per_state", 10)),
        seed=seed,
    )
    agents = _agents_from_cohort(run_id, sampled)
    agents = agents.merge(
        sampled[["ces_id", "sample_rank", "sample_flag", "headline_sample"]],
        left_on="base_ces_id",
        right_on="ces_id",
        how="left",
    ).drop(columns=["ces_id"])
    donor_maps = {
        "state": build_memory_donor_map(agents, scope="state", seed=seed),
        "party": build_memory_donor_map(agents, scope="party", seed=seed),
    }
    targets_wide = _target_wide(targets)
    responses, prompts, runtime_log, llm_metadata = _run_llm_ablation(
        cfg=cfg,
        paths=paths,
        sampled=sampled,
        agents=agents,
        question=question,
        strict_memory_facts=strict_memory_facts,
        poll_memory_facts=poll_memory_facts,
        context=context,
        targets_wide=targets_wide,
        donor_maps=donor_maps,
    )
    responses = _add_sample_metadata(responses, agents)
    active_ids = set(responses["base_ces_id"].astype(str))
    sampled_agents = sampled[sampled["ces_id"].astype(str).isin(active_ids)].copy()
    active_agents = agents[agents["base_ces_id"].astype(str).isin(active_ids)].copy()
    headline_responses = responses[responses["headline_sample"].astype(bool)].copy()
    individual_metrics = pd.DataFrame(
        [
            *benchmark_metric_rows(headline_responses, sampled_agents, targets, run_id, metric_scope="individual_main"),
            *benchmark_metric_rows(responses, sampled_agents, targets, run_id, metric_scope="individual_all"),
        ]
    )
    subgroup_metrics = compute_subgroup_metrics(responses, sampled_agents, targets, run_id)
    diagnostic_metrics = _diagnostic_group_metrics(responses, sampled_agents, targets, run_id)
    if not diagnostic_metrics.empty:
        subgroup_metrics = pd.concat([subgroup_metrics, diagnostic_metrics], ignore_index=True)
    calibration_bins = turnout_calibration_bins(headline_responses, targets, run_id)
    confidence_bins = pd.DataFrame()
    state_predictions = _state_prediction_rows(
        responses=headline_responses,
        mit_truth=mit_truth,
        run_id=run_id,
        states=states,
        sample_size=int(llm_metadata.get("effective_main_agents_per_state") or cfg.get("main_agents_per_state", 50)),
    )
    aggregate_metrics = pd.DataFrame(swing_aggregate_metric_rows(state_predictions, run_id))
    deltas = ablation_delta_rows(individual_metrics, run_id, aggregate_metrics)
    placebo_diag = memory_placebo_diagnostics(prompts, run_id)
    parse_diag = parse_diagnostics(responses, run_id)

    write_table(sampled_agents, run_dir / "sampled_agents.parquet")
    write_table(active_agents, run_dir / "agents.parquet")
    write_table(prompts, run_dir / "prompts.parquet")
    write_table(responses, run_dir / "responses.parquet")
    write_table(individual_metrics, run_dir / "individual_metrics.parquet")
    write_table(subgroup_metrics, run_dir / "subgroup_metrics.parquet")
    write_table(calibration_bins, run_dir / "calibration_bins.parquet")
    write_table(state_predictions, run_dir / "state_predictions.parquet")
    write_table(aggregate_metrics, run_dir / "aggregate_metrics.parquet")
    write_table(deltas, run_dir / "ablation_deltas.parquet")
    write_table(placebo_diag, run_dir / "memory_placebo_diagnostics.parquet")
    write_table(placebo_diag, run_dir / "memory_donor_map.parquet")
    write_table(parse_diag, run_dir / "parse_diagnostics.parquet")
    write_table(runtime_log, run_dir / "runtime_log.parquet")
    figures = write_ablation_figures(
        run_dir=run_dir,
        individual_metrics=individual_metrics,
        subgroup_metrics=subgroup_metrics,
        calibration_bins=calibration_bins,
        confidence_bins=confidence_bins,
        responses=responses,
        targets=targets,
        state_predictions=state_predictions,
        deltas=deltas,
        parse_diag=parse_diag,
    )
    report = write_ablation_report(
        run_dir=run_dir,
        cfg=cfg,
        sampled_agents=sampled_agents,
        responses=responses,
        individual_metrics=individual_metrics,
        aggregate_metrics=aggregate_metrics,
        deltas=deltas,
        placebo_diag=placebo_diag,
        parse_diag=parse_diag,
        figures=figures,
        llm_metadata=llm_metadata,
    )
    runtime = {
        "run_id": run_id,
        "git_commit": git_commit(),
        "states": states,
        "baselines": [name for name in cfg.get("baselines", LADDER_BASELINES) if name in LADDER_BASELINES],
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
        "agents": run_dir / "agents.parquet",
        "responses": run_dir / "responses.parquet",
        "prompts": run_dir / "prompts.parquet",
        "individual_metrics": run_dir / "individual_metrics.parquet",
        "subgroup_metrics": run_dir / "subgroup_metrics.parquet",
        "calibration_bins": run_dir / "calibration_bins.parquet",
        "state_predictions": run_dir / "state_predictions.parquet",
        "aggregate_metrics": run_dir / "aggregate_metrics.parquet",
        "ablation_deltas": run_dir / "ablation_deltas.parquet",
        "memory_placebo_diagnostics": run_dir / "memory_placebo_diagnostics.parquet",
        "memory_donor_map": run_dir / "memory_donor_map.parquet",
        "parse_diagnostics": run_dir / "parse_diagnostics.parquet",
        "runtime_log": run_dir / "runtime_log.parquet",
        "runtime": run_dir / "runtime.json",
        "report": report,
    }
