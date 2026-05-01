"""Swing-state aggregate CES election benchmark runner."""

from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .ces_benchmark import (
    CANDIDATE_CLASSES,
    PartyIdRuleBaseline,
    ResponseCalibrator,
    SklearnLogitBenchmarkBaseline,
    _agents_from_cohort,
    _canonical_raw,
    _git_commit,
    _gpu_peak_summary,
    _gpu_snapshot,
    _group_context,
    _group_memory_facts,
    _post_hoc_oracle_raw,
    _raw_choice_diagnostics,
    _response_rows,
    _target_wide,
    build_benchmark_cohort,
    crossfit_partitions,
)
from .ces_schema import CES_TURNOUT_VOTE_CHOICES, parse_turnout_vote_json
from .config import ModelConfig
from .io import ensure_dir, load_yaml, stable_json, write_json, write_table
from .llm import build_llm_client
from .transforms import stable_hash


DEFAULT_SWING_STATES = ["PA", "MI", "WI", "GA", "AZ", "NV", "NC"]
DEFAULT_SAMPLE_SIZES = [500, 1000, 2000]
LLM_BASELINES = ["survey_memory_llm_strict", "survey_memory_llm_poll_informed"]
RESPONDENT_BASELINES = [
    "party_id_baseline",
    "sklearn_logit_pre_only_crossfit",
    "survey_memory_llm_strict",
    "sklearn_logit_poll_informed",
    "survey_memory_llm_poll_informed",
    "ces_post_self_report_aggregate_oracle",
]
ALL_BASELINES = [
    "mit_2020_state_prior",
    "uniform_national_swing_from_2020",
    *RESPONDENT_BASELINES,
]
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
    "cache_hit",
    "latency_ms",
    "created_at",
]


@dataclass
class AggregateBenchmarkPaths:
    run_dir: Path
    figures_dir: Path
    cache_path: Path


def build_nested_state_sample(
    cohort: pd.DataFrame,
    *,
    states: list[str],
    sample_sizes: list[int],
    seed: int,
    weight_column: str = "sample_weight",
) -> pd.DataFrame:
    """Draw one deterministic weighted-without-replacement max sample per state."""

    max_requested = max(sample_sizes)
    work = cohort[cohort["state_po"].isin(states)].copy()
    parts = []
    for state in states:
        group = work[work["state_po"] == state].copy()
        if group.empty:
            continue
        n = min(max_requested, len(group))
        weights = group[weight_column].fillna(0).astype(float) if weight_column in group.columns else None
        if weights is not None and float(weights.sum()) <= 0:
            weights = None
        random_state = int(stable_hash("swing_nested_sample", seed, state, length=8), 16)
        rng = np.random.default_rng(random_state)
        if weights is None:
            selected_idx = rng.choice(len(group), size=n, replace=False)
        else:
            prob = weights.to_numpy(dtype=float)
            prob = prob / prob.sum()
            selected_idx = rng.choice(len(group), size=n, replace=False, p=prob)
        sampled = group.iloc[selected_idx].reset_index(drop=True)
        sampled["sample_rank"] = np.arange(1, len(sampled) + 1)
        sampled["max_requested_n_agents"] = max_requested
        sampled["max_effective_n_agents"] = len(sampled)
        parts.append(sampled)
    if not parts:
        return work.head(0).copy()
    state_order = {state: idx for idx, state in enumerate(states)}
    out = pd.concat(parts, ignore_index=True)
    out["_state_order"] = out["state_po"].map(state_order)
    return out.sort_values(["_state_order", "sample_rank"]).drop(columns=["_state_order"]).reset_index(drop=True)


def build_sample_membership(sampled: pd.DataFrame, agents: pd.DataFrame, sample_sizes: list[int]) -> pd.DataFrame:
    """Return long-form nested sample membership for every requested size."""

    work = sampled[["ces_id", "state_po", "sample_rank", "max_effective_n_agents"]].merge(
        agents[["agent_id", "base_ces_id"]],
        left_on="ces_id",
        right_on="base_ces_id",
        how="left",
        validate="one_to_one",
    )
    rows = []
    for size in sorted(sample_sizes):
        for state, group in work.groupby("state_po", sort=False):
            effective = min(size, int(group["max_effective_n_agents"].iloc[0]))
            included = group[group["sample_rank"] <= effective].copy()
            included["sample_size"] = size
            included["requested_n_agents"] = size
            included["effective_n_agents"] = effective
            rows.append(included)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)[
        [
            "sample_size",
            "requested_n_agents",
            "effective_n_agents",
            "state_po",
            "sample_rank",
            "agent_id",
            "base_ces_id",
        ]
    ]


def _truth_year(mit_truth: pd.DataFrame, year: int, states: list[str] | None = None) -> pd.DataFrame:
    truth = mit_truth.copy()
    if "year" in truth.columns:
        truth = truth[truth["year"] == year].copy()
    if "geo_level" in truth.columns:
        truth = truth[truth["geo_level"] == "state"].copy()
    if states is not None:
        truth = truth[truth["state_po"].isin(states)].copy()
    return truth


def _winner_from_margin(margin: float) -> str:
    if margin > 0:
        return "democrat"
    if margin < 0:
        return "republican"
    return "tie"


def _prediction_row(
    *,
    run_id: str,
    sample_size: int,
    state_po: str,
    baseline: str,
    model_name: str,
    pred_dem_2p: float,
    true_dem_2p: float,
    true_margin: float,
    true_winner: str,
    effective_n_agents: int | None,
    requested_n_agents: int | None = None,
    fallback_count: int = 0,
    response_count: int = 0,
) -> dict[str, Any]:
    pred_dem_2p = float(np.clip(pred_dem_2p, 0.0, 1.0))
    pred_margin = 2.0 * pred_dem_2p - 1.0
    pred_winner = _winner_from_margin(pred_margin)
    return {
        "run_id": run_id,
        "sample_size": int(sample_size),
        "requested_n_agents": requested_n_agents,
        "effective_n_agents": effective_n_agents,
        "state_po": state_po,
        "baseline": baseline,
        "model_name": model_name,
        "pred_dem_2p": pred_dem_2p,
        "true_dem_2p": float(true_dem_2p),
        "dem_2p_error": pred_dem_2p - float(true_dem_2p),
        "pred_margin": pred_margin,
        "true_margin": float(true_margin),
        "error": pred_margin - float(true_margin),
        "pred_winner": pred_winner,
        "true_winner": true_winner,
        "winner_correct": pred_winner == true_winner,
        "aggregation_fallback_used": bool(fallback_count > 0),
        "fallback_response_count": int(fallback_count),
        "response_count": int(response_count),
        "fallback_rate": float(fallback_count / response_count) if response_count else 0.0,
        "created_at": pd.Timestamp.now(tz="UTC"),
    }


def mit_2020_state_prior_predictions(
    mit_truth: pd.DataFrame,
    *,
    run_id: str,
    states: list[str],
    sample_sizes: list[int],
) -> pd.DataFrame:
    prior = _truth_year(mit_truth, 2020, states)
    truth_2024 = _truth_year(mit_truth, 2024, states)
    truth_by_state = truth_2024.set_index("state_po")
    rows = []
    for size in sample_sizes:
        for _, row in prior.iterrows():
            state = row["state_po"]
            truth = truth_by_state.loc[state]
            rows.append(
                _prediction_row(
                    run_id=run_id,
                    sample_size=size,
                    state_po=state,
                    baseline="mit_2020_state_prior",
                    model_name="mit_2020_state_prior_v1",
                    pred_dem_2p=float(row["dem_share_2p"]),
                    true_dem_2p=float(truth["dem_share_2p"]),
                    true_margin=float(truth["margin_2p"]),
                    true_winner=str(truth["winner"]),
                    requested_n_agents=size,
                    effective_n_agents=None,
                )
            )
    return pd.DataFrame(rows)


def _national_dem_2p(truth: pd.DataFrame) -> float:
    dem = float(truth["dem_votes"].sum())
    rep = float(truth["rep_votes"].sum())
    return dem / (dem + rep) if dem + rep else np.nan


def uniform_national_swing_from_2020_predictions(
    mit_truth: pd.DataFrame,
    *,
    run_id: str,
    states: list[str],
    sample_sizes: list[int],
) -> pd.DataFrame:
    truth_2020_all = _truth_year(mit_truth, 2020, None)
    truth_2024_all = _truth_year(mit_truth, 2024, None)
    national_swing = _national_dem_2p(truth_2024_all) - _national_dem_2p(truth_2020_all)
    prior = _truth_year(mit_truth, 2020, states)
    truth_2024 = _truth_year(mit_truth, 2024, states)
    truth_by_state = truth_2024.set_index("state_po")
    rows = []
    for size in sample_sizes:
        for _, row in prior.iterrows():
            state = row["state_po"]
            truth = truth_by_state.loc[state]
            rows.append(
                _prediction_row(
                    run_id=run_id,
                    sample_size=size,
                    state_po=state,
                    baseline="uniform_national_swing_from_2020",
                    model_name="realized_2024_national_swing_v1",
                    pred_dem_2p=float(row["dem_share_2p"]) + float(national_swing),
                    true_dem_2p=float(truth["dem_share_2p"]),
                    true_margin=float(truth["margin_2p"]),
                    true_winner=str(truth["winner"]),
                    requested_n_agents=size,
                    effective_n_agents=None,
                )
            )
    return pd.DataFrame(rows)


def _parsed_response_frame(responses: pd.DataFrame) -> pd.DataFrame:
    parsed_rows = []
    for _, row in responses.iterrows():
        parsed = parse_turnout_vote_json(row["raw_response"])
        parsed_rows.append(
            {
                "base_ces_id": str(row["base_ces_id"]),
                "baseline": row["baseline"],
                "model_name": row["model_name"],
                "parse_status": parsed["parse_status"],
                "turnout_probability": parsed["turnout_probability"],
                "vote_prob_democrat": parsed["vote_prob_democrat"],
                "vote_prob_republican": parsed["vote_prob_republican"],
                "vote_prob_other": parsed["vote_prob_other"],
                "vote_prob_undecided": parsed["vote_prob_undecided"],
                "most_likely_choice": parsed["most_likely_choice"],
                "confidence": parsed["confidence"],
            }
        )
    return pd.DataFrame(parsed_rows)


def _make_response_rows(
    *,
    run_id: str,
    baseline: str,
    predictions: pd.DataFrame,
    sampled: pd.DataFrame,
    agents: pd.DataFrame,
    is_llm: bool,
    prompt_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cohort = sampled.copy()
    if "_agent_ordinal" not in cohort.columns:
        cohort["_agent_ordinal"] = np.arange(1, len(cohort) + 1)
    rows = _response_rows(
        run_id=run_id,
        baseline=baseline,
        predictions=predictions,
        cohort=cohort,
        prediction_scope="max_sample_pool",
        is_llm=is_llm,
        prompt_by_id=prompt_by_id,
    )
    agent_by_ces = agents.set_index("base_ces_id", drop=False)
    sampled_by_ces = sampled.set_index("ces_id", drop=False)
    for row in rows:
        ces_id = str(row["base_ces_id"])
        if ces_id in agent_by_ces.index:
            row["agent_id"] = agent_by_ces.loc[ces_id, "agent_id"]
        if ces_id in sampled_by_ces.index:
            source = sampled_by_ces.loc[ces_id]
            row["state_po"] = source["state_po"]
            row["sample_rank"] = int(source["sample_rank"])
            row["max_effective_n_agents"] = int(source["max_effective_n_agents"])
        row["aggregation_fallback_used"] = False
        row["fallback_baseline"] = None
        for col in [
            "turnout_probability",
            "vote_prob_democrat",
            "vote_prob_republican",
            "vote_prob_other",
            "vote_prob_undecided",
        ]:
            row[f"agg_{col}"] = row[col]
    return rows


def _run_party_baseline(run_id: str, sampled: pd.DataFrame, agents: pd.DataFrame) -> list[dict[str, Any]]:
    baseline = PartyIdRuleBaseline()
    baseline.fit(set(), sampled, pd.DataFrame(), pd.DataFrame())
    preds = baseline.predict(set(sampled["ces_id"].astype(str)), sampled)
    return _make_response_rows(
        run_id=run_id,
        baseline="party_id_baseline",
        predictions=preds,
        sampled=sampled,
        agents=agents,
        is_llm=False,
    )


def _run_oracle_baseline(
    *,
    run_id: str,
    sampled: pd.DataFrame,
    agents: pd.DataFrame,
    targets_wide: pd.DataFrame,
) -> list[dict[str, Any]]:
    agent_by_ces = agents.set_index("base_ces_id", drop=False)
    preds = []
    for ces_id in sampled["ces_id"].astype(str):
        raw = _post_hoc_oracle_raw(agent_by_ces.loc[ces_id], targets_wide)
        preds.append({"ces_id": ces_id, "raw_response": raw, "model_name": "ces_post_self_report_oracle_v1"})
    return _make_response_rows(
        run_id=run_id,
        baseline="ces_post_self_report_aggregate_oracle",
        predictions=pd.DataFrame(preds),
        sampled=sampled,
        agents=agents,
        is_llm=False,
    )


def _run_crossfit_sklearn_baseline(
    *,
    run_id: str,
    baseline_name: str,
    feature_mode: str,
    cohort: pd.DataFrame,
    sampled: pd.DataFrame,
    agents: pd.DataFrame,
    answers: pd.DataFrame,
    targets_wide: pd.DataFrame,
    feature_cache: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    sample_ids = set(sampled["ces_id"].astype(str))
    for fold in range(5):
        train_ids, dev_ids, eval_ids = crossfit_partitions(cohort, fold)
        eval_ids = eval_ids & sample_ids
        if not eval_ids:
            continue
        baseline = SklearnLogitBenchmarkBaseline(baseline_name, feature_mode, feature_cache)
        baseline.fit(train_ids, cohort, answers, targets_wide)
        dev_predictions = baseline.predict(dev_ids, cohort)
        calibrator = ResponseCalibrator()
        calibrator.fit(dev_predictions, targets_wide)
        notes.extend([f"{baseline_name}:fold{fold}:{note}" for note in calibrator.notes])
        fold_predictions = calibrator.apply(baseline.predict(eval_ids, cohort))
        rows.extend(
            _make_response_rows(
                run_id=run_id,
                baseline=baseline_name,
                predictions=fold_predictions,
                sampled=sampled,
                agents=agents,
                is_llm=False,
            )
        )
    return rows, notes


class AggregateLlmCache:
    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()
        self.rows: dict[str, dict[str, Any]] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    self.rows[row["cache_key"]] = row

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self.rows.get(key)

    def set(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.rows[row["cache_key"]] = row
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")


def llm_cache_key(
    *,
    model_name: str,
    baseline: str,
    prompt_hash: str,
    temperature: float,
    max_tokens: int,
    response_format: str,
) -> str:
    return stable_hash(model_name, baseline, prompt_hash, temperature, max_tokens, response_format, length=40)


def complete_llm_task_with_cache(
    *,
    client: Any,
    cache: AggregateLlmCache,
    cache_key: str,
    run_id: str,
    model_name: str,
    baseline: str,
    prompt_hash: str,
    prompt_text: str,
) -> tuple[str, bool, int | None]:
    cached = cache.get(cache_key)
    if cached is not None:
        return str(cached["raw_response"]), True, cached.get("latency_ms")
    started = time.time()
    raw = client.complete(prompt_text, CES_TURNOUT_VOTE_CHOICES)
    latency_ms = int((time.time() - started) * 1000)
    cache.set(
        {
            "cache_key": cache_key,
            "run_id": run_id,
            "model_name": model_name,
            "baseline": baseline,
            "prompt_hash": prompt_hash,
            "raw_response": raw,
            "latency_ms": latency_ms,
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }
    )
    return raw, False, latency_ms


def _fast_ces_prompt(
    agent: pd.Series,
    question: pd.Series,
    *,
    memory_facts: dict[str, pd.DataFrame],
    context: dict[str, list[dict[str, Any]]],
    fact_roles: list[str],
    memory_section_title: str,
    max_memory_facts: int,
) -> tuple[str, list[str]]:
    base_ces_id = str(agent.get("base_ces_id") or agent.get("source_respondent_id"))
    facts = memory_facts.get(base_ces_id)
    if facts is None or facts.empty:
        selected = facts.head(0) if facts is not None else pd.DataFrame()
    else:
        selected = facts
        if "fact_role" in selected.columns:
            selected = selected[selected["fact_role"].fillna("safe_pre").astype(str).isin(set(fact_roles))]
        if "fact_priority" in selected.columns:
            selected = selected.sort_values(["fact_priority", "source_variable"], ascending=[False, True])
        selected = selected.head(max_memory_facts)
    fact_texts = selected["fact_text"].astype(str).tolist() if selected is not None and not selected.empty else []
    fact_ids = selected["memory_fact_id"].astype(str).tolist() if selected is not None and not selected.empty else []
    candidates = list(context.get(base_ces_id, [])) or [
        {"candidate_party": "Democratic", "candidate_name": "Kamala Harris"},
        {"candidate_party": "Republican", "candidate_name": "Donald Trump"},
    ]
    lines = [
        "You are simulating how a specific U.S. eligible voter would behave in the 2024 general election.",
        "Answer as this voter would behave, not as a political analyst.",
        "",
        "Voter profile:",
        f"- State: {agent.get('state_po') or 'unknown'}",
        f"- Age group: {agent.get('age_group') or 'unknown'}",
        f"- Gender: {agent.get('gender') or 'unknown'}",
        f"- Race/ethnicity: {agent.get('race_ethnicity') or 'unknown'}",
        f"- Education: {agent.get('education_binary') or 'unknown'}",
        f"- Party identification: {agent.get('party_id_3') or 'unknown'}",
        f"- 7-point party ID: {agent.get('party_id_7') or 'unknown'}",
        f"- Ideology: {agent.get('ideology_3') or 'unknown'}",
        "",
    ]
    if fact_texts:
        lines.append(f"{memory_section_title}:")
        lines.extend(f"- {fact}" for fact in fact_texts)
        lines.append("")
    lines.extend(["Election context:", "- Office: President"])
    for candidate in candidates:
        lines.append(f"- {candidate.get('candidate_party', 'Unknown')} candidate: {candidate.get('candidate_name', 'unknown')}")
    lines.extend(
        [
            "",
            "Task:",
            "Choose the single election behavior this voter would most likely take.",
            "",
            "Allowed choices:",
            "- not_vote: does not vote",
            "- democrat: votes for the Democratic candidate",
            "- republican: votes for the Republican candidate",
            "",
            "Return JSON only with this schema:",
            '{"choice": "not_vote|democrat|republican"}',
        ]
    )
    _ = question
    return "\n".join(lines), fact_ids


def _llm_prompt(
    *,
    baseline: str,
    agent: pd.Series,
    question: pd.Series,
    strict_memory: dict[str, pd.DataFrame],
    poll_memory: dict[str, pd.DataFrame],
    context: dict[str, list[dict[str, Any]]],
    max_memory_facts: int,
) -> tuple[str, list[str]]:
    if baseline == "survey_memory_llm_strict":
        return _fast_ces_prompt(
            agent,
            question,
            memory_facts=strict_memory,
            context=context,
            fact_roles=["safe_pre"],
            memory_section_title="Strict pre-election survey-derived background facts",
            max_memory_facts=max_memory_facts,
        )
    if baseline == "survey_memory_llm_poll_informed":
        return _fast_ces_prompt(
            agent,
            question,
            memory_facts=poll_memory,
            context=context,
            fact_roles=["safe_pre", "poll_prior"],
            memory_section_title="Survey-derived background facts, including poll-prior facts",
            max_memory_facts=max_memory_facts,
        )
    raise ValueError(f"Unknown LLM baseline: {baseline}")


def _iter_llm_tasks(
    *,
    cfg: dict[str, Any],
    agents: pd.DataFrame,
    question: pd.Series,
    strict_memory: dict[str, pd.DataFrame],
    poll_memory: dict[str, pd.DataFrame],
    context_by_id: dict[str, list[dict[str, Any]]],
    model_name: str,
    model_cfg: ModelConfig,
) -> Iterator[dict[str, Any]]:
    max_memory_facts = int(cfg.get("memory", {}).get("max_memory_facts", 24))
    baseline_names = cfg.get("baselines", {}).get("llm", LLM_BASELINES)
    ordered_agents = agents.sort_values(["sample_rank", "state_po", "base_ces_id"]).copy()
    for _, agent in ordered_agents.iterrows():
        for baseline in baseline_names:
            if baseline not in LLM_BASELINES:
                continue
            prompt_text, fact_ids = _llm_prompt(
                baseline=baseline,
                agent=agent,
                question=question,
                strict_memory=strict_memory,
                poll_memory=poll_memory,
                context=context_by_id,
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
            yield {
                "baseline": baseline,
                "agent_id": agent["agent_id"],
                "base_ces_id": str(agent["base_ces_id"]),
                "state_po": agent["state_po"],
                "sample_rank": int(agent["sample_rank"]),
                "prompt_id": prompt_id,
                "prompt_hash": prompt_hash,
                "prompt_text": prompt_text,
                "memory_fact_ids_used": fact_ids,
                "cache_key": cache_key,
            }


def _build_llm_tasks(**kwargs: Any) -> list[dict[str, Any]]:
    return list(_iter_llm_tasks(**kwargs))


def _run_llm_predictions(
    *,
    cfg: dict[str, Any],
    paths: AggregateBenchmarkPaths,
    sampled: pd.DataFrame,
    agents: pd.DataFrame,
    question: pd.Series,
    context: pd.DataFrame,
    strict_memory_facts: pd.DataFrame,
    poll_memory_facts: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    model_cfg = ModelConfig.model_validate(cfg.get("model", {}))
    client = build_llm_client(model_cfg)
    model_name = getattr(client, "model_name", model_cfg.model_name)
    cache = AggregateLlmCache(paths.cache_path)
    llm_cfg = cfg.get("llm", {})
    timing_responses = int(llm_cfg.get("timing_responses", 50))
    max_runtime_hours = float(llm_cfg.get("max_runtime_hours", 4.0))
    min_sample_size = int(llm_cfg.get("min_sample_size", min(cfg.get("sample_sizes", DEFAULT_SAMPLE_SIZES))))
    workers = max(1, int(llm_cfg.get("workers", 1)))
    checkpoint_every = max(1, int(llm_cfg.get("checkpoint_every", 25)))
    gpu_sample_every = max(1, int(llm_cfg.get("gpu_sample_every", checkpoint_every)))
    baseline_names = [name for name in cfg.get("baselines", {}).get("llm", LLM_BASELINES) if name in LLM_BASELINES]
    if not baseline_names:
        return [], pd.DataFrame(columns=PROMPT_COLUMNS), [], {
            "llm_enabled": False,
            "workers": workers,
            "n_full_tasks": 0,
            "n_selected_tasks": 0,
        }
    ordered_agents = agents.sort_values(["sample_rank", "state_po", "base_ces_id"]).copy()
    timing_agent_count = min(len(ordered_agents), max(1, int(np.ceil(timing_responses / max(1, len(baseline_names))))))
    strict_memory = _group_memory_facts(strict_memory_facts)
    poll_memory = _group_memory_facts(poll_memory_facts)
    context_by_id = _group_context(context)
    llm_started = time.time()
    gpu_snapshots = [_gpu_snapshot("llm_start")]
    timing_tasks = _build_llm_tasks(
        cfg=cfg,
        agents=ordered_agents.head(timing_agent_count),
        question=question,
        strict_memory=strict_memory,
        poll_memory=poll_memory,
        context_by_id=context_by_id,
        model_name=model_name,
        model_cfg=model_cfg,
    )

    result_by_key: dict[str, dict[str, Any]] = {}
    runtime_rows: list[dict[str, Any]] = []
    observed_latencies: list[float] = []
    uncached_timing = 0

    def finish_task(task: dict[str, Any]) -> tuple[dict[str, Any], str, bool, int | None, str | None]:
        result = result_by_key.get(task["cache_key"])
        if result is not None:
            return task, str(result["raw"]), bool(result["cache_hit"]), result.get("latency_ms"), result.get("transport_error")
        started = time.time()
        try:
            raw, cache_hit, latency_ms = complete_llm_task_with_cache(
                client=client,
                cache=cache,
                cache_key=task["cache_key"],
                run_id=cfg["run_id"],
                model_name=model_name,
                baseline=task["baseline"],
                prompt_hash=task["prompt_hash"],
                prompt_text=task["prompt_text"],
            )
            return task, raw, cache_hit, latency_ms, None
        except Exception as exc:
            latency_ms = int((time.time() - started) * 1000)
            return task, "", False, latency_ms, f"{type(exc).__name__}: {exc}"

    def runtime_row(
        *,
        event: str,
        task: dict[str, Any],
        raw: str,
        cache_hit: bool,
        latency_ms: int | None,
        transport_error: str | None,
    ) -> dict[str, Any]:
        parsed = parse_turnout_vote_json(raw)
        parse_status = "transport_error" if transport_error else parsed["parse_status"]
        raw_diag = _raw_choice_diagnostics(raw)
        return {
            "run_id": cfg["run_id"],
            "event": event,
            "prompt_id": task.get("prompt_id"),
            "agent_id": task.get("agent_id"),
            "baseline": task["baseline"],
            "model_name": model_name,
            "base_ces_id": task["base_ces_id"],
            "state_po": task.get("state_po"),
            "sample_rank": task.get("sample_rank"),
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

    for task in timing_tasks:
        if uncached_timing >= timing_responses:
            break
        cached_before = cache.get(task["cache_key"]) is not None
        task, raw, cache_hit, latency_ms, transport_error = finish_task(task)
        if not cached_before:
            uncached_timing += 1
        if latency_ms is not None:
            observed_latencies.append(float(latency_ms) / 1000.0)
        result_by_key[task["cache_key"]] = {
            "raw": raw,
            "cache_hit": cache_hit,
            "latency_ms": latency_ms,
            "transport_error": transport_error,
        }
        runtime_rows.append(
            runtime_row(
                event="llm_timing",
                task=task,
                raw=raw,
                cache_hit=cache_hit,
                latency_ms=latency_ms,
                transport_error=transport_error,
            )
        )

    median_latency = statistics.median(observed_latencies) if observed_latencies else 0.0
    full_task_count = len(agents) * len(baseline_names)
    projected_full_hours = (median_latency * full_task_count / max(1, workers) / 3600.0) if median_latency else 0.0
    max_sample_size = max(cfg.get("sample_sizes", DEFAULT_SAMPLE_SIZES))
    configured_max_sample_size = llm_cfg.get("max_sample_size")
    if configured_max_sample_size is not None:
        configured_max_sample_size = max(1, min(max_sample_size, int(configured_max_sample_size)))
    limit_reasons: list[str] = []
    if projected_full_hours and projected_full_hours > max_runtime_hours:
        effective_llm_sample_size = min_sample_size
        limit_reasons.append("projected_runtime")
    else:
        effective_llm_sample_size = max_sample_size
    if configured_max_sample_size is not None and configured_max_sample_size < effective_llm_sample_size:
        effective_llm_sample_size = configured_max_sample_size
        limit_reasons.append("configured_max_sample_size")
    pilot_limited = effective_llm_sample_size < max_sample_size

    selected_agents = agents[agents["sample_rank"] <= effective_llm_sample_size].copy()
    selected_task_count = len(selected_agents) * len(baseline_names)
    selected_tasks = list(_iter_llm_tasks(
        cfg=cfg,
        agents=selected_agents,
        question=question,
        strict_memory=strict_memory,
        poll_memory=poll_memory,
        context_by_id=context_by_id,
        model_name=model_name,
        model_cfg=model_cfg,
    ))
    prompt_rows: list[dict[str, Any]] = []
    pred_rows = []
    cache_hits = 0
    ollama_calls = 0
    processed = 0

    def partial_response_rows() -> list[dict[str, Any]]:
        if not pred_rows:
            return []
        rows: list[dict[str, Any]] = []
        pred_df = pd.DataFrame(pred_rows)
        for baseline, group in pred_df.groupby("baseline", sort=False):
            prompt_by_id = {
                str(row["ces_id"]): {
                    "prompt_id": row["prompt_id"],
                    "cache_hit": bool(row["cache_hit"]),
                    "latency_ms": row["latency_ms"],
                }
                for _, row in group.iterrows()
            }
            rows.extend(
                _make_response_rows(
                    run_id=cfg["run_id"],
                    baseline=baseline,
                    predictions=group[["ces_id", "raw_response", "model_name"]],
                    sampled=sampled,
                    agents=agents,
                    is_llm=True,
                    prompt_by_id=prompt_by_id,
                )
            )
        return rows

    def write_checkpoint() -> None:
        if prompt_rows:
            write_table(pd.DataFrame(prompt_rows), paths.run_dir / "prompts.partial.parquet")
        partial_rows = partial_response_rows()
        if partial_rows:
            write_table(pd.DataFrame(partial_rows), paths.run_dir / "responses.partial.parquet")
        if runtime_rows:
            write_table(pd.DataFrame(runtime_rows), paths.run_dir / "runtime_log.partial.parquet")

    def record_completed(task: dict[str, Any], raw: str, cache_hit: bool, latency_ms: int | None, transport_error: str | None) -> None:
        nonlocal cache_hits, ollama_calls, processed
        cache_hits += int(cache_hit)
        ollama_calls += int(not cache_hit)
        prompt_rows.append(
            {
                "run_id": cfg["run_id"],
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
        pred_rows.append(
            {
                "ces_id": task["base_ces_id"],
                "baseline": task["baseline"],
                "raw_response": raw,
                "model_name": model_name,
                "prompt_id": task["prompt_id"],
                "cache_hit": cache_hit,
                "latency_ms": latency_ms,
            }
        )
        runtime_rows.append(
            runtime_row(
                event="llm_response",
                task=task,
                raw=raw,
                cache_hit=cache_hit,
                latency_ms=latency_ms,
                transport_error=transport_error,
            )
        )
        processed += 1
        if processed % gpu_sample_every == 0:
            gpu_snapshots.append(_gpu_snapshot(f"llm_after_response_{processed}"))
        if processed % checkpoint_every == 0:
            write_checkpoint()

    if workers == 1:
        for task in selected_tasks:
            record_completed(*finish_task(task))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task = {executor.submit(finish_task, task): task for task in selected_tasks}
            for future in as_completed(future_to_task):
                record_completed(*future.result())
    gpu_snapshots.append(_gpu_snapshot("llm_end"))
    write_checkpoint()
    sort_key = lambda row: (int(row.get("sample_rank", 10**9) or 10**9), str(row.get("state_po", "")), str(row.get("base_ces_id", "")), str(row.get("baseline", "")))
    prompt_rows.sort(key=sort_key)
    pred_rows.sort(key=lambda row: (str(row.get("baseline", "")), str(row.get("ces_id", ""))))
    runtime_rows.sort(key=lambda row: (str(row.get("event", "")), int(row.get("sample_rank", 10**9) or 10**9), str(row.get("state_po", "")), str(row.get("base_ces_id", "")), str(row.get("baseline", ""))))

    response_rows: list[dict[str, Any]] = []
    if pred_rows:
        pred_df = pd.DataFrame(pred_rows)
        for baseline, group in pred_df.groupby("baseline", sort=False):
            prompt_by_id = {
                str(row["ces_id"]): {
                    "prompt_id": row["prompt_id"],
                    "cache_hit": bool(row["cache_hit"]),
                    "latency_ms": row["latency_ms"],
                }
                for _, row in group.iterrows()
            }
            response_rows.extend(
                _make_response_rows(
                    run_id=cfg["run_id"],
                    baseline=baseline,
                    predictions=group[["ces_id", "raw_response", "model_name"]],
                    sampled=sampled,
                    agents=agents,
                    is_llm=True,
                    prompt_by_id=prompt_by_id,
                )
            )
    llm_runtime_seconds = time.time() - llm_started
    runtime_df = pd.DataFrame(runtime_rows)
    response_runtime = runtime_df[runtime_df.get("event", pd.Series(dtype=str)) == "llm_response"].copy() if not runtime_df.empty else pd.DataFrame()
    latency = pd.to_numeric(response_runtime.get("latency_ms", pd.Series(dtype=float)), errors="coerce").dropna()
    parse_status = response_runtime.get("parse_status", pd.Series(dtype=str)).fillna("").astype(str) if not response_runtime.empty else pd.Series(dtype=str)
    invalid_choice = response_runtime.get("invalid_choice", pd.Series(dtype=bool)).fillna(False).astype(bool) if not response_runtime.empty else pd.Series(dtype=bool)
    forbidden_choice = response_runtime.get("forbidden_choice", pd.Series(dtype=bool)).fillna(False).astype(bool) if not response_runtime.empty else pd.Series(dtype=bool)
    legacy_schema = response_runtime.get("legacy_probability_schema", pd.Series(dtype=bool)).fillna(False).astype(bool) if not response_runtime.empty else pd.Series(dtype=bool)
    transport_errors = response_runtime.get("transport_error", pd.Series(dtype=object)).notna() if not response_runtime.empty else pd.Series(dtype=bool)
    parse_ok_rate = float((parse_status == "ok").mean()) if len(parse_status) else None
    invalid_choice_rate = float(invalid_choice.mean()) if len(invalid_choice) else None
    forbidden_choice_rate = float(forbidden_choice.mean()) if len(forbidden_choice) else None
    legacy_schema_rate = float(legacy_schema.mean()) if len(legacy_schema) else None
    transport_error_rate = float(transport_errors.mean()) if len(transport_errors) else None
    metadata = {
        "llm_enabled": selected_task_count > 0,
        "pilot_limited": pilot_limited,
        "requested_max_sample_size": max_sample_size,
        "effective_llm_sample_size": effective_llm_sample_size,
        "configured_max_sample_size": configured_max_sample_size,
        "limit_reasons": limit_reasons,
        "n_full_tasks": full_task_count,
        "n_selected_tasks": selected_task_count,
        "median_latency_seconds": median_latency if observed_latencies else None,
        "projected_full_hours": projected_full_hours,
        "max_runtime_hours": max_runtime_hours,
        "workers": workers,
        "cache_hits": cache_hits,
        "ollama_calls": ollama_calls,
        "cache_hit_rate": cache_hits / selected_task_count if selected_task_count else None,
        "llm_runtime_seconds": llm_runtime_seconds,
        "p90_latency_seconds": float(latency.quantile(0.9) / 1000.0) if not latency.empty else None,
        "throughput_responses_per_second": float(selected_task_count / llm_runtime_seconds) if llm_runtime_seconds > 0 else None,
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
        "gpu_snapshots": gpu_snapshots,
        **_gpu_peak_summary(gpu_snapshots),
    }
    return response_rows, pd.DataFrame(prompt_rows, columns=PROMPT_COLUMNS), runtime_rows, metadata


def _apply_llm_fallback(responses: pd.DataFrame) -> pd.DataFrame:
    if responses.empty:
        return responses
    out = responses.copy()
    for col in [
        "turnout_probability",
        "vote_prob_democrat",
        "vote_prob_republican",
        "vote_prob_other",
        "vote_prob_undecided",
    ]:
        agg_col = f"agg_{col}"
        if agg_col not in out.columns:
            out[agg_col] = out[col]
    out["aggregation_fallback_used"] = out.get("aggregation_fallback_used", False).fillna(False).astype(bool)
    out["fallback_baseline"] = out.get("fallback_baseline", None)
    party = out[out["baseline"] == "party_id_baseline"].set_index("base_ces_id", drop=False)
    llm_mask = out["baseline"].isin(LLM_BASELINES) & (out["parse_status"] != "ok")
    for idx, row in out[llm_mask].iterrows():
        ces_id = row["base_ces_id"]
        if ces_id not in party.index:
            continue
        fallback = party.loc[ces_id]
        for col in [
            "turnout_probability",
            "vote_prob_democrat",
            "vote_prob_republican",
            "vote_prob_other",
            "vote_prob_undecided",
        ]:
            out.at[idx, f"agg_{col}"] = fallback[col]
        out.at[idx, "aggregation_fallback_used"] = True
        out.at[idx, "fallback_baseline"] = "party_id_baseline"
    return out


def _state_predictions_from_responses(
    *,
    responses: pd.DataFrame,
    sampled: pd.DataFrame,
    sample_membership: pd.DataFrame,
    mit_truth: pd.DataFrame,
    run_id: str,
    states: list[str],
    sample_sizes: list[int],
) -> pd.DataFrame:
    truth = _truth_year(mit_truth, 2024, states).set_index("state_po")
    sampled_cols = sampled[["ces_id", "state_po", "sample_rank", "sample_weight"]].copy()
    sampled_cols["base_ces_id"] = sampled_cols["ces_id"].astype(str)
    effective_by_size_state = sample_membership.groupby(["sample_size", "state_po"], dropna=False)[
        "effective_n_agents"
    ].first()
    merged = responses.merge(sampled_cols, on="base_ces_id", how="left", suffixes=("", "_sample"))
    rows = []
    for size in sorted(sample_sizes):
        part = merged[merged["sample_rank"] <= size].copy()
        for (baseline, model_name, state), group in part.groupby(["baseline", "model_name", "state_po_sample"], dropna=False):
            if state not in truth.index:
                continue
            expected_effective = int(effective_by_size_state.get((size, state), min(size, group["base_ces_id"].nunique())))
            observed_effective = int(group["base_ces_id"].nunique())
            if observed_effective < expected_effective:
                continue
            weights = group["sample_weight_sample"].fillna(1.0).astype(float)
            turnout = group["agg_turnout_probability"].fillna(0.0).astype(float).clip(0.0, 1.0)
            dem = float((weights * turnout * group["agg_vote_prob_democrat"].fillna(0.0).astype(float)).sum())
            rep = float((weights * turnout * group["agg_vote_prob_republican"].fillna(0.0).astype(float)).sum())
            pred_dem_2p = dem / (dem + rep) if dem + rep else 0.5
            truth_row = truth.loc[state]
            fallback_count = int(group["aggregation_fallback_used"].fillna(False).astype(bool).sum())
            rows.append(
                _prediction_row(
                    run_id=run_id,
                    sample_size=size,
                    state_po=state,
                    baseline=baseline,
                    model_name=model_name,
                    pred_dem_2p=pred_dem_2p,
                    true_dem_2p=float(truth_row["dem_share_2p"]),
                    true_margin=float(truth_row["margin_2p"]),
                    true_winner=str(truth_row["winner"]),
                    requested_n_agents=size,
                    effective_n_agents=observed_effective,
                    fallback_count=fallback_count,
                    response_count=int(len(group)),
                )
            )
    return pd.DataFrame(rows)


def swing_aggregate_metric_rows(state_predictions: pd.DataFrame, run_id: str) -> list[dict[str, Any]]:
    rows = []
    for (sample_size, baseline, model_name), group in state_predictions.groupby(
        ["sample_size", "baseline", "model_name"], dropna=False
    ):
        dem_error = group["pred_dem_2p"].astype(float) - group["true_dem_2p"].astype(float)
        margin_error = group["error"].astype(float)
        winner_correct = group["pred_winner"] == group["true_winner"]
        corr = np.nan
        if len(group) > 1 and group["pred_margin"].nunique() > 1 and group["true_margin"].nunique() > 1:
            corr = float(group["pred_margin"].corr(group["true_margin"]))
        metrics = {
            "dem_2p_rmse": float(np.sqrt((dem_error**2).mean())),
            "margin_mae": float(margin_error.abs().mean()),
            "margin_bias": float(margin_error.mean()),
            "winner_accuracy": float(winner_correct.mean()),
            "battleground_rmse": float(np.sqrt((margin_error**2).mean())),
            "correlation_with_true_margin": corr,
            "winner_flip_count": float((~winner_correct).sum()),
        }
        for metric_name, metric_value in metrics.items():
            rows.append(
                {
                    "run_id": run_id,
                    "sample_size": int(sample_size),
                    "baseline": baseline,
                    "model_name": model_name,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "n_states": int(len(group)),
                    "created_at": pd.Timestamp.now(tz="UTC"),
                }
            )
    return rows


def _parse_diagnostics(responses: pd.DataFrame, run_id: str) -> pd.DataFrame:
    if responses.empty:
        return pd.DataFrame()
    rows = []
    for (baseline, model_name), group in responses.groupby(["baseline", "model_name"], dropna=False):
        rows.append(
            {
                "run_id": run_id,
                "baseline": baseline,
                "model_name": model_name,
                "n": int(len(group)),
                "parse_ok_rate": float((group["parse_status"] == "ok").mean()),
                "fallback_rate": float(group["aggregation_fallback_used"].fillna(False).astype(bool).mean()),
                "cache_hit_rate": float(group["cache_hit"].fillna(False).astype(bool).mean())
                if "cache_hit" in group.columns and group["cache_hit"].notna().any()
                else None,
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    return pd.DataFrame(rows)


def _save_figure(fig: Any, out_base: Path) -> Path:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    return out_base.with_suffix(".png")


def write_aggregate_benchmark_figures(
    *,
    run_dir: Path,
    state_predictions: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    parse_diagnostics: pd.DataFrame,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir = ensure_dir(run_dir / "figures")
    written: list[Path] = []
    metric_subset = aggregate_metrics[
        aggregate_metrics["metric_name"].isin(["dem_2p_rmse", "margin_mae", "margin_bias", "winner_accuracy"])
    ].copy()
    if not metric_subset.empty:
        metric_subset["row"] = metric_subset["baseline"] + " @ " + metric_subset["sample_size"].astype(str)
        pivot = metric_subset.pivot_table(index="row", columns="metric_name", values="metric_value", aggfunc="first")
        fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(pivot))))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title("Swing-State Aggregate Metrics")
        written.append(_save_figure(fig, fig_dir / "baseline_metric_heatmap_by_sample_size"))
        plt.close(fig)

    common_size = int(state_predictions.groupby("baseline")["sample_size"].max().min()) if not state_predictions.empty else None
    if common_size is not None:
        common = state_predictions[state_predictions["sample_size"] == common_size].copy()
        if not common.empty:
            plot = common.sort_values(["state_po", "error"])
            fig, ax = plt.subplots(figsize=(12, max(5, 0.28 * len(plot))))
            y = np.arange(len(plot))
            ax.hlines(y=y, xmin=0, xmax=plot["error"], color="#9ca3af", linewidth=1)
            ax.scatter(plot["error"], y, c=np.where(plot["error"] >= 0, "#2563eb", "#dc2626"), s=28)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(plot["state_po"] + " | " + plot["baseline"])
            ax.set_xlabel("Margin error (predicted - true)")
            ax.set_title(f"State Margin Error, Common Sample Size {common_size}")
            written.append(_save_figure(fig, fig_dir / "state_margin_error_lollipop_common"))
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7, 6))
            sns.scatterplot(data=common, x="true_margin", y="pred_margin", hue="baseline", style="state_po", ax=ax)
            low = min(common["true_margin"].min(), common["pred_margin"].min())
            high = max(common["true_margin"].max(), common["pred_margin"].max())
            ax.plot([low, high], [low, high], color="black", linewidth=1)
            ax.set_title(f"Predicted vs True Margin, Sample Size {common_size}")
            written.append(_save_figure(fig, fig_dir / "predicted_vs_true_margin_scatter"))
            plt.close(fig)

            flips = common.pivot_table(index="state_po", columns="baseline", values="winner_correct", aggfunc="first")
            fig, ax = plt.subplots(figsize=(12, 4.5))
            sns.heatmap(flips.astype(float), annot=flips.replace({True: "OK", False: "Flip"}), fmt="", cmap="RdYlGn", cbar=False, ax=ax)
            ax.set_title(f"Winner Calls, Sample Size {common_size}")
            written.append(_save_figure(fig, fig_dir / "winner_flip_tile_chart"))
            plt.close(fig)

            delta_metric = aggregate_metrics[
                (aggregate_metrics["sample_size"] == common_size) & (aggregate_metrics["metric_name"] == "margin_mae")
            ].copy()
            if {"mit_2020_state_prior", "uniform_national_swing_from_2020"} <= set(delta_metric["baseline"]):
                a0 = float(delta_metric.loc[delta_metric["baseline"] == "mit_2020_state_prior", "metric_value"].iloc[0])
                a1 = float(
                    delta_metric.loc[
                        delta_metric["baseline"] == "uniform_national_swing_from_2020", "metric_value"
                    ].iloc[0]
                )
                delta_metric["delta_vs_2020_prior"] = delta_metric["metric_value"] - a0
                delta_metric["delta_vs_national_swing"] = delta_metric["metric_value"] - a1
                melt = delta_metric.melt(
                    id_vars=["baseline"],
                    value_vars=["delta_vs_2020_prior", "delta_vs_national_swing"],
                    var_name="reference",
                    value_name="margin_mae_delta",
                )
                fig, ax = plt.subplots(figsize=(12, 5))
                sns.barplot(data=melt, x="baseline", y="margin_mae_delta", hue="reference", ax=ax)
                ax.axhline(0, color="black", linewidth=0.8)
                ax.tick_params(axis="x", rotation=45)
                ax.set_title("Margin MAE Delta vs Aggregate References")
                written.append(_save_figure(fig, fig_dir / "a0_a1_delta_chart"))
                plt.close(fig)

    non_llm = state_predictions[
        (state_predictions["sample_size"] == max(state_predictions["sample_size"]))
        & (~state_predictions["baseline"].isin(LLM_BASELINES))
    ].copy()
    if not non_llm.empty:
        plot = non_llm.sort_values(["baseline", "error"])
        fig, ax = plt.subplots(figsize=(12, max(5, 0.25 * len(plot))))
        y = np.arange(len(plot))
        ax.hlines(y=y, xmin=0, xmax=plot["error"], color="#9ca3af", linewidth=1)
        ax.scatter(plot["error"], y, c=np.where(plot["error"] >= 0, "#2563eb", "#dc2626"), s=28)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(plot["state_po"] + " | " + plot["baseline"])
        ax.set_xlabel("Margin error (predicted - true)")
        ax.set_title("Non-LLM and Oracle State Margin Error at Max Sample Size")
        written.append(_save_figure(fig, fig_dir / "state_margin_error_lollipop_non_llm_max"))
        plt.close(fig)

    sensitivity = aggregate_metrics[aggregate_metrics["metric_name"].isin(["margin_mae", "dem_2p_rmse"])].copy()
    if not sensitivity.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=sensitivity, x="sample_size", y="metric_value", hue="baseline", style="metric_name", marker="o", ax=ax)
        ax.set_title("Sample-Size Sensitivity")
        written.append(_save_figure(fig, fig_dir / "sample_size_sensitivity"))
        plt.close(fig)

    if not parse_diagnostics.empty:
        diag = parse_diagnostics[parse_diagnostics["baseline"].isin(LLM_BASELINES)].copy()
        if not diag.empty:
            melt = diag.melt(
                id_vars=["baseline"],
                value_vars=["parse_ok_rate", "fallback_rate", "cache_hit_rate"],
                var_name="diagnostic",
                value_name="rate",
            )
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.barplot(data=melt, x="baseline", y="rate", hue="diagnostic", ax=ax)
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="x", rotation=20)
            ax.set_title("LLM Parse, Fallback, and Cache Diagnostics")
            written.append(_save_figure(fig, fig_dir / "parse_fallback_diagnostics"))
            plt.close(fig)

    return written


def write_aggregate_benchmark_report(
    *,
    run_dir: Path,
    cfg: dict[str, Any],
    sampled: pd.DataFrame,
    sample_membership: pd.DataFrame,
    state_predictions: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    parse_diagnostics: pd.DataFrame,
    figures: list[Path],
    llm_metadata: dict[str, Any],
    calibration_notes: list[str],
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

    sample_summary = sample_membership.groupby(["sample_size", "state_po"], dropna=False).agg(
        effective_n_agents=("base_ces_id", "nunique")
    ).reset_index()
    key_metrics = aggregate_metrics[
        aggregate_metrics["metric_name"].isin(["dem_2p_rmse", "margin_mae", "margin_bias", "winner_accuracy", "winner_flip_count"])
    ].sort_values(["sample_size", "baseline", "metric_name"])
    common_size = int(state_predictions.groupby("baseline")["sample_size"].max().min()) if not state_predictions.empty else None
    common_metrics = key_metrics[key_metrics["sample_size"] == common_size] if common_size is not None else key_metrics.head(0)
    figure_rows = pd.DataFrame({"figure": [str(path.relative_to(run_dir)) for path in figures]})
    lines = [
        f"# Swing-State Aggregate Election Benchmark: {cfg['run_id']}",
        "",
        "## Run Summary",
        f"- States: {', '.join(cfg.get('states', DEFAULT_SWING_STATES))}",
        f"- Sample sizes requested: {cfg.get('sample_sizes', DEFAULT_SAMPLE_SIZES)}",
        f"- Max sampled respondents: {len(sampled):,}",
        f"- LLM model: `{cfg.get('model', {}).get('model_name', 'qwen3.5:0.8b')}`",
        f"- LLM metadata: `{stable_json(llm_metadata)}`",
        "- A1 uses realized 2024 national two-party swing and is a retrospective reference, not a deployable forecast.",
        "- Winner accuracy is reported, but dem_2p_rmse, margin_mae, and margin_bias are the primary aggregate diagnostics.",
        "",
        "## Effective Sample Sizes",
        table(sample_summary, ["sample_size", "state_po", "effective_n_agents"], head=80),
        "",
        f"## Core Metrics at Common Sample Size {common_size}",
        table(common_metrics, ["sample_size", "baseline", "model_name", "metric_name", "metric_value", "n_states"], head=120),
        "",
        "## All Aggregate Metrics",
        table(key_metrics, ["sample_size", "baseline", "model_name", "metric_name", "metric_value", "n_states"], head=220),
        "",
        "## Core State Prediction Table",
        table(
            state_predictions.sort_values(["sample_size", "baseline", "state_po"]),
            [
                "sample_size",
                "state_po",
                "baseline",
                "pred_dem_2p",
                "true_dem_2p",
                "pred_margin",
                "true_margin",
                "error",
                "pred_winner",
                "true_winner",
                "effective_n_agents",
            ],
            head=220,
        ),
        "",
        "## Parse Diagnostics",
        table(parse_diagnostics, ["baseline", "model_name", "n", "parse_ok_rate", "fallback_rate", "cache_hit_rate"], head=80),
        "",
        "## Figures",
        table(figure_rows, ["figure"], head=80),
        "",
        "## Calibration Notes",
        "\n".join(f"- {note}" for note in calibration_notes[:200]) if calibration_notes else "- None.",
        "",
        "## Output Files",
        "- `sampled_agents.parquet`, `sample_membership.parquet`",
        "- `responses.parquet`, `prompts.parquet`",
        "- `state_predictions.parquet`, `aggregate_metrics.parquet`",
        "- `parse_diagnostics.parquet`, `runtime_log.parquet`",
        "- `runtime.json`, `llm_cache.jsonl`",
        "- `figures/*.png`, `figures/*.svg`",
        "",
    ]
    out = run_dir / "benchmark_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run_ces_aggregate_benchmark(config_path: str | Path) -> dict[str, Path]:
    run_started = time.time()
    cfg = load_yaml(config_path)
    run_id = cfg["run_id"]
    run_dir = ensure_dir(cfg.get("paths", {}).get("run_dir", f"data/runs/{run_id}"))
    paths = AggregateBenchmarkPaths(run_dir=run_dir, figures_dir=ensure_dir(run_dir / "figures"), cache_path=run_dir / "llm_cache.jsonl")
    states = list(cfg.get("states", DEFAULT_SWING_STATES))
    sample_sizes = sorted(int(size) for size in cfg.get("sample_sizes", DEFAULT_SAMPLE_SIZES))
    seed = int(cfg.get("seed", 20260426))
    gpu_snapshots = [_gpu_snapshot("run_start")]

    respondents = pd.read_parquet(cfg["paths"]["ces_respondents"])
    answers = pd.read_parquet(cfg["paths"]["ces_answers"])
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

    cohort_all = build_benchmark_cohort(respondents, seed, "all")
    sampled = build_nested_state_sample(
        cohort_all,
        states=states,
        sample_sizes=sample_sizes,
        seed=seed,
        weight_column="sample_weight",
    )
    agents = _agents_from_cohort(run_id, sampled)
    agents = agents.merge(sampled[["ces_id", "sample_rank", "max_effective_n_agents"]], left_on="base_ces_id", right_on="ces_id", how="left")
    agents = agents.drop(columns=["ces_id"])
    sample_membership = build_sample_membership(sampled, agents, sample_sizes)
    targets_wide = _target_wide(targets)

    response_rows: list[dict[str, Any]] = []
    calibration_notes: list[str] = []
    runtime_rows: list[dict[str, Any]] = []
    started = time.time()
    response_rows.extend(_run_party_baseline(run_id, sampled, agents))
    runtime_rows.append({"run_id": run_id, "event": "party_id_baseline_complete", "latency_ms": int((time.time() - started) * 1000), "created_at": pd.Timestamp.now(tz="UTC")})

    feature_cache: dict[str, pd.DataFrame] = {}
    for baseline_name, feature_mode in [
        ("sklearn_logit_pre_only_crossfit", "strict_pre"),
        ("sklearn_logit_poll_informed", "poll_informed"),
    ]:
        started = time.time()
        rows, notes = _run_crossfit_sklearn_baseline(
            run_id=run_id,
            baseline_name=baseline_name,
            feature_mode=feature_mode,
            cohort=cohort_all,
            sampled=sampled,
            agents=agents,
            answers=answers,
            targets_wide=targets_wide,
            feature_cache=feature_cache,
        )
        response_rows.extend(rows)
        calibration_notes.extend(notes)
        runtime_rows.append({"run_id": run_id, "event": f"{baseline_name}_complete", "latency_ms": int((time.time() - started) * 1000), "created_at": pd.Timestamp.now(tz="UTC")})

    response_rows.extend(_run_oracle_baseline(run_id=run_id, sampled=sampled, agents=agents, targets_wide=targets_wide))

    llm_rows, prompt_rows, llm_runtime_rows, llm_metadata = _run_llm_predictions(
        cfg=cfg,
        paths=paths,
        sampled=sampled,
        agents=agents,
        question=question,
        context=context,
        strict_memory_facts=strict_memory_facts,
        poll_memory_facts=poll_memory_facts,
    )
    response_rows.extend(llm_rows)
    runtime_rows.extend(llm_runtime_rows)

    responses = _apply_llm_fallback(pd.DataFrame(response_rows))
    prompts = prompt_rows
    prior_predictions = pd.concat(
        [
            mit_2020_state_prior_predictions(mit_truth, run_id=run_id, states=states, sample_sizes=sample_sizes),
            uniform_national_swing_from_2020_predictions(mit_truth, run_id=run_id, states=states, sample_sizes=sample_sizes),
        ],
        ignore_index=True,
    )
    respondent_predictions = _state_predictions_from_responses(
        responses=responses,
        sampled=sampled,
        sample_membership=sample_membership,
        mit_truth=mit_truth,
        run_id=run_id,
        states=states,
        sample_sizes=sample_sizes,
    )
    state_predictions = pd.concat([prior_predictions, respondent_predictions], ignore_index=True)
    aggregate_metrics = pd.DataFrame(swing_aggregate_metric_rows(state_predictions, run_id))
    parse_diagnostics = _parse_diagnostics(responses, run_id)
    runtime_log = pd.DataFrame(runtime_rows)

    write_table(sampled, run_dir / "sampled_agents.parquet")
    write_table(sample_membership, run_dir / "sample_membership.parquet")
    write_table(responses, run_dir / "responses.parquet")
    write_table(prompts, run_dir / "prompts.parquet")
    write_table(state_predictions, run_dir / "state_predictions.parquet")
    write_table(aggregate_metrics, run_dir / "aggregate_metrics.parquet")
    write_table(parse_diagnostics, run_dir / "parse_diagnostics.parquet")
    write_table(runtime_log, run_dir / "runtime_log.parquet")

    figures = write_aggregate_benchmark_figures(
        run_dir=run_dir,
        state_predictions=state_predictions,
        aggregate_metrics=aggregate_metrics,
        parse_diagnostics=parse_diagnostics,
    )
    report = write_aggregate_benchmark_report(
        run_dir=run_dir,
        cfg=cfg,
        sampled=sampled,
        sample_membership=sample_membership,
        state_predictions=state_predictions,
        aggregate_metrics=aggregate_metrics,
        parse_diagnostics=parse_diagnostics,
        figures=figures,
        llm_metadata=llm_metadata,
        calibration_notes=calibration_notes,
    )
    gpu_snapshots.extend(llm_metadata.get("gpu_snapshots", []))
    gpu_snapshots.append(_gpu_snapshot("run_end"))
    runtime_seconds = time.time() - run_started
    response_runtime = runtime_log[runtime_log.get("event", pd.Series(dtype=str)) == "llm_response"].copy() if not runtime_log.empty else pd.DataFrame()
    latency = pd.to_numeric(response_runtime.get("latency_ms", pd.Series(dtype=float)), errors="coerce").dropna()
    runtime = {
        "run_id": run_id,
        "git_commit": _git_commit(),
        "model_name": cfg.get("model", {}).get("model_name"),
        "provider": cfg.get("model", {}).get("provider"),
        "temperature": cfg.get("model", {}).get("temperature"),
        "max_tokens": cfg.get("model", {}).get("max_tokens"),
        "workers": llm_metadata.get("workers"),
        "states": states,
        "sample_sizes": sample_sizes,
        "effective_llm_sample_size": llm_metadata.get("effective_llm_sample_size"),
        "n_sampled_agents": int(len(sampled)),
        "n_llm_tasks": int(llm_metadata.get("n_selected_tasks", 0)),
        "runtime_seconds": runtime_seconds,
        "llm_runtime_seconds": llm_metadata.get("llm_runtime_seconds"),
        "median_latency_seconds": float(latency.median() / 1000.0) if not latency.empty else llm_metadata.get("median_latency_seconds"),
        "p90_latency_seconds": float(latency.quantile(0.9) / 1000.0) if not latency.empty else llm_metadata.get("p90_latency_seconds"),
        "throughput_responses_per_second": llm_metadata.get("throughput_responses_per_second"),
        "cache_hit_rate": llm_metadata.get("cache_hit_rate"),
        "ollama_calls": int(llm_metadata.get("ollama_calls", 0)),
        "parse_ok_rate": llm_metadata.get("parse_ok_rate"),
        "invalid_choice_rate": llm_metadata.get("invalid_choice_rate"),
        "forbidden_choice_rate": llm_metadata.get("forbidden_choice_rate"),
        "legacy_probability_schema_rate": llm_metadata.get("legacy_probability_schema_rate"),
        "transport_error_rate": llm_metadata.get("transport_error_rate"),
        "all_gates_passed": bool(llm_metadata.get("all_gates_passed", True)),
        **_gpu_peak_summary(gpu_snapshots),
        "gpu_snapshots": gpu_snapshots,
        "input_artifact_paths": {key: str(value) for key, value in cfg.get("paths", {}).items()},
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    write_json(runtime, run_dir / "runtime.json")
    return {
        "sampled_agents": run_dir / "sampled_agents.parquet",
        "sample_membership": run_dir / "sample_membership.parquet",
        "responses": run_dir / "responses.parquet",
        "prompts": run_dir / "prompts.parquet",
        "state_predictions": run_dir / "state_predictions.parquet",
        "aggregate_metrics": run_dir / "aggregate_metrics.parquet",
        "parse_diagnostics": run_dir / "parse_diagnostics.parquet",
        "runtime_log": run_dir / "runtime_log.parquet",
        "runtime": run_dir / "runtime.json",
        "report": report,
    }
