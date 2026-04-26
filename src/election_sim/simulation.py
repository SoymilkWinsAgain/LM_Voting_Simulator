"""End-to-end simulation orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .aggregation import write_aggregate_state_results
from .anes import build_anes, build_memory_cards
from .baselines import LLMBaseline, build_baselines
from .ces import build_ces_cells
from .config import RunConfig, load_cell_schema, load_run_config
from .evaluation import write_eval_metrics
from .io import ensure_dir, stable_json, write_table
from .llm import build_llm_client
from .mit import build_mit_results
from .population import agent_response_id, build_agents_from_frames
from .prompts import build_prompt, options_from_question, parse_json_answer
from .questions import load_question_config
from .report import write_eval_report
from .transforms import stable_hash


def _path(cfg: RunConfig, key: str) -> Path:
    if key not in cfg.paths:
        raise KeyError(f"Run config missing paths.{key}")
    return Path(cfg.paths[key])


def ensure_processed_inputs(cfg: RunConfig) -> dict[str, Path]:
    """Build processed fixture/intermediate files needed by a run."""

    processed_dir = ensure_dir(cfg.processed_dir)
    anes_dir = ensure_dir(processed_dir / "anes")
    ces_dir = ensure_dir(processed_dir / "ces")
    mit_dir = ensure_dir(processed_dir / "mit")

    anes_paths = build_anes(
        _path(cfg, "anes_config"),
        _path(cfg, "anes_profile_crosswalk"),
        _path(cfg, "anes_question_crosswalk"),
        anes_dir,
    )
    memory_paths = build_memory_cards(
        anes_paths["respondents"],
        anes_paths["answers"],
        _path(cfg, "anes_fact_templates"),
        cfg.anes_memory.get("memory_policy", "safe_survey_memory_v1"),
        anes_dir,
        max_facts=int(cfg.anes_memory.get("max_memory_facts", cfg.anes_memory.get("max_facts", 24))),
    )
    ces_paths = build_ces_cells(
        _path(cfg, "ces_config"),
        _path(cfg, "ces_crosswalk"),
        _path(cfg, "cell_schema"),
        ces_dir,
    )
    mit_path = build_mit_results(
        _path(cfg, "mit_config"),
        int(cfg.evaluation.get("mit_results_year", cfg.scenario.year)),
        mit_dir / "mit_election_results.parquet",
    )
    return {
        "anes_respondents": anes_paths["respondents"],
        "anes_answers": anes_paths["answers"],
        "anes_memory_facts": memory_paths["facts"],
        "anes_memory_cards": memory_paths["cards"],
        "question_bank": _path(cfg, "question_set") if "question_set" in cfg.paths else anes_paths["question_bank"],
        "ces_respondents": ces_paths["respondents"],
        "ces_cell_distribution": ces_paths["cell_distribution"],
        "mit_results": mit_path,
    }


def _prompt_record(
    *,
    cfg: RunConfig,
    agent: pd.Series,
    question: pd.Series,
    baseline: str,
    model_name: str,
    prompt_text: str,
    fact_ids: list[str],
) -> dict[str, Any]:
    prompt_hash = stable_hash(model_name, baseline, prompt_text, cfg.model.temperature, length=32)
    prompt_id = stable_hash(cfg.run_id, agent["agent_id"], question["question_id"], baseline, length=20)
    return {
        "run_id": cfg.run_id,
        "prompt_id": prompt_id,
        "agent_id": agent["agent_id"],
        "question_id": question["question_id"],
        "baseline": baseline,
        "model_name": model_name,
        "prompt_template": baseline,
        "prompt_version": "v1",
        "prompt_hash": prompt_hash,
        "prompt_text": prompt_text,
        "memory_fact_ids_used": fact_ids,
        "context_card_ids_used": [],
        "created_at": pd.Timestamp.now(tz="UTC"),
    }


def _response_record(
    *,
    cfg: RunConfig,
    prompt_id: str,
    agent: pd.Series,
    question: pd.Series,
    baseline: str,
    model_name: str,
    raw_response: str,
    answer: str | None,
    confidence: float | None,
    parse_status: str,
    probabilities: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": cfg.run_id,
        "response_id": agent_response_id(cfg.run_id, agent["agent_id"], question["question_id"], baseline),
        "prompt_id": prompt_id,
        "agent_id": agent["agent_id"],
        "base_anes_id": agent["base_anes_id"],
        "question_id": question["question_id"],
        "baseline": baseline,
        "model_name": model_name,
        "raw_response": raw_response,
        "parsed_answer_code": answer,
        "parsed_answer_label": answer,
        "confidence": confidence,
        "parse_status": parse_status,
        "repair_attempts": 0,
        "latency_ms": None,
        "cost_usd": None,
        "probabilities_json": stable_json(probabilities) if probabilities is not None else None,
        "created_at": pd.Timestamp.now(tz="UTC"),
    }


def run_simulation(run_config_path: str | Path) -> dict[str, Path]:
    cfg = load_run_config(run_config_path)
    run_dir = ensure_dir(cfg.run_dir)
    processed = ensure_processed_inputs(cfg)

    cell_schema = load_cell_schema(_path(cfg, "cell_schema"))
    agents = build_agents_from_frames(
        cfg,
        pd.read_parquet(processed["ces_cell_distribution"]),
        pd.read_parquet(processed["anes_respondents"]),
        pd.read_parquet(processed["anes_memory_cards"]),
        cell_schema,
    )
    agents_path = run_dir / "agents.parquet"
    write_table(agents, agents_path)

    questions = load_question_config(processed["question_bank"])
    ces_respondents = pd.read_parquet(processed["ces_respondents"])
    memory_facts = pd.read_parquet(processed["anes_memory_facts"])
    mit_results = pd.read_parquet(processed["mit_results"])
    llm_client = build_llm_client(cfg.model)
    baselines = build_baselines(cfg.baselines, ces_respondents, cell_schema["columns"], llm_client)
    model_name = getattr(llm_client, "model_name", cfg.model.model_name)

    prompt_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    memory_policy = cfg.anes_memory.get("memory_policy", "safe_survey_memory_v1")
    max_memory = int(cfg.anes_memory.get("max_memory_facts", 24))

    for _, agent in agents.iterrows():
        for _, question in questions.iterrows():
            allowed = list(question["allowed_answer_codes"])
            for baseline_name, baseline in baselines.items():
                if isinstance(baseline, LLMBaseline):
                    prompt_text, fact_ids = build_prompt(
                        agent,
                        question,
                        baseline.prompt_mode,
                        memory_facts=memory_facts,
                        memory_policy=memory_policy,
                        max_memory_facts=max_memory,
                    )
                    prompt = _prompt_record(
                        cfg=cfg,
                        agent=agent,
                        question=question,
                        baseline=baseline_name,
                        model_name=model_name,
                        prompt_text=prompt_text,
                        fact_ids=fact_ids,
                    )
                    pred = baseline.predict(prompt_text, allowed)
                    parsed = parse_json_answer(pred.raw_response, allowed)
                    answer = parsed["answer"]
                    confidence = parsed["confidence"]
                    parse_status = parsed["parse_status"]
                else:
                    options = options_from_question(question)
                    prompt_text = (
                        f"Non-LLM baseline {baseline_name} for {question['question_id']} "
                        f"with options {json.dumps(options, sort_keys=True)}"
                    )
                    prompt = _prompt_record(
                        cfg=cfg,
                        agent=agent,
                        question=question,
                        baseline=baseline_name,
                        model_name="non_llm",
                        prompt_text=prompt_text,
                        fact_ids=[],
                    )
                    pred = baseline.predict(agent=agent, question=question)
                    answer = pred.answer
                    confidence = pred.confidence
                    parse_status = "ok" if answer in allowed else "invalid_option"
                prompt_rows.append(prompt)
                response_rows.append(
                    _response_record(
                        cfg=cfg,
                        prompt_id=prompt["prompt_id"],
                        agent=agent,
                        question=question,
                        baseline=baseline_name,
                        model_name=model_name if isinstance(baseline, LLMBaseline) else "non_llm",
                        raw_response=pred.raw_response,
                        answer=answer,
                        confidence=confidence,
                        parse_status=parse_status,
                        probabilities=pred.probabilities,
                    )
                )

    prompts = pd.DataFrame(prompt_rows)
    responses = pd.DataFrame(response_rows)
    prompts_path = run_dir / "prompts.parquet"
    responses_path = run_dir / "responses.parquet"
    write_table(prompts, prompts_path)
    write_table(responses, responses_path)

    aggregate = write_aggregate_state_results(
        responses,
        agents,
        cfg.run_id,
        int(cfg.scenario.year),
        run_dir / "aggregate_state_results.parquet",
    )
    metrics = write_eval_metrics(
        aggregate,
        mit_results,
        cfg.run_id,
        run_dir / "eval_metrics.parquet",
    )
    write_eval_report(
        run_id=cfg.run_id,
        run_dir=run_dir,
        aggregate=aggregate,
        metrics=metrics,
        mit_results=mit_results,
        agents=agents,
        responses=responses,
    )

    return {
        "agents": agents_path,
        "prompts": prompts_path,
        "responses": responses_path,
        "aggregate": run_dir / "aggregate_state_results.parquet",
        "metrics": run_dir / "eval_metrics.parquet",
        "report": run_dir / "eval_report.md",
    }


def evaluate_run(run_id: str, run_dir: str | Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    aggregate = pd.read_parquet(run_dir / "aggregate_state_results.parquet")
    agents = pd.read_parquet(run_dir / "agents.parquet")
    responses = pd.read_parquet(run_dir / "responses.parquet")
    mit_path = next((run_dir.parent.parent / "processed").glob("**/mit_election_results.parquet"), None)
    if mit_path is None:
        raise FileNotFoundError("Could not locate processed mit_election_results.parquet")
    mit_results = pd.read_parquet(mit_path)
    metrics = write_eval_metrics(aggregate, mit_results, run_id, run_dir / "eval_metrics.parquet")
    report = write_eval_report(
        run_id=run_id,
        run_dir=run_dir,
        aggregate=aggregate,
        metrics=metrics,
        mit_results=mit_results,
        agents=agents,
        responses=responses,
    )
    return {"metrics": run_dir / "eval_metrics.parquet", "report": report}
