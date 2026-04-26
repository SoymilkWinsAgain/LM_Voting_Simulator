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
from .report import write_eval_report, write_individual_report
from .transforms import stable_hash


def _path(cfg: RunConfig, key: str) -> Path:
    if key not in cfg.paths:
        raise KeyError(f"Run config missing paths.{key}")
    return Path(cfg.paths[key])


def _memory_policy(cfg: RunConfig) -> str:
    return cfg.anes_memory.get("memory_policy", "safe_survey_memory_v1")


def _max_memory_facts(cfg: RunConfig) -> int:
    return int(cfg.anes_memory.get("max_memory_facts", cfg.anes_memory.get("max_facts", 24)))


def _question_bank_path(cfg: RunConfig, default_path: Path) -> Path:
    return _path(cfg, "question_set") if "question_set" in cfg.paths else default_path


def _build_anes_artifacts(cfg: RunConfig, out_dir: Path) -> dict[str, Path]:
    anes_paths = build_anes(
        _path(cfg, "anes_config"),
        _path(cfg, "anes_profile_crosswalk"),
        _path(cfg, "anes_question_crosswalk"),
        out_dir,
    )
    memory_paths = build_memory_cards(
        anes_paths["respondents"],
        anes_paths["answers"],
        _path(cfg, "anes_fact_templates"),
        _memory_policy(cfg),
        out_dir,
        max_facts=_max_memory_facts(cfg),
    )
    return {
        "anes_respondents": anes_paths["respondents"],
        "anes_answers": anes_paths["answers"],
        "anes_memory_facts": memory_paths["facts"],
        "anes_memory_cards": memory_paths["cards"],
        "question_bank": _question_bank_path(cfg, anes_paths["question_bank"]),
    }


def ensure_processed_inputs(cfg: RunConfig) -> dict[str, Path]:
    """Build processed fixture/intermediate files needed by a run."""

    processed_dir = ensure_dir(cfg.processed_dir)
    anes_dir = ensure_dir(processed_dir / "anes")
    ces_dir = ensure_dir(processed_dir / "ces")
    mit_dir = ensure_dir(processed_dir / "mit")

    anes_artifacts = _build_anes_artifacts(cfg, anes_dir)
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
        **anes_artifacts,
        "ces_respondents": ces_paths["respondents"],
        "ces_cell_distribution": ces_paths["cell_distribution"],
        "mit_results": mit_path,
    }


def ensure_anes_processed_inputs(cfg: RunConfig) -> dict[str, Path]:
    """Build ANES-only processed files for individual benchmark smoke runs."""

    processed_dir = ensure_dir(cfg.processed_dir)
    return _build_anes_artifacts(cfg, processed_dir)


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


def _select_individual_smoke_agent(
    cfg: RunConfig,
    respondents: pd.DataFrame,
    answers: pd.DataFrame,
    memory_facts: pd.DataFrame,
    question: pd.Series,
) -> tuple[pd.Series, str]:
    smoke_cfg = cfg.model_extra.get("smoke", {}) if cfg.model_extra else {}
    requested_id = smoke_cfg.get("respondent_id")
    target_answers = answers[answers["question_id"] == question["question_id"]].copy()
    allowed = set(question["allowed_answer_codes"])
    target_answers = target_answers[target_answers["canonical_value"].isin(allowed)]
    if not bool(smoke_cfg.get("allow_unknown_target", False)):
        target_answers = target_answers[target_answers["canonical_value"] != "not_vote_or_unknown"]
    if requested_id:
        target_answers = target_answers[target_answers["anes_id"].astype(str) == str(requested_id)]
    memory_ids = set(memory_facts["anes_id"].astype(str)) if not memory_facts.empty else set()
    for _, target in target_answers.sort_values("anes_id").iterrows():
        if str(target["anes_id"]) in memory_ids:
            respondent = respondents[respondents["anes_id"].astype(str) == str(target["anes_id"])].iloc[0]
            return respondent, str(target["canonical_value"])
    if target_answers.empty:
        raise ValueError(f"No valid target answers for {question['question_id']}")
    target = target_answers.sort_values("anes_id").iloc[0]
    respondent = respondents[respondents["anes_id"].astype(str) == str(target["anes_id"])].iloc[0]
    return respondent, str(target["canonical_value"])


def _agent_from_anes_respondent(cfg: RunConfig, respondent: pd.Series) -> pd.DataFrame:
    agent = {
        "run_id": cfg.run_id,
        "agent_id": f"{cfg.run_id}_{respondent['anes_id']}",
        "year": int(cfg.scenario.year),
        "state_po": respondent.get("state_po"),
        "cell_schema": "anes_individual_v1",
        "cell_id": f"ANES|{respondent['anes_id']}",
        "base_anes_id": respondent["anes_id"],
        "memory_card_id": f"{respondent['anes_id']}_{cfg.anes_memory.get('memory_policy', 'safe_survey_memory_v1')}",
        "match_level": 0,
        "match_distance": 0.0,
        "sample_weight": 1.0,
        "age_group": respondent["age_group"],
        "gender": respondent["gender"],
        "race_ethnicity": respondent["race_ethnicity"],
        "education_binary": respondent["education_binary"],
        "party_id_3": respondent["party_id_3"],
        "ideology_3": respondent["ideology_3"],
        "created_at": pd.Timestamp.now(tz="UTC"),
    }
    return pd.DataFrame([agent])


def _individual_metrics(
    cfg: RunConfig,
    responses: pd.DataFrame,
    baseline: str,
) -> pd.DataFrame:
    correct = responses["correct"].astype(bool)
    parse_ok = responses["parse_status"] == "ok"
    rows = [
        {
            "run_id": cfg.run_id,
            "metric_scope": "individual",
            "baseline": baseline,
            "model_name": cfg.model.model_name,
            "metric_name": "accuracy",
            "metric_value": float(correct.mean()) if len(correct) else 0.0,
            "state_po": None,
            "group_key": None,
            "question_id": responses["question_id"].iloc[0] if len(responses) else None,
            "confidence_low": None,
            "confidence_high": None,
            "created_at": pd.Timestamp.now(tz="UTC"),
        },
        {
            "run_id": cfg.run_id,
            "metric_scope": "individual",
            "baseline": baseline,
            "model_name": cfg.model.model_name,
            "metric_name": "parse_ok_rate",
            "metric_value": float(parse_ok.mean()) if len(parse_ok) else 0.0,
            "state_po": None,
            "group_key": None,
            "question_id": responses["question_id"].iloc[0] if len(responses) else None,
            "confidence_low": None,
            "confidence_high": None,
            "created_at": pd.Timestamp.now(tz="UTC"),
        },
    ]
    return pd.DataFrame(rows)


def run_individual_benchmark(run_config_path: str | Path) -> dict[str, Path]:
    cfg = load_run_config(run_config_path)
    run_dir = ensure_dir(cfg.run_dir)
    processed = ensure_anes_processed_inputs(cfg)

    respondents = pd.read_parquet(processed["anes_respondents"])
    answers = pd.read_parquet(processed["anes_answers"])
    memory_facts = pd.read_parquet(processed["anes_memory_facts"])
    questions = load_question_config(processed["question_bank"])
    target_question_ids = cfg.questions.get("target_question_ids") or [questions.iloc[0]["question_id"]]
    question = questions[questions["question_id"] == target_question_ids[0]].iloc[0]
    respondent, target_answer = _select_individual_smoke_agent(cfg, respondents, answers, memory_facts, question)
    agents = _agent_from_anes_respondent(cfg, respondent)
    agent = agents.iloc[0]

    llm_client = build_llm_client(cfg.model)
    model_name = getattr(llm_client, "model_name", cfg.model.model_name)
    baseline_name = cfg.baselines[0] if cfg.baselines else "survey_memory_llm"
    baseline_modes = {
        "demographic_only_llm": "demographic_only",
        "party_ideology_llm": "party_ideology",
        "survey_memory_llm": "survey_memory",
    }
    if baseline_name not in baseline_modes:
        raise ValueError(f"Individual benchmark supports only LLM baselines, got {baseline_name}")
    baseline = LLMBaseline(baseline_name, baseline_modes[baseline_name], llm_client)
    prompt_text, fact_ids = build_prompt(
        agent,
        question,
        baseline.prompt_mode,
        memory_facts=memory_facts,
        memory_policy=_memory_policy(cfg),
        max_memory_facts=_max_memory_facts(cfg),
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
    allowed = list(question["allowed_answer_codes"])
    pred = baseline.predict(prompt_text, allowed)
    parsed = parse_json_answer(pred.raw_response, allowed)
    response = _response_record(
        cfg=cfg,
        prompt_id=prompt["prompt_id"],
        agent=agent,
        question=question,
        baseline=baseline_name,
        model_name=model_name,
        raw_response=pred.raw_response,
        answer=parsed["answer"],
        confidence=parsed["confidence"],
        parse_status=parsed["parse_status"],
        probabilities=None,
    )
    response["target_answer_code"] = target_answer
    response["correct"] = parsed["answer"] == target_answer

    prompts = pd.DataFrame([prompt])
    responses = pd.DataFrame([response])
    metrics = _individual_metrics(cfg, responses, baseline_name)

    agents_path = run_dir / "agents.parquet"
    prompts_path = run_dir / "prompts.parquet"
    responses_path = run_dir / "responses.parquet"
    metrics_path = run_dir / "eval_metrics.parquet"
    write_table(agents, agents_path)
    write_table(prompts, prompts_path)
    write_table(responses, responses_path)
    write_table(metrics, metrics_path)

    preview = [
        f"# Prompt Preview: {cfg.run_id}",
        "",
        f"- Agent ID: `{agent['agent_id']}`",
        f"- Base ANES ID: `{agent['base_anes_id']}`",
        f"- Target answer: `{target_answer}`",
        f"- Parsed answer: `{parsed['answer']}`",
        f"- Parse status: `{parsed['parse_status']}`",
        "",
        "## Prompt",
        "```text",
        prompt_text,
        "```",
        "",
        "## Raw response",
        "```text",
        pred.raw_response,
        "```",
        "",
    ]
    prompt_preview_path = run_dir / "prompt_preview.md"
    prompt_preview_path.write_text("\n".join(preview), encoding="utf-8")
    report = write_individual_report(
        run_id=cfg.run_id,
        run_dir=run_dir,
        agents=agents,
        prompts=prompts,
        responses=responses,
        metrics=metrics,
    )
    return {
        "agents": agents_path,
        "prompts": prompts_path,
        "prompt_preview": prompt_preview_path,
        "responses": responses_path,
        "metrics": metrics_path,
        "report": report,
    }


def run_simulation(run_config_path: str | Path) -> dict[str, Path]:
    cfg = load_run_config(run_config_path)
    if cfg.mode == "individual_benchmark":
        return run_individual_benchmark(run_config_path)
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
    memory_policy = _memory_policy(cfg)
    max_memory = _max_memory_facts(cfg)

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
