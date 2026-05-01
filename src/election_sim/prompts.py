"""Prompt rendering and JSON response parsing."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import pandas as pd
from jinja2 import Template

from .ces_schema import parse_turnout_vote_json
from .survey_memory import LeakageGuard


DEMOGRAPHIC_TEMPLATE = """You are simulating a U.S. eligible voter in {{ year }}.

Voter profile:
- State: {{ state_po }}
- Age group: {{ age_group }}
- Gender: {{ gender }}
- Race/ethnicity: {{ race_ethnicity }}
- Education: {{ education_binary }}

Question:
{{ question_text }}

Options:
{% for code, label in options.items() -%}
- {{ code }}: {{ label }}
{% endfor %}

Return JSON only with this schema:
{"answer": "<option code>", "confidence": <number between 0 and 1>}
"""

PARTY_IDEOLOGY_TEMPLATE = """You are simulating a U.S. eligible voter in {{ year }}.

Voter profile:
- State: {{ state_po }}
- Age group: {{ age_group }}
- Gender: {{ gender }}
- Race/ethnicity: {{ race_ethnicity }}
- Education: {{ education_binary }}
- Party identification: {{ party_id_3 }}
- Ideology: {{ ideology_3 }}

Question:
{{ question_text }}

Options:
{% for code, label in options.items() -%}
- {{ code }}: {{ label }}
{% endfor %}

Return JSON only with this schema:
{"answer": "<option code>", "confidence": <number between 0 and 1>}
"""

SURVEY_MEMORY_TEMPLATE = """You are simulating a U.S. eligible voter in {{ year }}.
Answer as this voter would answer, not as a political analyst.

Voter profile:
- State: {{ state_po }}
- Age group: {{ age_group }}
- Gender: {{ gender }}
- Race/ethnicity: {{ race_ethnicity }}
- Education: {{ education_binary }}
- Party identification: {{ party_id_3 }}
- Ideology: {{ ideology_3 }}

Additional survey-derived background facts:
{% for fact in memory_facts -%}
- {{ fact }}
{% endfor %}

Question:
{{ question_text }}

Options:
{% for code, label in options.items() -%}
- {{ code }}: {{ label }}
{% endfor %}

Return JSON only with this schema:
{"answer": "<option code>", "confidence": <number between 0 and 1>}
"""

CES_PRESIDENT_VOTE_TEMPLATE = """You are simulating how a specific U.S. eligible voter would behave in the 2024 general election.
Answer as this voter would behave, not as a political analyst.

Voter profile:
- State: {{ state_po }}
- Age group: {{ age_group }}
- Gender: {{ gender }}
- Race/ethnicity: {{ race_ethnicity }}
- Education: {{ education_binary }}
{% if include_party_ideology -%}
- Party identification: {{ party_id_3 }}
- 7-point party ID: {{ party_id_7 }}
- Ideology: {{ ideology_3 }}
{% endif %}

{% if memory_facts -%}
{{ memory_section_title }}:
{% for fact in memory_facts -%}
- {{ fact }}
{% endfor %}
{% endif %}

{% if inferred_persona_facts -%}
{{ inferred_persona_section_title }}:
{% for fact in inferred_persona_facts -%}
- {{ fact }}
{% endfor %}
{% endif %}

Election context:
- Office: President
{% for candidate in candidates -%}
- {{ candidate.candidate_party }} candidate: {{ candidate.candidate_name }}
{% endfor %}

Task:
Estimate this voter's turnout probability and presidential vote choice.

Return JSON only with this schema:
{
  "turnout_probability": 0.0,
  "vote_probabilities": {
    "democrat": 0.0,
    "republican": 0.0,
    "other": 0.0,
    "undecided": 0.0
  },
  "most_likely_choice": "democrat|republican|other|undecided|not_vote",
  "confidence": 0.0
}
"""

TEMPLATES = {
    "demographic_only": DEMOGRAPHIC_TEMPLATE,
    "party_ideology": PARTY_IDEOLOGY_TEMPLATE,
    "survey_memory": SURVEY_MEMORY_TEMPLATE,
    "ces_president_vote_v1": CES_PRESIDENT_VOTE_TEMPLATE,
}

CES_PROMPT_MODE_CONFIGS = {
    "ces_demographic_only": {
        "include_party_ideology": False,
        "include_memory": False,
        "fact_roles": [],
        "inferred_persona_roles": [],
        "memory_section_title": "Survey-derived background facts",
        "inferred_persona_section_title": "ANES-matched inferred persona context",
    },
    "ces_party_ideology": {
        "include_party_ideology": True,
        "include_memory": False,
        "fact_roles": [],
        "inferred_persona_roles": [],
        "memory_section_title": "Survey-derived background facts",
        "inferred_persona_section_title": "ANES-matched inferred persona context",
    },
    "ces_survey_memory": {
        "include_party_ideology": True,
        "include_memory": True,
        "fact_roles": ["safe_pre"],
        "inferred_persona_roles": [],
        "memory_section_title": "Strict pre-election survey-derived background facts",
        "inferred_persona_section_title": "ANES-matched inferred persona context",
    },
    "ces_poll_informed": {
        "include_party_ideology": True,
        "include_memory": True,
        "fact_roles": ["safe_pre", "poll_prior"],
        "inferred_persona_roles": [],
        "memory_section_title": "Survey-derived background facts, including poll-prior facts",
        "inferred_persona_section_title": "ANES-matched inferred persona context",
    },
    "ces_anes_persona": {
        "include_party_ideology": True,
        "include_memory": True,
        "fact_roles": ["safe_pre"],
        "inferred_persona_roles": ["inferred_persona"],
        "memory_section_title": "Strict pre-election CES survey-derived background facts",
        "inferred_persona_section_title": "ANES-matched inferred persona context",
    },
}

CES_LLM_BASELINE_PROMPT_MODES = {
    "ces_demographic_only_llm": "ces_demographic_only",
    "ces_party_ideology_llm": "ces_party_ideology",
    "ces_survey_memory_llm": "ces_survey_memory",
    "ces_poll_informed_llm": "ces_poll_informed",
    "ces_anes_persona_llm": "ces_anes_persona",
    # Backward-compatible aliases used by early CES configs.
    "demographic_only_llm": "ces_demographic_only",
    "party_ideology_llm": "ces_party_ideology",
    "survey_memory_llm": "ces_survey_memory",
}


def options_from_question(question: pd.Series | dict[str, Any]) -> dict[str, str]:
    options = question["options_json"]
    if isinstance(options, str):
        return json.loads(options)
    return dict(options)


def memory_facts_for_agent(
    agent: pd.Series,
    question: pd.Series,
    memory_facts: pd.DataFrame,
    *,
    memory_policy: str,
    max_facts: int,
) -> tuple[list[str], list[str]]:
    facts = memory_facts[memory_facts["anes_id"] == agent["base_anes_id"]]
    filtered = LeakageGuard().filter_facts(facts, question, memory_policy).head(max_facts)
    return filtered["fact_text"].tolist(), filtered["memory_fact_id"].tolist()


def build_prompt(
    agent: pd.Series,
    question: pd.Series,
    mode: str,
    *,
    memory_facts: pd.DataFrame | None = None,
    memory_policy: str = "safe_survey_memory_v1",
    max_memory_facts: int = 24,
) -> tuple[str, list[str]]:
    options = options_from_question(question)
    prompt_facts: list[str] = []
    fact_ids: list[str] = []
    if mode == "survey_memory" and memory_facts is not None:
        prompt_facts, fact_ids = memory_facts_for_agent(
            agent,
            question,
            memory_facts,
            memory_policy=memory_policy,
            max_facts=max_memory_facts,
        )
    template = Template(TEMPLATES[mode])
    state_po = agent["state_po"] if pd.notna(agent.get("state_po")) and agent.get("state_po") else "unknown"
    text = template.render(
        year=agent["year"],
        state_po=state_po,
        age_group=agent["age_group"] or "unknown",
        gender=agent["gender"],
        race_ethnicity=agent["race_ethnicity"],
        education_binary=agent["education_binary"],
        party_id_3=agent["party_id_3"],
        ideology_3=agent["ideology_3"],
        memory_facts=prompt_facts,
        question_text=question["question_text"],
        options=options,
    )
    return text, fact_ids


def ces_memory_facts_for_agent(
    agent: pd.Series,
    question: pd.Series | dict[str, Any],
    memory_facts: pd.DataFrame | Mapping[str, pd.DataFrame],
    *,
    memory_policy: str,
    max_facts: int,
    fact_roles: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    base_ces_id = str(agent.get("base_ces_id") or agent.get("source_respondent_id"))
    if isinstance(memory_facts, Mapping):
        facts = memory_facts.get(base_ces_id)
        if facts is None or facts.empty:
            return [], []
    else:
        if memory_facts.empty:
            return [], []
        facts = memory_facts[memory_facts["ces_id"].astype(str) == base_ces_id]
    filtered = LeakageGuard().filter_facts(facts, question, memory_policy)
    if fact_roles is not None:
        allowed_roles = set(fact_roles)
        if "fact_role" in filtered.columns:
            roles = filtered["fact_role"].fillna("safe_pre").astype(str)
            filtered = filtered[roles.isin(allowed_roles)]
        elif "safe_pre" not in allowed_roles:
            filtered = filtered.head(0)
    if "fact_priority" in filtered.columns:
        filtered = filtered.sort_values(["fact_priority", "source_variable"], ascending=[False, True])
    filtered = filtered.head(max_facts)
    return filtered["fact_text"].tolist(), filtered["memory_fact_id"].tolist()


def build_ces_prompt(
    agent: pd.Series,
    question: pd.Series | dict[str, Any],
    *,
    memory_facts: pd.DataFrame | Mapping[str, pd.DataFrame],
    context: pd.DataFrame | Mapping[str, list[dict[str, Any]]],
    memory_policy: str = "strict_pre_no_vote_v1",
    max_memory_facts: int = 24,
    prompt_mode: str = "ces_survey_memory",
) -> tuple[str, list[str]]:
    if prompt_mode not in CES_PROMPT_MODE_CONFIGS:
        raise ValueError(f"Unknown CES prompt mode: {prompt_mode}")
    mode_cfg = CES_PROMPT_MODE_CONFIGS[prompt_mode]
    prompt_facts: list[str] = []
    fact_ids: list[str] = []
    inferred_persona_facts: list[str] = []
    inferred_persona_fact_ids: list[str] = []
    if mode_cfg["include_memory"]:
        prompt_facts, fact_ids = ces_memory_facts_for_agent(
            agent,
            question,
            memory_facts,
            memory_policy=memory_policy,
            max_facts=max_memory_facts,
            fact_roles=list(mode_cfg["fact_roles"]),
        )
        inferred_roles = list(mode_cfg.get("inferred_persona_roles", []))
        if inferred_roles:
            inferred_persona_facts, inferred_persona_fact_ids = ces_memory_facts_for_agent(
                agent,
                question,
                memory_facts,
                memory_policy=memory_policy,
                max_facts=max_memory_facts,
                fact_roles=inferred_roles,
            )
    base_ces_id = str(agent.get("base_ces_id") or agent.get("source_respondent_id"))
    if isinstance(context, Mapping):
        candidates = list(context.get(base_ces_id, []))
    elif context.empty or "ces_id" not in context.columns:
        candidates = []
    else:
        candidates = context[context["ces_id"].astype(str) == base_ces_id].to_dict("records")
    if not candidates:
        candidates = [
            {"candidate_party": "Democratic", "candidate_name": "Kamala Harris"},
            {"candidate_party": "Republican", "candidate_name": "Donald Trump"},
        ]
    template = Template(TEMPLATES["ces_president_vote_v1"])
    text = template.render(
        state_po=agent.get("state_po") or "unknown",
        age_group=agent.get("age_group") or "unknown",
        gender=agent.get("gender") or "unknown",
        race_ethnicity=agent.get("race_ethnicity") or "unknown",
        education_binary=agent.get("education_binary") or "unknown",
        include_party_ideology=bool(mode_cfg["include_party_ideology"]),
        party_id_3=agent.get("party_id_3") or "unknown",
        party_id_7=agent.get("party_id_7") or "unknown",
        ideology_3=agent.get("ideology_3") or "unknown",
        memory_facts=prompt_facts,
        memory_section_title=mode_cfg["memory_section_title"],
        inferred_persona_facts=inferred_persona_facts,
        inferred_persona_section_title=mode_cfg["inferred_persona_section_title"],
        candidates=candidates,
    )
    return text, fact_ids + inferred_persona_fact_ids


def parse_json_answer(raw_response: str, allowed: list[str]) -> dict[str, Any]:
    text = raw_response.strip()
    payload: dict[str, Any]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {"answer": None, "confidence": None, "parse_status": "failed"}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"answer": None, "confidence": None, "parse_status": "failed"}
    answer = payload.get("answer")
    if answer not in allowed:
        return {"answer": None, "confidence": payload.get("confidence"), "parse_status": "invalid_option"}
    confidence = payload.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0
    return {"answer": answer, "confidence": confidence, "parse_status": "ok"}
