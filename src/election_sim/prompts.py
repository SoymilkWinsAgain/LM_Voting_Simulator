"""Prompt rendering and JSON response parsing."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from jinja2 import Template

from .anes import LeakageGuard


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

TEMPLATES = {
    "demographic_only": DEMOGRAPHIC_TEMPLATE,
    "party_ideology": PARTY_IDEOLOGY_TEMPLATE,
    "survey_memory": SURVEY_MEMORY_TEMPLATE,
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
