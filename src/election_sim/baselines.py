"""Non-LLM and LLM-backed baseline strategies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .ces import empirical_vote_probabilities

VOTE_CODES = ["democrat", "republican", "other", "not_vote_or_unknown"]


@dataclass
class BaselinePrediction:
    answer: str
    confidence: float
    raw_response: str
    probabilities: dict[str, float] | None = None


class MajorityBaseline:
    name = "majority"

    def __init__(self, ces_respondents: pd.DataFrame):
        counts = ces_respondents.groupby("vote_choice_president")["common_weight"].sum()
        if counts.empty:
            self.answer = "not_vote_or_unknown"
            self.confidence = 0.0
        else:
            self.answer = str(counts.idxmax())
            self.confidence = float(counts.max() / counts.sum())

    def predict(self, *_: Any, **__: Any) -> BaselinePrediction:
        probs = {code: 1.0 if code == self.answer else 0.0 for code in VOTE_CODES}
        return BaselinePrediction(
            answer=self.answer,
            confidence=self.confidence,
            raw_response=json.dumps({"answer": self.answer, "confidence": self.confidence}),
            probabilities=probs,
        )


class CesEmpiricalCellBaseline:
    name = "ces_empirical_cell"

    def __init__(self, ces_respondents: pd.DataFrame, cell_cols: list[str]):
        probs = empirical_vote_probabilities(ces_respondents, cell_cols)
        self.by_state_cell: dict[tuple[str, str], dict[str, float]] = {}
        for (state, cell_id), group in probs.groupby(["state_po", "cell_id"]):
            row_probs = {code: 0.0 for code in VOTE_CODES}
            for _, row in group.iterrows():
                row_probs[str(row["vote_choice_president"])] = float(row["probability"])
            self.by_state_cell[(state, cell_id)] = row_probs
        national_counts = ces_respondents.groupby("vote_choice_president")["common_weight"].sum()
        total = float(national_counts.sum()) or 1.0
        self.national = {code: float(national_counts.get(code, 0.0) / total) for code in VOTE_CODES}

    def predict(self, agent: pd.Series, *_: Any, **__: Any) -> BaselinePrediction:
        probs = self.by_state_cell.get((agent["state_po"], agent["cell_id"]), self.national)
        answer = max(probs, key=probs.get)
        return BaselinePrediction(
            answer=answer,
            confidence=float(probs.get(answer, 0.0)),
            raw_response=json.dumps({"answer": answer, "confidence": probs.get(answer, 0.0), "probabilities": probs}),
            probabilities=probs,
        )


class LLMBaseline:
    def __init__(self, name: str, prompt_mode: str, client: Any):
        self.name = name
        self.prompt_mode = prompt_mode
        self.client = client

    def predict(self, prompt_text: str, allowed: list[str]) -> BaselinePrediction:
        raw = self.client.complete(prompt_text, allowed)
        from .prompts import parse_json_answer

        parsed = parse_json_answer(raw, allowed)
        return BaselinePrediction(
            answer=parsed["answer"],
            confidence=float(parsed.get("confidence") or 0.0),
            raw_response=raw,
            probabilities=None,
        )


def build_baselines(
    names: list[str],
    ces_respondents: pd.DataFrame,
    cell_cols: list[str],
    llm_client: Any,
) -> dict[str, Any]:
    baselines: dict[str, Any] = {}
    for name in names:
        if name == "majority":
            baselines[name] = MajorityBaseline(ces_respondents)
        elif name == "ces_empirical_cell":
            baselines[name] = CesEmpiricalCellBaseline(ces_respondents, cell_cols)
        elif name == "demographic_only_llm":
            baselines[name] = LLMBaseline(name, "demographic_only", llm_client)
        elif name == "party_ideology_llm":
            baselines[name] = LLMBaseline(name, "party_ideology", llm_client)
        elif name == "survey_memory_llm":
            baselines[name] = LLMBaseline(name, "survey_memory", llm_client)
        else:
            raise ValueError(f"Unknown baseline: {name}")
    return baselines
