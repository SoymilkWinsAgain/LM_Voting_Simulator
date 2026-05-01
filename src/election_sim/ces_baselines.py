"""CES turnout + vote non-LLM baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .ces_schema import CES_TURNOUT_VOTE_CHOICES, format_turnout_vote_choice_response
from .survey_memory import is_leakage_variable


PROFILE_FEATURES = [
    "state_po",
    "age_group",
    "gender",
    "race_ethnicity",
    "education_binary",
    "income_bin",
    "party_id_3_pre",
    "party_id_7_pre",
    "ideology_3",
    "registered_self_pre",
    "party_registration_self",
    "citizenship",
]


@dataclass
class CesTurnoutVotePrediction:
    raw_response: str
    model_name: str


def _vote_probs_for_party(agent: pd.Series) -> dict[str, float]:
    party = str(agent.get("party_id_3") or agent.get("party_id_3_pre") or "unknown")
    ideology = str(agent.get("ideology_3") or "unknown")
    if party == "democrat":
        return {"democrat": 0.78, "republican": 0.12}
    if party == "republican":
        return {"democrat": 0.12, "republican": 0.78}
    if ideology == "liberal":
        return {"democrat": 0.58, "republican": 0.24}
    if ideology == "conservative":
        return {"democrat": 0.24, "republican": 0.58}
    return {"democrat": 0.37, "republican": 0.37}


def _turnout_for_agent(agent: pd.Series) -> float:
    registered = str(agent.get("registered_self_pre") or "unknown")
    if registered == "yes":
        return 0.84
    if registered == "no":
        return 0.28
    return 0.62


def _choice(turnout: float, vote_probs: dict[str, float]) -> str:
    turnout = float(np.clip(turnout, 0.0, 1.0))
    total = float(vote_probs.get("democrat", 0.0)) + float(vote_probs.get("republican", 0.0))
    dem = float(vote_probs.get("democrat", 0.0)) / total if total > 0 else 0.5
    rep = float(vote_probs.get("republican", 0.0)) / total if total > 0 else 0.5
    scores = {"not_vote": 1.0 - turnout, "democrat": turnout * dem, "republican": turnout * rep}
    return max(CES_TURNOUT_VOTE_CHOICES, key=lambda code: (scores[code], -CES_TURNOUT_VOTE_CHOICES.index(code)))


class PartyIdBaseline:
    name = "party_id_baseline"
    model_name = "party_id_baseline_v1"
    feature_policy = "strict_pre_no_vote_v1"
    is_poll_informed = False

    def predict(self, agent: pd.Series) -> CesTurnoutVotePrediction:
        turnout = _turnout_for_agent(agent)
        probs = _vote_probs_for_party(agent)
        choice = _choice(turnout, probs)
        raw = format_turnout_vote_choice_response(choice)
        return CesTurnoutVotePrediction(raw_response=raw, model_name=self.model_name)


def _answers_wide(answers: pd.DataFrame, policy: str) -> pd.DataFrame:
    if answers.empty:
        return pd.DataFrame(columns=["ces_id"])
    work = answers.copy()
    if "is_missing" in work.columns:
        work = work[~work["is_missing"].astype(bool)]
    work = work[~work["source_variable"].apply(lambda var: is_leakage_variable(var, policy))]
    if policy == "strict_pre_no_vote_v1" and "allowed_for_memory_strict" in work.columns:
        work = work[work["allowed_for_memory_strict"].astype(bool)]
    if work.empty:
        return pd.DataFrame(columns=["ces_id"])
    wide = work.pivot_table(
        index="ces_id",
        columns="source_variable",
        values="canonical_value",
        aggfunc="first",
    ).reset_index()
    wide.columns = [str(col) for col in wide.columns]
    return wide


def build_feature_frame(respondents: pd.DataFrame, answers: pd.DataFrame, policy: str) -> pd.DataFrame:
    profile_cols = ["ces_id", *[col for col in PROFILE_FEATURES if col in respondents.columns]]
    features = respondents[profile_cols].copy()
    wide = _answers_wide(answers, policy)
    if not wide.empty:
        features = features.merge(wide, on="ces_id", how="left")
    for col in features.columns:
        if col != "ces_id":
            features[col] = features[col].fillna("unknown").astype(str)
    return features


def _target_wide(targets: pd.DataFrame) -> pd.DataFrame:
    return targets.pivot_table(
        index="ces_id",
        columns="target_id",
        values="canonical_value",
        aggfunc="first",
    ).reset_index()


class SklearnLogitBaseline:
    def __init__(
        self,
        *,
        name: str,
        policy: str,
        respondents: pd.DataFrame,
        answers: pd.DataFrame,
        targets: pd.DataFrame,
        eval_ces_ids: set[str],
    ):
        self.name = name
        self.model_name = f"{name}_v1"
        self.feature_policy = policy
        self.is_poll_informed = policy == "poll_informed_pre_v1"
        self.fallback = PartyIdBaseline()
        self.features = build_feature_frame(respondents, answers, policy)
        self.feature_by_id = self.features.set_index("ces_id", drop=False)
        self.feature_columns = [col for col in self.features.columns if col != "ces_id"]
        labels = _target_wide(targets)
        train = self.features.merge(labels, on="ces_id", how="left")
        train = train[~train["ces_id"].astype(str).isin(eval_ces_ids)]
        self.turnout_model = self._fit_binary(train, "turnout_2024_self_report")
        self.vote_model = self._fit_multiclass(train, "president_vote_2024")
        self.raw_prediction_by_id = self._precompute_raw_predictions(eval_ces_ids)

    @staticmethod
    def _model_pipeline():
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.pipeline import make_pipeline

        return make_pipeline(
            DictVectorizer(sparse=True),
            OneVsRestClassifier(LogisticRegression(max_iter=200, solver="liblinear")),
        )

    def _records(self, df: pd.DataFrame) -> list[dict[str, str]]:
        return df[self.feature_columns].fillna("unknown").astype(str).to_dict("records")

    def _fit_binary(self, train: pd.DataFrame, target_col: str):
        known = train[train[target_col].isin(["voted", "not_voted"])].copy()
        if known.empty or known[target_col].nunique() < 2:
            return None
        y = (known[target_col] == "voted").astype(int)
        model = self._model_pipeline()
        model.fit(self._records(known), y)
        return model

    def _fit_multiclass(self, train: pd.DataFrame, target_col: str):
        known = train[train[target_col].isin(["democrat", "republican"])].copy()
        if known.empty or known[target_col].nunique() < 2:
            return None
        model = self._model_pipeline()
        model.fit(self._records(known), known[target_col].astype(str))
        return model

    def _feature_row(self, agent: pd.Series) -> pd.DataFrame | None:
        ces_id = str(agent.get("base_ces_id") or agent.get("source_respondent_id"))
        if ces_id not in self.feature_by_id.index:
            return None
        return self.feature_by_id.loc[[ces_id]].copy()

    def _turnout_probability(self, row: pd.DataFrame, agent: pd.Series) -> float:
        if self.turnout_model is None:
            return _turnout_for_agent(agent)
        classes = list(self.turnout_model.classes_)
        probs = self.turnout_model.predict_proba(self._records(row))[0]
        return float(probs[classes.index(1)]) if 1 in classes else 0.0

    def _vote_probabilities(self, row: pd.DataFrame, agent: pd.Series) -> dict[str, float]:
        if self.vote_model is None:
            return _vote_probs_for_party(agent)
        probs = {code: 0.0 for code in ["democrat", "republican"]}
        classes = [str(value) for value in self.vote_model.classes_]
        pred = self.vote_model.predict_proba(self._records(row))[0]
        for klass, value in zip(classes, pred, strict=False):
            if klass in probs:
                probs[klass] = float(value)
        return probs

    def _batch_turnout_probabilities(self, rows: pd.DataFrame) -> np.ndarray:
        if self.turnout_model is None:
            return rows.apply(_turnout_for_agent, axis=1).astype(float).to_numpy()
        classes = list(self.turnout_model.classes_)
        pred = self.turnout_model.predict_proba(self._records(rows))
        return pred[:, classes.index(1)] if 1 in classes else np.zeros(len(rows))

    def _batch_vote_probabilities(self, rows: pd.DataFrame) -> list[dict[str, float]]:
        if self.vote_model is None:
            return [_vote_probs_for_party(row) for _, row in rows.iterrows()]
        classes = [str(value) for value in self.vote_model.classes_]
        pred = self.vote_model.predict_proba(self._records(rows))
        out: list[dict[str, float]] = []
        for row_probs in pred:
            probs = {code: 0.0 for code in ["democrat", "republican"]}
            for klass, value in zip(classes, row_probs, strict=False):
                if klass in probs:
                    probs[klass] = float(value)
            out.append(probs)
        return out

    def _precompute_raw_predictions(self, eval_ces_ids: set[str]) -> dict[str, str]:
        if not eval_ces_ids:
            return {}
        rows = self.features[self.features["ces_id"].astype(str).isin(eval_ces_ids)].copy()
        if rows.empty:
            return {}
        turnout_values = self._batch_turnout_probabilities(rows)
        vote_values = self._batch_vote_probabilities(rows)
        cache: dict[str, str] = {}
        for (_, row), turnout, probs in zip(rows.iterrows(), turnout_values, vote_values, strict=False):
            choice = _choice(float(turnout), probs)
            cache[str(row["ces_id"])] = format_turnout_vote_choice_response(choice)
        return cache

    def predict(self, agent: pd.Series) -> CesTurnoutVotePrediction:
        ces_id = str(agent.get("base_ces_id") or agent.get("source_respondent_id"))
        if ces_id in self.raw_prediction_by_id:
            return CesTurnoutVotePrediction(raw_response=self.raw_prediction_by_id[ces_id], model_name=self.model_name)
        row = self._feature_row(agent)
        if row is None:
            return self.fallback.predict(agent)
        turnout = self._turnout_probability(row, agent)
        probs = self._vote_probabilities(row, agent)
        choice = _choice(turnout, probs)
        raw = format_turnout_vote_choice_response(choice)
        return CesTurnoutVotePrediction(raw_response=raw, model_name=self.model_name)


def build_ces_non_llm_baselines(
    names: list[str],
    *,
    respondents: pd.DataFrame,
    answers: pd.DataFrame,
    targets: pd.DataFrame,
    agents: pd.DataFrame,
) -> dict[str, Any]:
    eval_ids = set(agents["base_ces_id"].dropna().astype(str))
    baselines: dict[str, Any] = {}
    for name in names:
        if name == "party_id_baseline":
            baselines[name] = PartyIdBaseline()
        elif name == "sklearn_logit_pre_only":
            baselines[name] = SklearnLogitBaseline(
                name=name,
                policy="strict_pre_no_vote_v1",
                respondents=respondents,
                answers=answers,
                targets=targets,
                eval_ces_ids=eval_ids,
            )
        elif name == "sklearn_logit_poll_informed":
            baselines[name] = SklearnLogitBaseline(
                name=name,
                policy="poll_informed_pre_v1",
                respondents=respondents,
                answers=answers,
                targets=targets,
                eval_ces_ids=eval_ids,
            )
    return baselines
