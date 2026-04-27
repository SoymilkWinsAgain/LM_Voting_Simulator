"""CES respondent-level individual benchmark runner."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, log_loss, roc_auc_score

from .aggregation import aggregate_turnout_vote_state_results
from .ces_schema import CES_VOTE_PROBABILITY_CODES, format_turnout_vote_response, parse_turnout_vote_json
from .config import ModelConfig
from .evaluation import turnout_vote_election_metrics
from .io import ensure_dir, load_yaml, stable_json, write_table
from .llm import build_llm_client
from .prompts import build_ces_prompt
from .survey_memory import is_direct_pre_vote_variable, is_leakage_variable, is_post_vote_or_turnout_variable
from .transforms import stable_hash


VOTE_CLASSES = ["democrat", "republican", "other", "not_vote"]
CANDIDATE_CLASSES = ["democrat", "republican", "other"]
POLL_PRIOR_VARIABLES = {"CC24_363", "CC24_364a"}
DEFAULT_SUBGROUPS = [
    "party_id_3",
    "ideology_3",
    "race_ethnicity",
    "education_binary",
    "age_group",
    "gender",
    "state_po",
    "state_party_id_3",
]
NON_LLM_BASELINES = [
    "majority_by_state",
    "party_id_baseline",
    "sklearn_logit_demographic_only",
    "sklearn_logit_pre_only",
    "sklearn_logit_poll_informed",
]
LLM_BASELINES = [
    "ces_demographic_only_llm",
    "ces_party_ideology_llm",
    "ces_survey_memory_llm_strict",
    "ces_survey_memory_llm_poll_informed",
    "post_hoc_oracle_llm",
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
    "created_at",
]


PROFILE_FEATURES_STRICT = [
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
DEMOGRAPHIC_FEATURES = ["state_po", "age_group", "gender", "race_ethnicity", "education_binary"]


@dataclass
class BenchmarkPaths:
    run_dir: Path
    figures_dir: Path
    cache_path: Path


def assign_ces_splits(ces_ids: pd.Series, seed: int | str) -> pd.DataFrame:
    """Return deterministic benchmark bucket, split, and fold for CES ids."""

    seed_str = str(seed)
    out = pd.DataFrame({"ces_id": ces_ids.astype(str).to_numpy()})
    out["split_bucket"] = out["ces_id"].map(lambda value: int(stable_hash(value, seed_str, length=8), 16) % 100)
    out["split"] = out["split_bucket"].map(lambda bucket: "train" if bucket < 60 else "dev" if bucket < 80 else "test")
    out["fold"] = (out["split_bucket"] // 20).astype(int)
    return out


def crossfit_partitions(cohort: pd.DataFrame, heldout_fold: int) -> tuple[set[str], set[str], set[str]]:
    """Return train/dev/eval CES ids for a 5-fold cross-fit partition."""

    dev_fold = (int(heldout_fold) + 1) % 5
    eval_ids = set(cohort.loc[cohort["fold"] == heldout_fold, "ces_id"].astype(str))
    dev_ids = set(cohort.loc[cohort["fold"] == dev_fold, "ces_id"].astype(str))
    train_ids = set(cohort.loc[~cohort["fold"].isin([heldout_fold, dev_fold]), "ces_id"].astype(str))
    return train_ids, dev_ids, eval_ids


def _target_wide(targets: pd.DataFrame) -> pd.DataFrame:
    return targets.pivot_table(
        index="ces_id",
        columns="target_id",
        values="canonical_value",
        aggfunc="first",
    ).reset_index()


def build_benchmark_cohort(respondents: pd.DataFrame, seed: int, states: list[str] | str = "all") -> pd.DataFrame:
    work = respondents.copy()
    work["ces_id"] = work["ces_id"].astype(str)
    if states != "all":
        work = work[work["state_po"].isin(states)].copy()
    work = work[work["tookpost"].astype(bool)].copy()
    if "citizenship" in work.columns:
        work = work[work["citizenship"] == "yes"].copy()
    splits = assign_ces_splits(work["ces_id"], seed)
    cohort = work.merge(splits, on="ces_id", how="left", validate="one_to_one")
    cohort["sample_weight"] = pd.to_numeric(cohort.get("weight_common_post", 1.0), errors="coerce")
    cohort["sample_weight"] = cohort["sample_weight"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    cohort.loc[cohort["sample_weight"] <= 0, "sample_weight"] = 1.0
    cohort["state_party_id_3"] = cohort["state_po"].astype(str) + " x " + cohort["party_id_3_pre"].astype(str)
    cohort["_agent_ordinal"] = np.arange(1, len(cohort) + 1)
    return cohort.reset_index(drop=True)


def _agents_from_cohort(run_id: str, cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in cohort.reset_index(drop=True).iterrows():
        rows.append(
            {
                "run_id": run_id,
                "agent_id": f"{run_id}_ces_{int(row['_agent_ordinal']):06d}",
                "year": 2024,
                "state_po": row.get("state_po"),
                "source_dataset": "ces",
                "source_respondent_id": row["ces_id"],
                "cell_schema": "ces_rows_v1",
                "cell_id": f"CES|{row.get('state_po')}|{row['ces_id']}",
                "base_anes_id": None,
                "base_ces_id": row["ces_id"],
                "memory_card_id": None,
                "match_level": 0,
                "match_distance": 0.0,
                "sample_weight": row["sample_weight"],
                "weight_column": "weight_common_post",
                "weight_missing_reason": None,
                "age_group": row.get("age_group"),
                "gender": row.get("gender"),
                "race_ethnicity": row.get("race_ethnicity"),
                "education_binary": row.get("education_binary"),
                "income_bin": row.get("income_bin"),
                "party_id_3": row.get("party_id_3_pre"),
                "party_id_7": row.get("party_id_7_pre"),
                "ideology_3": row.get("ideology_3"),
                "ideology_7": row.get("ideology_self_7"),
                "registered_self_pre": row.get("registered_self_pre"),
                "validated_registration": row.get("validated_registration"),
                "split": row.get("split"),
                "split_bucket": int(row.get("split_bucket")),
                "fold": int(row.get("fold")),
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    return pd.DataFrame(rows)


def _answer_feature_wide(answers: pd.DataFrame, policy: str) -> pd.DataFrame:
    if answers.empty:
        return pd.DataFrame(columns=["ces_id"])
    work = answers.copy()
    work["source_variable"] = work["source_variable"].astype(str)
    if "is_missing" in work.columns:
        work = work[~work["is_missing"].astype(bool)]
    if policy == "strict_pre_no_vote_v1":
        work = work[~work["source_variable"].apply(lambda var: is_leakage_variable(var, policy))]
        if "allowed_for_memory_strict" in work.columns:
            work = work[work["allowed_for_memory_strict"].astype(bool)]
    elif policy == "poll_informed_pre_v1":
        allowed = []
        for var in work["source_variable"]:
            blocked = is_post_vote_or_turnout_variable(var) or var.upper().startswith("TS_")
            strict_safe = not is_leakage_variable(var, "strict_pre_no_vote_v1")
            poll_prior = var in POLL_PRIOR_VARIABLES and is_direct_pre_vote_variable(var)
            allowed.append((not blocked) and (strict_safe or poll_prior))
        work = work[pd.Series(allowed, index=work.index)]
    else:
        raise ValueError(f"Unsupported feature policy: {policy}")
    if work.empty:
        return pd.DataFrame(columns=["ces_id"])
    wide = work.pivot_table(index="ces_id", columns="source_variable", values="canonical_value", aggfunc="first")
    wide = wide.reset_index()
    wide.columns = [str(col) for col in wide.columns]
    return wide


def benchmark_feature_frame(
    cohort: pd.DataFrame,
    answers: pd.DataFrame,
    *,
    feature_mode: str,
) -> pd.DataFrame:
    if feature_mode == "demographic_only":
        cols = ["ces_id", *[col for col in DEMOGRAPHIC_FEATURES if col in cohort.columns]]
        features = cohort[cols].copy()
    elif feature_mode in {"strict_pre", "poll_informed"}:
        cols = ["ces_id", *[col for col in PROFILE_FEATURES_STRICT if col in cohort.columns]]
        features = cohort[cols].copy()
        policy = "strict_pre_no_vote_v1" if feature_mode == "strict_pre" else "poll_informed_pre_v1"
        wide = _answer_feature_wide(answers, policy)
        if not wide.empty:
            features = features.merge(wide, on="ces_id", how="left")
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
    for col in features.columns:
        if col != "ces_id":
            features[col] = features[col].fillna("unknown").astype(str)
    return features


def _safe_distribution(values: pd.Series, classes: list[str]) -> dict[str, float]:
    counts = values[values.isin(classes)].value_counts(normalize=True)
    if counts.empty:
        return {code: 1.0 / len(classes) for code in classes}
    out = {code: float(counts.get(code, 0.0)) for code in classes}
    total = sum(out.values()) or 1.0
    return {code: value / total for code, value in out.items()}


def _canonical_raw(turnout: float, candidate_probs: dict[str, float]) -> str:
    probs = {code: max(0.0, float(candidate_probs.get(code, 0.0))) for code in CANDIDATE_CLASSES}
    total = sum(probs.values())
    if total <= 0:
        probs = {"democrat": 1 / 3, "republican": 1 / 3, "other": 1 / 3}
    else:
        probs = {code: value / total for code, value in probs.items()}
    vote_probs = {**probs, "undecided": 0.0}
    turnout = float(np.clip(turnout, 0.0, 1.0))
    choice = "not_vote" if turnout < 0.5 else max(CES_VOTE_PROBABILITY_CODES, key=lambda code: vote_probs.get(code, 0.0))
    confidence = max([turnout, 1.0 - turnout, *vote_probs.values()])
    return format_turnout_vote_response(
        turnout_probability=turnout,
        vote_probabilities=vote_probs,
        most_likely_choice=choice,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
    )


class BenchmarkBaseline:
    name: str
    model_name: str

    def fit(self, train_ids: set[str], cohort: pd.DataFrame, answers: pd.DataFrame, targets_wide: pd.DataFrame) -> None:
        raise NotImplementedError

    def predict(self, eval_ids: set[str], cohort: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class MajorityByStateBaseline(BenchmarkBaseline):
    name = "majority_by_state"
    model_name = "majority_by_state_v1"

    def fit(self, train_ids: set[str], cohort: pd.DataFrame, answers: pd.DataFrame, targets_wide: pd.DataFrame) -> None:
        train = cohort[cohort["ces_id"].isin(train_ids)][["ces_id", "state_po"]].merge(targets_wide, on="ces_id", how="left")
        known_national = train[train["turnout_2024_self_report"].isin(["voted", "not_voted"])]
        self.national_turnout = (
            float((known_national["turnout_2024_self_report"] == "voted").mean()) if not known_national.empty else 0.5
        )
        self.national_vote = _safe_distribution(train["president_vote_2024"], CANDIDATE_CLASSES)
        self.turnout_by_state = {}
        self.vote_by_state = {}
        for state, group in train.groupby("state_po", dropna=False):
            known_turnout = group[group["turnout_2024_self_report"].isin(["voted", "not_voted"])]
            self.turnout_by_state[state] = (
                float((known_turnout["turnout_2024_self_report"] == "voted").mean())
                if not known_turnout.empty
                else self.national_turnout
            )
            self.vote_by_state[state] = _safe_distribution(group["president_vote_2024"], CANDIDATE_CLASSES)

    def predict(self, eval_ids: set[str], cohort: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in cohort[cohort["ces_id"].isin(eval_ids)].iterrows():
            turnout = self.turnout_by_state.get(row["state_po"], self.national_turnout)
            probs = self.vote_by_state.get(row["state_po"], self.national_vote)
            rows.append({"ces_id": row["ces_id"], "raw_response": _canonical_raw(turnout, probs), "model_name": self.model_name})
        return pd.DataFrame(rows, columns=["ces_id", "raw_response", "model_name"])


class PartyIdRuleBaseline(BenchmarkBaseline):
    name = "party_id_baseline"
    model_name = "party_id_baseline_v1"

    def fit(self, train_ids: set[str], cohort: pd.DataFrame, answers: pd.DataFrame, targets_wide: pd.DataFrame) -> None:
        _ = (train_ids, cohort, answers, targets_wide)

    def predict(self, eval_ids: set[str], cohort: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in cohort[cohort["ces_id"].isin(eval_ids)].iterrows():
            registered = str(row.get("registered_self_pre") or "unknown")
            turnout = 0.84 if registered == "yes" else 0.28 if registered == "no" else 0.62
            party = str(row.get("party_id_3_pre") or "unknown")
            ideology = str(row.get("ideology_3") or "unknown")
            if party == "democrat":
                probs = {"democrat": 0.78, "republican": 0.12, "other": 0.10}
            elif party == "republican":
                probs = {"democrat": 0.12, "republican": 0.78, "other": 0.10}
            elif ideology == "liberal":
                probs = {"democrat": 0.58, "republican": 0.24, "other": 0.18}
            elif ideology == "conservative":
                probs = {"democrat": 0.24, "republican": 0.58, "other": 0.18}
            else:
                probs = {"democrat": 0.42, "republican": 0.42, "other": 0.16}
            rows.append({"ces_id": row["ces_id"], "raw_response": _canonical_raw(turnout, probs), "model_name": self.model_name})
        return pd.DataFrame(rows, columns=["ces_id", "raw_response", "model_name"])


class SklearnLogitBenchmarkBaseline(BenchmarkBaseline):
    def __init__(self, name: str, feature_mode: str, feature_cache: dict[str, pd.DataFrame] | None = None):
        self.name = name
        self.model_name = f"{name}_v1"
        self.feature_mode = feature_mode
        self.feature_cache = feature_cache or {}

    @staticmethod
    def _model_pipeline():
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        return make_pipeline(DictVectorizer(sparse=True), LogisticRegression(max_iter=300, solver="liblinear"))

    def _records(self, rows: pd.DataFrame) -> list[dict[str, str]]:
        return rows[self.feature_columns].fillna("unknown").astype(str).to_dict("records")

    def fit(self, train_ids: set[str], cohort: pd.DataFrame, answers: pd.DataFrame, targets_wide: pd.DataFrame) -> None:
        self.features = self.feature_cache.get(self.feature_mode)
        if self.features is None:
            self.features = benchmark_feature_frame(cohort, answers, feature_mode=self.feature_mode)
            self.feature_cache[self.feature_mode] = self.features
        self.feature_columns = [col for col in self.features.columns if col != "ces_id"]
        train = self.features[self.features["ces_id"].isin(train_ids)].merge(targets_wide, on="ces_id", how="left")
        turnout_known = train[train["turnout_2024_self_report"].isin(["voted", "not_voted"])].copy()
        self.turnout_prior = float((turnout_known["turnout_2024_self_report"] == "voted").mean()) if not turnout_known.empty else 0.5
        if turnout_known["turnout_2024_self_report"].nunique() >= 2:
            self.turnout_model = self._model_pipeline()
            self.turnout_model.fit(self._records(turnout_known), (turnout_known["turnout_2024_self_report"] == "voted").astype(int))
        else:
            self.turnout_model = None
        vote_known = train[train["president_vote_2024"].isin(CANDIDATE_CLASSES)].copy()
        self.vote_prior = _safe_distribution(vote_known["president_vote_2024"], CANDIDATE_CLASSES)
        if vote_known["president_vote_2024"].nunique() >= 2:
            self.vote_model = self._model_pipeline()
            self.vote_model.fit(self._records(vote_known), vote_known["president_vote_2024"].astype(str))
        else:
            self.vote_model = None

    def _predict_turnout(self, rows: pd.DataFrame) -> np.ndarray:
        if self.turnout_model is None:
            return np.repeat(self.turnout_prior, len(rows))
        classes = list(self.turnout_model.classes_)
        probs = self.turnout_model.predict_proba(self._records(rows))
        return probs[:, classes.index(1)] if 1 in classes else np.zeros(len(rows))

    def _predict_vote(self, rows: pd.DataFrame) -> list[dict[str, float]]:
        if self.vote_model is None:
            return [dict(self.vote_prior) for _ in range(len(rows))]
        classes = [str(value) for value in self.vote_model.classes_]
        raw = self.vote_model.predict_proba(self._records(rows))
        out = []
        for row_probs in raw:
            probs = {code: 0.0 for code in CANDIDATE_CLASSES}
            for klass, value in zip(classes, row_probs, strict=False):
                if klass in probs:
                    probs[klass] = float(value)
            total = sum(probs.values()) or 1.0
            out.append({code: value / total for code, value in probs.items()})
        return out

    def predict(self, eval_ids: set[str], cohort: pd.DataFrame) -> pd.DataFrame:
        rows = self.features[self.features["ces_id"].isin(eval_ids)].copy()
        turnout = self._predict_turnout(rows)
        vote_probs = self._predict_vote(rows)
        return pd.DataFrame(
            {
                "ces_id": rows["ces_id"].astype(str).to_numpy(),
                "raw_response": [_canonical_raw(t, p) for t, p in zip(turnout, vote_probs, strict=False)],
                "model_name": self.model_name,
            }
        )


def _baseline_factory(name: str, feature_cache: dict[str, pd.DataFrame] | None = None) -> BenchmarkBaseline:
    if name == "majority_by_state":
        return MajorityByStateBaseline()
    if name == "party_id_baseline":
        return PartyIdRuleBaseline()
    if name == "sklearn_logit_demographic_only":
        return SklearnLogitBenchmarkBaseline(name, "demographic_only", feature_cache)
    if name == "sklearn_logit_pre_only":
        return SklearnLogitBenchmarkBaseline(name, "strict_pre", feature_cache)
    if name == "sklearn_logit_poll_informed":
        return SklearnLogitBenchmarkBaseline(name, "poll_informed", feature_cache)
    raise ValueError(f"Unknown benchmark baseline: {name}")


class ResponseCalibrator:
    def __init__(self) -> None:
        self.turnout: IsotonicRegression | None = None
        self.vote: dict[str, IsotonicRegression] = {}
        self.notes: list[str] = []

    def fit(self, predictions: pd.DataFrame, targets_wide: pd.DataFrame) -> None:
        if predictions.empty:
            self.notes.append("calibration_skipped_empty_dev_predictions")
            return
        merged = _parsed_predictions(predictions).merge(targets_wide, on="ces_id", how="left")
        known_turnout = merged[merged["turnout_2024_self_report"].isin(["voted", "not_voted"])].copy()
        if known_turnout["turnout_2024_self_report"].nunique() >= 2:
            y = (known_turnout["turnout_2024_self_report"] == "voted").astype(int)
            self.turnout = IsotonicRegression(out_of_bounds="clip")
            self.turnout.fit(known_turnout["turnout_probability"].astype(float), y)
        else:
            self.notes.append("turnout_calibration_skipped_missing_dev_classes")
        voter_rows = merged[merged["president_vote_2024"].isin(CANDIDATE_CLASSES)].copy()
        for code in CANDIDATE_CLASSES:
            y = (voter_rows["president_vote_2024"] == code).astype(int)
            col = f"vote_prob_{code}"
            if len(voter_rows) and y.nunique() >= 2:
                model = IsotonicRegression(out_of_bounds="clip")
                model.fit(voter_rows[col].astype(float), y)
                self.vote[code] = model
            else:
                self.notes.append(f"vote_calibration_skipped_missing_dev_class:{code}")

    def apply(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if predictions.empty:
            return pd.DataFrame(columns=["ces_id", "raw_response", "model_name"])
        parsed = _parsed_predictions(predictions)
        out_rows = []
        for _, row in parsed.iterrows():
            turnout = float(row["turnout_probability"])
            if self.turnout is not None:
                turnout = float(self.turnout.predict([turnout])[0])
            probs = {code: float(row[f"vote_prob_{code}"]) for code in CANDIDATE_CLASSES}
            for code, model in self.vote.items():
                probs[code] = float(model.predict([probs[code]])[0])
            total = sum(max(0.0, value) for value in probs.values())
            probs = {code: (max(0.0, value) / total if total else 1 / 3) for code, value in probs.items()}
            out_rows.append(
                {
                    "ces_id": row["ces_id"],
                    "raw_response": _canonical_raw(turnout, probs),
                    "model_name": row["model_name"],
                }
            )
        return pd.DataFrame(out_rows)


def _parsed_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in predictions.iterrows():
        parsed = parse_turnout_vote_json(row["raw_response"])
        rows.append(
            {
                "ces_id": str(row["ces_id"]),
                "model_name": row.get("model_name"),
                **parsed,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "ces_id",
            "model_name",
            "parse_status",
            "turnout_probability",
            "vote_prob_democrat",
            "vote_prob_republican",
            "vote_prob_other",
            "vote_prob_undecided",
            "most_likely_choice",
            "confidence",
        ],
    )


def _response_rows(
    *,
    run_id: str,
    baseline: str,
    predictions: pd.DataFrame,
    cohort: pd.DataFrame,
    prediction_scope: str,
    fold: int | None = None,
    is_llm: bool = False,
    prompt_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cohort_index = cohort.set_index("ces_id", drop=False)
    rows = []
    for _, pred in predictions.iterrows():
        ces_id = str(pred["ces_id"])
        parsed = parse_turnout_vote_json(pred["raw_response"])
        source = cohort_index.loc[ces_id]
        prompt_meta = (prompt_by_id or {}).get(ces_id, {})
        rows.append(
            {
                "run_id": run_id,
                "response_id": stable_hash(run_id, baseline, ces_id, prediction_scope, fold, length=20),
                "prompt_id": prompt_meta.get("prompt_id"),
                "agent_id": f"{run_id}_ces_{int(source['_agent_ordinal']):06d}",
                "source_respondent_id": ces_id,
                "base_ces_id": ces_id,
                "question_id": "president_turnout_vote_2024",
                "task_id": "president_turnout_vote",
                "baseline": baseline,
                "model_name": pred.get("model_name"),
                "raw_response": pred["raw_response"],
                "parse_status": parsed["parse_status"],
                "turnout_probability": parsed["turnout_probability"],
                "vote_prob_democrat": parsed["vote_prob_democrat"],
                "vote_prob_republican": parsed["vote_prob_republican"],
                "vote_prob_other": parsed["vote_prob_other"],
                "vote_prob_undecided": parsed["vote_prob_undecided"],
                "most_likely_choice": parsed["most_likely_choice"],
                "confidence": parsed["confidence"],
                "prediction_scope": prediction_scope,
                "split": source.get("split"),
                "split_bucket": int(source.get("split_bucket")),
                "fold": fold if fold is not None else int(source.get("fold")),
                "sample_weight": float(source.get("sample_weight", 1.0)),
                "is_llm": is_llm,
                "cache_hit": prompt_meta.get("cache_hit"),
                "latency_ms": prompt_meta.get("latency_ms"),
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    return rows


def _combined_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    turnout = df["turnout_probability"].fillna(0.0).astype(float).clip(0.0, 1.0)
    out = pd.DataFrame(index=df.index)
    out["democrat"] = turnout * df["vote_prob_democrat"].fillna(0.0).astype(float).clip(0.0, 1.0)
    out["republican"] = turnout * df["vote_prob_republican"].fillna(0.0).astype(float).clip(0.0, 1.0)
    out["other"] = turnout * df["vote_prob_other"].fillna(0.0).astype(float).clip(0.0, 1.0)
    undecided = df["vote_prob_undecided"].fillna(0.0).astype(float).clip(0.0, 1.0)
    out["not_vote"] = (1.0 - turnout) + turnout * undecided
    denom = out.sum(axis=1).replace(0.0, 1.0)
    return out.div(denom, axis=0)


def _weighted_mean(values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray | None) -> float:
    arr = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.mean(arr)) if len(arr) else np.nan
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(w))
    return float(np.sum(arr * w) / denom) if denom else np.nan


def _safe_weighted_auc(y: pd.Series, p: pd.Series, weights: pd.Series | None) -> float | None:
    if y.nunique() < 2:
        return None
    try:
        return float(roc_auc_score(y, p, sample_weight=weights))
    except ValueError:
        return None


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    weights: pd.Series | np.ndarray | None = None,
    *,
    n_bins: int = 10,
) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(np.sum(w)) or 1.0
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:], strict=False):
        if high == 1.0:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)
        if not mask.any():
            continue
        bin_w = w[mask]
        conf = _weighted_mean(p[mask], bin_w)
        acc = _weighted_mean(y[mask], bin_w)
        ece += float(np.sum(bin_w)) / total * abs(conf - acc)
    return float(ece)


def multiclass_brier(
    y_true: pd.Series,
    probs: pd.DataFrame,
    weights: pd.Series | None = None,
    classes: list[str] | None = None,
) -> float:
    classes = classes or VOTE_CLASSES
    onehot = np.zeros((len(y_true), len(classes)))
    class_index = {code: idx for idx, code in enumerate(classes)}
    for row_idx, value in enumerate(y_true.astype(str)):
        if value in class_index:
            onehot[row_idx, class_index[value]] = 1.0
    losses = ((probs[classes].to_numpy() - onehot) ** 2).sum(axis=1)
    return _weighted_mean(losses, weights)


def _metric_row(
    run_id: str,
    baseline: str,
    model_name: str,
    metric_scope: str,
    metric_name: str,
    metric_value: float | None,
    *,
    weighted: bool,
    group_key: str | None = None,
    n: int | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "baseline": baseline,
        "model_name": model_name,
        "metric_scope": metric_scope,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "weighted": weighted,
        "group_key": group_key,
        "n": n,
        "created_at": pd.Timestamp.now(tz="UTC"),
    }


def benchmark_metric_rows(
    responses: pd.DataFrame,
    cohort: pd.DataFrame,
    targets: pd.DataFrame,
    run_id: str,
    *,
    metric_scope: str,
    group_key: str | None = None,
) -> list[dict[str, Any]]:
    targets_wide = _target_wide(targets)
    merged = responses.merge(targets_wide, left_on="base_ces_id", right_on="ces_id", how="left", suffixes=("", "_target"))
    subgroup_cols = [col for col in DEFAULT_SUBGROUPS if col in cohort.columns]
    merged = merged.merge(cohort[["ces_id", *subgroup_cols]], left_on="base_ces_id", right_on="ces_id", how="left", suffixes=("", "_cohort"))
    rows = []
    for (baseline, model_name), group in merged.groupby(["baseline", "model_name"], dropna=False):
        for weighted in [False, True]:
            weights = group["sample_weight"] if weighted and "sample_weight" in group.columns else None
            parse_ok = (group["parse_status"] == "ok").astype(float)
            rows.append(_metric_row(run_id, baseline, model_name, metric_scope, "parse_ok_rate", _weighted_mean(parse_ok, weights), weighted=weighted, group_key=group_key, n=len(group)))
            known_turnout = group[group["turnout_2024_self_report"].isin(["voted", "not_voted"])].copy()
            if not known_turnout.empty:
                turnout_y = (known_turnout["turnout_2024_self_report"] == "voted").astype(int)
                turnout_p = known_turnout["turnout_probability"].fillna(0.0).astype(float).clip(0, 1)
                tw = known_turnout["sample_weight"] if weighted else None
                rows.extend(
                    [
                        _metric_row(run_id, baseline, model_name, metric_scope, "turnout_brier", _weighted_mean((turnout_p - turnout_y) ** 2, tw), weighted=weighted, group_key=group_key, n=len(known_turnout)),
                        _metric_row(run_id, baseline, model_name, metric_scope, "turnout_auc", _safe_weighted_auc(turnout_y, turnout_p, tw), weighted=weighted, group_key=group_key, n=len(known_turnout)),
                        _metric_row(run_id, baseline, model_name, metric_scope, "turnout_ece", expected_calibration_error(turnout_y, turnout_p, tw), weighted=weighted, group_key=group_key, n=len(known_turnout)),
                        _metric_row(run_id, baseline, model_name, metric_scope, "turnout_accuracy_at_0_5", _weighted_mean(((turnout_p >= 0.5) == (turnout_y == 1)).astype(float), tw), weighted=weighted, group_key=group_key, n=len(known_turnout)),
                    ]
                )
            known_vote = group[group["president_vote_2024"].isin(VOTE_CLASSES)].copy()
            if not known_vote.empty:
                probs = _combined_probabilities(known_vote)[VOTE_CLASSES]
                y_true = known_vote["president_vote_2024"].astype(str)
                y_pred = probs.idxmax(axis=1)
                vw = known_vote["sample_weight"] if weighted else None
                log_loss_value = None
                try:
                    ordered_classes = sorted(VOTE_CLASSES)
                    log_loss_value = float(
                        log_loss(y_true, probs[ordered_classes], labels=ordered_classes, sample_weight=vw)
                    )
                except ValueError:
                    log_loss_value = None
                rows.extend(
                    [
                        _metric_row(run_id, baseline, model_name, metric_scope, "vote_accuracy", _weighted_mean((y_pred == y_true).astype(float), vw), weighted=weighted, group_key=group_key, n=len(known_vote)),
                        _metric_row(run_id, baseline, model_name, metric_scope, "vote_macro_f1", float(f1_score(y_true, y_pred, labels=VOTE_CLASSES, average="macro", zero_division=0, sample_weight=vw)), weighted=weighted, group_key=group_key, n=len(known_vote)),
                        _metric_row(run_id, baseline, model_name, metric_scope, "vote_log_loss", log_loss_value, weighted=weighted, group_key=group_key, n=len(known_vote)),
                        _metric_row(run_id, baseline, model_name, metric_scope, "vote_brier_multiclass", multiclass_brier(y_true, probs, vw), weighted=weighted, group_key=group_key, n=len(known_vote)),
                    ]
                )
                confusion = pd.crosstab(y_true, y_pred)
                for truth in VOTE_CLASSES:
                    for pred in VOTE_CLASSES:
                        rows.append(
                            _metric_row(
                                run_id,
                                baseline,
                                model_name,
                                metric_scope,
                                "vote_confusion_count",
                                float(confusion.get(pred, pd.Series(dtype=float)).get(truth, 0.0)),
                                weighted=weighted,
                                group_key=f"{group_key + ';' if group_key else ''}truth={truth};pred={pred}",
                                n=len(known_vote),
                            )
                        )
    return rows


def compute_subgroup_metrics(responses: pd.DataFrame, cohort: pd.DataFrame, targets: pd.DataFrame, run_id: str) -> pd.DataFrame:
    rows = []
    cohort_index = cohort.set_index("ces_id", drop=False)
    response_ids = responses["base_ces_id"].astype(str)
    for subgroup in DEFAULT_SUBGROUPS:
        if subgroup not in cohort.columns:
            continue
        values = response_ids.map(cohort_index[subgroup])
        for value in sorted(values.dropna().astype(str).unique()):
            mask = values.astype(str) == value
            rows.extend(
                benchmark_metric_rows(
                    responses[mask].copy(),
                    cohort,
                    targets,
                    run_id,
                    metric_scope="subgroup",
                    group_key=f"{subgroup}={value}",
                )
            )
    return pd.DataFrame(rows)


def turnout_calibration_bins(responses: pd.DataFrame, targets: pd.DataFrame, run_id: str, n_bins: int = 10) -> pd.DataFrame:
    targets_wide = _target_wide(targets)
    merged = responses.merge(targets_wide, left_on="base_ces_id", right_on="ces_id", how="left")
    merged = merged[merged["turnout_2024_self_report"].isin(["voted", "not_voted"])].copy()
    rows = []
    for (baseline, model_name), group in merged.groupby(["baseline", "model_name"], dropna=False):
        y = (group["turnout_2024_self_report"] == "voted").astype(float)
        p = group["turnout_probability"].fillna(0.0).astype(float).clip(0, 1)
        weights = group["sample_weight"].fillna(1.0).astype(float)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        for idx, (low, high) in enumerate(zip(bins[:-1], bins[1:], strict=False)):
            upper = p <= high if high == 1.0 else p < high
            mask = (p >= low) & upper
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
                    "mean_predicted": _weighted_mean(p[mask], weights[mask]),
                    "observed_rate": _weighted_mean(y[mask], weights[mask]),
                    "created_at": pd.Timestamp.now(tz="UTC"),
                }
            )
    return pd.DataFrame(rows)


def _run_non_llm_predictions(
    *,
    run_id: str,
    baseline_names: list[str],
    cohort: pd.DataFrame,
    answers: pd.DataFrame,
    targets_wide: pd.DataFrame,
    eval_ids: set[str],
    train_ids: set[str],
    dev_ids: set[str],
    prediction_scope: str,
    fold: int | None = None,
    feature_cache: dict[str, pd.DataFrame] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for name in baseline_names:
        baseline = _baseline_factory(name, feature_cache)
        baseline.fit(train_ids, cohort, answers, targets_wide)
        dev_predictions = baseline.predict(dev_ids, cohort)
        calibrator = ResponseCalibrator()
        calibrator.fit(dev_predictions, targets_wide)
        notes.extend([f"{name}:{note}" for note in calibrator.notes])
        eval_predictions = calibrator.apply(baseline.predict(eval_ids, cohort))
        rows.extend(
            _response_rows(
                run_id=run_id,
                baseline=name,
                predictions=eval_predictions,
                cohort=cohort,
                prediction_scope=prediction_scope,
                fold=fold,
                is_llm=False,
            )
        )
    return rows, notes


def _group_memory_facts(memory_facts: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if memory_facts.empty:
        return {}
    work = memory_facts.copy()
    work["ces_id"] = work["ces_id"].astype(str)
    return {ces_id: group.copy() for ces_id, group in work.groupby("ces_id", sort=False)}


def _group_context(context: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if context.empty:
        return {}
    work = context.copy()
    work["ces_id"] = work["ces_id"].astype(str)
    return {ces_id: group.to_dict("records") for ces_id, group in work.groupby("ces_id", sort=False)}


def _post_hoc_oracle_prompt(agent: pd.Series, targets_wide: pd.DataFrame) -> str:
    target = targets_wide.set_index("ces_id").loc[str(agent["base_ces_id"])]
    turnout = target.get("turnout_2024_self_report", "unknown")
    vote = target.get("president_vote_2024", "unknown")
    return f"""You are simulating a CES respondent for a leakage diagnostic.

Voter profile:
- State: {agent.get('state_po') or 'unknown'}
- Age group: {agent.get('age_group') or 'unknown'}
- Gender: {agent.get('gender') or 'unknown'}
- Race/ethnicity: {agent.get('race_ethnicity') or 'unknown'}
- Education: {agent.get('education_binary') or 'unknown'}
- Party identification: {agent.get('party_id_3') or 'unknown'}
- Ideology: {agent.get('ideology_3') or 'unknown'}

Post-election diagnostic information:
- The respondent's post-election turnout label is: {turnout}
- The respondent's post-election presidential vote label is: {vote}

Return JSON only with this schema:
{{
  "turnout_probability": 0.0,
  "vote_probabilities": {{
    "democrat": 0.0,
    "republican": 0.0,
    "other": 0.0,
    "undecided": 0.0
  }},
  "most_likely_choice": "democrat|republican|other|undecided|not_vote",
  "confidence": 0.0
}}
"""


def _post_hoc_oracle_raw(agent: pd.Series, targets_wide: pd.DataFrame) -> str:
    """Leakage diagnostic: emit the canonical response implied by post labels."""

    target = targets_wide.set_index("ces_id").loc[str(agent["base_ces_id"])]
    turnout_label = str(target.get("turnout_2024_self_report", "unknown"))
    vote_label = str(target.get("president_vote_2024", "unknown"))
    if turnout_label == "voted":
        turnout = 0.99
    elif turnout_label == "not_voted":
        turnout = 0.01
    else:
        turnout = 0.5
    if vote_label in CANDIDATE_CLASSES:
        probs = {code: 0.005 for code in CANDIDATE_CLASSES}
        probs[vote_label] = 0.99
    elif vote_label == "not_vote":
        probs = {code: 1.0 / len(CANDIDATE_CLASSES) for code in CANDIDATE_CLASSES}
        turnout = min(turnout, 0.01)
    else:
        probs = {code: 1.0 / len(CANDIDATE_CLASSES) for code in CANDIDATE_CLASSES}
    return _canonical_raw(turnout, probs)


def select_llm_pilot_ids(cohort: pd.DataFrame, targets: pd.DataFrame, *, cap: int, per_state_target: int, seed: int) -> list[str]:
    test = cohort[cohort["split"] == "test"].copy()
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for _, state_group in test.groupby("state_po", sort=True):
        parts = []
        party_groups = list(state_group.groupby("party_id_3_pre", dropna=False, sort=True))
        per_party = max(1, int(np.ceil(per_state_target / max(1, len(party_groups)))))
        for _, party_group in party_groups:
            n = min(per_party, len(party_group))
            parts.extend(party_group.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1)))["ces_id"].astype(str).tolist())
        selected.extend(parts[:per_state_target])
    targets_wide = _target_wide(targets)
    diagnostic = test.merge(targets_wide, on="ces_id", how="left")
    diagnostic = diagnostic[
        diagnostic["president_vote_2024"].isin(["not_vote", "other"])
        | diagnostic["party_id_3_pre"].isin(["unknown", "independent_or_other"])
    ]
    if not diagnostic.empty:
        n_diag = min(max(0, cap - len(set(selected))), len(diagnostic))
        selected.extend(
            diagnostic.sample(n=n_diag, random_state=seed + 17)["ces_id"].astype(str).tolist()
        )
    unique = list(dict.fromkeys(selected))
    if len(unique) > cap:
        unique = list(pd.Series(unique).sample(n=cap, random_state=seed).tolist())
    return unique


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[row["cache_key"]] = row
    return cache


def _append_cache(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _llm_prompt_for_baseline(
    baseline: str,
    agent: pd.Series,
    question: pd.Series,
    *,
    strict_memory: dict[str, pd.DataFrame],
    poll_memory: dict[str, pd.DataFrame],
    context: dict[str, list[dict[str, Any]]],
    targets_wide: pd.DataFrame,
    max_memory_facts: int,
) -> tuple[str, list[str]]:
    if baseline == "ces_demographic_only_llm":
        return build_ces_prompt(agent, question, memory_facts=strict_memory, context=context, prompt_mode="ces_demographic_only", memory_policy="strict_pre_no_vote_v1", max_memory_facts=max_memory_facts)
    if baseline == "ces_party_ideology_llm":
        return build_ces_prompt(agent, question, memory_facts=strict_memory, context=context, prompt_mode="ces_party_ideology", memory_policy="strict_pre_no_vote_v1", max_memory_facts=max_memory_facts)
    if baseline == "ces_survey_memory_llm_strict":
        return build_ces_prompt(agent, question, memory_facts=strict_memory, context=context, prompt_mode="ces_survey_memory", memory_policy="strict_pre_no_vote_v1", max_memory_facts=max_memory_facts)
    if baseline == "ces_survey_memory_llm_poll_informed":
        return build_ces_prompt(agent, question, memory_facts=poll_memory, context=context, prompt_mode="ces_poll_informed", memory_policy="poll_informed_pre_v1", max_memory_facts=max_memory_facts)
    if baseline == "post_hoc_oracle_llm":
        return _post_hoc_oracle_prompt(agent, targets_wide), []
    raise ValueError(f"Unknown LLM baseline: {baseline}")


def _run_llm_pilot(
    *,
    cfg: dict[str, Any],
    paths: BenchmarkPaths,
    cohort: pd.DataFrame,
    agents: pd.DataFrame,
    targets: pd.DataFrame,
    question: pd.Series,
    context: pd.DataFrame,
    strict_memory_facts: pd.DataFrame,
    poll_memory_facts: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    llm_cfg = cfg.get("llm_pilot", {})
    baseline_names = [name for name in cfg.get("baselines", {}).get("llm", LLM_BASELINES) if name in LLM_BASELINES]
    if not baseline_names:
        return [], pd.DataFrame(columns=PROMPT_COLUMNS), {"llm_enabled": False}
    cap = int(llm_cfg.get("max_respondents", 750))
    per_state_target = int(llm_cfg.get("per_state_target", 10))
    timing_responses = int(llm_cfg.get("timing_responses", 50))
    threshold = float(llm_cfg.get("timing_threshold_seconds", 6.0))
    max_runtime_minutes = float(llm_cfg.get("max_runtime_minutes", 60.0))
    seed = int(cfg.get("seed", 20260426))
    pilot_ids = select_llm_pilot_ids(cohort, targets, cap=cap, per_state_target=per_state_target, seed=seed)

    model_cfg = ModelConfig.model_validate(cfg.get("model", {}))
    client = build_llm_client(model_cfg)
    model_name = getattr(client, "model_name", model_cfg.model_name)
    cache = _load_cache(paths.cache_path)
    strict_memory = _group_memory_facts(strict_memory_facts)
    poll_memory = _group_memory_facts(poll_memory_facts)
    context_by_id = _group_context(context)
    targets_wide = _target_wide(targets)
    agents_by_id = agents.set_index("base_ces_id", drop=False)

    prompt_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    observed_latencies: list[float] = []
    cache_hits = 0
    ollama_calls = 0
    deterministic_rows = 0
    effective_cap = cap
    timing_checked = False
    processed_pairs = 0
    for baseline in baseline_names:
        for ces_id in pilot_ids:
            if pilot_ids.index(ces_id) >= effective_cap:
                continue
            agent = agents_by_id.loc[ces_id]
            prompt_text, fact_ids = _llm_prompt_for_baseline(
                baseline,
                agent,
                question,
                strict_memory=strict_memory,
                poll_memory=poll_memory,
                context=context_by_id,
                targets_wide=targets_wide,
                max_memory_facts=int(cfg.get("memory", {}).get("max_memory_facts", 24)),
            )
            response_model_name = "post_hoc_oracle_v1" if baseline == "post_hoc_oracle_llm" else model_name
            prompt_temperature = 0.0 if baseline == "post_hoc_oracle_llm" else model_cfg.temperature
            prompt_hash = stable_hash(response_model_name, baseline, prompt_text, prompt_temperature, length=32)
            cache_key = stable_hash(cfg["run_id"], response_model_name, baseline, prompt_hash, length=40)
            cache_hit = cache_key in cache
            latency_ms = None
            if baseline == "post_hoc_oracle_llm":
                raw = _post_hoc_oracle_raw(agent, targets_wide)
                latency_ms = 0
                cache_hit = False
                deterministic_rows += 1
            elif cache_hit:
                raw = cache[cache_key]["raw_response"]
                latency_ms = cache[cache_key].get("latency_ms")
                cache_hits += 1
                if latency_ms is not None:
                    observed_latencies.append(float(latency_ms) / 1000.0)
            else:
                started = time.time()
                raw = client.complete(prompt_text, ["democrat", "republican", "other", "undecided", "not_vote"])
                latency_ms = int((time.time() - started) * 1000)
                latencies.append(latency_ms / 1000.0)
                observed_latencies.append(latency_ms / 1000.0)
                ollama_calls += 1
                cache_row = {
                    "cache_key": cache_key,
                    "run_id": cfg["run_id"],
                    "model_name": response_model_name,
                    "baseline": baseline,
                    "prompt_hash": prompt_hash,
                    "raw_response": raw,
                    "latency_ms": latency_ms,
                    "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
                }
                cache[cache_key] = cache_row
                _append_cache(paths.cache_path, cache_row)
            prompt_id = stable_hash(cfg["run_id"], baseline, ces_id, prompt_hash, length=20)
            prompt_rows.append(
                {
                    "run_id": cfg["run_id"],
                    "prompt_id": prompt_id,
                    "agent_id": agent["agent_id"],
                    "base_ces_id": ces_id,
                    "baseline": baseline,
                    "model_name": response_model_name,
                    "prompt_hash": prompt_hash,
                    "prompt_text": prompt_text,
                    "memory_fact_ids_used": fact_ids,
                    "cache_hit": cache_hit,
                    "created_at": pd.Timestamp.now(tz="UTC"),
                }
            )
            response_rows.extend(
                _response_rows(
                    run_id=cfg["run_id"],
                    baseline=baseline,
                    predictions=pd.DataFrame([{"ces_id": ces_id, "raw_response": raw, "model_name": response_model_name}]),
                    cohort=cohort,
                    prediction_scope="llm_pilot",
                    is_llm=True,
                    prompt_by_id={ces_id: {"prompt_id": prompt_id, "cache_hit": cache_hit, "latency_ms": latency_ms}},
                )
            )
            processed_pairs += 1
            if not timing_checked and len(latencies) >= timing_responses:
                median_latency = statistics.median(latencies)
                if median_latency > threshold:
                    effective_cap = min(effective_cap, int(llm_cfg.get("slow_cap_respondents", 300)))
                projected = median_latency * len(baseline_names) * effective_cap / 60.0
                if projected > max_runtime_minutes:
                    runtime_cap = max(25, int(max_runtime_minutes * 60.0 / (median_latency * len(baseline_names))))
                    effective_cap = min(effective_cap, runtime_cap)
                timing_checked = True
    metadata = {
        "llm_enabled": True,
        "requested_cap": cap,
        "effective_cap": effective_cap,
        "n_pilot_ids_initial": len(pilot_ids),
        "n_response_rows": len(response_rows),
        "median_latency_seconds": statistics.median(observed_latencies) if observed_latencies else None,
        "cache_hits": cache_hits,
        "ollama_calls": ollama_calls,
        "deterministic_rows": deterministic_rows,
        "timing_checked": timing_checked,
        "processed_pairs": processed_pairs,
    }
    return response_rows, pd.DataFrame(prompt_rows, columns=PROMPT_COLUMNS), metadata


def _save_figure(fig: Any, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")


def write_benchmark_figures(
    *,
    run_dir: Path,
    metrics: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    calibration_bins: pd.DataFrame,
    responses: pd.DataFrame,
    targets: pd.DataFrame,
    aggregate: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure_dir = ensure_dir(run_dir / "figures")
    written: list[Path] = []
    core = metrics[
        (metrics["metric_scope"] == "individual")
        & (metrics["weighted"].astype(bool))
        & (metrics["metric_name"].isin(["turnout_brier", "turnout_auc", "vote_accuracy", "vote_macro_f1", "vote_log_loss"]))
    ].copy()
    if not core.empty:
        pivot = core.pivot_table(index="baseline", columns="metric_name", values="metric_value", aggfunc="first")
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(pivot))))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title("Weighted Individual Metrics by Baseline")
        _save_figure(fig, figure_dir / "baseline_metric_heatmap")
        written.append(figure_dir / "baseline_metric_heatmap.png")
        plt.close(fig)
    for name, pairs in {
        "llm_memory_delta": ("ces_demographic_only_llm", "ces_survey_memory_llm_strict"),
        "party_ideology_increment": ("ces_demographic_only_llm", "ces_party_ideology_llm"),
        "strict_vs_poll_informed": ("ces_survey_memory_llm_strict", "ces_survey_memory_llm_poll_informed"),
    }.items():
        base, comp = pairs
        subset = core[core["baseline"].isin([base, comp])]
        if subset["baseline"].nunique() == 2:
            wide = subset.pivot_table(index="metric_name", columns="baseline", values="metric_value", aggfunc="first")
            wide["delta"] = wide[comp] - wide[base]
            fig, ax = plt.subplots(figsize=(8, 4))
            wide["delta"].sort_values().plot(kind="barh", ax=ax)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(name.replace("_", " ").title())
            _save_figure(fig, figure_dir / name)
            written.append(figure_dir / f"{name}.png")
            plt.close(fig)
    if not calibration_bins.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        for baseline, group in calibration_bins.groupby("baseline"):
            ax.plot(group["mean_predicted"], group["observed_rate"], marker="o", label=baseline)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("Mean predicted turnout")
        ax.set_ylabel("Observed turnout")
        ax.set_title("Turnout Calibration")
        ax.legend(fontsize=7)
        _save_figure(fig, figure_dir / "turnout_calibration_curves")
        written.append(figure_dir / "turnout_calibration_curves.png")
        plt.close(fig)
    ece = core[core["metric_name"] == "turnout_ece"] if "turnout_ece" in set(core["metric_name"]) else metrics[
        (metrics["metric_name"] == "turnout_ece") & (metrics["weighted"].astype(bool))
    ]
    if not ece.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ece.sort_values("metric_value").plot(kind="bar", x="baseline", y="metric_value", ax=ax, legend=False)
        ax.set_title("Weighted Turnout ECE")
        ax.set_ylabel("ECE")
        ax.tick_params(axis="x", rotation=45)
        _save_figure(fig, figure_dir / "turnout_ece_bars")
        written.append(figure_dir / "turnout_ece_bars.png")
        plt.close(fig)
    targets_wide = _target_wide(targets)
    merged = responses.merge(targets_wide, left_on="base_ces_id", right_on="ces_id", how="left")
    for baseline in ["ces_survey_memory_llm_strict", "sklearn_logit_pre_only"]:
        part = merged[(merged["baseline"] == baseline) & (merged["president_vote_2024"].isin(VOTE_CLASSES))]
        if part.empty:
            continue
        probs = _combined_probabilities(part)[VOTE_CLASSES]
        confusion = pd.crosstab(part["president_vote_2024"], probs.idxmax(axis=1)).reindex(index=VOTE_CLASSES, columns=VOTE_CLASSES, fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion, annot=True, fmt=".0f", cmap="Blues", ax=ax)
        ax.set_title(f"Vote Confusion: {baseline}")
        _save_figure(fig, figure_dir / f"confusion_{baseline}")
        written.append(figure_dir / f"confusion_{baseline}.png")
        plt.close(fig)
    minority = subgroup_metrics[
        subgroup_metrics["metric_name"].isin(["vote_accuracy", "vote_macro_f1"])
        & subgroup_metrics["weighted"].astype(bool)
        & subgroup_metrics["group_key"].fillna("").str.startswith(("race_ethnicity=", "state_party_id_3="))
    ].head(80)
    if not minority.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot = minority[minority["metric_name"] == "vote_accuracy"].copy()
        plot["label"] = plot["baseline"].astype(str) + " | " + plot["group_key"].astype(str)
        plot.sort_values("metric_value").tail(25).plot(kind="barh", x="label", y="metric_value", ax=ax, legend=False)
        ax.set_title("Subgroup Vote Accuracy Diagnostics")
        _save_figure(fig, figure_dir / "minority_nonvoter_third_party_panel")
        written.append(figure_dir / "minority_nonvoter_third_party_panel.png")
        plt.close(fig)
    if not aggregate.empty and not aggregate_metrics.empty:
        state_errors = aggregate_metrics[aggregate_metrics["metric_name"] == "state_margin_error"].copy()
        if not state_errors.empty:
            state_errors["abs_error"] = state_errors["metric_value"].abs()
            plot = (
                state_errors.sort_values(["baseline", "abs_error"], ascending=[True, False])
                .groupby("baseline", group_keys=False)
                .head(12)
                .sort_values("metric_value")
            )
            fig, ax = plt.subplots(figsize=(10, max(5, min(14, 0.28 * len(plot)))))
            plot["label"] = plot["baseline"].astype(str) + " | " + plot["state_po"].astype(str)
            plot.plot(kind="barh", x="label", y="metric_value", ax=ax, legend=False)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title("Cross-Fit State Margin Error (Largest Absolute Errors)")
            _save_figure(fig, figure_dir / "state_aggregate_margin_error")
            written.append(figure_dir / "state_aggregate_margin_error.png")
            plt.close(fig)
    return written


def _answer_research_questions(metrics: pd.DataFrame, subgroup_metrics: pd.DataFrame) -> list[str]:
    weighted = metrics[(metrics["metric_scope"] == "individual") & (metrics["weighted"].astype(bool))]

    def metric(baseline: str, name: str) -> float | None:
        row = weighted[(weighted["baseline"] == baseline) & (weighted["metric_name"] == name)]
        return None if row.empty else float(row["metric_value"].iloc[0])

    lines = []
    demo_acc = metric("ces_demographic_only_llm", "vote_accuracy")
    strict_acc = metric("ces_survey_memory_llm_strict", "vote_accuracy")
    if demo_acc is not None and strict_acc is not None:
        lines.append(f"- Strict survey-memory LLM vs demographic-only LLM vote accuracy delta: {strict_acc - demo_acc:+.3f}.")
    party_acc = metric("ces_party_ideology_llm", "vote_accuracy")
    if demo_acc is not None and party_acc is not None:
        lines.append(f"- Party/ideology LLM increment over demographic-only vote accuracy: {party_acc - demo_acc:+.3f}.")
    sklearn_acc = metric("sklearn_logit_pre_only", "vote_accuracy")
    if strict_acc is not None and sklearn_acc is not None:
        lines.append(f"- Strict survey-memory LLM vs sklearn pre-only vote accuracy delta: {strict_acc - sklearn_acc:+.3f}.")
    poll_acc = metric("ces_survey_memory_llm_poll_informed", "vote_accuracy")
    if strict_acc is not None and poll_acc is not None:
        lines.append(f"- Poll-informed LLM increment over strict survey memory vote accuracy: {poll_acc - strict_acc:+.3f}.")
    minority = subgroup_metrics[
        (subgroup_metrics["metric_name"] == "vote_accuracy")
        & (subgroup_metrics["weighted"].astype(bool))
        & subgroup_metrics["group_key"].fillna("").str.contains("race_ethnicity=|truth=other|truth=not_vote", regex=True)
    ]
    if not minority.empty:
        worst = minority.sort_values("metric_value").head(5)
        labels = "; ".join(f"{r['baseline']} {r['group_key']}={float(r['metric_value']):.3f}" for _, r in worst.iterrows())
        lines.append(f"- Lowest diagnostic subgroup/class vote accuracy rows: {labels}.")
    if not lines:
        lines.append("- LLM comparison rows are unavailable; inspect non-LLM full-test metrics and any partial LLM cache output.")
    return lines


def write_benchmark_report(
    *,
    run_dir: Path,
    cfg: dict[str, Any],
    cohort: pd.DataFrame,
    responses: pd.DataFrame,
    metrics: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    figures: list[Path],
    calibration_notes: list[str],
    llm_metadata: dict[str, Any],
) -> Path:
    def table(df: pd.DataFrame, cols: list[str], head: int = 40) -> str:
        if df.empty:
            return "_No rows._"
        show = df[[col for col in cols if col in df.columns]].head(head).copy()
        for col in show.columns:
            show[col] = show[col].map(lambda value: "" if pd.isna(value) else str(value))
        lines = ["| " + " | ".join(show.columns) + " |", "| " + " | ".join("---" for _ in show.columns) + " |"]
        for _, row in show.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in show.columns) + " |")
        return "\n".join(lines)

    split_counts = cohort.groupby("split").size().reset_index(name="n")
    baseline_summary = responses.groupby(["baseline", "model_name", "prediction_scope", "parse_status"], dropna=False).size().reset_index(name="n")
    metric_summary = metrics[
        (metrics["metric_scope"] == "individual")
        & (metrics["weighted"].astype(bool))
        & metrics["metric_name"].isin(["turnout_brier", "turnout_auc", "turnout_ece", "vote_accuracy", "vote_macro_f1", "vote_log_loss", "vote_brier_multiclass"])
    ].sort_values(["baseline", "metric_name"])
    figure_rows = pd.DataFrame({"figure": [str(path.relative_to(run_dir)) for path in figures]})
    lines = [
        f"# CES Respondent-Level Individual Benchmark: {cfg['run_id']}",
        "",
        "## Run Summary",
        f"- Eligible respondents: {len(cohort):,}",
        f"- States/DC: {cohort['state_po'].nunique()}",
        f"- Seed: `{cfg.get('seed', 20260426)}`",
        f"- LLM model: `{cfg.get('model', {}).get('model_name', 'qwen3.5:0.8b')}`",
        f"- LLM metadata: `{stable_json(llm_metadata)}`",
        "",
        "## Split Counts",
        table(split_counts, ["split", "n"]),
        "",
        "## Baseline / Parse Summary",
        table(baseline_summary, ["baseline", "model_name", "prediction_scope", "parse_status", "n"], head=80),
        "",
        "## Research Question Readout",
        "\n".join(_answer_research_questions(metrics, subgroup_metrics)),
        "",
        "## Weighted Individual Metrics",
        table(metric_summary, ["baseline", "model_name", "metric_name", "metric_value", "n"], head=120),
        "",
        "## Cross-Fit Aggregate Metrics",
        table(aggregate_metrics, ["baseline", "model_name", "metric_scope", "metric_name", "metric_value", "state_po", "n"], head=160),
        "",
        "## Figures",
        table(figure_rows, ["figure"], head=80),
        "",
        "## Calibration Notes",
        "\n".join(f"- {note}" for note in calibration_notes[:200]) if calibration_notes else "- None.",
        "",
        "## Output Files",
        "- `cohort.parquet`, `splits.parquet`, `agents.parquet`",
        "- `responses.parquet`, `crossfit_responses.parquet`, `prompts.parquet`",
        "- `individual_metrics.parquet`, `subgroup_metrics.parquet`, `calibration_bins.parquet`",
        "- `aggregate_crossfit_state_results.parquet`, `aggregate_eval_metrics.parquet`",
        "- `figures/*.png`, `figures/*.svg`",
        "",
    ]
    out = run_dir / "benchmark_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run_ces_individual_benchmark(config_path: str | Path) -> dict[str, Path]:
    cfg = load_yaml(config_path)
    run_id = cfg["run_id"]
    run_dir = ensure_dir(cfg.get("paths", {}).get("run_dir", f"data/runs/{run_id}"))
    paths = BenchmarkPaths(run_dir=run_dir, figures_dir=ensure_dir(run_dir / "figures"), cache_path=run_dir / "llm_cache.jsonl")

    respondents = pd.read_parquet(cfg["paths"]["ces_respondents"])
    answers = pd.read_parquet(cfg["paths"]["ces_answers"])
    targets = pd.read_parquet(cfg["paths"]["ces_targets"])
    context = pd.read_parquet(cfg["paths"]["ces_context"])
    strict_memory_facts = pd.read_parquet(cfg["paths"]["ces_memory_facts_strict"])
    poll_memory_facts = pd.read_parquet(cfg["paths"]["ces_memory_facts_poll"])
    question = pd.read_parquet(cfg["paths"].get("question_set_parquet", "data/processed/ces/2024_common_vv/ces_question_bank.parquet")).iloc[0] if cfg["paths"].get("question_set_parquet") else pd.Series(load_yaml(cfg["paths"]["question_set"]))

    cohort = build_benchmark_cohort(respondents, int(cfg.get("seed", 20260426)), cfg.get("cohort", {}).get("states", "all"))
    targets_wide = _target_wide(targets)
    agents = _agents_from_cohort(run_id, cohort)

    write_table(cohort, run_dir / "cohort.parquet")
    write_table(cohort[["ces_id", "split_bucket", "split", "fold"]], run_dir / "splits.parquet")
    write_table(agents, run_dir / "agents.parquet")

    train_ids = set(cohort.loc[cohort["split"] == "train", "ces_id"].astype(str))
    dev_ids = set(cohort.loc[cohort["split"] == "dev", "ces_id"].astype(str))
    test_ids = set(cohort.loc[cohort["split"] == "test", "ces_id"].astype(str))

    non_llm_names = [name for name in cfg.get("baselines", {}).get("non_llm", NON_LLM_BASELINES) if name in NON_LLM_BASELINES]
    feature_cache: dict[str, pd.DataFrame] = {}
    response_rows, calibration_notes = _run_non_llm_predictions(
        run_id=run_id,
        baseline_names=non_llm_names,
        cohort=cohort,
        answers=answers,
        targets_wide=targets_wide,
        eval_ids=test_ids,
        train_ids=train_ids,
        dev_ids=dev_ids,
        prediction_scope="test_full",
        feature_cache=feature_cache,
    )

    crossfit_rows: list[dict[str, Any]] = []
    for fold in range(5):
        fold_train, fold_dev, fold_eval = crossfit_partitions(cohort, fold)
        fold_rows, fold_notes = _run_non_llm_predictions(
            run_id=run_id,
            baseline_names=non_llm_names,
            cohort=cohort,
            answers=answers,
            targets_wide=targets_wide,
            eval_ids=fold_eval,
            train_ids=fold_train,
            dev_ids=fold_dev,
            prediction_scope="crossfit_all",
            fold=fold,
            feature_cache=feature_cache,
        )
        crossfit_rows.extend(fold_rows)
        calibration_notes.extend([f"fold{fold}:{note}" for note in fold_notes])

    llm_rows, prompt_rows, llm_metadata = _run_llm_pilot(
        cfg=cfg,
        paths=paths,
        cohort=cohort,
        agents=agents,
        targets=targets,
        question=question,
        context=context,
        strict_memory_facts=strict_memory_facts,
        poll_memory_facts=poll_memory_facts,
    )
    response_rows.extend(llm_rows)

    responses = pd.DataFrame(response_rows)
    crossfit_responses = pd.DataFrame(crossfit_rows)
    prompts = prompt_rows
    write_table(responses, run_dir / "responses.parquet")
    write_table(crossfit_responses, run_dir / "crossfit_responses.parquet")
    write_table(prompts, run_dir / "prompts.parquet")

    metrics = pd.DataFrame(benchmark_metric_rows(responses, cohort, targets, run_id, metric_scope="individual"))
    subgroup_metrics = compute_subgroup_metrics(responses, cohort, targets, run_id)
    calibration_bins = turnout_calibration_bins(responses, targets, run_id)
    write_table(metrics, run_dir / "individual_metrics.parquet")
    write_table(subgroup_metrics, run_dir / "subgroup_metrics.parquet")
    write_table(calibration_bins, run_dir / "calibration_bins.parquet")

    aggregate_input = crossfit_responses.drop(columns=["sample_weight"], errors="ignore")
    aggregate = aggregate_turnout_vote_state_results(aggregate_input, agents, run_id, 2024)
    mit_truth = pd.read_parquet(cfg["paths"]["mit_state_truth"])
    if "year" in mit_truth.columns:
        mit_truth = mit_truth[mit_truth["year"] == 2024].copy()
    if "geo_level" in mit_truth.columns:
        mit_truth = mit_truth[mit_truth["geo_level"] == "state"].copy()
    aggregate_metrics = turnout_vote_election_metrics(aggregate, mit_truth, run_id)
    write_table(aggregate, run_dir / "aggregate_crossfit_state_results.parquet")
    write_table(aggregate_metrics, run_dir / "aggregate_eval_metrics.parquet")

    figures = write_benchmark_figures(
        run_dir=run_dir,
        metrics=metrics,
        subgroup_metrics=subgroup_metrics,
        calibration_bins=calibration_bins,
        responses=responses,
        targets=targets,
        aggregate=aggregate,
        aggregate_metrics=aggregate_metrics,
    )
    report = write_benchmark_report(
        run_dir=run_dir,
        cfg=cfg,
        cohort=cohort,
        responses=responses,
        metrics=metrics,
        subgroup_metrics=subgroup_metrics,
        aggregate=aggregate,
        aggregate_metrics=aggregate_metrics,
        figures=figures,
        calibration_notes=calibration_notes,
        llm_metadata=llm_metadata,
    )
    return {
        "cohort": run_dir / "cohort.parquet",
        "splits": run_dir / "splits.parquet",
        "agents": run_dir / "agents.parquet",
        "responses": run_dir / "responses.parquet",
        "crossfit_responses": run_dir / "crossfit_responses.parquet",
        "prompts": run_dir / "prompts.parquet",
        "individual_metrics": run_dir / "individual_metrics.parquet",
        "subgroup_metrics": run_dir / "subgroup_metrics.parquet",
        "calibration_bins": run_dir / "calibration_bins.parquet",
        "aggregate": run_dir / "aggregate_crossfit_state_results.parquet",
        "aggregate_metrics": run_dir / "aggregate_eval_metrics.parquet",
        "report": report,
    }
