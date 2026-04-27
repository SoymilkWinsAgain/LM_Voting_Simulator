"""Synthetic agent construction from CES cells and ANES archetypes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RunConfig, load_cell_schema, load_run_config
from .io import write_table
from .transforms import clean_string, stable_hash


AGENT_COLUMNS = [
    "run_id",
    "agent_id",
    "year",
    "state_po",
    "source_dataset",
    "source_respondent_id",
    "cell_schema",
    "cell_id",
    "base_anes_id",
    "base_ces_id",
    "memory_card_id",
    "match_level",
    "match_distance",
    "sample_weight",
    "weight_column",
    "weight_missing_reason",
    "age_group",
    "gender",
    "race_ethnicity",
    "education_binary",
    "income_bin",
    "party_id_3",
    "party_id_7",
    "ideology_3",
    "ideology_7",
    "registered_self_pre",
    "validated_registration",
    "created_at",
]


def deterministic_counts(dist: pd.DataFrame, n_agents: int) -> pd.DataFrame:
    work = dist.copy()
    expected = work["weighted_share_smoothed"] * n_agents
    work["n_agents"] = np.floor(expected).astype(int)
    remainder = int(n_agents - work["n_agents"].sum())
    if remainder > 0:
        order = (expected - np.floor(expected)).sort_values(ascending=False).index[:remainder]
        work.loc[order, "n_agents"] += 1
    return work


def _row_matches(candidate: pd.Series, cell: pd.Series, columns: list[str]) -> bool:
    for col in columns:
        if col not in candidate.index or col not in cell.index:
            continue
        if clean_string(candidate[col]) != clean_string(cell[col]):
            return False
    return True


def categorical_distance(candidate: pd.Series, cell: pd.Series, weights: dict[str, float]) -> float:
    distance = 0.0
    for col, weight in weights.items():
        if col not in candidate.index or col not in cell.index:
            continue
        if clean_string(candidate[col]) != clean_string(cell[col]):
            distance += float(weight)
    return distance


def match_archetypes(
    cell: pd.Series,
    anes_pool: pd.DataFrame,
    memory_cards: pd.DataFrame,
    cell_schema: dict[str, Any],
    n_needed: int,
    *,
    min_candidates_per_cell: int = 1,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    backoff_levels = cell_schema.get("backoff_levels") or [cell_schema["columns"]]
    weights = cell_schema.get("weights", {})
    cards_by_anes = memory_cards.set_index("anes_id")

    chosen: pd.DataFrame | None = None
    match_level = 999
    for level, cols in enumerate(backoff_levels):
        candidates = anes_pool[anes_pool.apply(lambda row: _row_matches(row, cell, cols), axis=1)]
        if len(candidates) >= min_candidates_per_cell:
            chosen = candidates
            match_level = level
            break
    if chosen is None or chosen.empty:
        scored = anes_pool.copy()
        scored["_distance"] = scored.apply(lambda row: categorical_distance(row, cell, weights), axis=1)
        chosen = scored.sort_values("_distance").head(max(min_candidates_per_cell, 1))

    rows: list[dict[str, Any]] = []
    indices = rng.choice(chosen.index.to_numpy(), size=n_needed, replace=True)
    for idx in indices:
        respondent = chosen.loc[idx]
        distance = categorical_distance(respondent, cell, weights)
        card_id = None
        if respondent["anes_id"] in cards_by_anes.index:
            card_id = cards_by_anes.loc[respondent["anes_id"], "memory_card_id"]
        rows.append(
            {
                "base_anes_id": respondent["anes_id"],
                "memory_card_id": card_id,
                "match_level": match_level,
                "match_distance": float(distance),
            }
        )
    return rows


def build_agents_from_frames(
    cfg: RunConfig,
    cell_distribution: pd.DataFrame,
    anes_respondents: pd.DataFrame,
    memory_cards: pd.DataFrame,
    cell_schema: dict[str, Any],
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, Any]] = []
    n_per_state = int(cfg.population.n_agents_per_state)
    min_candidates = int(
        cfg.population.model_extra.get("archetype_matching", {}).get("min_candidates_per_cell", 1)
        if cfg.population.model_extra
        else 1
    )

    agent_counter = 0
    for state in cfg.scenario.states:
        state_dist = cell_distribution[cell_distribution["state_po"] == state].copy()
        if state_dist.empty:
            raise ValueError(f"No cell distribution for state {state}")
        allocated = deterministic_counts(state_dist, n_per_state)
        for _, cell in allocated.iterrows():
            n_needed = int(cell["n_agents"])
            if n_needed <= 0:
                continue
            matches = match_archetypes(
                cell,
                anes_respondents,
                memory_cards,
                cell_schema,
                n_needed,
                min_candidates_per_cell=min_candidates,
                rng=rng,
            )
            for match in matches:
                agent_counter += 1
                agent_id = f"{cfg.run_id}_{state}_{agent_counter:06d}"
                rows.append(
                    {
                        "run_id": cfg.run_id,
                        "agent_id": agent_id,
                        "year": int(cfg.scenario.year),
                        "state_po": state,
                        "source_dataset": "synthetic",
                        "source_respondent_id": match["base_anes_id"],
                        "cell_schema": cell_schema["name"],
                        "cell_id": cell["cell_id"],
                        "base_anes_id": match["base_anes_id"],
                        "base_ces_id": None,
                        "memory_card_id": match["memory_card_id"],
                        "match_level": match["match_level"],
                        "match_distance": match["match_distance"],
                        "sample_weight": 1.0,
                        "weight_column": None,
                        "weight_missing_reason": None,
                        "age_group": cell["age_group"],
                        "gender": cell["gender"],
                        "race_ethnicity": cell["race_ethnicity"],
                        "education_binary": cell["education_binary"],
                        "income_bin": cell.get("income_bin"),
                        "party_id_3": cell["party_id_3"],
                        "party_id_7": None,
                        "ideology_3": cell["ideology_3"],
                        "ideology_7": None,
                        "registered_self_pre": None,
                        "validated_registration": None,
                        "created_at": pd.Timestamp.now(tz="UTC"),
                    }
                )
    return pd.DataFrame(rows, columns=AGENT_COLUMNS)


def _extra_dict(model: Any) -> dict[str, Any]:
    return dict(getattr(model, "model_extra", None) or {})


def _filter_ces_rows(cfg: RunConfig, ces_respondents: pd.DataFrame) -> pd.DataFrame:
    extra = _extra_dict(cfg.population)
    selection = extra.get("selection", {})
    work = ces_respondents.copy()
    states = selection.get("states") or cfg.scenario.states
    if states != "all":
        work = work[work["state_po"].isin(states)]
    if selection.get("tookpost_required", False):
        work = work[work["tookpost"].astype(bool)]
    if selection.get("citizen_required", False) and "citizenship" in work.columns:
        work = work[work["citizenship"] == "yes"]
    return work


def _sample_ces_rows(cfg: RunConfig, ces_respondents: pd.DataFrame, weight_column: str) -> pd.DataFrame:
    extra = _extra_dict(cfg.population)
    sampling = extra.get("sampling", {})
    mode = sampling.get("mode", "weighted_sample")
    rng_seed = int(sampling.get("random_seed", cfg.seed))
    work = ces_respondents.copy()
    if mode == "all_rows":
        return work
    replace = bool(sampling.get("replace", False))
    if mode == "stratified_state_sample":
        n_per_state = int(sampling.get("n_agents_per_state", cfg.population.n_agents_per_state))
        parts = []
        for _, group in work.groupby("state_po", dropna=False):
            n = min(n_per_state, len(group))
            weights = group[weight_column].fillna(0).astype(float) if weight_column in group.columns else None
            if weights is not None and float(weights.sum()) <= 0:
                weights = None
            try:
                parts.append(group.sample(n=n, replace=replace, weights=weights, random_state=rng_seed))
            except ValueError:
                if weights is None or replace:
                    raise
                parts.append(group.sample(n=n, replace=True, weights=weights, random_state=rng_seed))
        return pd.concat(parts, ignore_index=True) if parts else work.head(0)
    n_total = int(sampling.get("n_total_agents", sampling.get("n_agents", 10)))
    n_total = min(n_total, len(work))
    weights = work[weight_column].fillna(0).astype(float) if weight_column in work.columns else None
    if weights is not None and float(weights.sum()) <= 0:
        weights = None
    try:
        return work.sample(n=n_total, replace=replace, weights=weights, random_state=rng_seed).reset_index(drop=True)
    except ValueError:
        if weights is None or replace:
            raise
        return work.sample(n=n_total, replace=True, weights=weights, random_state=rng_seed).reset_index(drop=True)


def build_agents_from_ces_rows(
    cfg: RunConfig,
    ces_respondents: pd.DataFrame,
    memory_cards: pd.DataFrame,
) -> pd.DataFrame:
    extra = _extra_dict(cfg.population)
    requested_weight_column = extra.get("weight", {}).get("column", "commonpostweight")
    weight_column = requested_weight_column
    if weight_column not in ces_respondents.columns and f"weight_{weight_column}" in ces_respondents.columns:
        weight_column = f"weight_{weight_column}"
    filtered = _filter_ces_rows(cfg, ces_respondents)
    if filtered.empty:
        raise ValueError("No CES respondents remain after population selection")
    sampled = _sample_ces_rows(cfg, filtered, weight_column)
    cards_by_ces = memory_cards.set_index(memory_cards["ces_id"].astype(str)) if not memory_cards.empty else None
    rows: list[dict[str, Any]] = []
    for idx, respondent in sampled.reset_index(drop=True).iterrows():
        ces_id = str(respondent["ces_id"])
        memory_card_id = None
        if cards_by_ces is not None and ces_id in cards_by_ces.index:
            memory_card_id = cards_by_ces.loc[ces_id, "memory_card_id"]
        weight_missing_reason = None
        if weight_column not in respondent.index:
            sample_weight = 1.0
            weight_missing_reason = f"weight_column_not_found:{requested_weight_column}"
        else:
            sample_weight = respondent.get(weight_column)
        try:
            sample_weight = float(sample_weight)
        except (TypeError, ValueError):
            sample_weight = 1.0
            weight_missing_reason = weight_missing_reason or "weight_value_missing_or_invalid"
        if not np.isfinite(sample_weight) or sample_weight <= 0:
            sample_weight = 1.0
            weight_missing_reason = weight_missing_reason or "weight_value_nonpositive_or_nonfinite"
        rows.append(
            {
                "run_id": cfg.run_id,
                "agent_id": f"{cfg.run_id}_ces_{idx + 1:06d}",
                "year": int(cfg.scenario.year),
                "state_po": respondent.get("state_po"),
                "source_dataset": "ces",
                "source_respondent_id": ces_id,
                "cell_schema": "ces_rows_v1",
                "cell_id": f"CES|{respondent.get('state_po')}|{ces_id}",
                "base_anes_id": None,
                "base_ces_id": ces_id,
                "memory_card_id": memory_card_id,
                "match_level": 0,
                "match_distance": 0.0,
                "sample_weight": sample_weight,
                "weight_column": weight_column,
                "weight_missing_reason": weight_missing_reason,
                "age_group": respondent.get("age_group"),
                "gender": respondent.get("gender"),
                "race_ethnicity": respondent.get("race_ethnicity"),
                "education_binary": respondent.get("education_binary"),
                "income_bin": respondent.get("income_bin"),
                "party_id_3": respondent.get("party_id_3_pre"),
                "party_id_7": respondent.get("party_id_7_pre"),
                "ideology_3": respondent.get("ideology_3"),
                "ideology_7": respondent.get("ideology_self_7"),
                "registered_self_pre": respondent.get("registered_self_pre"),
                "validated_registration": respondent.get("validated_registration"),
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
    return pd.DataFrame(rows, columns=AGENT_COLUMNS)


def build_agents(run_config_path: str | Path, out_path: str | Path) -> Path:
    cfg = load_run_config(run_config_path)
    paths = cfg.paths
    if cfg.population.source == "ces_rows":
        agents = build_agents_from_ces_rows(
            cfg,
            pd.read_parquet(paths["ces_respondents"]),
            pd.read_parquet(paths["ces_memory_cards"]),
        )
        write_table(agents, out_path)
        return Path(out_path)
    cell_schema = load_cell_schema(paths["cell_schema"])
    ces_cell_distribution = paths.get(
        "ces_cell_distribution",
        str(cfg.processed_dir / "ces" / "ces_cell_distribution.parquet"),
    )
    anes_respondents = paths.get(
        "anes_respondents",
        str(cfg.processed_dir / "anes" / "anes_respondents.parquet"),
    )
    anes_memory_cards = paths.get(
        "anes_memory_cards",
        str(cfg.processed_dir / "anes" / "anes_memory_cards.parquet"),
    )
    agents = build_agents_from_frames(
        cfg,
        pd.read_parquet(ces_cell_distribution),
        pd.read_parquet(anes_respondents),
        pd.read_parquet(anes_memory_cards),
        cell_schema,
    )
    write_table(agents, out_path)
    return Path(out_path)


def agent_profile(agent: pd.Series) -> dict[str, Any]:
    return {
        "state_po": agent["state_po"],
        "age_group": agent["age_group"],
        "gender": agent["gender"],
        "race_ethnicity": agent["race_ethnicity"],
        "education_binary": agent["education_binary"],
        "party_id_3": agent["party_id_3"],
        "ideology_3": agent["ideology_3"],
    }


def agent_response_id(run_id: str, agent_id: str, question_id: str, baseline: str) -> str:
    return stable_hash(run_id, agent_id, question_id, baseline, length=20)
