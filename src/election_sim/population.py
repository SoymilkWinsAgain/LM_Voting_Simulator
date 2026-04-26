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
    "cell_schema",
    "cell_id",
    "base_anes_id",
    "memory_card_id",
    "match_level",
    "match_distance",
    "sample_weight",
    "age_group",
    "gender",
    "race_ethnicity",
    "education_binary",
    "party_id_3",
    "ideology_3",
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
                        "cell_schema": cell_schema["name"],
                        "cell_id": cell["cell_id"],
                        "base_anes_id": match["base_anes_id"],
                        "memory_card_id": match["memory_card_id"],
                        "match_level": match["match_level"],
                        "match_distance": match["match_distance"],
                        "sample_weight": 1.0,
                        "age_group": cell["age_group"],
                        "gender": cell["gender"],
                        "race_ethnicity": cell["race_ethnicity"],
                        "education_binary": cell["education_binary"],
                        "party_id_3": cell["party_id_3"],
                        "ideology_3": cell["ideology_3"],
                        "created_at": pd.Timestamp.now(tz="UTC"),
                    }
                )
    return pd.DataFrame(rows, columns=AGENT_COLUMNS)


def build_agents(run_config_path: str | Path, out_path: str | Path) -> Path:
    cfg = load_run_config(run_config_path)
    paths = cfg.paths
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
