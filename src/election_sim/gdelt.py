"""GDELT context-card stub adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io import load_yaml, read_table
from .validation import require_columns


GDELT_CONTEXT_COLUMNS = [
    "context_card_id",
    "year",
    "state_po",
    "geo_scope",
    "start_date",
    "end_date",
    "topic",
    "candidate_ids",
    "event_count",
    "mention_count",
    "avg_tone",
    "top_themes",
    "top_persons",
    "top_organizations",
    "top_locations",
    "summary",
    "source",
    "created_at",
]


def _normalize_card(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_card_id": raw["context_card_id"],
        "year": int(raw["year"]),
        "state_po": raw.get("state_po"),
        "geo_scope": raw.get("geo_scope", "national"),
        "start_date": pd.to_datetime(raw["start_date"]).date(),
        "end_date": pd.to_datetime(raw["end_date"]).date(),
        "topic": raw["topic"],
        "candidate_ids": list(raw.get("candidate_ids", [])),
        "event_count": raw.get("event_count"),
        "mention_count": raw.get("mention_count"),
        "avg_tone": raw.get("avg_tone"),
        "top_themes": list(raw.get("top_themes", [])),
        "top_persons": list(raw.get("top_persons", [])),
        "top_organizations": list(raw.get("top_organizations", [])),
        "top_locations": list(raw.get("top_locations", [])),
        "summary": raw["summary"],
        "source": raw.get("source", "manual_stub"),
        "created_at": raw.get("created_at", pd.Timestamp.now(tz="UTC")),
    }


def load_context_cards(path: str | Path) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        raw = load_yaml(path)
        if isinstance(raw, list):
            cards = raw
        else:
            cards = raw.get("cards", [raw])
        df = pd.DataFrame([_normalize_card(card) for card in cards])
    else:
        df = read_table(path)
    validate_context_cards(df)
    return df


def validate_context_cards(df: pd.DataFrame) -> None:
    require_columns(df, GDELT_CONTEXT_COLUMNS, "gdelt_context_cards")
    invalid_scope = set(df["geo_scope"]) - {"national", "state", "county"}
    if invalid_scope:
        raise ValueError(f"Invalid GDELT geo_scope values: {sorted(invalid_scope)}")
    missing_summary = df["summary"].isna() | (df["summary"].astype(str).str.len() == 0)
    if missing_summary.any():
        raise ValueError("GDELT context cards require non-empty summary text")


def select_context_cards(
    cards: pd.DataFrame,
    *,
    year: int,
    states: list[str],
    simulation_date: str,
    topics: list[str] | None = None,
) -> pd.DataFrame:
    cutoff = pd.to_datetime(simulation_date).date()
    out = cards[cards["year"] == year].copy()
    out = out[out["end_date"].map(lambda value: pd.to_datetime(value).date() <= cutoff)]
    out = out[out["state_po"].isna() | out["state_po"].isin(states)]
    if topics:
        out = out[out["topic"].isin(topics)]
    return out.reset_index(drop=True)
