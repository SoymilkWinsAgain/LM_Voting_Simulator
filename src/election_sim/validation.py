"""Data contract validation helpers."""

from __future__ import annotations

import pandas as pd

from .constants import CANONICAL_SETS, STATE_PO


def require_columns(df: pd.DataFrame, columns: list[str], table_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")


def validate_categories(df: pd.DataFrame, table_name: str) -> None:
    for col, allowed in CANONICAL_SETS.items():
        if col not in df.columns:
            continue
        values = set(df[col].dropna().astype(str))
        invalid = sorted(values - allowed)
        if invalid:
            raise ValueError(f"{table_name}.{col} has invalid categories: {invalid[:10]}")
    if "state_po" in df.columns:
        values = set(df["state_po"].dropna().astype(str))
        invalid = sorted(values - STATE_PO)
        if invalid:
            raise ValueError(f"{table_name}.state_po has invalid state codes: {invalid[:10]}")


def validate_probability_simplex(
    df: pd.DataFrame,
    group_col: str,
    probability_col: str,
    *,
    tolerance: float = 1e-6,
) -> None:
    grouped = df.groupby(group_col, dropna=False)[probability_col].sum()
    bad = grouped[(grouped - 1.0).abs() > tolerance]
    if not bad.empty:
        raise ValueError(f"{probability_col} does not sum to 1 by {group_col}: {bad.to_dict()}")
