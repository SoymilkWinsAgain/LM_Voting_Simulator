"""Small file I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def project_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def ensure_parent(path: str | Path) -> Path:
    out = project_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def ensure_dir(path: str | Path) -> Path:
    out = project_path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_yaml(path: str | Path) -> Any:
    with project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(data: Any, path: str | Path) -> None:
    out = ensure_parent(path)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def read_table(path: str | Path) -> pd.DataFrame:
    src = project_path(path)
    suffix = src.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(src)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(src)
    if suffix == ".json":
        return pd.read_json(src)
    if suffix == ".jsonl":
        return pd.read_json(src, lines=True)
    raise ValueError(f"Unsupported table format: {src}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    out = ensure_parent(path)
    suffix = out.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(out, index=False)
        return
    if suffix == ".csv":
        df.to_csv(out, index=False)
        return
    if suffix == ".jsonl":
        df.to_json(out, orient="records", lines=True)
        return
    raise ValueError(f"Unsupported table format: {out}")


def write_json(data: Any, path: str | Path) -> None:
    out = ensure_parent(path)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
