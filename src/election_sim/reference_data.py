"""Load packaged reference data used by transforms and policy guards."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache
def load_reference_json(name: str) -> Any:
    resource = Path(__file__).resolve().parent / "reference" / name
    with resource.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache
def us_state_records() -> tuple[dict[str, Any], ...]:
    return tuple(load_reference_json("us_states.json"))


@lru_cache
def state_fips_to_po_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for record in us_state_records():
        fips = str(record["state_fips"]).zfill(2)
        mapping[fips] = record["state_po"]
        mapping[fips.lstrip("0")] = record["state_po"]
    return mapping


@lru_cache
def leakage_policy_reference() -> dict[str, Any]:
    return load_reference_json("leakage_policies.json")


STATE_PO = {record["state_po"] for record in us_state_records()}
STATE_FIPS_TO_PO = state_fips_to_po_map()
SWING_STATES_2024 = [
    record["state_po"] for record in us_state_records() if bool(record.get("swing_state_2024"))
]
