"""CES-to-ANES persona enrichment for CES respondent simulations."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import ensure_dir, load_yaml, write_json, write_table
from .transforms import (
    age_to_group,
    anes_2024_education_to_binary,
    anes_2024_party_id_to_party3,
    clean_string,
    ideology7_to_ideology3,
    is_missing,
    stable_hash,
    state_fips_to_po,
)


PERSONA_MEMORY_POLICY = "strict_pre_no_vote_with_anes_persona_v1"

ANES_ID_COL = "V240001"
ANES_AGE_COL = "V241458x"
ANES_REGISTRATION_STATE_COL = "V241023"

THERMOMETER_VARIABLES = {
    "V241156": "Democratic presidential candidate",
    "V241157": "Joe Biden",
    "V241158": "Robert F. Kennedy Jr.",
    "V241164": "Democratic vice-presidential candidate",
    "V241165": "Republican vice-presidential candidate",
    "V241166": "Democratic Party",
    "V241167": "Republican Party",
}

EMOTION_VARIABLES = {
    "V241117": "hopeful",
    "V241118": "afraid",
    "V241119": "outraged",
    "V241120": "angry",
    "V241121": "happy",
    "V241122": "worried",
    "V241123": "proud",
    "V241124": "irritated",
    "V241125": "nervous",
    "V241126": "concerned",
}

IDENTITY_EFFICACY_VARIABLES = {
    "V241228": "party_identity_importance",
    "V241235": "government_responsiveness",
}

OPEN_END_VARIABLES = {
    "V241110": "likes about Democratic presidential candidate",
    "V241112": "dislikes about Democratic presidential candidate",
    "V241114": "likes about Republican presidential candidate",
    "V241116": "dislikes about Republican presidential candidate",
    "V241170": "likes about Democratic Party",
    "V241172": "dislikes about Democratic Party",
    "V241174": "likes about Republican Party",
    "V241176": "dislikes about Republican Party",
}

SAFE_PERSONA_SOURCE_VARIABLES = (
    tuple(THERMOMETER_VARIABLES)
    + tuple(EMOTION_VARIABLES)
    + tuple(IDENTITY_EFFICACY_VARIABLES)
    + tuple(OPEN_END_VARIABLES)
)

SYNTHETIC_PERSONA_SOURCE_VARIABLES = {
    "ANES_PERSONA_THERMOMETERS",
    "ANES_PERSONA_EMOTIONS",
    "ANES_PERSONA_IDENTITY",
    "ANES_PERSONA_EFFICACY",
    "ANES_PERSONA_OPEN_TEXT",
}

STATE_TO_REGION = {
    "CT": "northeast",
    "ME": "northeast",
    "MA": "northeast",
    "NH": "northeast",
    "RI": "northeast",
    "VT": "northeast",
    "NJ": "northeast",
    "NY": "northeast",
    "PA": "northeast",
    "IL": "midwest",
    "IN": "midwest",
    "MI": "midwest",
    "OH": "midwest",
    "WI": "midwest",
    "IA": "midwest",
    "KS": "midwest",
    "MN": "midwest",
    "MO": "midwest",
    "NE": "midwest",
    "ND": "midwest",
    "SD": "midwest",
    "DE": "south",
    "DC": "south",
    "FL": "south",
    "GA": "south",
    "MD": "south",
    "NC": "south",
    "SC": "south",
    "VA": "south",
    "WV": "south",
    "AL": "south",
    "KY": "south",
    "MS": "south",
    "TN": "south",
    "AR": "south",
    "LA": "south",
    "OK": "south",
    "TX": "south",
    "AZ": "west",
    "CO": "west",
    "ID": "west",
    "MT": "west",
    "NV": "west",
    "NM": "west",
    "UT": "west",
    "WY": "west",
    "AK": "west",
    "CA": "west",
    "HI": "west",
    "OR": "west",
    "WA": "west",
}

CES_REGION_MAP = {
    "1": "northeast",
    "2": "midwest",
    "3": "south",
    "4": "west",
}

DEFAULT_PERSONA_CONFIG: dict[str, Any] = {
    "policy": PERSONA_MEMORY_POLICY,
    "source_year": 2024,
    "matching": {
        "k_retrieve": 20,
        "k_min_support": 5,
        "tau": 0.15,
        "chunk_size": 512,
        "max_persona_facts": 6,
        "features": [
            {
                "name": "party_id_3",
                "kind": "nominal",
                "weight": 2.0,
                "ces_column": "party_id_3_pre",
                "anes_variable": "V241221",
                "anes_transform": "anes_party_id_3",
            },
            {
                "name": "ideology_3",
                "kind": "nominal",
                "weight": 1.5,
                "ces_column": "ideology_3",
                "anes_variable": "V241177",
                "anes_transform": "ideology_3",
            },
            {
                "name": "age_group",
                "kind": "ordinal",
                "weight": 1.0,
                "order": ["18_29", "30_44", "45_64", "65_plus"],
                "ces_column": "age_group",
                "anes_variable": ANES_AGE_COL,
                "anes_transform": "age_group",
            },
            {
                "name": "education_binary",
                "kind": "nominal",
                "weight": 1.0,
                "ces_column": "education_binary",
                "anes_variable": "V241463",
                "anes_transform": "anes_education_binary",
            },
            {
                "name": "race_ethnicity",
                "kind": "nominal",
                "weight": 1.0,
                "ces_column": "race_ethnicity",
                "anes_variable": "V241501x",
                "anes_transform": "anes_race_ethnicity",
            },
            {
                "name": "gender",
                "kind": "nominal",
                "weight": 0.75,
                "ces_column": "gender",
                "anes_variable": "V241550",
                "anes_transform": "anes_gender",
            },
            {
                "name": "income_bin",
                "kind": "ordinal",
                "weight": 0.75,
                "ces_column": "income_bin",
                "anes_variable": None,
                "anes_transform": "string",
            },
            {
                "name": "region",
                "kind": "nominal",
                "weight": 0.75,
                "ces_column": "region",
                "ces_transform": "ces_region",
                "anes_variable": ANES_REGISTRATION_STATE_COL,
                "anes_transform": "state_region",
                "source_note": "ANES side is derived from registration state, not residence state.",
            },
            {
                "name": "registration_state_po",
                "kind": "nominal",
                "weight": 0.25,
                "ces_column": "state_po",
                "anes_variable": ANES_REGISTRATION_STATE_COL,
                "anes_transform": "state_fips_to_po",
                "source_note": "Weak ANES registration-state feature; not a residence-state block.",
            },
        ],
    },
    "persona": {
        "thermometers": THERMOMETER_VARIABLES,
        "emotions": EMOTION_VARIABLES,
        "identity_efficacy": IDENTITY_EFFICACY_VARIABLES,
        "open_ends": OPEN_END_VARIABLES,
        "theme_min_share": 0.2,
    },
}

OPEN_TEXT_THEME_KEYWORDS = {
    "economy_prices": (
        "economy",
        "economic",
        "inflation",
        "price",
        "prices",
        "gas",
        "food",
        "jobs",
        "wage",
        "tax",
        "taxes",
        "cost",
    ),
    "immigration_border": (
        "immigration",
        "immigrant",
        "immigrants",
        "illegal",
        "border",
        "asylum",
        "wall",
        "deport",
        "migrant",
    ),
    "democracy_institutions": (
        "democracy",
        "democratic",
        "constitution",
        "law",
        "rights",
        "freedom",
        "country",
        "america",
        "nation",
        "court",
    ),
    "abortion_reproductive": (
        "abortion",
        "reproductive",
        "women",
        "woman",
        "choice",
        "pregnancy",
        "body",
        "roe",
    ),
    "character_leadership": (
        "leader",
        "leadership",
        "honest",
        "dishonest",
        "liar",
        "criminal",
        "competent",
        "incompetent",
        "qualified",
        "morals",
        "values",
        "strong",
    ),
    "healthcare_insurance": (
        "healthcare",
        "health care",
        "insurance",
        "medicare",
        "medicaid",
        "medical",
        "drug",
    ),
    "climate_environment": (
        "climate",
        "environment",
        "environmental",
        "epa",
        "energy",
        "carbon",
        "pollution",
    ),
    "foreign_policy_war": (
        "war",
        "ukraine",
        "israel",
        "gaza",
        "foreign",
        "military",
        "china",
        "russia",
    ),
    "identity_civil_rights": (
        "race",
        "racial",
        "gender",
        "lgbt",
        "lgbtq",
        "trans",
        "equality",
        "diversity",
        "civil rights",
    ),
    "guns_crime_policing": (
        "gun",
        "guns",
        "crime",
        "criminal",
        "police",
        "safety",
        "violence",
    ),
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_persona_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load persona config with module defaults filled in."""

    if config_path is None:
        return _deep_update({}, DEFAULT_PERSONA_CONFIG)
    loaded = load_yaml(config_path)
    return _deep_update(DEFAULT_PERSONA_CONFIG, loaded if isinstance(loaded, dict) else {})


def _valid_text(value: Any) -> str:
    text = clean_string(value)
    if not text or text.lower() in {"unknown", "nan", "none"}:
        return ""
    return text


def _profile_missing(values: pd.Series) -> pd.Series:
    return values.map(lambda value: _valid_text(value).lower() in {"", "unknown"})


def _numeric_or_nan(value: Any) -> float:
    if is_missing(value):
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not math.isfinite(number) or number < 0:
        return np.nan
    return number


def _anes_gender(value: Any) -> str:
    key = clean_string(value)
    if key == "1":
        return "male"
    if key == "2":
        return "female"
    return "other_or_unknown"


def _anes_race_ethnicity(value: Any) -> str:
    key = clean_string(value)
    if key == "1":
        return "white"
    if key == "2":
        return "black"
    if key == "3":
        return "hispanic"
    if key == "4":
        return "asian"
    if key in {"5", "6"}:
        return "other_or_unknown"
    return "unknown"


def _state_region(value: Any) -> str:
    state = state_fips_to_po(value) or clean_string(value).upper()
    return STATE_TO_REGION.get(state, "unknown")


def _ces_region(value: Any) -> str:
    key = clean_string(value)
    lowered = key.lower()
    if lowered in {"northeast", "midwest", "south", "west"}:
        return lowered
    return CES_REGION_MAP.get(key, "unknown")


def _state_po(value: Any) -> str:
    return state_fips_to_po(value) or clean_string(value).upper() or "unknown"


def _transform_series(values: pd.Series, transform: str | None) -> pd.Series:
    name = transform or "string"
    if name == "string":
        return values.map(clean_string)
    if name == "lower_string":
        return values.map(lambda value: clean_string(value).lower())
    if name == "age_group":
        return values.map(age_to_group)
    if name == "anes_gender":
        return values.map(_anes_gender)
    if name == "anes_race_ethnicity":
        return values.map(_anes_race_ethnicity)
    if name == "anes_education_binary":
        return values.map(anes_2024_education_to_binary)
    if name == "anes_party_id_3":
        return values.map(anes_2024_party_id_to_party3)
    if name == "ideology_3":
        return values.map(ideology7_to_ideology3)
    if name == "state_fips_to_po":
        return values.map(_state_po)
    if name == "state_region":
        return values.map(_state_region)
    if name == "ces_region":
        return values.map(_ces_region)
    if name == "numeric":
        return values.map(_numeric_or_nan)
    raise ValueError(f"Unknown persona transform: {name}")


def _feature_specs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return list(cfg.get("matching", {}).get("features", []))


def _anes_usecols(cfg: dict[str, Any]) -> list[str]:
    variables = {ANES_ID_COL}
    for feature in _feature_specs(cfg):
        var = feature.get("anes_variable")
        if var:
            variables.add(str(var))
    persona = cfg.get("persona", {})
    for block in ("thermometers", "emotions", "identity_efficacy", "open_ends"):
        values = persona.get(block, {})
        if isinstance(values, dict):
            variables.update(str(var) for var in values)
        else:
            variables.update(str(var) for var in values)
    return sorted(variables)


def build_ces_bridge_profiles(
    ces_respondents: pd.DataFrame,
    ces_answers: pd.DataFrame | None,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the CES side of the cross-dataset matching profile."""

    del ces_answers
    cfg = cfg or load_persona_config()
    rows = pd.DataFrame(
        {
            "ces_id": ces_respondents["ces_id"].astype(str),
            "source_year": ces_respondents.get("source_year", pd.Series(2024, index=ces_respondents.index)),
        }
    )
    for feature in _feature_specs(cfg):
        name = str(feature["name"])
        col = feature.get("ces_column", name)
        if col in ces_respondents.columns:
            source = ces_respondents[col]
            rows[name] = _transform_series(source, feature.get("ces_transform"))
        else:
            rows[name] = None
    if "registration_state_po" not in rows.columns and "state_po" in ces_respondents.columns:
        rows["registration_state_po"] = ces_respondents["state_po"].map(_state_po)
    return rows


def load_anes_raw_for_persona(raw_path: str | Path, cfg: dict[str, Any]) -> pd.DataFrame:
    usecols = _anes_usecols(cfg)
    available = pd.read_csv(raw_path, nrows=0).columns
    available_set = set(available)
    existing = [col for col in usecols if col in available_set]
    if ANES_ID_COL not in existing:
        raise ValueError(f"ANES raw file is missing required id column {ANES_ID_COL}")
    return pd.read_csv(raw_path, usecols=existing)


def build_anes_bridge_profiles(anes_raw: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build the ANES side of the cross-dataset matching profile."""

    cfg = cfg or load_persona_config()
    rows = pd.DataFrame({"anes_id": anes_raw[ANES_ID_COL].map(clean_string)})
    for feature in _feature_specs(cfg):
        name = str(feature["name"])
        var = feature.get("anes_variable")
        if var and var in anes_raw.columns:
            rows[name] = _transform_series(anes_raw[var], feature.get("anes_transform"))
        else:
            rows[name] = None
    if ANES_REGISTRATION_STATE_COL in anes_raw.columns:
        rows["registration_state_po_raw"] = anes_raw[ANES_REGISTRATION_STATE_COL].map(_state_po)
        rows["registration_state_is_weak"] = True
    return rows


def _ordinal_codes(values: pd.Series, order: list[Any] | None) -> pd.Series:
    if order:
        mapping = {clean_string(value): idx for idx, value in enumerate(order)}
        return values.map(lambda value: mapping.get(clean_string(value), np.nan)).astype(float)
    numeric = values.map(_numeric_or_nan).astype(float)
    return numeric


def _distance_for_feature(
    ces_values: pd.Series,
    anes_values: pd.Series,
    feature: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    kind = feature.get("kind", "nominal")
    if kind == "ordinal":
        c = _ordinal_codes(ces_values, feature.get("order")).to_numpy(dtype=float)[:, None]
        a = _ordinal_codes(anes_values, feature.get("order")).to_numpy(dtype=float)[None, :]
        valid = np.isfinite(c) & np.isfinite(a)
        if feature.get("order"):
            denom = max(1.0, float(len(feature["order"]) - 1))
        else:
            combined = np.concatenate([c[np.isfinite(c)], a[np.isfinite(a)]])
            denom = float(np.nanmax(combined) - np.nanmin(combined)) if combined.size else 1.0
            denom = max(1.0, denom)
        return np.abs(c - a) / denom, valid
    if kind == "continuous":
        c = ces_values.map(_numeric_or_nan).to_numpy(dtype=float)[:, None]
        a = anes_values.map(_numeric_or_nan).to_numpy(dtype=float)[None, :]
        valid = np.isfinite(c) & np.isfinite(a)
        combined = np.concatenate([c[np.isfinite(c)], a[np.isfinite(a)]])
        denom = float(np.nanstd(combined)) if combined.size else 1.0
        denom = max(1.0, denom)
        return np.abs(c - a) / denom, valid
    if kind == "multiselect":
        c_sets = ces_values.map(_as_token_set).tolist()
        a_sets = anes_values.map(_as_token_set).tolist()
        dist = np.ones((len(c_sets), len(a_sets)), dtype=float)
        valid = np.zeros_like(dist, dtype=bool)
        for i, cset in enumerate(c_sets):
            if not cset:
                continue
            for j, aset in enumerate(a_sets):
                if not aset:
                    continue
                valid[i, j] = True
                dist[i, j] = 1.0 - (len(cset & aset) / len(cset | aset))
        return dist, valid
    c = ces_values.map(lambda value: _valid_text(value).lower())
    a = anes_values.map(lambda value: _valid_text(value).lower())
    c_arr = c.to_numpy(dtype=object)[:, None]
    a_arr = a.to_numpy(dtype=object)[None, :]
    valid = (~_profile_missing(c).to_numpy()[:, None]) & (~_profile_missing(a).to_numpy()[None, :])
    return (c_arr != a_arr).astype(float), valid


def _as_token_set(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = re.split(r"[|,;]", clean_string(value))
    return {clean_string(item).lower() for item in values if clean_string(item)}


def match_ces_to_anes(
    ces_profiles: pd.DataFrame,
    anes_profiles: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find weighted nearest ANES donors for each CES respondent."""

    cfg = cfg or load_persona_config()
    matching = cfg["matching"]
    k = int(matching.get("k_retrieve", 20))
    tau = float(matching.get("tau", 0.15))
    chunk_size = int(matching.get("chunk_size", 512))
    if len(anes_profiles) == 0:
        raise ValueError("ANES persona matching requires at least one donor row")
    features = [
        f
        for f in _feature_specs(cfg)
        if _feature_has_overlap(f, ces_profiles=ces_profiles, anes_profiles=anes_profiles)
    ]
    if not features:
        raise ValueError("No shared non-missing CES/ANES persona matching features are available")

    anes_n = len(anes_profiles)
    k_eff_global = min(k, anes_n)
    match_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    anes_ids = anes_profiles["anes_id"].astype(str).to_numpy()
    for start in range(0, len(ces_profiles), chunk_size):
        ces_chunk = ces_profiles.iloc[start : start + chunk_size].reset_index(drop=True)
        dist_sum = np.zeros((len(ces_chunk), anes_n), dtype=np.float64)
        weight_sum = np.zeros_like(dist_sum)
        feature_count = np.zeros_like(dist_sum)
        for feature in features:
            name = str(feature["name"])
            weight = float(feature.get("weight", 1.0))
            dist, valid = _distance_for_feature(ces_chunk[name], anes_profiles[name], feature)
            dist_sum += np.where(valid, dist * weight, 0.0)
            weight_sum += np.where(valid, weight, 0.0)
            feature_count += valid.astype(float)
        avg_dist = np.divide(
            dist_sum,
            weight_sum,
            out=np.full_like(dist_sum, np.inf, dtype=np.float64),
            where=weight_sum > 0,
        )
        top_unsorted = np.argpartition(avg_dist, kth=k_eff_global - 1, axis=1)[:, :k_eff_global]
        for i, indices in enumerate(top_unsorted):
            ces_id = str(ces_chunk.loc[i, "ces_id"])
            finite_indices = [idx for idx in indices if np.isfinite(avg_dist[i, idx])]
            finite_indices.sort(key=lambda idx: (avg_dist[i, idx], anes_ids[idx]))
            if not finite_indices:
                summary_rows.append(
                    {
                        "ces_id": ces_id,
                        "top_anes_id": None,
                        "top_distance": np.nan,
                        "donor_count": 0,
                        "effective_sample_size": 0.0,
                        "n_eligible_donors": 0,
                    }
                )
                continue
            distances = np.array([avg_dist[i, idx] for idx in finite_indices], dtype=float)
            similarities = np.exp(-distances / max(tau, 1e-9))
            if not np.isfinite(similarities).any() or float(similarities.sum()) <= 0:
                donor_weights = np.full(len(finite_indices), 1.0 / len(finite_indices))
            else:
                donor_weights = similarities / similarities.sum()
            ess = float(1.0 / np.square(donor_weights).sum())
            registration_state_match = None
            if "registration_state_po" in ces_chunk.columns and "registration_state_po" in anes_profiles.columns:
                registration_state_match = [
                    clean_string(ces_chunk.loc[i, "registration_state_po"])
                    == clean_string(anes_profiles.iloc[idx]["registration_state_po"])
                    for idx in finite_indices
                ]
            for rank, (idx, donor_weight) in enumerate(zip(finite_indices, donor_weights, strict=False), start=1):
                row = {
                    "ces_id": ces_id,
                    "anes_id": str(anes_ids[idx]),
                    "donor_rank": rank,
                    "match_distance": float(avg_dist[i, idx]),
                    "donor_weight": float(donor_weight),
                    "common_feature_weight": float(weight_sum[i, idx]),
                    "n_common_features": int(feature_count[i, idx]),
                }
                if registration_state_match is not None:
                    row["registration_state_match"] = bool(registration_state_match[rank - 1])
                match_rows.append(row)
            summary_rows.append(
                {
                    "ces_id": ces_id,
                    "top_anes_id": str(anes_ids[finite_indices[0]]),
                    "top_distance": float(distances[0]),
                    "donor_count": int(len(finite_indices)),
                    "effective_sample_size": ess,
                    "n_eligible_donors": int(np.isfinite(avg_dist[i]).sum()),
                }
            )
    return pd.DataFrame(match_rows), pd.DataFrame(summary_rows)


def _feature_has_overlap(
    feature: dict[str, Any],
    *,
    ces_profiles: pd.DataFrame,
    anes_profiles: pd.DataFrame,
) -> bool:
    name = str(feature.get("name", ""))
    if name not in ces_profiles or name not in anes_profiles:
        return False
    return bool((~_profile_missing(ces_profiles[name])).any() and (~_profile_missing(anes_profiles[name])).any())


def read_anes_open_ends(xlsx_path: str | Path, sheet_names: dict[str, str] | list[str]) -> pd.DataFrame:
    """Read selected ANES redacted open-ended sheets into a long table."""

    if isinstance(sheet_names, dict):
        variables = list(sheet_names)
    else:
        variables = list(sheet_names)
    frames: list[pd.DataFrame] = []
    for variable in variables:
        sheet = pd.read_excel(xlsx_path, sheet_name=variable, dtype=str)
        if ANES_ID_COL not in sheet.columns:
            raise ValueError(f"Open-ended sheet {variable} is missing {ANES_ID_COL}")
        text_columns = [col for col in sheet.columns if col != ANES_ID_COL]
        if not text_columns:
            continue
        text_col = text_columns[0]
        part = sheet[[ANES_ID_COL, text_col]].copy()
        part.columns = ["anes_id", "open_text"]
        part["anes_id"] = part["anes_id"].map(clean_string)
        part["source_variable"] = variable
        part["prompt_label"] = sheet_names[variable] if isinstance(sheet_names, dict) else variable
        part["open_text"] = part["open_text"].fillna("").astype(str).str.replace("[CHAR(10)]", " ", regex=False)
        part = part[part["open_text"].map(lambda text: bool(clean_string(text)))]
        frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["anes_id", "source_variable", "prompt_label", "open_text", "themes"])
    out = pd.concat(frames, ignore_index=True)
    out["themes"] = out["open_text"].map(classify_open_text_themes)
    return out[["anes_id", "source_variable", "prompt_label", "open_text", "themes"]]


def classify_open_text_themes(text: str) -> list[str]:
    lowered = clean_string(text).lower()
    if not lowered:
        return []
    themes = []
    for theme, keywords in OPEN_TEXT_THEME_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            themes.append(theme)
    return themes


def _valid_thermometer(value: Any) -> float:
    number = _numeric_or_nan(value)
    if np.isnan(number) or number > 100:
        return np.nan
    return float(number)


def _valid_scale_1_5(value: Any) -> float:
    number = _numeric_or_nan(value)
    if np.isnan(number) or number < 1 or number > 5:
        return np.nan
    return float(number)


def _identity_label(value: Any) -> str:
    return {
        "1": "extremely important",
        "2": "very important",
        "3": "moderately important",
        "4": "a little important",
        "5": "not important",
    }.get(clean_string(value), "")


def _efficacy_label(value: Any) -> str:
    return {"1": "a good deal", "2": "some", "3": "not much"}.get(clean_string(value), "")


def build_anes_persona_payloads(
    anes_raw: pd.DataFrame,
    open_ends: pd.DataFrame | None,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build per-ANES respondent persona payloads used for donor aggregation."""

    cfg = cfg or load_persona_config()
    persona = cfg.get("persona", {})
    rows = pd.DataFrame({"anes_id": anes_raw[ANES_ID_COL].map(clean_string)})
    for variable in persona.get("thermometers", THERMOMETER_VARIABLES):
        if variable in anes_raw.columns:
            rows[variable] = anes_raw[variable].map(_valid_thermometer)
    for variable in persona.get("emotions", EMOTION_VARIABLES):
        if variable in anes_raw.columns:
            rows[variable] = anes_raw[variable].map(_valid_scale_1_5)
    if "V241228" in anes_raw.columns:
        rows["V241228_label"] = anes_raw["V241228"].map(_identity_label)
    if "V241235" in anes_raw.columns:
        rows["V241235_label"] = anes_raw["V241235"].map(_efficacy_label)
    if open_ends is not None and not open_ends.empty:
        theme_rows = []
        for anes_id, group in open_ends.groupby("anes_id", sort=False):
            counter: Counter[str] = Counter()
            source_vars = set()
            for _, row in group.iterrows():
                counter.update(row["themes"])
                source_vars.add(str(row["source_variable"]))
            theme_rows.append(
                {
                    "anes_id": str(anes_id),
                    "open_text_n": int(len(group)),
                    "open_text_source_variables": sorted(source_vars),
                    "open_text_themes": sorted(counter),
                }
            )
        themes = pd.DataFrame(theme_rows)
        rows = rows.merge(themes, on="anes_id", how="left")
    if "open_text_n" not in rows.columns:
        rows["open_text_n"] = 0
        rows["open_text_source_variables"] = [[] for _ in range(len(rows))]
        rows["open_text_themes"] = [[] for _ in range(len(rows))]
    rows["open_text_n"] = rows["open_text_n"].fillna(0).astype(int)
    rows["open_text_source_variables"] = rows["open_text_source_variables"].map(_list_or_empty)
    rows["open_text_themes"] = rows["open_text_themes"].map(_list_or_empty)
    return rows


def _list_or_empty(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [str(value)]


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    nums = values.map(_numeric_or_nan).astype(float)
    valid = nums.notna()
    if not valid.any():
        return np.nan
    w = weights[valid].astype(float)
    if float(w.sum()) <= 0:
        return float(nums[valid].mean())
    return float(np.average(nums[valid], weights=w))


def _weighted_mode(labels: pd.Series, weights: pd.Series) -> tuple[str, float]:
    scores: dict[str, float] = {}
    for label, weight in zip(labels, weights, strict=False):
        text = clean_string(label)
        if not text:
            continue
        scores[text] = scores.get(text, 0.0) + float(weight)
    if not scores:
        return "", 0.0
    label, score = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0]
    return label, score


def _support_confidence(donor_count: int, ess: float, k_min_support: int) -> float:
    if donor_count <= 0:
        return 0.0
    return round(min(1.0, (ess / max(1, k_min_support)) * (donor_count / max(donor_count, k_min_support))), 3)


def build_persona_facts(
    ces_profiles: pd.DataFrame,
    matches: pd.DataFrame,
    match_summary: pd.DataFrame,
    payloads: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Aggregate matched ANES donors into inferred CES persona facts."""

    cfg = cfg or load_persona_config()
    matching = cfg["matching"]
    k_min_support = int(matching.get("k_min_support", 5))
    max_facts = int(matching.get("max_persona_facts", 6))
    policy = str(cfg.get("policy", PERSONA_MEMORY_POLICY))
    source_year = int(cfg.get("source_year", 2024))
    payloads_by_id = payloads.set_index("anes_id", drop=False)
    summary_by_ces = match_summary.set_index("ces_id", drop=False)
    matches_by_ces = {
        str(ces_id): group.reset_index(drop=True)
        for ces_id, group in matches.groupby(matches["ces_id"].astype(str), sort=False)
    }
    fact_rows: list[dict[str, Any]] = []

    for _, ces_row in ces_profiles.iterrows():
        ces_id = str(ces_row["ces_id"])
        donor_matches = matches_by_ces.get(ces_id)
        if donor_matches is None:
            continue
        if donor_matches.empty or len(donor_matches) < k_min_support:
            continue
        donor_payloads = payloads_by_id.reindex(donor_matches["anes_id"].astype(str)).reset_index(drop=True)
        if donor_payloads.empty:
            continue
        weights = donor_matches["donor_weight"].reset_index(drop=True).astype(float)
        summary = summary_by_ces.loc[ces_id] if ces_id in summary_by_ces.index else {}
        donor_ids = donor_matches["anes_id"].astype(str).tolist()
        ess = float(summary.get("effective_sample_size", 0.0)) if isinstance(summary, pd.Series) else 0.0
        confidence = _support_confidence(len(donor_ids), ess, k_min_support)
        facts = _persona_fact_candidates(
            donor_payloads,
            weights,
            donor_ids=donor_ids,
            donor_count=len(donor_ids),
            ess=ess,
            confidence=confidence,
            theme_min_share=float(cfg.get("persona", {}).get("theme_min_share", 0.2)),
        )
        for rank, fact in enumerate(facts[:max_facts], start=1):
            source_variables = fact["source_variables"]
            fact_id = "ces_anes_persona_{}_{}_{}".format(
                ces_id,
                stable_hash(fact["source_variable"], fact["fact_text"], rank, length=12),
                policy,
            )
            fact_rows.append(
                {
                    "memory_fact_id": fact_id,
                    "ces_id": ces_id,
                    "source_year": source_year,
                    "source_variable": fact["source_variable"],
                    "source_variables": source_variables,
                    "question_id": None,
                    "topic": fact["topic"],
                    "subtopic": fact["subtopic"],
                    "fact_text": fact["fact_text"],
                    "fact_priority": int(fact["fact_priority"]),
                    "fact_strength": fact.get("fact_strength"),
                    "safe_as_memory": True,
                    "allowed_memory_policies": [policy],
                    "excluded_target_question_ids": [],
                    "excluded_target_topics": [],
                    "memory_policy": policy,
                    "fact_role": "inferred_persona",
                    "leakage_group": "anes_inferred_persona",
                    "donor_anes_ids": donor_ids,
                    "donor_weights": [float(x) for x in weights.tolist()],
                    "donor_count": int(len(donor_ids)),
                    "effective_sample_size": ess,
                    "support_confidence": confidence,
                    "created_at": pd.Timestamp.now(tz="UTC"),
                }
            )
    return pd.DataFrame(fact_rows, columns=_persona_fact_columns())


def _persona_fact_columns() -> list[str]:
    return [
        "memory_fact_id",
        "ces_id",
        "source_year",
        "source_variable",
        "source_variables",
        "question_id",
        "topic",
        "subtopic",
        "fact_text",
        "fact_priority",
        "fact_strength",
        "safe_as_memory",
        "allowed_memory_policies",
        "excluded_target_question_ids",
        "excluded_target_topics",
        "memory_policy",
        "fact_role",
        "leakage_group",
        "donor_anes_ids",
        "donor_weights",
        "donor_count",
        "effective_sample_size",
        "support_confidence",
        "created_at",
    ]


def _persona_fact_candidates(
    donor_payloads: pd.DataFrame,
    weights: pd.Series,
    *,
    donor_ids: list[str],
    donor_count: int,
    ess: float,
    confidence: float,
    theme_min_share: float = 0.2,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    therm_means = {
        var: _weighted_mean(donor_payloads[var], weights)
        for var in THERMOMETER_VARIABLES
        if var in donor_payloads.columns
    }
    party_fact = _party_thermometer_fact(therm_means, donor_count, ess, confidence)
    if party_fact:
        candidates.append(party_fact)
    emotion_fact = _emotion_fact(donor_payloads, weights, donor_count, ess, confidence)
    if emotion_fact:
        candidates.append(emotion_fact)
    identity_fact = _label_fact(
        donor_payloads,
        weights,
        label_col="V241228_label",
        source_variable="ANES_PERSONA_IDENTITY",
        source_variables=["V241228"],
        topic="party_identification",
        subtopic="identity_importance",
        prefix="Matched ANES respondents suggest party identity tends to be",
        donor_count=donor_count,
        ess=ess,
        confidence=confidence,
        priority=40,
    )
    if identity_fact:
        candidates.append(identity_fact)
    efficacy_fact = _label_fact(
        donor_payloads,
        weights,
        label_col="V241235_label",
        source_variable="ANES_PERSONA_EFFICACY",
        source_variables=["V241235"],
        topic="political_efficacy",
        subtopic="government_responsiveness",
        prefix="Matched ANES respondents suggest government responsiveness is perceived as",
        donor_count=donor_count,
        ess=ess,
        confidence=confidence,
        priority=39,
    )
    if efficacy_fact:
        candidates.append(efficacy_fact)
    open_fact = _open_theme_fact(donor_payloads, weights, donor_count, ess, confidence, theme_min_share)
    if open_fact:
        candidates.append(open_fact)
    candidates.sort(key=lambda fact: (-int(fact["fact_priority"]), fact["source_variable"]))
    return candidates


def _party_thermometer_fact(
    therm_means: dict[str, float],
    donor_count: int,
    ess: float,
    confidence: float,
) -> dict[str, Any] | None:
    dem = therm_means.get("V241166")
    rep = therm_means.get("V241167")
    if dem is None or rep is None or np.isnan(dem) or np.isnan(rep):
        return None
    if abs(dem - rep) < 5:
        direction = "similar warmth toward the Democratic and Republican parties"
    else:
        warmer = "the Democratic Party" if dem > rep else "the Republican Party"
        cooler = "the Republican Party" if dem > rep else "the Democratic Party"
        direction = f"warmer feelings toward {warmer} than {cooler}"
    return {
        "source_variable": "ANES_PERSONA_THERMOMETERS",
        "source_variables": ["V241166", "V241167"],
        "topic": "party_affect",
        "subtopic": "party_thermometers",
        "fact_priority": 45,
        "fact_strength": confidence,
        "fact_text": (
            f"Matched ANES respondents suggest {direction} "
            f"(weighted thermometer means: Democratic Party {dem:.0f}, Republican Party {rep:.0f}; "
            f"donors={donor_count}, ESS={ess:.1f})."
        ),
    }

def _emotion_fact(
    donor_payloads: pd.DataFrame,
    weights: pd.Series,
    donor_count: int,
    ess: float,
    confidence: float,
) -> dict[str, Any] | None:
    means = {
        var: _weighted_mean(donor_payloads[var], weights)
        for var in EMOTION_VARIABLES
        if var in donor_payloads.columns
    }
    means = {var: value for var, value in means.items() if not np.isnan(value)}
    if not means:
        return None
    top = sorted(means.items(), key=lambda item: (-item[1], item[0]))[:3]
    labels = ", ".join(f"{EMOTION_VARIABLES[var]} ({value:.1f}/5)" for var, value in top)
    return {
        "source_variable": "ANES_PERSONA_EMOTIONS",
        "source_variables": [var for var, _ in top],
        "topic": "political_emotion",
        "subtopic": "national_mood",
        "fact_priority": 42,
        "fact_strength": confidence,
        "fact_text": (
            f"Matched ANES respondents suggest the strongest national mood signals are {labels} "
            f"(donors={donor_count}, ESS={ess:.1f})."
        ),
    }


def _label_fact(
    donor_payloads: pd.DataFrame,
    weights: pd.Series,
    *,
    label_col: str,
    source_variable: str,
    source_variables: list[str],
    topic: str,
    subtopic: str,
    prefix: str,
    donor_count: int,
    ess: float,
    confidence: float,
    priority: int,
) -> dict[str, Any] | None:
    if label_col not in donor_payloads.columns:
        return None
    label, share = _weighted_mode(donor_payloads[label_col], weights)
    if not label:
        return None
    return {
        "source_variable": source_variable,
        "source_variables": source_variables,
        "topic": topic,
        "subtopic": subtopic,
        "fact_priority": priority,
        "fact_strength": confidence,
        "fact_text": (
            f"{prefix} {label} (weighted donor share {share:.2f}; donors={donor_count}, ESS={ess:.1f})."
        ),
    }


def _open_theme_fact(
    donor_payloads: pd.DataFrame,
    weights: pd.Series,
    donor_count: int,
    ess: float,
    confidence: float,
    min_share: float,
) -> dict[str, Any] | None:
    if "open_text_themes" not in donor_payloads.columns:
        return None
    theme_scores: dict[str, float] = {}
    source_vars: set[str] = set()
    for row_idx, (_, row) in enumerate(donor_payloads.iterrows()):
        themes = set(_list_or_empty(row.get("open_text_themes")))
        for theme in themes:
            theme_scores[theme] = theme_scores.get(theme, 0.0) + float(weights.iloc[row_idx])
        source_vars.update(_list_or_empty(row.get("open_text_source_variables")))
    if not theme_scores:
        return None
    theme, share = sorted(theme_scores.items(), key=lambda item: (-item[1], item[0]))[0]
    if share < min_share:
        return None
    readable = theme.replace("_", " ")
    return {
        "source_variable": "ANES_PERSONA_OPEN_TEXT",
        "source_variables": sorted(source_vars) or list(OPEN_END_VARIABLES),
        "topic": "open_ended_persona",
        "subtopic": theme,
        "fact_priority": 38,
        "fact_strength": confidence,
        "fact_text": (
            f"Matched ANES open-ended responses suggest {readable} is a recurring consideration "
            f"(weighted donor share {share:.2f}; donors={donor_count}, ESS={ess:.1f})."
        ),
    }


def _extend_allowed_policies(value: Any, policy: str) -> list[str]:
    raw_items: list[Any]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raw_items = []
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    elif hasattr(value, "tolist"):
        raw_items = list(value.tolist())
    else:
        raw_items = [value]
    policies = []
    for item in raw_items:
        if isinstance(item, (list, tuple, set)):
            policies.extend(str(inner) for inner in item)
        elif hasattr(item, "tolist") and not isinstance(item, str):
            nested = item.tolist()
            if isinstance(nested, list):
                policies.extend(str(inner) for inner in nested)
            else:
                policies.append(str(nested))
        else:
            policies.append(str(item))
    if "strict_pre_no_vote_v1" in policies and policy not in policies:
        policies.append(policy)
    return policies


def enrich_ces_memory(
    ces_respondents: pd.DataFrame,
    ces_memory_facts: pd.DataFrame,
    persona_facts: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append persona facts and rebuild memory cards for the new policy."""

    cfg = cfg or load_persona_config()
    policy = str(cfg.get("policy", PERSONA_MEMORY_POLICY))
    matching = cfg.get("matching", {})
    max_observed = int(matching.get("max_observed_facts", 24))
    max_persona = int(matching.get("max_persona_facts", 6))
    observed = ces_memory_facts.copy()
    if "allowed_memory_policies" in observed.columns:
        observed["allowed_memory_policies"] = observed["allowed_memory_policies"].map(
            lambda value: _extend_allowed_policies(value, policy)
        )
    for col in _persona_fact_columns():
        if col not in observed.columns:
            observed[col] = None
    for col in observed.columns:
        if col not in persona_facts.columns:
            persona_facts[col] = None
    enriched = pd.concat([observed, persona_facts[observed.columns]], ignore_index=True)
    cards = _build_enriched_cards(ces_respondents, enriched, policy, max_observed, max_persona)
    return enriched, cards


def _build_enriched_cards(
    ces_respondents: pd.DataFrame,
    facts: pd.DataFrame,
    policy: str,
    max_observed: int,
    max_persona: int,
) -> pd.DataFrame:
    facts = facts.copy()
    facts["ces_id"] = facts["ces_id"].astype(str)
    role = facts["fact_role"].fillna("safe_pre").astype(str)
    observed = facts[role != "inferred_persona"].copy()
    persona = facts[role == "inferred_persona"].copy()
    sort_cols = ["ces_id", "fact_priority", "source_variable"]
    selected_parts: list[pd.DataFrame] = []
    if not observed.empty:
        selected_parts.append(
            observed.sort_values(sort_cols, ascending=[True, False, True])
            .groupby("ces_id", sort=False, group_keys=False)
            .head(max_observed)
        )
    if not persona.empty:
        selected_parts.append(
            persona.sort_values(sort_cols, ascending=[True, False, True])
            .groupby("ces_id", sort=False, group_keys=False)
            .head(max_persona)
        )
    selected_facts = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else facts.head(0)
    )
    if not selected_facts.empty:
        selected_facts["_role_order"] = (
            selected_facts["fact_role"].fillna("safe_pre").astype(str).eq("inferred_persona").astype(int)
        )
        selected_facts = selected_facts.sort_values(
            ["ces_id", "_role_order", "fact_priority", "source_variable"],
            ascending=[True, True, False, True],
        ).drop(columns=["_role_order"])
    grouped = {
        ces_id: group
        for ces_id, group in selected_facts.groupby("ces_id", sort=False)
    }
    empty = selected_facts.head(0)
    rows: list[dict[str, Any]] = []
    now = pd.Timestamp.now(tz="UTC")
    for _, respondent in ces_respondents.iterrows():
        ces_id = str(respondent["ces_id"])
        selected = grouped.get(ces_id, empty)
        fact_ids = selected["memory_fact_id"].astype(str).tolist()
        rows.append(
            {
                "memory_card_id": f"{ces_id}_{policy}",
                "ces_id": ces_id,
                "source_year": int(respondent.get("source_year", 0) or 0),
                "memory_policy": policy,
                "fact_ids": fact_ids,
                "memory_text": "\n".join(f"- {text}" for text in selected["fact_text"].fillna("").astype(str)),
                "n_facts": int(len(selected)),
                "all_fact_ids": fact_ids,
                "max_facts": int(max_observed + max_persona),
                "created_at": now,
            }
        )
    return pd.DataFrame(rows)


def persona_feature_audit(ces_profiles: pd.DataFrame, anes_profiles: pd.DataFrame, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for feature in _feature_specs(cfg):
        name = str(feature["name"])
        if name not in ces_profiles or name not in anes_profiles:
            rows.append({"feature": name, "available": False, "reason": "missing_profile_column"})
            continue
        rows.append(
            {
                "feature": name,
                "available": True,
                "kind": feature.get("kind", "nominal"),
                "weight": float(feature.get("weight", 1.0)),
                "ces_nonmissing_share": round(float((~_profile_missing(ces_profiles[name])).mean()), 4),
                "anes_nonmissing_share": round(float((~_profile_missing(anes_profiles[name])).mean()), 4),
                "source_note": feature.get("source_note"),
            }
        )
    return rows


def build_ces_anes_persona(
    *,
    ces_respondents_path: str | Path,
    ces_answers_path: str | Path,
    ces_memory_facts_path: str | Path,
    ces_memory_cards_path: str | Path,
    anes_raw_path: str | Path,
    anes_open_ends_path: str | Path,
    config_path: str | Path,
    out_dir: str | Path,
    limit_ces: int | None = None,
    limit_anes: int | None = None,
) -> dict[str, Path]:
    """Run the offline CES-ANES persona enrichment pipeline."""

    cfg = load_persona_config(config_path)
    out = ensure_dir(out_dir)

    ces_respondents = pd.read_parquet(ces_respondents_path)
    ces_answers = pd.read_parquet(ces_answers_path)
    ces_memory_facts = pd.read_parquet(ces_memory_facts_path)
    # Keep this input explicit in the public interface, even though facts are the
    # source of truth for rebuilding new cards.
    pd.read_parquet(ces_memory_cards_path)

    if limit_ces is not None:
        ces_respondents = ces_respondents.head(int(limit_ces)).copy()
        ids = set(ces_respondents["ces_id"].astype(str))
        ces_answers = ces_answers[ces_answers["ces_id"].astype(str).isin(ids)].copy()
        ces_memory_facts = ces_memory_facts[ces_memory_facts["ces_id"].astype(str).isin(ids)].copy()

    anes_raw = load_anes_raw_for_persona(anes_raw_path, cfg)
    if limit_anes is not None:
        anes_raw = anes_raw.head(int(limit_anes)).copy()

    open_ends = read_anes_open_ends(
        anes_open_ends_path,
        cfg.get("persona", {}).get("open_ends", OPEN_END_VARIABLES),
    )
    if limit_anes is not None:
        allowed_anes = set(anes_raw[ANES_ID_COL].map(clean_string))
        open_ends = open_ends[open_ends["anes_id"].isin(allowed_anes)].copy()

    ces_profiles = build_ces_bridge_profiles(ces_respondents, ces_answers, cfg)
    anes_profiles = build_anes_bridge_profiles(anes_raw, cfg)
    matches, match_summary = match_ces_to_anes(ces_profiles, anes_profiles, cfg)
    payloads = build_anes_persona_payloads(anes_raw, open_ends, cfg)
    persona_facts = build_persona_facts(ces_profiles, matches, match_summary, payloads, cfg)
    enriched_facts, enriched_cards = enrich_ces_memory(ces_respondents, ces_memory_facts, persona_facts, cfg)

    paths = {
        "ces_bridge_profiles": out / "ces_bridge_profiles.parquet",
        "anes_bridge_profiles": out / "anes_bridge_profiles.parquet",
        "anes_persona_payloads": out / "anes_persona_payloads.parquet",
        "ces_anes_matches": out / "ces_anes_matches.parquet",
        "ces_anes_match_summary": out / "ces_anes_match_summary.parquet",
        "ces_anes_persona_facts": out / "ces_anes_persona_facts.parquet",
        "ces_memory_facts_enriched": out / "ces_memory_facts_enriched.parquet",
        "ces_memory_cards_enriched": out / "ces_memory_cards_enriched.parquet",
        "audit": out / "ces_anes_persona_audit.json",
    }
    write_table(ces_profiles, paths["ces_bridge_profiles"])
    write_table(anes_profiles, paths["anes_bridge_profiles"])
    write_table(payloads, paths["anes_persona_payloads"])
    write_table(matches, paths["ces_anes_matches"])
    write_table(match_summary, paths["ces_anes_match_summary"])
    write_table(persona_facts, paths["ces_anes_persona_facts"])
    write_table(enriched_facts, paths["ces_memory_facts_enriched"])
    write_table(enriched_cards, paths["ces_memory_cards_enriched"])

    audit = {
        "policy": cfg.get("policy", PERSONA_MEMORY_POLICY),
        "limits": {"limit_ces": limit_ces, "limit_anes": limit_anes},
        "input_rows": {
            "ces_respondents": int(len(ces_respondents)),
            "ces_answers": int(len(ces_answers)),
            "ces_memory_facts": int(len(ces_memory_facts)),
            "anes_raw": int(len(anes_raw)),
            "anes_open_ends": int(len(open_ends)),
        },
        "output_rows": {
            "matches": int(len(matches)),
            "match_summary": int(len(match_summary)),
            "persona_facts": int(len(persona_facts)),
            "enriched_facts": int(len(enriched_facts)),
            "enriched_cards": int(len(enriched_cards)),
        },
        "feature_audit": persona_feature_audit(ces_profiles, anes_profiles, cfg),
        "safe_persona_source_variables": list(SAFE_PERSONA_SOURCE_VARIABLES),
        "synthetic_persona_source_variables": sorted(SYNTHETIC_PERSONA_SOURCE_VARIABLES),
    }
    write_json(audit, paths["audit"])
    return paths
