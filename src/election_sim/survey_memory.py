"""Survey-derived memory fact rendering and leakage filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io import load_yaml, write_table
from .reference_data import leakage_policy_reference
from .transforms import clean_string


_LEAKAGE_POLICY_REFERENCE = leakage_policy_reference()
TARGET_POST_VARIABLES = {str(var).upper() for var in _LEAKAGE_POLICY_REFERENCE["target_post_variables"]}
TARGETSMART_PREFIXES = tuple(str(prefix).upper() for prefix in _LEAKAGE_POLICY_REFERENCE["targetsmart_prefixes"])
POST_TARGET_PREFIXES = tuple(str(prefix).upper() for prefix in _LEAKAGE_POLICY_REFERENCE["post_target_prefixes"])
DIRECT_PRE_VOTE_PREFIXES = tuple(
    str(prefix).upper() for prefix in _LEAKAGE_POLICY_REFERENCE["direct_pre_vote_prefixes"]
)
ANES_PERSONA_ALLOWED_VARIABLES = {
    str(var).upper() for var in _LEAKAGE_POLICY_REFERENCE.get("anes_persona_allowed_variables", [])
}
ANES_PERSONA_BLOCKED_VARIABLES = {
    str(var).upper() for var in _LEAKAGE_POLICY_REFERENCE.get("anes_persona_blocked_variables", [])
}
AUDIT_PROBE_VARIABLES = tuple(str(var) for var in _LEAKAGE_POLICY_REFERENCE["audit_probe_variables"])
SUPPORTED_MEMORY_POLICIES = set(_LEAKAGE_POLICY_REFERENCE["supported_memory_policies"])
STRICT_PRE_NO_DIRECT_VOTE_POLICIES = {
    "strict_pre_no_vote_v1",
    "strict_pre_no_vote_with_anes_persona_v1",
}


def is_targetsmart_variable(source_variable: Any) -> bool:
    var = clean_string(source_variable).upper()
    return any(var.startswith(prefix) for prefix in TARGETSMART_PREFIXES)


def is_direct_pre_vote_variable(source_variable: Any) -> bool:
    var = clean_string(source_variable).upper()
    return any(var.startswith(prefix) for prefix in DIRECT_PRE_VOTE_PREFIXES)


def is_post_vote_or_turnout_variable(source_variable: Any) -> bool:
    var = clean_string(source_variable).upper()
    return var in TARGET_POST_VARIABLES or any(var.startswith(prefix) for prefix in POST_TARGET_PREFIXES)


def fact_role_for_variable(source_variable: Any, policy: str) -> str:
    if policy == "poll_informed_pre_v1" and is_direct_pre_vote_variable(source_variable):
        return "poll_prior"
    if policy == "post_hoc_explanation_v1":
        return "post_hoc_context"
    return "safe_pre"


def leakage_reason(source_variable: Any, policy: str) -> str | None:
    if policy not in SUPPORTED_MEMORY_POLICIES:
        return "unsupported_memory_policy"
    var = clean_string(source_variable).upper()
    if is_targetsmart_variable(source_variable):
        return "targetsmart_evaluation_only"
    if is_post_vote_or_turnout_variable(source_variable):
        return "policy_blocks_post_vote_or_turnout"
    if policy in STRICT_PRE_NO_DIRECT_VOTE_POLICIES and is_direct_pre_vote_variable(source_variable):
        return "strict_policy_blocks_direct_pre_vote_intention"
    if policy == "strict_pre_no_vote_with_anes_persona_v1":
        if var in ANES_PERSONA_BLOCKED_VARIABLES:
            return "strict_policy_blocks_anes_vote_or_post_vote_variable"
        if var.startswith("V242067"):
            return "strict_policy_blocks_anes_post_vote_variable"
    return None


def is_leakage_variable(source_variable: Any, policy: str) -> bool:
    """Return whether a source variable is blocked by a memory policy."""

    return leakage_reason(source_variable, policy) is not None


def _anes_persona_source_variables_allowed(source_variables: Any) -> bool:
    variables = {clean_string(var).upper() for var in _as_list(source_variables) if clean_string(var)}
    if not variables:
        return False
    if variables & ANES_PERSONA_BLOCKED_VARIABLES:
        return False
    if any(var.startswith("V242067") for var in variables):
        return False
    anes_vars = {var for var in variables if var.startswith("V") or var.startswith("ANES_PERSONA_")}
    return bool(anes_vars) and anes_vars <= ANES_PERSONA_ALLOWED_VARIABLES


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean_string(value).lower() in {"1", "true", "yes", "y"}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return [value]


class LeakageGuard:
    """Filter memory facts that leak target answers or violate policy."""

    @staticmethod
    def _as_set(value: Any) -> set[Any]:
        return set(_as_list(value))

    def filter_facts(
        self,
        facts: pd.DataFrame,
        target_question: pd.Series | dict[str, Any],
        memory_policy: str,
    ) -> pd.DataFrame:
        if facts.empty:
            return facts.copy()
        question = dict(target_question)
        out = facts.copy()
        if "safe_as_memory" in out.columns:
            out = out[out["safe_as_memory"].map(_as_bool)]
        if "allowed_memory_policies" in out.columns:
            out = out[out["allowed_memory_policies"].apply(lambda policies: memory_policy in self._as_set(policies))]
        if "source_variable" in out.columns:
            out = out[~out["source_variable"].apply(lambda var: is_leakage_variable(var, memory_policy))]
        if memory_policy == "strict_pre_no_vote_with_anes_persona_v1" and "source_variables" in out.columns:
            inferred = out.get("fact_role", pd.Series("", index=out.index)).fillna("").astype(str) == "inferred_persona"
            allowed = out["source_variables"].apply(_anes_persona_source_variables_allowed)
            out = out[(~inferred) | allowed]
        if "excluded_target_question_ids" in out.columns:
            out = out[
                ~out["excluded_target_question_ids"].apply(
                    lambda ids: question.get("question_id") in self._as_set(ids)
                )
            ]
        if "excluded_target_topics" in out.columns:
            out = out[
                ~out["excluded_target_topics"].apply(lambda topics: question.get("topic") in self._as_set(topics))
            ]
        excluded_vars = set(question.get("excluded_memory_variables") or [])
        if memory_policy in {"poll_informed_pre_v1", "post_hoc_explanation_v1"}:
            excluded_vars = {var for var in excluded_vars if not is_direct_pre_vote_variable(var)}
        if excluded_vars and "source_variable" in out.columns:
            out = out[~out["source_variable"].isin(excluded_vars)]
        excluded_topics = set(question.get("excluded_memory_topics") or [])
        if excluded_topics and "topic" in out.columns:
            out = out[~out["topic"].isin(excluded_topics)]
        return out


def _profile_facts(row: pd.Series) -> list[str]:
    state = row.get("state_po") or "an unknown state"
    return [
        f"The respondent lives in {state}.",
        f"The respondent is in the {row.get('age_group') or 'unknown'} age group.",
        f"The respondent identifies as {row.get('gender') or 'unknown'}.",
        f"The respondent's race/ethnicity category is {row.get('race_ethnicity') or 'unknown'}.",
        f"The respondent's education category is {row.get('education_binary') or 'unknown'}.",
        f"The respondent's party identification is {row.get('party_id_3') or 'unknown'}.",
        f"The respondent's ideology is {row.get('ideology_3') or 'unknown'}.",
    ]


def build_survey_memory_cards(
    respondents: pd.DataFrame,
    answers: pd.DataFrame,
    fact_templates_path: str | Path,
    policy: str,
    out_path: str | Path,
    *,
    id_col: str,
    output_prefix: str,
    max_facts: int = 24,
    include_profile_facts: bool = False,
) -> dict[str, Path]:
    templates_raw = load_yaml(fact_templates_path)
    templates = templates_raw.get("templates", templates_raw) if isinstance(templates_raw, dict) else templates_raw
    templates = templates or []
    template_by_var = {tpl["source_variable"]: tpl for tpl in templates}

    available_source_vars = set(answers["source_variable"]) if "source_variable" in answers.columns else set()
    fact_frames: list[pd.DataFrame] = []
    for source_var, tpl in template_by_var.items():
        if is_leakage_variable(source_var, policy) or source_var not in available_source_vars:
            continue
        part = answers[answers["source_variable"] == source_var].copy()
        if part.empty:
            continue
        missing = part.get("is_missing", False)
        if not isinstance(missing, pd.Series):
            missing = pd.Series(False, index=part.index)
        canonical = part.get("canonical_value", pd.Series("", index=part.index)).fillna("").astype(str)
        if tpl.get("missing_policy", "skip") == "skip":
            part = part[~missing.astype(bool) & ~canonical.isin(["", "unknown"])]
        if part.empty:
            continue
        value_templates = {str(k): v for k, v in tpl.get("value_templates", {}).items()}
        if value_templates:
            canonical_keys = part.get("canonical_value", pd.Series("", index=part.index)).map(clean_string)
            answer_keys = part.get("answer_code", part.get("raw_value", pd.Series("", index=part.index))).map(clean_string)
            text = canonical_keys.map(value_templates).fillna(answer_keys.map(value_templates))
            part = part[text.notna()].copy()
            text = text[text.notna()].astype(str)
        elif tpl.get("template"):
            template_text = str(tpl["template"])
            labels = part.get("answer_label", pd.Series("", index=part.index)).fillna("").astype(str)
            canon_values = part.get("canonical_value", pd.Series("", index=part.index)).fillna("").astype(str)
            question_texts = part.get("question_text", pd.Series("", index=part.index)).fillna("").astype(str)
            text = pd.Series(
                [
                    template_text.format(
                        answer_label=label or canonical_value,
                        answer_label_lower=(label or canonical_value).lower(),
                        canonical_value=canonical_value,
                        question_text=question_text,
                    )
                    for label, canonical_value, question_text in zip(labels, canon_values, question_texts, strict=False)
                ],
                index=part.index,
            )
        else:
            continue
        if part.empty:
            continue
        allowed_policies = list(tpl.get("allowed_memory_policies", [policy]))
        fact_role = tpl.get("fact_role") or fact_role_for_variable(source_var, policy)
        fact_ids = (
            output_prefix
            + "_"
            + part[id_col].astype(str)
            + "_"
            + source_var
            + "_"
            + part.get("canonical_value", pd.Series("", index=part.index)).fillna("").astype(str)
            + "_"
            + policy
        )
        fact_part = pd.DataFrame(
            {
                "memory_fact_id": fact_ids.to_numpy(),
                id_col: part[id_col].astype(str).to_numpy(),
                "source_year": part.get("source_year", pd.Series(0, index=part.index)).fillna(0).astype(int).to_numpy(),
                "source_variable": source_var,
                "question_id": part.get("question_id", pd.Series(None, index=part.index)).to_numpy(),
                "topic": tpl.get("topic", part["topic"].iloc[0] if "topic" in part.columns else None),
                "subtopic": tpl.get("subtopic"),
                "fact_text": text.to_numpy(),
                "fact_priority": int(tpl.get("priority", tpl.get("fact_priority", 50))),
                "fact_strength": tpl.get("fact_strength"),
                "safe_as_memory": bool(tpl.get("safe_as_memory", True)),
                "allowed_memory_policies": [allowed_policies] * len(part),
                "excluded_target_question_ids": [list(tpl.get("excluded_target_question_ids", []))] * len(part),
                "excluded_target_topics": [list(tpl.get("excluded_target_topics", []))] * len(part),
                "memory_policy": policy,
                "fact_role": fact_role,
                "leakage_group": part.get(
                    "leakage_group",
                    pd.Series(tpl.get("leakage_group", "safe_pre"), index=part.index),
                ).to_numpy(),
                "created_at": pd.Timestamp.now(tz="UTC"),
            }
        )
        fact_frames.append(fact_part)

    fact_columns = [
        "memory_fact_id",
        id_col,
        "source_year",
        "source_variable",
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
        "created_at",
    ]
    facts = pd.concat(fact_frames, ignore_index=True) if fact_frames else pd.DataFrame(columns=fact_columns)

    if not facts.empty:
        sorted_facts = facts.sort_values([id_col, "fact_priority", "source_variable"], ascending=[True, False, True])
        facts_by_id = {
            str(respondent_id): group.head(max_facts).copy()
            for respondent_id, group in sorted_facts.groupby(id_col, sort=False)
        }
    else:
        facts_by_id = {}

    empty_facts = facts.head(0)
    card_rows: list[dict[str, Any]] = []
    for _, respondent in respondents.iterrows():
        respondent_id = str(respondent[id_col])
        person_facts = facts_by_id.get(respondent_id, empty_facts)
        row = {
            "memory_card_id": f"{respondent_id}_{policy}",
            id_col: respondent_id,
            "source_year": int(respondent.get("source_year", 0) or 0),
            "memory_policy": policy,
            "fact_ids": person_facts["memory_fact_id"].tolist(),
            "memory_text": "\n".join(f"- {text}" for text in person_facts["fact_text"].tolist()),
            "n_facts": int(len(person_facts)),
            "all_fact_ids": person_facts["memory_fact_id"].tolist(),
            "max_facts": max_facts,
            "created_at": pd.Timestamp.now(tz="UTC"),
        }
        if include_profile_facts:
            by_topic = {
                topic: group["fact_text"].tolist()
                for topic, group in person_facts.groupby("topic", sort=True)
            }
            row.update(
                {
                    "profile_facts": _profile_facts(respondent),
                    "political_facts": by_topic.get("political_engagement", []),
                    "media_facts": by_topic.get("media_use", []),
                    "issue_facts": [
                        text
                        for topic, texts in by_topic.items()
                        if topic not in {"political_engagement", "media_use", "party_affect"}
                        for text in texts
                    ],
                    "affect_facts": by_topic.get("party_affect", []),
                    "open_end_facts": [],
                }
            )
        card_rows.append(row)

    cards = pd.DataFrame(card_rows)
    out = Path(out_path)
    if out.suffix:
        card_path = out
        facts_path = out.with_name(f"{output_prefix}_memory_facts.parquet")
    else:
        out.mkdir(parents=True, exist_ok=True)
        card_path = out / f"{output_prefix}_memory_cards.parquet"
        facts_path = out / f"{output_prefix}_memory_facts.parquet"
    write_table(facts, facts_path)
    write_table(cards, card_path)
    return {"facts": facts_path, "cards": card_path}


def build_leakage_audit(
    answers: pd.DataFrame,
    fact_templates_path: str | Path,
    policy: str,
    out_path: str | Path,
) -> Path:
    templates_raw = load_yaml(fact_templates_path)
    templates = templates_raw.get("templates", templates_raw) if isinstance(templates_raw, dict) else templates_raw
    templates = templates or []
    template_by_var = {tpl["source_variable"]: tpl for tpl in templates}
    templated_vars = set(template_by_var)
    rows: list[dict[str, Any]] = []
    for source_var in sorted(set(answers.get("source_variable", pd.Series(dtype=str)).dropna().astype(str))):
        reason = leakage_reason(source_var, policy)
        excluded = reason is not None
        tpl = template_by_var.get(source_var, {})
        fact_role = tpl.get("fact_role") or fact_role_for_variable(source_var, policy)
        leakage_group = tpl.get("leakage_group")
        if leakage_group is None and "leakage_group" in answers.columns:
            matches = answers.loc[answers["source_variable"] == source_var, "leakage_group"].dropna()
            leakage_group = matches.iloc[0] if not matches.empty else None
        if source_var not in templated_vars and not excluded:
            reason = "no_fact_template"
            excluded = True
        else:
            reason = reason or "allowed"
        rows.append(
            {
                "source_variable": source_var,
                "question_id": None,
                "policy": policy,
                "excluded": excluded,
                "reason": reason,
                "target_id": None,
                "fact_role": fact_role,
                "leakage_group": leakage_group,
                "potential_leakage_warning": bool(
                    not excluded and fact_role in {"poll_prior", "post_hoc_context"}
                ),
            }
        )
    for source_var in AUDIT_PROBE_VARIABLES:
        if source_var not in {row["source_variable"] for row in rows}:
            reason = leakage_reason(source_var, policy)
            excluded = reason is not None or source_var not in templated_vars
            rows.append(
                {
                    "source_variable": source_var,
                    "question_id": None,
                    "policy": policy,
                    "excluded": excluded,
                    "reason": reason or "no_fact_template",
                    "target_id": None,
                    "fact_role": template_by_var.get(source_var, {}).get(
                        "fact_role", fact_role_for_variable(source_var, policy)
                    ),
                    "leakage_group": template_by_var.get(source_var, {}).get("leakage_group"),
                    "potential_leakage_warning": bool(
                        not excluded
                        and fact_role_for_variable(source_var, policy) in {"poll_prior", "post_hoc_context"}
                    ),
                }
            )
    audit = pd.DataFrame(rows)
    out = Path(out_path)
    audit_path = out / "ces_leakage_audit.parquet" if not out.suffix else out
    write_table(audit, audit_path)
    return audit_path
