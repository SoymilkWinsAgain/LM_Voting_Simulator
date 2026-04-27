"""Shared CES turnout and presidential vote response schema."""

from __future__ import annotations

import json
import math
import re
from typing import Any


CES_VOTE_PROBABILITY_CODES = ["democrat", "republican", "other", "undecided"]
CES_MOST_LIKELY_CHOICES = [*CES_VOTE_PROBABILITY_CODES, "not_vote"]
CES_RESPONSE_COLUMNS = [
    "turnout_probability",
    "vote_prob_democrat",
    "vote_prob_republican",
    "vote_prob_other",
    "vote_prob_undecided",
    "most_likely_choice",
    "confidence",
    "parse_status",
]


def _empty_parse(status: str) -> dict[str, Any]:
    return {
        "parse_status": status,
        "turnout_probability": None,
        "vote_prob_democrat": None,
        "vote_prob_republican": None,
        "vote_prob_other": None,
        "vote_prob_undecided": None,
        "most_likely_choice": None,
        "confidence": None,
    }


def _json_payload(raw_response: str) -> dict[str, Any] | None:
    text = raw_response.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _probability(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        return None
    return parsed


def parse_turnout_vote_json(raw_response: str) -> dict[str, Any]:
    """Parse the canonical CES turnout + vote JSON schema.

    The schema is deliberately stricter than the legacy hard-label parser:
    malformed probabilities and invalid choices are recorded as parse failures
    instead of being silently repaired.
    """

    payload = _json_payload(raw_response)
    if payload is None:
        return _empty_parse("failed")
    if not {"turnout_probability", "vote_probabilities", "most_likely_choice", "confidence"} <= set(payload):
        return _empty_parse("invalid_schema")
    turnout = _probability(payload.get("turnout_probability"))
    confidence = _probability(payload.get("confidence"))
    probs = payload.get("vote_probabilities")
    if turnout is None or confidence is None:
        return _empty_parse("invalid_probability")
    if not isinstance(probs, dict) or not set(CES_VOTE_PROBABILITY_CODES) <= set(probs):
        return _empty_parse("invalid_schema")
    parsed_probs = {code: _probability(probs.get(code)) for code in CES_VOTE_PROBABILITY_CODES}
    if any(value is None for value in parsed_probs.values()):
        return _empty_parse("invalid_probability")
    total = sum(float(value) for value in parsed_probs.values())
    if total <= 0.0:
        return _empty_parse("invalid_probability")
    normalized = {code: float(value) / total for code, value in parsed_probs.items()}
    choice = payload.get("most_likely_choice")
    if choice not in CES_MOST_LIKELY_CHOICES:
        return _empty_parse("invalid_choice")
    return {
        "parse_status": "ok",
        "turnout_probability": turnout,
        "vote_prob_democrat": normalized["democrat"],
        "vote_prob_republican": normalized["republican"],
        "vote_prob_other": normalized["other"],
        "vote_prob_undecided": normalized["undecided"],
        "most_likely_choice": choice,
        "confidence": confidence,
    }


def format_turnout_vote_response(
    *,
    turnout_probability: float,
    vote_probabilities: dict[str, float],
    most_likely_choice: str,
    confidence: float,
) -> str:
    """Serialize a CES turnout + vote prediction using the canonical schema."""

    return json.dumps(
        {
            "turnout_probability": float(turnout_probability),
            "vote_probabilities": {code: float(vote_probabilities.get(code, 0.0)) for code in CES_VOTE_PROBABILITY_CODES},
            "most_likely_choice": most_likely_choice,
            "confidence": float(confidence),
        },
        sort_keys=True,
    )
