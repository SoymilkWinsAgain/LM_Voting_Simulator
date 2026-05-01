"""Shared CES turnout and presidential vote response schema."""

from __future__ import annotations

import json
import re
from typing import Any


CES_TURNOUT_VOTE_CHOICES = ["not_vote", "democrat", "republican"]
CES_VOTE_PROBABILITY_CODES = ["democrat", "republican", "other", "undecided"]
CES_MOST_LIKELY_CHOICES = CES_TURNOUT_VOTE_CHOICES
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
        "raw_choice": None,
        "legacy_probability_schema": False,
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


def parse_turnout_vote_json(raw_response: str) -> dict[str, Any]:
    """Parse the canonical CES turnout + vote hard-choice JSON schema.

    The LLM-facing contract is deliberately a hard choice rather than a
    probability estimate. Compatibility probability columns are one-hot system
    derivatives used by the existing aggregation and evaluation code.
    """

    payload = _json_payload(raw_response)
    if payload is None:
        return _empty_parse("failed")
    if "vote_probabilities" in payload or "turnout_probability" in payload:
        return _empty_parse("legacy_probability_schema") | {"legacy_probability_schema": True}
    if set(payload) != {"choice"}:
        return _empty_parse("invalid_schema")
    choice = payload.get("choice")
    if choice not in CES_MOST_LIKELY_CHOICES:
        return _empty_parse("invalid_choice")
    turnout = 0.0 if choice == "not_vote" else 1.0
    return {
        "parse_status": "ok",
        "turnout_probability": turnout,
        "vote_prob_democrat": 1.0 if choice == "democrat" else 0.0,
        "vote_prob_republican": 1.0 if choice == "republican" else 0.0,
        "vote_prob_other": 0.0,
        "vote_prob_undecided": 0.0,
        "most_likely_choice": choice,
        "confidence": None,
        "raw_choice": choice,
        "legacy_probability_schema": False,
    }


def format_turnout_vote_choice_response(choice: str) -> str:
    """Serialize a CES turnout + vote prediction using the canonical hard-choice schema."""

    if choice not in CES_TURNOUT_VOTE_CHOICES:
        raise ValueError(f"Invalid CES turnout-vote choice: {choice}")
    return json.dumps({"choice": choice}, sort_keys=True)


def format_turnout_vote_response(**kwargs: Any) -> str:
    """Backward-compatible wrapper for older callers.

    New CES turnout-vote code should use ``format_turnout_vote_choice_response``.
    This wrapper converts legacy probability arguments to a hard choice.
    """

    if "choice" in kwargs:
        return format_turnout_vote_choice_response(str(kwargs["choice"]))
    turnout = float(kwargs.get("turnout_probability", 0.0) or 0.0)
    probs = kwargs.get("vote_probabilities") or {}
    scores = {
        "not_vote": 1.0 - turnout,
        "democrat": turnout * float(probs.get("democrat", 0.0) or 0.0),
        "republican": turnout * float(probs.get("republican", 0.0) or 0.0),
    }
    choice = max(CES_TURNOUT_VOTE_CHOICES, key=lambda code: (scores[code], -CES_TURNOUT_VOTE_CHOICES.index(code)))
    return json.dumps(
        {"choice": choice},
        sort_keys=True,
    )
