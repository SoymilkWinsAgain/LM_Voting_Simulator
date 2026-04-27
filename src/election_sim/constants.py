"""Canonical categories used across data sources."""

from .reference_data import STATE_PO, SWING_STATES_2024

AGE_GROUPS = {"18_29", "30_44", "45_64", "65_plus", "unknown"}
GENDERS = {"male", "female", "non_binary", "other", "other_or_unknown"}
RACE_ETHNICITIES = {"white", "black", "hispanic", "asian", "other_or_unknown"}
EDUCATION_BINARY = {"non_college", "college_plus", "unknown"}
PARTY_ID_3 = {"democrat", "republican", "independent_or_other", "unknown"}
IDEOLOGY_3 = {"liberal", "moderate", "conservative", "unknown"}
VOTE_CHOICES = {"democrat", "republican", "other", "not_vote_or_unknown"}

CANONICAL_SETS = {
    "age_group": AGE_GROUPS,
    "gender": GENDERS,
    "race_ethnicity": RACE_ETHNICITIES,
    "education_binary": EDUCATION_BINARY,
    "party_id_3": PARTY_ID_3,
    "ideology_3": IDEOLOGY_3,
    "vote_choice_president": VOTE_CHOICES,
}
