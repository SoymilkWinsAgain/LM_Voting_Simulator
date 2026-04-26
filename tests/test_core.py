from __future__ import annotations

import json

import pandas as pd

from election_sim.aggregation import aggregate_state_results
from election_sim.anes import LeakageGuard, build_anes, build_memory_cards
from election_sim.ces import build_ces_cells
from election_sim.config import load_cell_schema, load_run_config
from election_sim.evaluation import election_metrics
from election_sim.gdelt import load_context_cards, select_context_cards
from election_sim.mit import normalize_mit_results, state_truth_table
from election_sim.population import build_agents_from_frames
from election_sim.prompts import build_prompt, parse_json_answer
from election_sim.questions import load_question_config
from election_sim.transforms import age_to_group, education_to_binary, ideology7_to_ideology3, party7_to_party3


RUN_CONFIG = "configs/runs/first_e2e_2024_pa_fixture.yaml"


def test_category_mapping_helpers():
    assert age_to_group(24) == "18_29"
    assert age_to_group(65) == "65_plus"
    assert education_to_binary("college_plus") == "college_plus"
    assert party7_to_party3("republican") == "republican"
    assert ideology7_to_ideology3("liberal") == "liberal"


def test_anes_profile_crosswalk_maps_categories_correctly(tmp_path):
    paths = build_anes(
        "configs/datasets/anes_2024_fixture.yaml",
        "configs/crosswalks/anes_2024_fixture_profile.yaml",
        "configs/crosswalks/anes_2024_fixture_questions.yaml",
        tmp_path / "anes",
    )
    respondents = pd.read_parquet(paths["respondents"])
    answers = pd.read_parquet(paths["answers"])
    assert set(respondents["age_group"]) <= {"18_29", "30_44", "45_64", "65_plus"}
    assert set(respondents["party_id_3"]) <= {"democrat", "republican", "independent_or_other"}
    assert "vote_choice_president_2024" in set(answers["question_id"])


def test_memory_card_respects_max_facts_and_skips_missing_values(tmp_path):
    paths = build_anes(
        "configs/datasets/anes_2024_fixture.yaml",
        "configs/crosswalks/anes_2024_fixture_profile.yaml",
        "configs/crosswalks/anes_2024_fixture_questions.yaml",
        tmp_path / "anes",
    )
    memory_paths = build_memory_cards(
        paths["respondents"],
        paths["answers"],
        "configs/fact_templates/anes_2024_fixture_facts.yaml",
        "safe_survey_memory_v1",
        tmp_path / "anes",
        max_facts=2,
    )
    cards = pd.read_parquet(memory_paths["cards"])
    facts = pd.read_parquet(memory_paths["facts"])
    assert cards["all_fact_ids"].map(len).max() <= 2
    assert facts["fact_text"].str.len().min() > 0


def test_leakage_guard_removes_vote_choice_facts():
    facts = pd.DataFrame(
        [
            {
                "memory_fact_id": "safe",
                "source_variable": "economy_view",
                "topic": "economy",
                "fact_text": "Economy fact",
                "safe_as_memory": True,
                "allowed_memory_policies": ["safe_survey_memory_v1"],
                "excluded_target_question_ids": [],
                "excluded_target_topics": [],
            },
            {
                "memory_fact_id": "leak",
                "source_variable": "post_vote_choice",
                "topic": "vote_choice",
                "fact_text": "Vote choice fact",
                "safe_as_memory": False,
                "allowed_memory_policies": [],
                "excluded_target_question_ids": ["vote_choice_president_2024"],
                "excluded_target_topics": ["vote_choice"],
            },
        ]
    )
    question = {
        "question_id": "vote_choice_president_2024",
        "topic": "vote_choice",
        "excluded_memory_variables": ["post_vote_choice"],
        "excluded_memory_topics": ["vote_choice"],
    }
    filtered = LeakageGuard().filter_facts(facts, question, "safe_survey_memory_v1")
    assert filtered["memory_fact_id"].tolist() == ["safe"]


def test_question_bank_options_are_valid():
    questions = load_question_config("configs/questions/vote_choice_2024.yaml")
    row = questions.iloc[0]
    options = json.loads(row["options_json"])
    assert set(row["allowed_answer_codes"]) == set(options)


def test_ces_weighted_cell_distribution_sums_to_one_by_state(tmp_path):
    paths = build_ces_cells(
        "configs/datasets/ces_2024_fixture.yaml",
        "configs/crosswalks/ces_2024_fixture_profile.yaml",
        "configs/cell_schemas/mvp_state_cell_v1.yaml",
        tmp_path / "ces",
    )
    dist = pd.read_parquet(paths["cell_distribution"])
    sums = dist.groupby("state_po")["weighted_share_smoothed"].sum()
    assert all(abs(value - 1.0) < 1e-9 for value in sums)
    assert dist["smoothing_lambda"].between(0, 1).all()


def test_archetype_matcher_returns_required_count(tmp_path):
    cfg = load_run_config(RUN_CONFIG)
    anes_paths = build_anes(
        "configs/datasets/anes_2024_fixture.yaml",
        "configs/crosswalks/anes_2024_fixture_profile.yaml",
        "configs/crosswalks/anes_2024_fixture_questions.yaml",
        tmp_path / "anes",
    )
    memory_paths = build_memory_cards(
        anes_paths["respondents"],
        anes_paths["answers"],
        "configs/fact_templates/anes_2024_fixture_facts.yaml",
        "safe_survey_memory_v1",
        tmp_path / "anes",
        max_facts=12,
    )
    ces_paths = build_ces_cells(
        "configs/datasets/ces_2024_fixture.yaml",
        "configs/crosswalks/ces_2024_fixture_profile.yaml",
        "configs/cell_schemas/mvp_state_cell_v1.yaml",
        tmp_path / "ces",
    )
    agents = build_agents_from_frames(
        cfg,
        pd.read_parquet(ces_paths["cell_distribution"]),
        pd.read_parquet(anes_paths["respondents"]),
        pd.read_parquet(memory_paths["cards"]),
        load_cell_schema("configs/cell_schemas/mvp_state_cell_v1.yaml"),
    )
    assert len(agents) == cfg.population.n_agents_per_state
    assert agents["match_level"].notna().all()


def test_mit_candidate_party_aggregation():
    results = normalize_mit_results("configs/datasets/mit_president_county_fixture.yaml", 2024)
    truth = state_truth_table(results)
    row = truth[truth["state_po"] == "PA"].iloc[0]
    assert row["true_winner"] == "democrat"
    assert abs(row["true_dem_2p"] - (510000 / 925000)) < 1e-12


def test_prompt_builder_includes_allowed_facts_only():
    agent = pd.Series(
        {
            "year": 2024,
            "state_po": "PA",
            "age_group": "45_64",
            "gender": "female",
            "race_ethnicity": "white",
            "education_binary": "non_college",
            "party_id_3": "republican",
            "ideology_3": "conservative",
            "base_anes_id": "A1",
        }
    )
    question = load_question_config("configs/questions/vote_choice_2024.yaml").iloc[0]
    facts = pd.DataFrame(
        [
            {
                "anes_id": "A1",
                "memory_fact_id": "economy",
                "source_variable": "economy_view",
                "topic": "economy",
                "fact_text": "Allowed economy fact.",
                "safe_as_memory": True,
                "allowed_memory_policies": ["safe_survey_memory_v1"],
                "excluded_target_question_ids": [],
                "excluded_target_topics": [],
            },
            {
                "anes_id": "A1",
                "memory_fact_id": "vote",
                "source_variable": "post_vote_choice",
                "topic": "vote_choice",
                "fact_text": "Leaking vote fact.",
                "safe_as_memory": False,
                "allowed_memory_policies": [],
                "excluded_target_question_ids": ["vote_choice_president_2024"],
                "excluded_target_topics": ["vote_choice"],
            },
        ]
    )
    prompt, fact_ids = build_prompt(agent, question, "survey_memory", memory_facts=facts)
    assert "Allowed economy fact." in prompt
    assert "Leaking vote fact." not in prompt
    assert fact_ids == ["economy"]


def test_json_parser_accepts_valid_response_and_rejects_invalid_option():
    allowed = ["democrat", "republican"]
    assert parse_json_answer('{"answer": "democrat", "confidence": 0.7}', allowed)["parse_status"] == "ok"
    parsed = parse_json_answer('{"answer": "green", "confidence": 0.7}', allowed)
    assert parsed["parse_status"] == "invalid_option"
    assert parsed["answer"] is None


def test_vote_share_aggregation_and_election_rmse_metric():
    agents = pd.DataFrame(
        [
            {"agent_id": "a1", "state_po": "PA", "sample_weight": 1.0},
            {"agent_id": "a2", "state_po": "PA", "sample_weight": 1.0},
        ]
    )
    responses = pd.DataFrame(
        [
            {
                "agent_id": "a1",
                "baseline": "b",
                "parsed_answer_code": "democrat",
                "probabilities_json": None,
            },
            {
                "agent_id": "a2",
                "baseline": "b",
                "parsed_answer_code": "republican",
                "probabilities_json": None,
            },
        ]
    )
    aggregate = aggregate_state_results(responses, agents, "r", 2024)
    assert aggregate.iloc[0]["dem_share_raw"] == 0.5
    mit = normalize_mit_results("configs/datasets/mit_president_county_fixture.yaml", 2024)
    metrics = election_metrics(aggregate, mit, "r")
    assert {"winner_accuracy", "vote_share_rmse", "margin_mae"} <= set(metrics["metric_name"])


def test_gdelt_context_card_time_leakage_filter():
    cards = load_context_cards("configs/gdelt/context_cards_2024_fixture.yaml")
    allowed = select_context_cards(cards, year=2024, states=["PA"], simulation_date="2024-11-05")
    blocked = select_context_cards(cards, year=2024, states=["PA"], simulation_date="2024-10-01")
    assert len(allowed) == 1
    assert blocked.empty
