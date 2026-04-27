from __future__ import annotations

import json
import sys

import pandas as pd

from election_sim.aggregation import aggregate_state_results, aggregate_turnout_vote_state_results
from election_sim.anes import LeakageGuard, build_anes, build_memory_cards
from election_sim.ces_baselines import build_ces_non_llm_baselines
from election_sim.ces_schema import format_turnout_vote_response
from election_sim.ces import build_ces, build_ces_cells, build_ces_memory_cards
from election_sim.config import RunConfig, load_cell_schema, load_run_config
from election_sim.evaluation import election_metrics, individual_turnout_vote_metrics, turnout_vote_election_metrics
from election_sim.gdelt import load_context_cards, select_context_cards
from election_sim.io import load_yaml
from election_sim.mit import normalize_mit_results, state_truth_table, write_mit_processed_artifacts
from election_sim.population import build_agents_from_ces_rows, build_agents_from_frames
from election_sim.prompts import build_prompt, parse_json_answer, parse_turnout_vote_json
from election_sim.questions import load_question_config
from election_sim.reference_data import STATE_FIPS_TO_PO, STATE_PO, SWING_STATES_2024, leakage_policy_reference
from election_sim.simulation import run_simulation
from election_sim.transforms import (
    age_to_group,
    anes_2024_education_to_binary,
    anes_2024_hispanic_to_race_ethnicity,
    anes_2024_party_id_to_party3,
    anes_vote_choice_president,
    ces_education_binary,
    ces_gender4,
    ces_ideo5_to_ideology3,
    ces_pid3_to_party3,
    ces_president_nonvoter_preference,
    ces_president_vote_choice,
    ces_race_ethnicity,
    ces_turnout_self_report,
    ces_validated_turnout,
    education_to_binary,
    ideology7_to_ideology3,
    party7_to_party3,
    state_fips_to_po,
)


RUN_CONFIG = "configs/runs/first_e2e_2024_pa_fixture.yaml"


def test_category_mapping_helpers():
    assert age_to_group(24) == "18_29"
    assert age_to_group(65) == "65_plus"
    assert education_to_binary("college_plus") == "college_plus"
    assert party7_to_party3("republican") == "republican"
    assert ideology7_to_ideology3("liberal") == "liberal"
    assert anes_2024_education_to_binary(12) == "non_college"
    assert anes_2024_education_to_binary(13) == "college_plus"
    assert anes_2024_hispanic_to_race_ethnicity(1) == "hispanic"
    assert anes_2024_hispanic_to_race_ethnicity(2) == "other_or_unknown"
    assert anes_2024_party_id_to_party3(1) == "democrat"
    assert anes_2024_party_id_to_party3(2) == "republican"
    assert anes_2024_party_id_to_party3(3) == "independent_or_other"
    assert anes_vote_choice_president(1) == "democrat"
    assert anes_vote_choice_president(2) == "republican"
    assert anes_vote_choice_president(6) == "other"
    assert anes_vote_choice_president(-1) == "not_vote_or_unknown"
    assert state_fips_to_po(42) == "PA"
    assert state_fips_to_po("01") == "AL"
    assert ces_gender4(3) == "non_binary"
    assert ces_education_binary(5) == "college_plus"
    assert ces_race_ethnicity(1, 2) == "white"
    assert ces_race_ethnicity(1, 1) == "hispanic"
    assert ces_pid3_to_party3(2) == "republican"
    assert ces_ideo5_to_ideology3(4) == "conservative"
    assert ces_turnout_self_report(5) == "voted"
    assert ces_turnout_self_report(3) == "not_voted"
    assert ces_president_vote_choice(1) == "democrat"
    assert ces_president_vote_choice(9) == "not_vote"
    assert ces_president_nonvoter_preference(9) == "undecided"
    assert ces_validated_turnout(6) == "voted"
    assert ces_validated_turnout(7) == "not_voted"


def test_packaged_reference_data_contracts():
    assert len(STATE_PO) == 51
    assert STATE_FIPS_TO_PO["42"] == "PA"
    assert STATE_FIPS_TO_PO["04"] == "AZ"
    assert set(SWING_STATES_2024) == {"PA", "MI", "WI", "GA", "AZ", "NC", "NV"}
    leakage = leakage_policy_reference()
    assert "strict_pre_no_vote_v1" in leakage["supported_memory_policies"]
    assert "CC24_410" in leakage["target_post_variables"]
    assert "TS_" in leakage["targetsmart_prefixes"]


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


def _write_tiny_ces_fixture(tmp_path):
    profile = load_yaml("configs/crosswalks/ces_2024_profile.yaml")
    questions = load_yaml("configs/crosswalks/ces_2024_pre_questions.yaml")
    targets = load_yaml("configs/crosswalks/ces_2024_targets.yaml")
    cols = {profile["respondent_id"]}
    for spec in profile["fields"].values():
        cols.add(spec["variable"])
    for question in questions["questions"]:
        cols.add(question["source_variable"])
    for target in targets["targets"]:
        cols.add(target["source_variable"])
    base = {col: 1 for col in cols}
    base.update(
        {
            "caseid": 101,
            "tookpost": 2,
            "inputstate": 42,
            "birthyr": 1980,
            "gender4": 2,
            "educ": 5,
            "race": 1,
            "hispanic": 2,
            "pid3": 1,
            "pid7": 1,
            "CC24_pid7": 1,
            "ideo5": 2,
            "votereg": 1,
            "votereg_post": 1,
            "cit1": 1,
            "commonweight": 1.0,
            "commonpostweight": 1.2,
            "vvweight": 1.1,
            "vvweight_post": 1.3,
            "TS_voterstatus": 1,
            "TS_g2024": 4,
            "TS_partyreg": 2,
            "CC24_401": 5,
            "CC24_410": 1,
            "CC24_410_nv": None,
            "CC24_303": 1,
            "CC24_312a": 4,
            "CC24_321c": 1,
            "CC24_323a": 2,
            "CC24_361b": 2,
        }
    )
    other = dict(base)
    other.update({"caseid": 102, "pid3": 2, "pid7": 7, "ideo5": 5, "CC24_410": 2, "TS_g2024": 7})
    raw_path = tmp_path / "ces_tiny.csv"
    pd.DataFrame([base, other]).to_csv(raw_path, index=False)
    config_path = tmp_path / "ces_tiny.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: ces_2024_tiny",
                "source: ces",
                "year: 2024",
                f"path: {raw_path}",
                "respondent_id: caseid",
                "codebook: configs/codebooks/ces_2024_value_labels.yaml",
                "schema_version: ces_2024_tiny_v1",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_build_ces_fixture_outputs_contract_tables(tmp_path):
    config_path = _write_tiny_ces_fixture(tmp_path)
    paths = build_ces(
        config_path,
        "configs/crosswalks/ces_2024_profile.yaml",
        "configs/crosswalks/ces_2024_pre_questions.yaml",
        "configs/crosswalks/ces_2024_targets.yaml",
        "configs/crosswalks/ces_2024_context.yaml",
        tmp_path / "ces",
    )
    respondents = pd.read_parquet(paths["respondents"])
    answers = pd.read_parquet(paths["answers"])
    targets = pd.read_parquet(paths["targets"])
    context = pd.read_parquet(paths["context"])
    assert len(respondents) == 2
    assert respondents.iloc[0]["state_po"] == "PA"
    assert respondents.iloc[0]["education_binary"] == "college_plus"
    assert len(answers) > 0
    assert {"turnout_2024_self_report", "president_vote_2024", "turnout_2024_validated"} <= set(targets["target_id"])
    assert set(context["candidate_name"]) == {"Kamala Harris", "Donald Trump"}


def test_ces_memory_policy_and_row_agents(tmp_path):
    config_path = _write_tiny_ces_fixture(tmp_path)
    paths = build_ces(
        config_path,
        "configs/crosswalks/ces_2024_profile.yaml",
        "configs/crosswalks/ces_2024_pre_questions.yaml",
        "configs/crosswalks/ces_2024_targets.yaml",
        "configs/crosswalks/ces_2024_context.yaml",
        tmp_path / "ces",
    )
    memory_paths = build_ces_memory_cards(
        paths["respondents"],
        paths["answers"],
        "configs/fact_templates/ces_2024_common_facts.yaml",
        "strict_pre_no_vote_v1",
        tmp_path / "ces",
        max_facts=3,
    )
    facts = pd.read_parquet(memory_paths["facts"])
    cards = pd.read_parquet(memory_paths["cards"])
    audit = pd.read_parquet(memory_paths["audit"])
    assert not {"CC24_401", "CC24_410", "TS_g2024", "CC24_363", "CC24_364a"} & set(facts["source_variable"])
    assert cards["n_facts"].max() <= 3
    assert audit[audit["source_variable"].isin(["CC24_401", "CC24_410", "TS_g2024"])]["excluded"].all()
    assert audit[audit["source_variable"].isin(["TS_partyreg"])]["excluded"].all()

    cfg = RunConfig.model_validate(
        {
            "run_id": "tiny_ces",
            "scenario": {"year": 2024, "office": "president", "states": ["PA"]},
            "population": {
                "source": "ces_rows",
                "selection": {"states": ["PA"], "tookpost_required": True, "citizen_required": True},
                "sampling": {"mode": "weighted_sample", "n_total_agents": 1, "random_seed": 1},
                "weight": {"column": "weight_common_post"},
            },
        }
    )
    agents = build_agents_from_ces_rows(cfg, pd.read_parquet(paths["respondents"]), cards)
    assert len(agents) == 1
    assert agents.iloc[0]["base_ces_id"] in {"101", "102"}
    assert agents.iloc[0]["memory_card_id"]
    assert "weight_missing_reason" in agents.columns


def test_ces_poll_informed_policy_marks_poll_prior(tmp_path):
    config_path = _write_tiny_ces_fixture(tmp_path)
    paths = build_ces(
        config_path,
        "configs/crosswalks/ces_2024_profile.yaml",
        "configs/crosswalks/ces_2024_pre_questions.yaml",
        "configs/crosswalks/ces_2024_targets.yaml",
        "configs/crosswalks/ces_2024_context.yaml",
        tmp_path / "ces",
    )
    memory_paths = build_ces_memory_cards(
        paths["respondents"],
        paths["answers"],
        "configs/fact_templates/ces_2024_common_facts.yaml",
        "poll_informed_pre_v1",
        tmp_path / "ces_poll",
        max_facts=10,
    )
    facts = pd.read_parquet(memory_paths["facts"])
    audit = pd.read_parquet(memory_paths["audit"])
    poll_facts = facts[facts["source_variable"].isin(["CC24_363", "CC24_364a"])]
    assert not poll_facts.empty
    assert set(poll_facts["fact_role"]) == {"poll_prior"}
    assert audit[audit["source_variable"].isin(["TS_g2024", "TS_partyreg"])]["excluded"].all()
    assert not audit[audit["source_variable"].isin(["CC24_363", "CC24_364a"])]["excluded"].any()


def test_turnout_vote_parser_and_aggregation():
    parsed = parse_turnout_vote_json(
        json.dumps(
            {
                "turnout_probability": 0.8,
                "vote_probabilities": {"democrat": 0.7, "republican": 0.2, "other": 0.05, "undecided": 0.05},
                "most_likely_choice": "democrat",
                "confidence": 0.9,
            }
        )
    )
    assert parsed["parse_status"] == "ok"
    assert parsed["most_likely_choice"] == "democrat"
    assert parse_turnout_vote_json('{"answer": "democrat"}')["parse_status"] == "invalid_schema"
    assert (
        parse_turnout_vote_json(
            json.dumps(
                {
                    "turnout_probability": 1.2,
                    "vote_probabilities": {"democrat": 0.7, "republican": 0.2, "other": 0.05, "undecided": 0.05},
                    "most_likely_choice": "democrat",
                    "confidence": 0.9,
                }
            )
        )["parse_status"]
        == "invalid_probability"
    )
    assert (
        parse_turnout_vote_json(
            json.dumps(
                {
                    "turnout_probability": 0.8,
                    "vote_probabilities": {"democrat": 0.7, "republican": 0.2, "other": 0.05, "undecided": 0.05},
                    "most_likely_choice": "green",
                    "confidence": 0.9,
                }
            )
        )["parse_status"]
        == "invalid_choice"
    )

    agents = pd.DataFrame(
        [
            {"agent_id": "a1", "state_po": "PA", "sample_weight": 2.0, "weight_column": "w"},
            {"agent_id": "a2", "state_po": "PA", "sample_weight": 1.0, "weight_column": "w"},
        ]
    )
    responses = pd.DataFrame(
        [
            {
                "agent_id": "a1",
                "baseline": "b",
                "model_name": "m",
                "turnout_probability": 0.5,
                "vote_prob_democrat": 0.8,
                "vote_prob_republican": 0.2,
                "vote_prob_other": 0.0,
                "vote_prob_undecided": 0.0,
            },
            {
                "agent_id": "a2",
                "baseline": "b",
                "model_name": "m",
                "turnout_probability": 1.0,
                "vote_prob_democrat": 0.0,
                "vote_prob_republican": 1.0,
                "vote_prob_other": 0.0,
                "vote_prob_undecided": 0.0,
            },
        ]
    )
    aggregate = aggregate_turnout_vote_state_results(responses, agents, "r", 2024)
    row = aggregate.iloc[0]
    assert abs(row["expected_dem_votes"] - 0.8) < 1e-12
    assert abs(row["expected_rep_votes"] - 1.2) < 1e-12
    assert abs(row["dem_share_2p"] - 0.4) < 1e-12


def test_ces_non_llm_baselines_emit_canonical_schema(tmp_path):
    config_path = _write_tiny_ces_fixture(tmp_path)
    paths = build_ces(
        config_path,
        "configs/crosswalks/ces_2024_profile.yaml",
        "configs/crosswalks/ces_2024_pre_questions.yaml",
        "configs/crosswalks/ces_2024_targets.yaml",
        "configs/crosswalks/ces_2024_context.yaml",
        tmp_path / "ces",
    )
    respondents = pd.read_parquet(paths["respondents"])
    answers = pd.read_parquet(paths["answers"])
    targets = pd.read_parquet(paths["targets"])
    cards = pd.DataFrame({"ces_id": respondents["ces_id"], "memory_card_id": ["m1", "m2"]})
    cfg = RunConfig.model_validate(
        {
            "run_id": "tiny_ces",
            "scenario": {"year": 2024, "office": "president", "states": ["PA"]},
            "population": {
                "source": "ces_rows",
                "selection": {"states": ["PA"]},
                "sampling": {"mode": "all_rows"},
                "weight": {"column": "weight_common_post"},
            },
        }
    )
    agents = build_agents_from_ces_rows(cfg, respondents, cards)
    baselines = build_ces_non_llm_baselines(
        ["party_id_baseline", "sklearn_logit_pre_only", "sklearn_logit_poll_informed"],
        respondents=respondents,
        answers=answers,
        targets=targets,
        agents=agents.head(1),
    )
    for baseline in baselines.values():
        parsed = parse_turnout_vote_json(baseline.predict(agents.iloc[0]).raw_response)
        assert parsed["parse_status"] == "ok"


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


def test_real_mit_president_processing_outputs_truth_tables(tmp_path):
    paths = write_mit_processed_artifacts("configs/datasets/mit_president_returns.yaml", tmp_path / "mit")
    county_returns = pd.read_parquet(paths["county_returns"])
    state_truth = pd.read_parquet(paths["state_truth"])
    county_truth = pd.read_parquet(paths["county_truth"])
    features = pd.read_parquet(paths["historical_features"])
    audit = pd.read_parquet(paths["audit"])

    assert set(county_returns["year"].unique()) == {2000, 2004, 2008, 2012, 2016, 2020, 2024}
    assert county_returns["state_po"].nunique() == 51
    assert county_returns["county_fips"].astype(str).str.len().eq(5).all()
    assert {"total_row", "summed_modes", "single_mode"} & set(county_returns["mode_policy_used"])
    assert not county_returns["candidate_norm"].isin(["TOTAL VOTES CAST", "OVERVOTES", "UNDERVOTES", "SPOILED"]).any()
    assert {"administrative_row_excluded", "candidatevotes_missing_filled_zero", "county_fips_synthetic"} <= set(
        audit["audit_type"]
    )

    state_2024 = state_truth[state_truth["year"] == 2024].copy()
    assert len(state_2024) == 51
    assert set(state_2024["truth_source"]) == {"mit_county_rollup"}
    assert {"dem_votes", "rep_votes", "dem_share_2p", "margin_2p", "winner"} <= set(state_2024.columns)
    ref = pd.read_csv("data/raw/mit/2024-better-evaluation.csv")
    merged = state_2024.merge(ref, on="state_po", suffixes=("_truth", "_ref"))
    assert len(merged) == 51
    assert (merged["dem_votes_truth"] - merged["dem_votes_ref"]).abs().max() == 0
    assert (merged["rep_votes_truth"] - merged["rep_votes_ref"]).abs().max() == 0
    assert (merged["dem_share_2p_truth"] - merged["dem_share_2p_ref"]).abs().max() < 1e-10

    county_2024 = county_truth[county_truth["year"] == 2024]
    assert {"county_fips", "county_name", "margin_2p"} <= set(county_2024.columns)
    assert features["year"].max() == 2020
    truth_table = state_truth_table(state_2024)
    assert {"true_dem_votes", "true_rep_votes", "true_dem_2p", "true_margin", "true_winner"} <= set(
        truth_table.columns
    )


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


def test_ces_individual_and_aggregate_metrics_include_diagnostics():
    responses = pd.DataFrame(
        [
            {
                "agent_id": "a1",
                "base_ces_id": "101",
                "baseline": "b",
                "model_name": "m",
                "parse_status": "ok",
                "turnout_probability": 0.9,
                "vote_prob_democrat": 0.8,
                "vote_prob_republican": 0.1,
                "vote_prob_other": 0.05,
                "vote_prob_undecided": 0.05,
                "most_likely_choice": "democrat",
            },
            {
                "agent_id": "a2",
                "base_ces_id": "102",
                "baseline": "b",
                "model_name": "m",
                "parse_status": "ok",
                "turnout_probability": 0.2,
                "vote_prob_democrat": 0.2,
                "vote_prob_republican": 0.7,
                "vote_prob_other": 0.05,
                "vote_prob_undecided": 0.05,
                "most_likely_choice": "not_vote",
            },
        ]
    )
    targets = pd.DataFrame(
        [
            {"ces_id": "101", "target_id": "turnout_2024_self_report", "canonical_value": "voted"},
            {"ces_id": "101", "target_id": "president_vote_2024", "canonical_value": "democrat"},
            {"ces_id": "102", "target_id": "turnout_2024_self_report", "canonical_value": "not_voted"},
            {"ces_id": "102", "target_id": "president_vote_2024", "canonical_value": "not_vote"},
        ]
    )
    agents = pd.DataFrame(
        [
            {"agent_id": "a1", "party_id_3": "democrat", "state_po": "PA"},
            {"agent_id": "a2", "party_id_3": "republican", "state_po": "PA"},
        ]
    )
    metrics = individual_turnout_vote_metrics(
        responses,
        targets,
        "r",
        agents=agents,
        subgroup_columns=["party_id"],
        small_n_threshold=30,
    )
    assert {"turnout_auc", "vote_macro_f1", "vote_log_loss", "vote_confusion_count"} <= set(metrics["metric_name"])
    assert metrics[metrics["metric_scope"] == "subgroup"]["small_n"].all()

    aggregate = pd.DataFrame(
        [
            {
                "state_po": "PA",
                "baseline": "b",
                "model_name": "m",
                "dem_share_2p": 0.52,
                "margin_2p": 0.04,
                "winner": "democrat",
            }
        ]
    )
    mit = normalize_mit_results("configs/datasets/mit_president_county_fixture.yaml", 2024)
    agg_metrics = turnout_vote_election_metrics(aggregate, mit, "r")
    assert {"state_dem_2p_rmse", "state_margin_mae", "winner_accuracy", "state_margin_error"} <= set(
        agg_metrics["metric_name"]
    )


def test_gdelt_context_card_time_leakage_filter():
    cards = load_context_cards("configs/gdelt/context_cards_2024_fixture.yaml")
    allowed = select_context_cards(cards, year=2024, states=["PA"], simulation_date="2024-11-05")
    blocked = select_context_cards(cards, year=2024, states=["PA"], simulation_date="2024-10-01")
    assert len(allowed) == 1
    assert blocked.empty


def test_real_anes_2024_questionnaire_parser_finds_curated_variables():
    sys.path.insert(0, "data/raw/anes")
    import parse_anes_questionnaire as parser

    schema = parser.parse_pdf("data/raw/anes/2024/anes_2024_questionnaire.pdf")
    for variable in ["V241049", "V242067", "V241221", "V241177", "V241463"]:
        assert variable in schema["variables"]


def test_real_anes_2020_questionnaire_parser_has_legacy_fallback():
    sys.path.insert(0, "data/raw/anes")
    import parse_anes_questionnaire as parser

    schema = parser.parse_pdf("data/raw/anes/2020/anes_timeseries_2020_questionnaire.pdf")
    assert schema["n_items"] > 0
    assert "V202072" in schema["variables"]


def test_real_anes_2024_profile_crosswalk_maps_minimal_categories(tmp_path):
    paths = build_anes(
        "configs/datasets/anes_2024_real_min.yaml",
        "configs/crosswalks/anes_2024_real_min_profile.yaml",
        "configs/crosswalks/anes_2024_real_min_questions.yaml",
        tmp_path / "anes_real",
    )
    respondents = pd.read_parquet(paths["respondents"])
    answers = pd.read_parquet(paths["answers"])
    assert len(respondents) == 5521
    assert set(respondents["race_ethnicity"]) <= {"hispanic", "other_or_unknown"}
    assert set(respondents["education_binary"]) <= {"non_college", "college_plus", "unknown"}
    assert "V241500a" not in set(answers["source_variable"])
    target = answers[answers["source_variable"] == "V242067"]
    assert {"democrat", "republican", "other", "not_vote_or_unknown"} == set(target["canonical_value"])


def test_real_anes_leakage_guard_removes_pre_and_post_vote_choice():
    question = load_question_config("configs/questions/vote_choice_2024_real_anes.yaml").iloc[0]
    facts = pd.DataFrame(
        [
            {
                "anes_id": "1",
                "memory_fact_id": "economy",
                "source_variable": "V241236",
                "topic": "economy",
                "fact_text": "Allowed economy fact.",
                "safe_as_memory": True,
                "allowed_memory_policies": ["safe_survey_memory_v1"],
                "excluded_target_question_ids": [],
                "excluded_target_topics": [],
            },
            {
                "anes_id": "1",
                "memory_fact_id": "pre_vote",
                "source_variable": "V241049",
                "topic": "vote_choice",
                "fact_text": "Pre vote intent leak.",
                "safe_as_memory": True,
                "allowed_memory_policies": ["safe_survey_memory_v1"],
                "excluded_target_question_ids": ["vote_choice_president_2024"],
                "excluded_target_topics": ["vote_choice"],
            },
            {
                "anes_id": "1",
                "memory_fact_id": "post_vote",
                "source_variable": "V242067",
                "topic": "vote_choice",
                "fact_text": "Post vote leak.",
                "safe_as_memory": False,
                "allowed_memory_policies": [],
                "excluded_target_question_ids": ["vote_choice_president_2024"],
                "excluded_target_topics": ["vote_choice"],
            },
        ]
    )
    filtered = LeakageGuard().filter_facts(facts, question, "safe_survey_memory_v1")
    assert filtered["memory_fact_id"].tolist() == ["economy"]


def test_real_anes_one_agent_mock_smoke_writes_prompt_response_report():
    outputs = run_simulation("configs/runs/real_anes_2024_one_agent_mock.yaml")
    responses = pd.read_parquet(outputs["responses"])
    assert len(responses) == 1
    assert responses.iloc[0]["baseline"] == "survey_memory_llm"
    assert responses.iloc[0]["parse_status"] == "ok"
    assert outputs["prompt_preview"].read_text(encoding="utf-8").count("Additional survey-derived background facts") == 1
    assert outputs["report"].exists()
