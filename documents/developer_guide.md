# Developer Guide

This guide is for a new human or AI maintainer taking over the repository.

## Purpose

LM Voting Simulator tests whether LLM-based voter agents can reproduce
individual survey outcomes and aggregate election results. The current mainline
uses 2024 CES respondents, leakage-controlled survey memory, local or hosted
LLMs, non-LLM baselines, and MIT Election Lab official returns.

The project is not a dashboard or service. It is a reproducible batch pipeline
that writes Parquet artifacts and Markdown reports.

## Code Map

Core orchestration:

- `src/election_sim/cli.py`: Typer CLI entry points.
- `src/election_sim/simulation.py`: run orchestration for fixture, ANES
  individual benchmark, and CES election simulation modes.
- `src/election_sim/config.py`: Pydantic run config models.
- `src/election_sim/io.py`: YAML/table/json I/O helpers.
- `src/election_sim/validation.py`: lightweight dataframe contract checks.

Data ingest:

- `src/election_sim/anes.py`: ANES fixture/real minimal ingestion and ANES
  memory wrapper.
- `src/election_sim/ces.py`: CES respondent, answer, target, context, and
  legacy cell-distribution builders.
- `src/election_sim/mit.py`: MIT Election Lab presidential returns processor.
- `src/election_sim/gdelt.py`: fixture context card loader and time filtering.

Agent and prompt stack:

- `src/election_sim/population.py`: CES-row agents and legacy CES-cell/ANES
  archetype agents.
- `src/election_sim/survey_memory.py`: survey fact rendering, leakage policy
  guard, and leakage audit.
- `src/election_sim/prompts.py`: prompt templates and parser wrappers.
- `src/election_sim/llm.py`: mock, Ollama, and OpenAI-compatible clients.
- `src/election_sim/ces_schema.py`: canonical CES turnout + vote response
  schema and parser.

Prediction and evaluation:

- `src/election_sim/baselines.py`: legacy fixture baselines.
- `src/election_sim/ces_baselines.py`: CES non-LLM baselines.
- `src/election_sim/aggregation.py`: vote/turnout state aggregation.
- `src/election_sim/evaluation.py`: individual, subgroup, and aggregate metrics.
- `src/election_sim/report.py`: Markdown report generation.

Reference and transforms:

- `src/election_sim/transforms.py`: category transforms and deterministic IDs.
- `src/election_sim/constants.py`: canonical category sets.
- `src/election_sim/reference/us_states.json`: USPS/FIPS/swing-state reference.
- `src/election_sim/reference/leakage_policies.json`: leakage policy reference.

## Config Layout

Important config directories:

- `configs/datasets/`: raw dataset path and dataset-level metadata.
- `configs/crosswalks/`: source-variable to canonical-field mappings.
- `configs/fact_templates/`: survey answer to prompt fact text mappings.
- `configs/questions/`: LLM task/question definitions.
- `configs/runs/`: run configs consumed by `run-simulation`.
- `configs/codebooks/`: static value labels extracted from manuals.

Run configs are intentionally explicit. If a run depends on a processed file,
the path should appear under `paths:` so the report can audit it.

## Data Contracts

CES build writes:

```text
ces_respondents.parquet
ces_answers.parquet
ces_targets.parquet
ces_context.parquet
ces_question_bank.parquet
ces_ingest_report.md
```

CES memory build writes:

```text
ces_memory_facts.parquet
ces_memory_cards.parquet
ces_leakage_audit.parquet
```

MIT build writes:

```text
election_returns_county_2000_2024.parquet
election_returns_state_1976_2024.parquet
president_state_truth.parquet
president_county_truth.parquet
president_historical_features.parquet
mit_ingest_audit.parquet
mit_ingest_report.md
```

Simulation writes:

```text
agents.parquet
prompts.parquet
prompt_preview.md
responses.parquet
individual_eval_metrics.parquet
aggregate_state_results.parquet
aggregate_eval_metrics.parquet
eval_report.md
```

## CES Response Schema

All CES `president_turnout_vote` predictions must parse through
`parse_turnout_vote_json()` in `ces_schema.py`.

Expected JSON:

```json
{
  "turnout_probability": 0.86,
  "vote_probabilities": {
    "democrat": 0.45,
    "republican": 0.45,
    "other": 0.05,
    "undecided": 0.05
  },
  "most_likely_choice": "democrat",
  "confidence": 0.7
}
```

`responses.parquet` keeps raw model output plus normalized columns:

```text
turnout_probability
vote_prob_democrat
vote_prob_republican
vote_prob_other
vote_prob_undecided
most_likely_choice
confidence
parse_status
```

Do not add a second CES parser for the same task. Extend the shared schema if
the task contract changes.

## Leakage Rules

The formal prediction policy is `strict_pre_no_vote_v1`.

Strict policy blocks:

- post-election turnout and vote choice
- TargetSmart validation fields
- direct pre-election vote intention or candidate preference

Poll-informed policy allows direct pre-election intention/preference only as
`fact_role=poll_prior`. The report must make that visible.

TargetSmart fields are evaluation-only under strict and poll-informed policies.
Do not place `TS_*` variables in prompt facts.

## MIT Truth Rules

2024 state truth comes from `countypres_2000-2024.csv` county rollup. If county
rows include `TOTAL` mode, only `TOTAL` is used. Otherwise available modes are
summed. Administrative rows such as overvotes and undervotes are audited but do
not enter candidate truth.

`2024-better-evaluation.csv` is a validation reference, not the primary truth
source.

Historical features should not include 2024 by default when simulating 2024.

## Full-Chain Test

Run these in order after changing ingest, memory, simulation, evaluation, or
report code:

```bash
conda run -n jigsaw python -m election_sim.cli build-ces \
  --config configs/datasets/ces_2024_real_vv.yaml \
  --profile-crosswalk configs/crosswalks/ces_2024_profile.yaml \
  --question-crosswalk configs/crosswalks/ces_2024_pre_questions.yaml \
  --target-crosswalk configs/crosswalks/ces_2024_targets.yaml \
  --context-crosswalk configs/crosswalks/ces_2024_context.yaml \
  --out data/processed/ces/2024_common_vv

conda run -n jigsaw python -m election_sim.cli build-ces-memory \
  --respondents data/processed/ces/2024_common_vv/ces_respondents.parquet \
  --answers data/processed/ces/2024_common_vv/ces_answers.parquet \
  --fact-templates configs/fact_templates/ces_2024_common_facts.yaml \
  --policy strict_pre_no_vote_v1 \
  --out data/processed/ces/2024_common_vv \
  --max-facts 24

conda run -n jigsaw python -m election_sim.cli build-ces-memory \
  --respondents data/processed/ces/2024_common_vv/ces_respondents.parquet \
  --answers data/processed/ces/2024_common_vv/ces_answers.parquet \
  --fact-templates configs/fact_templates/ces_2024_common_facts.yaml \
  --policy poll_informed_pre_v1 \
  --out data/processed/ces/2024_common_vv_poll \
  --max-facts 24

conda run -n jigsaw python -m election_sim.cli build-mit-president \
  --config configs/datasets/mit_president_returns.yaml

conda run -n jigsaw python -m election_sim.cli run-simulation \
  --run-config configs/runs/ces_2024_president_swing_strict_pre.yaml

conda run -n jigsaw python -m election_sim.cli run-simulation \
  --run-config configs/runs/ces_2024_president_swing_poll_informed.yaml

conda run -n jigsaw python -m pytest
```

Optional real-model smoke:

```bash
conda run -n jigsaw python -m election_sim.cli run-simulation \
  --run-config configs/runs/ces_2024_president_qwen08b_3_agent.yaml
```

## Acceptance Checks

After a full-chain run:

- `ces_respondents.parquet` should have 60,000 rows for the current real CES
  file.
- strict memory should have zero prompt facts from `TS_*`, `CC24_401`,
  `CC24_410`, and direct pre-vote variables.
- swing runs should have 700 agents by default.
- mock CES response parse rate should be 100 percent.
- aggregate metrics should include `dem_2p_rmse`, `margin_mae`,
  `winner_accuracy`, state-level errors, and national dem 2p error.
- reports should include dataset artifacts, leakage audit, baseline comparison,
  MIT truth source, MIT audit flags, and known limitations.

## Modification Guidance

Prefer data/config changes over code changes when adding variables:

- New source variables: edit crosswalks and fact templates first.
- New leakage behavior: edit `reference/leakage_policies.json` and tests.
- New candidate/election truth mappings: edit MIT crosswalk configs.
- New LLM provider: implement one client in `llm.py`, then keep responses on
  the shared schema.
- New baseline: add it in `ces_baselines.py` and ensure it emits the shared CES
  turnout + vote JSON schema.

Keep generated files under `data/processed/` and `data/runs/`. Do not commit
real raw data, processed Parquet outputs, or run outputs.
