# LM Voting Simulator E06-Large 10-State Subgroup and Calibration Reliability Report

Generated at: 2026-05-01

This report summarizes the large-scale E06: Subgroup and Calibration Reliability. This E06-large run does not call Ollama; it only reuses the completed large-scale E01/E05 artifacts to check whether hard-choice outputs are reliable across subgroups, fairness slices, distribution compression, and deterministic turnout calibration.

## 1. Experiment Setup

### 1.1 Inputs and Outputs

Formal output directory:

`data/runs/eval_suite_local/06_subgroup_calibration_large_10state`

Standalone report file:

`documents/lm_voting_simulator_e06_large_10state_report.md`

Config file:

`configs/eval_suite/e06_subgroup_calibration_large_10state.yaml`

This run reuses two large-scale input sources:

| Source label in E06 | Actual input directory | Role |
| --- | --- | --- |
| `E01_individual_persona` | `data/runs/eval_suite_local/01_individual_persona_large_10state_1000` | 10-state individual persona fidelity, about 1000 LLM agents, plus non-LLM baselines |
| `E05_ablation_placebo` | `data/runs/eval_suite_local/05_ablation_placebo_large_10state_full_ladder_600` | 10-state full-ladder ablation/placebo, 600 agents x 10 LLM baselines |

E02-large and E04-large are only background references and do not enter the E06 table calculations. The current E06 runner only consumes `responses.parquet` and `agents.parquet` from E01/E05, then merges CES targets.

### 1.2 Analysis Dimensions

E06 generates the following subgroup dimensions:

| Dimension | Meaning |
| --- | --- |
| `overall` | Overall full sample |
| `party_id_3` | Three-category party identification |
| `party_id_7` | Seven-category party identification |
| `ideology_3` | Three-category ideology |
| `race_ethnicity` | Race/ethnicity |
| `education_binary` | College degree or above |
| `age_group` | Age group |
| `gender` | Gender |
| `state_po` | State |
| `state_po_x_party_id_3` | State x three-category party |
| `state_po_x_race_ethnicity` | State x race/ethnicity |

`small_n_threshold=30`. Small-sample subgroups are retained but marked as `small_n=True`. The worst-subgroup conclusions below prioritize non-small groups.

### 1.3 Hard-Choice Interpretation Principles

This run uses the three-class hard-choice contract:

- `not_vote`
- `democrat`
- `republican`

Therefore, the probability columns in response tables are not LLM subjective probabilities; they are one-hot compatibility columns derived from hard choices. Calibration in E06 is also not "how calibrated the LLM probabilities are," but a posterior reliability diagnostic for deterministic 0/1 turnout choices.

## 2. Runtime and Quality Gates

### 2.1 Prechecks and Probe

Prechecks passed:

- Compilation passed:
  - `src/election_sim/ces_subgroup_calibration_benchmark.py`
  - `src/election_sim/eval_suite.py`
  - `src/election_sim/cli.py`
- E06 smoke tests passed:
  - `test_ces_subgroup_calibration_benchmark_runner_smoke`
  - `test_turnout_vote_parser_and_aggregation`

Read-only input probe results:

| Source | responses | agents | states | parse_ok | invalid | forbidden | legacy schema | transport |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| E01-large | 13,844 | 13,401 | AZ, CO, GA, MI, MN, NC, NV, PA, VA, WI | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| E05-large | 6,000 | 600 | AZ, CO, GA, MI, MN, NC, NV, PA, VA, WI | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

All key subgroup columns have no missing values:

- `party_id_3`
- `party_id_7`
- `ideology_3`
- `race_ethnicity`
- `education_binary`
- `age_group`
- `gender`
- `state_po`

Probe run:

| Run | Directory | workers | Runtime | Responses | Subgroup rows | Figures | Gates | LLM calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Probe | `data/runs/eval_suite_local/e06_large_probe_10state_w2` | 2 | 28.99s | 19,844 | 1,991 | 6 | PASS | 0 |

### 2.2 Formal Run

The formal run passed:

| Field | Value |
| --- | ---: |
| workers | 6 |
| runtime_seconds | 47.28 |
| n_sources | 2 |
| n_responses | 19,844 |
| n_subgroup_rows | 1,991 |
| n_distribution_rows | 1,991 |
| n_calibration_rows | 23 |
| figures_png | 6 |
| n_llm_tasks | 0 |
| llm_calls_made | 0 |
| all_gates_passed | true |

All formal output files are present:

- `subgroup_reliability_metrics.parquet`
- `distribution_diagnostics.parquet`
- `calibration_bins.parquet`
- `worst_subgroups.parquet`
- `quality_gates.parquet`
- `runtime.json`
- `config_snapshot.yaml`
- `benchmark_report.md`
- `figures/*`

All quality gates passed. The formal `runtime.json` explicitly records `n_llm_tasks=0` and `llm_calls_made=0`.

## 3. Overall Metrics

### 3.1 E01-Large Overall

In E01-large, the individual-level results for LLM baselines are:

| Baseline | n | Vote accuracy | Macro F1 | Turnout accuracy | Turnout Brier | Pred Dem 2P | True Dem 2P | Dem 2P error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ces_demographic_only_llm` | 1,000 | 0.518 | 0.338 | 0.968 | 0.0315 | 0.405 | 0.466 | -0.061 |
| `ces_party_ideology_llm` | 1,000 | 0.917 | 0.611 | 0.968 | 0.0315 | 0.374 | 0.466 | -0.092 |
| `ces_survey_memory_llm_strict` | 1,000 | 0.871 | 0.578 | 0.968 | 0.0315 | 0.309 | 0.466 | -0.157 |

E01-large non-LLM references:

| Baseline | n | Vote accuracy | Macro F1 | Turnout Brier | Pred Dem 2P | True Dem 2P | Dem 2P error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `party_id_baseline` | 2,711 | 0.868 | 0.595 | 0.0457 | 0.563 | 0.501 | +0.063 |
| `sklearn_logit_demographic_only` | 2,711 | 0.627 | 0.412 | 0.0295 | 0.663 | 0.501 | +0.162 |
| `sklearn_logit_pre_only` | 2,711 | 0.947 | 0.647 | 0.0451 | 0.497 | 0.501 | -0.004 |
| `sklearn_logit_poll_informed` | 2,711 | 0.954 | 0.652 | 0.0348 | 0.490 | 0.501 | -0.010 |

Interpretation:

- E01-large again confirms that `party/ideology` is the most effective LLM information source.
- Overall vote accuracy for `strict survey memory` is lower than `party/ideology`, and Dem 2P is also more Republican-skewed.
- Non-LLM `sklearn_logit_pre_only` and `sklearn_logit_poll_informed` are clearly stronger than the current small-model LLM.

### 3.2 E05-Large Overall

Overall metrics for the complete E05-large ladder:

| Baseline | n | Vote accuracy | Macro F1 | Turnout Brier | Pred Dem 2P | True Dem 2P | Dem 2P error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `L1_demographic_only_llm` | 600 | 0.539 | 0.234 | 0.0503 | 0.000 | 0.461 | -0.461 |
| `L2_demographic_state_llm` | 600 | 0.539 | 0.234 | 0.0503 | 0.000 | 0.461 | -0.461 |
| `L3_party_ideology_llm` | 600 | 0.911 | 0.609 | 0.0563 | 0.362 | 0.461 | -0.098 |
| `L4_party_ideology_context_llm` | 600 | 0.557 | 0.261 | 0.0503 | 0.019 | 0.461 | -0.442 |
| `L5_strict_memory_llm` | 600 | 0.789 | 0.513 | 0.0503 | 0.229 | 0.461 | -0.232 |
| `L6_strict_memory_context_llm` | 600 | 0.596 | 0.315 | 0.0503 | 0.059 | 0.461 | -0.402 |
| `L7_poll_informed_memory_context_llm` | 600 | 0.635 | 0.365 | 0.0552 | 0.087 | 0.461 | -0.373 |
| `L8_post_hoc_oracle_memory_context_llm` | 600 | 0.993 | 0.662 | 0.0077 | 0.445 | 0.461 | -0.016 |
| `P1_memory_shuffled_within_state_llm` | 600 | 0.542 | 0.238 | 0.0503 | 0.006 | 0.461 | -0.454 |
| `P2_memory_shuffled_within_party_llm` | 600 | 0.543 | 0.239 | 0.0503 | 0.003 | 0.461 | -0.457 |

Interpretation:

- `L3_party_ideology_llm` remains the most stable non-oracle LLM condition.
- `L5_strict_memory_llm` contains some information, but is weaker than `L3`.
- Once candidate/context is added, `L4`, `L6`, and `L7` clearly move toward Republican collapse.
- `L8` oracle approaches the upper bound, showing that the runner and hard-choice contract can carry the correct answer; the main problem is small-model inference bias under real prompts.

## 4. Distribution Diagnostics: Collapse or Not

### 4.1 Overall Choice Distribution

| Source | Baseline | Pred Democrat | Pred Republican | Pred not_vote | Entropy ratio | Variance ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E01 | `ces_party_ideology_llm` | 0.374 | 0.626 | 0.000 | 0.947 | 0.941 |
| E01 | `ces_survey_memory_llm_strict` | 0.309 | 0.690 | 0.001 | 0.896 | 0.858 |
| E05 | `L3_party_ideology_llm` | 0.357 | 0.628 | 0.015 | 1.047 | 0.924 |
| E05 | `L4_party_ideology_context_llm` | 0.019 | 0.981 | 0.000 | 0.135 | 0.074 |
| E05 | `L5_strict_memory_llm` | 0.229 | 0.771 | 0.000 | 0.780 | 0.711 |
| E05 | `L6_strict_memory_context_llm` | 0.059 | 0.941 | 0.000 | 0.324 | 0.222 |
| E05 | `L7_poll_informed_memory_context_llm` | 0.087 | 0.905 | 0.009 | 0.497 | 0.318 |
| E05 | `L8_post_hoc_oracle_memory_context_llm` | 0.394 | 0.492 | 0.114 | 1.396 | 0.961 |

Interpretation:

- `L4/L6/L7` have very high overall Republican share, and entropy/variance ratios are clearly below 1, indicating strong distribution compression.
- `L1/L2/P1/P2` are basically all Republican, with overall entropy near 0.
- `L3` has an overall entropy ratio near 1, but still underestimates Democratic two-party share.
- `L8` oracle has higher `not_vote`; this is because oracle memory explicitly includes turnout/post-hoc information. It is not a leakage-free simulator and should only be treated as an upper bound/diagnostic.

### 4.2 Caveats for Distribution Metrics

An entropy ratio greater than 1 is not necessarily good. It can occur when the true subgroup itself is almost entirely one-sided. For example, if a Democratic subgroup is truly almost all Democrat, but the model outputs some Republican choices, the predicted distribution can have "more entropy" than the true distribution and the ratio can exceed 1. In that case, it means more within-subgroup error, not better calibration.

## 5. Party Subgroup Results

### 5.1 Three-Category Party

| Source | Baseline | Subgroup | n | Vote accuracy | Pred Dem 2P | True Dem 2P | Dem 2P error |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| E01 | `ces_party_ideology_llm` | democrat | 340 | 0.958 | 0.991 | 0.959 | +0.033 |
| E01 | `ces_party_ideology_llm` | independent/other | 341 | 0.794 | 0.243 | 0.493 | -0.250 |
| E01 | `ces_party_ideology_llm` | republican | 319 | 0.981 | 0.000 | 0.019 | -0.019 |
| E01 | `ces_survey_memory_llm_strict` | democrat | 340 | 0.957 | 0.935 | 0.959 | -0.023 |
| E01 | `ces_survey_memory_llm_strict` | independent/other | 341 | 0.642 | 0.106 | 0.493 | -0.387 |
| E01 | `ces_survey_memory_llm_strict` | republican | 319 | 0.981 | 0.000 | 0.019 | -0.019 |
| E05 | `L3_party_ideology_llm` | democrat | 165 | 0.978 | 0.990 | 0.982 | +0.007 |
| E05 | `L3_party_ideology_llm` | independent/other | 258 | 0.797 | 0.243 | 0.465 | -0.222 |
| E05 | `L3_party_ideology_llm` | republican | 177 | 0.980 | 0.000 | 0.020 | -0.020 |
| E05 | `L4_party_ideology_context_llm` | democrat | 165 | 0.079 | 0.072 | 0.982 | -0.911 |
| E05 | `L5_strict_memory_llm` | democrat | 165 | 0.861 | 0.848 | 0.982 | -0.135 |
| E05 | `L6_strict_memory_context_llm` | democrat | 165 | 0.216 | 0.225 | 0.982 | -0.758 |
| E05 | `L7_poll_informed_memory_context_llm` | democrat | 165 | 0.354 | 0.337 | 0.982 | -0.645 |
| E05 | `L8_post_hoc_oracle_memory_context_llm` | democrat | 165 | 1.000 | 0.983 | 0.982 | +0.000 |

Core conclusions:

- The `party/ideology` prompt handles clear Democrat and Republican personas well.
- The largest instability point is independent/other: the model often pushes this group toward Republican and clearly underestimates Dem 2P.
- After candidate/context is added, the Democratic subgroup is severely and incorrectly pushed toward Republican. `L4/L6/L7` have the largest errors on the Democratic subgroup.
- `L5_strict_memory_llm` is clearly better than `L6_strict_memory_context_llm`, showing that the "strict memory harm" in this round mainly comes from prompt behavior after combining memory with candidate/context, not from the memory table being completely uninformative.

### 5.2 Worst Seven-Category Party Rows

The worst non-small subgroups are mainly Democratic identifiers:

| Baseline | Worst subgroup | n | Vote accuracy | Dem 2P error |
| --- | --- | ---: | ---: | ---: |
| `L4_party_ideology_context_llm` | `party_id_7=Lean Democrat` | 77 | 0.019 | -0.981 |
| `L5_strict_memory_llm` | `party_id_7=Lean Democrat` | 77 | 0.093 | -0.914 |
| `L6_strict_memory_context_llm` | `party_id_7=Lean Democrat` | 77 | 0.019 | -0.981 |
| `L7_poll_informed_memory_context_llm` | `party_id_7=Lean Democrat` | 77 | 0.019 | -0.981 |
| `L3_party_ideology_llm` | `party_id_7=Independent` | 76 | 0.481 | -0.519 |

This shows that the current small model is not uniformly "a bit Republican-skewed"; it has very strong directional errors in several Democratic-leaning subgroups.

## 6. Race/Ethnicity Subgroup Results

Key subgroups:

| Source | Baseline | Race | n | Vote accuracy | Pred Dem 2P | True Dem 2P | Dem 2P error |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| E01 | `ces_party_ideology_llm` | black | 78 | 0.883 | 0.769 | 0.860 | -0.091 |
| E01 | `ces_survey_memory_llm_strict` | black | 78 | 0.813 | 0.576 | 0.860 | -0.284 |
| E05 | `L3_party_ideology_llm` | black | 58 | 0.900 | 0.698 | 0.833 | -0.134 |
| E05 | `L4_party_ideology_context_llm` | black | 58 | 0.183 | 0.013 | 0.833 | -0.820 |
| E05 | `L5_strict_memory_llm` | black | 58 | 0.617 | 0.412 | 0.833 | -0.421 |
| E05 | `L6_strict_memory_context_llm` | black | 58 | 0.233 | 0.113 | 0.833 | -0.719 |
| E05 | `L7_poll_informed_memory_context_llm` | black | 58 | 0.361 | 0.150 | 0.833 | -0.682 |
| E05 | `L8_post_hoc_oracle_memory_context_llm` | black | 58 | 0.996 | 0.829 | 0.833 | -0.004 |

Interpretation:

- The Black subgroup is a clear stress test: true Democratic share is high.
- `L3` still underestimates, but is relatively usable.
- `L4/L6/L7` severely underestimate Democratic share for the Black subgroup, showing that Republican skew triggered by candidate/context prompts is amplified in truly Democratic subgroups.
- Oracle recovers a value close to truth, indicating that this is not a data-label or aggregation-code problem.

## 7. State Dimension and the Added Three States

### 7.1 10-State State-Level Subgroup

Main E01-large LLM results:

| Baseline | State | Vote accuracy | Pred Dem 2P | True Dem 2P | Error |
| --- | --- | ---: | ---: | ---: | ---: |
| `ces_party_ideology_llm` | AZ | 0.967 | 0.374 | 0.402 | -0.028 |
| `ces_party_ideology_llm` | CO | 0.836 | 0.424 | 0.598 | -0.175 |
| `ces_party_ideology_llm` | MI | 0.926 | 0.282 | 0.462 | -0.180 |
| `ces_party_ideology_llm` | VA | 0.949 | 0.449 | 0.576 | -0.127 |
| `ces_party_ideology_llm` | WI | 0.815 | 0.280 | 0.480 | -0.201 |
| `ces_survey_memory_llm_strict` | CO | 0.734 | 0.290 | 0.598 | -0.308 |
| `ces_survey_memory_llm_strict` | NC | 0.752 | 0.302 | 0.546 | -0.245 |
| `ces_survey_memory_llm_strict` | WI | 0.749 | 0.217 | 0.480 | -0.263 |

Key E05-large baselines:

| Baseline | State | Vote accuracy | Pred Dem 2P | True Dem 2P | Error |
| --- | --- | ---: | ---: | ---: | ---: |
| `L3_party_ideology_llm` | CO | 0.811 | 0.452 | 0.702 | -0.250 |
| `L3_party_ideology_llm` | WI | 0.749 | 0.415 | 0.510 | -0.096 |
| `L4_party_ideology_context_llm` | CO | 0.336 | 0.033 | 0.702 | -0.668 |
| `L5_strict_memory_llm` | CO | 0.650 | 0.311 | 0.702 | -0.391 |
| `L6_strict_memory_context_llm` | CO | 0.324 | 0.023 | 0.702 | -0.679 |
| `L7_poll_informed_memory_context_llm` | CO | 0.422 | 0.110 | 0.702 | -0.592 |
| `L8_post_hoc_oracle_memory_context_llm` | CO | 0.980 | 0.666 | 0.702 | -0.036 |

CO is the clearest stress test among the added three states because its true Democratic share is high. Non-oracle conditions severely underestimate CO's Democratic share, especially variants with candidate/context.

### 7.2 Original 7 States vs Added 3 States

Comparison aggregated by state rows:

| Source | Baseline | Group | State count | n sum | Weighted vote accuracy | Mean Dem 2P error | Mean abs Dem 2P error |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| E01 | `ces_party_ideology_llm` | old7 swing | 7 | 701 | 0.924 | -0.081 | 0.081 |
| E01 | `ces_party_ideology_llm` | new3 MN/VA/CO | 3 | 299 | 0.902 | -0.123 | 0.123 |
| E01 | `ces_survey_memory_llm_strict` | old7 swing | 7 | 701 | 0.879 | -0.144 | 0.144 |
| E01 | `ces_survey_memory_llm_strict` | new3 MN/VA/CO | 3 | 299 | 0.856 | -0.174 | 0.174 |
| E05 | `L3_party_ideology_llm` | old7 swing | 7 | 420 | 0.920 | -0.080 | 0.080 |
| E05 | `L3_party_ideology_llm` | new3 MN/VA/CO | 3 | 180 | 0.884 | -0.138 | 0.138 |
| E05 | `L4_party_ideology_context_llm` | old7 swing | 7 | 420 | 0.586 | -0.404 | 0.404 |
| E05 | `L4_party_ideology_context_llm` | new3 MN/VA/CO | 3 | 180 | 0.475 | -0.505 | 0.505 |
| E05 | `L5_strict_memory_llm` | old7 swing | 7 | 420 | 0.795 | -0.225 | 0.225 |
| E05 | `L5_strict_memory_llm` | new3 MN/VA/CO | 3 | 180 | 0.767 | -0.243 | 0.243 |
| E05 | `L6_strict_memory_context_llm` | old7 swing | 7 | 420 | 0.639 | -0.364 | 0.364 |
| E05 | `L6_strict_memory_context_llm` | new3 MN/VA/CO | 3 | 180 | 0.477 | -0.500 | 0.500 |
| E05 | `L7_poll_informed_memory_context_llm` | old7 swing | 7 | 420 | 0.656 | -0.354 | 0.354 |
| E05 | `L7_poll_informed_memory_context_llm` | new3 MN/VA/CO | 3 | 180 | 0.573 | -0.412 | 0.412 |
| E05 | `L8_post_hoc_oracle_memory_context_llm` | old7 swing | 7 | 420 | 0.994 | -0.012 | 0.021 |
| E05 | `L8_post_hoc_oracle_memory_context_llm` | new3 MN/VA/CO | 3 | 180 | 0.987 | -0.017 | 0.020 |

The added MN/VA/CO states do not "fix" the systematic Republican skew. Instead, they make the problem clearer: in Democratic-winning or Democratic-leaning states, non-oracle LLM conditions underestimate Dem 2P more severely.

## 8. Worst Subgroups

### 8.1 Worst Vote Subgroup for Each Key Baseline

| Source | Baseline | Worst non-small subgroup | n | Vote accuracy | Dem 2P error | Turnout Brier |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| E01 | `ces_party_ideology_llm` | `ideology_3=unknown` | 41 | 0.397 | -0.720 | 0.528 |
| E01 | `ces_survey_memory_llm_strict` | `state_po_x_party_id_3=CO x independent_or_other` | 34 | 0.313 | -0.698 | 0.067 |
| E05 | `L3_party_ideology_llm` | `party_id_7=Independent` | 76 | 0.481 | -0.519 | 0.113 |
| E05 | `L4_party_ideology_context_llm` | `party_id_7=Lean Democrat` | 77 | 0.019 | -0.981 | 0.083 |
| E05 | `L5_strict_memory_llm` | `party_id_7=Lean Democrat` | 77 | 0.093 | -0.914 | 0.083 |
| E05 | `L6_strict_memory_context_llm` | `party_id_7=Lean Democrat` | 77 | 0.019 | -0.981 | 0.083 |
| E05 | `L7_poll_informed_memory_context_llm` | `party_id_7=Lean Democrat` | 77 | 0.019 | -0.981 | 0.083 |
| E05 | `L8_post_hoc_oracle_memory_context_llm` | `state_po_x_race_ethnicity=CO x white` | 40 | 0.948 | -0.077 | 0.000 |

### 8.2 Interpretation

- E01's `party/ideology` is strong overall, but `ideology_3=unknown` and some independent subgroups are high risk.
- E05's `L3` mainly fails on Independent.
- The worst rows for `L4/L6/L7` all land on Lean Democrat, and are almost completely wrong in the Republican direction.
- `L5` also fails clearly on Lean Democrat, but not as extremely as `L4/L6/L7`.
- The worst row for `L8` oracle is still high, indicating that E06 metrics and target labels do not have an obvious systematic error.

## 9. Turnout Calibration

E06 turnout calibration is posterior reliability for hard-choice 0/1 turnout:

| Source | Baseline | ECE | MCE | Notes |
| --- | --- | ---: | ---: | --- |
| E01 | all three LLM baselines | 0.0315 | 0.0315 | LLM almost always predicts turnout=1; true turnout is about 0.968 |
| E05 | `L1/L2/L4/L5/L6/P1/P2` | 0.0503 | 0.0503 | Almost always predicts turnout=1; true turnout is about 0.950 |
| E05 | `L3_party_ideology_llm` | 0.0563 | 0.6844 | There are 3 predicted not_vote cases, but weighted true turnout is high, causing high MCE |
| E05 | `L7_poll_informed_memory_context_llm` | 0.0552 | 1.0000 | There are 4 predicted not_vote cases, and all of them actually turned out |
| E05 | `L8_post_hoc_oracle_memory_context_llm` | 0.0077 | 0.1334 | Oracle memory clearly improves turnout choice |

Conclusion:

- Non-oracle LLM conditions almost always choose voting, so turnout calibration mostly reflects the sample's true turnout rate.
- The current MVP's main problem is not turnout, but directional bias in vote choice.
- Oracle can improve turnout, showing that if memory explicitly includes post-hoc turnout information, the system can express `not_vote`; but this is not a leakage-free main simulation.

## 10. Figures

The formal directory generated six figures:

- `figures/e06_subgroup_vote_accuracy_heatmap.png`
- `figures/e06_subgroup_turnout_brier_heatmap.png`
- `figures/e06_subgroup_dem_2p_error.png`
- `figures/e06_turnout_reliability.png`
- `figures/e06_entropy_ratio_by_subgroup.png`
- `figures/e06_variance_ratio_by_subgroup.png`

E06 figure focus baselines have been supplemented with `L4_party_ideology_context_llm` and `L5_strict_memory_llm`, avoiding omission of the key context effect and no-context strict-memory result from E05-large.

## 11. Conclusion

### 11.1 Main Conclusions

1. E06-large runs reliably, all quality gates pass, and there are no additional LLM calls.

2. Large-sample subgroup results reinforce the earlier E01/E05-large conclusion: the current `qwen3.5:2b` gets most of its value from party/ideology information, not strict memory.

3. The `party/ideology` baseline performs well overall and on clear party subgroups, but is still clearly Republican-skewed for Independent/unknown/some state x party subgroups.

4. After adding candidate/context, the model shows severe Republican collapse. `L4/L6/L7` have especially large errors on Democratic identifiers, the Black subgroup, and Democratic-leaning states such as CO/VA/MN.

5. `L5_strict_memory_llm` is better than `L6_strict_memory_context_llm`, indicating that strict memory itself is not completely useless; much of the problem comes from world-knowledge/candidate-context bias triggered after combining memory with candidate/context prompts.

6. `L8` oracle is close to correct, showing that the code pipeline, hard-choice parsing, and subgroup calculations themselves do not have obvious failures; model bias is an experimental result, not a format failure.

7. Adding MN/VA/CO does not change the overall assessment; it more clearly exposes Dem share underestimation in Democratic-winning states.

### 11.2 Assessment of the Simulator

As an MVP, this system's engineering pipeline is now stable:

- The hard-choice contract is stable.
- Concurrency and cache pipelines are stable.
- E01-E06 and the large-series outputs are reusable.
- Quality gates do not expose parsing or schema problems.

But as a "reliable voter simulator," the current small model still fails:

- It does not stably infer individual choices from persona/memory.
- It strongly depends on party labels.
- It shows systematic Republican skew under candidate/context.
- It lacks reliability for Independent, Democratic-leaning subgroups, the Black subgroup, and Democratic-winning states.

More accurately, the current system can be used as an evaluation framework and experiment platform, but `qwen3.5:2b` should not be treated as the reliable final simulator model.

### 11.3 Whether This Changes Earlier Large-Experiment Conclusions

It does not change them; it strengthens them:

- E01-large: `party/ideology` is most effective, and strict memory does not provide a stable gain.
- E02-large: LLM strict still has systematic bias in state-level aggregation.
- E03-large: prompt wording is not the only problem; directional bias remains the core issue.
- E04-large: candidate/party/world-knowledge leakage risk is real.
- E05-large: the complete ladder shows that candidate/context significantly harms Democratic subgroups.

E06-large's new contribution is locating these conclusions in concrete subgroups: Independent, Lean Democrat, Black respondents, and Democratic-leaning state rows such as CO/VA/MN.

## 12. Follow-Up Recommendations

1. If evaluating stronger models later, first reuse the same hard-choice contract and E06-large metrics.

2. If improving prompts, prioritize the Republican collapse triggered by candidate/context, rather than further tightening JSON format constraints.

3. If keeping the LLM simulator path, use the `party/ideology` prompt as the current strongest leakage-free baseline, and treat the strict memory/context prompt as an object needing repair.

4. At the reporting layer, include E06-large subgroup findings in the final summary report, because they explain why overall accuracy can look acceptable while aggregate/state/subgroup reliability remains poor.
