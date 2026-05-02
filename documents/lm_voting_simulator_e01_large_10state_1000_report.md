# E01-Large 10-State / 1000-Agent Individual Persona Fidelity Report

Generated at: 2026-05-01
Experiment directory: `data/runs/eval_suite_local/01_individual_persona_large_10state_1000`
Config file: `configs/eval_suite/e01_individual_persona_large_10state_1000.yaml`

## 1. Experiment Goal

This run expands the individual-level persona fidelity evaluation beyond the completed small-scale E01 experiment. The core questions still follow `documents/lm_voting_simulator_evaluation_plan.md`:

1. Can LLM agents reproduce each CES respondent's own 2024 presidential voting behavior at the individual level?
2. How do results change across different information levels?
   - `ces_demographic_only_llm`: demographics only.
   - `ces_party_ideology_llm`: adds party and ideology.
   - `ces_survey_memory_llm_strict`: adds strict pre-election survey memory.
3. After expanding to more states, do the previous conclusions change?
4. After adding Democratic-winning states, does the run mitigate or expose the model's systematic Republican skew?

This run does not overwrite old results. The previous official E01 directory remains:

`data/runs/eval_suite_local/01_individual_persona`

The new directory for this run is:

`data/runs/eval_suite_local/01_individual_persona_large_10state_1000`

## 2. State and Sample Design

This run uses 10 states, about 100 LLM agents per state, for a total of 1000 LLM agents.

The original seven 2024 swing states are retained:

| State | Description |
| --- | --- |
| PA | 2024 Republican win, core swing state |
| MI | 2024 Republican win, core swing state |
| WI | 2024 Republican win, core swing state |
| GA | 2024 Republican win, core swing state |
| AZ | 2024 Republican win, core swing state |
| NV | 2024 Republican win, core swing state |
| NC | 2024 Republican win, core swing state |

Three Democratic-winning states are added:

| State | Selection rationale |
| --- | --- |
| MN | Midwest, similar to WI/MI, but a 2024 Democratic win |
| VA | Southern / Atlantic state, a contrast to NC/GA |
| CO | Western state, a contrast to AZ/NV |

Final LLM agent distribution:

| State | LLM agents |
| --- | ---: |
| PA | 101 |
| MI | 100 |
| WI | 100 |
| GA | 100 |
| AZ | 100 |
| NV | 100 |
| NC | 100 |
| MN | 100 |
| VA | 100 |
| CO | 99 |

The total is 1000 LLM agents. The small 101/99 imbalance in PA/CO comes from the current runner's sampling and deduplication behavior; at this scale it satisfies the run design.

## 3. Runtime Configuration

LLM runtime configuration:

| Item | Value |
| --- | --- |
| Ollama base URL | `http://172.26.48.1:11434` |
| Model | `qwen3.5:2b` |
| Response contract | hard-choice JSON: `{"choice": "not_vote|democrat|republican"}` |
| Max tokens | 80 |
| Formal workers | 4 |
| LLM baselines | 3 |
| LLM agents | 1000 |
| LLM requests | 3000 |

This run continues to exclude poll-informed LLM conditions, so the main LLM experiment does not introduce polling-information leakage. Non-LLM reference baselines include:

| Baseline | Purpose |
| --- | --- |
| `party_id_baseline` | Simple party-rule reference |
| `sklearn_logit_demographic_only` | Demographic statistical model |
| `sklearn_logit_pre_only` | Pre-election information statistical model |
| `sklearn_logit_poll_informed` | Poll-informed reference upper bound |

## 4. Preflight and Probes

All code and API preflight checks passed:

| Check | Result |
| --- | --- |
| `py_compile` | Passed |
| E01/parser related pytest | Passed |
| Ollama `/api/tags` | `qwen3.5:2b` visible |
| Initial GPU | RTX 5070 Laptop GPU, 8151 MiB total, about 6844 MiB free |

Probe results:

| Probe | Scale | Workers | LLM runtime | Median latency | P90 latency | Throughput | GPU peak used | Min free | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | 30 calls | 3 | 9.0s | 0.533s | 1.042s | 3.34/s | 5052 MiB | 2840 MiB | PASS |
| B | 90 calls | 4 | 19.0s | 0.728s | 1.081s | 4.74/s | 5052 MiB | 2840 MiB | PASS |
| C | 90 calls | 5 | 20.4s | 0.973s | 1.483s | 4.41/s | 5051 MiB | 2841 MiB | PASS |

Workers=5 did not fail, and p90 was still around 1.5s, but throughput was lower than workers=4 and latency was worse. Therefore the formal run used workers=4.

The probes found no:

| Exception type | Result |
| --- | --- |
| transport error | 0 |
| invalid choice | 0 |
| forbidden choice | 0 |
| legacy probability schema | 0 |
| parse failure | 0 |

The added three-state probe did not trigger stop-loss rules. In the tiny CO probe sample, `demographic-only` was visibly Republican-skewed, but `party/ideology` and `strict memory` did not show a full three-state collapse, so the formal experiment proceeded.

## 5. Formal Run Overview

The formal run completed with no interruption and no schema exceptions.

| Metric | Value |
| --- | ---: |
| Total wall-clock | 732.1s, about 12.2 min |
| LLM runtime | 581.9s, about 9.7 min |
| Workers | 4 |
| LLM tasks | 3000 |
| Median latency | 0.686s |
| P90 latency | 0.899s |
| Throughput | 5.16 responses/s |
| Cache hit rate | 0.0 |
| GPU peak used | 5054 MiB |
| GPU min free | 2838 MiB |
| GPU peak utilization | 85% |

Quality gates:

| Gate | Value | Result |
| --- | ---: | --- |
| parse_ok_rate | 1.000 | PASS |
| invalid_choice_rate | 0.000 | PASS |
| forbidden_choice_rate | 0.000 | PASS |
| legacy_probability_schema_rate | 0.000 | PASS |
| transport_error_rate | 0.000 | PASS |
| all_gates_passed | true | PASS |

Output files exist and are complete, including:

| File | Description |
| --- | --- |
| `agents.parquet` | Agent table |
| `cohort.parquet` | 10-state cohort |
| `prompts.parquet` | LLM prompts |
| `responses.parquet` | LLM and non-LLM responses |
| `crossfit_responses.parquet` | Crossfit statistical model responses |
| `individual_metrics.parquet` | Individual-level metrics |
| `subgroup_metrics.parquet` | Subgroup metrics |
| `aggregate_eval_metrics.parquet` | Non-LLM aggregate election metrics |
| `runtime_log.parquet` | Per-request LLM runtime log |
| `runtime.json` | Overall runtime observations |
| `llm_cache.jsonl` | LLM cache |
| `benchmark_report.md` | Runner-generated report |
| `figures/*` | Figures |

Main table sizes:

| Table | Shape |
| --- | ---: |
| `responses.parquet` | 13844 x 34 |
| `runtime_log.parquet` | 3000 x 16 |
| `individual_metrics.parquet` | 224 x 10 |
| `subgroup_metrics.parquet` | 13150 x 10 |
| `aggregate_eval_metrics.parquet` | 104 x 14 |
| `cohort.parquet` | 13401 x 47 |
| `agents.parquet` | 13401 x 31 |

Note: `cohort` and `agents` contain the full evaluable cohort. The formal LLM pilot sampled only 1000 agents from it. `responses.parquet` contains both LLM and non-LLM baseline responses.

## 6. Individual-Level Overall Results

The following are weighted individual metrics. The old probability columns should be interpreted only as hard-choice one-hot derived columns, not as LLM subjective probabilities.

| Baseline | parse_ok | Turnout acc | Turnout brier | Vote acc | Vote macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ces_demographic_only_llm` | 1.000 | 0.968 | 0.032 | 0.518 | 0.338 |
| `ces_party_ideology_llm` | 1.000 | 0.968 | 0.032 | 0.917 | 0.611 |
| `ces_survey_memory_llm_strict` | 1.000 | 0.968 | 0.032 | 0.871 | 0.578 |
| `party_id_baseline` | 1.000 | 0.954 | 0.046 | 0.868 | 0.595 |
| `sklearn_logit_demographic_only` | 1.000 | 0.971 | 0.029 | 0.627 | 0.412 |
| `sklearn_logit_pre_only` | 1.000 | 0.955 | 0.045 | 0.947 | 0.647 |
| `sklearn_logit_poll_informed` | 1.000 | 0.965 | 0.035 | 0.954 | 0.652 |

Core conclusions:

1. `party/ideology` is the strongest LLM baseline.
2. `strict memory` not only fails to stably exceed `party/ideology`, it declines:
   - vote accuracy: 0.917 -> 0.871, a drop of 0.046.
   - macro F1: 0.611 -> 0.578, a drop of 0.033.
3. `demographic-only` is clearly insufficient, with vote accuracy only 0.518, close to a weak classifier.
4. Non-LLM `sklearn_logit_pre_only` and `sklearn_logit_poll_informed` remain substantially stronger than all LLM baselines.
5. Turnout accuracy looks high, but it must be interpreted very carefully: the LLM almost always chooses one of the two major-party candidates and rarely chooses `not_vote`. High turnout accuracy largely comes from the high voter share in the sample, not from the model truly learning turnout.

## 7. Comparison with the Previous Small E01 Run

The previous E01 was 4 states, 100 LLM agents, and 300 LLM calls. This run expands to 10 states, 1000 LLM agents, and 3000 LLM calls.

| Baseline | Old vote acc | Large vote acc | Delta | Old F1 | Large F1 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ces_demographic_only_llm` | 0.577 | 0.518 | -0.059 | 0.369 | 0.338 | -0.031 |
| `ces_party_ideology_llm` | 0.977 | 0.917 | -0.060 | 0.651 | 0.611 | -0.040 |
| `ces_survey_memory_llm_strict` | 0.977 | 0.871 | -0.105 | 0.651 | 0.578 | -0.073 |

After scaling up, every LLM baseline declines, indicating that the previous 4-state 100-agent result was optimistic. The most important change is that `strict memory` moved from near `party/ideology` to clearly below `party/ideology`.

This reinforces one directional conclusion from the previous report: the LLM most effectively uses explicit party and ideology information; additional strict survey memory does not provide a stable gain and may introduce noise or make the model more likely to default to Republican outputs.

## 8. Original 7 States vs Added 3 States

| Group | Baseline | N responses | parse_ok | Turnout acc | Turnout brier | Vote acc | Vote macro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original_7 | `ces_demographic_only_llm` | 701 | 1.000 | 0.968 | 0.032 | 0.552 | 0.362 |
| original_7 | `ces_party_ideology_llm` | 701 | 1.000 | 0.968 | 0.032 | 0.924 | 0.615 |
| original_7 | `ces_survey_memory_llm_strict` | 701 | 1.000 | 0.968 | 0.032 | 0.878 | 0.580 |
| new_3 | `ces_demographic_only_llm` | 299 | 1.000 | 0.969 | 0.031 | 0.444 | 0.285 |
| new_3 | `ces_party_ideology_llm` | 299 | 1.000 | 0.969 | 0.031 | 0.903 | 0.602 |
| new_3 | `ces_survey_memory_llm_strict` | 299 | 1.000 | 0.969 | 0.031 | 0.857 | 0.571 |

After adding MN/VA/CO, `party/ideology` still performs best, but the new three states are harder than the original seven:

| Baseline | Original 7 vote acc | New 3 vote acc | Drop |
| --- | ---: | ---: | ---: |
| `demographic-only` | 0.552 | 0.444 | -0.108 |
| `party/ideology` | 0.924 | 0.903 | -0.021 |
| `strict memory` | 0.878 | 0.857 | -0.021 |

The added three states mainly expose Republican skew at the aggregate level, rather than a full collapse of individual-level vote accuracy.

## 9. State-Level Individual Results

Weighted vote accuracy / macro F1:

| Baseline | State | Vote acc | Vote macro F1 | Turnout acc |
| --- | --- | ---: | ---: | ---: |
| `ces_demographic_only_llm` | PA | 0.668 | 0.440 | 1.000 |
| `ces_demographic_only_llm` | MI | 0.528 | 0.350 | 0.850 |
| `ces_demographic_only_llm` | WI | 0.596 | 0.392 | 0.950 |
| `ces_demographic_only_llm` | GA | 0.521 | 0.346 | 0.998 |
| `ces_demographic_only_llm` | AZ | 0.351 | 0.190 | 1.000 |
| `ces_demographic_only_llm` | NV | 0.629 | 0.339 | 0.978 |
| `ces_demographic_only_llm` | NC | 0.496 | 0.330 | 1.000 |
| `ces_demographic_only_llm` | MN | 0.427 | 0.270 | 0.933 |
| `ces_demographic_only_llm` | VA | 0.519 | 0.332 | 1.000 |
| `ces_demographic_only_llm` | CO | 0.406 | 0.196 | 0.974 |
| `ces_party_ideology_llm` | PA | 0.912 | 0.608 | 1.000 |
| `ces_party_ideology_llm` | MI | 0.926 | 0.617 | 0.850 |
| `ces_party_ideology_llm` | WI | 0.815 | 0.541 | 0.950 |
| `ces_party_ideology_llm` | GA | 0.926 | 0.615 | 0.998 |
| `ces_party_ideology_llm` | AZ | 0.967 | 0.645 | 1.000 |
| `ces_party_ideology_llm` | NV | 0.973 | 0.642 | 0.978 |
| `ces_party_ideology_llm` | NC | 0.923 | 0.615 | 1.000 |
| `ces_party_ideology_llm` | MN | 0.953 | 0.635 | 0.933 |
| `ces_party_ideology_llm` | VA | 0.949 | 0.632 | 1.000 |
| `ces_party_ideology_llm` | CO | 0.836 | 0.557 | 0.974 |
| `ces_survey_memory_llm_strict` | PA | 0.896 | 0.597 | 1.000 |
| `ces_survey_memory_llm_strict` | MI | 0.907 | 0.603 | 0.850 |
| `ces_survey_memory_llm_strict` | WI | 0.749 | 0.488 | 0.950 |
| `ces_survey_memory_llm_strict` | GA | 0.960 | 0.638 | 0.998 |
| `ces_survey_memory_llm_strict` | AZ | 0.962 | 0.641 | 1.000 |
| `ces_survey_memory_llm_strict` | NV | 0.911 | 0.578 | 0.978 |
| `ces_survey_memory_llm_strict` | NC | 0.752 | 0.498 | 1.000 |
| `ces_survey_memory_llm_strict` | MN | 0.952 | 0.634 | 0.933 |
| `ces_survey_memory_llm_strict` | VA | 0.940 | 0.627 | 1.000 |
| `ces_survey_memory_llm_strict` | CO | 0.734 | 0.489 | 0.974 |

Clear patterns:

1. `party/ideology` is stronger than `demographic-only` in every state.
2. `strict memory` is close to or slightly stronger in GA/AZ/MN/VA, but clearly weaker than `party/ideology` in WI/NC/CO.
3. The state-level performance of `demographic-only` is highly unstable, especially in AZ/CO/MN.

## 10. Choice Distribution

Overall LLM output distribution:

| Baseline | N | Democrat | Republican | Not vote |
| --- | ---: | ---: | ---: | ---: |
| `ces_demographic_only_llm` | 1000 | 0.355 | 0.645 | 0.000 |
| `ces_party_ideology_llm` | 1000 | 0.431 | 0.569 | 0.000 |
| `ces_survey_memory_llm_strict` | 1000 | 0.375 | 0.624 | 0.001 |

State-level output distribution:

| Baseline | State | D | R | Not vote |
| --- | --- | ---: | ---: | ---: |
| `demographic-only` | PA | 0.238 | 0.762 | 0.000 |
| `demographic-only` | MI | 0.640 | 0.360 | 0.000 |
| `demographic-only` | WI | 0.710 | 0.290 | 0.000 |
| `demographic-only` | GA | 0.500 | 0.500 | 0.000 |
| `demographic-only` | AZ | 0.150 | 0.850 | 0.000 |
| `demographic-only` | NV | 0.170 | 0.830 | 0.000 |
| `demographic-only` | NC | 0.280 | 0.720 | 0.000 |
| `demographic-only` | MN | 0.640 | 0.360 | 0.000 |
| `demographic-only` | VA | 0.200 | 0.800 | 0.000 |
| `demographic-only` | CO | 0.020 | 0.980 | 0.000 |
| `party/ideology` | PA | 0.475 | 0.525 | 0.000 |
| `party/ideology` | MI | 0.420 | 0.580 | 0.000 |
| `party/ideology` | WI | 0.430 | 0.570 | 0.000 |
| `party/ideology` | GA | 0.420 | 0.580 | 0.000 |
| `party/ideology` | AZ | 0.430 | 0.570 | 0.000 |
| `party/ideology` | NV | 0.370 | 0.630 | 0.000 |
| `party/ideology` | NC | 0.480 | 0.520 | 0.000 |
| `party/ideology` | MN | 0.450 | 0.550 | 0.000 |
| `party/ideology` | VA | 0.400 | 0.600 | 0.000 |
| `party/ideology` | CO | 0.434 | 0.566 | 0.000 |
| `strict memory` | PA | 0.376 | 0.624 | 0.000 |
| `strict memory` | MI | 0.370 | 0.620 | 0.010 |
| `strict memory` | WI | 0.370 | 0.630 | 0.000 |
| `strict memory` | GA | 0.370 | 0.630 | 0.000 |
| `strict memory` | AZ | 0.390 | 0.610 | 0.000 |
| `strict memory` | NV | 0.330 | 0.670 | 0.000 |
| `strict memory` | NC | 0.400 | 0.600 | 0.000 |
| `strict memory` | MN | 0.400 | 0.600 | 0.000 |
| `strict memory` | VA | 0.380 | 0.620 | 0.000 |
| `strict memory` | CO | 0.364 | 0.636 | 0.000 |

Interpretation:

1. `demographic-only` output has obvious state-level drift and does not look like a stable voter simulator.
2. `party/ideology` output is closer to a reasonable two-party distribution, but is still overall Republican-skewed.
3. `strict memory` output is more Republican-skewed and almost never uses `not_vote`.
4. The `not_vote` choice almost disappears; this is a turnout modeling problem that the hard-choice system must handle separately later.

## 11. Confusion Matrix

The following are unweighted LLM confusion counts. Vote metrics are computed only for true `democrat/republican`; the `not_vote` row is included to observe turnout hard-choice behavior.

| Baseline | True | Pred D | Pred not_vote | Pred R |
| --- | --- | ---: | ---: | ---: |
| `demographic-only` | democrat | 186 | 0 | 274 |
| `demographic-only` | not_vote | 1 | 0 | 3 |
| `demographic-only` | republican | 119 | 0 | 301 |
| `party/ideology` | democrat | 400 | 0 | 60 |
| `party/ideology` | not_vote | 0 | 0 | 4 |
| `party/ideology` | republican | 12 | 0 | 408 |
| `strict memory` | democrat | 355 | 0 | 105 |
| `strict memory` | not_vote | 0 | 0 | 4 |
| `strict memory` | republican | 5 | 0 | 415 |

The errors for `party/ideology` are relatively balanced: 60 true Democrats were misclassified as Republican, and 12 true Republicans were misclassified as Democrat. `strict memory` more strongly reduces Republicans misclassified as Democrats, but at the cost of pushing more true Democrats toward Republican: 105 true Democrats were misclassified as Republican. This explains why strict memory has worse overall accuracy and aggregate margin.

## 12. Aggregation to State-Level Election Truth

The current runner's `aggregate_eval_metrics.parquet` contains only non-LLM aggregate metrics. Therefore this section separates the runner's non-LLM metrics from the LLM metrics that I manually aggregated from `responses.parquet`.

Manual LLM aggregation method:

1. For each state and each LLM baseline, aggregate hard-choice one-hot outputs weighted by `sample_weight`.
2. `pred_dem_2p = weighted_democrat / (weighted_democrat + weighted_republican)`.
3. Compare against the 2024 state-level MIT truth in `data/processed/mit/president_state_truth.parquet`.
4. `margin_error = 2 * (pred_dem_2p - true_dem_2p)`.

### 12.1 Official Non-LLM Aggregate Metrics

| Baseline | Dem 2P RMSE | Margin MAE | Margin bias | Winner acc |
| --- | ---: | ---: | ---: | ---: |
| `party_id_baseline` | 0.079 | 0.145 | 0.145 | 0.400 |
| `sklearn_logit_demographic_only` | 0.224 | 0.386 | 0.386 | 0.300 |
| `sklearn_logit_poll_informed` | 0.023 | 0.042 | -0.021 | 0.900 |
| `sklearn_logit_pre_only` | 0.024 | 0.041 | -0.023 | 0.900 |

### 12.2 Manual LLM Aggregate Metrics

| Baseline | Dem 2P RMSE | Margin MAE | Margin bias | Winner acc |
| --- | ---: | ---: | ---: | ---: |
| `ces_demographic_only_llm` | 0.265 | 0.467 | -0.136 | 0.500 |
| `ces_party_ideology_llm` | 0.154 | 0.262 | -0.262 | 0.700 |
| `ces_survey_memory_llm_strict` | 0.207 | 0.382 | -0.382 | 0.700 |

Interpretation:

1. `party/ideology` is the best aggregate baseline among LLM conditions, but it remains clearly weaker than `sklearn_logit_pre_only` and `sklearn_logit_poll_informed`.
2. `strict memory` is more Republican-skewed than `party/ideology`, with margin bias expanding from -0.262 to -0.382.
3. LLM winner accuracy of 0.700 does not mean the model predicts accurately; 7 of the 10 states were 2024 Republican wins, so a Republican-skewed model gets a structural advantage on winner accuracy.
4. After adding MN/VA/CO, neither `party/ideology` nor `strict memory` predicted the three Democratic-winning states.

### 12.3 LLM State-Level Details

| Baseline | State | Pred Dem 2P | True Dem 2P | Margin error | Winner correct |
| --- | --- | ---: | ---: | ---: | --- |
| `demographic-only` | PA | 0.353 | 0.491 | -0.276 | true |
| `demographic-only` | MI | 0.728 | 0.493 | 0.470 | false |
| `demographic-only` | WI | 0.710 | 0.496 | 0.430 | false |
| `demographic-only` | GA | 0.622 | 0.489 | 0.266 | false |
| `demographic-only` | AZ | 0.295 | 0.472 | -0.354 | true |
| `demographic-only` | NV | 0.270 | 0.484 | -0.429 | true |
| `demographic-only` | NC | 0.398 | 0.484 | -0.172 | true |
| `demographic-only` | MN | 0.766 | 0.522 | 0.489 | true |
| `demographic-only` | VA | 0.188 | 0.530 | -0.682 | false |
| `demographic-only` | CO | 0.004 | 0.556 | -1.106 | false |
| `party/ideology` | PA | 0.480 | 0.491 | -0.023 | true |
| `party/ideology` | MI | 0.282 | 0.493 | -0.422 | true |
| `party/ideology` | WI | 0.280 | 0.496 | -0.432 | true |
| `party/ideology` | GA | 0.357 | 0.489 | -0.265 | true |
| `party/ideology` | AZ | 0.374 | 0.472 | -0.196 | true |
| `party/ideology` | NV | 0.212 | 0.484 | -0.544 | true |
| `party/ideology` | NC | 0.467 | 0.484 | -0.034 | true |
| `party/ideology` | MN | 0.381 | 0.522 | -0.281 | false |
| `party/ideology` | VA | 0.449 | 0.530 | -0.161 | false |
| `party/ideology` | CO | 0.424 | 0.556 | -0.265 | false |
| `strict memory` | PA | 0.406 | 0.491 | -0.171 | true |
| `strict memory` | MI | 0.270 | 0.493 | -0.445 | true |
| `strict memory` | WI | 0.217 | 0.496 | -0.556 | true |
| `strict memory` | GA | 0.298 | 0.489 | -0.382 | true |
| `strict memory` | AZ | 0.359 | 0.472 | -0.226 | true |
| `strict memory` | NV | 0.155 | 0.484 | -0.659 | true |
| `strict memory` | NC | 0.302 | 0.484 | -0.364 | true |
| `strict memory` | MN | 0.368 | 0.522 | -0.307 | false |
| `strict memory` | VA | 0.441 | 0.530 | -0.176 | false |
| `strict memory` | CO | 0.290 | 0.556 | -0.533 | false |

Key observations:

1. Republicans did win seven swing states in 2024, so "predicting Republican more often" is not itself the problem.
2. The problem is that, relative to truth, the LLM's Democratic two-party share is systematically too low, especially in MN/VA/CO.
3. `party/ideology` has margins very close to truth in PA and NC, but is much too Republican in MI/WI/NV.
4. `strict memory` is more Republican-skewed than `party/ideology` and does not improve the added Democratic-winning states.
5. `demographic-only` has the most unstable aggregate behavior: CO has predicted Dem 2P of only 0.004, which is clearly not a credible simulation.

## 13. Subgroup Risk

The following are the LLM subgroups with the worst weighted vote accuracy, requiring subgroup n >= 25.

| Baseline | Subgroup | N | Vote acc |
| --- | --- | ---: | ---: |
| `demographic-only` | `state_party_id_3=AZ x democrat` | 31 | 0.082 |
| `demographic-only` | `state_party_id_3=CO x democrat` | 32 | 0.088 |
| `demographic-only` | `state_party_id_3=MN x republican` | 30 | 0.169 |
| `demographic-only` | `state_party_id_3=NC x democrat` | 32 | 0.204 |
| `demographic-only` | `state_party_id_3=NV x democrat` | 33 | 0.244 |
| `demographic-only` | `state_party_id_3=VA x democrat` | 34 | 0.271 |
| `demographic-only` | `state_party_id_3=CO x independent_or_other` | 25 | 0.272 |
| `demographic-only` | `state_party_id_3=MN x independent_or_other` | 27 | 0.312 |
| `strict memory` | `state_party_id_3=CO x independent_or_other` | 25 | 0.313 |
| `demographic-only` | `state_po=AZ` | 90 | 0.351 |
| `demographic-only` | `state_party_id_3=MI x republican` | 28 | 0.365 |
| `strict memory` | `state_party_id_3=NC x independent_or_other` | 27 | 0.388 |
| `demographic-only` | `state_po=CO` | 86 | 0.406 |
| `demographic-only` | `state_party_id_3=MI x independent_or_other` | 27 | 0.415 |
| `demographic-only` | `ideology_3=liberal` | 281 | 0.419 |
| `demographic-only` | `state_po=MN` | 90 | 0.427 |
| `strict memory` | `race_ethnicity=other_or_unknown` | 31 | 0.434 |
| `demographic-only` | `state_party_id_3=PA x independent_or_other` | 27 | 0.448 |
| `strict memory` | `state_party_id_3=WI x independent_or_other` | 30 | 0.461 |
| `demographic-only` | `state_party_id_3=AZ x republican` | 29 | 0.466 |

Subgroup risk is concentrated in two areas:

1. `demographic-only` is very poor at recognizing state x party combinations, especially Democratic respondents in AZ/CO/NC/NV/VA.
2. `strict memory` is unstable among independent/other party-id groups, especially CO, NC, and WI.

## 14. Assessment of System Capability

### 14.1 Engineering Pipeline

The engineering pipeline performed well in this run.

1. The hard-choice response contract was stable, with all 3000 LLM calls parsed successfully.
2. Parallel workers=4 ran stably, with throughput around 5.16 responses/s.
3. GPU peak memory was about 5054 MiB, with minimum free memory about 2838 MiB, so there was no OOM risk.
4. Partial checkpoints, runtime logs, cache, and final artifacts were all generated normally.
5. The 1000-agent E01 completed in about 12.2 minutes, well under the 30-minute budget.

This shows that the current system can support larger local LLM evaluations.

### 14.2 Simulator Validity

When the simulator is treated as an individual vote hard-choice classifier, the strongest LLM condition is `party/ideology`:

| Comparison | Conclusion |
| --- | --- |
| party/ideology vs demographic-only | Large improvement, vote acc 0.518 -> 0.917 |
| strict memory vs party/ideology | Decline, vote acc 0.917 -> 0.871 |
| party/ideology vs party_id_baseline | Slightly stronger, 0.917 vs 0.868 |
| party/ideology vs sklearn_pre_only | Clearly weaker, 0.917 vs 0.947 |
| party/ideology vs sklearn_poll_informed | Clearly weaker, 0.917 vs 0.954 |

This indicates that qwen3.5:2b can effectively use explicit party/ideology information, but strict memory does not show a stable benefit of making responses "more like the real respondent."

### 14.3 As a State-Level Election Simulator

As an aggregate election simulator, the LLM remains insufficient.

1. `party/ideology` has winner accuracy of 0.700, but that is because 7 of the 10 state truths are Republican wins; winner accuracy cannot be read in isolation.
2. The more important margin and dem_2p metrics show systematic Republican bias in the LLM.
3. After adding the Democratic-winning states MN/VA/CO, neither `party/ideology` nor `strict memory` predicted a Democratic winner.
4. `sklearn_logit_pre_only` and `sklearn_logit_poll_informed` are much stronger than the LLM at the aggregate level.

Therefore, the current qwen3.5:2b + hard-choice persona prompt is not suitable as a direct state-level election predictor. It is more like an individual classifier that can read party fields than a reliable election simulator.

## 15. Whether This Changes the Previous Conclusions

This run does not overturn the previous conclusions; it makes them clearer:

1. **party/ideology is the most effective information source.**
   This is the most stable finding in this run. After adding party and ideology, LLM individual vote accuracy rises from 0.518 to 0.917.

2. **strict memory does not provide a stable gain.**
   In the small E01, strict memory was close to party/ideology; after 10 states and 1000 agents, strict memory clearly degrades.

3. **The LLM has Republican skew.**
   This is not just because Republicans won seven swing states in 2024. Compared with MIT truth, the LLM systematically underestimates Democratic 2P share, especially in MN/VA/CO.

4. **turnout remains unresolved.**
   The hard-choice contract is an engineering success, but the LLM almost never chooses `not_vote`. Current turnout accuracy is high mainly because voters are overrepresented in the sample, not because the model can genuinely identify nonvoters.

5. **Statistical models remain strong references.**
   `sklearn_logit_pre_only` and `sklearn_logit_poll_informed` are clearly better than the LLM on both individual and aggregate metrics.

## 16. Recommendations

If the system continues to scale or is redesigned, I recommend prioritizing three things:

1. **Split turnout and vote choice into a two-stage task.**
   The current three-class hard-choice setup is reliable as engineering, but the LLM basically does not use `not_vote`. The model can first judge `vote/not_vote`; if vote, then judge `democrat/republican`, with separate evaluations.

2. **Keep party/ideology as the main LLM baseline and use strict memory cautiously.**
   Strict memory currently appears to introduce extra noise or trigger a default Republican tendency. Future work should decompose the memory fields and inspect which fields cause degradation.

3. **Future large aggregate experiments should include Democratic-winning states.**
   Looking only at seven Republican-winning swing states overestimates the winner accuracy of a Republican-skewed model. Contrast states like MN/VA/CO are very valuable.

Overall assessment: E01-large formally passes, the engineering pipeline is reliable, and the results are reusable. But from a scientific evaluation perspective, the current qwen3.5:2b LLM persona simulator still cannot be treated as a reliable state-level election simulator. Its most stable capability is using explicit party/ideology to perform individual vote classification.
