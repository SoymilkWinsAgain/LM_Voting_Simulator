# LM Voting Simulator Evaluation Report

Date: 2026-05-01  
Evaluation root: `data/runs/eval_suite_local`  
Primary model: `qwen3.5:2b` via Windows Ollama at `http://172.26.48.1:11434`  
Git commit recorded by runs: `58275c09cca786fcd811a2c77474d4b4c611eaf7`

## 1. Executive Summary

This evaluation suite tested the simulator as a simulator, not just as a software pipeline. The engineering system is now stable: all formal experiments completed, all data-contract gates passed, all LLM outputs followed the hard-choice JSON response contract, and runtime artifacts are reproducible enough for later analysis.

The scientific result is more mixed. The simulator can reliably produce parsable hard choices and can recover individual vote direction when party/ideology is visible. However, the current `qwen3.5:2b` LLM is not a reliable standalone election simulator. Its strongest useful signal is party/ideology. Strict survey-memory prompts did not improve results over party/ideology prompts, and in several aggregate or ablation settings they made predictions more Republican-skewed. Aggregate winner accuracy is sometimes high because the 2024 swing states were all Republican wins, but margin error is often very large.

The strongest conclusion is:

- The framework is usable for controlled evaluation.
- The hard-choice response contract fixed the previous invalid-probability problem.
- `qwen3.5:2b` should not be treated as a calibrated probabilistic voter simulator.
- In this run, party/ideology prompts were usually more useful than strict memory prompts.
- Memory/context prompts often induced a strong Republican default rather than a nuanced persona simulation.
- Future work should test stronger models, candidate-order balancing, repeated sampling, and redesigned memory use before using the simulator for substantive claims.

## 2. Response Contract and Interpretation

The evaluated system no longer asks the LLM to invent probability vectors. Each CES turnout-vote response is a hard choice:

```json
{"choice": "not_vote|democrat|republican"}
```

The system then derives compatibility columns:

- `turnout_probability = 1` for `democrat` or `republican`; `0` for `not_vote`.
- `vote_prob_democrat = 1` iff `choice == "democrat"`.
- `vote_prob_republican = 1` iff `choice == "republican"`.
- `vote_prob_other = 0`.
- `vote_prob_undecided = 0`.

Important interpretation rule: these columns are one-hot system-derived values. They are not subjective LLM probabilities. Therefore, old probability calibration metrics are either removed, downgraded, or interpreted only as deterministic hard-choice reliability diagnostics.

## 3. Experiment Inventory

Formal outputs:

- E00 Preflight: `data/runs/eval_suite_local/00_preflight`
- E01 Individual Persona Fidelity: `data/runs/eval_suite_local/01_individual_persona`
- E02 Aggregate Election Accuracy: `data/runs/eval_suite_local/02_aggregate_accuracy`
- E03 Prompt Robustness: `data/runs/eval_suite_local/03_prompt_robustness`
- E04 Historical / World-Knowledge Leakage: `data/runs/eval_suite_local/04_leakage`
- E05 Information Ablation and Placebo Memory: `data/runs/eval_suite_local/05_ablation_placebo`
- E06 Subgroup and Calibration Reliability: `data/runs/eval_suite_local/06_subgroup_calibration`
- Suite summary: `data/runs/eval_suite_local/eval_suite_summary.md`

Probe outputs are also preserved under `data/runs/eval_suite_local/e0*_probe_*`, but conclusions below use formal directories only.

## 4. Runtime and Quality Overview

All formal quality gates passed. No formal experiment showed invalid choices, forbidden `other/undecided` choices, legacy probability schemas, or transport errors.

| experiment | workers | llm_tasks | runtime_min | median_s | p90_s | throughput_s | gpu_peak_mib | gpu_free_min_mib | gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E00 Preflight | 3 | 12 | 0.116 | 0.812 | 0.913 | 1.725 | 4560 | 3332 | True |
| E01 Individual | 3 | 300 | 2.925 | 0.545 | 0.700 | 4.791 | 4561 | 3331 | True |
| E02 Aggregate | 4 | 490 | 4.788 | 0.979 | 1.025 | 3.819 | 4561 | 3331 | True |
| E03 Robustness | 4 | 300 | 1.250 | 0.950 | 1.007 | 4.140 | 4563 | 3329 | True |
| E04 Leakage | 3 | 420 | 2.371 | 0.847 | 0.896 | 3.409 | 4839 | 3053 | True |
| E05 Ablation | 3 | 420 | 2.765 | 0.755 | 0.783 | 4.072 | 4902 | 2990 | True |
| E06 Subgroup | 4 | 0 | 0.283 |  |  |  |  |  | True |

Formal E01-E06 wall-clock time was about 14.38 minutes. E06 did not call Ollama.

GPU memory was stable. Peak observed memory use was about 4.9 GiB, with at least about 3.0 GiB free in formal LLM runs. There was no OOM signal.

## 5. E00 Preflight and Data Contract Validation

E00 was the readiness gate before substantive experiments. It verified:

- Windows Ollama connectivity.
- `qwen3.5:2b` response quality under the hard-choice contract.
- strict-memory and poll-informed leakage rules.
- basic runtime logging.

Quality gates:

| gate_name | threshold | observed | passed |
| --- | --- | --- | --- |
| parse_ok_rate | 0.950 | 1.000 | True |
| invalid_choice_rate | 0.020 | 0.000 | True |
| forbidden_choice_rate | 0.000 | 0.000 | True |
| legacy_probability_schema_rate | 0.000 | 0.000 | True |

Leakage checks:

| check_name | passed | details |
| --- | --- | --- |
| strict_blocks_post_direct_and_targetsmart | True |  |
| poll_informed_has_poll_prior_role | True | `CC24_363,CC24_364a` |
| poll_informed_blocks_post_and_targetsmart | True |  |

Conclusion: the system was safe to evaluate. E00 should remain mandatory before any larger rerun.

## 6. E01 Individual Persona Fidelity

Report: `data/runs/eval_suite_local/01_individual_persona/benchmark_report.md`

### 6.1 Goal

E01 asks whether the simulator can predict each CES respondent's own turnout/vote hard choice. Formal setup:

- States: PA, GA, AZ, WI
- LLM sample: 100 agents, 25 per state
- LLM baselines:
  - `ces_demographic_only_llm`
  - `ces_party_ideology_llm`
  - `ces_survey_memory_llm_strict`
- Non-LLM references:
  - `party_id_baseline`
  - `sklearn_logit_demographic_only`
  - `sklearn_logit_pre_only`
  - `sklearn_logit_poll_informed`

### 6.2 Key Individual Metrics

Weighted metrics:

| baseline | model_name | turnout_accuracy_at_0_5 | turnout_brier | vote_accuracy | vote_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| `ces_demographic_only_llm` | `qwen3.5:2b` | 1.000 | 0.000 | 0.577 | 0.369 |
| `ces_party_ideology_llm` | `qwen3.5:2b` | 1.000 | 0.000 | 0.977 | 0.651 |
| `ces_survey_memory_llm_strict` | `qwen3.5:2b` | 1.000 | 0.000 | 0.977 | 0.651 |
| `party_id_baseline` | `party_id_baseline_v1` | 0.944 | 0.056 | 0.884 | 0.611 |
| `sklearn_logit_demographic_only` | `sklearn_logit_demographic_only_v1` | 0.976 | 0.024 | 0.627 | 0.415 |
| `sklearn_logit_poll_informed` | `sklearn_logit_poll_informed_v1` | 0.967 | 0.033 | 0.965 | 0.669 |
| `sklearn_logit_pre_only` | `sklearn_logit_pre_only_v1` | 0.976 | 0.024 | 0.977 | 0.651 |

Choice distributions:

| baseline | n | democrat | republican | not_vote |
| --- | --- | --- | --- | --- |
| `ces_demographic_only_llm` | 100 | 0.390 | 0.610 | 0.000 |
| `ces_party_ideology_llm` | 100 | 0.420 | 0.580 | 0.000 |
| `ces_survey_memory_llm_strict` | 100 | 0.400 | 0.590 | 0.010 |
| `party_id_baseline` | 1356 | 0.580 | 0.374 | 0.046 |
| `sklearn_logit_demographic_only` | 1356 | 0.458 | 0.542 | 0.000 |
| `sklearn_logit_poll_informed` | 1356 | 0.515 | 0.442 | 0.044 |
| `sklearn_logit_pre_only` | 1356 | 0.549 | 0.451 | 0.000 |

### 6.3 Interpretation

Demographics alone are weak. Adding party/ideology causes a large jump from `0.577` to `0.977` weighted vote accuracy. Strict survey memory does not improve over party/ideology and is essentially tied with it.

This is an important scientific result: the model is not obviously using rich memory in a way that improves individual simulation. The usable signal comes mostly from party/ideology, which a simple statistical model also captures.

Turnout metrics are artificially high for LLM baselines because most sampled respondents are validated/self-reported voters and the hard-choice outputs rarely choose `not_vote`. This should not be read as a strong turnout model.

## 7. E02 Aggregate Election Accuracy

Report: `data/runs/eval_suite_local/02_aggregate_accuracy/benchmark_report.md`

### 7.1 Goal

E02 tests whether state-level aggregation reproduces 2024 two-party Democratic share, margin, and winner.

Formal LLM setup:

- States: PA, MI, WI, GA, AZ, NV, NC
- LLM baseline: `survey_memory_llm_strict`
- LLM sample: 70 agents per state, 490 total LLM calls
- Non-LLM baselines ran at sample sizes 50, 70, 100, 500, 1000, 2000

### 7.2 Main Aggregate Metrics at Sample Size 70

| baseline | model_name | dem_2p_rmse | margin_bias | margin_mae | winner_accuracy | winner_flip_count |
| --- | --- | --- | --- | --- | --- | --- |
| `ces_post_self_report_aggregate_oracle` | `ces_post_self_report_oracle_v1` | 0.039 | -0.024 | 0.064 | 0.857 | 1 |
| `mit_2020_state_prior` | `mit_2020_state_prior_v1` | 0.019 | 0.035 | 0.035 | 0.143 | 6 |
| `party_id_baseline` | `party_id_baseline_v1` | 0.094 | 0.147 | 0.150 | 0.286 | 5 |
| `sklearn_logit_poll_informed` | `sklearn_logit_poll_informed_v1` | 0.052 | -0.069 | 0.088 | 0.857 | 1 |
| `sklearn_logit_pre_only_crossfit` | `sklearn_logit_pre_only_crossfit_v1` | 0.055 | -0.092 | 0.092 | 1.000 | 0 |
| `survey_memory_llm_strict` | `qwen3.5:2b` | 0.180 | -0.351 | 0.351 | 1.000 | 0 |
| `uniform_national_swing_from_2020` | `realized_2024_national_swing_v1` | 0.015 | -0.025 | 0.025 | 1.000 | 0 |

### 7.3 LLM State-Level Errors at Sample Size 70

| state | pred_dem_2p | true_dem_2p | pred_margin | true_margin | margin_error | winner_correct |
| --- | --- | --- | --- | --- | --- | --- |
| AZ | 0.277 | 0.472 | -0.445 | -0.056 | -0.390 | True |
| GA | 0.366 | 0.489 | -0.267 | -0.022 | -0.245 | True |
| MI | 0.284 | 0.493 | -0.432 | -0.014 | -0.417 | True |
| NC | 0.270 | 0.484 | -0.460 | -0.033 | -0.427 | True |
| NV | 0.299 | 0.484 | -0.403 | -0.032 | -0.371 | True |
| PA | 0.390 | 0.491 | -0.221 | -0.017 | -0.203 | True |
| WI | 0.293 | 0.496 | -0.413 | -0.009 | -0.404 | True |

### 7.4 Interpretation

The LLM gets all seven winners correct because all seven states were true Republican wins in 2024. But it predicts much larger Republican margins than reality. Its `margin_mae = 0.351` is far worse than `sklearn_logit_pre_only_crossfit = 0.092`, `sklearn_logit_poll_informed = 0.088`, and `uniform_national_swing_from_2020 = 0.025`.

The key lesson is that winner accuracy is not enough in this setting. When all seven target states have the same winner, an over-Republican model can look good on winner accuracy while being poor on margin and vote share.

## 8. E03 Prompt Robustness

Report: `data/runs/eval_suite_local/03_prompt_robustness/report.md`

### 8.1 Goal

E03 tests whether the hard-choice LLM output is stable under prompt variants.

Formal setup:

- States: PA, GA, AZ, WI
- Agents: 60
- Prompt variants:
  - `base_json`
  - `json_strict_nonzero`
  - `candidate_order_reversed`
  - `interviewer_style`
  - `analyst_style`
- Total LLM calls: 300

### 8.2 Metrics by Prompt Variant

| prompt_variant | median_latency_seconds | parse_ok_rate | turnout_brier | vote_accuracy |
| --- | --- | --- | --- | --- |
| `analyst_style` | 0.931 | 1.000 | 0.012 | 0.926 |
| `base_json` | 0.967 | 1.000 | 0.012 | 0.926 |
| `candidate_order_reversed` | 0.936 | 1.000 | 0.012 | 0.865 |
| `interviewer_style` | 0.944 | 1.000 | 0.012 | 0.926 |
| `json_strict_nonzero` | 0.939 | 1.000 | 0.012 | 0.926 |

Pairwise shifts versus `base_json`:

| prompt_variant | metric_name | metric_value | n |
| --- | --- | --- | --- |
| `json_strict_nonzero` | choice_flip_rate | 0.000 | 60 |
| `json_strict_nonzero` | turnout_choice_flip_rate | 0.000 | 60 |
| `candidate_order_reversed` | choice_flip_rate | 0.067 | 60 |
| `candidate_order_reversed` | turnout_choice_flip_rate | 0.000 | 60 |
| `interviewer_style` | choice_flip_rate | 0.000 | 60 |
| `interviewer_style` | turnout_choice_flip_rate | 0.000 | 60 |
| `analyst_style` | choice_flip_rate | 0.000 | 60 |
| `analyst_style` | turnout_choice_flip_rate | 0.000 | 60 |

Choice distribution:

| prompt_variant | n | democrat | republican | not_vote |
| --- | --- | --- | --- | --- |
| `analyst_style` | 60 | 0.367 | 0.633 | 0.000 |
| `base_json` | 60 | 0.367 | 0.633 | 0.000 |
| `candidate_order_reversed` | 60 | 0.300 | 0.700 | 0.000 |
| `interviewer_style` | 60 | 0.367 | 0.633 | 0.000 |
| `json_strict_nonzero` | 60 | 0.367 | 0.633 | 0.000 |

### 8.3 Interpretation

The hard-choice schema is robust: parse success is perfect across all prompt variants. Most stylistic changes have no effect.

The candidate-order reversal is the exception. It causes a `6.7%` choice flip rate and reduces vote accuracy from `0.926` to `0.865`. It also shifts the state margin in PA substantially in the pairwise diagnostics.

This means future aggregate experiments should either fix and document candidate order or explicitly randomize/balance candidate order and average over variants.

## 9. E04 Historical / World-Knowledge Leakage

Report: `data/runs/eval_suite_local/04_leakage/benchmark_report.md`

### 9.1 Goal

E04 tests whether the LLM relies on candidate names, party labels, state identity, year knowledge, or other world-knowledge shortcuts rather than persona information.

Formal setup:

- States: PA, MN, GA, VA, AZ, CO
- Agents: 10 per state, 60 total
- Conditions:
  - `named_candidates`
  - `party_only_candidates`
  - `anonymous_candidates`
  - `masked_year`
  - `masked_state`
  - `state_swap_placebo`
  - `candidate_swap_placebo`
- Total LLM calls: 420

### 9.2 Individual Metrics

| condition | turnout_accuracy_at_0_5 | turnout_brier | vote_accuracy | vote_macro_f1 |
| --- | --- | --- | --- | --- |
| `anonymous_candidates` | 1.000 | 0.000 | 0.980 | 0.653 |
| `candidate_swap_placebo` | 1.000 | 0.000 | 0.511 | 0.226 |
| `masked_state` | 1.000 | 0.000 | 0.531 | 0.254 |
| `masked_year` | 1.000 | 0.000 | 0.511 | 0.226 |
| `named_candidates` | 1.000 | 0.000 | 0.536 | 0.261 |
| `party_only_candidates` | 1.000 | 0.000 | 0.511 | 0.226 |
| `state_swap_placebo` | 1.000 | 0.000 | 0.615 | 0.358 |

### 9.3 Aggregate Metrics

| condition | dem_2p_rmse | margin_bias | margin_mae | winner_accuracy | winner_flip_count |
| --- | --- | --- | --- | --- | --- |
| `anonymous_candidates` | 0.210 | -0.012 | 0.411 | 0.333 | 4 |
| `candidate_swap_placebo` | 0.511 | -1.020 | 1.020 | 0.500 | 3 |
| `masked_state` | 0.490 | -0.976 | 0.976 | 0.500 | 3 |
| `masked_year` | 0.511 | -1.020 | 1.020 | 0.500 | 3 |
| `named_candidates` | 0.485 | -0.967 | 0.967 | 0.500 | 3 |
| `party_only_candidates` | 0.511 | -1.020 | 1.020 | 0.500 | 3 |
| `state_swap_placebo` | 0.448 | -0.777 | 0.848 | 0.333 | 4 |

Choice distribution:

| condition | n | democrat | republican | not_vote |
| --- | --- | --- | --- | --- |
| `anonymous_candidates` | 60 | 0.500 | 0.500 | 0.000 |
| `candidate_swap_placebo` | 60 | 0.000 | 1.000 | 0.000 |
| `masked_state` | 60 | 0.033 | 0.967 | 0.000 |
| `masked_year` | 60 | 0.000 | 1.000 | 0.000 |
| `named_candidates` | 60 | 0.050 | 0.950 | 0.000 |
| `party_only_candidates` | 60 | 0.000 | 1.000 | 0.000 |
| `state_swap_placebo` | 60 | 0.083 | 0.917 | 0.000 |

### 9.4 Leakage Diagnostics

Important diagnostics:

- `anonymous_candidates` is much less Republican-skewed than named or party-only conditions.
- `party_only_candidates`, `masked_year`, and `candidate_swap_placebo` collapse to all Republican.
- `named_candidates` is 95% Republican.
- Candidate swap diagnostics show strong party/label following rather than name following:
  - `candidate_name_following_index` ranges from about `-1.000` to `-0.846`.
  - Mean name-minus-party following is about `-0.947`.

State-swap diagnostics are noisy. One large shift appears for PA displayed as MN, but most swaps barely track true state-margin shifts.

### 9.5 Interpretation

This is one of the most important experiments. It shows that `qwen3.5:2b` often defaults to Republican under candidate/party/year conditions, even in states where the true 2024 winner was Democratic. The problem is not merely that the model predicts Republicans; the problem is that it predicts extreme Republican margins and loses state-level resolution.

Anonymous candidates produce a more balanced 50/50 distribution and better aggregate margin MAE, but low winner accuracy. Named and party-only candidates produce higher Republican collapse.

The evidence points to prompt-conditioned shortcut behavior, not robust persona simulation.

## 10. E05 Information Ablation and Placebo Memory

Report: `data/runs/eval_suite_local/05_ablation_placebo/benchmark_report.md`

### 10.1 Goal

E05 asks which information sources actually improve prediction and whether real memory beats placebo memory.

Formal setup:

- States: PA, GA, AZ, WI
- Agents: 60 total
  - 40 main agents
  - 20 diagnostic boost agents
- Conditions:
  - `L1_demographic_only_llm`
  - `L3_party_ideology_llm`
  - `L6_strict_memory_context_llm`
  - `L7_poll_informed_memory_context_llm`
  - `L8_post_hoc_oracle_memory_context_llm`
  - `P1_memory_shuffled_within_state_llm`
  - `P2_memory_shuffled_within_party_llm`
- Total LLM calls: 420

### 10.2 Main-Sample Individual Metrics

| condition | turnout_accuracy_at_0_5 | turnout_brier | vote_accuracy | vote_macro_f1 |
| --- | --- | --- | --- | --- |
| `L1_demographic_only_llm` | 0.893 | 0.107 | 0.471 | 0.213 |
| `L3_party_ideology_llm` | 0.785 | 0.215 | 0.820 | 0.581 |
| `L6_strict_memory_context_llm` | 0.893 | 0.107 | 0.594 | 0.359 |
| `L7_poll_informed_memory_context_llm` | 0.893 | 0.107 | 0.594 | 0.359 |
| `L8_post_hoc_oracle_memory_context_llm` | 1.000 | 0.000 | 1.000 | 0.667 |
| `P1_memory_shuffled_within_state_llm` | 0.893 | 0.107 | 0.471 | 0.213 |
| `P2_memory_shuffled_within_party_llm` | 0.893 | 0.107 | 0.488 | 0.237 |

### 10.3 Aggregate Metrics

| condition | dem_2p_rmse | margin_bias | margin_mae | winner_accuracy | winner_flip_count |
| --- | --- | --- | --- | --- | --- |
| `L1_demographic_only_llm` | 0.487 | -0.974 | 0.974 | 1.000 | 0 |
| `L3_party_ideology_llm` | 0.189 | -0.300 | 0.300 | 1.000 | 0 |
| `L6_strict_memory_context_llm` | 0.367 | -0.661 | 0.661 | 1.000 | 0 |
| `L7_poll_informed_memory_context_llm` | 0.440 | -0.862 | 0.862 | 1.000 | 0 |
| `L8_post_hoc_oracle_memory_context_llm` | 0.213 | -0.087 | 0.348 | 0.500 | 2 |
| `P1_memory_shuffled_within_state_llm` | 0.487 | -0.974 | 0.974 | 1.000 | 0 |
| `P2_memory_shuffled_within_party_llm` | 0.473 | -0.946 | 0.946 | 1.000 | 0 |

Choice distribution:

| condition | n | democrat | republican | not_vote |
| --- | --- | --- | --- | --- |
| `L1_demographic_only_llm` | 60 | 0.000 | 1.000 | 0.000 |
| `L3_party_ideology_llm` | 60 | 0.333 | 0.633 | 0.033 |
| `L6_strict_memory_context_llm` | 60 | 0.067 | 0.933 | 0.000 |
| `L7_poll_informed_memory_context_llm` | 60 | 0.050 | 0.950 | 0.000 |
| `L8_post_hoc_oracle_memory_context_llm` | 60 | 0.367 | 0.550 | 0.083 |
| `P1_memory_shuffled_within_state_llm` | 60 | 0.000 | 1.000 | 0.000 |
| `P2_memory_shuffled_within_party_llm` | 60 | 0.033 | 0.967 | 0.000 |

### 10.4 Ablation Deltas

Important deltas:

- `L6` over `P1`: vote accuracy +0.123, macro F1 +0.145.
- `L6` over `P2`: vote accuracy +0.106, macro F1 +0.122.
- `L7` over `L6`: no improvement on main individual metrics.
- `L8` over `L7`: vote accuracy +0.406, macro F1 +0.308.

### 10.5 Interpretation

Party/ideology (`L3`) is the best non-oracle LLM condition in E05. It beats strict memory (`L6`) and poll-informed memory (`L7`) by a large margin.

Strict memory has some signal because it beats shuffled memory placebos, but it does not beat the simpler party/ideology prompt. Poll-informed memory does not help and appears to worsen aggregate margin bias.

The oracle condition proves the pipeline can encode target-relevant information when explicitly given, but its aggregate winner accuracy is hurt by small-sample composition. This is a useful reminder that individual accuracy and aggregate accuracy are related but not identical.

## 11. E06 Subgroup and Calibration Reliability

Report: `data/runs/eval_suite_local/06_subgroup_calibration/benchmark_report.md`

### 11.1 Goal

E06 reuses E01 and E05 outputs to inspect subgroup reliability, distribution collapse, and deterministic hard-choice calibration.

It did not call Ollama.

### 11.2 Overall Reliability Rows

| source | baseline | n | vote_accuracy | vote_macro_f1 | turnout_accuracy | turnout_brier | mean_predicted_dem_2p | mean_true_dem_2p | dem_2p_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E01 | `ces_party_ideology_llm` | 100 | 0.977 | 0.651 | 1.000 | 0.000 | 0.402 | 0.432 | -0.030 |
| E01 | `ces_demographic_only_llm` | 100 | 0.577 | 0.369 | 1.000 | 0.000 | 0.327 | 0.432 | -0.104 |
| E01 | `ces_survey_memory_llm_strict` | 100 | 0.977 | 0.651 | 1.000 | 0.000 | 0.401 | 0.432 | -0.031 |
| E05 | `L1_demographic_only_llm` | 60 | 0.554 | 0.238 | 0.890 | 0.110 | 0.000 | 0.446 | -0.446 |
| E05 | `L3_party_ideology_llm` | 60 | 0.867 | 0.598 | 0.813 | 0.187 | 0.353 | 0.446 | -0.093 |
| E05 | `L6_strict_memory_context_llm` | 60 | 0.641 | 0.361 | 0.890 | 0.110 | 0.124 | 0.446 | -0.322 |
| E05 | `L7_poll_informed_memory_context_llm` | 60 | 0.641 | 0.361 | 0.890 | 0.110 | 0.063 | 0.446 | -0.383 |
| E05 | `L8_post_hoc_oracle_memory_context_llm` | 60 | 0.995 | 0.663 | 1.000 | 0.000 | 0.399 | 0.446 | -0.047 |
| E05 | `P1_memory_shuffled_within_state_llm` | 60 | 0.554 | 0.238 | 0.890 | 0.110 | 0.000 | 0.446 | -0.446 |
| E05 | `P2_memory_shuffled_within_party_llm` | 60 | 0.566 | 0.257 | 0.890 | 0.110 | 0.009 | 0.446 | -0.437 |

### 11.3 Worst Subgroup Patterns

Worst subgroup examples from `worst_subgroups.parquet`:

- E01 `ces_demographic_only_llm`, `ideology_3=liberal`: vote accuracy `0.289`, Democratic two-party error `-0.718`.
- E01 `ces_demographic_only_llm`, `party_id_3=democrat`: vote accuracy `0.437`, Democratic two-party error `-0.569`.
- E05 `L1_demographic_only_llm`, `race_ethnicity=white`: vote accuracy about `0.509`, Democratic two-party error about `-0.491`.
- E05 `P1_memory_shuffled_within_state_llm`, `race_ethnicity=white`: similar to L1, indicating placebo memory does not fix collapse.

Distribution diagnostics show severe collapse in weak-information E05 conditions:

- `L1` and `P1` predict 100% Republican.
- `P2` predicts 96.7% Republican.
- `L6` predicts 93.3% Republican.
- `L7` predicts 95.0% Republican.
- `L3` is much less collapsed: 33.3% Democrat, 63.3% Republican, 3.3% not vote.

### 11.4 Interpretation

Subgroup failures are not random. They concentrate where the prompt lacks party/ideology or where memory/context causes over-Republican predictions. Hard-choice calibration is deterministic and should not be overinterpreted as probabilistic calibration.

E06 reinforces the E05 conclusion: the useful information source is party/ideology, not the current memory prompt design.

## 12. Cross-Experiment Synthesis

### 12.1 Engineering Reliability

The current system is robust as an experiment runner:

- All formal runs completed.
- All formal quality gates passed.
- Runtime logs, raw responses, caches, partial checkpoints, and reports are preserved.
- Windows Ollama through WSL worked reliably.
- Parallelism of 3-4 workers gave stable throughput and safe GPU memory usage.

This is a strong result for the framework itself.

### 12.2 Scientific Validity of the Current LLM Simulator

The current `qwen3.5:2b` simulator is not scientifically strong enough for final election simulation claims.

Positive findings:

- It follows hard-choice JSON well.
- It can use party/ideology to recover individual vote direction.
- It can be evaluated cleanly using this suite.

Negative findings:

- It is not calibrated.
- It rarely models non-voting in these samples.
- It is sensitive to candidate order.
- It tends to overpredict Republican choices in aggregate and leakage/ablation settings.
- Strict memory does not reliably improve individual or aggregate metrics.
- Poll-informed memory does not improve results in E05.
- Winner accuracy can be misleading because all seven E02 swing states were true Republican wins.

### 12.3 Party / Ideology Dominates Memory

E01 and E05 both show that party/ideology is the most useful prompt information.

In E01:

- demographic-only LLM: vote accuracy `0.577`
- party/ideology LLM: vote accuracy `0.977`
- strict-memory LLM: vote accuracy `0.977`

In E05:

- demographic-only LLM: vote accuracy `0.471`
- party/ideology LLM: vote accuracy `0.820`
- strict-memory/context LLM: vote accuracy `0.594`

Strict memory sometimes contains real signal, but in this model/prompt design it is not being used reliably.

### 12.4 Aggregate Winner Accuracy Is Not Enough

E02 demonstrates that `survey_memory_llm_strict` gets all seven swing-state winners right while having very large margin errors:

- LLM margin MAE: `0.351`
- `sklearn_logit_pre_only_crossfit` margin MAE: `0.092`
- `uniform_national_swing_from_2020` margin MAE: `0.025`

Because every E02 target state was a Republican win in 2024, a Republican-skewed simulator can score high on winner accuracy. The report should always emphasize margin MAE, dem_2p RMSE, and bias alongside winner accuracy.

### 12.5 The Republican Collapse Pattern

Several experiments show an extreme Republican default:

- E04 named candidates: 95% Republican.
- E04 party-only: 100% Republican.
- E04 masked-year: 100% Republican.
- E05 L1 demographic-only: 100% Republican.
- E05 L6 strict memory/context: 93.3% Republican.
- E05 L7 poll-informed memory/context: 95.0% Republican.

This is not equivalent to correctly recognizing that Trump won all seven 2024 swing states. In E04, the sampled states include Democratic true-winner states such as CO, MN, and VA, and the model still often predicts Republican. The issue is loss of state/persona resolution.

## 13. Limitations

Important limitations of this evaluation:

- The formal LLM sample sizes were intentionally small to keep runtime under control.
- E02 LLM aggregate results use 70 agents per state; larger LLM samples may reduce sampling noise but will not necessarily fix systematic bias.
- The hard-choice contract removes invalid probability issues but also removes direct probabilistic uncertainty.
- Turnout modeling is weakly tested because many sampled respondents are voters and LLM outputs rarely choose `not_vote`.
- `qwen3.5:2b` is a small local model; results should not be generalized to stronger models without rerunning the suite.
- E05 oracle aggregate behavior shows that small state samples can distort aggregate results even when individual labels are nearly perfect.

## 14. Recommendations

Recommended next steps before making substantive claims:

1. Treat this run as a framework validation plus baseline model diagnostic, not as a final election-simulation result.
2. Run the same suite on a stronger local or API model.
3. Add candidate-order balancing to aggregate experiments.
4. Consider repeated stochastic hard-choice sampling per agent, then aggregate frequencies, instead of one deterministic response per agent.
5. Redesign memory prompts so memory does not swamp or distort party/ideology signal.
6. Keep party/ideology statistical baselines as mandatory references.
7. Report margin error and bias before winner accuracy.
8. Expand turnout-specific evaluation with more non-voter-rich samples.
9. For final publication-style reporting, include uncertainty intervals from repeated samples or bootstrap over agents.

## 15. Final Verdict

The evaluation suite succeeded. The simulator infrastructure is now measurable, reproducible, and much less fragile than before the hard-choice contract change.

The evaluated `qwen3.5:2b` simulator does not yet meet the standard for a credible voter/election simulator. It is best viewed as an experimental LLM behavior component inside a broader evaluation framework. The strongest practical baseline from this run is party/ideology-based modeling; the current strict-memory LLM path is not reliably superior and often worsens aggregate realism.

The next meaningful milestone is not just scaling sample size. It is improving the model/prompt design and rerunning this suite to see whether memory can add real predictive value without inducing aggregate collapse.
