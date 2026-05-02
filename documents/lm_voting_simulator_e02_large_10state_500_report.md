# E02-Large 10-State / 500-Agent Aggregate Election Accuracy Report

Generated at: 2026-05-01
Experiment directory: `data/runs/eval_suite_local/02_aggregate_accuracy_large_10state_500`
Config file: `configs/eval_suite/e02_aggregate_accuracy_large_10state_500.yaml`
Old E02 directory retained: `data/runs/eval_suite_local/02_aggregate_accuracy`

## 1. Experiment Goal

This run is the expanded version of E02 Aggregate Election Accuracy. Its goal is to evaluate whether the `qwen3.5:2b` strict-memory LLM can aggregate agent hard-choice outputs into results close to 2024 state-level presidential election truth across a larger state set and larger per-state samples.

This run focuses on three questions:

1. Can `survey_memory_llm_strict` recover state-level Democratic two-party share, margin, and winner?
2. Is LLM strict better than simple non-LLM baselines such as `party_id_baseline`, `mit_2020_state_prior`, and `uniform_national_swing_from_2020`?
3. After adding the three 2024 Democratic-winning states MN/VA/CO, does the run expose or mitigate the LLM's Republican skew?

This run continues to use the three-class hard-choice contract:

```json
{"choice": "not_vote|democrat|republican"}
```

The old probability columns should only be interpreted as one-hot or aggregate frequencies derived from the hard choice, not as LLM subjective probabilities.

## 2. Design

### 2.1 States

This run uses 10 states:

| Group | States | Description |
| --- | --- | --- |
| Original 7 swing states | PA, MI, WI, GA, AZ, NV, NC | All were 2024 Republican winners |
| Added 3 Democratic-winning states | MN, VA, CO | Used to break the interpretive bias that all target states are R wins |

2024 MIT state-level truth:

| State | True Dem 2P | True margin | True winner |
| --- | ---: | ---: | --- |
| PA | 0.491 | -0.017 | republican |
| MI | 0.493 | -0.014 | republican |
| WI | 0.496 | -0.009 | republican |
| GA | 0.489 | -0.022 | republican |
| AZ | 0.472 | -0.056 | republican |
| NV | 0.484 | -0.032 | republican |
| NC | 0.484 | -0.033 | republican |
| MN | 0.522 | 0.043 | democrat |
| VA | 0.530 | 0.059 | democrat |
| CO | 0.556 | 0.113 | democrat |

### 2.2 Sample Sizes

Configured sample sizes:

```yaml
sample_sizes: [50, 100, 200, 300, 500, 1000, 2000]
```

LLM runs only up to 500 per state:

```yaml
llm.max_sample_size: 500
```

Therefore:

- LLM `survey_memory_llm_strict` has five result levels: 50/100/200/300/500.
- Non-LLM baselines have seven result levels: 50/100/200/300/500/1000/2000.

### 2.3 Baselines

LLM baseline:

| Baseline | Description |
| --- | --- |
| `survey_memory_llm_strict` | strict pre-election survey memory, `qwen3.5:2b` |

Non-LLM baselines:

| Baseline | Description |
| --- | --- |
| `mit_2020_state_prior` | 2020 state-level prior |
| `uniform_national_swing_from_2020` | 2020 + national swing baseline |
| `party_id_baseline` | Party-rule baseline |
| `sklearn_logit_pre_only_crossfit` | Pre-election crossfit logit |
| `sklearn_logit_poll_informed` | Poll-informed logit upper-bound reference |
| `ces_post_self_report_aggregate_oracle` | Post-self-report oracle, used only as an upper bound |

This run does not include a poll-informed LLM, to avoid doubling LLM calls and to keep the main LLM condition leakage-free.

## 3. Prechecks and Probes

Precheck results:

| Check | Result |
| --- | --- |
| `py_compile` | Passed |
| E02/parser/cache smoke tests | Passed |
| Ollama `/api/tags` | `qwen3.5:2b` visible |
| processed CES/MIT artifacts | All present |
| Initial GPU | RTX 5070 Laptop GPU, 8151 MiB total, about 6786 MiB free |

Probe results:

| Probe | States | Calls | Workers | LLM runtime | Median | P90 | Throughput | GPU peak used | Min free | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | PA, GA, MN, VA, CO | 25 | 1 | 13.0s | 0.409s | 0.449s | 1.92/s | 4963 MiB | 2929 MiB | PASS |
| B | 10 states | 100 | 4 | 26.2s | 0.880s | 0.913s | 3.82/s | 4963 MiB | 2929 MiB | PASS |
| C | 10 states | 100 | 5 | 29.4s | 1.305s | 1.342s | 3.41/s | 4963 MiB | 2929 MiB | PASS |

Workers=5 had no errors, but was slower than workers=4. Therefore the formal run used workers=4.

The early 10-state choice distributions in Probe B/C were consistent:

| State | Democrat | Not vote | Republican |
| --- | ---: | ---: | ---: |
| AZ | 4 | 0 | 6 |
| CO | 1 | 0 | 9 |
| GA | 2 | 0 | 8 |
| MI | 3 | 1 | 6 |
| MN | 5 | 0 | 5 |
| NC | 2 | 0 | 8 |
| NV | 4 | 0 | 6 |
| PA | 2 | 0 | 8 |
| VA | 4 | 0 | 6 |
| WI | 4 | 0 | 6 |

CO was Republican-skewed in the probe, but not all Republican; MN/VA also did not trigger the stop-loss rule that all added Democratic-winning states were predicted Republican.

## 4. Formal Run Overview

The formal run completed without interruption.

| Metric | Value |
| --- | ---: |
| Workers | 4 |
| States | 10 |
| Effective LLM sample size | 500 per state |
| LLM tasks | 5000 |
| Ollama calls | 4988 |
| Cache hit rate | 0.002 |
| Total wall-clock | 1381.5s, about 23.0 min |
| LLM runtime | 1213.1s, about 20.2 min |
| Median latency | 0.897s |
| P90 latency | 0.967s |
| Throughput | 4.12 responses/s |
| GPU peak used | 4998 MiB |
| GPU min free | 2894 MiB |
| GPU peak utilization | 83% |

The formal run stayed within the 30-minute budget. GPU utilization was high and did not approach OOM.

## 5. Output Artifacts

Formal directory:

`data/runs/eval_suite_local/02_aggregate_accuracy_large_10state_500`

Core outputs:

| File | Shape / Description |
| --- | --- |
| `sampled_agents.parquet` | 12805 x 50 |
| `sample_membership.parquet` | 33632 x 7 |
| `responses.parquet` | 56220 x 38 |
| `prompts.parquet` | 5000 x 12 |
| `state_predictions.parquet` | 470 x 21 |
| `aggregate_metrics.parquet` | 329 x 8 |
| `parse_diagnostics.parquet` | 5 x 8 |
| `runtime_log.parquet` | 5023 x 18 |
| `runtime.json` | Runtime observations |
| `llm_cache.jsonl` | LLM cache |
| `benchmark_report.md` | Runner-generated report |
| `figures/*` | Figures |

Figures include:

| Figure |
| --- |
| `baseline_metric_heatmap_by_sample_size.png` |
| `predicted_vs_true_margin_scatter.png` |
| `sample_size_sensitivity.png` |
| `state_margin_error_lollipop_common.png` |
| `state_margin_error_lollipop_non_llm_max.png` |
| `winner_flip_tile_chart.png` |
| `parse_fallback_diagnostics.png` |
| `a0_a1_delta_chart.png` |

## 6. Quality Gates

Parse diagnostics:

| Baseline | N | Parse OK | Fallback rate | Cache hit |
| --- | ---: | ---: | ---: | ---: |
| `ces_post_self_report_aggregate_oracle` | 12805 | 1.000 | 0.000 | NA |
| `party_id_baseline` | 12805 | 1.000 | 0.000 | NA |
| `sklearn_logit_poll_informed` | 12805 | 1.000 | 0.000 | NA |
| `sklearn_logit_pre_only_crossfit` | 12805 | 1.000 | 0.000 | NA |
| `survey_memory_llm_strict` | 5000 | 1.000 | 0.000 | 0.002 |

Runtime gates:

| Gate | Value | Result |
| --- | ---: | --- |
| parse_ok_rate | 1.000 | PASS |
| invalid_choice_rate | 0.000 | PASS |
| forbidden_choice_rate | 0.000 | PASS |
| legacy_probability_schema_rate | 0.000 | PASS |
| transport_error_rate | 0.000 | PASS |
| fallback_rate | 0.000 | PASS |
| all_gates_passed | true | PASS |

Engineering conclusion: the hard-choice LLM pipeline was stable in this run. There were no JSON/schema problems and no transport errors.

## 7. LLM Choice Distribution

Overall LLM hard choices:

| Choice | N | Share |
| --- | ---: | ---: |
| republican | 2995 | 0.599 |
| democrat | 1999 | 0.400 |
| not_vote | 6 | 0.001 |

By state:

| State | N | Democrat | Republican | Not vote |
| --- | ---: | ---: | ---: | ---: |
| PA | 500 | 0.392 | 0.608 | 0.000 |
| MI | 500 | 0.422 | 0.576 | 0.002 |
| WI | 500 | 0.380 | 0.614 | 0.006 |
| GA | 500 | 0.374 | 0.624 | 0.002 |
| AZ | 500 | 0.334 | 0.666 | 0.000 |
| NV | 500 | 0.446 | 0.554 | 0.000 |
| NC | 500 | 0.352 | 0.648 | 0.000 |
| MN | 500 | 0.500 | 0.498 | 0.002 |
| VA | 500 | 0.384 | 0.616 | 0.000 |
| CO | 500 | 0.414 | 0.586 | 0.000 |

Observations:

1. The LLM almost never outputs `not_vote`, consistent with E01-large.
2. The LLM is overall Republican-skewed, with R share about 59.9%.
3. Among the added Democratic-winning states, MN is almost 50/50, but VA/CO remain clearly Republican-skewed.
4. Note: this raw choice distribution is not the final weighted aggregate share; the final state-level `pred_dem_2p` uses sample weights and turnout-aware aggregation.

## 8. Aggregate Metrics at Sample Size 500

The following are the 10-state overall metrics at common sample size 500, which is the fairest horizontal comparison against LLM strict.

| Baseline | Dem 2P RMSE | Margin MAE | Margin bias | Winner acc | Winner flips | Corr true margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `survey_memory_llm_strict` | 0.162 | 0.314 | -0.314 | 0.700 | 3 | 0.286 |
| `party_id_baseline` | 0.074 | 0.126 | 0.124 | 0.400 | 6 | 0.626 |
| `sklearn_logit_pre_only_crossfit` | 0.047 | 0.082 | -0.073 | 0.800 | 2 | 0.647 |
| `sklearn_logit_poll_informed` | 0.048 | 0.079 | -0.073 | 0.800 | 2 | 0.535 |
| `mit_2020_state_prior` | 0.019 | 0.034 | 0.034 | 0.400 | 6 | 0.956 |
| `uniform_national_swing_from_2020` | 0.015 | 0.026 | -0.026 | 1.000 | 0 | 0.956 |
| `ces_post_self_report_aggregate_oracle` | 0.031 | 0.055 | -0.039 | 0.700 | 3 | 0.740 |

Core conclusions:

1. LLM strict has winner accuracy of 0.700, but that number is structurally misleading: 7 of the 10 state truths are Republican winners, so predicting all states Republican also yields 0.700.
2. The more important metric is margin: LLM strict has `margin_bias=-0.314`, meaning average margin is biased toward Republicans by about 31.4 percentage points.
3. LLM strict has `dem_2p_rmse=0.162` and `margin_mae=0.314`, clearly weaker than all key non-LLM baselines.
4. `uniform_national_swing_from_2020` is very strong across these 10 states, with winner accuracy 1.000 and margin MAE 0.026. This shows that 2024 state-level results in these states can be well explained by a simple historical swing prior.
5. `mit_2020_state_prior` has low share/margin RMSE, but winner accuracy only 0.400 because the 2020 prior predicted too many Democratic winners among near-50/50 swing states.

## 9. Non-LLM at Sample Size 2000

Non-LLM baselines can run up to 2000, which is useful for observing upper bounds from statistical models and oracle at larger sample sizes.

| Baseline | Dem 2P RMSE | Margin MAE | Margin bias | Winner acc | Winner flips | Corr true margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `party_id_baseline` | 0.080 | 0.146 | 0.146 | 0.400 | 6 | 0.731 |
| `sklearn_logit_pre_only_crossfit` | 0.029 | 0.050 | -0.028 | 0.900 | 1 | 0.734 |
| `sklearn_logit_poll_informed` | 0.028 | 0.050 | -0.027 | 0.800 | 2 | 0.698 |
| `mit_2020_state_prior` | 0.019 | 0.034 | 0.034 | 0.400 | 6 | 0.956 |
| `uniform_national_swing_from_2020` | 0.015 | 0.026 | -0.026 | 1.000 | 0 | 0.956 |
| `ces_post_self_report_aggregate_oracle` | 0.019 | 0.032 | 0.001 | 0.800 | 2 | 0.836 |

The larger non-LLM sample does not change the main conclusion: statistical models and historical swing priors are clearly stronger than the current LLM strict condition.

## 10. LLM Sample Size Sensitivity

`survey_memory_llm_strict` as sample size increases:

| Sample size | Dem 2P RMSE | Margin MAE | Margin bias | Winner acc | Winner flips |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.204 | 0.393 | -0.393 | 0.700 | 3 |
| 100 | 0.176 | 0.342 | -0.342 | 0.700 | 3 |
| 200 | 0.173 | 0.333 | -0.333 | 0.700 | 3 |
| 300 | 0.171 | 0.332 | -0.332 | 0.700 | 3 |
| 500 | 0.162 | 0.314 | -0.314 | 0.700 | 3 |

Increasing sample size does reduce variance, and RMSE/MAE improve from 50 to 500. But winner accuracy stays at 0.700 because the LLM predicts all 10 states as Republican winners at every sample size, missing MN/VA/CO.

This shows that larger samples cannot fix systematic bias; they only make the bias estimate more stable.

## 11. State-Level LLM Results at Sample Size 500

`survey_memory_llm_strict` state-level details:

| State | Pred Dem 2P | True Dem 2P | Dem 2P error | Pred margin | True margin | Margin error | Pred winner | True winner | Correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| AZ | 0.326 | 0.472 | -0.146 | -0.347 | -0.056 | -0.291 | republican | republican | true |
| CO | 0.342 | 0.556 | -0.215 | -0.317 | 0.113 | -0.430 | republican | democrat | false |
| GA | 0.317 | 0.489 | -0.172 | -0.366 | -0.022 | -0.344 | republican | republican | true |
| MI | 0.318 | 0.493 | -0.175 | -0.365 | -0.014 | -0.350 | republican | republican | true |
| MN | 0.424 | 0.522 | -0.097 | -0.151 | 0.043 | -0.195 | republican | democrat | false |
| NC | 0.282 | 0.484 | -0.202 | -0.436 | -0.033 | -0.403 | republican | republican | true |
| NV | 0.363 | 0.484 | -0.121 | -0.273 | -0.032 | -0.242 | republican | republican | true |
| PA | 0.409 | 0.491 | -0.083 | -0.183 | -0.017 | -0.165 | republican | republican | true |
| VA | 0.347 | 0.530 | -0.182 | -0.306 | 0.059 | -0.365 | republican | democrat | false |
| WI | 0.320 | 0.496 | -0.176 | -0.361 | -0.009 | -0.352 | republican | republican | true |

All 10 states have negative `dem_2p_error`; this is not a normal consequence of "Republicans won seven swing states", but a systematic underestimation of Democratic two-party share relative to truth.

## 12. Group Metrics

At common sample size 500, decomposed by state group:

| Group | Baseline | N states | Dem 2P RMSE | Margin MAE | Margin bias | Winner acc | Winner flips |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all10 | `survey_memory_llm_strict` | 10 | 0.162 | 0.314 | -0.314 | 0.700 | 3 |
| all10 | `party_id_baseline` | 10 | 0.074 | 0.126 | 0.124 | 0.400 | 6 |
| all10 | `sklearn_logit_pre_only_crossfit` | 10 | 0.047 | 0.082 | -0.073 | 0.800 | 2 |
| all10 | `sklearn_logit_poll_informed` | 10 | 0.048 | 0.079 | -0.073 | 0.800 | 2 |
| all10 | `mit_2020_state_prior` | 10 | 0.019 | 0.034 | 0.034 | 0.400 | 6 |
| all10 | `uniform_national_swing_from_2020` | 10 | 0.015 | 0.026 | -0.026 | 1.000 | 0 |
| all10 | `ces_post_self_report_aggregate_oracle` | 10 | 0.031 | 0.055 | -0.039 | 0.700 | 3 |
| original7 | `survey_memory_llm_strict` | 7 | 0.158 | 0.307 | -0.307 | 1.000 | 0 |
| original7 | `party_id_baseline` | 7 | 0.072 | 0.117 | 0.115 | 0.143 | 6 |
| original7 | `sklearn_logit_pre_only_crossfit` | 7 | 0.048 | 0.083 | -0.079 | 1.000 | 0 |
| original7 | `sklearn_logit_poll_informed` | 7 | 0.044 | 0.071 | -0.071 | 1.000 | 0 |
| original7 | `mit_2020_state_prior` | 7 | 0.019 | 0.035 | 0.035 | 0.143 | 6 |
| original7 | `uniform_national_swing_from_2020` | 7 | 0.015 | 0.025 | -0.025 | 1.000 | 0 |
| original7 | `ces_post_self_report_aggregate_oracle` | 7 | 0.033 | 0.061 | -0.042 | 0.714 | 2 |
| new3 | `survey_memory_llm_strict` | 3 | 0.172 | 0.330 | -0.330 | 0.000 | 3 |
| new3 | `party_id_baseline` | 3 | 0.076 | 0.147 | 0.147 | 1.000 | 0 |
| new3 | `sklearn_logit_pre_only_crossfit` | 3 | 0.044 | 0.077 | -0.060 | 0.333 | 2 |
| new3 | `sklearn_logit_poll_informed` | 3 | 0.055 | 0.100 | -0.078 | 0.333 | 2 |
| new3 | `mit_2020_state_prior` | 3 | 0.017 | 0.033 | 0.033 | 1.000 | 0 |
| new3 | `uniform_national_swing_from_2020` | 3 | 0.014 | 0.027 | -0.027 | 1.000 | 0 |
| new3 | `ces_post_self_report_aggregate_oracle` | 3 | 0.025 | 0.044 | -0.032 | 0.667 | 1 |

Group interpretation:

1. On the original seven states, LLM strict winner accuracy is 1.000 because all original seven state truths were Republican winners and the LLM also predicted them all Republican.
2. But LLM margin MAE on the original seven states remains as high as 0.307, so the winners are correct but the margins are badly wrong.
3. On the added three states MN/VA/CO, LLM strict winner accuracy is 0.000: all three Democratic-winning states are predicted Republican.
4. The added three states expose how sensitive winner accuracy is to the composition of the state set, and prove that looking only at seven Republican-winning swing states significantly overestimates a Republican-skewed model.

## 13. Relationship to Old E02

Old E02 ran only seven swing states, with at most 70 LLM agents per state. Old E02 LLM strict metrics:

| Sample size | Dem 2P RMSE | Margin MAE | Margin bias | Winner acc | Winner flips |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.203 | 0.404 | -0.404 | 1.000 | 0 |
| 70 | 0.180 | 0.351 | -0.351 | 1.000 | 0 |

New E02-large LLM strict on the original seven states, 500 per state:

| Group | Dem 2P RMSE | Margin MAE | Margin bias | Winner acc | Winner flips |
| --- | ---: | ---: | ---: | ---: | ---: |
| original7, sample 500 | 0.158 | 0.307 | -0.307 | 1.000 | 0 |

With a larger sample, LLM margin error on the original seven states declines somewhat, but remains very large. After adding MN/VA/CO, full 10-state winner accuracy falls from the superficial 1.000 to 0.700, revealing that old E02 winner accuracy was strongly affected by the state-set composition.

## 14. Conclusion

### 14.1 Engineering Conclusion

E02-large passes as an engineering pipeline:

1. All 5000 LLM calls parsed successfully.
2. No invalid choice, forbidden choice, legacy probability schema, or transport error.
3. No fallback.
4. workers=4 was stable, with formal wall-clock around 23 minutes, below the 30-minute budget.
5. GPU peak memory was about 4998 MiB, with minimum free memory about 2894 MiB, so there was no OOM risk.

The current system can stably run a 10-state x 500-per-state aggregate LLM experiment.

### 14.2 Scientific Conclusion

The current `qwen3.5:2b` strict-memory LLM is not suitable as a reliable state-level election simulator:

1. It underestimates Democratic two-party share in all 10 states.
2. It predicts all 10 states as Republican winners, so it gets the seven Republican-winning states right and misses all three Democratic-winning states MN/VA/CO.
3. Its 10-state margin bias is -0.314, meaning average margin is biased toward Republicans by about 31.4 percentage points.
4. It is clearly weaker than non-LLM baselines such as `sklearn_logit_pre_only_crossfit`, `sklearn_logit_poll_informed`, `uniform_national_swing_from_2020`, and `mit_2020_state_prior`.
5. Increasing sample size only reduces variance; it does not fix systematic Republican skew.

### 14.3 Impact on E01-Large Conclusions

This E02-large run reinforces the E01-large assessment:

1. strict memory does not show a stable gain.
2. The current LLM can use political cues in the persona, but forms strong systematic bias at the aggregate level.
3. Looking only at winner accuracy is misleading, especially when most state truths in the set are wins for the same party.
4. Future reports must prioritize `dem_2p_rmse`, `margin_mae`, and `margin_bias`; winner accuracy should only be a secondary metric.

## 15. Recommendations

If E02 is further scaled or redesigned, I recommend:

1. Do not treat LLM strict as the main predictor result for now; use it as a diagnostic object.
2. Aggregate interpretations in E02/E04/E05 must always include Democratic-winning contrast states to avoid inflated winner accuracy from Republican skew.
3. If improving the LLM simulator, prioritize:
   - turnout almost never outputs `not_vote`;
   - strict memory causes stronger Republican bias;
   - aggregate calibration is missing;
   - there is no reliable calibration layer between state-level priors and persona vote choice.
4. The next round can test "two-stage turnout/vote choice" or "persona hard choice + posterior calibration layer" instead of simply increasing LLM sample size.

E02-large formally passes and the results are reusable, but the scientific conclusion is unfavorable for the current LLM aggregate simulator.
