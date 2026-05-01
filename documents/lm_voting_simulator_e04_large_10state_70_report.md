# LM Voting Simulator E04-Large Report: 10-State / 70-Agent Leakage Stress Test

## 1. Experiment Overview

This report summarizes the larger E04 Historical / World-Knowledge Leakage experiment.
It does not replace the earlier E04 report or the full evaluation report.

- Formal run directory: `data/runs/eval_suite_local/04_leakage_large_10state_70`
- Formal config: `configs/eval_suite/e04_leakage_large_10state_70.yaml`
- Previous E04 directory retained: `data/runs/eval_suite_local/04_leakage`
- Model: `qwen3.5:2b`
- Ollama endpoint: `http://172.26.48.1:11434`
- Response contract: hard choice JSON, `{"choice": "not_vote|democrat|republican"}`
- States: `PA, MI, WI, GA, AZ, NV, NC, MN, VA, CO`
- Formal sample: 70 agents per state, 700 agents total
- Conditions: 7 leakage conditions, 4900 LLM calls total

The 10-state design keeps the seven 2024 Republican-won swing states and adds three Democratic-won states:

- Original 7: `PA, MI, WI, GA, AZ, NV, NC`
- Added 3: `MN, VA, CO`

The larger state set is important because the smaller all-or-mostly-Republican target set can hide a model that simply predicts Republican. The added Democratic-won states expose whether the simulator can recover the direction of Democratic state outcomes.

## 2. Implementation Notes

The E04 runner already supported parallel LLM calls, runtime logs, partial checkpoints, GPU snapshots, and hard-choice parsing. For this larger run, the 10-state leakage mappings were extended so new states do not become no-op state swaps.

State swap mapping used in this run:

| Original | Displayed in `state_swap_placebo` |
| --- | --- |
| PA | MN |
| MN | PA |
| GA | VA |
| VA | GA |
| AZ | CO |
| CO | AZ |
| MI | NC |
| NC | MI |
| WI | NV |
| NV | WI |

Masked-state fictitious codes were also extended:

| State | Masked code |
| --- | --- |
| PA | F01 |
| GA | F02 |
| AZ | F03 |
| MN | F04 |
| VA | F05 |
| CO | F06 |
| MI | F07 |
| WI | F08 |
| NV | F09 |
| NC | F10 |

## 3. Preflight, Probes, and Worker Choice

Prechecks passed:

- `python -m py_compile src/election_sim/ces_leakage_benchmark.py src/election_sim/eval_suite.py src/election_sim/cli.py`
- E04-related pytest targets passed: 5 tests
- Ollama `/api/tags` showed `qwen3.5:2b`
- Initial GPU state was safe: about 1.1 GiB used and 6.7 GiB free

Probe summary:

| Probe | Workers | Calls | Runtime sec | LLM sec | Throughput resp/s | Median latency sec | P90 latency sec | Peak used MiB | Min free MiB | Gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A 10-state w1 | 1 | 70 | 47.74 | 34.58 | 2.024 | 0.435 | 0.499 | 5002 | 2890 | PASS |
| B 10-state w3 | 3 | 140 | 49.74 | 36.27 | 3.860 | 0.714 | 0.820 | 5002 | 2890 | PASS |
| C 10-state w4 | 4 | 140 | 50.33 | 36.60 | 3.825 | 0.979 | 1.073 | 5002 | 2890 | PASS |
| D 10-state w5 | 5 | 140 | 47.48 | 33.49 | 4.181 | 1.109 | 1.141 | 5002 | 2890 | PASS |

The formal run used `workers=3`. Although `workers=5` was slightly faster than `workers=3`, the gain was below the planned 10% threshold and latency rose. `workers=3` was the more conservative stable choice.

## 4. Formal Runtime and Quality Gates

Formal runtime:

| Metric | Value |
| --- | ---: |
| Agents | 700 |
| Prompts / LLM tasks | 4900 |
| Effective agents per state | 70 |
| Total runtime sec | 1201.95 |
| LLM runtime sec | 1179.86 |
| Throughput resp/s | 4.153 |
| Median latency sec | 0.673 |
| P90 latency sec | 0.742 |
| Cache hit rate | 0.000 |
| Ollama calls | 4900 |
| GPU peak memory used MiB | 5164 |
| GPU minimum free memory MiB | 2728 |
| GPU peak utilization pct | 84 |

Quality gates:

| Gate | Value | Status |
| --- | ---: | --- |
| `parse_ok_rate` | 1.000 | PASS |
| `invalid_choice_rate` | 0.000 | PASS |
| `forbidden_choice_rate` | 0.000 | PASS |
| `legacy_probability_schema_rate` | 0.000 | PASS |
| `transport_error_rate` | 0.000 | PASS |

All required output files were present, including final parquet outputs, `runtime.json`, `runtime_log.parquet`, `llm_cache.jsonl`, `benchmark_report.md`, and figures.

## 5. Choice Distribution

No condition produced `not_vote` in the final run. Every parsed response was either `democrat` or `republican`.

| Condition | Democrat | Republican | Democrat rate | Republican rate |
| --- | ---: | ---: | ---: | ---: |
| anonymous_candidates | 372 | 328 | 0.531 | 0.469 |
| candidate_swap_placebo | 0 | 700 | 0.000 | 1.000 |
| masked_state | 34 | 666 | 0.049 | 0.951 |
| masked_year | 2 | 698 | 0.003 | 0.997 |
| named_candidates | 49 | 651 | 0.070 | 0.930 |
| party_only_candidates | 7 | 693 | 0.010 | 0.990 |
| state_swap_placebo | 44 | 656 | 0.063 | 0.937 |

The central behavioral result is simple: except for the anonymous policy-summary condition, the model overwhelmingly predicts Republican. This is not a parsing failure. It is a model behavior signal.

## 6. Aggregate Metrics

Aggregate metrics use original-state MIT 2024 truth even when prompts mask or swap displayed state.

| Condition | Dem 2P RMSE | Margin MAE | Margin Bias | Winner Accuracy | Winner Flips |
| --- | ---: | ---: | ---: | ---: | ---: |
| anonymous_candidates | 0.074 | 0.133 | 0.049 | 0.500 | 5 |
| candidate_swap_placebo | 0.502 | 1.003 | -1.003 | 0.700 | 3 |
| masked_state | 0.469 | 0.933 | -0.933 | 0.700 | 3 |
| masked_year | 0.500 | 0.998 | -0.998 | 0.700 | 3 |
| named_candidates | 0.446 | 0.881 | -0.881 | 0.700 | 3 |
| party_only_candidates | 0.496 | 0.990 | -0.990 | 0.700 | 3 |
| state_swap_placebo | 0.459 | 0.907 | -0.907 | 0.700 | 3 |

The 0.700 winner accuracy in most non-anonymous conditions is misleading. It comes from predicting Republican in all or nearly all states: the original seven states are Republican-won, while the three added states are Democratic-won.

Group-level split:

| Condition | Group | Dem 2P RMSE | Margin MAE | Margin Bias | Winner Accuracy | Flips |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| anonymous_candidates | all10 | 0.074 | 0.133 | 0.049 | 0.500 | 5 |
| anonymous_candidates | old7 | 0.074 | 0.132 | 0.035 | 0.429 | 4 |
| anonymous_candidates | new3 | 0.074 | 0.136 | 0.082 | 0.667 | 1 |
| named_candidates | all10 | 0.446 | 0.881 | -0.881 | 0.700 | 3 |
| named_candidates | old7 | 0.425 | 0.839 | -0.839 | 1.000 | 0 |
| named_candidates | new3 | 0.492 | 0.981 | -0.981 | 0.000 | 3 |
| party_only_candidates | all10 | 0.496 | 0.990 | -0.990 | 0.700 | 3 |
| party_only_candidates | old7 | 0.480 | 0.961 | -0.961 | 1.000 | 0 |
| party_only_candidates | new3 | 0.530 | 1.059 | -1.059 | 0.000 | 3 |
| masked_year | all10 | 0.500 | 0.998 | -0.998 | 0.700 | 3 |
| masked_year | old7 | 0.484 | 0.967 | -0.967 | 1.000 | 0 |
| masked_year | new3 | 0.536 | 1.072 | -1.072 | 0.000 | 3 |
| masked_state | all10 | 0.469 | 0.933 | -0.933 | 0.700 | 3 |
| masked_state | old7 | 0.454 | 0.904 | -0.904 | 1.000 | 0 |
| masked_state | new3 | 0.502 | 1.003 | -1.003 | 0.000 | 3 |
| state_swap_placebo | all10 | 0.459 | 0.907 | -0.907 | 0.700 | 3 |
| state_swap_placebo | old7 | 0.442 | 0.870 | -0.870 | 1.000 | 0 |
| state_swap_placebo | new3 | 0.497 | 0.993 | -0.993 | 0.000 | 3 |
| candidate_swap_placebo | all10 | 0.502 | 1.003 | -1.003 | 0.700 | 3 |
| candidate_swap_placebo | old7 | 0.487 | 0.974 | -0.974 | 1.000 | 0 |
| candidate_swap_placebo | new3 | 0.536 | 1.072 | -1.072 | 0.000 | 3 |

## 7. State-Level Predictions

Named-candidate condition:

| State | Pred Dem 2P | True Dem 2P | Error | Pred winner | True winner | Correct |
| --- | ---: | ---: | ---: | --- | --- | --- |
| AZ | 0.158 | 0.472 | -0.314 | republican | republican | yes |
| CO | 0.024 | 0.556 | -0.533 | republican | democrat | no |
| GA | 0.006 | 0.489 | -0.483 | republican | republican | yes |
| MI | 0.024 | 0.493 | -0.468 | republican | republican | yes |
| MN | 0.061 | 0.522 | -0.461 | republican | democrat | no |
| NC | 0.006 | 0.484 | -0.478 | republican | republican | yes |
| NV | 0.035 | 0.484 | -0.449 | republican | republican | yes |
| PA | 0.179 | 0.491 | -0.312 | republican | republican | yes |
| VA | 0.052 | 0.530 | -0.478 | republican | democrat | no |
| WI | 0.065 | 0.496 | -0.431 | republican | republican | yes |

Anonymous-candidate condition:

| State | Pred Dem 2P | True Dem 2P | Error | Pred winner | True winner | Correct |
| --- | ---: | ---: | ---: | --- | --- | --- |
| AZ | 0.473 | 0.472 | 0.001 | republican | republican | yes |
| CO | 0.664 | 0.556 | 0.108 | democrat | democrat | yes |
| GA | 0.580 | 0.489 | 0.091 | democrat | republican | no |
| MI | 0.533 | 0.493 | 0.041 | democrat | republican | no |
| MN | 0.578 | 0.522 | 0.056 | democrat | democrat | yes |
| NC | 0.591 | 0.484 | 0.108 | democrat | republican | no |
| NV | 0.395 | 0.484 | -0.089 | republican | republican | yes |
| PA | 0.412 | 0.491 | -0.080 | republican | republican | yes |
| VA | 0.489 | 0.530 | -0.040 | republican | democrat | no |
| WI | 0.548 | 0.496 | 0.052 | democrat | republican | no |

Anonymous candidates are much less Republican-collapsed and have far better margin metrics. However, this condition is not a clean predictive win for the simulator: it removes names and parties and replaces them with policy summaries. It is best interpreted as a diagnostic that the model is highly sensitive to candidate/party framing.

## 8. Leakage Contrasts

Named candidates versus other conditions:

| Comparison | Metric | Named value | Comparison value | Named improvement |
| --- | --- | ---: | ---: | ---: |
| party_only_candidates | margin_mae | 0.881 | 0.990 | 0.109 |
| party_only_candidates | dem_2p_rmse | 0.446 | 0.496 | 0.050 |
| anonymous_candidates | margin_mae | 0.881 | 0.133 | -0.749 |
| anonymous_candidates | dem_2p_rmse | 0.446 | 0.074 | -0.372 |
| masked_year | margin_mae | 0.881 | 0.998 | 0.117 |
| masked_year | dem_2p_rmse | 0.446 | 0.500 | 0.054 |
| masked_state | margin_mae | 0.881 | 0.933 | 0.052 |
| masked_state | dem_2p_rmse | 0.446 | 0.469 | 0.023 |
| state_swap_placebo | margin_mae | 0.881 | 0.907 | 0.026 |
| state_swap_placebo | dem_2p_rmse | 0.446 | 0.459 | 0.013 |
| candidate_swap_placebo | margin_mae | 0.881 | 1.003 | 0.122 |
| candidate_swap_placebo | dem_2p_rmse | 0.446 | 0.502 | 0.056 |

Named candidates are slightly better than party-only, masked-year, masked-state, state-swap, and candidate-swap. But the absolute errors remain very large, and the model still predicts the three Democratic-won states as Republican. This is not evidence of a reliable named-candidate simulator.

## 9. State-Swap Diagnostic

State-swap diagnostic:

| State | Displayed state | Pred shift | Truth shift |
| --- | --- | ---: | ---: |
| AZ | CO | -0.282 | 0.169 |
| CO | AZ | 0.033 | -0.169 |
| GA | VA | 0.002 | 0.081 |
| MI | NC | -0.028 | -0.018 |
| MN | PA | -0.056 | -0.061 |
| NC | MI | 0.025 | 0.018 |
| NV | WI | -0.040 | 0.023 |
| PA | MN | 0.125 | 0.061 |
| VA | GA | -0.014 | -0.081 |
| WI | NV | -0.023 | -0.023 |

Summary:

- `pred_shift` vs `truth_shift` correlation: -0.460
- slope: -0.508

The model does not consistently move predictions toward the displayed state's true 2024 margin. This weakens a simple "it just knows the state outcome and follows the displayed state" explanation. The stronger issue is broader Republican collapse under named/party/year/state-framed prompts.

## 10. Candidate-Swap Diagnostic

Candidate-swap results:

| State | Party-following score | Name-following score | Name-minus-party index |
| --- | ---: | ---: | ---: |
| AZ | 0.842 | 0.158 | -0.684 |
| CO | 0.976 | 0.024 | -0.953 |
| GA | 0.994 | 0.006 | -0.989 |
| MI | 0.976 | 0.024 | -0.951 |
| MN | 0.940 | 0.061 | -0.879 |
| NC | 0.994 | 0.006 | -0.988 |
| NV | 0.965 | 0.035 | -0.930 |
| PA | 0.821 | 0.179 | -0.642 |
| VA | 0.948 | 0.052 | -0.896 |
| WI | 0.935 | 0.065 | -0.871 |

Mean name-minus-party index: -0.878.

Under candidate swap, the model outputs Republican for all 700 cases. It follows the party label much more than the candidate name. The E04 result therefore does not look like simple "Trump-name following"; it looks more like Republican/party-label dominance.

## 11. Individual Metrics

Weighted individual metrics:

| Condition | Turnout accuracy | Turnout ECE | Vote accuracy | Vote macro F1 |
| --- | ---: | ---: | ---: | ---: |
| anonymous_candidates | 0.956 | 0.044 | 0.956 | 0.637 |
| candidate_swap_placebo | 0.956 | 0.044 | 0.532 | 0.232 |
| masked_state | 0.956 | 0.044 | 0.575 | 0.294 |
| masked_year | 0.956 | 0.044 | 0.534 | 0.235 |
| named_candidates | 0.956 | 0.044 | 0.597 | 0.323 |
| party_only_candidates | 0.956 | 0.044 | 0.540 | 0.243 |
| state_swap_placebo | 0.956 | 0.044 | 0.584 | 0.306 |

Turnout metrics are not very informative here because the model never selected `not_vote`. That makes predicted turnout essentially always 1. The useful signal is in vote choice and aggregate behavior.

## 12. Comparison to Earlier E04

Earlier E04 used 60 agents and 420 LLM calls. This larger run used 700 agents and 4900 LLM calls.

The earlier run already suggested strong Republican skew in named/party/masked conditions. The larger run confirms and sharpens that result:

- The output contract and runner are stable at larger scale.
- The Republican skew persists at 10-state scale.
- The added Democratic-won states expose that 0.700 winner accuracy is mostly an artifact of seven Republican-won states.
- Anonymous policy-summary prompts produce much more balanced outputs and much better aggregate margins, showing that prompt/candidate framing strongly changes behavior.

## 13. Conclusion

E04-large passed as an engineering run: all 4900 responses parsed correctly, no schema regressions occurred, no transport errors occurred, GPU memory stayed safe, and all required artifacts were saved.

As a simulator evaluation, the result is unfavorable for the current `qwen3.5:2b` strict-memory LLM path:

- The model is not reliably simulating individual voter choice under named/party/year/state-framed prompts.
- The dominant failure mode is not a JSON/schema issue; it is a behavioral collapse toward Republican choices.
- Named candidates are slightly better than several masked comparisons, but the absolute aggregate errors are too large to treat that as useful predictive leakage.
- Candidate-swap results show party-label following rather than candidate-name following.
- State-swap results do not show clean displayed-state-prior tracking.
- The anonymous condition is much better numerically, but it is diagnostically different because it replaces candidate/party labels with policy summaries.

This large E04 does not overturn the previous large-experiment conclusion: the current small model and hard-choice setup are useful for stress-testing the simulator framework, but the `qwen3.5:2b` LLM path is not a reliable election simulator at state aggregate level.
