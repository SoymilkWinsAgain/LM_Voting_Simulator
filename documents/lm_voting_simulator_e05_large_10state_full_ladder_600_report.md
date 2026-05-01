# LM Voting Simulator E05-Large Report: 10-State / Full-Ladder Ablation

## 1. Experiment Overview

This report summarizes the larger E05 Information Ablation and Placebo Memory experiment.
It does not replace the earlier E05 report or the full evaluation report.

- Formal run directory: `data/runs/eval_suite_local/05_ablation_placebo_large_10state_full_ladder_600`
- Formal config: `configs/eval_suite/e05_ablation_placebo_large_10state_full_ladder_600.yaml`
- Previous E05 directory retained: `data/runs/eval_suite_local/05_ablation_placebo`
- Model: `qwen3.5:2b`
- Ollama endpoint: `http://172.26.48.1:11434`
- Response contract: hard choice JSON, `{"choice": "not_vote|democrat|republican"}`
- States: `PA, MI, WI, GA, AZ, NV, NC, MN, VA, CO`
- Formal sample: 50 main + 10 diagnostic boost per state, 600 agents total
- Formal task count: 600 agents x 10 baselines = 6000 responses

The main difference from the previous E05 is that this run uses the full 10-condition ablation ladder. The earlier formal E05 used 4 states, 60 agents, 420 responses, and 7 conditions; it did not include `L2`, `L4`, or `L5`.

## 2. Full Ladder

The run used all planned E05 conditions:

| Code | Meaning |
| --- | --- |
| `L1_demographic_only_llm` | demographics only |
| `L2_demographic_state_llm` | demographics + state |
| `L3_party_ideology_llm` | demographics + state + party/ideology |
| `L4_party_ideology_context_llm` | L3 + candidate context |
| `L5_strict_memory_llm` | strict pre-election memory, no candidate context |
| `L6_strict_memory_context_llm` | strict pre-election memory + candidate context |
| `L7_poll_informed_memory_context_llm` | poll-informed memory + candidate context |
| `L8_post_hoc_oracle_memory_context_llm` | post-election oracle upper bound |
| `P1_memory_shuffled_within_state_llm` | strict memory shuffled within state |
| `P2_memory_shuffled_within_party_llm` | strict memory shuffled within party |

The `ablation_deltas.parquet` output was extended to include both `individual_main` and `aggregate` deltas.

## 3. Prechecks and Probes

Prechecks passed:

- `python -m py_compile src/election_sim/ces_ablation_benchmark.py src/election_sim/eval_suite.py src/election_sim/cli.py`
- E05 and hard-choice parser pytest targets passed: 6 tests
- Ollama `/api/tags` showed `qwen3.5:2b`
- Required CES/MIT parquet inputs were present
- Initial GPU state was safe, about 1.3 GiB used and 6.6 GiB free

Probe summary:

| Probe | Workers | Calls | Runtime sec | LLM sec | Throughput resp/s | Median latency sec | P90 latency sec | Peak used MiB | Min free MiB | Gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A 5-state w1 | 1 | 150 | 124.91 | 64.86 | 2.313 | 0.401 | 0.499 | 5162 | 2730 | PASS |
| B 10-state w3 | 3 | 300 | 124.05 | 60.72 | 4.941 | 0.617 | 0.765 | 5162 | 2730 | PASS |
| C 10-state w4 | 4 | 300 | 127.13 | 61.37 | 4.889 | 0.831 | 0.988 | 5162 | 2730 | PASS |
| D 10-state w5 | 5 | 300 | 123.82 | 59.89 | 5.009 | 0.998 | 1.207 | 5162 | 2730 | PASS |

The formal run used `workers=3`. `workers=5` was only about 1.4% faster than `workers=3` and had higher p90 latency, so the stable lower-concurrency setting was preferred.

## 4. Formal Runtime and Quality Gates

Formal runtime:

| Metric | Value |
| --- | ---: |
| Agents | 600 |
| Prompts / LLM tasks | 6000 |
| Requested main agents per state | 50 |
| Effective main agents per state | 50 |
| Diagnostic boost per state | 10 |
| Total runtime sec | 1105.14 |
| LLM runtime sec | 1027.39 |
| Throughput resp/s | 5.840 |
| Median latency sec | 0.642 |
| P90 latency sec | 0.719 |
| Cache hit rate | 0.182 |
| Ollama calls | 4907 |
| GPU peak memory used MiB | 5162 |
| GPU minimum free memory MiB | 2730 |
| GPU peak utilization pct | 84 |

Quality gates:

| Gate | Value | Status |
| --- | ---: | --- |
| `parse_ok_rate` | 1.000 | PASS |
| `invalid_choice_rate` | 0.000 | PASS |
| `forbidden_choice_rate` | 0.000 | PASS |
| `legacy_probability_schema_rate` | 0.000 | PASS |
| `transport_error_rate` | 0.000 | PASS |

All required output files were present, including final parquet outputs, `runtime.json`, `runtime_log.parquet`, `llm_cache.jsonl`, `benchmark_report.md`, and 32 figure files.

## 5. Choice Distribution

| Baseline | Democrat | Not vote | Republican | Democrat rate | Not-vote rate | Republican rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 demographic only | 0 | 0 | 600 | 0.000 | 0.000 | 1.000 |
| L2 demographic + state | 0 | 0 | 600 | 0.000 | 0.000 | 1.000 |
| L3 party/ideology | 205 | 3 | 392 | 0.342 | 0.005 | 0.653 |
| L4 party/ideology + context | 17 | 0 | 583 | 0.028 | 0.000 | 0.972 |
| L5 strict memory only | 144 | 0 | 456 | 0.240 | 0.000 | 0.760 |
| L6 strict memory + context | 31 | 0 | 569 | 0.052 | 0.000 | 0.948 |
| L7 poll-informed + context | 48 | 6 | 546 | 0.080 | 0.010 | 0.910 |
| L8 post-hoc oracle | 262 | 39 | 299 | 0.437 | 0.065 | 0.498 |
| P1 shuffled within state | 3 | 0 | 597 | 0.005 | 0.000 | 0.995 |
| P2 shuffled within party | 3 | 0 | 597 | 0.005 | 0.000 | 0.995 |

The key pattern is that party/ideology alone is much less collapsed than candidate-context conditions. Adding explicit candidate context sharply moves the model back toward Republican choices.

## 6. Individual Main-Sample Metrics

Weighted main-sample individual metrics:

| Baseline | Turnout accuracy | Turnout ECE | Vote accuracy | Vote macro F1 |
| --- | ---: | ---: | ---: | ---: |
| L1 demographic only | 0.959 | 0.041 | 0.537 | 0.233 |
| L2 demographic + state | 0.959 | 0.041 | 0.537 | 0.233 |
| L3 party/ideology | 0.952 | 0.048 | 0.924 | 0.619 |
| L4 party/ideology + context | 0.959 | 0.041 | 0.556 | 0.263 |
| L5 strict memory only | 0.959 | 0.041 | 0.812 | 0.532 |
| L6 strict memory + context | 0.959 | 0.041 | 0.599 | 0.322 |
| L7 poll-informed + context | 0.953 | 0.047 | 0.643 | 0.376 |
| L8 post-hoc oracle | 0.993 | 0.008 | 0.993 | 0.662 |
| P1 shuffled within state | 0.959 | 0.041 | 0.540 | 0.238 |
| P2 shuffled within party | 0.959 | 0.041 | 0.541 | 0.239 |

`L3_party_ideology_llm` is the best non-oracle condition by individual vote accuracy and macro F1. `L5_strict_memory_llm` is second. `L6` and `L7`, both with candidate context, are much worse than `L3` and `L5`.

Turnout is less informative because most non-oracle conditions rarely choose `not_vote`.

## 7. Aggregate Metrics

State-level aggregate metrics:

| Baseline | Dem 2P RMSE | Margin MAE | Margin Bias | Winner Accuracy | Winner Flips |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1 demographic only | 0.502 | 1.003 | -1.003 | 0.700 | 3 |
| L2 demographic + state | 0.502 | 1.003 | -1.003 | 0.700 | 3 |
| L3 party/ideology | 0.139 | 0.244 | -0.244 | 0.700 | 3 |
| L4 party/ideology + context | 0.478 | 0.953 | -0.953 | 0.700 | 3 |
| L5 strict memory only | 0.257 | 0.496 | -0.496 | 0.700 | 3 |
| L6 strict memory + context | 0.448 | 0.885 | -0.885 | 0.700 | 3 |
| L7 poll-informed + context | 0.409 | 0.808 | -0.808 | 0.700 | 3 |
| L8 post-hoc oracle | 0.114 | 0.184 | -0.113 | 0.600 | 4 |
| P1 shuffled within state | 0.493 | 0.984 | -0.984 | 0.700 | 3 |
| P2 shuffled within party | 0.498 | 0.995 | -0.995 | 0.700 | 3 |

The 0.700 winner accuracy is misleading for most conditions. It comes from predicting Republican in all 7 Republican-won states and missing the 3 Democratic-won states. The group split makes this explicit:

| Baseline | Group | Margin MAE | Margin Bias | Winner Accuracy |
| --- | --- | ---: | ---: | ---: |
| L3 party/ideology | old 7 R-won states | 0.264 | -0.264 | 1.000 |
| L3 party/ideology | new 3 D-won states | 0.197 | -0.197 | 0.000 |
| L5 strict memory only | old 7 R-won states | 0.527 | -0.527 | 1.000 |
| L5 strict memory only | new 3 D-won states | 0.423 | -0.423 | 0.000 |
| L6 strict memory + context | old 7 R-won states | 0.836 | -0.836 | 1.000 |
| L6 strict memory + context | new 3 D-won states | 0.999 | -0.999 | 0.000 |
| L8 post-hoc oracle | old 7 R-won states | 0.197 | -0.153 | 0.714 |
| L8 post-hoc oracle | new 3 D-won states | 0.153 | -0.019 | 0.333 |

Even the post-hoc oracle is not perfect at aggregate winner prediction because the hard-choice sampled aggregate can still deviate from state truth under small state sample sizes and survey weighting. But it is much better on margin/RMSE than non-oracle context-heavy conditions.

## 8. Full Ladder Deltas

Selected individual vote deltas:

| Delta | From | To | Vote accuracy delta | Macro F1 delta |
| --- | --- | --- | ---: | ---: |
| State increment | L1 | L2 | 0.000 | 0.000 |
| Party/ideology increment | L2 | L3 | +0.388 | +0.386 |
| Candidate context increment | L3 | L4 | -0.368 | -0.357 |
| Strict memory increment | L4 | L6 | +0.043 | +0.060 |
| Strict context increment | L5 | L6 | -0.213 | -0.210 |
| Poll increment | L6 | L7 | +0.044 | +0.053 |
| Oracle gap from strict | L6 | L8 | +0.393 | +0.340 |
| Oracle gap from poll | L7 | L8 | +0.350 | +0.286 |
| State placebo gap | P1 | L6 | +0.060 | +0.084 |
| Party placebo gap | P2 | L6 | +0.059 | +0.083 |

Selected aggregate deltas:

| Delta | From | To | Margin MAE delta | Dem 2P RMSE delta |
| --- | --- | --- | ---: | ---: |
| State increment | L1 | L2 | 0.000 | 0.000 |
| Party/ideology increment | L2 | L3 | -0.760 | -0.363 |
| Candidate context increment | L3 | L4 | +0.710 | +0.339 |
| Strict memory increment | L4 | L6 | -0.068 | -0.030 |
| Strict context increment | L5 | L6 | +0.389 | +0.191 |
| Poll increment | L6 | L7 | -0.076 | -0.039 |
| Oracle gap from strict | L6 | L8 | -0.701 | -0.334 |
| Oracle gap from poll | L7 | L8 | -0.625 | -0.295 |
| State placebo gap | P1 | L6 | -0.099 | -0.045 |
| Party placebo gap | P2 | L6 | -0.110 | -0.051 |

The most important deltas are:

- Party/ideology is the largest useful information increment.
- Candidate context is strongly harmful in this small-model setup.
- Strict memory helps a little over context-only `L4`, but strict memory without context `L5` is much better than `L6`.
- Poll-informed memory helps over `L6`, but not enough to overcome the context-induced Republican skew.
- Real strict memory beats shuffled memory only modestly.

## 9. Placebo Memory

Placebo memory donor diagnostics:

| Baseline | Scope | Fallback reason | Rows |
| --- | --- | --- | ---: |
| P1 shuffled within state | state | none | 600 |
| P2 shuffled within party | party | none | 600 |

No donor fallback was needed. Therefore the placebo comparison is valid for this sample.

`L6` beats both shuffled-memory baselines, but the gain is small relative to the oracle gap:

- `L6 - P1`: vote accuracy +0.060, macro F1 +0.084
- `L6 - P2`: vote accuracy +0.059, macro F1 +0.083
- `L8 - L6`: vote accuracy +0.393, macro F1 +0.340

This suggests respondent-specific strict memory is being used somewhat, but not enough to make the simulator reliable.

## 10. Interpretation

The large E05 result is more diagnostic than the smaller run because it includes all missing ladder rungs.

Main findings:

1. `L3_party_ideology_llm` is the strongest non-oracle condition.
2. Adding candidate context sharply degrades performance: `L3 -> L4` is a large negative step.
3. Strict memory without candidate context (`L5`) is much better than strict memory with candidate context (`L6`).
4. Poll-informed memory (`L7`) improves over `L6`, but remains far below `L3` and `L5`.
5. Real memory beats shuffled memory, but only modestly.
6. The Republican skew remains the dominant failure mode in context-heavy prompts.
7. The added Democratic-won states remain useful: they reveal that many 0.700 winner-accuracy results are just all-Republican prediction artifacts.

This does not overturn the earlier conclusion. It refines it: the current small model can use party/ideology signals, and it can use oracle information when directly given. But candidate context and memory-context prompts push it toward a strong Republican collapse, making the strict-memory LLM path unreliable as an election simulator.

## 11. Artifacts

Key artifacts:

- `data/runs/eval_suite_local/05_ablation_placebo_large_10state_full_ladder_600/runtime.json`
- `data/runs/eval_suite_local/05_ablation_placebo_large_10state_full_ladder_600/runtime_log.parquet`
- `data/runs/eval_suite_local/05_ablation_placebo_large_10state_full_ladder_600/responses.parquet`
- `data/runs/eval_suite_local/05_ablation_placebo_large_10state_full_ladder_600/individual_metrics.parquet`
- `data/runs/eval_suite_local/05_ablation_placebo_large_10state_full_ladder_600/aggregate_metrics.parquet`
- `data/runs/eval_suite_local/05_ablation_placebo_large_10state_full_ladder_600/ablation_deltas.parquet`
- `data/runs/eval_suite_local/05_ablation_placebo_large_10state_full_ladder_600/memory_placebo_diagnostics.parquet`
- `data/runs/eval_suite_local/05_ablation_placebo_large_10state_full_ladder_600/benchmark_report.md`
- `data/runs/eval_suite_local/05_ablation_placebo_large_10state_full_ladder_600/figures/`

The run is valid for follow-up analysis and can be used as the large-scale E05 reference for any later large E06-style subgroup/calibration rerun.
