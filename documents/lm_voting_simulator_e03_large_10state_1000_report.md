# E03-Large 10-State / 1000-Agent Prompt Robustness Report

Generated at: 2026-05-01
Experiment directory: `data/runs/eval_suite_local/03_prompt_robustness_large_10state_1000`
Config file: `configs/eval_suite/e03_prompt_robustness_large_10state_1000.yaml`
Old E03 directory retained: `data/runs/eval_suite_local/03_prompt_robustness`

## 1. Experiment Goal

This run is the expanded version of E03 Prompt Robustness. Its goal is to evaluate whether `qwen3.5:2b` is stable across different prompt phrasings when using the same persona and the same strict pre-election memory information.

Core questions:

1. Do semantically equivalent or approximately equivalent prompt variants change hard-choice outputs?
2. Does changing candidate order systematically change Democrat/Republican choices?
3. Are interviewer-style, analyst-style, and strict JSON wording only surface changes, or do they change simulation conclusions?
4. Does the larger sample confirm the candidate-order sensitivity observed in old E03's small sample?

This run continues to use the hard-choice contract:

```json
{"choice": "not_vote|democrat|republican"}
```

Old probability robustness metrics are no longer primary metrics. All old probability columns in the report should only be interpreted as one-hot or aggregate frequencies derived from hard choices.

## 2. Design

### 2.1 States and Sample

This run uses 10 states, with 100 agents per state:

| Group | States | Agents |
| --- | --- | ---: |
| Original 7 2024 Republican-winning swing states | PA, MI, WI, GA, AZ, NV, NC | 700 |
| Added 3 2024 Democratic-winning states | MN, VA, CO | 300 |
| Total | PA, MI, WI, GA, AZ, NV, NC, MN, VA, CO | 1000 |

Formal scale:

```text
1000 agents x 5 prompt variants = 5000 LLM calls
```

### 2.2 Prompt Variants

This run reuses the five variants from old E03 to preserve comparability:

| Variant | Description |
| --- | --- |
| `base_json` | Current strict-memory hard-choice JSON prompt |
| `json_strict_nonzero` | Same information, but with stronger emphasis on JSON-only and hard-choice schema |
| `candidate_order_reversed` | Same information, but candidate/context order reversed |
| `interviewer_style` | Reworded in a survey interviewer style |
| `analyst_style` | Reworded in a more concise analyst-style hard-choice form |

All pairwise metrics use `base_json` as the reference.

## 3. Prechecks and Probes

All prechecks passed:

| Check | Result |
| --- | --- |
| `py_compile` | Passed |
| E03/parser target pytest | Passed |
| Ollama `/api/tags` | `qwen3.5:2b` visible |
| processed CES artifacts | All present |
| Initial GPU | RTX 5070 Laptop GPU, 8151 MiB total, about 6748 MiB free |

Probe results:

| Probe | States | Agents | Calls | Workers | LLM runtime | Median | P90 | Throughput | GPU peak used | Min free | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | PA, GA, MN, VA, CO | 10 | 50 | 1 | 27.2s | 0.429s | 0.453s | 1.84/s | 5002 MiB | 2890 MiB | PASS |
| B | 10 states | 30 | 150 | 4 | 38.0s | 0.935s | 1.075s | 3.94/s | 5002 MiB | 2890 MiB | PASS |
| C | 10 states | 30 | 150 | 5 | 37.6s | 1.146s | 1.332s | 3.99/s | 5002 MiB | 2890 MiB | PASS |

Workers=5 was only about 1% faster than workers=4, which did not meet the "at least 10% faster" adoption condition, and p90 latency was higher. Therefore the formal run used workers=4.

The probes had no:

| Exception | Result |
| --- | --- |
| parse failure | 0 |
| invalid choice | 0 |
| forbidden choice | 0 |
| legacy probability schema | 0 |
| transport error | 0 |
| GPU OOM | 0 |

## 4. Formal Run Overview

The formal run completed without interruption.

| Metric | Value |
| --- | ---: |
| Workers | 4 |
| Agents | 1000 |
| Prompt variants | 5 |
| Prompts / responses | 5000 |
| Ollama calls | 4995 |
| Cache hit rate | 0.001 |
| Total wall-clock | 1169.2s, about 19.5 min |
| LLM runtime | 1147.6s, about 19.1 min |
| Median latency | 0.874s |
| P90 latency | 0.932s |
| Throughput | 4.36 responses/s |
| GPU peak used | 5004 MiB |
| GPU min free | 2888 MiB |
| GPU peak utilization | 83% |

The formal run stayed below the 30-minute budget, with high GPU utilization and no approach to OOM.

## 5. Output Artifacts

Formal directory:

`data/runs/eval_suite_local/03_prompt_robustness_large_10state_1000`

Core outputs:

| File | Shape / Description |
| --- | --- |
| `agents.parquet` | 1000 x 32 |
| `prompts.parquet` | 5000 x 14 |
| `responses.parquet` | 5000 x 29 |
| `prompt_variant_metadata.parquet` | 5 x 2 |
| `robustness_metrics.parquet` | 55 x 7 |
| `pairwise_variant_metrics.parquet` | 48 x 8 |
| `runtime_log.parquet` | 5000 x 16 |
| `runtime.json` | Runtime observations |
| `llm_cache.jsonl` | LLM cache |
| `report.md` | Runner-generated report |
| `figures/*` | E03 figures |

Figures:

| Figure |
| --- |
| `e03_choice_flip_rate_vs_base.png` |
| `e03_parse_ok_by_variant.png` |
| `e03_state_margin_shift_by_variant.png` |
| `e03_turnout_choice_flip_rate_vs_base.png` |

## 6. Quality Gates

Overall runtime gates:

| Gate | Value | Result |
| --- | ---: | --- |
| parse_ok_rate | 1.000 | PASS |
| invalid_choice_rate | 0.000 | PASS |
| forbidden_choice_rate | 0.000 | PASS |
| legacy_probability_schema_rate | 0.000 | PASS |
| transport_error_rate | 0.000 | PASS |
| all_gates_passed | true | PASS |

By variant:

| Variant | Parse OK | Invalid choice | Legacy schema | Transport error | Median latency | Vote acc | Turnout brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_json` | 1.000 | 0.000 | 0.000 | 0.000 | 0.893 | 0.886 | 0.022 |
| `json_strict_nonzero` | 1.000 | 0.000 | 0.000 | 0.000 | 0.869 | 0.900 | 0.024 |
| `candidate_order_reversed` | 1.000 | 0.000 | 0.000 | 0.000 | 0.868 | 0.822 | 0.022 |
| `interviewer_style` | 1.000 | 0.000 | 0.000 | 0.000 | 0.864 | 0.900 | 0.022 |
| `analyst_style` | 1.000 | 0.000 | 0.000 | 0.000 | 0.866 | 0.881 | 0.022 |

The engineering conclusion is clear: E03's problem is not parse failure or JSON contract failure. Every variant strictly follows the hard-choice schema.

## 7. Choice Distribution by Variant

Overall choice distribution:

| Variant | N | Democrat | Republican | Not vote |
| --- | ---: | ---: | ---: | ---: |
| `base_json` | 1000 | 0.401 | 0.598 | 0.001 |
| `json_strict_nonzero` | 1000 | 0.415 | 0.570 | 0.015 |
| `candidate_order_reversed` | 1000 | 0.329 | 0.671 | 0.000 |
| `interviewer_style` | 1000 | 0.414 | 0.585 | 0.001 |
| `analyst_style` | 1000 | 0.385 | 0.615 | 0.000 |

Observations:

1. All variants are overall Republican-skewed.
2. `candidate_order_reversed` is clearly the most Republican-skewed, with Democrat share falling from base's 40.1% to 32.9%.
3. `interviewer_style` and `json_strict_nonzero` instead slightly increase Democrat share.
4. `json_strict_nonzero` is the only variant that noticeably increases `not_vote`, but only to 1.5%.
5. The turnout problem remains: every variant has a very low `not_vote` share.

## 8. Pairwise Robustness vs Base

Compared with `base_json`:

| Variant | Choice flip rate | Turnout choice flip rate |
| --- | ---: | ---: |
| `json_strict_nonzero` | 0.039 | 0.014 |
| `candidate_order_reversed` | 0.073 | 0.001 |
| `interviewer_style` | 0.021 | 0.000 |
| `analyst_style` | 0.017 | 0.001 |

Interpretation:

1. By hard-choice flip rate, all variants remain below the old 10% threshold.
2. `candidate_order_reversed` is the most sensitive variant, with choice flip rate 7.3%.
3. `interviewer_style` and `analyst_style` are relatively stable, with flip rates 2.1% and 1.7%.
4. `json_strict_nonzero` has a somewhat higher turnout choice flip because it outputs a few more `not_vote` choices.

If looking only at individual choice flip, E03-large would be judged "mostly robust." But aggregate margin shift shows that candidate order has a more serious effect than the flip rate suggests.

## 9. State Margin Shift

`state_margin_shift = variant_margin - base_json_margin`. Negative values mean the variant is more Republican-skewed than base.

| Variant | Max abs state margin shift | Mean abs state margin shift |
| --- | ---: | ---: |
| `json_strict_nonzero` | 0.081 | 0.038 |
| `candidate_order_reversed` | 0.398 | 0.128 |
| `interviewer_style` | 0.065 | 0.024 |
| `analyst_style` | 0.062 | 0.014 |

State-level details:

| Variant | State | Margin shift |
| --- | --- | ---: |
| `candidate_order_reversed` | AZ | -0.398 |
| `candidate_order_reversed` | MI | -0.160 |
| `candidate_order_reversed` | PA | -0.156 |
| `candidate_order_reversed` | NC | -0.141 |
| `candidate_order_reversed` | MN | -0.122 |
| `candidate_order_reversed` | WI | -0.115 |
| `candidate_order_reversed` | CO | -0.072 |
| `candidate_order_reversed` | GA | -0.055 |
| `candidate_order_reversed` | NV | -0.039 |
| `candidate_order_reversed` | VA | -0.024 |
| `json_strict_nonzero` | PA | 0.081 |
| `json_strict_nonzero` | CO | 0.071 |
| `json_strict_nonzero` | VA | 0.055 |
| `json_strict_nonzero` | MI | 0.054 |
| `interviewer_style` | NC | 0.065 |
| `interviewer_style` | VA | 0.053 |
| `analyst_style` | MI | -0.062 |

Core conclusions:

1. `candidate_order_reversed` is a clear source of prompt-order sensitivity.
2. Its margin shift is negative in all 10 states, meaning every state moves in the Republican direction.
3. The largest shift is AZ at -0.398, which is a very large aggregate shift.
4. In old E03, PA had a large -0.425 shift; after the larger sample, PA shift falls to -0.156, meaning the old PA extreme was partly amplified by small sample size, but the direction and sensitivity were not accidental.
5. New E03-large shows a broader problem: the candidate-order shift does not occur only in PA; all 10 states shift in the same Republican direction.

## 10. Group-Level Robustness

Decomposed by all 10 states, original 7 states, and added 3 states:

| Group | Variant | N agents | Choice flip | Turnout flip | Base margin | Variant margin | Margin shift |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all10 | `json_strict_nonzero` | 1000 | 0.039 | 0.014 | -0.284 | -0.251 | 0.033 |
| original7 | `json_strict_nonzero` | 700 | 0.036 | 0.016 | -0.315 | -0.292 | 0.023 |
| new3 | `json_strict_nonzero` | 300 | 0.047 | 0.010 | -0.216 | -0.164 | 0.053 |
| all10 | `candidate_order_reversed` | 1000 | 0.073 | 0.001 | -0.284 | -0.403 | -0.119 |
| original7 | `candidate_order_reversed` | 700 | 0.076 | 0.001 | -0.315 | -0.457 | -0.141 |
| new3 | `candidate_order_reversed` | 300 | 0.067 | 0.000 | -0.216 | -0.286 | -0.069 |
| all10 | `interviewer_style` | 1000 | 0.021 | 0.000 | -0.284 | -0.258 | 0.026 |
| original7 | `interviewer_style` | 700 | 0.020 | 0.000 | -0.315 | -0.293 | 0.022 |
| new3 | `interviewer_style` | 300 | 0.023 | 0.000 | -0.216 | -0.182 | 0.034 |
| all10 | `analyst_style` | 1000 | 0.017 | 0.001 | -0.284 | -0.297 | -0.013 |
| original7 | `analyst_style` | 700 | 0.020 | 0.001 | -0.315 | -0.329 | -0.014 |
| new3 | `analyst_style` | 300 | 0.010 | 0.000 | -0.216 | -0.228 | -0.012 |

Interpretation:

1. `candidate_order_reversed` moves in the Republican direction in both the original seven states and the added three states.
2. The shift is larger in the original seven states: -0.141; the added three states have a shift of -0.069.
3. `interviewer_style` and `json_strict_nonzero` move slightly toward Democrats, but by much less than the candidate-order shift.
4. `analyst_style` is almost equivalent to base, with an overall margin shift of only -0.013.

## 11. Comparison with Old E03

Old E03:

| Metric | Value |
| --- | ---: |
| Agents | 60 |
| Prompts | 300 |
| Workers | 4 |
| Runtime | 75.0s |
| Parse OK | 1.000 |

Old E03 pairwise choice flip:

| Variant | Old choice flip | Large choice flip |
| --- | ---: | ---: |
| `json_strict_nonzero` | 0.000 | 0.039 |
| `candidate_order_reversed` | 0.067 | 0.073 |
| `interviewer_style` | 0.000 | 0.021 |
| `analyst_style` | 0.000 | 0.017 |

Conclusion:

1. Old E03 already indicated that `candidate_order_reversed` was the most sensitive variant.
2. The larger sample confirms that conclusion: choice flip moves from 6.7% to 7.3%, which is very close.
3. Other variants looked completely stable in old E03, but the larger sample shows that they also have small 1.7%-3.9% flips.
4. The larger sample sharpens the conclusion: prompt wording is generally controllable, but candidate order produces a systematic aggregate shift.

## 12. Impact on E01/E02-Large Conclusions

This E03-large run does not overturn E01/E02-large; it adds an important mechanism:

1. The current model's main problem is still systematic Republican skew.
2. Prompt parsing itself is stable and is not the main failure source.
3. Candidate order in the prompt amplifies Republican skew, especially in `candidate_order_reversed`.
4. Interviewer/analyst wording itself has a smaller impact, showing that not every wording change causes instability.
5. All future formal prompts must fix candidate order and treat candidate-order sensitivity as a known risk in reports.

## 13. Conclusion

### 13.1 Engineering Conclusion

E03-large passes as an engineering pipeline:

1. All 5000 LLM calls parsed successfully.
2. No invalid choice, forbidden choice, legacy probability schema, or transport error.
3. workers=4 was stable, with formal wall-clock around 19.5 minutes, below the 30-minute budget.
4. GPU peak memory was about 5004 MiB, with minimum free memory about 2888 MiB, so there was no OOM risk.
5. Partial checkpoints and final artifacts are complete.

### 13.2 Scientific Conclusion

The current `qwen3.5:2b` hard-choice simulator is basically stable under ordinary wording changes, but clearly sensitive to candidate order.

More specifically:

1. `interviewer_style` and `analyst_style` differ little from base.
2. `json_strict_nonzero` slightly increases `not_vote` and Democrat share, but is still overall stable.
3. `candidate_order_reversed` is the main risk: choice flip is 7.3%, and margins in all 10 states move in the Republican direction.
4. This shows that the model is not simply reading the persona and making a stable classification; it is still systematically affected by prompt surface form.
5. For an MVP and a small model, this bias is expected, but reports must label it honestly and should not overinterpret aggregate results from a single prompt.

## 14. Recommendations

Future recommendations:

1. Fix and version the candidate order in formal prompts.
2. In final evaluation reports, treat E03's candidate-order sensitivity as a model risk, not an engineering bug.
3. If the system is improved later, consider counterbalanced prompting over candidate order, then aggregate or consistency-check outputs from both orders.
4. Continue prioritizing the Republican skew and the turnout problem exposed by E01/E02, where the model almost never outputs `not_vote`.

E03-large formally passes and the results are reusable. The conclusion is not "the system is broken," but that "a small-model MVP has measurable, explainable bias in prompt surface form."
