# LM Voting Simulator Large-Suite Evaluation Report

Date: 2026-05-01  
Primary evaluation root: `data/runs/eval_suite_local`  
Primary model: `qwen3.5:2b` via Windows Ollama at `http://172.26.48.1:11434`  
Response contract: hard-choice JSON, `{"choice": "not_vote|democrat|republican"}`  

本报告总结第二轮大规模 E01-E06 实验，并把第一轮小规模 E01-E06 作为辅助参照。旧总报告 `documents/lm_voting_simulator_evaluation_report.md` 保留不覆盖。

## 1. Executive Summary

第二轮大规模实验的结论比第一轮更清楚。

工程层面，系统已经比较稳定。E01-large 到 E06-large 全部完成，质量门全部通过；LLM 输出 100% 遵守 hard-choice contract，没有 invalid choice、forbidden choice、legacy probability schema 或 transport error。并发运行稳定，GPU 没有 OOM。E06-large 明确没有调用 Ollama，只复用前序产物做本地表分析。

科学层面，当前 `qwen3.5:2b` 不能被视为可靠的 voter simulator。它可以稳定输出格式正确的 hard choice，也能在党派/意识形态明确时恢复个体投票方向；但它并没有稳定地从 rich survey memory 中做细粒度 persona simulation。它最可靠的信息源是 party/ideology。加入候选人、年份、州名、candidate context 或 strict memory context 后，模型经常出现强 Republican skew，尤其伤害 Lean Democrat、Independent、Black respondents、CO/VA/MN 等 Democratic-leaning subgroup 或州。

最重要的跨实验结论：

- hard-choice 契约是正确的工程修复，彻底避免了旧概率 JSON 的归一化和瞎写概率问题。
- large runs 没有发现解析、schema、并发或缓存层面的致命问题。
- `party/ideology` 是当前小模型最有效的非 oracle LLM 条件。
- `strict memory` 本身有一定信息，但在当前 prompt 设计里没有稳定超过 party/ideology。
- `candidate/context` 语境会显著放大 Republican collapse。
- E02-large 加入 MN/VA/CO 后，LLM strict 不能再靠“七个摇摆州全是 Republican winner”掩盖问题；它把 MN/VA/CO 都预测成 Republican。
- E03-large 确认 candidate order 不是小样本偶然，`candidate_order_reversed` 仍是最敏感 prompt variant。
- E04-large 显示匿名候选人时模型接近平衡，但一旦出现党派/候选人/年份语境，输出强烈转向 Republican。
- E05-large 的 full ladder 说明：party/ideology 带来最大增益，candidate context 带来最大伤害，oracle 可以恢复正确答案，placebo memory 基本无效。
- E06-large 把总体问题定位到具体 subgroup：Independent、Lean Democrat、Black respondents、Democratic-winning states 的错误最突出。

总体判断：这套评估框架已经可以作为系统性实验平台继续使用；但当前 `qwen3.5:2b` 只能作为 MVP 小模型基线，不能作为最终模拟器模型。

## 2. Response Contract and Interpretation

本轮所有 CES turnout-vote LLM 任务都使用三分类 hard-choice contract：

```json
{"choice": "not_vote|democrat|republican"}
```

系统再派生兼容列：

- `turnout_probability = 1` iff `choice` 是 `democrat` 或 `republican`
- `turnout_probability = 0` iff `choice` 是 `not_vote`
- `vote_prob_democrat = 1` iff `choice == "democrat"`
- `vote_prob_republican = 1` iff `choice == "republican"`
- `vote_prob_other = 0`
- `vote_prob_undecided = 0`

这些概率列不是 LLM 主观概率。它们只是 hard choice 的 one-hot 派生值。报告里所有 calibration / Brier / entropy 诊断都按这个原则解释，不能解读为“模型概率是否校准”。

## 3. Experiment Inventory

### 3.1 Large formal runs

| Experiment | Directory | Main scale |
| --- | --- | --- |
| E01-large Individual Persona Fidelity | `data/runs/eval_suite_local/01_individual_persona_large_10state_1000` | 10 states, 1000 LLM agents, 3 LLM baselines, 3000 LLM calls |
| E02-large Aggregate Election Accuracy | `data/runs/eval_suite_local/02_aggregate_accuracy_large_10state_500` | 10 states, LLM max 500/state, 5000 LLM calls |
| E03-large Prompt Robustness | `data/runs/eval_suite_local/03_prompt_robustness_large_10state_1000` | 10 states, 1000 agents, 5 prompt variants, 5000 LLM calls |
| E04-large Historical / World-Knowledge Leakage | `data/runs/eval_suite_local/04_leakage_large_10state_70` | 10 states, 70 agents/state, 7 conditions, 4900 LLM calls |
| E05-large Information Ablation and Placebo Memory | `data/runs/eval_suite_local/05_ablation_placebo_large_10state_full_ladder_600` | 10 states, 600 agents, 10 baselines, 6000 LLM calls |
| E06-large Subgroup and Calibration Reliability | `data/runs/eval_suite_local/06_subgroup_calibration_large_10state` | Reuses E01/E05-large, 19,844 responses, 0 LLM calls |

### 3.2 Large standalone reports

- `documents/lm_voting_simulator_e01_large_10state_1000_report.md`
- `documents/lm_voting_simulator_e02_large_10state_500_report.md`
- `documents/lm_voting_simulator_e03_large_10state_1000_report.md`
- `documents/lm_voting_simulator_e04_large_10state_70_report.md`
- `documents/lm_voting_simulator_e05_large_10state_full_ladder_600_report.md`
- `documents/lm_voting_simulator_e06_large_10state_report.md`

### 3.3 Previous small formal runs used as secondary reference

- E01-small: `data/runs/eval_suite_local/01_individual_persona`
- E02-small: `data/runs/eval_suite_local/02_aggregate_accuracy`
- E03-small: `data/runs/eval_suite_local/03_prompt_robustness`
- E04-small: `data/runs/eval_suite_local/04_leakage`
- E05-small: `data/runs/eval_suite_local/05_ablation_placebo`
- E06-small: `data/runs/eval_suite_local/06_subgroup_calibration`

## 4. Runtime and Quality Overview

### 4.1 Large runtime

| Experiment | Workers | LLM tasks / responses | Wall-clock | LLM time | Median latency | P90 latency | Throughput | GPU peak used | Min GPU free | Gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E01-large | 4 | 3000 | 732.14s | 581.94s | 0.686s | 0.899s | 5.16/s | 5054 MiB | 2838 MiB | PASS |
| E02-large | 4 | 5000 | 1381.53s | 1213.14s | 0.897s | 0.967s | 4.12/s | 4998 MiB | 2894 MiB | PASS |
| E03-large | 4 | 5000 | 1169.16s | 1147.55s | 0.874s | 0.932s | 4.36/s | 5004 MiB | 2888 MiB | PASS |
| E04-large | 3 | 4900 | 1201.95s | 1179.86s | 0.673s | 0.742s | 4.15/s | 5164 MiB | 2728 MiB | PASS |
| E05-large | 3 | 6000 | 1105.14s | 1027.39s | 0.642s | 0.719s | 5.84/s | 5162 MiB | 2730 MiB | PASS |
| E06-large | 6 | 19,844 local rows | 47.28s | 0 | n/a | n/a | n/a | n/a | n/a | PASS |

E01-E06-large 总墙钟约 93.95 分钟，其中 E06 没有 LLM 调用。所有 LLM large runs 都控制在 30 分钟以内。

### 4.2 Small runtime reference

| Experiment | Workers | LLM tasks / responses | Wall-clock | Median latency | P90 latency | GPU peak used | Gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E01-small | 3 | 300 | 175.48s | 0.545s | 0.700s | 4561 MiB | PASS |
| E02-small | 4 | 490 | 287.27s | 0.979s | 1.025s | 4561 MiB | PASS |
| E03-small | 4 | 300 | 74.98s | 0.950s | 1.007s | 4563 MiB | PASS |
| E04-small | 3 | 420 | 142.27s | 0.847s | 0.896s | 4839 MiB | PASS |
| E05-small | 3 | 420 | 165.89s | 0.755s | 0.783s | 4902 MiB | PASS |
| E06-small | 4 | 6144 local rows | 16.99s | n/a | n/a | n/a | PASS |

Large runs 的速度明显好于最初预估。GPU 峰值约 5.0-5.2 GiB，最低剩余显存约 2.7-2.9 GiB，说明 3-4 workers 是稳定高吞吐配置。

### 4.3 Quality gates

所有 large formal runs 的解析质量门都通过：

- `parse_ok_rate = 1.0`
- `invalid_choice_rate = 0`
- `forbidden_choice_rate = 0`
- `legacy_probability_schema_rate = 0`
- `transport_error_rate = 0`

这意味着本轮结论主要是模型行为结论，不是格式失败、并发失败或解析失败造成的假象。

## 5. E01-Large: Individual Persona Fidelity

### 5.1 Goal and setup

E01-large 检查 LLM agent 能否在个体层面复现 CES 受访者自己的 2024 turnout/vote hard choice。

设计：

- States: PA, MI, WI, GA, AZ, NV, NC, MN, VA, CO
- LLM agents: 1000
- LLM baselines:
  - `ces_demographic_only_llm`
  - `ces_party_ideology_llm`
  - `ces_survey_memory_llm_strict`
- Non-LLM baselines:
  - `party_id_baseline`
  - `sklearn_logit_demographic_only`
  - `sklearn_logit_pre_only`
  - `sklearn_logit_poll_informed`

### 5.2 Individual metrics

| Baseline | Parse OK | Vote accuracy | Vote macro F1 | Turnout accuracy | Turnout Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sklearn_logit_poll_informed` | 1.000 | 0.977 | 0.683 | 0.954 | 0.046 |
| `sklearn_logit_pre_only` | 1.000 | 0.974 | 0.682 | 0.946 | 0.054 |
| `ces_party_ideology_llm` | 1.000 | 0.914 | 0.611 | 0.947 | 0.053 |
| `party_id_baseline` | 1.000 | 0.896 | 0.636 | 0.948 | 0.052 |
| `ces_survey_memory_llm_strict` | 1.000 | 0.871 | 0.582 | 0.947 | 0.053 |
| `sklearn_logit_demographic_only` | 1.000 | 0.633 | 0.420 | 0.944 | 0.056 |
| `ces_demographic_only_llm` | 1.000 | 0.551 | 0.363 | 0.947 | 0.053 |

### 5.3 Interpretation

E01-large 的排序很清楚：

1. 统计模型 `sklearn_logit_pre_only` / `poll_informed` 最强。
2. LLM 里 `party/ideology` 最强。
3. `strict memory` 低于 `party/ideology`。
4. demographics-only 很弱。

与 E01-small 对照：

- E01-small 中 `party/ideology` 和 `strict memory` 几乎并列，vote accuracy 都约 0.977。
- E01-large 中两者分开：`party/ideology=0.914`，`strict memory=0.871`。

这说明小样本中 strict memory 看起来“不差”，但扩大到 10 州 1000 agents 后，它没有稳定增益。当前小模型主要利用党派/意识形态，而不是 rich survey memory。

## 6. E02-Large: Aggregate Election Accuracy

### 6.1 Goal and setup

E02-large 检查 state-level 聚合能否复现 2024 总统选举 Democratic two-party share、margin 和 winner。

设计：

- States: PA, MI, WI, GA, AZ, NV, NC, MN, VA, CO
- 原 7 州均为 2024 Republican winner
- 新增 MN/VA/CO 为 2024 Democratic winner
- LLM baseline: `survey_memory_llm_strict`
- LLM max sample size: 500 per state
- LLM calls: 5000

### 6.2 Aggregate metrics at sample size 500

| Baseline | Dem 2P RMSE | Margin MAE | Margin bias | Winner accuracy | Winner flips | Corr true margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `uniform_national_swing_from_2020` | 0.0148 | 0.0259 | -0.0259 | 1.000 | 0 | 0.956 |
| `mit_2020_state_prior` | 0.0187 | 0.0345 | +0.0345 | 0.400 | 6 | 0.956 |
| `ces_post_self_report_aggregate_oracle` | 0.0307 | 0.0555 | -0.0386 | 0.700 | 3 | 0.740 |
| `sklearn_logit_poll_informed` | 0.0479 | 0.0795 | -0.0728 | 0.800 | 2 | 0.535 |
| `sklearn_logit_pre_only_crossfit` | 0.0469 | 0.0815 | -0.0732 | 0.800 | 2 | 0.647 |
| `party_id_baseline` | 0.0735 | 0.1264 | +0.1245 | 0.400 | 6 | 0.626 |
| `survey_memory_llm_strict` | 0.1623 | 0.3136 | -0.3136 | 0.700 | 3 | 0.286 |

### 6.3 LLM state-level results at sample size 500

| State | Pred Dem 2P | True Dem 2P | Error | Pred winner | True winner | Correct |
| --- | ---: | ---: | ---: | --- | --- | --- |
| AZ | 0.326 | 0.472 | -0.146 | Republican | Republican | True |
| CO | 0.342 | 0.556 | -0.215 | Republican | Democrat | False |
| GA | 0.317 | 0.489 | -0.172 | Republican | Republican | True |
| MI | 0.318 | 0.493 | -0.175 | Republican | Republican | True |
| MN | 0.424 | 0.522 | -0.097 | Republican | Democrat | False |
| NC | 0.282 | 0.484 | -0.202 | Republican | Republican | True |
| NV | 0.363 | 0.484 | -0.121 | Republican | Republican | True |
| PA | 0.409 | 0.491 | -0.083 | Republican | Republican | True |
| VA | 0.347 | 0.530 | -0.182 | Republican | Democrat | False |
| WI | 0.320 | 0.496 | -0.176 | Republican | Republican | True |

LLM choice distribution in E02-large:

| Choice | Share |
| --- | ---: |
| Republican | 0.5990 |
| Democrat | 0.3998 |
| Not vote | 0.0012 |

### 6.4 Interpretation

E02-small 在 7 个全 Republican winner 州上得到 LLM winner accuracy 1.0，但 margin MAE 很差。E02-large 加入 MN/VA/CO 后，问题暴露：LLM strict 把 MN、VA、CO 也预测成 Republican，winner accuracy 降到 0.7，margin MAE 仍高达 0.314。

结论：LLM strict 不是可靠 aggregate predictor。它有较强 Republican skew，无法稳定恢复 Democratic-winning states。

## 7. E03-Large: Prompt Robustness

### 7.1 Goal and setup

E03-large 检查同一 persona、同一 strict memory 下，prompt wording 是否改变 hard-choice 输出。

设计：

- States: 10
- Agents: 1000
- Variants: 5
- LLM calls: 5000

Prompt variants:

- `base_json`
- `json_strict_nonzero`
- `candidate_order_reversed`
- `interviewer_style`
- `analyst_style`

### 7.2 Metrics by variant

| Prompt variant | Parse OK | Vote accuracy | Turnout Brier | Median latency | Democrat | Republican | Not vote |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `analyst_style` | 1.000 | 0.881 | 0.022 | 0.866s | 0.385 | 0.615 | 0.000 |
| `base_json` | 1.000 | 0.886 | 0.022 | 0.894s | 0.401 | 0.598 | 0.001 |
| `candidate_order_reversed` | 1.000 | 0.822 | 0.022 | 0.868s | 0.329 | 0.671 | 0.000 |
| `interviewer_style` | 1.000 | 0.900 | 0.022 | 0.864s | 0.414 | 0.585 | 0.001 |
| `json_strict_nonzero` | 1.000 | 0.900 | 0.024 | 0.869s | 0.415 | 0.570 | 0.015 |

Pairwise against `base_json`:

| Variant | Choice flip rate | Turnout choice flip rate |
| --- | ---: | ---: |
| `analyst_style` | 0.017 | 0.001 |
| `candidate_order_reversed` | 0.073 | 0.001 |
| `interviewer_style` | 0.021 | 0.000 |
| `json_strict_nonzero` | 0.039 | 0.014 |

Largest state margin shifts:

| Variant | State | Margin shift |
| --- | --- | ---: |
| `candidate_order_reversed` | AZ | -0.398 |
| `candidate_order_reversed` | MI | -0.160 |
| `candidate_order_reversed` | PA | -0.156 |
| `candidate_order_reversed` | NC | -0.141 |
| `candidate_order_reversed` | MN | -0.122 |

### 7.3 Interpretation

E03-large confirms E03-small:

- Formatting and style variants are mostly stable.
- `candidate_order_reversed` remains the sensitive condition.
- Hard-choice parsing is robust; prompt sensitivity is semantic/model behavior, not schema failure.

Candidate order changes can shift aggregate margins materially even when overall flip rate is only 7.3%. Future serious experiments should balance candidate order or explicitly average across order variants.

## 8. E04-Large: Historical / World-Knowledge Leakage

### 8.1 Goal and setup

E04-large checks whether model behavior depends on candidate names, party labels, state labels, year labels, or other world-knowledge shortcuts.

Design:

- States: PA, MI, WI, GA, AZ, NV, NC, MN, VA, CO
- Agents: 700
- Conditions: 7
- LLM calls: 4900

Conditions:

- `named_candidates`
- `party_only_candidates`
- `anonymous_candidates`
- `masked_year`
- `masked_state`
- `state_swap_placebo`
- `candidate_swap_placebo`

### 8.2 Aggregate metrics

| Condition | Dem 2P RMSE | Margin MAE | Margin bias | Winner accuracy | Winner flips | Corr true margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `anonymous_candidates` | 0.074 | 0.133 | +0.049 | 0.500 | 5 | 0.537 |
| `named_candidates` | 0.446 | 0.881 | -0.881 | 0.700 | 3 | -0.279 |
| `state_swap_placebo` | 0.459 | 0.907 | -0.907 | 0.700 | 3 | 0.002 |
| `masked_state` | 0.469 | 0.933 | -0.933 | 0.700 | 3 | -0.228 |
| `party_only_candidates` | 0.496 | 0.990 | -0.990 | 0.700 | 3 | 0.317 |
| `masked_year` | 0.500 | 0.998 | -0.998 | 0.700 | 3 | -0.145 |
| `candidate_swap_placebo` | 0.502 | 1.003 | -1.003 | 0.700 | 3 | n/a |

### 8.3 Choice distribution

| Condition | Democrat | Republican | Not vote |
| --- | ---: | ---: | ---: |
| `anonymous_candidates` | 0.531 | 0.469 | 0.000 |
| `candidate_swap_placebo` | 0.000 | 1.000 | 0.000 |
| `masked_state` | 0.049 | 0.951 | 0.000 |
| `masked_year` | 0.003 | 0.997 | 0.000 |
| `named_candidates` | 0.070 | 0.930 | 0.000 |
| `party_only_candidates` | 0.010 | 0.990 | 0.000 |
| `state_swap_placebo` | 0.063 | 0.937 | 0.000 |

### 8.4 Leakage diagnostics

Important contrasts:

- `anonymous_candidates` is far less collapsed and has much lower margin MAE than `named_candidates`.
- `party_only_candidates`, `masked_year`, and `candidate_swap_placebo` are essentially all Republican.
- `named_candidates` is 93% Republican.
- `candidate_swap_placebo` is 100% Republican, so candidate swap does not produce healthy name-following behavior.
- State swap does not track true state shifts well. The aggregate `pred_shift_vs_truth_shift` diagnostic shows weak/no useful state-prior response.

Candidate swap diagnostics:

| State | named pred Dem 2P | candidate-swap pred Dem 2P | party following | name following | name-minus-party index |
| --- | ---: | ---: | ---: | ---: | ---: |
| AZ | 0.158 | 0.000 | 0.842 | 0.158 | -0.684 |
| CO | 0.024 | 0.000 | 0.976 | 0.024 | -0.952 |
| GA | 0.006 | 0.000 | 0.994 | 0.006 | -0.989 |
| MI | 0.024 | 0.000 | 0.976 | 0.024 | -0.951 |
| MN | 0.061 | 0.000 | 0.939 | 0.061 | -0.879 |
| NC | 0.006 | 0.000 | 0.994 | 0.006 | -0.988 |
| NV | 0.035 | 0.000 | 0.965 | 0.035 | -0.930 |
| PA | 0.179 | 0.000 | 0.821 | 0.179 | -0.642 |
| VA | 0.052 | 0.000 | 0.948 | 0.052 | -0.896 |
| WI | 0.065 | 0.000 | 0.935 | 0.065 | -0.871 |

### 8.5 Interpretation

E04-large is the strongest leakage/prompt shortcut warning. Anonymous candidates produce a roughly balanced distribution and much lower margin error. Once party/candidate/year framing enters, outputs collapse toward Republican. This is not simply “2024 was Republican-favorable”; the problem is that predicted margins become extreme and Democratic-winning states are not recovered.

E04-small already showed the same direction. E04-large confirms it at 10-state scale.

## 9. E05-Large: Information Ablation and Placebo Memory

### 9.1 Goal and setup

E05-large tests which information sources actually help, and whether real memory beats placebo memory.

Design:

- States: 10
- Agents: 600
- Baselines: 10
- LLM calls: 6000

Full ladder:

- `L1_demographic_only_llm`
- `L2_demographic_state_llm`
- `L3_party_ideology_llm`
- `L4_party_ideology_context_llm`
- `L5_strict_memory_llm`
- `L6_strict_memory_context_llm`
- `L7_poll_informed_memory_context_llm`
- `L8_post_hoc_oracle_memory_context_llm`
- `P1_memory_shuffled_within_state_llm`
- `P2_memory_shuffled_within_party_llm`

### 9.2 Main-sample individual metrics

| Baseline | Parse OK | Vote accuracy | Vote macro F1 | Turnout accuracy | Turnout Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| `L8_post_hoc_oracle_memory_context_llm` | 1.000 | 0.987 | 0.658 | 0.996 | 0.004 |
| `L3_party_ideology_llm` | 1.000 | 0.893 | 0.596 | 0.985 | 0.015 |
| `L5_strict_memory_llm` | 1.000 | 0.806 | 0.531 | 0.987 | 0.013 |
| `L7_poll_informed_memory_context_llm` | 1.000 | 0.610 | 0.354 | 0.979 | 0.021 |
| `L6_strict_memory_context_llm` | 1.000 | 0.576 | 0.313 | 0.987 | 0.013 |
| `L4_party_ideology_context_llm` | 1.000 | 0.546 | 0.274 | 0.987 | 0.013 |
| `P2_memory_shuffled_within_party_llm` | 1.000 | 0.518 | 0.235 | 0.987 | 0.013 |
| `P1_memory_shuffled_within_state_llm` | 1.000 | 0.516 | 0.232 | 0.987 | 0.013 |
| `L2_demographic_state_llm` | 1.000 | 0.512 | 0.226 | 0.987 | 0.013 |
| `L1_demographic_only_llm` | 1.000 | 0.512 | 0.226 | 0.987 | 0.013 |

### 9.3 Aggregate metrics

| Baseline | Dem 2P RMSE | Margin MAE | Margin bias | Winner accuracy | Winner flips | Corr true margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `L8_post_hoc_oracle_memory_context_llm` | 0.114 | 0.184 | -0.113 | 0.600 | 4 | 0.595 |
| `L3_party_ideology_llm` | 0.139 | 0.244 | -0.244 | 0.700 | 3 | 0.525 |
| `L5_strict_memory_llm` | 0.257 | 0.496 | -0.496 | 0.700 | 3 | 0.538 |
| `L7_poll_informed_memory_context_llm` | 0.409 | 0.808 | -0.808 | 0.700 | 3 | 0.320 |
| `L6_strict_memory_context_llm` | 0.448 | 0.885 | -0.885 | 0.700 | 3 | -0.315 |
| `L4_party_ideology_context_llm` | 0.478 | 0.953 | -0.953 | 0.700 | 3 | 0.309 |
| `P1_memory_shuffled_within_state_llm` | 0.493 | 0.984 | -0.984 | 0.700 | 3 | 0.157 |
| `P2_memory_shuffled_within_party_llm` | 0.498 | 0.995 | -0.995 | 0.700 | 3 | 0.191 |
| `L2_demographic_state_llm` | 0.502 | 1.003 | -1.003 | 0.700 | 3 | n/a |
| `L1_demographic_only_llm` | 0.502 | 1.003 | -1.003 | 0.700 | 3 | n/a |

### 9.4 Choice distribution

| Baseline | Democrat | Republican | Not vote |
| --- | ---: | ---: | ---: |
| `L1_demographic_only_llm` | 0.000 | 1.000 | 0.000 |
| `L2_demographic_state_llm` | 0.000 | 1.000 | 0.000 |
| `L3_party_ideology_llm` | 0.342 | 0.653 | 0.005 |
| `L4_party_ideology_context_llm` | 0.028 | 0.972 | 0.000 |
| `L5_strict_memory_llm` | 0.240 | 0.760 | 0.000 |
| `L6_strict_memory_context_llm` | 0.052 | 0.948 | 0.000 |
| `L7_poll_informed_memory_context_llm` | 0.080 | 0.910 | 0.010 |
| `L8_post_hoc_oracle_memory_context_llm` | 0.437 | 0.498 | 0.065 |
| `P1_memory_shuffled_within_state_llm` | 0.005 | 0.995 | 0.000 |
| `P2_memory_shuffled_within_party_llm` | 0.005 | 0.995 | 0.000 |

### 9.5 Key deltas

| Delta | From -> To | Vote accuracy delta | Macro F1 delta | Margin MAE delta |
| --- | --- | ---: | ---: | ---: |
| State increment | L1 -> L2 | +0.000 | +0.000 | +0.000 |
| Party/ideology increment | L2 -> L3 | +0.388 | +0.386 | -0.760 |
| Candidate context increment | L3 -> L4 | -0.368 | -0.357 | +0.710 |
| Strict memory increment over party/context | L4 -> L6 | +0.043 | +0.060 | -0.068 |
| Strict context increment | L5 -> L6 | -0.213 | -0.210 | +0.389 |
| Poll increment | L6 -> L7 | +0.044 | +0.053 | -0.076 |
| State-placebo gap | P1 -> L6 | +0.059 | +0.084 | -0.099 |
| Party-placebo gap | P2 -> L6 | +0.059 | +0.083 | -0.110 |
| Oracle gap from strict | L6 -> L8 | +0.393 | +0.339 | -0.701 |
| Oracle gap from poll | L7 -> L8 | +0.350 | +0.286 | -0.625 |

Negative `margin_mae` delta means improvement; positive means worse.

### 9.6 Interpretation

E05-large is the most informative causal ladder:

- Adding state alone does nothing.
- Adding party/ideology is the biggest non-oracle gain.
- Adding candidate context is the biggest harm.
- Strict memory without context (`L5`) is useful but below `L3`.
- Strict memory with candidate context (`L6`) is much worse than `L5`.
- Poll-informed memory/context (`L7`) slightly improves over `L6`, but remains very Republican-skewed and far below `L3`.
- Real memory beats shuffled memory slightly, so memory contains some usable signal.
- Oracle proves the pipeline can recover answers when target-relevant information is explicitly available.

E05-small showed the same broad conclusion; E05-large makes it stronger and more granular by adding L2/L4/L5.

## 10. E06-Large: Subgroup and Calibration Reliability

### 10.1 Goal and setup

E06-large reuses E01-large and E05-large to evaluate subgroup reliability, distribution compression, and deterministic hard-choice calibration.

It made zero LLM calls.

Formal runtime:

- Responses analyzed: 19,844
- Subgroup rows: 1,991
- Distribution rows: 1,991
- Calibration rows: 23
- Workers: 6
- Runtime: 47.28s
- Gates: PASS

### 10.2 Overall reliability rows

| Source | Baseline | n | Vote acc | Macro F1 | Turnout Brier | Pred Dem 2P | True Dem 2P | Dem 2P error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E01 | `ces_demographic_only_llm` | 1000 | 0.518 | 0.338 | 0.032 | 0.405 | 0.466 | -0.061 |
| E01 | `ces_party_ideology_llm` | 1000 | 0.917 | 0.611 | 0.032 | 0.374 | 0.466 | -0.092 |
| E01 | `ces_survey_memory_llm_strict` | 1000 | 0.871 | 0.578 | 0.032 | 0.309 | 0.466 | -0.157 |
| E05 | `L1_demographic_only_llm` | 600 | 0.539 | 0.234 | 0.050 | 0.000 | 0.461 | -0.461 |
| E05 | `L3_party_ideology_llm` | 600 | 0.911 | 0.609 | 0.056 | 0.362 | 0.461 | -0.098 |
| E05 | `L4_party_ideology_context_llm` | 600 | 0.557 | 0.261 | 0.050 | 0.019 | 0.461 | -0.442 |
| E05 | `L5_strict_memory_llm` | 600 | 0.789 | 0.513 | 0.050 | 0.229 | 0.461 | -0.232 |
| E05 | `L6_strict_memory_context_llm` | 600 | 0.596 | 0.315 | 0.050 | 0.059 | 0.461 | -0.402 |
| E05 | `L7_poll_informed_memory_context_llm` | 600 | 0.635 | 0.365 | 0.055 | 0.087 | 0.461 | -0.373 |
| E05 | `L8_post_hoc_oracle_memory_context_llm` | 600 | 0.993 | 0.662 | 0.008 | 0.445 | 0.461 | -0.016 |

### 10.3 Distribution diagnostics

| Source | Baseline | Pred Dem | Pred Rep | Pred not_vote | Entropy ratio | Variance ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E01 | `ces_party_ideology_llm` | 0.374 | 0.626 | 0.000 | 0.947 | 0.941 |
| E01 | `ces_survey_memory_llm_strict` | 0.309 | 0.690 | 0.001 | 0.896 | 0.858 |
| E05 | `L3_party_ideology_llm` | 0.357 | 0.628 | 0.015 | 1.047 | 0.924 |
| E05 | `L4_party_ideology_context_llm` | 0.019 | 0.981 | 0.000 | 0.135 | 0.074 |
| E05 | `L5_strict_memory_llm` | 0.229 | 0.771 | 0.000 | 0.780 | 0.711 |
| E05 | `L6_strict_memory_context_llm` | 0.059 | 0.941 | 0.000 | 0.324 | 0.222 |
| E05 | `L7_poll_informed_memory_context_llm` | 0.087 | 0.905 | 0.009 | 0.497 | 0.318 |
| E05 | `L8_post_hoc_oracle_memory_context_llm` | 0.394 | 0.492 | 0.114 | 1.396 | 0.961 |

### 10.4 Worst subgroup patterns

Key non-small worst subgroup examples:

| Baseline | Worst subgroup | n | Vote accuracy | Dem 2P error |
| --- | --- | ---: | ---: | ---: |
| `ces_party_ideology_llm` | `ideology_3=unknown` | 41 | 0.397 | -0.720 |
| `ces_survey_memory_llm_strict` | `state_po_x_party_id_3=CO x independent_or_other` | 34 | 0.313 | -0.698 |
| `L3_party_ideology_llm` | `party_id_7=Independent` | 76 | 0.481 | -0.519 |
| `L4_party_ideology_context_llm` | `party_id_7=Lean Democrat` | 77 | 0.019 | -0.981 |
| `L5_strict_memory_llm` | `party_id_7=Lean Democrat` | 77 | 0.093 | -0.914 |
| `L6_strict_memory_context_llm` | `party_id_7=Lean Democrat` | 77 | 0.019 | -0.981 |
| `L7_poll_informed_memory_context_llm` | `party_id_7=Lean Democrat` | 77 | 0.019 | -0.981 |
| `L8_post_hoc_oracle_memory_context_llm` | `state_po_x_race_ethnicity=CO x white` | 40 | 0.948 | -0.077 |

Race/ethnicity stress test:

- `L3` Black subgroup: vote accuracy 0.900, Dem 2P error -0.134.
- `L4` Black subgroup: vote accuracy 0.183, Dem 2P error -0.820.
- `L6` Black subgroup: vote accuracy 0.233, Dem 2P error -0.719.
- `L8` Black subgroup: vote accuracy 0.996, Dem 2P error -0.004.

State-group comparison:

- E05 `L3`: old7 mean abs Dem 2P error 0.080, new MN/VA/CO 0.138.
- E05 `L6`: old7 0.364, new MN/VA/CO 0.500.
- E05 `L7`: old7 0.354, new MN/VA/CO 0.412.

### 10.5 Interpretation

E06-large shows that the system's problems are concentrated, not random. Independent/other, Lean Democrat, Black subgroup, and Democratic-winning states are where Republican skew becomes most damaging. Oracle condition works, so these failures are model/prompt behavior rather than data-contract failure.

## 11. Cross-Experiment Synthesis

### 11.1 Engineering system

The framework is now substantially stronger than at the start:

- Hard-choice response contract eliminated meaningless LLM-written probability vectors.
- Runtime logs, cache, partial checkpoints, raw responses, parsed responses, reports, and figures are preserved.
- Windows Ollama through WSL is stable.
- 3-4 LLM workers give good throughput without OOM.
- Quality gates catch the right failure modes.
- E06 can reuse prior artifacts without additional LLM cost.

This is a good MVP evaluation harness.

### 11.2 Model behavior

The model is not a reliable simulator.

The strongest useful LLM condition is party/ideology:

- E01-large `ces_party_ideology_llm`: vote accuracy 0.914.
- E05-large `L3_party_ideology_llm`: vote accuracy 0.893, margin MAE 0.244.

Strict memory is mixed:

- E01-large strict memory: vote accuracy 0.871, below party/ideology.
- E05-large `L5_strict_memory_llm`: vote accuracy 0.806, below L3 but above context-heavy variants.
- E05-large `L6_strict_memory_context_llm`: vote accuracy 0.576, much worse than L5.

Candidate/context is dangerous:

- E05 `L3 -> L4`: vote accuracy -0.368, margin MAE +0.710.
- E05 `L5 -> L6`: vote accuracy -0.213, margin MAE +0.389.
- E04 named/party/year/candidate-swap conditions mostly collapse toward Republican.

### 11.3 Aggregate winner accuracy can mislead

Small E02 used seven states that were all Republican wins in 2024. The strict-memory LLM got all seven winners right, but with very large Republican margin bias. Large E02 added MN/VA/CO; the model then missed all three Democratic-winning states.

Therefore, margin MAE, Dem 2P RMSE, margin bias, and subgroup behavior are more informative than winner accuracy alone.

### 11.4 Turnout remains under-modeled

Most LLM conditions predict turnout almost always. Turnout Brier can look good because the sampled CES cohorts are high-turnout. E06 calibration confirms this is deterministic reliability, not probabilistic calibration. The current simulator is primarily a vote-choice simulator, not a robust turnout simulator.

### 11.5 Previous small suite vs large suite

Large runs did not overturn the small-suite conclusions; they sharpened them.

| Theme | Small suite | Large suite |
| --- | --- | --- |
| Contract stability | Hard-choice contract passed | Confirmed at ~23.9k LLM calls |
| Party/ideology | Best non-oracle signal | Confirmed, strongest non-oracle condition |
| Strict memory | No clear stable gain | Below party/ideology; context variant harmful |
| Candidate order | Sensitive in E03-small | Confirmed with 1000 agents |
| Leakage / shortcut behavior | Republican collapse in named/party conditions | Confirmed across 10 states, including Democratic states |
| Aggregate winner accuracy | Misleading in all-R swing-state set | MN/VA/CO expose false Republican calls |
| Subgroup reliability | Failures visible but noisy | Failures localized to Independent, Lean Democrat, Black subgroup, Democratic-winning states |

## 12. Overall Verdict

### 12.1 What works

The evaluation framework works:

- It can run controlled experiments at larger scale.
- It can enforce schema and quality gates.
- It can produce reusable artifacts.
- It can detect prompt sensitivity, leakage, ablation effects, and subgroup failures.
- It can distinguish model behavior failures from software failures.

### 12.2 What does not work yet

The current `qwen3.5:2b` simulator does not yet meet the bar for substantive election simulation:

- It is too dependent on party/ideology labels.
- It does not consistently use survey memory in a helpful way.
- It has strong Republican skew under candidate/context prompts.
- It misses Democratic-winning states in aggregate.
- It fails badly for some Democratic-leaning and minority subgroups.
- It does not model turnout in a meaningful way.

### 12.3 Practical recommendation

For future experiments, use the current system as an evaluation harness, not as a finished simulator. The next scientifically useful step is to test stronger models and redesigned prompts against the same hard-choice contract and the same large E01-E06 suite.

Recommended next direction:

1. Keep `L3_party_ideology_llm` as the current strongest leakage-free LLM baseline.
2. Treat `strict memory + candidate/context` as a failure mode needing redesign.
3. Balance or randomize candidate order in serious aggregate runs.
4. Keep E04/E05/E06 as mandatory diagnostics for any new model.
5. Add repeated sampling or multi-seed aggregation only after a stronger model shows less collapse.
6. Consider separately modeling turnout, since current LLM behavior nearly always chooses to vote.

## 13. Bottom Line

This MVP has succeeded as an evaluation system and failed, in an informative way, as a final simulator.

That is a useful outcome. The framework now exposes exactly where the model breaks: not in JSON formatting or runtime reliability, but in substantive political behavior simulation. The current small model mostly learns or guesses from party cues, and candidate/context wording often pushes it toward an extreme Republican default. The next iteration should focus on model quality and prompt design, not on more schema repair.
