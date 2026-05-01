# LM Voting Simulator E06-Large 10-State Subgroup and Calibration Reliability Report

生成时间：2026-05-01

本报告总结大规模 E06：Subgroup and Calibration Reliability。本轮 E06-large 不调用 Ollama，只复用已经完成的大规模 E01/E05 产物，检查 hard-choice 输出在 subgroup、公平性、分布压缩和 deterministic turnout calibration 上是否可靠。

## 1. 实验设置

### 1.1 输入与输出

正式输出目录：

`data/runs/eval_suite_local/06_subgroup_calibration_large_10state`

独立报告文件：

`documents/lm_voting_simulator_e06_large_10state_report.md`

配置文件：

`configs/eval_suite/e06_subgroup_calibration_large_10state.yaml`

本轮复用两个大规模输入源：

| Source label in E06 | Actual input directory | Role |
| --- | --- | --- |
| `E01_individual_persona` | `data/runs/eval_suite_local/01_individual_persona_large_10state_1000` | 10 州 individual persona fidelity，大约 1000 个 LLM agents，加非 LLM baselines |
| `E05_ablation_placebo` | `data/runs/eval_suite_local/05_ablation_placebo_large_10state_full_ladder_600` | 10 州 full-ladder ablation/placebo，600 agents x 10 LLM baselines |

E02-large 和 E04-large 只作为背景参考，没有进入 E06 表计算。E06 runner 当前只消费 E01/E05 的 `responses.parquet` 和 `agents.parquet`，再合并 CES targets。

### 1.2 分析维度

E06 生成以下 subgroup 维度：

| Dimension | 含义 |
| --- | --- |
| `overall` | 全样本总体 |
| `party_id_3` | 三分类党派认同 |
| `party_id_7` | 七分类党派认同 |
| `ideology_3` | 三分类意识形态 |
| `race_ethnicity` | 种族/族裔 |
| `education_binary` | 是否大学及以上 |
| `age_group` | 年龄组 |
| `gender` | 性别 |
| `state_po` | 州 |
| `state_po_x_party_id_3` | 州 x 三分类党派 |
| `state_po_x_race_ethnicity` | 州 x 种族/族裔 |

`small_n_threshold=30`。小样本 subgroup 保留，但打标为 `small_n=True`。下面的最差 subgroup 结论优先看非小样本组。

### 1.3 Hard-choice 解释原则

本轮系统使用三分类硬选择契约：

- `not_vote`
- `democrat`
- `republican`

因此响应表中的概率列不是 LLM 主观概率，而是系统从 hard choice 派生的 one-hot 兼容列。E06 中的 calibration 也不是“LLM 对概率有多校准”，而是 deterministic 0/1 turnout choice 的后验可靠性诊断。

## 2. 运行与质量门

### 2.1 预检查和探针

预检查通过：

- 编译通过：
  - `src/election_sim/ces_subgroup_calibration_benchmark.py`
  - `src/election_sim/eval_suite.py`
  - `src/election_sim/cli.py`
- E06 smoke tests 通过：
  - `test_ces_subgroup_calibration_benchmark_runner_smoke`
  - `test_turnout_vote_parser_and_aggregation`

只读输入探针结果：

| Source | responses | agents | states | parse_ok | invalid | forbidden | legacy schema | transport |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| E01-large | 13,844 | 13,401 | AZ, CO, GA, MI, MN, NC, NV, PA, VA, WI | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| E05-large | 6,000 | 600 | AZ, CO, GA, MI, MN, NC, NV, PA, VA, WI | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

所有关键 subgroup 列无缺失：

- `party_id_3`
- `party_id_7`
- `ideology_3`
- `race_ethnicity`
- `education_binary`
- `age_group`
- `gender`
- `state_po`

probe run：

| Run | Directory | workers | Runtime | Responses | Subgroup rows | Figures | Gates | LLM calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Probe | `data/runs/eval_suite_local/e06_large_probe_10state_w2` | 2 | 28.99s | 19,844 | 1,991 | 6 | PASS | 0 |

### 2.2 正式运行

正式运行通过：

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

正式输出文件齐全：

- `subgroup_reliability_metrics.parquet`
- `distribution_diagnostics.parquet`
- `calibration_bins.parquet`
- `worst_subgroups.parquet`
- `quality_gates.parquet`
- `runtime.json`
- `config_snapshot.yaml`
- `benchmark_report.md`
- `figures/*`

质量门全部通过。正式 `runtime.json` 明确记录 `n_llm_tasks=0`、`llm_calls_made=0`。

## 3. 总体指标

### 3.1 E01-large 总体

E01-large 中，LLM baselines 的个体层结果如下：

| Baseline | n | Vote accuracy | Macro F1 | Turnout accuracy | Turnout Brier | Pred Dem 2P | True Dem 2P | Dem 2P error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ces_demographic_only_llm` | 1,000 | 0.518 | 0.338 | 0.968 | 0.0315 | 0.405 | 0.466 | -0.061 |
| `ces_party_ideology_llm` | 1,000 | 0.917 | 0.611 | 0.968 | 0.0315 | 0.374 | 0.466 | -0.092 |
| `ces_survey_memory_llm_strict` | 1,000 | 0.871 | 0.578 | 0.968 | 0.0315 | 0.309 | 0.466 | -0.157 |

E01-large 非 LLM 参考：

| Baseline | n | Vote accuracy | Macro F1 | Turnout Brier | Pred Dem 2P | True Dem 2P | Dem 2P error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `party_id_baseline` | 2,711 | 0.868 | 0.595 | 0.0457 | 0.563 | 0.501 | +0.063 |
| `sklearn_logit_demographic_only` | 2,711 | 0.627 | 0.412 | 0.0295 | 0.663 | 0.501 | +0.162 |
| `sklearn_logit_pre_only` | 2,711 | 0.947 | 0.647 | 0.0451 | 0.497 | 0.501 | -0.004 |
| `sklearn_logit_poll_informed` | 2,711 | 0.954 | 0.652 | 0.0348 | 0.490 | 0.501 | -0.010 |

解读：

- E01-large 再次确认：`party/ideology` 是最有效的 LLM 信息源。
- `strict survey memory` 的总体 vote accuracy 低于 `party/ideology`，Dem 2P 也更偏 Republican。
- 非 LLM `sklearn_logit_pre_only` 和 `sklearn_logit_poll_informed` 明显强于当前小模型 LLM。

### 3.2 E05-large 总体

E05-large 的完整 ladder 总体指标：

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

解读：

- `L3_party_ideology_llm` 仍是非 oracle LLM 中最稳定的一档。
- `L5_strict_memory_llm` 有一定信息，但弱于 `L3`。
- 一加入候选人/context 后，`L4`、`L6`、`L7` 明显向 Republican collapse 方向移动。
- `L8` oracle 接近上界，说明 runner 和 hard-choice 契约本身可以承载正确答案；主要问题是小模型在真实 prompt 下的推断偏差。

## 4. 分布诊断：是否塌缩

### 4.1 总体选择分布

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

解释：

- `L4/L6/L7` 的总体 Republican share 非常高，且 entropy/variance ratio 明显低于 1，属于强分布压缩。
- `L1/L2/P1/P2` 基本全 Republican，整体 entropy 接近 0。
- `L3` 的总体 entropy ratio 接近 1，但仍低估 Democratic two-party share。
- `L8` oracle 出现较高 `not_vote`，这是因为 oracle memory 明确带入 turnout/post-hoc 信息，不能作为 leakage-free 模拟器，只能作为上界/诊断。

### 4.2 分布指标的注意事项

Entropy ratio 大于 1 不一定代表好，也可能出现在真实 subgroup 本身几乎纯一边倒时。例如 Democratic subgroup 真实几乎全 Democrat，如果模型输出了一些 Republican，预测分布反而比真实分布“更有熵”，ratio 会大于 1。这种情况代表 subgroup 内部错误变多，而不是校准更好。

## 5. Party Subgroup 结果

### 5.1 三分类党派

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

核心结论：

- `party/ideology` prompt 能很好处理明确 Democrat 和 Republican persona。
- 最大不稳定点是 independent/other：模型经常把这组推向 Republican，Dem 2P 明显低估。
- 加候选人/context 后，Democratic subgroup 被严重错误地推向 Republican。`L4/L6/L7` 在 Democratic subgroup 上的错误最大。
- `L5_strict_memory_llm` 比 `L6_strict_memory_context_llm` 明显好，说明这轮结果中的“strict memory 伤害”主要来自和候选人/context 组合后的 prompt 行为，而不是 memory 表本身完全无信息。

### 5.2 七分类党派最差项

非小样本最差 subgroup 主要集中在 Democratic identifiers：

| Baseline | Worst subgroup | n | Vote accuracy | Dem 2P error |
| --- | --- | ---: | ---: | ---: |
| `L4_party_ideology_context_llm` | `party_id_7=Lean Democrat` | 77 | 0.019 | -0.981 |
| `L5_strict_memory_llm` | `party_id_7=Lean Democrat` | 77 | 0.093 | -0.914 |
| `L6_strict_memory_context_llm` | `party_id_7=Lean Democrat` | 77 | 0.019 | -0.981 |
| `L7_poll_informed_memory_context_llm` | `party_id_7=Lean Democrat` | 77 | 0.019 | -0.981 |
| `L3_party_ideology_llm` | `party_id_7=Independent` | 76 | 0.481 | -0.519 |

这说明当前小模型不是均匀地“有点偏共和党”，而是在若干 Democratic-leaning subgroup 上出现非常强的方向性错误。

## 6. Race/Ethnicity Subgroup 结果

重点 subgroup：

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

解读：

- Black subgroup 是很清楚的 stress test：真实 Democratic share 很高。
- `L3` 仍有低估，但相对可用。
- `L4/L6/L7` 对 Black subgroup 的 Democratic share 严重低估，说明候选人/context prompt 触发的 Republican skew 会在真实 Democratic subgroup 上被放大。
- Oracle 能恢复接近真实值，说明不是数据标签或聚合代码问题。

## 7. 州维度和新三州

### 7.1 10 州 state-level subgroup

E01-large 主要 LLM：

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

E05-large key baselines：

| Baseline | State | Vote accuracy | Pred Dem 2P | True Dem 2P | Error |
| --- | --- | ---: | ---: | ---: | ---: |
| `L3_party_ideology_llm` | CO | 0.811 | 0.452 | 0.702 | -0.250 |
| `L3_party_ideology_llm` | WI | 0.749 | 0.415 | 0.510 | -0.096 |
| `L4_party_ideology_context_llm` | CO | 0.336 | 0.033 | 0.702 | -0.668 |
| `L5_strict_memory_llm` | CO | 0.650 | 0.311 | 0.702 | -0.391 |
| `L6_strict_memory_context_llm` | CO | 0.324 | 0.023 | 0.702 | -0.679 |
| `L7_poll_informed_memory_context_llm` | CO | 0.422 | 0.110 | 0.702 | -0.592 |
| `L8_post_hoc_oracle_memory_context_llm` | CO | 0.980 | 0.666 | 0.702 | -0.036 |

CO 是新增三州中最明显的 stress test，因为真实 Democratic share 高。非 oracle 严重低估 CO 的 Democratic share，尤其是带 candidate/context 的 variants。

### 7.2 原 7 州 vs 新增 3 州

按 state rows 聚合的对比：

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

新增 MN/VA/CO 没有“修复”系统性 Republican skew，反而让问题更清楚：在 Democratic-winning 或 Democratic-leaning 州，非 oracle LLM 的 Dem 2P 低估更严重。

## 8. 最差 subgroup

### 8.1 每个重点 baseline 的最差 vote subgroup

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

- E01 的 `party/ideology` 在总体上强，但 `ideology_3=unknown` 和部分 independent subgroup 风险高。
- E05 的 `L3` 主要失败在 Independent。
- `L4/L6/L7` 最差项都落在 Lean Democrat，且几乎完全错向 Republican。
- `L5` 也在 Lean Democrat 上失败明显，但没有 `L4/L6/L7` 那么极端。
- `L8` oracle 的最差项仍然很高，说明 E06 指标和目标标签没有明显系统性错误。

## 9. Turnout calibration

E06 的 turnout calibration 是 hard-choice 0/1 的后验可靠性：

| Source | Baseline | ECE | MCE | Notes |
| --- | --- | ---: | ---: | --- |
| E01 | all three LLM baselines | 0.0315 | 0.0315 | LLM 几乎全预测 turnout=1，真实 turnout 约 0.968 |
| E05 | `L1/L2/L4/L5/L6/P1/P2` | 0.0503 | 0.0503 | 几乎全预测 turnout=1，真实 turnout 约 0.950 |
| E05 | `L3_party_ideology_llm` | 0.0563 | 0.6844 | 有 3 个预测 not_vote，但加权真实 turnout 高，导致 MCE 高 |
| E05 | `L7_poll_informed_memory_context_llm` | 0.0552 | 1.0000 | 有 4 个预测 not_vote，且这几例真实都 turnout |
| E05 | `L8_post_hoc_oracle_memory_context_llm` | 0.0077 | 0.1334 | oracle memory 明显改善 turnout choice |

结论：

- 非 oracle LLM 几乎总是选择投票，因此 turnout calibration 主要反映样本真实 turnout 率。
- 当前 MVP 最主要问题不是 turnout，而是 vote choice 方向偏差。
- Oracle 能改善 turnout，说明如果 memory 明确包含 post-hoc turnout 信息，系统可以表达 not_vote；但这不是 leakage-free 主模拟。

## 10. Figures

正式目录生成 6 张图：

- `figures/e06_subgroup_vote_accuracy_heatmap.png`
- `figures/e06_subgroup_turnout_brier_heatmap.png`
- `figures/e06_subgroup_dem_2p_error.png`
- `figures/e06_turnout_reliability.png`
- `figures/e06_entropy_ratio_by_subgroup.png`
- `figures/e06_variance_ratio_by_subgroup.png`

E06 图表 focus baseline 已补充 `L4_party_ideology_context_llm` 和 `L5_strict_memory_llm`，避免漏掉 E05-large 中最关键的 context effect 和 no-context strict-memory 结果。

## 11. 结论

### 11.1 主要结论

1. E06-large 运行可靠，质量门全通过，且没有任何新增 LLM 调用。

2. 大样本 subgroup 结果强化了前面 E01/E05-large 的结论：当前 `qwen3.5:2b` 最大价值来自 party/ideology 信息，而不是 strict memory。

3. `party/ideology` baseline 在总体和明确党派 subgroup 上表现较好，但对 Independent/unknown/部分 state x party subgroup 仍然明显偏 Republican。

4. 加入候选人/context 后，模型出现严重 Republican collapse。`L4/L6/L7` 在 Democratic identifiers、Black subgroup、CO/VA/MN 等 Democratic-leaning 州上错误尤其大。

5. `L5_strict_memory_llm` 比 `L6_strict_memory_context_llm` 好，说明 strict memory 本身不是完全无用；问题很大程度来自 memory 与 candidate/context prompt 组合后触发的世界知识/候选人语境偏差。

6. `L8` oracle 接近正确，说明代码链路、hard-choice 解析和 subgroup 计算本身没有明显故障；模型偏差是实验结果，不是格式失败。

7. 新增 MN/VA/CO 没有改变总体判断，反而更清楚地暴露了 Democratic-winning states 上的 Dem share 低估。

### 11.2 对模拟器的评价

作为 MVP，这套系统的工程链路现在是稳定的：

- hard-choice 契约稳定。
- 并发和缓存链路稳定。
- E01-E06 及 large 系列输出可复用。
- 质量门没有暴露解析或 schema 问题。

但作为“可靠 voter simulator”，当前小模型仍不合格：

- 它不是在稳定地从 persona/memory 推断个体选择。
- 它强依赖 party label。
- 它在 candidate/context 下会出现系统性 Republican skew。
- 它对 Independent、Democratic-leaning subgroup、Black subgroup 和 Democratic-winning states 的可靠性不足。

更准确地说，当前系统可以作为评估框架和实验平台使用，但 `qwen3.5:2b` 不应被视为可靠的最终模拟器模型。

### 11.3 是否改变前序 large 实验结论

没有改变，反而加强：

- E01-large：`party/ideology` 最有效，strict memory 没有稳定增益。
- E02-large：州级聚合中 LLM strict 仍有系统性偏差。
- E03-large：prompt wording 不是唯一问题，核心仍是方向性偏差。
- E04-large：candidate/party/world-knowledge leakage 风险真实存在。
- E05-large：完整 ladder 显示 candidate/context 会显著伤害 Democratic subgroup。

E06-large 的新贡献是把这些结论定位到了具体 subgroup：Independent、Lean Democrat、Black respondents、CO/VA/MN 等 Democratic-leaning state rows。

## 12. 后续建议

1. 后续如果继续评估更强模型，优先复用同一 hard-choice contract 和 E06-large 指标。

2. 如果要改善 prompt，优先处理 candidate/context 触发的 Republican collapse，而不是继续强化 JSON 格式约束。

3. 如果要保留 LLM simulator 路线，建议把 `party/ideology` prompt 作为当前最强 leakage-free baseline，把 strict memory/context prompt 视为待修复对象。

4. 报告层面建议把 E06-large 的 subgroup findings 放进最终总报告，因为它能解释为什么总体 accuracy 看起来尚可但 aggregate/state/subgroup 仍不可靠。
