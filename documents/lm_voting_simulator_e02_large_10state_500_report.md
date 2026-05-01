# E02-Large 10-State / 500-Agent Aggregate Election Accuracy Report

生成时间：2026-05-01  
实验目录：`data/runs/eval_suite_local/02_aggregate_accuracy_large_10state_500`  
配置文件：`configs/eval_suite/e02_aggregate_accuracy_large_10state_500.yaml`  
旧 E02 目录保留：`data/runs/eval_suite_local/02_aggregate_accuracy`

## 1. 实验目标

本轮实验是 E02 Aggregate Election Accuracy 的扩大版，目标是评估 `qwen3.5:2b` strict-memory LLM 在更大州集合和更大每州样本下，是否能通过 agent hard-choice 输出聚合出接近 2024 州级总统选举真值的结果。

本轮关注三个问题：

1. `survey_memory_llm_strict` 是否能恢复州级 Democratic two-party share、margin 和 winner？
2. LLM strict 是否优于简单非 LLM baseline，例如 `party_id_baseline`、`mit_2020_state_prior` 和 `uniform_national_swing_from_2020`？
3. 加入 MN/VA/CO 三个 2024 民主党胜州后，是否暴露或缓解 LLM 的 Republican skew？

本轮继续使用三分类 hard-choice 契约：

```json
{"choice": "not_vote|democrat|republican"}
```

旧概率列只解释为系统从 hard choice 派生的 one-hot 或聚合频率，不代表 LLM 主观概率。

## 2. 设计

### 2.1 States

本轮采用 10 州：

| Group | States | 说明 |
| --- | --- | --- |
| 原 7 个摇摆州 | PA, MI, WI, GA, AZ, NV, NC | 2024 均为 Republican winner |
| 新增 3 个民主党胜州 | MN, VA, CO | 用于打破“目标州全是 R 胜州”的解释偏差 |

2024 MIT 州级真值：

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

### 2.2 Sample sizes

配置中的 sample sizes：

```yaml
sample_sizes: [50, 100, 200, 300, 500, 1000, 2000]
```

LLM 只跑到每州 500：

```yaml
llm.max_sample_size: 500
```

因此：

- LLM `survey_memory_llm_strict` 有 50/100/200/300/500 五档结果。
- 非 LLM baselines 有 50/100/200/300/500/1000/2000 七档结果。

### 2.3 Baselines

LLM baseline：

| Baseline | 说明 |
| --- | --- |
| `survey_memory_llm_strict` | strict pre-election survey memory，`qwen3.5:2b` |

非 LLM baselines：

| Baseline | 说明 |
| --- | --- |
| `mit_2020_state_prior` | 2020 州级 prior |
| `uniform_national_swing_from_2020` | 2020 + national swing baseline |
| `party_id_baseline` | 党派规则 baseline |
| `sklearn_logit_pre_only_crossfit` | pre-election crossfit logit |
| `sklearn_logit_poll_informed` | poll-informed logit 上界参考 |
| `ces_post_self_report_aggregate_oracle` | post-self-report oracle，只作上界 |

本轮不加入 poll-informed LLM，避免 LLM 调用翻倍并保持主 LLM 条件 leakage-free。

## 3. 预检查和探针

预检查结果：

| 检查项 | 结果 |
| --- | --- |
| `py_compile` | 通过 |
| E02/parser/cache smoke tests | 通过 |
| Ollama `/api/tags` | 可见 `qwen3.5:2b` |
| processed CES/MIT artifacts | 全部存在 |
| 初始 GPU | RTX 5070 Laptop GPU，8151 MiB total，约 6786 MiB free |

探针结果：

| Probe | States | Calls | Workers | LLM runtime | Median | P90 | Throughput | GPU peak used | Min free | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | PA, GA, MN, VA, CO | 25 | 1 | 13.0s | 0.409s | 0.449s | 1.92/s | 4963 MiB | 2929 MiB | PASS |
| B | 10 states | 100 | 4 | 26.2s | 0.880s | 0.913s | 3.82/s | 4963 MiB | 2929 MiB | PASS |
| C | 10 states | 100 | 5 | 29.4s | 1.305s | 1.342s | 3.41/s | 4963 MiB | 2929 MiB | PASS |

Workers=5 没有错误，但比 workers=4 慢。因此正式运行采用 workers=4。

Probe B/C 的 10 州早期 choice 分布一致：

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

CO 在探针中偏 Republican，但不是全 Republican；MN/VA 也没有触发“新增民主党胜州全部预测 Republican”的止损规则。

## 4. 正式运行概况

正式运行完成，未中断。

| 指标 | 值 |
| --- | ---: |
| Workers | 4 |
| States | 10 |
| Effective LLM sample size | 500 per state |
| LLM tasks | 5000 |
| Ollama calls | 4988 |
| Cache hit rate | 0.002 |
| Total wall-clock | 1381.5s，约 23.0 min |
| LLM runtime | 1213.1s，约 20.2 min |
| Median latency | 0.897s |
| P90 latency | 0.967s |
| Throughput | 4.12 responses/s |
| GPU peak used | 4998 MiB |
| GPU min free | 2894 MiB |
| GPU peak utilization | 83% |

正式运行满足 30 分钟预算。GPU 吃得比较充分，且没有接近 OOM。

## 5. 输出产物

正式目录：

`data/runs/eval_suite_local/02_aggregate_accuracy_large_10state_500`

核心输出：

| 文件 | Shape / 说明 |
| --- | --- |
| `sampled_agents.parquet` | 12805 x 50 |
| `sample_membership.parquet` | 33632 x 7 |
| `responses.parquet` | 56220 x 38 |
| `prompts.parquet` | 5000 x 12 |
| `state_predictions.parquet` | 470 x 21 |
| `aggregate_metrics.parquet` | 329 x 8 |
| `parse_diagnostics.parquet` | 5 x 8 |
| `runtime_log.parquet` | 5023 x 18 |
| `runtime.json` | 运行观测 |
| `llm_cache.jsonl` | LLM cache |
| `benchmark_report.md` | runner 自动报告 |
| `figures/*` | 图表 |

图表包括：

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

## 6. 质量门

Parse diagnostics：

| Baseline | N | Parse OK | Fallback rate | Cache hit |
| --- | ---: | ---: | ---: | ---: |
| `ces_post_self_report_aggregate_oracle` | 12805 | 1.000 | 0.000 | NA |
| `party_id_baseline` | 12805 | 1.000 | 0.000 | NA |
| `sklearn_logit_poll_informed` | 12805 | 1.000 | 0.000 | NA |
| `sklearn_logit_pre_only_crossfit` | 12805 | 1.000 | 0.000 | NA |
| `survey_memory_llm_strict` | 5000 | 1.000 | 0.000 | 0.002 |

Runtime gates：

| Gate | 值 | 结果 |
| --- | ---: | --- |
| parse_ok_rate | 1.000 | PASS |
| invalid_choice_rate | 0.000 | PASS |
| forbidden_choice_rate | 0.000 | PASS |
| legacy_probability_schema_rate | 0.000 | PASS |
| transport_error_rate | 0.000 | PASS |
| fallback_rate | 0.000 | PASS |
| all_gates_passed | true | PASS |

工程结论：本轮 hard-choice LLM 链路稳定。没有 JSON/schema 问题，也没有 transport error。

## 7. LLM Choice Distribution

总体 LLM hard choices：

| Choice | N | Share |
| --- | ---: | ---: |
| republican | 2995 | 0.599 |
| democrat | 1999 | 0.400 |
| not_vote | 6 | 0.001 |

按州：

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

观察：

1. LLM 几乎不输出 `not_vote`，这和 E01-large 一致。
2. LLM overall 偏 Republican，R share 约 59.9%。
3. 新增民主党胜州中，MN 几乎 50/50，但 VA/CO 仍明显偏 Republican。
4. 注意：这里的 raw choice distribution 不是最终 weighted aggregate share；最终州级 `pred_dem_2p` 使用 sample weight 和 turnout-aware 聚合。

## 8. Aggregate Metrics at Sample Size 500

以下是共同 sample size 500 下的 10 州总体指标，是和 LLM strict 最公平的横向比较。

| Baseline | Dem 2P RMSE | Margin MAE | Margin bias | Winner acc | Winner flips | Corr true margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `survey_memory_llm_strict` | 0.162 | 0.314 | -0.314 | 0.700 | 3 | 0.286 |
| `party_id_baseline` | 0.074 | 0.126 | 0.124 | 0.400 | 6 | 0.626 |
| `sklearn_logit_pre_only_crossfit` | 0.047 | 0.082 | -0.073 | 0.800 | 2 | 0.647 |
| `sklearn_logit_poll_informed` | 0.048 | 0.079 | -0.073 | 0.800 | 2 | 0.535 |
| `mit_2020_state_prior` | 0.019 | 0.034 | 0.034 | 0.400 | 6 | 0.956 |
| `uniform_national_swing_from_2020` | 0.015 | 0.026 | -0.026 | 1.000 | 0 | 0.956 |
| `ces_post_self_report_aggregate_oracle` | 0.031 | 0.055 | -0.039 | 0.700 | 3 | 0.740 |

核心结论：

1. LLM strict 的 winner accuracy 是 0.700，但这个数字有结构性迷惑性：10 州里 7 州真值是 Republican winner，LLM 全部预测 Republican，也会得到 0.700。
2. 更关键的是 margin：LLM strict `margin_bias=-0.314`，表示平均 margin 向 Republican 方向偏约 31.4 个百分点。
3. LLM strict 的 `dem_2p_rmse=0.162` 和 `margin_mae=0.314`，明显弱于所有关键非 LLM baseline。
4. `uniform_national_swing_from_2020` 在 10 州上非常强，winner accuracy 1.000，margin MAE 0.026。这说明 2024 州级结果在这些州中可以被简单历史 swing prior 很好解释。
5. `mit_2020_state_prior` 的 share/margin RMSE 很低，但 winner accuracy 只有 0.400，因为 2020 先验在接近 50/50 的摇摆州中预测了过多 Democrat winner。

## 9. Non-LLM at Sample Size 2000

非 LLM baseline 可跑到 2000，用于观察统计模型和 oracle 在更大样本下的上界。

| Baseline | Dem 2P RMSE | Margin MAE | Margin bias | Winner acc | Winner flips | Corr true margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `party_id_baseline` | 0.080 | 0.146 | 0.146 | 0.400 | 6 | 0.731 |
| `sklearn_logit_pre_only_crossfit` | 0.029 | 0.050 | -0.028 | 0.900 | 1 | 0.734 |
| `sklearn_logit_poll_informed` | 0.028 | 0.050 | -0.027 | 0.800 | 2 | 0.698 |
| `mit_2020_state_prior` | 0.019 | 0.034 | 0.034 | 0.400 | 6 | 0.956 |
| `uniform_national_swing_from_2020` | 0.015 | 0.026 | -0.026 | 1.000 | 0 | 0.956 |
| `ces_post_self_report_aggregate_oracle` | 0.019 | 0.032 | 0.001 | 0.800 | 2 | 0.836 |

非 LLM 的更大样本并没有改变大结论：统计模型和历史 swing prior 明显优于当前 LLM strict。

## 10. LLM Sample Size Sensitivity

`survey_memory_llm_strict` 随 sample size 增大：

| Sample size | Dem 2P RMSE | Margin MAE | Margin bias | Winner acc | Winner flips |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.204 | 0.393 | -0.393 | 0.700 | 3 |
| 100 | 0.176 | 0.342 | -0.342 | 0.700 | 3 |
| 200 | 0.173 | 0.333 | -0.333 | 0.700 | 3 |
| 300 | 0.171 | 0.332 | -0.332 | 0.700 | 3 |
| 500 | 0.162 | 0.314 | -0.314 | 0.700 | 3 |

增大样本确实降低了方差，RMSE 和 MAE 从 50 到 500 有所改善。但 winner accuracy 一直是 0.700，因为 LLM 在所有样本规模下都预测 10 州全为 Republican winner，错掉 MN/VA/CO。

这说明更大样本不能修复系统性偏差，只能让偏差估计更稳定。

## 11. State-Level LLM Results at Sample Size 500

`survey_memory_llm_strict` 州级明细：

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

所有 10 州的 `dem_2p_error` 都为负；这不是“共和党赢得 7 个摇摆州”的正常现象，而是和真值相比系统性低估 Democratic two-party share。

## 12. Group Metrics

共同 sample size 500 下，按州组分解：

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

分组解释：

1. 在原 7 州上，LLM strict winner accuracy 是 1.000，因为原 7 州真实都为 Republican winner，且 LLM 也全部预测 Republican。
2. 但原 7 州 LLM margin MAE 仍高达 0.307，说明 winner 对了但 margin 错得很大。
3. 在新增 3 州 MN/VA/CO 上，LLM strict winner accuracy 是 0.000，三个民主党胜州全部预测为 Republican。
4. 新增三州暴露了 winner accuracy 对州集合构成的高度敏感性，也证明只看七个共和党胜摇摆州会显著高估偏 Republican 模型。

## 13. 与旧 E02 的关系

旧 E02 只跑 7 个摇摆州，每州最多 70 个 LLM agents。旧 E02 的 LLM strict 指标：

| Sample size | Dem 2P RMSE | Margin MAE | Margin bias | Winner acc | Winner flips |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.203 | 0.404 | -0.404 | 1.000 | 0 |
| 70 | 0.180 | 0.351 | -0.351 | 1.000 | 0 |

新 E02-large 在原 7 州、每州 500 的 LLM strict：

| Group | Dem 2P RMSE | Margin MAE | Margin bias | Winner acc | Winner flips |
| --- | ---: | ---: | ---: | ---: | ---: |
| original7, sample 500 | 0.158 | 0.307 | -0.307 | 1.000 | 0 |

样本变大后，LLM 在原 7 州上的 margin error 有所下降，但仍然非常大。加入 MN/VA/CO 后，全 10 州 winner accuracy 从表面上的 1.000 下降到 0.700，揭示了旧 E02 的 winner accuracy 受州集合构成影响很强。

## 14. 结论

### 14.1 工程结论

E02-large 工程链路通过：

1. 5000 次 LLM 调用全部 parse 成功。
2. 无 invalid choice、forbidden choice、legacy probability schema、transport error。
3. 无 fallback。
4. workers=4 稳定，正式墙钟约 23 分钟，低于 30 分钟预算。
5. GPU 峰值显存约 4998 MiB，最低剩余约 2894 MiB，没有 OOM 风险。

当前系统可以稳定跑 10 州 x 每州 500 的 aggregate LLM 实验。

### 14.2 科学结论

当前 `qwen3.5:2b` strict-memory LLM 不适合作为可靠的州级选举模拟器：

1. 它在全部 10 州都低估 Democratic two-party share。
2. 它预测 10 州全为 Republican winner，因此在 7 个 Republican 胜州上 winner 正确，在 MN/VA/CO 三个 Democratic 胜州上全部错误。
3. 它的 10 州 margin bias 为 -0.314，说明平均向 Republican 方向偏约 31.4 个百分点。
4. 它明显弱于 `sklearn_logit_pre_only_crossfit`、`sklearn_logit_poll_informed`、`uniform_national_swing_from_2020` 和 `mit_2020_state_prior` 等非 LLM baseline。
5. 增大样本量只能降低方差，不能修复系统性 Republican skew。

### 14.3 对 E01-large 结论的影响

本轮 E02-large 强化了 E01-large 的判断：

1. strict memory 没有表现出稳定增益。
2. 当前 LLM 能利用 persona 中的政治线索，但在 aggregate 层会形成强系统性偏差。
3. 只看 winner accuracy 会误导，尤其当州集合多数真值为同一党胜时。
4. 后续报告必须优先看 `dem_2p_rmse`、`margin_mae`、`margin_bias`，winner accuracy 只能作为辅助指标。

## 15. 建议

后续若继续扩大或改造 E02，我建议：

1. 暂时不要把 LLM strict 当作预测器主结果，只作为诊断对象。
2. E02/E04/E05 的 aggregate 解读必须固定加入民主党胜州对照组，避免 winner accuracy 被 Republican skew 虚高。
3. 如果要继续改善 LLM simulator，应优先解决：
   - turnout 几乎不输出 `not_vote`；
   - strict memory 导致更强 Republican bias；
   - aggregate calibration 缺失；
   - 州级 prior 和 persona vote choice 之间没有可靠校准层。
4. 下一轮可考虑实验“两阶段 turnout/vote choice”或“persona hard choice + 后验校准层”，而不是继续单纯增加 LLM 样本量。

本轮 E02-large 正式通过，结果可复用，但科学结论对当前 LLM aggregate simulator 不利。
