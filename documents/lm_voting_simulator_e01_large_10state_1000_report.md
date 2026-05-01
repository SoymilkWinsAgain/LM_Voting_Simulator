# E01-Large 10-State / 1000-Agent Individual Persona Fidelity Report

生成时间：2026-05-01  
实验目录：`data/runs/eval_suite_local/01_individual_persona_large_10state_1000`  
配置文件：`configs/eval_suite/e01_individual_persona_large_10state_1000.yaml`

## 1. 实验目标

本轮实验是在已完成的 E01 小规模实验基础上，扩大个体层面 persona fidelity 评估规模。核心问题仍然遵循 `documents/lm_voting_simulator_evaluation_plan.md`：

1. LLM agent 能否在个体层面复现 CES 受访者自己的 2024 总统投票行为？
2. 不同输入信息层级的效果如何变化？
   - `ces_demographic_only_llm`：只给人口统计信息。
   - `ces_party_ideology_llm`：加入党派和意识形态。
   - `ces_survey_memory_llm_strict`：加入 strict pre-election survey memory。
3. 扩展到更多州后，原先的结论是否改变？
4. 新增民主党胜州后，是否缓解或暴露模型的系统性共和党偏斜？

本轮不覆盖旧结果。上一轮 E01 正式目录仍为：

`data/runs/eval_suite_local/01_individual_persona`

本轮新目录为：

`data/runs/eval_suite_local/01_individual_persona_large_10state_1000`

## 2. 州和样本设计

本轮采用 10 州，每州约 100 个 LLM agents，总计 1000 个 LLM agents。

保留原 7 个 2024 摇摆州：

| State | 说明 |
| --- | --- |
| PA | 2024 共和党胜，核心摇摆州 |
| MI | 2024 共和党胜，核心摇摆州 |
| WI | 2024 共和党胜，核心摇摆州 |
| GA | 2024 共和党胜，核心摇摆州 |
| AZ | 2024 共和党胜，核心摇摆州 |
| NV | 2024 共和党胜，核心摇摆州 |
| NC | 2024 共和党胜，核心摇摆州 |

新增 3 个民主党胜州：

| State | 选择理由 |
| --- | --- |
| MN | 中西部，和 WI/MI 相近，但 2024 民主党胜 |
| VA | 南方/大西洋州，和 NC/GA 形成对照 |
| CO | 西部州，和 AZ/NV 形成对照 |

正式 LLM agents 分布：

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

总计 1000 个 LLM agents。PA/CO 的 101/99 轻微偏差来自现有 runner 的抽样和去重行为，规模上满足本轮设计。

## 3. 运行配置

LLM 运行配置：

| 项 | 值 |
| --- | --- |
| Ollama base URL | `http://172.26.48.1:11434` |
| Model | `qwen3.5:2b` |
| Response contract | hard-choice JSON: `{"choice": "not_vote|democrat|republican"}` |
| Max tokens | 80 |
| 正式 workers | 4 |
| LLM baselines | 3 |
| LLM agents | 1000 |
| LLM requests | 3000 |

本轮继续不使用 poll-informed LLM，以保持主 LLM 实验不引入民调信息泄漏。非 LLM 参考基线包括：

| Baseline | 用途 |
| --- | --- |
| `party_id_baseline` | 简单党派规则参考 |
| `sklearn_logit_demographic_only` | 人口统计统计模型 |
| `sklearn_logit_pre_only` | pre-election 信息统计模型 |
| `sklearn_logit_poll_informed` | poll-informed 参考上界 |

## 4. 预检和探针

代码和接口预检全部通过：

| 检查项 | 结果 |
| --- | --- |
| `py_compile` | 通过 |
| E01/parser 相关 pytest | 通过 |
| Ollama `/api/tags` | 可见 `qwen3.5:2b` |
| 初始 GPU | RTX 5070 Laptop GPU，8151 MiB total，约 6844 MiB free |

探针结果：

| Probe | 规模 | Workers | LLM runtime | Median latency | P90 latency | Throughput | GPU peak used | Min free | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | 30 calls | 3 | 9.0s | 0.533s | 1.042s | 3.34/s | 5052 MiB | 2840 MiB | PASS |
| B | 90 calls | 4 | 19.0s | 0.728s | 1.081s | 4.74/s | 5052 MiB | 2840 MiB | PASS |
| C | 90 calls | 5 | 20.4s | 0.973s | 1.483s | 4.41/s | 5051 MiB | 2841 MiB | PASS |

Workers=5 没有出错，p90 也仍在 1.5s 左右，但吞吐低于 workers=4，latency 更差。因此正式运行选择 workers=4。

探针中没有发现：

| 异常类型 | 结果 |
| --- | --- |
| transport error | 0 |
| invalid choice | 0 |
| forbidden choice | 0 |
| legacy probability schema | 0 |
| parse failure | 0 |

新增三州探针没有触发止损规则。`demographic-only` 在 CO 的极小探针样本中明显偏共和党，但 `party/ideology` 和 `strict memory` 没有出现三州全量 collapse，因此继续正式实验。

## 5. 正式运行概况

正式运行完成，无中断，无 schema 异常。

| 指标 | 值 |
| --- | ---: |
| Total wall-clock | 732.1s，约 12.2 min |
| LLM runtime | 581.9s，约 9.7 min |
| Workers | 4 |
| LLM tasks | 3000 |
| Median latency | 0.686s |
| P90 latency | 0.899s |
| Throughput | 5.16 responses/s |
| Cache hit rate | 0.0 |
| GPU peak used | 5054 MiB |
| GPU min free | 2838 MiB |
| GPU peak utilization | 85% |

质量门：

| Gate | 值 | 结果 |
| --- | ---: | --- |
| parse_ok_rate | 1.000 | PASS |
| invalid_choice_rate | 0.000 | PASS |
| forbidden_choice_rate | 0.000 | PASS |
| legacy_probability_schema_rate | 0.000 | PASS |
| transport_error_rate | 0.000 | PASS |
| all_gates_passed | true | PASS |

输出文件存在并完整，包括：

| 文件 | 说明 |
| --- | --- |
| `agents.parquet` | agent 表 |
| `cohort.parquet` | 10 州 cohort |
| `prompts.parquet` | LLM prompt |
| `responses.parquet` | LLM 和非 LLM 响应 |
| `crossfit_responses.parquet` | crossfit 统计模型响应 |
| `individual_metrics.parquet` | 个体层指标 |
| `subgroup_metrics.parquet` | subgroup 指标 |
| `aggregate_eval_metrics.parquet` | 非 LLM 聚合选举指标 |
| `runtime_log.parquet` | 每条 LLM 请求运行日志 |
| `runtime.json` | 总运行观测 |
| `llm_cache.jsonl` | LLM cache |
| `benchmark_report.md` | runner 自动报告 |
| `figures/*` | 图表 |

主要表规模：

| 表 | Shape |
| --- | ---: |
| `responses.parquet` | 13844 x 34 |
| `runtime_log.parquet` | 3000 x 16 |
| `individual_metrics.parquet` | 224 x 10 |
| `subgroup_metrics.parquet` | 13150 x 10 |
| `aggregate_eval_metrics.parquet` | 104 x 14 |
| `cohort.parquet` | 13401 x 47 |
| `agents.parquet` | 13401 x 31 |

注意：`cohort` 和 `agents` 包含完整可评估 cohort；LLM pilot 正式只抽取其中的 1000 个 agents。`responses.parquet` 同时包含 LLM 和非 LLM baseline 响应。

## 6. 个体层总体结果

以下为 weighted individual metrics。旧概率列只作为 hard-choice one-hot 派生列解释，不代表 LLM 主观概率。

| Baseline | parse_ok | Turnout acc | Turnout brier | Vote acc | Vote macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ces_demographic_only_llm` | 1.000 | 0.968 | 0.032 | 0.518 | 0.338 |
| `ces_party_ideology_llm` | 1.000 | 0.968 | 0.032 | 0.917 | 0.611 |
| `ces_survey_memory_llm_strict` | 1.000 | 0.968 | 0.032 | 0.871 | 0.578 |
| `party_id_baseline` | 1.000 | 0.954 | 0.046 | 0.868 | 0.595 |
| `sklearn_logit_demographic_only` | 1.000 | 0.971 | 0.029 | 0.627 | 0.412 |
| `sklearn_logit_pre_only` | 1.000 | 0.955 | 0.045 | 0.947 | 0.647 |
| `sklearn_logit_poll_informed` | 1.000 | 0.965 | 0.035 | 0.954 | 0.652 |

核心结论：

1. `party/ideology` 是最强 LLM baseline。
2. `strict memory` 不但没有稳定超过 `party/ideology`，反而下降：
   - vote accuracy：0.917 -> 0.871，下降 0.046。
   - macro F1：0.611 -> 0.578，下降 0.033。
3. `demographic-only` 明显不足，vote accuracy 只有 0.518，接近弱分类器。
4. 非 LLM `sklearn_logit_pre_only` 和 `sklearn_logit_poll_informed` 仍显著强于所有 LLM baseline。
5. turnout accuracy 看起来高，但解释时要非常小心：LLM 几乎总是选择投票给两党候选人，很少选择 `not_vote`。高 turnout accuracy 很大程度来自样本中投票者比例高，而不是模型真正学会 turnout。

## 7. 与上一轮小规模 E01 的比较

上一轮 E01 是 4 州、100 LLM agents、300 LLM calls。本轮扩大到 10 州、1000 LLM agents、3000 LLM calls。

| Baseline | Old vote acc | Large vote acc | Delta | Old F1 | Large F1 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ces_demographic_only_llm` | 0.577 | 0.518 | -0.059 | 0.369 | 0.338 | -0.031 |
| `ces_party_ideology_llm` | 0.977 | 0.917 | -0.060 | 0.651 | 0.611 | -0.040 |
| `ces_survey_memory_llm_strict` | 0.977 | 0.871 | -0.105 | 0.651 | 0.578 | -0.073 |

大规模后，所有 LLM baseline 指标都下降，说明上一轮 4 州 100-agent 的结果偏乐观。最重要的变化是：`strict memory` 从接近 `party/ideology`，变成明显弱于 `party/ideology`。

这强化了前一轮报告中的一个方向性结论：LLM 最有效使用的是显式党派和意识形态信息；额外 strict survey memory 没有带来稳定增益，反而可能引入噪声或让模型更偏向默认共和党输出。

## 8. 原 7 州 vs 新增 3 州

| Group | Baseline | N responses | parse_ok | Turnout acc | Turnout brier | Vote acc | Vote macro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original_7 | `ces_demographic_only_llm` | 701 | 1.000 | 0.968 | 0.032 | 0.552 | 0.362 |
| original_7 | `ces_party_ideology_llm` | 701 | 1.000 | 0.968 | 0.032 | 0.924 | 0.615 |
| original_7 | `ces_survey_memory_llm_strict` | 701 | 1.000 | 0.968 | 0.032 | 0.878 | 0.580 |
| new_3 | `ces_demographic_only_llm` | 299 | 1.000 | 0.969 | 0.031 | 0.444 | 0.285 |
| new_3 | `ces_party_ideology_llm` | 299 | 1.000 | 0.969 | 0.031 | 0.903 | 0.602 |
| new_3 | `ces_survey_memory_llm_strict` | 299 | 1.000 | 0.969 | 0.031 | 0.857 | 0.571 |

新增 MN/VA/CO 后，`party/ideology` 仍然表现最好，但新 3 州比原 7 州更难：

| Baseline | Original 7 vote acc | New 3 vote acc | Drop |
| --- | ---: | ---: | ---: |
| `demographic-only` | 0.552 | 0.444 | -0.108 |
| `party/ideology` | 0.924 | 0.903 | -0.021 |
| `strict memory` | 0.878 | 0.857 | -0.021 |

新增三州最主要暴露的是 aggregate 层面的共和党偏斜，而不是个体层 vote accuracy 的全面崩溃。

## 9. 按州个体层结果

Weighted vote accuracy / macro F1：

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

明显模式：

1. `party/ideology` 在所有州都强于 `demographic-only`。
2. `strict memory` 在 GA/AZ/MN/VA 接近或略强，但在 WI/NC/CO 明显弱于 `party/ideology`。
3. `demographic-only` 的州别表现非常不稳定，AZ/CO/MN 尤其差。

## 10. Choice distribution

总体 LLM 输出分布：

| Baseline | N | Democrat | Republican | Not vote |
| --- | ---: | ---: | ---: | ---: |
| `ces_demographic_only_llm` | 1000 | 0.355 | 0.645 | 0.000 |
| `ces_party_ideology_llm` | 1000 | 0.431 | 0.569 | 0.000 |
| `ces_survey_memory_llm_strict` | 1000 | 0.375 | 0.624 | 0.001 |

按州输出分布：

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

解释：

1. `demographic-only` 的输出有明显州别漂移，不像一个稳定的 voter simulator。
2. `party/ideology` 的输出更接近合理的两党分布，但仍整体偏 Republican。
3. `strict memory` 输出更偏 Republican，且几乎完全不使用 `not_vote`。
4. `not_vote` 选择几乎消失，这是 hard-choice 系统后续需要单独处理的 turnout 建模问题。

## 11. Confusion matrix

以下为 unweighted LLM confusion counts，vote 指标只对真实 `democrat/republican` 计算；`not_vote` 行用于观察 turnout hard-choice 行为。

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

`party/ideology` 的错误相对均衡：60 个民主党真值被误判为共和党，12 个共和党真值被误判为民主党。`strict memory` 更强烈地减少了共和党误判为民主党，但代价是把更多民主党真值推向共和党：105 个民主党真值被误判为共和党。这解释了为什么 strict memory 的总体准确率和 aggregate margin 都更差。

## 12. 聚合到州级选举真值

`aggregate_eval_metrics.parquet` 当前 runner 只包含非 LLM 聚合指标。因此，本节将非 LLM 的 runner 指标和我从 `responses.parquet` 手动聚合出的 LLM 指标分开列出。

手动 LLM 聚合方法：

1. 对每个州、每个 LLM baseline，用 `sample_weight` 加权汇总 hard-choice one-hot 输出。
2. `pred_dem_2p = weighted_democrat / (weighted_democrat + weighted_republican)`。
3. 和 `data/processed/mit/president_state_truth.parquet` 中 2024 州级 MIT 真值比较。
4. `margin_error = 2 * (pred_dem_2p - true_dem_2p)`。

### 12.1 非 LLM 官方聚合指标

| Baseline | Dem 2P RMSE | Margin MAE | Margin bias | Winner acc |
| --- | ---: | ---: | ---: | ---: |
| `party_id_baseline` | 0.079 | 0.145 | 0.145 | 0.400 |
| `sklearn_logit_demographic_only` | 0.224 | 0.386 | 0.386 | 0.300 |
| `sklearn_logit_poll_informed` | 0.023 | 0.042 | -0.021 | 0.900 |
| `sklearn_logit_pre_only` | 0.024 | 0.041 | -0.023 | 0.900 |

### 12.2 LLM 手动聚合指标

| Baseline | Dem 2P RMSE | Margin MAE | Margin bias | Winner acc |
| --- | ---: | ---: | ---: | ---: |
| `ces_demographic_only_llm` | 0.265 | 0.467 | -0.136 | 0.500 |
| `ces_party_ideology_llm` | 0.154 | 0.262 | -0.262 | 0.700 |
| `ces_survey_memory_llm_strict` | 0.207 | 0.382 | -0.382 | 0.700 |

解释：

1. `party/ideology` 是 LLM 中最好的 aggregate baseline，但仍明显弱于 `sklearn_logit_pre_only` 和 `sklearn_logit_poll_informed`。
2. `strict memory` 比 `party/ideology` 更偏 Republican，margin bias 从 -0.262 扩大到 -0.382。
3. LLM winner accuracy 为 0.700 的原因并不代表它预测得很准，而是 10 州中有 7 个是 2024 Republican 胜州；只要模型偏 Republican，就能在 winner accuracy 上吃到结构性优势。
4. 新增 MN/VA/CO 后，`party/ideology` 和 `strict memory` 均没有预测出三个民主党胜州。

### 12.3 LLM 州级明细

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

关键观察：

1. 2024 年共和党确实赢下 7 个摇摆州，所以“预测共和党更多”本身不是问题。
2. 问题在于和真值比，LLM 的 Democratic two-party share 系统性偏低，尤其在 MN/VA/CO。
3. `party/ideology` 在 PA 和 NC 的 margin 非常接近真值，但在 MI/WI/NV 偏 Republican 很多。
4. `strict memory` 比 `party/ideology` 更偏 Republican，且没有改善新增民主党胜州。
5. `demographic-only` 的 aggregate 行为最不稳定：CO 预测 Dem 2P 只有 0.004，明显不是可信模拟。

## 13. Subgroup 风险

以下为 weighted vote accuracy 最差的 LLM subgroup，要求 subgroup n >= 25。

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

Subgroup 风险主要集中在两类：

1. `demographic-only` 对州 x 党派组合的识别非常差，尤其是 AZ/CO/NC/NV/VA 的民主党受访者。
2. `strict memory` 在 independent/other party-id 人群中不稳定，尤其 CO、NC、WI。

## 14. 对系统能力的判断

### 14.1 工程链路

本轮工程链路表现良好。

1. hard-choice 响应契约稳定，3000 次 LLM 调用全部 parse 成功。
2. 并行 workers=4 能稳定运行，吞吐约 5.16 responses/s。
3. GPU 显存峰值约 5054 MiB，最低剩余约 2838 MiB，没有 OOM 风险。
4. partial checkpoint、runtime log、cache 和最终产物都正常生成。
5. 1000-agent E01 在约 12.2 分钟完成，远低于 30 分钟预算。

这说明当前系统已经能支撑更大规模的本地 LLM evaluation。

### 14.2 模拟器有效性

模拟器作为“个体 vote hard-choice 分类器”时，最强 LLM 条件是 `party/ideology`：

| 比较 | 结论 |
| --- | --- |
| party/ideology vs demographic-only | 大幅提升，vote acc 0.518 -> 0.917 |
| strict memory vs party/ideology | 下降，vote acc 0.917 -> 0.871 |
| party/ideology vs party_id_baseline | 略强，0.917 vs 0.868 |
| party/ideology vs sklearn_pre_only | 明显弱，0.917 vs 0.947 |
| party/ideology vs sklearn_poll_informed | 明显弱，0.917 vs 0.954 |

这表明 qwen3.5:2b 能有效使用显式党派/意识形态信息，但 strict memory 没有表现出“更像真实受访者”的稳定收益。

### 14.3 作为州级选举模拟器

作为 aggregate election simulator，LLM 仍然不足。

1. `party/ideology` 的 winner accuracy 是 0.700，但这是因为 10 州里 7 个真值是 Republican win；不能单独看 winner accuracy。
2. 更关键的 margin 和 dem_2p 指标显示 LLM 有系统性 Republican bias。
3. 新增民主党胜州 MN/VA/CO 后，`party/ideology` 和 `strict memory` 都未能预测出 Democratic winner。
4. `sklearn_logit_pre_only` 和 `sklearn_logit_poll_informed` 在 aggregate 层远强于 LLM。

因此，当前 qwen3.5:2b + hard-choice persona prompt 不适合直接作为州级选举预测器。它更像一个“能读懂党派字段的个体分类器”，而不是一个可靠的 election simulator。

## 15. 是否改变前一轮结论

本轮没有推翻前一轮结论，而是让结论更清晰：

1. **party/ideology 是最有效的信息源。**  
   这是本轮最稳定的发现。加入党派和意识形态后，LLM 个体 vote accuracy 从 0.518 提升到 0.917。

2. **strict memory 没有稳定增益。**  
   在小规模 E01 中 strict memory 接近 party/ideology；在 10 州 1000-agent 后，strict memory 明显退化。

3. **LLM 有 Republican skew。**  
   这不是因为 2024 共和党赢下 7 个摇摆州本身，而是因为和 MIT 真值比较后，LLM 在 Democratic 2P share 上系统性偏低，尤其在 MN/VA/CO。

4. **turnout 仍未真正解决。**  
   hard-choice contract 工程上成功，但 LLM 几乎不选择 `not_vote`。当前 turnout accuracy 高，主要来自样本中投票者占比高，而不是模型真的能判断不投票。

5. **统计模型仍是强参考。**  
   `sklearn_logit_pre_only` 和 `sklearn_logit_poll_informed` 在个体和聚合指标上都明显优于 LLM。

## 16. 建议

下一步如果继续扩大或改造系统，我建议优先做三件事：

1. **把 turnout 和 vote choice 拆成两阶段任务。**  
   现在三分类 hard-choice 在工程上可靠，但 LLM 基本不用 `not_vote`。可以先让模型判断 `vote/not_vote`，若 vote 再判断 `democrat/republican`，并分别评估。

2. **保留 party/ideology 作为 LLM 主 baseline，谨慎使用 strict memory。**  
   strict memory 当前像是在引入额外噪声或触发默认 Republican 倾向。后续需要拆解 memory 字段，检查哪些字段造成退化。

3. **后续大规模 aggregate 实验应加入民主党胜州。**  
   只看 7 个共和党胜摇摆州会高估偏 Republican 模型的 winner accuracy。MN/VA/CO 这类对照州非常有价值。

总体判断：E01-large 正式通过，工程链路可靠，结果可复用；但从科学评估角度看，当前 qwen3.5:2b LLM persona simulator 仍不能被视为可靠的州级选举模拟器。最稳定的能力是利用显式党派/意识形态做个体 vote classification。
