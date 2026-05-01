# E03-Large 10-State / 1000-Agent Prompt Robustness Report

生成时间：2026-05-01  
实验目录：`data/runs/eval_suite_local/03_prompt_robustness_large_10state_1000`  
配置文件：`configs/eval_suite/e03_prompt_robustness_large_10state_1000.yaml`  
旧 E03 目录保留：`data/runs/eval_suite_local/03_prompt_robustness`

## 1. 实验目标

本轮实验是 E03 Prompt Robustness 的扩大版，目标是评估 `qwen3.5:2b` 在同一 persona、同一 strict pre-election memory 信息下，对不同 prompt 表述是否稳定。

核心问题：

1. 语义等价或近似等价的 prompt variant 是否会改变 hard-choice 输出？
2. candidate order 变化是否会系统性改变 Democrat/Republican 选择？
3. interviewer-style、analyst-style、strict JSON wording 是否只是表面变化，还是会改变模拟结论？
4. 大样本是否确认旧 E03 小样本中看到的 candidate-order sensitivity？

本轮继续使用 hard-choice 契约：

```json
{"choice": "not_vote|democrat|republican"}
```

旧概率鲁棒性指标不再作为主指标。报告中所有旧概率列只解释为 hard-choice 派生 one-hot 或聚合频率。

## 2. 设计

### 2.1 States and Sample

本轮采用 10 州，每州 100 agents：

| Group | States | Agents |
| --- | --- | ---: |
| 原 7 个 2024 Republican 胜摇摆州 | PA, MI, WI, GA, AZ, NV, NC | 700 |
| 新增 3 个 2024 Democratic 胜州 | MN, VA, CO | 300 |
| 合计 | PA, MI, WI, GA, AZ, NV, NC, MN, VA, CO | 1000 |

正式规模：

```text
1000 agents x 5 prompt variants = 5000 LLM calls
```

### 2.2 Prompt Variants

本轮沿用旧 E03 的 5 个 variant，保证可比：

| Variant | 说明 |
| --- | --- |
| `base_json` | 当前 strict-memory hard-choice JSON prompt |
| `json_strict_nonzero` | 同信息，但更强调 JSON-only 和 hard-choice schema |
| `candidate_order_reversed` | 同信息，但反转候选人/上下文顺序 |
| `interviewer_style` | 调整为调查访问员式措辞 |
| `analyst_style` | 调整为更简洁的 analyst-style hard-choice 措辞 |

Pairwise 指标都以 `base_json` 为参照。

## 3. 预检查和探针

预检查全部通过：

| 检查项 | 结果 |
| --- | --- |
| `py_compile` | 通过 |
| E03/parser 目标 pytest | 通过 |
| Ollama `/api/tags` | 可见 `qwen3.5:2b` |
| processed CES artifacts | 全部存在 |
| 初始 GPU | RTX 5070 Laptop GPU，8151 MiB total，约 6748 MiB free |

探针结果：

| Probe | States | Agents | Calls | Workers | LLM runtime | Median | P90 | Throughput | GPU peak used | Min free | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | PA, GA, MN, VA, CO | 10 | 50 | 1 | 27.2s | 0.429s | 0.453s | 1.84/s | 5002 MiB | 2890 MiB | PASS |
| B | 10 states | 30 | 150 | 4 | 38.0s | 0.935s | 1.075s | 3.94/s | 5002 MiB | 2890 MiB | PASS |
| C | 10 states | 30 | 150 | 5 | 37.6s | 1.146s | 1.332s | 3.99/s | 5002 MiB | 2890 MiB | PASS |

Workers=5 只比 workers=4 快约 1%，没有达到“至少快 10%”的采用条件，且 p90 latency 更高。因此正式运行使用 workers=4。

探针中没有出现：

| 异常 | 结果 |
| --- | --- |
| parse failure | 0 |
| invalid choice | 0 |
| forbidden choice | 0 |
| legacy probability schema | 0 |
| transport error | 0 |
| GPU OOM | 0 |

## 4. 正式运行概况

正式运行完成，无中断。

| 指标 | 值 |
| --- | ---: |
| Workers | 4 |
| Agents | 1000 |
| Prompt variants | 5 |
| Prompts / responses | 5000 |
| Ollama calls | 4995 |
| Cache hit rate | 0.001 |
| Total wall-clock | 1169.2s，约 19.5 min |
| LLM runtime | 1147.6s，约 19.1 min |
| Median latency | 0.874s |
| P90 latency | 0.932s |
| Throughput | 4.36 responses/s |
| GPU peak used | 5004 MiB |
| GPU min free | 2888 MiB |
| GPU peak utilization | 83% |

正式运行低于 30 分钟预算，GPU 利用率较高且没有接近 OOM。

## 5. 输出产物

正式目录：

`data/runs/eval_suite_local/03_prompt_robustness_large_10state_1000`

核心输出：

| 文件 | Shape / 说明 |
| --- | --- |
| `agents.parquet` | 1000 x 32 |
| `prompts.parquet` | 5000 x 14 |
| `responses.parquet` | 5000 x 29 |
| `prompt_variant_metadata.parquet` | 5 x 2 |
| `robustness_metrics.parquet` | 55 x 7 |
| `pairwise_variant_metrics.parquet` | 48 x 8 |
| `runtime_log.parquet` | 5000 x 16 |
| `runtime.json` | 运行观测 |
| `llm_cache.jsonl` | LLM cache |
| `report.md` | runner 自动报告 |
| `figures/*` | E03 图表 |

图表：

| Figure |
| --- |
| `e03_choice_flip_rate_vs_base.png` |
| `e03_parse_ok_by_variant.png` |
| `e03_state_margin_shift_by_variant.png` |
| `e03_turnout_choice_flip_rate_vs_base.png` |

## 6. 质量门

总体 runtime gates：

| Gate | 值 | 结果 |
| --- | ---: | --- |
| parse_ok_rate | 1.000 | PASS |
| invalid_choice_rate | 0.000 | PASS |
| forbidden_choice_rate | 0.000 | PASS |
| legacy_probability_schema_rate | 0.000 | PASS |
| transport_error_rate | 0.000 | PASS |
| all_gates_passed | true | PASS |

按 variant：

| Variant | Parse OK | Invalid choice | Legacy schema | Transport error | Median latency | Vote acc | Turnout brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_json` | 1.000 | 0.000 | 0.000 | 0.000 | 0.893 | 0.886 | 0.022 |
| `json_strict_nonzero` | 1.000 | 0.000 | 0.000 | 0.000 | 0.869 | 0.900 | 0.024 |
| `candidate_order_reversed` | 1.000 | 0.000 | 0.000 | 0.000 | 0.868 | 0.822 | 0.022 |
| `interviewer_style` | 1.000 | 0.000 | 0.000 | 0.000 | 0.864 | 0.900 | 0.022 |
| `analyst_style` | 1.000 | 0.000 | 0.000 | 0.000 | 0.866 | 0.881 | 0.022 |

工程结论很明确：E03 的问题不是解析失败，也不是 JSON 契约失败。所有 variant 都严格遵守 hard-choice schema。

## 7. Choice Distribution by Variant

总体 choice distribution：

| Variant | N | Democrat | Republican | Not vote |
| --- | ---: | ---: | ---: | ---: |
| `base_json` | 1000 | 0.401 | 0.598 | 0.001 |
| `json_strict_nonzero` | 1000 | 0.415 | 0.570 | 0.015 |
| `candidate_order_reversed` | 1000 | 0.329 | 0.671 | 0.000 |
| `interviewer_style` | 1000 | 0.414 | 0.585 | 0.001 |
| `analyst_style` | 1000 | 0.385 | 0.615 | 0.000 |

观察：

1. 所有 variant 都整体偏 Republican。
2. `candidate_order_reversed` 明显最偏 Republican，Democrat share 从 base 的 40.1% 降到 32.9%。
3. `interviewer_style` 和 `json_strict_nonzero` 反而略微提高 Democrat share。
4. `json_strict_nonzero` 是唯一明显增加 `not_vote` 的 variant，但也只有 1.5%。
5. turnout 问题仍然存在：所有 variant 的 `not_vote` 比例都很低。

## 8. Pairwise Robustness vs Base

与 `base_json` 对比：

| Variant | Choice flip rate | Turnout choice flip rate |
| --- | ---: | ---: |
| `json_strict_nonzero` | 0.039 | 0.014 |
| `candidate_order_reversed` | 0.073 | 0.001 |
| `interviewer_style` | 0.021 | 0.000 |
| `analyst_style` | 0.017 | 0.001 |

解释：

1. 按 hard-choice flip rate 看，所有 variant 都低于 10% 的旧阈值。
2. `candidate_order_reversed` 是最敏感 variant，choice flip rate 7.3%。
3. `interviewer_style` 和 `analyst_style` 相对稳定，flip rate 分别为 2.1% 和 1.7%。
4. `json_strict_nonzero` 的 turnout choice flip 稍高，因为它多输出了一些 `not_vote`。

如果只看 individual choice flip，E03-large 会被判为“基本鲁棒”。但 aggregate margin shift 显示，candidate order 的影响比 flip rate 看起来更严重。

## 9. State Margin Shift

`state_margin_shift = variant_margin - base_json_margin`。负数表示 variant 比 base 更偏 Republican。

| Variant | Max abs state margin shift | Mean abs state margin shift |
| --- | ---: | ---: |
| `json_strict_nonzero` | 0.081 | 0.038 |
| `candidate_order_reversed` | 0.398 | 0.128 |
| `interviewer_style` | 0.065 | 0.024 |
| `analyst_style` | 0.062 | 0.014 |

州级明细：

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

核心结论：

1. `candidate_order_reversed` 是明确的 prompt-order sensitivity 来源。
2. 它在所有 10 州的 margin shift 都为负，即全部向 Republican 方向移动。
3. 最大 shift 是 AZ 的 -0.398，这是非常大的 aggregate shift。
4. 旧 E03 中 PA 曾出现 -0.425 的大幅 shift；大样本后 PA shift 降到 -0.156，说明旧 PA 极值部分是小样本放大，但方向和敏感性不是偶然。
5. 新 E03-large 显示更普遍的问题：candidate-order shift 不只发生在 PA，而是 10 州全部同方向偏 Republican。

## 10. Group-Level Robustness

按 10 州总体、原 7 州、新增 3 州分解：

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

解释：

1. `candidate_order_reversed` 在原 7 州和新增 3 州都向 Republican 方向移动。
2. 原 7 州 shift 更大：-0.141；新增 3 州 shift 为 -0.069。
3. `interviewer_style` 和 `json_strict_nonzero` 轻微向 Democrat 方向移动，但幅度远小于 candidate-order shift。
4. `analyst_style` 和 base 几乎等价，整体 margin shift 只有 -0.013。

## 11. 与旧 E03 的比较

旧 E03：

| 指标 | 值 |
| --- | ---: |
| Agents | 60 |
| Prompts | 300 |
| Workers | 4 |
| Runtime | 75.0s |
| Parse OK | 1.000 |

旧 E03 pairwise choice flip：

| Variant | Old choice flip | Large choice flip |
| --- | ---: | ---: |
| `json_strict_nonzero` | 0.000 | 0.039 |
| `candidate_order_reversed` | 0.067 | 0.073 |
| `interviewer_style` | 0.000 | 0.021 |
| `analyst_style` | 0.000 | 0.017 |

结论：

1. 旧 E03 已经提示 `candidate_order_reversed` 是最敏感 variant。
2. 大样本后该结论被确认：choice flip 从 6.7% 到 7.3%，非常接近。
3. 旧 E03 中其他 variant 看似完全稳定，但大样本显示它们也有 1.7%-3.9% 的小幅 flip。
4. 大样本让结论更细：prompt wording 总体可控，但 candidate order 会产生系统性 aggregate shift。

## 12. 对 E01/E02-Large 结论的影响

本轮 E03-large 不推翻 E01/E02-large，反而补充了一个重要机制：

1. 当前模型的主要问题仍然是系统性 Republican skew。
2. prompt 解析本身稳定，不是主要失败源。
3. prompt 表述中的候选顺序会放大 Republican skew，尤其是 `candidate_order_reversed`。
4. interviewer/analyst wording 本身影响较小，说明不是所有 wording 改动都会导致不稳定。
5. 后续所有正式 prompt 必须固定候选人顺序，并把 candidate-order sensitivity 作为报告中的已知风险。

## 13. 结论

### 13.1 工程结论

E03-large 工程链路通过：

1. 5000 次 LLM 调用全部 parse 成功。
2. 无 invalid choice、forbidden choice、legacy probability schema、transport error。
3. workers=4 稳定，正式墙钟约 19.5 分钟，低于 30 分钟预算。
4. GPU 峰值显存约 5004 MiB，最低剩余约 2888 MiB，没有 OOM 风险。
5. partial checkpoint 和最终产物完整。

### 13.2 科学结论

当前 `qwen3.5:2b` hard-choice simulator 对普通 wording 改动基本稳定，但对 candidate order 有明确敏感性。

更具体地说：

1. `interviewer_style` 和 `analyst_style` 与 base 差异小。
2. `json_strict_nonzero` 略微增加 `not_vote` 和 Democrat share，但总体仍稳定。
3. `candidate_order_reversed` 是主要风险：choice flip 7.3%，全 10 州 margin 都向 Republican 方向移动。
4. 这说明模型并不只是读取 persona 后做稳定分类；它仍会受到 prompt surface form 的系统性影响。
5. 作为 MVP 和小模型，这个偏误正常，但在报告中必须如实标注，不能把单一 prompt 的 aggregate 结果解释得过强。

## 14. 建议

后续建议：

1. 固定并版本化正式 prompt 的候选人顺序。
2. 最终评估报告中把 E03 的 candidate-order sensitivity 作为模型风险，而不是工程 bug。
3. 若后续要改善系统，可考虑对候选顺序做 counterbalanced prompting，再对两个顺序的输出做聚合或一致性检查。
4. 继续优先解决 E01/E02 暴露的 Republican skew 和 turnout 几乎不输出 `not_vote` 的问题。

本轮 E03-large 正式通过，结果可复用。结论不是“系统坏了”，而是“小模型 MVP 在 prompt surface form 上有可测、可解释的偏误”。
