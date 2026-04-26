# CES 2024 接入工程蓝图

本文档面向 `LM_Voting_Simulator` 仓库的下一轮改造。目标是让开发者或 coding agent 看到本文档和现有仓库后，可以直接开始实现 CES 2024 接入，而不需要重新理解整个研究设想。

---

## 0. 当前仓库状态

仓库当前已经有一个可运行的 MVP：

- `README.md` 给出了 fixture-backed end-to-end run，以及 real ANES 2024 one-agent smoke run。
- CLI 已有 `build-anes`、`build-anes-memory`、`build-ces-cells`、`build-mit-results`、`build-agents`、`run-simulation`、`evaluate`。
- 当前主线大致是：

```text
ANES raw
  -> anes_respondents.parquet
  -> anes_answers.parquet
  -> anes_memory_facts.parquet
  -> anes_memory_cards.parquet

CES raw / fixture
  -> ces_respondents.parquet
  -> ces_cell_distribution.parquet

CES cells + ANES memory cards
  -> synthetic agents
  -> prompt
  -> LLM / baseline response
  -> state aggregation
  -> MIT evaluation
```

也就是说，现在 CES 已经存在，但只承担 “人口 cell 分布 / empirical baseline” 的角色；真实个体级政治态度和选后标签主要没有被使用。下一步的核心改造是把 CES 2024 Common Content 升级为主数据源。

---

## 1. 本次改造目标

### 1.1 核心目标

把 CES 2024 接入为完整的 respondent-level election simulation backbone：

```text
CES 2024 Common Content CSV
  -> CES respondent profile
  -> CES pre-election answers
  -> CES memory facts / memory cards
  -> CES row-level agents
  -> LLM turnout + vote-choice simulation
  -> CES post-election / TargetSmart individual evaluation
  -> MIT Election Lab aggregate evaluation
```

实现后，系统应该支持两条主线：

```text
主线 A：CES row-level simulation
真实 CES respondent row -> agent -> LLM prediction -> individual + aggregate evaluation

保留线 B：CES cells + ANES archetypes
CES cell distribution -> ANES respondent matching -> synthetic agents -> simulation
```

主线 A 是之后的默认路线。保留线 B 用于兼容旧 demo、实验对照和未来 ANES enrichment。

### 1.2 非目标

本次不要求：

- 一次性接入所有 CES 变量。
- 一次性接入 ANES + CES 个体级 join。
- 一次性做完整 GDELT 事件动态传播。
- 一次性完成 Senate / House / Governor 全部 office。
- 一次性实现复杂社交网络传播。

第一阶段只需要让总统选举的 CES respondent-level pipeline 跑通。

---

## 2. CES 2024 数据集在项目中的定位

### 2.1 CES 包含什么

CES 2024 Common Content 是一个 60,000 respondent 的美国选举问卷数据集。它包含：

```text
sample identifiers
profile / demographic questions
pre-election questions
post-election questions
contextual data
vote validation variables
```

在项目里可以拆成这些用途：

| 数据层 | 代表内容 | 项目用途 |
|---|---|---|
| profile | 年龄、性别、种族、教育、收入、州、地区、选区 | agent 基础身份 |
| political identity | pid3、pid7、ideology、自报注册党派 | agent 政治先验 |
| media exposure | 社媒、电视新闻、报纸、广播、新闻频道 | 信息暴露 proxy |
| economy | national economy、household income、prices、emergency expense | 经济投票机制 |
| issue attitudes | 枪支、移民、堕胎、环境、税、医保、外交 | issue vector / prompt facts |
| approval / perception | Biden、Harris、Trump、党派、国会、最高法院、州长、议员评价 | candidate / institution perception |
| pre-vote questions | 投票意向、总统/参院/众院/州长偏好 | 可选 poll-informed prior；strict mode 默认排除 |
| post-vote questions | 是否投票、投给谁、投票方式、没投原因 | individual labels |
| TargetSmart validation | 注册匹配、2024 turnout/mode、党派注册 | turnout validation |
| contextual fields | 候选人姓名、党派、州、选区、办公室 | candidate context |

### 2.2 CES 与 ANES / MIT / GDELT 的关系

新的数据角色：

```text
CES 2024
  主数据源。负责 agent 初始化、pre-election memory、post-election labels、turnout validation、权重。

ANES 2024
  辅助数据源。负责更深的政治心理变量、对照实验、未来 prompt enrichment。
  不再是总统 MVP 的必需项。

MIT Election Lab
  official aggregate ground truth。用于州级 / 全国级结果评估。

GDELT
  news/event stream。第一阶段暂不接入主线；第二阶段作为 event context 加入 prompt 或 salience update。
```

---

## 3. 需要新增或修改的功能

## 3.1 新增 `build-ces`：CES respondent-level ingest

当前仓库有 `build-ces-cells`，但它只产出 cell distribution。需要新增真正的 CES respondent-level ingest。

建议 CLI：

```bash
python -m election_sim.cli build-ces \
  --config configs/datasets/ces_2024_real_vv.yaml \
  --profile-crosswalk configs/crosswalks/ces_2024_profile.yaml \
  --question-crosswalk configs/crosswalks/ces_2024_pre_questions.yaml \
  --target-crosswalk configs/crosswalks/ces_2024_targets.yaml \
  --context-crosswalk configs/crosswalks/ces_2024_context.yaml \
  --out data/processed/ces/2024_common_vv
```

建议输出：

```text
data/processed/ces/2024_common_vv/
  ces_respondents.parquet
  ces_answers.parquet
  ces_targets.parquet
  ces_context.parquet
  ces_question_bank.parquet
  ces_ingest_report.md
```

其中：

### `ces_respondents.parquet`

一行一个 respondent。只放 agent profile 和常用权重，不放所有问卷长表。

建议字段：

```text
ces_id
source_year
tookpost

state_po
state_fips
county_fips
county_name
cdid
region

birthyr
age
age_group
gender
race_ethnicity
hispanic
education_detail
education_binary
income_bin
employment
religion
bornagain
marital_status

party_id_3_pre
party_id_7_pre
party_id_3_post
party_id_7_post
ideology_self_7
registered_self_pre
registered_self_post
party_registration_self
party_registration_validated

weight_common
weight_common_post
weight_vv
weight_vv_post

validated_registration
validated_turnout_2024
validated_vote_mode_2024

schema_version
```

### `ces_answers.parquet`

long format。只放可用于 prompt 的 pre-election answers 和少量非标签型 profile answers。

建议字段：

```text
ces_id
wave                         # pre / post / validation / context
question_id
source_variable
question_text
answer_code
answer_label
canonical_value
topic
is_multiselect
is_grid_item
is_pre_election
allowed_for_memory_strict
leakage_group
schema_version
```

### `ces_targets.parquet`

只放监督标签和评估标签，不进入 prompt。

建议字段：

```text
ces_id
target_id                    # turnout_2024, president_vote_2024, president_preference_2024, ...
source_variable
target_type                  # turnout / vote_choice / preference / vote_mode
answer_code
answer_label
canonical_value
truth_source                 # ces_post_self_report / targetsmart_validation / ces_pre_intent
weight_column_recommended
schema_version
```

### `ces_context.parquet`

放 candidate/context 变量。第一阶段总统可以很简单，但结构要预留。

建议字段：

```text
ces_id
year
office                       # president / senate / house / governor
state_po
district
candidate_slot
candidate_name
candidate_party
candidate_incumbent
context_source_variable
schema_version
```

---

## 3.2 新增 `build-ces-memory`：CES memory facts / cards

现有 `build-anes-memory` 的思路可以复用，但要把 memory 逻辑抽成通用 survey memory。

建议新增模块：

```text
src/election_sim/survey_memory.py
```

负责：

```text
build_memory_facts()
build_memory_cards()
LeakageGuard
memory policy
fact template rendering
```

然后：

```text
anes.py 只负责 ANES ingest
ces.py 只负责 CES ingest
survey_memory.py 负责 ANES/CES 通用 memory
```

建议 CLI：

```bash
python -m election_sim.cli build-ces-memory \
  --respondents data/processed/ces/2024_common_vv/ces_respondents.parquet \
  --answers data/processed/ces/2024_common_vv/ces_answers.parquet \
  --fact-templates configs/fact_templates/ces_2024_common_facts.yaml \
  --policy strict_pre_no_vote_v1 \
  --out data/processed/ces/2024_common_vv \
  --max-facts 24
```

建议输出：

```text
ces_memory_facts.parquet
ces_memory_cards.parquet
ces_leakage_audit.parquet
```

### `ces_memory_facts.parquet`

```text
memory_fact_id
ces_id
source_variable
question_id
topic
fact_text
fact_priority
memory_policy
leakage_group
created_at
```

### `ces_memory_cards.parquet`

```text
memory_card_id
ces_id
memory_policy
fact_ids
memory_text
n_facts
created_at
```

### `ces_leakage_audit.parquet`

记录被排除的变量和原因：

```text
source_variable
question_id
policy
excluded
reason
target_id
```

---

## 3.3 新增 leakage policy

CES 有 pre intention、post vote、TargetSmart turnout validation。必须明确哪些能进 prompt，哪些只能做 label。

建议第一版实现三种 policy：

```text
strict_pre_no_vote_v1
poll_informed_pre_v1
post_hoc_explanation_v1
```

### `strict_pre_no_vote_v1`

默认预测模式。

禁止进入 prompt：

```text
post-election vote choice
post-election turnout
TargetSmart validation
pre-election direct vote intention
pre-election direct candidate preference
```

允许进入 prompt：

```text
demographics
party ID
ideology
issue attitudes
economy perceptions
media use
approval ratings
candidate / party ideological placement
```

### `poll_informed_pre_v1`

民调增强模式。

禁止进入 prompt：

```text
post-election vote choice
post-election turnout
TargetSmart validation
```

允许进入 prompt：

```text
pre-election turnout intention
pre-election candidate preference
```

这些 facts 需要标记为 `poll_prior`，报告里必须单独说明。

### `post_hoc_explanation_v1`

只用于解释性分析，不用于正式预测指标。

可以允许更多 post-election 非 target 变量，但仍不得把当前 target answer 直接放进 prompt。

---

## 3.4 修改 agent construction：支持 `population.source = ces_rows`

当前 `population.py` 的主线是：

```text
CES cell distribution + ANES respondent pool + ANES memory cards -> synthetic agents
```

新增 CES row-level path：

```text
CES respondents + CES memory cards -> agents
```

建议 run config：

```yaml
population:
  source: ces_rows
  selection:
    states: ["PA", "MI", "WI", "GA", "AZ", "NV", "NC"]
    tookpost_required: true
    citizen_required: true
  sampling:
    mode: stratified_state_sample       # all_rows | weighted_sample | stratified_state_sample
    n_agents_per_state: 500
    random_seed: 2024
  weight:
    column: commonpostweight
```

建议 `agents.parquet` 支持兼容旧字段，同时加入 CES 字段：

```text
run_id
agent_id
year
state_po
source_dataset                 # ces / anes / synthetic
source_respondent_id           # CES: ces_id
base_anes_id                   # legacy nullable
base_ces_id                    # new
memory_card_id
sample_weight
weight_column
cell_schema
cell_id

age_group
gender
race_ethnicity
education_binary
income_bin
party_id_3
party_id_7
ideology_7
registered_self_pre
validated_registration

created_at
```

对旧模式：

```yaml
population:
  source: ces_cells_anes_archetypes
```

继续走现有 `match_archetypes()`。

---

## 3.5 修改 prompt rendering：从三种硬编码模板升级为 block-based template

当前 `prompts.py` 中模板只有：

```text
demographic_only
party_ideology
survey_memory
```

CES 接入后，建议新增一个可配置模板：

```text
ces_vote_v1
ces_turnout_v1
```

第一阶段不必实现非常复杂的模板系统，但 prompt 应至少分块：

```text
1. role instruction
2. voter profile
3. political identity
4. survey-derived facts
5. candidate context
6. task instruction
7. JSON schema
```

总统 vote prompt 示例结构：

```text
You are simulating how a specific U.S. eligible voter would behave in the 2024 general election.

Voter profile:
- State: PA
- Age group: 45-64
- Gender: Woman
- Race/ethnicity: White
- Education: Some college
- Party identification: Independent, leans Republican
- Ideology: Somewhat conservative

Survey-derived background facts:
- The voter says prices of everyday goods and services increased a lot over the past year.
- The voter strongly disapproves of President Biden.
- The voter supports increasing border patrols.
- ...

Election context:
- Office: President
- Democratic candidate: Kamala Harris
- Republican candidate: Donald Trump

Task:
Estimate the voter's turnout probability and presidential vote choice.

Return JSON only:
{
  "turnout_probability": 0.0,
  "vote_probabilities": {
    "democrat": 0.0,
    "republican": 0.0,
    "other": 0.0,
    "undecided": 0.0
  },
  "most_likely_choice": "democrat|republican|other|undecided|not_vote",
  "confidence": 0.0
}
```

---

## 3.6 修改 response schema：支持 turnout + probability vector

当前 response 主要是：

```text
parsed_answer_code
confidence
probabilities_json
```

建议扩展为：

```text
run_id
response_id
prompt_id
agent_id
question_id
task_id
baseline
model_name

raw_response
parse_status
repair_attempts

turnout_probability
vote_prob_democrat
vote_prob_republican
vote_prob_other
vote_prob_undecided
most_likely_choice
confidence

parsed_answer_code          # legacy fallback
probabilities_json          # legacy / optional
created_at
```

为了兼容旧 evaluator，可以先保留 `parsed_answer_code`。

---

## 3.7 修改 aggregation：turnout-aware weighted aggregation

当前 aggregate 是 hard vote share：

```text
sum(sample_weight * vote_indicator) / sum(sample_weight)
```

CES 主线需要 turnout-aware expected vote：

```text
expected_dem_votes =
  sum(sample_weight * turnout_probability * vote_prob_democrat)

expected_rep_votes =
  sum(sample_weight * turnout_probability * vote_prob_republican)

dem_2p =
  expected_dem_votes / (expected_dem_votes + expected_rep_votes)
```

建议输出：

```text
aggregate_state_results.parquet
```

字段：

```text
run_id
year
state_po
office
baseline
model_name
n_agents
weight_column

expected_turnout
expected_dem_votes
expected_rep_votes
expected_other_votes
expected_undecided_or_not_vote

dem_share_raw
rep_share_raw
other_share_raw
dem_share_2p
rep_share_2p
margin_2p
winner

created_at
```

---

## 3.8 修改 evaluation：同时支持 individual 与 aggregate

CES 主线新增 individual evaluation：

```text
individual turnout prediction vs TS_g2024 / CC24_401
individual vote choice vs CC24_410
```

保留 aggregate evaluation：

```text
state-level vote share vs MIT official results
```

建议输出：

```text
individual_eval_metrics.parquet
aggregate_eval_metrics.parquet
subgroup_eval_metrics.parquet
eval_report.md
```

### Individual metrics

```text
turnout:
  Brier score
  accuracy at threshold 0.5
  AUC if sklearn available

vote choice:
  accuracy
  log loss if probability vector available
  macro F1
  parse_ok_rate
```

### Aggregate metrics

```text
state dem_2p RMSE
state margin MAE
winner accuracy
national dem_2p error
```

### Subgroup metrics

第一阶段可选，但 schema 先预留：

```text
group_by:
  party_id_3
  race_ethnicity
  age_group
  education_binary
  gender
```

---

## 4. 数据 pipeline

## 4.1 第一阶段：CES presidential smoke run

目标：只跑少量 respondent，验证真实 CES prompt + LLM + label evaluation 能通。

```text
raw CES CSV
  -> build-ces
  -> build-ces-memory(policy=strict_pre_no_vote_v1)
  -> build-agents(source=ces_rows, n=10)
  -> run-simulation(task=president_turnout_vote)
  -> evaluate(individual only)
```

示例命令：

```bash
python -m election_sim.cli build-ces \
  --config configs/datasets/ces_2024_real_vv.yaml \
  --profile-crosswalk configs/crosswalks/ces_2024_profile.yaml \
  --question-crosswalk configs/crosswalks/ces_2024_pre_questions.yaml \
  --target-crosswalk configs/crosswalks/ces_2024_targets.yaml \
  --context-crosswalk configs/crosswalks/ces_2024_context.yaml \
  --out data/processed/ces/2024_common_vv

python -m election_sim.cli build-ces-memory \
  --respondents data/processed/ces/2024_common_vv/ces_respondents.parquet \
  --answers data/processed/ces/2024_common_vv/ces_answers.parquet \
  --fact-templates configs/fact_templates/ces_2024_common_facts.yaml \
  --policy strict_pre_no_vote_v1 \
  --out data/processed/ces/2024_common_vv \
  --max-facts 24

python -m election_sim.cli run-simulation \
  --run-config configs/runs/ces_2024_president_smoke.yaml
```

成功标准：

```text
能生成 agents.parquet
能生成 prompts.parquet
能生成 responses.parquet
prompt 不含 post vote / TargetSmart / direct pre vote leakage
response JSON 可解析
能对照 CES target 计算 individual metric
```

---

## 4.2 第二阶段：swing states state-level run

目标：在 7 个关键州做 state-level 聚合，并与 MIT 结果比较。

```text
CES processed artifacts
  -> build-agents(source=ces_rows, states=swing_states, n_agents_per_state=500)
  -> run-simulation
  -> turnout-aware aggregation
  -> MIT state evaluation
```

Run config 示例：

```yaml
run_id: ces_2024_president_swing_strict_pre
mode: ces_election_simulation

scenario:
  year: 2024
  office: president
  states: ["PA", "MI", "WI", "GA", "AZ", "NV", "NC"]
  election_day: "2024-11-05"

paths:
  ces_respondents: data/processed/ces/2024_common_vv/ces_respondents.parquet
  ces_answers: data/processed/ces/2024_common_vv/ces_answers.parquet
  ces_targets: data/processed/ces/2024_common_vv/ces_targets.parquet
  ces_context: data/processed/ces/2024_common_vv/ces_context.parquet
  ces_memory_cards: data/processed/ces/2024_common_vv/ces_memory_cards.parquet
  ces_memory_facts: data/processed/ces/2024_common_vv/ces_memory_facts.parquet
  mit_results: data/processed/mit/2024/mit_election_results.parquet

population:
  source: ces_rows
  selection:
    tookpost_required: true
    citizen_required: true
  sampling:
    mode: stratified_state_sample
    n_agents_per_state: 500
  weight:
    column: commonpostweight

memory:
  source: ces
  memory_policy: strict_pre_no_vote_v1
  max_memory_facts: 24

tasks:
  - president_turnout_vote

prompt:
  template: ces_president_vote_v1
  include_candidate_context: true
  include_gdelt_context: false

model:
  provider: ollama
  base_url: http://127.0.0.1:11434
  model_name: qwen3.5:9b-q4_K_M
  temperature: 0.0

baselines:
  - party_id_baseline
  - sklearn_logit_pre_only
  - survey_memory_llm

evaluation:
  individual:
    enabled: true
    turnout_truth: TS_g2024
    vote_truth: CC24_410
  aggregate:
    enabled: true
    truth: mit_president_state_2024
```

成功标准：

```text
每个州生成 dem_2p、margin、winner
报告 state RMSE / margin MAE / winner accuracy
报告 individual parse rate / accuracy / Brier score
报告使用的 weight column 与 leakage policy
```

---

## 4.3 第三阶段：all-state / national run

目标：全国所有州聚合，与 MIT official result 比较。

```yaml
scenario:
  states: all
population:
  sampling:
    mode: weighted_sample
    n_total_agents: 5000
```

第一版不建议直接让 60,000 行全调用 LLM。可以先做：

```text
5000 respondent weighted sample
or
每州 200-500 respondent stratified sample
```

如果使用非 LLM baseline 或本地廉价模型，可以全量跑。

---

## 4.4 第四阶段：GDELT event context

GDELT 只在 CES 主线跑通后加入。

新增数据：

```text
gdelt_events.parquet
event_context_cards.parquet
```

prompt 加：

```text
Recent news context:
- ...
```

实验对比：

```text
no_event_context
event_context_national
event_context_state_specific
event_context_issue_matched
```

---

## 5. 建议的模块改造

## 5.1 `ces.py`

当前 `ces.py` 可以保留 cell distribution 函数，但需要新增 respondent-level builder。

建议拆分：

```text
ces.py
  normalize_ces_respondents()
  build_ces_answers()
  build_ces_targets()
  build_ces_context()
  build_ces()
  build_ces_cells()              # legacy compatible
```

注意：

- 不要再把 CES schema 固定为十几个字段。
- 不要在 `ces_respondents` 中塞所有问卷列。
- 问卷列统一进 `ces_answers` long format。
- 标签统一进 `ces_targets`。

---

## 5.2 `survey_memory.py`

新增通用模块。

```text
survey_memory.py
  FactTemplate
  LeakageGuard
  build_memory_facts()
  build_memory_cards()
  filter_facts_for_target()
```

`anes.py` 中现有 `build_memory_cards` 和 `LeakageGuard` 迁移到这里。

---

## 5.3 `population.py`

新增：

```text
build_agents_from_ces_rows()
```

保留：

```text
build_agents_from_frames()       # legacy CES cells + ANES archetypes
match_archetypes()
```

入口根据 config 分流：

```python
if cfg.population.source == "ces_rows":
    return build_agents_from_ces_rows(...)
elif cfg.population.source == "ces_cells_anes_archetypes":
    return build_agents_from_frames(...)
```

---

## 5.4 `prompts.py`

新增：

```text
build_ces_prompt()
parse_turnout_vote_json()
```

建议短期内可以继续用 Python string / Jinja2 template，不必立刻做完整 prompt DSL。

必须支持：

```text
candidate context
CES memory facts
turnout probability
vote probability vector
```

---

## 5.5 `simulation.py`

当前 `run_simulation()` 里 fixture 主线默认会重建 ANES、CES cells、MIT。CES 主线不应该每次都强制重建 ANES。

建议新增 mode：

```text
ces_election_simulation
```

逻辑：

```python
if cfg.mode == "individual_benchmark":
    run_individual_benchmark()
elif cfg.mode == "ces_election_simulation":
    run_ces_election_simulation()
else:
    run_legacy_cell_anes_simulation()
```

`run_ces_election_simulation()` 只加载 CES processed artifacts，不依赖 ANES。

---

## 5.6 `aggregation.py`

保留 legacy hard-label aggregation；新增：

```text
aggregate_turnout_vote_state_results()
```

根据 response schema 自动选择：

```text
如果存在 turnout_probability 和 vote_prob_*，走 turnout-aware expected votes
否则回退到 parsed_answer_code hard labels
```

---

## 5.7 `evaluation.py`

新增：

```text
evaluate_individual_turnout()
evaluate_individual_vote_choice()
evaluate_aggregate_state_results()
evaluate_subgroups()
```

第一阶段最小实现：

```text
parse_ok_rate
individual vote accuracy
turnout Brier score
state dem_2p RMSE
state margin MAE
winner accuracy
```

---

## 5.8 `report.py`

新增 CES report sections：

```text
Run metadata
Dataset artifacts
Population summary
Weight choice
Leakage policy
Excluded leakage variables
Prompt fact coverage
Parse status summary
Individual metrics
Aggregate metrics
State table
Known limitations
```

---

## 6. Config 文件规划

新增目录和文件：

```text
configs/datasets/
  ces_2024_real_vv.yaml

configs/crosswalks/
  ces_2024_profile.yaml
  ces_2024_pre_questions.yaml
  ces_2024_targets.yaml
  ces_2024_context.yaml

configs/fact_templates/
  ces_2024_common_facts.yaml

configs/questions/
  ces_2024_president_turnout_vote.yaml

configs/runs/
  ces_2024_president_smoke.yaml
  ces_2024_president_swing_strict_pre.yaml
  ces_2024_president_swing_poll_informed.yaml
```

### `configs/datasets/ces_2024_real_vv.yaml`

```yaml
name: ces_2024_common_vv
year: 2024
path: data/raw/ces/CCES24_Common_OUTPUT_vv_topost_final.csv
format: csv
respondent_id: caseid
schema_version: ces_2024_common_vv_v1
```

`respondent_id` 具体列名以 CSV 实际列为准；如果不是 `caseid`，在实现时改成真实列。

### `configs/crosswalks/ces_2024_profile.yaml`

负责 profile 和权重字段。

```yaml
fields:
  state_po:
    variable: inputstate
    transform: state_fips_to_po
  gender:
    variable: gender4
    map:
      1: man
      2: woman
      3: non_binary
      4: other
  education_detail:
    variable: educ
  race_ethnicity:
    variable: race
  party_id_3_pre:
    variable: pid3
  party_id_7_pre:
    variable: pid7
  weight_common:
    variable: commonweight
  weight_common_post:
    variable: commonpostweight
  weight_vv:
    variable: vvweight
  weight_vv_post:
    variable: vvweight_post
```

### `configs/crosswalks/ces_2024_pre_questions.yaml`

只列第一阶段进入 memory 的变量，先不要贪多。

建议第一版包含：

```text
CC24_300 / 300a / 300b / 300c / 300d    media use
CC24_301                                national economy
CC24_302                                household income
CC24_303                                prices
CC24_305                                life changes
CC24_308a                               Ukraine
CC24_308b                               Israel/Gaza
CC24_309d                               emergency expense
CC24_309e                               health
CC24_312a/i/b/c/d/e/f/g/h               approval
CC24_321a-f                             gun policy
CC24_323a-d                             immigration
CC24_324a-d / CC24_325                  abortion
CC24_326a-f                             environment
CC24_328a-f                             tax / housing / healthcare / student loan
CC24_330a-h/e/f/g                       ideology placement
CC24_340a-f                             congressional bills
CC24_341a-d                             tax/infrastructure
```

### `configs/crosswalks/ces_2024_targets.yaml`

第一阶段：

```yaml
targets:
  turnout_self_report_post:
    variable: CC24_401
    target_id: turnout_2024_self_report
    truth_source: ces_post_self_report

  president_vote_post:
    variable: CC24_410
    target_id: president_vote_2024
    truth_source: ces_post_self_report

  president_preference_nonvoter_post:
    variable: CC24_410_nv
    target_id: president_preference_2024_nonvoter
    truth_source: ces_post_self_report

  turnout_validated:
    variable: TS_g2024
    target_id: turnout_2024_validated
    truth_source: targetsmart_validation
```

### `configs/fact_templates/ces_2024_common_facts.yaml`

例：

```yaml
- source_variable: CC24_301
  topic: economy
  priority: 90
  template: "The respondent says the nation's economy has {answer_label} over the past year."

- source_variable: CC24_303
  topic: inflation
  priority: 95
  template: "The respondent says prices of everyday goods and services have {answer_label} over the past year."

- source_variable: CC24_312a
  topic: approval
  priority: 100
  template: "The respondent {answer_label} of President Biden's job performance."

- source_variable: CC24_321c
  topic: guns
  priority: 70
  template: "The respondent {answer_label} requiring criminal background checks on all gun sales."
```

---

## 7. Baselines

保留现有 LLM baseline：

```text
demographic_only_llm
party_ideology_llm
survey_memory_llm
```

新增或重命名：

```text
party_id_baseline
sklearn_logit_pre_only
sklearn_logit_poll_informed
ces_empirical_cell_oracle
```

注意：

- `ces_empirical_cell_oracle` 如果使用 post vote choice，不能作为公平预测 baseline。
- `sklearn_logit_pre_only` 只能使用 strict policy 允许的变量。
- `sklearn_logit_poll_informed` 可以使用 pre vote intention / preference，但报告中要标记为 poll-informed。

---

## 8. 最小可交付版本

第一版 PR / coding agent 任务建议拆成 5 个小任务。

### Task 1：CES ingest

实现：

```text
build-ces
ces_respondents.parquet
ces_answers.parquet
ces_targets.parquet
ces_context.parquet
```

测试：

```text
读真实 CSV
输出行数 = 60000
权重列存在
post target 非空
pre answers 非空
```

### Task 2：CES memory

实现：

```text
survey_memory.py
build-ces-memory
strict_pre_no_vote_v1
leakage_audit
```

测试：

```text
memory facts 不包含 CC24_410 / TS_g2024 / direct vote vars
每个 respondent 最多 max_facts
fact_text 非空
```

### Task 3：CES row-level agents

实现：

```text
population.source = ces_rows
build_agents_from_ces_rows()
```

测试：

```text
能为 PA 生成 10 个 agents
agent 有 base_ces_id / memory_card_id / sample_weight
不依赖 ANES 文件
```

### Task 4：CES prompt + simulation

实现：

```text
ces_president_vote_v1 prompt
turnout + vote probability JSON parser
run_ces_election_simulation()
```

测试：

```text
mock model deterministic
ollama one-agent smoke run
responses.parquet 有 turnout_probability 和 vote_prob_*
```

### Task 5：evaluation + report

实现：

```text
individual metrics
turnout-aware state aggregation
MIT aggregate comparison
CES eval report
```

测试：

```text
individual accuracy 可计算
state aggregate 可计算
eval_report.md 包含 leakage policy / weight column / metrics
```

---

## 9. 推荐文件结构改造结果

目标结构：

```text
src/election_sim/
  anes.py
  ces.py
  mit.py
  gdelt.py

  survey_memory.py
  population.py
  prompts.py
  simulation.py
  aggregation.py
  evaluation.py
  report.py

  config.py
  io.py
  transforms.py
  validation.py
  baselines.py
  llm.py
  questions.py
  cli.py
```

`anes.py` 和 `ces.py` 只做 dataset adapter；通用功能不要塞回 dataset-specific module。

---

## 10. 运行路线

### 10.1 开发期 smoke

```bash
python -m election_sim.cli build-ces ...
python -m election_sim.cli build-ces-memory ...
python -m election_sim.cli run-simulation \
  --run-config configs/runs/ces_2024_president_smoke.yaml
```

### 10.2 本地小规模真实模型

```yaml
scenario:
  states: ["PA"]
population:
  sampling:
    mode: weighted_sample
    n_total_agents: 20
model:
  provider: ollama
```

### 10.3 Swing states

```yaml
scenario:
  states: ["PA", "MI", "WI", "GA", "AZ", "NV", "NC"]
population:
  sampling:
    mode: stratified_state_sample
    n_agents_per_state: 200
```

### 10.4 Full run

```yaml
scenario:
  states: all
population:
  sampling:
    mode: weighted_sample
    n_total_agents: 5000
```

---

## 11. 实现中的关键约束

1. CES post variables 和 TargetSmart variables 默认只能做 targets/evaluation，不得进入 prompt。
2. `strict_pre_no_vote_v1` 是默认正式预测模式。
3. prompt 和 report 必须记录 memory policy。
4. response 必须保留 raw response 和 parse_status。
5. state aggregation 必须记录使用的 weight column。
6. CES row-level 主线不能依赖 ANES processed artifacts。
7. 旧 fixture / ANES smoke run 尽量保持可运行；如果需要破坏兼容，先保留 legacy mode。
8. 大规模 LLM 调用前必须支持 `n_agents` sampling 和 mock provider。
9. 所有生成数据都写入 `data/processed/...` 或 `data/runs/...`，不要写回 `data/raw`。
10. 不把真实 CES/ANES/MIT 数据提交进 repo。

---

## 12. 第一阶段完成后的理想目录输出

```text
data/processed/ces/2024_common_vv/
  ces_respondents.parquet
  ces_answers.parquet
  ces_targets.parquet
  ces_context.parquet
  ces_question_bank.parquet
  ces_memory_facts.parquet
  ces_memory_cards.parquet
  ces_leakage_audit.parquet
  ces_ingest_report.md

data/runs/ces_2024_president_smoke/
  agents.parquet
  prompts.parquet
  prompt_preview.md
  responses.parquet
  individual_eval_metrics.parquet
  aggregate_state_results.parquet
  aggregate_eval_metrics.parquet
  eval_report.md
```

---

## 13. 最终目标形态

最终系统应该支持以下研究循环：

```text
Choose population:
  CES rows / CES cells + ANES archetypes / future synthetic electorate

Choose memory policy:
  strict pre-only / poll-informed / post-hoc

Choose context:
  no events / GDELT national / GDELT state-specific / issue-matched

Choose task:
  turnout / president vote / senate vote / house vote / governor vote

Run:
  build agents -> render prompts -> call model/baseline -> parse response

Evaluate:
  individual CES post/TargetSmart
  aggregate MIT official result
  subgroup calibration
```

总统 MVP 的最小闭环：

```text
CES pre-election persona
  -> LLM predicts turnout + vote
  -> CES post / TargetSmart evaluates individual behavior
  -> MIT evaluates state result
```
