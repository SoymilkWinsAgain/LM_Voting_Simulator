# Synthetic Election Simulation MVP 工程蓝图

> 本文档定义一个可实现、可评测、可扩展的模拟选举系统。系统使用 ANES 构造 voter-agent 的政治记忆与个体 benchmark，使用 CES/CCES Common Content 构造目标人口与政治联合分布，使用 MIT Election Lab 作为选举结果真值，使用 GDELT 2.0 GKG / Events 作为新闻事件上下文适配器。

---

## 1. 项目目标

### 1.1 系统产物

构建一个 **agent-based synthetic election simulation platform**：

```text
ANES respondent memory
+ CES/CCES target population distribution
+ MIT Election Lab result ground truth
+ optional GDELT news/event context
→ synthetic voter agents
→ LLM / baseline voter responses
→ state/county election simulation
→ benchmark, calibration, diagnostics, dashboard
```

系统必须能同时支持：

1. **个体级 benchmark**：给定 ANES respondent 的 profile 与安全 memory，预测 held-out ANES 问题。
2. **分布级 benchmark**：用 CES/CCES 加权分布作为目标，评估模拟 population 和 attitude/vote distribution。
3. **选举级 benchmark**：用 MIT Election Lab 结果评估州级或县级 vote share、margin、winner。
4. **事件上下文实验**：用 GDELT compact context cards 测试新闻事件环境对 voter agents 的影响。
5. **baseline comparison**：同时跑非 LLM baseline、LLM baseline、memory-augmented baseline、news-context baseline。

### 1.2 MVP 边界

MVP 第一版只做：

```text
Years:      2020, 2024 任选其一或两者都做
States:     PA, MI, WI, GA, AZ, NC, NV 优先
Level:      state-level presidential simulation
Population: CES/CCES state-cell weighted distribution
Agents:     ANES archetype matched agents
GDELT:      只实现 adapter 和 context-card schema；不全量 ingest
Frontend:   简单 dashboard 或 API + notebooks
```

MVP 第一版不做：

```text
- 全量 GDELT 下载与存储
- precinct-level simulation
- campaign strategist / candidate agent optimization
- real-time polling forecast
- 用户隐私敏感字段或 restricted-use geocode
```

---

## 2. 数据集角色

### 2.1 ANES

ANES 用于：

```text
1. voter-agent profile source
2. closed-question memory source
3. individual benchmark source
4. question bank source
5. optional open-ended memory source
```

ANES 的输出不是 population distribution 的主来源。ANES 的核心工程任务是把 respondent 的闭式问题答案转成安全的 prompt facts / memory cards。

### 2.2 CES/CCES Common Content

CES/CCES Common Content 用于：

```text
1. state-level weighted population distribution
2. political-demographic joint cells
3. empirical vote-choice baseline
4. distribution-level ground truth / calibration source
```

CES/CCES 是 synthetic electorate 的主分布来源。

### 2.3 MIT Election Lab

MIT Election Lab 用于：

```text
1. official election result ground truth
2. state/county vote share target
3. winner and margin evaluation
```

MIT 数据不进入 agent prompt，不进入 memory，不参与 individual behavior generation。

### 2.4 GDELT 2.0 GKG / Events

GDELT 用于：

```text
1. time-window news/event context
2. topic/candidate/state-level event cards
3. scenario experiments
```

GDELT 不直接决定人口分布或真实票数。MVP 阶段只实现 context-card pipeline 与少量抽样/手动 mock 数据。

---

## 3. 总体 pipeline

```text
configs
  ↓
ANES ingestion
  ↓
ANES canonical respondents
  ↓
ANES fact templates
  ↓
ANES memory cards
  ↓
ANES question bank

CES/CCES ingestion
  ↓
CES canonical respondents
  ↓
weighted state-cell distributions
  ↓
sparse-cell smoothing

MIT Election Lab ingestion
  ↓
normalized election results
  ↓
state/county ground truth

GDELT adapter
  ↓
context cards

CES state-cell distribution
+ ANES memory-card archetype pool
  ↓
ANES-to-CES archetype matching
  ↓
synthetic voter agents

agents
+ question bank
+ candidate config
+ optional GDELT context
  ↓
prompt builder
  ↓
LLM / non-LLM baselines
  ↓
simulated responses
  ↓
aggregation
  ↓
calibration
  ↓
evaluation
  ↓
reports / dashboard
```

---

## 4. 推荐目录结构

```text
synthetic-election-sim/
  README.md
  pyproject.toml
  .env.example

  configs/
    datasets/
      anes_2020.yaml
      anes_2024.yaml
      ces_2020.yaml
      ces_2024.yaml
      mit_president_state.yaml
      mit_president_county.yaml
      gdelt.yaml

    crosswalks/
      anes_2020_profile.yaml
      anes_2024_profile.yaml
      anes_2020_questions.yaml
      anes_2024_questions.yaml
      ces_2020_profile.yaml
      ces_2024_profile.yaml
      mit_results.yaml

    fact_templates/
      anes_2020_facts.yaml
      anes_2024_facts.yaml

    questions/
      mini_ppe_2020.yaml
      mini_ppe_2024.yaml
      vote_choice_2020.yaml
      vote_choice_2024.yaml

    candidates/
      president_2020.yaml
      president_2024.yaml

    cell_schemas/
      mvp_state_cell_v1.yaml
      extended_state_cell_v2.yaml

    prompts/
      demographic_only_v1.jinja2
      party_ideology_v1.jinja2
      survey_memory_v1.jinja2
      survey_memory_gdelt_v1.jinja2
      direct_json_answer_v1.jinja2

    runs/
      mvp_2024_swing_states.yaml
      individual_benchmark_anes_2024.yaml
      gdelt_ablation_2024_pa.yaml

  data/
    raw/
      anes/
      ces/
      mit/
      gdelt/
    interim/
    processed/
    runs/
      <run_id>/
        agents.parquet
        prompts.parquet
        responses.parquet
        aggregate_state_results.parquet
        aggregate_question_distributions.parquet
        eval_report.json
        eval_report.md

  src/
    election_sim/
      __init__.py

      config/
        loader.py
        schemas.py
        validate.py

      common/
        constants.py
        states.py
        categories.py
        logging.py
        hashing.py
        io.py
        random.py

      data/
        anes/
          ingest.py
          crosswalk.py
          facts.py
          memory.py
          questions.py
          leakage.py
        ces/
          ingest.py
          normalize.py
          cells.py
          smoothing.py
        mit/
          ingest.py
          normalize.py
          results.py
        gdelt/
          adapter.py
          bigquery_client.py
          local_loader.py
          context_cards.py

      population/
        cell_schema.py
        distribution.py
        archetype_matcher.py
        agent_builder.py

      baselines/
        majority.py
        empirical_cell.py
        logistic.py
        llm_baselines.py

      prompts/
        builder.py
        templates.py
        parser.py
        validator.py

      llm/
        client_base.py
        openai_client.py
        local_client.py
        batch.py
        cache.py

      simulation/
        run.py
        scheduler.py
        response_store.py
        retry.py

      aggregation/
        responses.py
        vote_share.py
        question_distribution.py
        subgroup.py

      calibration/
        confusion_matrix.py
        group_shrinkage.py
        poststratification.py
        bootstrap.py

      evaluation/
        individual.py
        distribution.py
        election.py
        robustness.py
        leakage.py
        report.py

      api/
        main.py
        routes_runs.py
        routes_agents.py
        routes_results.py
        routes_eval.py

  notebooks/
    00_data_checks.ipynb
    01_anes_memory_cards.ipynb
    02_ces_cell_distribution.ipynb
    03_individual_benchmark.ipynb
    04_state_simulation.ipynb
    05_eval_report.ipynb

  tests/
    test_anes_crosswalk.py
    test_fact_templates.py
    test_leakage_guard.py
    test_ces_cells.py
    test_archetype_matcher.py
    test_prompt_builder.py
    test_json_parser.py
    test_aggregation.py
    test_metrics.py
```

---

## 5. Canonical category system

所有数据源必须映射到统一 category system。不要在下游混用原始变量编码。

### 5.1 State

```text
state_fips: string, zero-padded, e.g. "42"
state_po:   USPS abbreviation, e.g. "PA"
state_name: full state name, e.g. "Pennsylvania"
```

### 5.2 Age group

MVP 默认：

```text
18_29
30_44
45_64
65_plus
unknown
```

### 5.3 Gender

```text
male
female
other_or_unknown
```

### 5.4 Race / ethnicity

MVP 默认：

```text
white
black
hispanic
asian
other_or_unknown
```

### 5.5 Education

MVP 默认二分：

```text
non_college
college_plus
unknown
```

可扩展版本：

```text
hs_or_less
some_college
college_degree
postgraduate
unknown
```

### 5.6 Party ID

```text
democrat
republican
independent_or_other
unknown
```

### 5.7 Ideology

```text
liberal
moderate
conservative
unknown
```

### 5.8 Vote choice

总统大选 MVP 默认：

```text
democrat
republican
other
not_vote_or_unknown
```

---

## 6. 配置文件规范

### 6.1 Run config

```yaml
run_id: mvp_2024_swing_states_v1
seed: 20260426
mode: election_simulation

scenario:
  year: 2024
  office: president
  election_day: "2024-11-05"
  states: ["PA", "MI", "WI", "GA", "AZ", "NC", "NV"]
  level: state

population:
  source: ces_common_content
  ces_year: 2024
  cell_schema: mvp_state_cell_v1
  n_agents_per_state: 5000
  smoothing:
    enabled: true
    tau: 500
    national_prior_weight: 0.15
  archetype_matching:
    source: anes
    anes_year: 2024
    with_replacement: true
    min_candidates_per_cell: 20
    allow_backoff: true

anes_memory:
  memory_policy: safe_survey_memory_v1
  include_profile_facts: true
  include_party_ideology: true
  include_candidate_thermometers: false
  include_party_thermometers: true
  include_issue_facts: true
  include_media_use: true
  include_open_ends: false
  max_memory_facts: 24

questions:
  question_set: vote_choice_2024
  target_question_ids:
    - vote_choice_president_2024

gdelt:
  enabled: false
  provider: stub
  context_window_days: 14
  topics: ["economy", "immigration", "crime", "healthcare", "abortion"]

model:
  provider: openai
  model_name: gpt-4o-mini
  temperature: 0.2
  max_tokens: 300
  response_format: json
  batch_size: 100
  cache_enabled: true

baselines:
  - majority
  - ces_empirical_cell
  - demographic_only_llm
  - party_ideology_llm
  - survey_memory_llm

evaluation:
  mit_results_year: 2024
  metrics:
    individual: [accuracy, micro_f1, macro_f1]
    distribution: [js_divergence, total_variation, hhi_difference]
    election: [winner_accuracy, vote_share_rmse, margin_mae]
  bootstrap:
    enabled: true
    n_bootstrap: 200
```

### 6.2 Cell schema config

```yaml
name: mvp_state_cell_v1
columns:
  - state_po
  - age_group
  - gender
  - race_ethnicity
  - education_binary
  - party_id_3
  - ideology_3

weights:
  state_po: 1.5
  age_group: 1.0
  gender: 0.5
  race_ethnicity: 2.0
  education_binary: 1.5
  party_id_3: 3.0
  ideology_3: 2.5

backoff_levels:
  - [state_po, age_group, gender, race_ethnicity, education_binary, party_id_3, ideology_3]
  - [state_po, age_group, race_ethnicity, education_binary, party_id_3, ideology_3]
  - [region, age_group, gender, race_ethnicity, education_binary, party_id_3, ideology_3]
  - [region, race_ethnicity, education_binary, party_id_3, ideology_3]
  - [race_ethnicity, education_binary, party_id_3, ideology_3]
  - [party_id_3, ideology_3]
```

### 6.3 Candidate config

```yaml
year: 2024
office: president
candidates:
  - candidate_id: dem_2024_president
    name: Kamala Harris
    party: democrat
    short_label: Harris
  - candidate_id: rep_2024_president
    name: Donald Trump
    party: republican
    short_label: Trump
  - candidate_id: other_2024_president
    name: Other candidate
    party: other
    short_label: Other
```

---

## 7. 数据表 contracts

所有 processed data 优先保存为 Parquet。每张表都必须包含 `source`, `year`, `created_at`, `schema_version` 或可追溯 metadata。

### 7.1 `anes_respondents.parquet`

用途：ANES respondent canonical profile。

字段：

```text
anes_id: string
source_year: int
sample_component: string|null
wave_available_pre: bool
wave_available_post: bool
state_po: string|null
region: string|null
age: int|null
age_group: string
gender: string
race_ethnicity: string
education_binary: string
education_detail: string|null
income_bin: string|null
party_id_3: string
party_id_7: string|null
ideology_3: string
ideology_7: string|null
political_interest: string|null
religion: string|null
urbanicity: string|null
weight_pre: float|null
weight_post: float|null
weight_full: float|null
schema_version: string
```

注意：如果公开 ANES 不含 state 或地理字段，则 `state_po` 可为空，archetype matching 通过 region/national backoff 完成。不要使用 restricted-use geocode 作为 MVP 输入。

### 7.2 `anes_answers.parquet`

用途：保存 ANES 原始问题答案的 canonical answer table。

字段：

```text
anes_id: string
source_year: int
wave: pre|post|panel|unknown
source_variable: string
question_id: string|null
topic: string|null
raw_value: string|int|float|null
raw_label: string|null
canonical_value: string|null
canonical_label: string|null
is_missing: bool
is_refusal: bool
is_dont_know: bool
is_valid_for_memory: bool
is_valid_for_target: bool
```

### 7.3 `anes_fact_templates.parquet`

用途：变量值到 prompt fact 的映射。

字段：

```text
source_year: int
source_variable: string
topic: string
subtopic: string|null
wave: pre|post|unknown
safe_as_memory: bool
allowed_memory_policies: list<string>
excluded_target_question_ids: list<string>
excluded_target_topics: list<string>
value: string|int|float
fact_text_template: string
missing_policy: skip|include_unknown
```

### 7.4 `anes_memory_facts.parquet`

用途：每个 respondent 的单条 memory fact。

字段：

```text
memory_fact_id: string
anes_id: string
source_year: int
source_variable: string
topic: string
subtopic: string|null
fact_text: string
fact_strength: float|null
safe_as_memory: bool
memory_policy: string
excluded_target_question_ids: list<string>
excluded_target_topics: list<string>
created_at: timestamp
```

### 7.5 `anes_memory_cards.parquet`

用途：每个 respondent 的完整 memory card。

字段：

```text
memory_card_id: string
anes_id: string
source_year: int
memory_policy: string
profile_facts: list<string>
political_facts: list<string>
media_facts: list<string>
issue_facts: list<string>
affect_facts: list<string>
open_end_facts: list<string>
all_fact_ids: list<string>
max_facts: int
created_at: timestamp
```

### 7.6 `question_bank.parquet`

用途：simulation / benchmark 的问题表。

字段：

```text
question_id: string
source: anes|custom|ces
source_year: int|null
source_variable: string|null
topic: string
subtopic: string|null
question_text: string
options_json: json
canonical_target_type: categorical|ordinal|binary|numeric
is_vote_choice: bool
is_candidate_eval: bool
is_party_eval: bool
is_issue_position: bool
allowed_answer_codes: list<string>
missing_answer_codes: list<string>
excluded_memory_variables: list<string>
excluded_memory_topics: list<string>
created_at: timestamp
```

Example `options_json`:

```json
{
  "democrat": "The Democratic candidate",
  "republican": "The Republican candidate",
  "other": "Another candidate",
  "not_vote_or_unknown": "Would not vote or unsure"
}
```

### 7.7 `ces_respondents.parquet`

用途：CES/CCES Common Content 的 canonical respondent table。

字段：

```text
ces_id: string
source_year: int
state_po: string
state_fips: string|null
age_group: string
gender: string
race_ethnicity: string
education_binary: string
income_bin: string|null
party_id_3: string
ideology_3: string
registered_voter: bool|null
validated_vote: string|null
vote_choice_president: string|null
common_weight: float
schema_version: string
```

### 7.8 `ces_cell_distribution.parquet`

用途：各州目标 population cell distribution。

字段：

```text
year: int
state_po: string
cell_schema: string
cell_id: string
age_group: string
gender: string
race_ethnicity: string
education_binary: string
party_id_3: string
ideology_3: string
weighted_n: float
raw_n: int
weighted_share_raw: float
weighted_share_smoothed: float
smoothing_lambda: float
national_prior_share: float
state_prior_share: float
```

### 7.9 `mit_election_results.parquet`

用途：MIT Election Lab ground truth。

字段：

```text
year: int
office: string
level: state|county|precinct
state_po: string
state_fips: string|null
county_name: string|null
county_fips: string|null
candidate: string
party_detailed: string|null
party_simplified: democrat|republican|other
candidatevotes: int
totalvotes: int
two_party_votes: int|null
two_party_share_dem: float|null
two_party_share_rep: float|null
source_file: string
```

### 7.10 `gdelt_context_cards.parquet`

用途：GDELT compact context cards。

字段：

```text
context_card_id: string
year: int
state_po: string|null
geo_scope: national|state|county
start_date: date
end_date: date
topic: string
candidate_ids: list<string>
event_count: int|null
mention_count: int|null
avg_tone: float|null
top_themes: list<string>
top_persons: list<string>
top_organizations: list<string>
top_locations: list<string>
summary: string
source: gdelt_events|gdelt_gkg|manual_stub
created_at: timestamp
```

### 7.11 `agents.parquet`

用途：synthetic voter agents。

字段：

```text
run_id: string
agent_id: string
year: int
state_po: string
cell_schema: string
cell_id: string
base_anes_id: string
memory_card_id: string
match_level: int
match_distance: float
sample_weight: float
age_group: string
gender: string
race_ethnicity: string
education_binary: string
party_id_3: string
ideology_3: string
created_at: timestamp
```

### 7.12 `prompts.parquet`

用途：可审计 prompt log。

字段：

```text
run_id: string
prompt_id: string
agent_id: string
question_id: string
baseline: string
model_name: string
prompt_template: string
prompt_version: string
prompt_hash: string
prompt_text: string
memory_fact_ids_used: list<string>
context_card_ids_used: list<string>
created_at: timestamp
```

### 7.13 `responses.parquet`

用途：LLM 和 non-LLM baseline responses。

字段：

```text
run_id: string
response_id: string
prompt_id: string
agent_id: string
base_anes_id: string
question_id: string
baseline: string
model_name: string
raw_response: string
parsed_answer_code: string|null
parsed_answer_label: string|null
confidence: float|null
parse_status: ok|failed|repaired|invalid_option
repair_attempts: int
latency_ms: int|null
cost_usd: float|null
created_at: timestamp
```

### 7.14 `aggregate_state_results.parquet`

用途：州级模拟结果。

字段：

```text
run_id: string
year: int
state_po: string
baseline: string
n_agents: int
dem_share_raw: float
rep_share_raw: float
other_share_raw: float
not_vote_or_unknown_share_raw: float
dem_share_calibrated: float|null
rep_share_calibrated: float|null
winner_raw: democrat|republican|other|tie
winner_calibrated: democrat|republican|other|tie|null
margin_raw: float
margin_calibrated: float|null
created_at: timestamp
```

### 7.15 `eval_metrics.parquet`

用途：统一 metrics table。

字段：

```text
run_id: string
metric_scope: individual|distribution|election|robustness|leakage
baseline: string
model_name: string
metric_name: string
metric_value: float
state_po: string|null
group_key: string|null
question_id: string|null
confidence_low: float|null
confidence_high: float|null
created_at: timestamp
```

---

## 8. ANES ingestion 与 memory-card pipeline

### 8.1 输入

```text
data/raw/anes/<year>/anes_<year>.csv
data/raw/anes/<year>/codebook 或 release variables
data/raw/anes/<year>/questionnaire
data/raw/anes/<year>/redacted_openends 可选
configs/crosswalks/anes_<year>_profile.yaml
configs/crosswalks/anes_<year>_questions.yaml
configs/fact_templates/anes_<year>_facts.yaml
```

### 8.2 Profile crosswalk

`anes_<year>_profile.yaml` 示例：

```yaml
source_year: 2024
respondent_id: V240001
weights:
  pre: V240101
  post: V240102
  full: V240103
fields:
  age:
    variable: V24XXXX_age
    transform: int
  age_group:
    variable: V24XXXX_age
    transform: age_to_group
  gender:
    variable: V24XXXX_gender
    mapping:
      1: male
      2: female
      default: other_or_unknown
  race_ethnicity:
    variable: V24XXXX_race_ethnicity
    mapping:
      1: white
      2: black
      3: hispanic
      4: asian
      default: other_or_unknown
  education_binary:
    variable: V24XXXX_education
    transform: education_to_binary
  party_id_3:
    variable: V24XXXX_partyid
    transform: party7_to_party3
  ideology_3:
    variable: V24XXXX_ideology
    transform: ideology7_to_ideology3
```

### 8.3 Fact template design

每个可作为 memory 的 ANES 变量都必须有 fact template。

```yaml
- source_variable: V24XXXX_political_interest
  topic: political_engagement
  subtopic: interest
  wave: pre
  safe_as_memory: true
  allowed_memory_policies: [safe_survey_memory_v1, rich_memory_v1]
  excluded_target_question_ids: []
  excluded_target_topics: []
  value_templates:
    "1": "The respondent follows politics most of the time."
    "2": "The respondent follows politics some of the time."
    "3": "The respondent only occasionally follows politics."
    "4": "The respondent hardly follows politics."
  missing_policy: skip

- source_variable: V24XXXX_vote_choice_post
  topic: vote_choice
  subtopic: president
  wave: post
  safe_as_memory: false
  allowed_memory_policies: []
  excluded_target_question_ids: [vote_choice_president_2024]
  excluded_target_topics: [vote_choice]
  value_templates: {}
  missing_policy: skip
```

### 8.4 Memory policy

#### `safe_survey_memory_v1`

允许：

```text
- demographics
- party ID
- ideology
- political interest
- media-use facts
- broad values / identity facts
- non-target issue facts
- party thermometer if target is not direct party-affect benchmark
```

禁止：

```text
- post-election vote choice
- validated vote
- direct pre-election vote intention if predicting vote choice
- exact target question answer
- near-duplicate same-topic item
- open-ended fact that directly states target answer
```

#### `rich_memory_v1`

允许更多政治态度变量，用于 stronger baseline 或 ablation。必须在 eval report 中单独标记。

#### `oracle_memory_v1`

用于泄漏上限诊断，不作为正式 baseline。可以故意加入 direct target-related variables，检查系统能够发现 B3-safe 与 oracle 的差距。

### 8.5 Leakage guard

所有 prompt 构造都必须经过 leakage guard。

```python
class LeakageGuard:
    def filter_facts(
        self,
        facts: list[MemoryFact],
        target_question: Question,
        memory_policy: str,
    ) -> list[MemoryFact]:
        out = []
        for fact in facts:
            if not fact.safe_as_memory:
                continue
            if memory_policy not in fact.allowed_memory_policies:
                continue
            if target_question.question_id in fact.excluded_target_question_ids:
                continue
            if target_question.topic in fact.excluded_target_topics:
                continue
            if fact.source_variable in target_question.excluded_memory_variables:
                continue
            if fact.topic in target_question.excluded_memory_topics:
                continue
            out.append(fact)
        return out
```

### 8.6 Memory card construction

Pseudo-code:

```python
def build_memory_card(respondent, facts, policy, max_facts):
    profile_facts = render_profile_facts(respondent)
    allowed_facts = [f for f in facts if policy in f.allowed_memory_policies]
    ranked_facts = rank_facts(allowed_facts, policy=policy)
    selected = ranked_facts[:max_facts]
    return MemoryCard(
        anes_id=respondent.anes_id,
        memory_policy=policy,
        profile_facts=profile_facts,
        political_facts=[f.text for f in selected if f.topic in POLITICAL_TOPICS],
        media_facts=[f.text for f in selected if f.topic == "media_use"],
        issue_facts=[f.text for f in selected if f.topic in ISSUE_TOPICS],
        affect_facts=[f.text for f in selected if f.topic in AFFECT_TOPICS],
        all_fact_ids=[f.id for f in selected],
    )
```

### 8.7 ANES individual benchmark split

必须避免同一 respondent 同时用于 prompt tuning 和 final test。

```text
split_key = hash(anes_id + seed) % 100
train: 0-59
dev:   60-79
test:  80-99
```

输出：

```text
anes_respondents_train.parquet
anes_respondents_dev.parquet
anes_respondents_test.parquet
```

---

## 9. Question bank pipeline

### 9.1 Question set types

```text
vote_choice_2020 / vote_choice_2024
  只包含总统投票选择问题。

mini_ppe_2020 / mini_ppe_2024
  10-20 个核心政治问题。

full_ppe_2020 / full_ppe_2024
  尽可能接近完整政治问卷 benchmark。

issue_only_<topic>
  单一议题实验，例如 immigration, economy, abortion。
```

### 9.2 Mini-PPE 推荐主题

```text
1. president vote choice
2. candidate favorability
3. party favorability
4. economy / inflation
5. immigration
6. abortion
7. health care
8. climate / environment
9. guns
10. crime
11. race / social justice
12. democracy / election trust
13. foreign policy
14. taxes / redistribution
15. social welfare
```

### 9.3 Question config 示例

```yaml
question_id: vote_choice_president_2024
source: custom
source_year: 2024
topic: vote_choice
subtopic: president
question_text: >
  In the 2024 U.S. presidential election, which option would this voter most likely choose?
options:
  democrat: "Kamala Harris, the Democratic candidate"
  republican: "Donald Trump, the Republican candidate"
  other: "Another candidate"
  not_vote_or_unknown: "Would not vote or is unsure"
canonical_target_type: categorical
is_vote_choice: true
excluded_memory_variables:
  - V24XXXX_post_vote_choice
  - V24XXXX_validated_vote
  - V24XXXX_pre_vote_intent
excluded_memory_topics:
  - vote_choice
  - direct_candidate_choice
```

---

## 10. CES/CCES population pipeline

### 10.1 输入

```text
data/raw/ces/<year>/ces_common_content_<year>.*
或 cumulative common content file
configs/crosswalks/ces_<year>_profile.yaml
configs/cell_schemas/mvp_state_cell_v1.yaml
```

### 10.2 Normalize CES respondents

Pseudo-code:

```python
def normalize_ces(raw_df, crosswalk):
    out = DataFrame()
    out["ces_id"] = raw_df[crosswalk.respondent_id]
    out["source_year"] = crosswalk.year
    out["state_po"] = map_state(raw_df[crosswalk.state])
    out["age_group"] = map_age(raw_df[crosswalk.age])
    out["gender"] = map_gender(raw_df[crosswalk.gender])
    out["race_ethnicity"] = map_race(raw_df[crosswalk.race])
    out["education_binary"] = map_education(raw_df[crosswalk.education])
    out["party_id_3"] = map_party(raw_df[crosswalk.party])
    out["ideology_3"] = map_ideology(raw_df[crosswalk.ideology])
    out["common_weight"] = raw_df[crosswalk.weight]
    out["vote_choice_president"] = map_vote(raw_df[crosswalk.vote_choice])
    return out
```

### 10.3 Weighted cell distribution

For each year and state:

\[
\hat p_s(c) =
\frac{\sum_i w_i \mathbf{1}(state_i=s, cell_i=c)}
{\sum_i w_i \mathbf{1}(state_i=s)}
\]

Pseudo-code:

```python
def build_cell_distribution(ces_df, cell_cols, weight_col):
    rows = []
    for state, df_s in ces_df.groupby("state_po"):
        grouped = df_s.groupby(cell_cols).agg(
            weighted_n=(weight_col, "sum"),
            raw_n=(weight_col, "size"),
        ).reset_index()
        grouped["weighted_share_raw"] = grouped["weighted_n"] / grouped["weighted_n"].sum()
        rows.append(grouped)
    return concat(rows)
```

### 10.4 Sparse-cell smoothing

Use national prior shrinkage:

\[
\tilde p_s(c) = \lambda_s \hat p_s(c) + (1 - \lambda_s)\hat p_{nat}(c)
\]

\[
\lambda_s = \frac{n_s}{n_s + \tau}
\]

Pseudo-code:

```python
def smooth_state_distribution(state_dist, national_dist, raw_n_state, tau=500):
    lam = raw_n_state / (raw_n_state + tau)
    merged = outer_join(state_dist, national_dist, on="cell_id").fillna(0)
    merged["weighted_share_smoothed"] = (
        lam * merged["weighted_share_raw"]
        + (1 - lam) * merged["national_prior_share"]
    )
    merged["weighted_share_smoothed"] /= merged["weighted_share_smoothed"].sum()
    merged["smoothing_lambda"] = lam
    return merged
```

### 10.5 CES empirical cell baseline

For vote choice:

\[
P(y | c, s) =
\frac{\sum_i w_i \mathbf{1}(state_i=s, cell_i=c, vote_i=y)}
{\sum_i w_i \mathbf{1}(state_i=s, cell_i=c)}
\]

If sparse:

```text
state-cell → region-cell → national-cell → party/ideology cell → state overall → national overall
```

This baseline is required. Any LLM baseline must be compared against it.

---

## 11. MIT Election Lab result pipeline

### 11.1 输入

```text
data/raw/mit/president_state_<year>.csv
data/raw/mit/president_county_<year>.csv optional
configs/crosswalks/mit_results.yaml
```

### 11.2 Normalization rules

1. Normalize state to `state_po` and `state_fips`.
2. Normalize party to `democrat`, `republican`, `other`.
3. Aggregate multiple party lines for the same candidate if present.
4. Compute total votes by state/county.
5. Compute two-party vote share.
6. Store both raw candidate rows and aggregated two-party rows.

### 11.3 Output metrics fields

For each state:

```text
dem_votes
rep_votes
other_votes
total_votes
dem_share_total = dem_votes / total_votes
rep_share_total = rep_votes / total_votes
dem_share_two_party = dem_votes / (dem_votes + rep_votes)
rep_share_two_party = rep_votes / (dem_votes + rep_votes)
margin_two_party = dem_share_two_party - rep_share_two_party
winner_two_party
```

---

## 12. GDELT context-card pipeline

### 12.1 MVP behavior

Implement the interface first. Support two providers:

```text
stub:
  reads manually authored context cards from YAML/CSV.

local_sample:
  reads a small local GDELT sample file.

bigquery:
  future implementation for querying GDELT 2.0 Events/GKG.
```

### 12.2 Context card YAML example

```yaml
- context_card_id: gdelt_stub_2024_PA_economy_2024-10-01_2024-10-15
  year: 2024
  state_po: PA
  geo_scope: state
  start_date: "2024-10-01"
  end_date: "2024-10-15"
  topic: economy
  candidate_ids: [dem_2024_president, rep_2024_president]
  event_count: 128
  mention_count: 940
  avg_tone: -1.8
  top_themes: [ECONOMY, INFLATION, COST_OF_LIVING, CAMPAIGN]
  top_persons: [Kamala Harris, Donald Trump]
  top_organizations: []
  top_locations: [Pennsylvania]
  summary: >
    Recent coverage emphasized inflation, grocery prices, jobs, and campaign messages
    about the economy in Pennsylvania.
```

### 12.3 Time leakage rule

For a simulation with `simulation_date = D`, allowed GDELT context must satisfy:

```text
context_card.end_date <= D
```

For election-day vote simulation:

```text
context_card.end_date <= election_day
```

---

## 13. ANES-to-CES archetype matching

### 13.1 Goal

Create synthetic voter agents with CES target distribution and ANES memory cards.

Input:

```text
ces_cell_distribution.parquet
anes_respondents.parquet
anes_memory_cards.parquet
cell_schema.yaml
```

Output:

```text
agents.parquet
```

### 13.2 Sampling cell counts

For each state:

```python
cell_counts = multinomial(
    n=n_agents_per_state,
    p=ces_cell_distribution[state].weighted_share_smoothed,
)
```

Alternative deterministic allocation:

```python
cell_counts = round(n_agents_per_state * p_cell)
fix rounding by largest remainder
```

Use deterministic allocation for reproducible baseline runs. Use multinomial for uncertainty runs.

### 13.3 Matching procedure

Pseudo-code:

```python
def match_anes_archetypes(cell, anes_pool, cell_schema, n_needed):
    for level, cols in enumerate(cell_schema.backoff_levels):
        candidates = filter_by_cols(anes_pool, cell, cols)
        if len(candidates) >= min_candidates_per_cell:
            return weighted_sample(candidates, n_needed, level=level)

    candidates = nearest_neighbors(anes_pool, cell, weights=cell_schema.weights)
    return sample_top_k(candidates, n_needed, level=999)
```

### 13.4 Distance function

For categorical features:

\[
d(i,c) = \sum_k \alpha_k \mathbf{1}(x_{ik} \ne c_k)
\]

Default weights:

```text
party_id_3:        3.0
ideology_3:        2.5
race_ethnicity:    2.0
education_binary:  1.5
state_po:          1.5
region:            1.0
age_group:         1.0
gender:            0.5
```

### 13.5 Agent object

```json
{
  "run_id": "mvp_2024_swing_states_v1",
  "agent_id": "mvp_2024_swing_states_v1_PA_000001",
  "year": 2024,
  "state_po": "PA",
  "cell_id": "PA|45_64|female|white|non_college|republican|conservative",
  "base_anes_id": "ANES2024_004812",
  "memory_card_id": "ANES2024_004812_safe_survey_memory_v1",
  "match_level": 2,
  "match_distance": 1.5,
  "sample_weight": 1.0,
  "profile": {
    "age_group": "45_64",
    "gender": "female",
    "race_ethnicity": "white",
    "education_binary": "non_college",
    "party_id_3": "republican",
    "ideology_3": "conservative"
  }
}
```

---

## 14. Baselines

### 14.1 Required baseline list

```text
B0_majority
B1_ces_empirical_cell
B2_demographic_only_llm
B3_party_ideology_llm
B4_survey_memory_llm
B5_survey_memory_ces_population_llm
B6_survey_memory_ces_population_gdelt_llm
```

### 14.2 B0 Majority

Individual benchmark:

```text
For each question, predict the most frequent answer in ANES train split.
```

Election simulation:

```text
For vote choice, predict majority class from CES train/year/state or national fallback.
```

### 14.3 B1 CES empirical cell

No LLM.

```text
Predict P(vote | state, cell) from CES/CCES weighted data.
```

If the simulation needs a single categorical answer, sample from the empirical distribution with fixed seed.

If the simulation needs aggregate result, directly aggregate expected probabilities:

\[
\hat V_s(y) = \sum_c \tilde p_s(c) P(y | s,c)
\]

### 14.4 B2 demographic-only LLM

Prompt contains:

```text
state
age_group
gender
race_ethnicity
education
income if available
```

Does not contain:

```text
party ID
ideology
ANES memory
GDELT context
candidate thermometer
issue facts
```

### 14.5 B3 party+ideology LLM

Prompt contains B2 plus:

```text
party_id_3
ideology_3
political_interest optional
```

### 14.6 B4 survey-memory LLM

Prompt contains B3 plus:

```text
safe ANES memory facts after leakage guard
```

Use ANES respondents directly for individual benchmark.

### 14.7 B5 survey-memory + CES population LLM

Pipeline:

```text
CES cell distribution
→ sample state agents
→ match ANES memory cards
→ run B4 prompts
→ aggregate state results
```

This is the main MVP election-simulation baseline.

### 14.8 B6 survey-memory + CES population + GDELT LLM

Pipeline:

```text
B5 + GDELT context card selected by state/topic/date
```

Use only for scenario and ablation experiments.

---

## 15. Prompt templates

### 15.1 Common output contract

All LLM prompts must require JSON only:

```json
{
  "answer": "<one of allowed answer codes>",
  "confidence": 0.0
}
```

Optional debug mode:

```json
{
  "answer": "<one of allowed answer codes>",
  "confidence": 0.0,
  "brief_reason": "one short sentence"
}
```

Evaluation should use `answer` only.

### 15.2 Demographic-only template

```jinja2
You are simulating a U.S. eligible voter in {{ year }}.

Voter profile:
- State: {{ state_po }}
- Age group: {{ age_group }}
- Gender: {{ gender }}
- Race/ethnicity: {{ race_ethnicity }}
- Education: {{ education_binary }}
{% if income_bin %}- Income: {{ income_bin }}{% endif %}

Question:
{{ question_text }}

Options:
{% for code, label in options.items() -%}
- {{ code }}: {{ label }}
{% endfor %}

Return JSON only with this schema:
{"answer": "<option code>", "confidence": <number between 0 and 1>}
```

### 15.3 Party+ideology template

```jinja2
You are simulating a U.S. eligible voter in {{ year }}.

Voter profile:
- State: {{ state_po }}
- Age group: {{ age_group }}
- Gender: {{ gender }}
- Race/ethnicity: {{ race_ethnicity }}
- Education: {{ education_binary }}
- Party identification: {{ party_id_3 }}
- Ideology: {{ ideology_3 }}
{% if political_interest %}- Political interest: {{ political_interest }}{% endif %}

Question:
{{ question_text }}

Options:
{% for code, label in options.items() -%}
- {{ code }}: {{ label }}
{% endfor %}

Return JSON only with this schema:
{"answer": "<option code>", "confidence": <number between 0 and 1>}
```

### 15.4 Survey-memory template

```jinja2
You are simulating a U.S. eligible voter in {{ year }}.
Answer as this voter would answer, not as a political analyst.

Voter profile:
- State: {{ state_po }}
- Age group: {{ age_group }}
- Gender: {{ gender }}
- Race/ethnicity: {{ race_ethnicity }}
- Education: {{ education_binary }}
- Party identification: {{ party_id_3 }}
- Ideology: {{ ideology_3 }}

Additional survey-derived background facts:
{% for fact in memory_facts -%}
- {{ fact }}
{% endfor %}

Question:
{{ question_text }}

Options:
{% for code, label in options.items() -%}
- {{ code }}: {{ label }}
{% endfor %}

Return JSON only with this schema:
{"answer": "<option code>", "confidence": <number between 0 and 1>}
```

### 15.5 Survey-memory + GDELT template

```jinja2
You are simulating a U.S. eligible voter in {{ year }}.
Answer as this voter would answer, not as a political analyst.

Voter profile:
- State: {{ state_po }}
- Age group: {{ age_group }}
- Gender: {{ gender }}
- Race/ethnicity: {{ race_ethnicity }}
- Education: {{ education_binary }}
- Party identification: {{ party_id_3 }}
- Ideology: {{ ideology_3 }}

Additional survey-derived background facts:
{% for fact in memory_facts -%}
- {{ fact }}
{% endfor %}

Recent news/event environment:
{% for card in context_cards -%}
- {{ card.summary }}
{% endfor %}

Question:
{{ question_text }}

Options:
{% for code, label in options.items() -%}
- {{ code }}: {{ label }}
{% endfor %}

Return JSON only with this schema:
{"answer": "<option code>", "confidence": <number between 0 and 1>}
```

---

## 16. Simulation engine

### 16.1 Run modes

```text
individual_benchmark
  Use ANES respondents directly.
  Predict held-out ANES questions.

population_diagnostic
  Build CES synthetic agents.
  Do not call LLM.
  Check synthetic population vs CES target distribution.

election_simulation
  Build CES synthetic agents.
  Run vote-choice prompts.
  Aggregate state results.
  Compare with MIT.

issue_distribution_simulation
  Build agents.
  Run issue-question prompts.
  Compare with ANES/CES weighted distributions if available.

gdelt_scenario_ablation
  Run same agents/questions with and without context cards.
  Compare answer shifts.
```

### 16.2 Orchestration pseudo-code

```python
def run_simulation(run_config):
    cfg = load_and_validate(run_config)
    set_seed(cfg.seed)

    if cfg.mode == "individual_benchmark":
        population = load_anes_benchmark_population(cfg)
    else:
        ces_dist = load_ces_cell_distribution(cfg)
        anes_pool = load_anes_archetype_pool(cfg)
        population = build_agents(ces_dist, anes_pool, cfg)

    questions = load_question_bank(cfg.questions)
    baselines = load_baselines(cfg.baselines)
    gdelt_context = load_context_cards(cfg) if cfg.gdelt.enabled else None

    for baseline in baselines:
        for agent in population:
            for question in questions:
                response = baseline.predict(
                    agent=agent,
                    question=question,
                    context=gdelt_context,
                    cfg=cfg,
                )
                store_response(response)

    aggregate_results(cfg.run_id)
    calibrate_if_enabled(cfg.run_id)
    evaluate(cfg.run_id)
    write_report(cfg.run_id)
```

### 16.3 Batch and cache

Prompt cache key:

```text
hash(model_name + prompt_template_version + prompt_text + temperature)
```

If cache hit:

```text
skip LLM call
reuse raw_response and parsed answer
```

### 16.4 Retry policy

If output parse fails:

```text
attempt 1: parse raw JSON substring
attempt 2: repair with deterministic parser prompt
attempt 3: mark parse_status=failed
```

If parsed answer not in options:

```text
attempt 1: fuzzy map to option label
attempt 2: mark invalid_option
```

Do not silently coerce invalid options without logging.

---

## 17. Aggregation

### 17.1 Individual response aggregation

For a categorical question:

```python
share(answer=a | group=g) =
    sum_i weight_i * 1(answer_i == a) / sum_i weight_i
```

Groups:

```text
state
cell
party_id_3
ideology_3
race_ethnicity
education_binary
age_group
state × party
state × race
state × education
```

### 17.2 Vote-share aggregation

Raw shares:

```python
dem_share_raw = weighted_mean(answer == "democrat")
rep_share_raw = weighted_mean(answer == "republican")
other_share_raw = weighted_mean(answer == "other")
unknown_share_raw = weighted_mean(answer == "not_vote_or_unknown")
```

Two-party share:

```python
dem_share_2p = dem_share_raw / (dem_share_raw + rep_share_raw)
rep_share_2p = rep_share_raw / (dem_share_raw + rep_share_raw)
margin_2p = dem_share_2p - rep_share_2p
```

Winner:

```python
winner = "democrat" if dem_share_2p > rep_share_2p else "republican"
```

### 17.3 Expected aggregation for probabilistic baselines

If baseline returns probabilities:

```python
dem_share = weighted_mean(p_democrat)
rep_share = weighted_mean(p_republican)
```

Do not sample individual votes unless evaluating stochastic variability.

---

## 18. Calibration

### 18.1 Calibration scope

Calibration is optional but should be implemented early. It must produce both raw and calibrated outputs.

```text
raw_response → raw aggregate → calibrated aggregate
```

Never overwrite raw outputs.

### 18.2 Confusion-matrix calibration

Use ANES dev split.

For each question:

\[
C_{ab} = P(\text{true}=b \mid \text{pred}=a)
\]

Then adjust aggregate predicted distribution:

\[
\hat p_{calibrated}(b) = \sum_a \hat p_{raw}(a) C_{ab}
\]

### 18.3 Group shrinkage

For each group `g`:

\[
\delta_g = true\_share_g - raw\_share_g
\]

Shrink group correction toward global correction:

\[
\tilde\delta_g = \lambda_g \delta_g + (1-\lambda_g)\delta_{global}
\]

\[
\lambda_g = \frac{n_g}{n_g + \tau}
\]

### 18.4 Bootstrap uncertainty

Bootstrap sources:

```text
1. CES respondents / cell distribution
2. ANES archetype matching
3. LLM stochasticity if temperature > 0
4. agent sample
```

MVP default:

```text
bootstrap over agents and cells only
n_bootstrap = 200
```

Output:

```text
metric confidence_low / confidence_high
state vote share CI
winner flip rate
```

---

## 19. Evaluation

### 19.1 Individual-level metrics

Ground truth: held-out ANES answers.

Metrics:

```text
accuracy
micro_f1
macro_f1
balanced_accuracy
per_question_accuracy
per_group_accuracy
confusion_matrix
```

Macro-F1 is required because many answer options are imbalanced.

### 19.2 Distribution-level metrics

Ground truth: ANES weighted distribution or CES weighted distribution.

For each question and group:

```text
JS divergence
Total variation distance
KL divergence optional
HHI difference
Entropy difference
Minority-option absolute error
```

Definitions:

\[
TV(P,Q)=\frac{1}{2}\sum_i |P_i-Q_i|
\]

\[
HHI(P)=\sum_i P_i^2
\]

High HHI difference means the simulated distribution is too concentrated or too diffuse.

### 19.3 Election-level metrics

Ground truth: MIT Election Lab.

Metrics:

```text
winner_accuracy
state_vote_share_rmse
state_vote_share_mae
margin_mae
margin_bias
battleground_rmse
winner_flip_count
correlation_with_true_margin
```

Formulas:

\[
RMSE = \sqrt{\frac{1}{|S|}\sum_{s\in S}(\hat p_s - p_s)^2}
\]

\[
MarginBias = \frac{1}{|S|}\sum_{s\in S}(\hat m_s - m_s)
\]

### 19.4 Robustness metrics

Run same config with different seeds / prompt versions / models.

```text
state winner flip rate
mean absolute vote share drift
question distribution drift
prompt sensitivity
model sensitivity
```

### 19.5 Leakage diagnostics

Run:

```text
safe_survey_memory_v1
rich_memory_v1
oracle_memory_v1
```

Report:

```text
safe_vs_oracle_accuracy_gap
safe_vs_oracle_vote_rmse_gap
leakage_flag_by_question
```

If oracle gap is large, inspect excluded variables and fact templates.

---

## 20. Reports

### 20.1 `eval_report.md` required sections

```text
1. Run metadata
2. Dataset versions
3. Population summary
4. Baseline list
5. Individual benchmark table
6. Distribution benchmark table
7. Election result table
8. State-by-state error table
9. Subgroup diagnostics
10. Calibration effect
11. Leakage diagnostics
12. Robustness summary
13. Known failures
```

### 20.2 State result table format

```text
state | baseline | pred_dem_2p | true_dem_2p | margin_error | pred_winner | true_winner | correct
```

### 20.3 Population diagnostic table

```text
state | cell_schema | max_abs_cell_error | js_to_ces | n_empty_cells | n_backoff_agents | avg_match_distance
```

### 20.4 Subgroup diagnostic table

```text
group_type | group_value | baseline | metric | value | n_agents | warning
```

Warnings:

```text
small_n
high_match_distance
high_prompt_failure
high_error
sparse_ces_cell
```

---

## 21. API design

MVP API can be FastAPI.

### 21.1 Endpoints

```text
GET /runs
GET /runs/{run_id}
GET /runs/{run_id}/states
GET /runs/{run_id}/states/{state_po}/results
GET /runs/{run_id}/metrics
GET /runs/{run_id}/agents?state=PA&limit=50
GET /runs/{run_id}/agents/{agent_id}
GET /runs/{run_id}/questions
GET /runs/{run_id}/distributions?question_id=...
GET /runs/{run_id}/prompts/{prompt_id}
```

### 21.2 Agent response

```json
{
  "agent_id": "mvp_2024_PA_000001",
  "state_po": "PA",
  "cell_id": "PA|45_64|female|white|non_college|republican|conservative",
  "base_anes_id": "ANES2024_004812",
  "match_level": 2,
  "match_distance": 1.5,
  "profile": {
    "age_group": "45_64",
    "gender": "female",
    "race_ethnicity": "white",
    "education_binary": "non_college",
    "party_id_3": "republican",
    "ideology_3": "conservative"
  },
  "memory_facts_preview": [
    "The respondent follows politics most of the time.",
    "The respondent reported using TV and internet sites for campaign information."
  ]
}
```

---

## 22. Dashboard MVP

### 22.1 Pages

```text
1. Run overview
2. State map / state table
3. Baseline comparison
4. Question distribution explorer
5. Subgroup diagnostics
6. Agent browser
7. Prompt audit viewer
8. Calibration and leakage diagnostics
```

### 22.2 Required visualizations

```text
- state-level predicted vs true two-party share
- state margin error bar chart
- baseline RMSE comparison
- answer distribution stacked bar
- subgroup error heatmap
- match-level distribution
- prompt parse failure rate
- raw vs calibrated comparison
```

---

## 23. CLI commands

### 23.1 Build ANES

```bash
python -m election_sim.data.anes.ingest \
  --config configs/datasets/anes_2024.yaml \
  --profile-crosswalk configs/crosswalks/anes_2024_profile.yaml \
  --question-crosswalk configs/crosswalks/anes_2024_questions.yaml \
  --out data/processed/anes/2024/
```

### 23.2 Build ANES memory cards

```bash
python -m election_sim.data.anes.memory \
  --respondents data/processed/anes/2024/anes_respondents.parquet \
  --answers data/processed/anes/2024/anes_answers.parquet \
  --fact-templates configs/fact_templates/anes_2024_facts.yaml \
  --policy safe_survey_memory_v1 \
  --out data/processed/anes/2024/anes_memory_cards.parquet
```

### 23.3 Build CES cell distribution

```bash
python -m election_sim.data.ces.cells \
  --config configs/datasets/ces_2024.yaml \
  --crosswalk configs/crosswalks/ces_2024_profile.yaml \
  --cell-schema configs/cell_schemas/mvp_state_cell_v1.yaml \
  --out data/processed/ces/2024/ces_cell_distribution.parquet
```

### 23.4 Build MIT results

```bash
python -m election_sim.data.mit.ingest \
  --config configs/datasets/mit_president_state.yaml \
  --year 2024 \
  --out data/processed/mit/mit_president_state_2024.parquet
```

### 23.5 Build agents

```bash
python -m election_sim.population.agent_builder \
  --run-config configs/runs/mvp_2024_swing_states.yaml \
  --out data/runs/mvp_2024_swing_states_v1/agents.parquet
```

### 23.6 Run simulation

```bash
python -m election_sim.simulation.run \
  --run-config configs/runs/mvp_2024_swing_states.yaml
```

### 23.7 Evaluate

```bash
python -m election_sim.evaluation.report \
  --run-id mvp_2024_swing_states_v1 \
  --run-dir data/runs/mvp_2024_swing_states_v1
```

---

## 24. Testing requirements

### 24.1 Unit tests

Required tests:

```text
test_anes_profile_crosswalk_maps_categories_correctly
test_anes_fact_template_skips_missing_values
test_memory_card_respects_max_facts
test_leakage_guard_removes_vote_choice_facts
test_question_bank_options_are_valid
test_ces_weighted_cell_distribution_sums_to_one_by_state
test_smoothing_preserves_probability_simplex
test_archetype_matcher_returns_required_count
test_archetype_matcher_backoff_level_logged
test_mit_candidate_party_aggregation
test_prompt_builder_includes_allowed_facts_only
test_json_parser_accepts_valid_response
test_json_parser_rejects_invalid_option
test_vote_share_aggregation
test_election_rmse_metric
```

### 24.2 Data validation checks

For each processed table:

```text
- no duplicate primary keys
- required columns exist
- canonical categories are valid
- state shares sum to 1
- memory facts are non-empty where expected
- prompt templates render without missing variables
- responses use allowed answer codes
```

### 24.3 Smoke test

A minimal smoke run:

```text
Year: 2024
States: PA
Agents: 50
Questions: vote_choice only
Baselines: majority, demographic_only_llm
Model: cheap/fast model or mock LLM
Expected: complete run without parse failures > 20%
```

---

## 25. Development milestones

### Milestone 1: Data contracts and ANES memory

Deliverables:

```text
- canonical category mapping
- ANES ingestion
- ANES profile crosswalk
- ANES answer table
- fact templates for at least 30 variables
- safe memory cards
- leakage guard
- unit tests
```

### Milestone 2: CES distribution and agent builder

Deliverables:

```text
- CES ingestion
- weighted state-cell distribution
- smoothing
- archetype matcher
- agents.parquet
- population diagnostics
```

### Milestone 3: MIT results and non-LLM baselines

Deliverables:

```text
- MIT state result ingestion
- majority baseline
- CES empirical cell baseline
- aggregation and election metrics
```

### Milestone 4: LLM simulation

Deliverables:

```text
- prompt templates
- LLM client
- response parser
- cache
- demographic-only baseline
- party+ideology baseline
- survey-memory baseline
```

### Milestone 5: Evaluation report

Deliverables:

```text
- individual benchmark
- distribution benchmark
- election benchmark
- leakage diagnostics
- markdown report
```

### Milestone 6: GDELT adapter and scenario ablation

Deliverables:

```text
- GDELT context-card schema
- stub provider
- local sample provider
- survey-memory + GDELT prompt
- no-context vs context ablation report
```

### Milestone 7: Dashboard/API

Deliverables:

```text
- FastAPI endpoints
- state result explorer
- baseline comparison
- agent browser
- prompt audit viewer
```

---

## 26. Minimum viable first run

Use this as the first end-to-end target.

```yaml
run_id: first_e2e_2024_pa_v0
mode: election_simulation
scenario:
  year: 2024
  office: president
  states: ["PA"]
  level: state
population:
  source: ces_common_content
  n_agents_per_state: 500
  cell_schema: mvp_state_cell_v1
anes_memory:
  memory_policy: safe_survey_memory_v1
  max_memory_facts: 12
questions:
  question_set: vote_choice_2024
gdelt:
  enabled: false
baselines:
  - majority
  - ces_empirical_cell
  - demographic_only_llm
  - party_ideology_llm
  - survey_memory_llm
model:
  provider: openai
  model_name: gpt-4o-mini
  temperature: 0.2
evaluation:
  metrics:
    election: [winner_accuracy, vote_share_rmse, margin_mae]
```

Expected outputs:

```text
data/runs/first_e2e_2024_pa_v0/agents.parquet
data/runs/first_e2e_2024_pa_v0/prompts.parquet
data/runs/first_e2e_2024_pa_v0/responses.parquet
data/runs/first_e2e_2024_pa_v0/aggregate_state_results.parquet
data/runs/first_e2e_2024_pa_v0/eval_metrics.parquet
data/runs/first_e2e_2024_pa_v0/eval_report.md
```

---

## 27. Implementation priorities

Priority order:

```text
1. Get data contracts right.
2. Build ANES memory cards with leakage guard.
3. Build CES weighted cell distribution.
4. Match ANES archetypes to CES cells.
5. Implement non-LLM baselines.
6. Implement LLM prompt/response loop.
7. Implement MIT election evaluation.
8. Add calibration.
9. Add GDELT context cards.
10. Add dashboard/API.
```

Do not start with dashboard. Do not start with complex GDELT ingestion. Do not start with multi-agent debate. The first useful artifact is a clean run directory with reproducible data, responses, metrics, and report.

---

## 28. Official data references

Use these as data documentation starting points:

```text
ANES 2024 Time Series Study:
https://electionstudies.org/data-center/2024-time-series-study/

ANES 2020 Time Series Study:
https://electionstudies.org/data-center/2020-time-series-study/

CES / CCES Common Content:
https://cces.gov.harvard.edu/book/common-content
https://tischcollege.tufts.edu/research-faculty/research-centers/cooperative-election-study/data-downloads

Cumulative CES Common Content guide:
https://csmweb-prod-02.ist.berkeley.edu/sdaweb/docs/ces-cumulative-2024-v10/DOC/guide_cumulative_2006-2024.pdf

MIT Election Lab data:
https://electionlab.mit.edu/data

GDELT data overview:
https://www.gdeltproject.org/data.html

GDELT Event Codebook V2.0:
https://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf

GDELT GKG Codebook V2.1:
https://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf
```

