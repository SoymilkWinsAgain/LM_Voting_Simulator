# MIT Election Lab 数据接入工程蓝图

本文档说明如何把 MIT Election Lab 总统选举结果数据正式接入 `LM_Voting_Simulator`。本次改造的核心目标是：把 MIT 数据从当前的 fixture / 临时对照数据，升级为系统性的 aggregate ground truth、历史地理先验与评估基准。

---

## 0. 本次改动目标

### 0.1 核心目标

新增一条稳定的 MIT Election Lab 数据 pipeline：

```text
MIT raw CSV
  -> MIT normalized election returns
  -> state / county presidential truth tables
  -> simulation aggregate evaluation
  -> historical geographic priors / calibration features
```

实现后，系统应该支持：

```text
1. 用 MIT 2024 官方结果评估 CES / ANES / LLM 模拟出的州级总统选举结果。
2. 用 MIT 2000-2024 county-level 数据支持 county-level truth 和历史地理特征。
3. 用 MIT 1976-2020 state-level 数据支持长期 state-level backtest。
4. 明确区分 historical prior 与 held-out truth，避免 2024 结果泄漏进 prompt。
```

### 0.2 非目标

本次不要求一次性完成：

```text
1. 完整 county-level agent population model。
2. ACS / CVAP / voter file 级别的人口后分层。
3. House / Senate / Governor 等其他 office。
4. GDELT 动态事件传播。
5. 复杂的 Bayesian calibration 或 MRP。
```

第一阶段只需要让真实 MIT presidential returns 能稳定产生 state-level truth，并被现有 CES election simulation aggregate evaluation 使用。

---

## 1. 与当前仓库的关系

当前仓库已经有这些相关模块：

```text
src/election_sim/mit.py
src/election_sim/aggregation.py
src/election_sim/evaluation.py
src/election_sim/simulation.py
configs/datasets/
configs/runs/
data/processed/
data/runs/
```

当前主线已经大体存在：

```text
CES respondents / agents
  -> prompts
  -> LLM or baseline responses
  -> aggregate_state_results.parquet
  -> aggregate_eval_metrics.parquet
  -> eval_report.md
```

本次改动不应该推翻这条主线，而是把 MIT 部分升级为：

```text
raw MIT CSV
  -> canonical returns
  -> state_truth / county_truth
  -> aggregate evaluation
```

也就是说，`mit.py` 应该从一个简单 CSV normalizer 升级成 MIT returns processing module；`evaluation.py` 和 `simulation.py` 只需要面向标准 truth schema，而不是关心 MIT 原始 CSV 的细节。

---

## 2. MIT 数据集内容与格式

本次本地下载了两份 MIT Election Lab presidential returns 数据。

### 2.1 `countypres_2000-2024.csv`

用途：county-level presidential returns，覆盖 2000、2004、2008、2012、2016、2020、2024。

本地读取结果：

```text
rows: 94,151
columns: 12
years: 2000, 2004, 2008, 2012, 2016, 2020, 2024
geo level: county
office: US PRESIDENT
states/jurisdictions: 51, including DC
```

字段：

```text
state
county_name
year
state_po
county_fips
office
candidate
party
candidatevotes
totalvotes
version
mode
```

代表性记录结构：

```text
state        county_name  year  state_po  county_fips  office        candidate          party       candidatevotes  totalvotes  version   mode
ALABAMA      AUTAUGA      2024  AL        1001         US PRESIDENT  KAMALA D HARRIS   DEMOCRAT    7439            28281       20260225  TOTAL
ALABAMA      AUTAUGA      2024  AL        1001         US PRESIDENT  DONALD J TRUMP    REPUBLICAN  20484           28281       20260225  TOTAL
```

重要 caveat：

```text
1. mode 不一定只有 TOTAL。
2. 2020 / 2024 中，一些州拆分了 ABSENTEE、EARLY、ELECTION DAY、PROVISIONAL 等 vote mode。
3. 有些州存在 TOTAL 行；有些州没有 TOTAL 行；有些州 mode 为空但实际可视作 total-like records。
4. 数据中存在 TOTAL VOTES CAST、OVERVOTES、UNDERVOTES、SPOILED 等非候选人记录。
5. candidatevotes 有少量缺失值，主要集中在 2024 New Mexico 的 Chase Oliver / Libertarian 若干 mode 行。
6. county_fips 应该标准化成 zero-padded string，不能保留 float。
```

### 2.2 `1976-2020-president.csv`

用途：state-level presidential returns，覆盖 1976-2020，每四年一次。

本地读取结果：

```text
rows: 4,287
columns: 15
years: 1976, 1980, ..., 2020
geo level: state
office: US PRESIDENT
```

字段：

```text
year
state
state_po
state_fips
state_cen
state_ic
office
candidate
party_detailed
writein
candidatevotes
totalvotes
version
notes
party_simplified
```

代表性记录结构：

```text
year  state    state_po  candidate        party_detailed  candidatevotes  totalvotes  party_simplified
1976  ALABAMA  AL        CARTER, JIMMY    DEMOCRAT        659170          1182850     DEMOCRAT
1976  ALABAMA  AL        FORD, GERALD     REPUBLICAN      504070          1182850     REPUBLICAN
```

这份数据更适合做长期 state-level backtest。它本身只到 2020；但 2024 state-level presidential returns 可以从 `countypres_2000-2024.csv` 按州聚合得到，用作 2024 aggregate truth。

### 2.3 2024 state-level 数据是否可以从 county file 直接构造

结论：**项目需要的 2024 state-level truth 可以直接从 `countypres_2000-2024.csv` 构造出来**。不需要额外寻找一个 2024 版 `1976-2020-president.csv` 才能做州级评估。

本地已按以下规则构造过一版 schema-compatible 的 2024 state-level extension：

```text
input:
  data/raw/mit/countypres_2000-2024.csv

mode policy:
  unit = (year, state_po, county_fips, candidate, party)

  if unit has mode == TOTAL:
      use TOTAL row
  else:
      sum all available mode rows

state rollup:
  group by (year, state_po, candidate, party)
  sum candidatevotes
  totalvotes = sum one county-level totalvotes per county

state code fields:
  state_fips / state_cen / state_ic copied from existing 1976-2020 state file mapping

party_simplified:
  DEMOCRAT -> DEMOCRAT
  REPUBLICAN -> REPUBLICAN
  LIBERTARIAN -> LIBERTARIAN
  otherwise -> OTHER
```

构造结果：

```text
2024 raw county rows: 21,534
rows after mode policy: 14,160
2024 state candidate rows generated: 202
states/jurisdictions covered: 51
extended state file rows: 4,287 + 202 = 4,489
```

推荐在工程中生成两个不同 artifact：

```text
data/processed/mit/1976-2024-president_derived_from_countypres.csv
data/processed/mit/2024-president-state-truth-derived_from_countypres.csv
```

其中第一个是对 `1976-2020-president.csv` 的 schema-compatible extension，第二个是 simulation evaluation 更应该直接使用的 truth table。

需要保留的 caveat：

```text
1. 这不是 MIT 官方单独发布的 1976-2024 state-level presidential file。
2. `countypres_2000-2024.csv` 的 2024 minor candidates 在很多州已经聚合成 OTHER，因此不能恢复所有 minor-party candidate 的逐人明细。
3. county file 没有 `writein` 字段，所以 2024 append rows 的 writein 应保留为空，而不是强行填 False。
4. `TOTAL VOTES CAST`、`OVERVOTES`、`UNDERVOTES`、`SPOILED` 等行政记录不应进入 candidate-level truth；如果保留，也必须放在 audit / administrative rows 中。
5. 对项目评估最重要的 Democratic / Republican / Other 聚合票数、two-party share、margin、winner 可以稳定构造。
```

因此，pipeline 设计上应该把 “2024 state truth” 视作 **由 county file 派生出的 state rollup**：

```text
countypres_2000-2024.csv
  -> mode-aware county normalization
  -> state rollup for 2024
  -> president_state_truth.parquet / csv
```

不要把它当成外部缺失数据；这部分应该由 `mit.py` 自动生成。

---

## 3. MIT 数据在系统中的角色

MIT 数据不应该直接变成 individual agent memory。它在系统中有三类角色。

### 3.1 Aggregate truth

这是第一优先级。

```text
simulation outputs:
  aggregate_state_results.parquet

MIT truth:
  state_president_truth.parquet

evaluation:
  compare predicted dem_2p / margin / winner against MIT truth
```

核心指标：

```text
state dem two-party share RMSE
state margin MAE
winner accuracy
state-level margin error
national popular-vote share error
```

### 3.2 Historical geographic prior

用于 2024 模拟时，可以使用 2020 及更早结果作为历史地理背景：

```text
state 2020 Democratic two-party share
county 2020 Democratic two-party share
state swing from 2016 to 2020
county swing from 2016 to 2020
long-run partisan baseline
```

但这些历史结果默认不直接进入 prompt。更推荐作为 calibration / poststratification feature。

### 3.3 Backtest benchmark

1976-2020 state-level 数据可以用于长期回测：

```text
for year in [2000, 2004, 2008, 2012, 2016, 2020]:
    build state truth
    run baseline or replay simulation
    compute state-level errors
```

county-level backtest 使用 `countypres_2000-2024.csv`。

---

## 4. 核心设计原则

### 4.1 Truth layer 与 prompt layer 分离

2024 MIT 结果是 held-out truth。它只能进入 evaluation，不得进入 agent prompt。

```text
Allowed for 2024 simulation prompt:
  - CES pre-election answers under selected leakage policy
  - ANES pre-election / non-label facts
  - historical MIT results from <= 2020, if explicitly configured

Not allowed for 2024 simulation prompt:
  - MIT 2024 state / county results
  - CES post-election vote labels
  - TargetSmart validation labels
```

### 4.2 原始 CSV 不直接给下游模块读

下游模块不应该直接依赖 MIT 原始字段。应先统一成 canonical schema。

```text
raw MIT CSV
  -> normalized election_returns.parquet
  -> state_truth / county_truth
  -> evaluation
```

这样以后 MIT 更新版本、字段名变化、mode 变化时，只需要改 normalizer。

### 4.3 Candidate choice 优先于 party line

总统投票任务问的是：

```text
Democratic nominee / Republican nominee / Other
```

而不只是 ballot party label。

在 fusion voting 州，同一个候选人可能出现在不同 party line 下。因此 truth builder 不应只按 `party` 或 `party_simplified` 分组，而应引入 candidate crosswalk：

```text
year
candidate_norm
major_choice
```

例如：

```text
2024  KAMALA D HARRIS  democrat
2024  DONALD J TRUMP   republican
2020  BIDEN JOSEPH R JR  democrat
2020  TRUMP DONALD J     republican
```

然后用 `major_choice` 构造 two-party truth。

### 4.4 County mode 处理必须显式化

`countypres_2000-2024.csv` 中，同一个 county-candidate-year 可能有多个 mode。必须统一规则：

```text
For each unit = (year, state_po, county_fips, candidate_norm):

if TOTAL rows exist:
    use TOTAL rows
else:
    sum all available mode rows
```

不能简单：

```text
df[df.mode == "TOTAL"]
```

也不能简单把所有 rows 加总，否则会在含 TOTAL + mode breakdown 的州重复计票。

---

## 5. 目标数据产物

### 5.1 Raw input

建议路径：

```text
data/raw/mit/countypres_2000-2024.csv
data/raw/mit/1976-2020-president.csv
```

### 5.2 Processed output

建议输出：

```text
data/processed/mit/
  election_returns_county_2000_2024.parquet
  election_returns_state_1976_2020.parquet
  president_state_truth.parquet
  president_county_truth.parquet
  president_historical_features.parquet
  mit_ingest_audit.parquet
  mit_ingest_report.md
```

### 5.3 Run-time output

simulation run 仍然输出到：

```text
data/runs/<run_id>/
  aggregate_state_results.parquet
  aggregate_eval_metrics.parquet
  eval_report.md
```

第一阶段不强制改现有 run output 命名。未来 county-level run 可以新增：

```text
aggregate_county_results.parquet
county_aggregate_eval_metrics.parquet
```

---

## 6. Canonical schema 设计

### 6.1 `election_returns_*.parquet`

一行代表一个规范化后的 candidate return record。

建议字段：

```text
year
office
geo_level                    # state / county
state
state_po
state_fips
county_name
county_fips                  # zero-padded string, nullable for state-level
geo_id                       # state:PA or county:PA:42003

candidate_raw
candidate_norm
party_raw
party_norm
major_choice                 # democrat / republican / other

candidatevotes
totalvotes

mode_policy_used             # total_row / summed_modes / single_mode / state_file
source_modes_used            # list or stable string
source_file
source_version

audit_flags                  # stable string list
schema_version
created_at
```

### 6.2 `president_state_truth.parquet`

一行代表一个 state-year 的 presidential truth。

```text
year
office
state_po
geo_id

dem_votes
rep_votes
other_votes
candidate_total_votes
two_party_total_votes
totalvotes

dem_share_raw
rep_share_raw
other_share_raw
dem_share_2p
rep_share_2p
margin_2p
winner

truth_source                 # mit_county_rollup / mit_state_file
source_version
audit_flags
schema_version
created_at
```

### 6.3 `president_county_truth.parquet`

一行代表一个 county-year 的 presidential truth。

```text
year
office
state_po
county_fips
county_name
geo_id

dem_votes
rep_votes
other_votes
candidate_total_votes
two_party_total_votes
totalvotes

dem_share_raw
rep_share_raw
other_share_raw
dem_share_2p
rep_share_2p
margin_2p
winner

truth_source                 # mit_county_file
source_version
audit_flags
schema_version
created_at
```

### 6.4 `president_historical_features.parquet`

用于 calibration 或未来 county-aware simulation，不是正式 truth。

```text
year
state_po
county_fips
geo_level
geo_id

dem_share_2p
rep_share_2p
margin_2p
winner
turnout_total
swing_from_previous
avg_margin_last_2
avg_margin_last_3
partisan_baseline

source_truth_table
schema_version
```

---

## 7. MIT processing pipeline

### 7.1 Pipeline 总览

```text
Raw MIT CSV
  -> load raw files
  -> validate columns
  -> normalize geography
  -> normalize candidate / party
  -> apply candidate crosswalk
  -> apply mode policy
  -> remove non-candidate rows
  -> aggregate to canonical returns
  -> build state / county truth
  -> build historical features
  -> write audit report
```

### 7.2 Geography normalization

规则：

```text
state_po:
  keep uppercase two-letter postal code

state_fips:
  normalize to two-digit string when present

county_fips:
  normalize to five-digit string
  preserve null only when genuinely state-level
```

Example:

```text
1001.0 -> "01001"
35003.0 -> "35003"
```

`geo_id`：

```text
state:PA
county:PA:42003
```

### 7.3 Candidate normalization

建议新增：

```text
configs/crosswalks/mit_president_candidate_crosswalk.yaml
```

结构：

```yaml
office: president
candidate_crosswalk:
  - year: 2024
    candidate_patterns:
      - "KAMALA D HARRIS"
      - "KAMALA HARRIS"
      - "HARRIS, KAMALA"
    candidate_norm: "KAMALA D HARRIS"
    major_choice: democrat

  - year: 2024
    candidate_patterns:
      - "DONALD J TRUMP"
      - "DONALD TRUMP"
      - "TRUMP, DONALD J."
    candidate_norm: "DONALD J TRUMP"
    major_choice: republican

  - year: 2020
    candidate_patterns:
      - "BIDEN, JOSEPH R. JR"
      - "JOE BIDEN"
      - "BIDEN JOSEPH R JR"
    candidate_norm: "JOSEPH R BIDEN JR"
    major_choice: democrat

  - year: 2020
    candidate_patterns:
      - "TRUMP, DONALD J."
      - "DONALD TRUMP"
    candidate_norm: "DONALD J TRUMP"
    major_choice: republican
```

Fallback：

```text
if party_norm is Democrat:
    major_choice = democrat
elif party_norm is Republican:
    major_choice = republican
else:
    major_choice = other
```

但 formal truth 应优先使用 candidate crosswalk。

### 7.4 Non-candidate row handling

需要过滤或标记：

```text
TOTAL VOTES CAST
OVERVOTES
UNDERVOTES
SPOILED
BLANK VOTES
WRITE-IN TOTAL where candidate is not identifiable
```

建议处理：

```text
1. 这些 rows 不进入 candidate vote aggregation。
2. 如果 totalvotes 与 candidate_total_votes 差异很大，写入 audit。
3. totalvotes 字段可保留为 denominator reference，但 two-party share 不直接依赖它。
```

### 7.5 Mode policy

对 county file：

```text
unit = (year, state_po, county_fips, candidate_norm, major_choice)

if any row in unit has mode == "TOTAL":
    use rows where mode == "TOTAL"
    mode_policy_used = "total_row"
else:
    use all rows for that unit
    mode_policy_used = "summed_modes"
```

对 state file：

```text
mode_policy_used = "state_file"
```

缺失值规则：

```text
candidatevotes missing:
  - if candidate is major-party nominee:
      flag as error or fail fast
  - if candidate is minor-party / other:
      coerce to 0 only if config explicitly allows it
      add audit flag: candidatevotes_missing_filled_zero
```

第一阶段可以对 minor-party missing values 填 0，但必须写 audit。

---

## 8. Truth construction

### 8.1 County truth

从 normalized county returns 生成：

```text
group by:
  year, state_po, county_fips, major_choice

sum:
  candidatevotes
```

然后构造：

```text
dem_votes = votes where major_choice == democrat
rep_votes = votes where major_choice == republican
other_votes = votes where major_choice == other

two_party_total = dem_votes + rep_votes

dem_share_2p = dem_votes / two_party_total
rep_share_2p = rep_votes / two_party_total
margin_2p = dem_share_2p - rep_share_2p
winner = democrat if margin_2p > 0 else republican if margin_2p < 0 else tie
```

形式上：

```text
s_{g,D}^{MIT} = V_{g,D} / (V_{g,D} + V_{g,R})

m_g^{MIT} = s_{g,D}^{MIT} - s_{g,R}^{MIT}
```

其中 `g` 是 county 或 state。

### 8.2 State truth

2024 state truth 优先从 county file roll up：

```text
county presidential returns
  -> aggregate by state_po, year, major_choice
  -> state truth
```

理由：

```text
1. 2024 只存在于 county file。
2. 现有 simulation aggregate 是 state-level。
3. county rollup 可以未来自然扩展到 county evaluation。
```

1976-2020 state truth 可以直接从 state file 构建。

### 8.3 Truth source priority

建议规则：

```text
For 2024:
  use mit_county_rollup

For 2000-2020 state-level backtest:
  default use mit_state_file
  optionally compare against mit_county_rollup where available

For 2000-2024 county-level:
  use mit_county_file
```

---

## 9. 与 simulation aggregate 的对接

### 9.1 当前 state aggregate 逻辑

现有 CES 主线已经适合接 MIT truth。simulation 的 response 层输出：

```text
turnout_probability
vote_prob_democrat
vote_prob_republican
vote_prob_other
vote_prob_undecided
```

aggregation 层应继续采用 expected vote，而不是 hard vote count：

```text
Vhat_{s,D} = sum_{i in s} w_i * p_i(turnout) * p_i(D)

Vhat_{s,R} = sum_{i in s} w_i * p_i(turnout) * p_i(R)

shat_{s,D}^{2p} = Vhat_{s,D} / (Vhat_{s,D} + Vhat_{s,R})
```

MIT truth 对应：

```text
s_{s,D}^{MIT} = V_{s,D}^{MIT} / (V_{s,D}^{MIT} + V_{s,R}^{MIT})
```

评估误差：

```text
e_s = shat_{s,D}^{2p} - s_{s,D}^{MIT}
```

### 9.2 Evaluation metrics

`aggregate_eval_metrics.parquet` 应至少包含：

```text
winner_accuracy
dem_2p_rmse
margin_mae
state_margin_error
state_dem_2p_error
national_dem_2p_error
```

推荐 metric schema 沿用现有 evaluator：

```text
run_id
metric_scope                 # election / election_state / election_national
baseline
model_name
metric_name
metric_value
state_po
group_key
question_id
n
small_n
confidence_low
confidence_high
created_at
```

---

## 10. 建议模块改造

### 10.1 `src/election_sim/mit.py`

从当前简单 normalizer 升级为 MIT processing module。

建议保留兼容入口：

```python
normalize_mit_results(config_path, year)
state_truth_table(results)
build_mit_results(config_path, year, out_path)
```

新增更通用入口：

```python
normalize_mit_county_president(config_path) -> pd.DataFrame
normalize_mit_state_president(config_path) -> pd.DataFrame
build_president_truth(returns, geo_level, years=None) -> pd.DataFrame
build_historical_features(truth) -> pd.DataFrame
write_mit_processed_artifacts(config_path, out_dir) -> dict[str, Path]
```

`normalize_mit_results(config_path, year)` 可以成为 wrapper，避免旧 run config 直接坏掉。

### 10.2 `configs/datasets/mit_countypres_2000_2024.yaml`

新增：

```yaml
name: mit_countypres_2000_2024
path: data/raw/mit/countypres_2000-2024.csv
format: csv
office: president
geo_level: county
schema_version: mit_countypres_v1

columns:
  state: state
  county_name: county_name
  year: year
  state_po: state_po
  county_fips: county_fips
  office: office
  candidate: candidate
  party: party
  candidatevotes: candidatevotes
  totalvotes: totalvotes
  version: version
  mode: mode

mode_policy:
  if_total_exists_use_total: true
  otherwise_sum_modes: true
  treat_null_mode_as_available_mode: true

candidate_vote_policy:
  major_party_missing: error
  minor_party_missing: fill_zero_with_audit

exclude_candidate_patterns:
  - "TOTAL VOTES CAST"
  - "OVERVOTES"
  - "UNDERVOTES"
  - "SPOILED"
```

### 10.3 `configs/datasets/mit_president_state_1976_2020.yaml`

新增：

```yaml
name: mit_president_state_1976_2020
path: data/raw/mit/1976-2020-president.csv
format: csv
office: president
geo_level: state
schema_version: mit_state_president_v1

columns:
  year: year
  state: state
  state_po: state_po
  state_fips: state_fips
  office: office
  candidate: candidate
  party_detailed: party_detailed
  party_simplified: party_simplified
  writein: writein
  candidatevotes: candidatevotes
  totalvotes: totalvotes
  version: version
  notes: notes
```

### 10.4 `configs/crosswalks/mit_president_candidate_crosswalk.yaml`

新增 candidate-level crosswalk，支持 1976-2024。第一版可以只覆盖：

```text
2000
2004
2008
2012
2016
2020
2024
```

长期 backtest 再补 1976-1996。

### 10.5 `aggregation.py`

短期保留 state-level output。建议预留泛化接口：

```python
aggregate_turnout_vote_geo_results(
    responses,
    agents,
    run_id,
    year,
    geo_cols,
    office="president",
)
```

然后：

```python
aggregate_turnout_vote_state_results(...)
```

作为 wrapper：

```python
return aggregate_turnout_vote_geo_results(
    responses,
    agents,
    run_id,
    year,
    geo_cols=["state_po"],
    office=office,
)
```

未来 county-level 只需要：

```python
geo_cols=["state_po", "county_fips"]
```

### 10.6 `evaluation.py`

新增 geo-level evaluator：

```python
evaluate_aggregate_geo_results(
    aggregate,
    truth,
    run_id,
    geo_cols,
)
```

现有 state-level evaluator 可以继续存在，但内部应尽量调用 geo-level evaluator。

### 10.7 `simulation.py`

当前 CES simulation 已经支持：

```text
if cfg.paths.get("mit_results"):
    read parquet truth/results
elif cfg.paths.get("mit_config"):
    normalize from config
```

建议升级为：

```text
paths:
  mit_state_truth: data/processed/mit/president_state_truth.parquet
  mit_county_truth: data/processed/mit/president_county_truth.parquet
  mit_historical_features: data/processed/mit/president_historical_features.parquet
```

第一阶段：

```text
evaluation.aggregate.truth_path = paths.mit_state_truth
evaluation.aggregate.year = 2024
evaluation.aggregate.geo_level = state
```

旧的 `mit_config` 可以保留为 fallback。

### 10.8 `report.py`

`eval_report.md` 增加 MIT truth section：

```text
MIT truth source
MIT source version
truth year
truth geo level
number of states evaluated
excluded rows / audit flags
mode policy summary
candidate crosswalk version
state error table
```

---

## 11. Config 设计示例

### 11.1 MIT build config

可以新增一个 higher-level config：

```yaml
# configs/datasets/mit_president_returns.yaml

name: mit_president_returns
schema_version: mit_president_returns_v1

inputs:
  county:
    config: configs/datasets/mit_countypres_2000_2024.yaml
  state:
    config: configs/datasets/mit_president_state_1976_2020.yaml

crosswalks:
  candidate: configs/crosswalks/mit_president_candidate_crosswalk.yaml

outputs:
  out_dir: data/processed/mit
  county_returns: election_returns_county_2000_2024.parquet
  state_returns: election_returns_state_1976_2020.parquet
  state_truth: president_state_truth.parquet
  county_truth: president_county_truth.parquet
  historical_features: president_historical_features.parquet
  audit: mit_ingest_audit.parquet
  report: mit_ingest_report.md

truth_policy:
  state_truth:
    "2024": county_rollup
    "2000-2020": state_file
  county_truth:
    "2000-2024": county_file
```

### 11.2 CLI 示例

新增或扩展 CLI：

```bash
python -m election_sim.cli build-mit-president \
  --config configs/datasets/mit_president_returns.yaml
```

输出：

```text
data/processed/mit/president_state_truth.parquet
data/processed/mit/president_county_truth.parquet
data/processed/mit/president_historical_features.parquet
```

### 11.3 CES swing run 使用真实 MIT truth

run config 中：

```yaml
paths:
  ces_respondents: data/processed/ces/2024_common_vv/ces_respondents.parquet
  ces_answers: data/processed/ces/2024_common_vv/ces_answers.parquet
  ces_targets: data/processed/ces/2024_common_vv/ces_targets.parquet
  ces_context: data/processed/ces/2024_common_vv/ces_context.parquet
  ces_memory_cards: data/processed/ces/2024_common_vv/ces_memory_cards.parquet
  ces_memory_facts: data/processed/ces/2024_common_vv/ces_memory_facts.parquet

  mit_state_truth: data/processed/mit/president_state_truth.parquet
  mit_historical_features: data/processed/mit/president_historical_features.parquet

evaluation:
  aggregate:
    enabled: true
    truth: mit_president_state
    truth_path: data/processed/mit/president_state_truth.parquet
    year: 2024
    geo_level: state
    geo_cols: ["state_po"]
```

---

## 12. 推荐实现顺序

### Task 1: MIT raw loader + validation

实现：

```text
load_mit_county_president()
load_mit_state_president()
validate required columns
normalize year / state_po / fips / candidatevotes
```

成功标准：

```text
能读取两份 raw CSV
能报告 rows / columns / years / states
county_fips 被标准化为五位 string
不会因为 2024 county_fips float 造成 join 问题
```

### Task 2: County mode-aware normalizer

实现：

```text
mode policy
non-candidate row exclusion
candidatevotes missing audit
candidate / party normalization
```

成功标准：

```text
不会重复计算 TOTAL + mode breakdown
能生成 election_returns_county_2000_2024.parquet
audit report 能显示每年每州使用了什么 mode policy
```

### Task 3: Candidate crosswalk + truth builder

实现：

```text
candidate crosswalk
major_choice mapping
state_truth builder
county_truth builder
```

成功标准：

```text
2024 state truth 能覆盖所有 states/jurisdictions
state truth 包含 dem_votes / rep_votes / dem_share_2p / margin_2p / winner
county truth 包含 county_fips / county_name / margin_2p
```

### Task 4: Simulation evaluation 接真实 MIT truth

实现：

```text
run config 支持 paths.mit_state_truth
evaluation.py 用 standard truth schema
report.py 输出 MIT truth source
```

成功标准：

```text
ces_2024_president_swing_strict_pre.yaml 能用真实 MIT 2024 state truth 跑出 aggregate_eval_metrics.parquet
eval_report.md 里出现 state-by-state MIT comparison
```

### Task 5: Historical features

实现：

```text
build_historical_features()
```

成功标准：

```text
能生成 state / county historical margin features
2024 simulation 默认只使用 <=2020 historical features
features 不会混入 2024 truth，除非显式用于 post-hoc analysis
```

---

## 13. 第一阶段最小可交付版本

第一版不需要 county-level simulation，只需要：

```text
1. 从 countypres_2000-2024.csv 生成 2024 state truth。
2. 从 1976-2020-president.csv 生成 historical state truth。
3. 让 CES 2024 swing-state run 对真实 MIT 2024 truth 计算 aggregate metrics。
4. 生成 audit report，确认没有重复计票。
```

最小数据流：

```text
data/raw/mit/countypres_2000-2024.csv
  -> data/processed/mit/election_returns_county_2000_2024.parquet
  -> data/processed/mit/president_state_truth.parquet

data/runs/ces_2024_president_swing_strict_pre/aggregate_state_results.parquet
  + data/processed/mit/president_state_truth.parquet
  -> data/runs/ces_2024_president_swing_strict_pre/aggregate_eval_metrics.parquet
  -> data/runs/ces_2024_president_swing_strict_pre/eval_report.md
```

---

## 14. 后续扩展方向

### 14.1 County-level simulation

需要新增或补强：

```text
agents.county_fips
county-level population assignment
county-level aggregation
county-level truth comparison
```

MIT county truth 只能提供选举结果，不能提供 county demographic distribution。真正严谨的 county-level simulation 需要 ACS / CVAP / voter file 或其他人口表。

### 14.2 Historical prior calibration

可引入轻量 calibration：

```text
logit p_i(D)
  = alpha
  + beta * logit p_i^LLM(D)
  + gamma^T X_i
  + eta_state(i)
  + rho * m_state(i),2020
```

其中 `m_state,2020` 来自 MIT historical truth，不是 2024 truth。

### 14.3 Backtesting

长期可以做：

```text
year t historical features up to t-4
  -> simulate / baseline predict year t
  -> compare against MIT truth year t
```

这样可以把系统从“只解释 2024”扩展成可复用 election simulator benchmark。

---

## 15. 关键检查项

实现时必须检查：

```text
1. 2024 truth 没有进入 prompt。
2. county mode 没有重复计票。
3. candidate crosswalk 能正确处理 Democratic / Republican nominees。
4. fusion voting 不会把 major-party nominee 的其他 ballot line 误算成 OTHER。
5. county_fips 是 string，不是 float。
6. candidatevotes missing 被 audit。
7. aggregate evaluation 使用 two-party share，而不是 totalvotes denominator。
8. report 明确写出 MIT source version 与 mode policy。
```

---

## 16. 最终系统图

```text
                   ┌─────────────────────────┐
                   │ MIT raw presidential CSV │
                   └───────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │ MIT normalizer / auditor  │
                  └───────────┬──────────────┘
                              │
          ┌───────────────────┼────────────────────┐
          ▼                   ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌─────────────────────┐
│ county returns    │ │ state returns     │ │ historical features  │
└────────┬─────────┘ └────────┬─────────┘ └─────────┬───────────┘
         │                    │                     │
         ▼                    ▼                     ▼
┌──────────────────┐ ┌──────────────────┐ ┌─────────────────────┐
│ county truth      │ │ state truth       │ │ calibration / prior  │
└────────┬─────────┘ └────────┬─────────┘ └─────────────────────┘
         │                    │
         │                    ▼
         │       ┌──────────────────────────┐
         │       │ simulation aggregate eval │
         │       └───────────┬──────────────┘
         │                   │
         ▼                   ▼
┌──────────────────┐ ┌──────────────────────┐
│ future county eval│ │ eval_report.md        │
└──────────────────┘ └──────────────────────┘
```

本次改造完成后，MIT Election Lab 数据在系统中的定位应非常清晰：

```text
CES / ANES:
  individual-level voter profile, memory, labels

LLM:
  individual-level turnout and vote-choice prediction

aggregation:
  expected votes by state / county

MIT:
  official aggregate ground truth and historical geographic priors

evaluation:
  compare simulated aggregate against official returns
```
