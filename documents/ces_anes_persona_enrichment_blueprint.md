# CES–ANES Persona Enrichment Blueprint

**Project context:** `LM_Voting_Simulator`  
**Purpose:** enrich CES-based voter agents with ANES-derived persona context without changing the CES population base, CES weights, or downstream state aggregation.

This document describes the final design target, not an experiment plan. It is written for a human developer and a coding assistant that can inspect the repository and local intermediate data but should not need to repeatedly reopen the original survey manuals.

---

## 1. Core Goal

The simulator should keep **CES respondents** as the actual simulated voter agents.

ANES should be used only as a **persona donor corpus**:

```text
CES respondent
  -> observed CES demographics / party / ideology / issue facts
  -> retrieve similar ANES respondents
  -> summarize ANES-only psychological / affective / open-text information
  -> inject this as inferred persona context
  -> ask the LLM to predict turnout / vote using both observed CES facts and inferred ANES persona context
```

The key design principle is:

> CES decides who exists in the simulated electorate and how much each respondent counts.  
> ANES only adds plausible psychological texture for similar respondents.

Do **not** append ANES respondents to the CES population.  
Do **not** let ANES respondents contribute directly to state-level vote shares.  
Do **not** use ANES direct vote choice or turnout intention as strict persona evidence.

---

## 2. Dataset Roles

### 2.1 CES 2024 Common Content

CES is the **population and prediction backbone**.

Use CES for:

- respondent-level agent construction;
- demographics;
- state / region / congressional district context;
- party identification;
- ideology;
- political registration / validated-voter fields;
- pre-election policy attitudes and economic evaluations;
- post-election vote / turnout labels where appropriate for target construction;
- state-level weighted aggregation;
- comparison against official state-level election truth.

Important CES characteristics:

- Large 2024 Common Content sample, about 60,000 cases.
- Conducted by YouGov in pre-election and post-election waves.
- Designed for election, representation, state, district, and subgroup analysis.
- Contains survey weights:
  - `commonweight`: adult population, pre-election / general adult estimates.
  - `commonpostweight`: adult population when using post-election items.
  - `vvweight`: validated registered adults.
  - `vvweight_post`: validated registered adults who completed both waves.
- Contains TargetSmart validation variables:
  - `TS_voterstatus`
  - `TS_g2024`
  - `TS_p2024`
  - `TS_p2024_party`
  - `TS_pp2024`
  - `TS_pp2024_party`
  - `TS_state`
  - `TS_partyreg`
- Contains direct post-election target variables such as self-reported turnout and presidential vote.

For this persona-enrichment design, CES provides the **query profile** for retrieval.

### 2.2 ANES 2024 Time Series

ANES is the **persona donor bank**.

Use ANES for:

- political psychology;
- political attention and campaign interest;
- candidate and party feeling thermometers;
- emotional states about the country;
- party identity strength / importance;
- trust, efficacy, and political responsiveness;
- open-ended likes/dislikes about candidates and parties;
- richer explanatory and affective texture than CES.

Important ANES characteristics:

- Smaller than CES, about 5,500 pre-election respondents.
- Much richer questionnaire, with over 1,700 variables in the local processed CSV.
- Pre-election and post-election waves.
- Mixed-mode survey design.
- Stronger for deep attitude/persona construction than for state-level population representation.

For this persona-enrichment design, ANES provides **payload information** after matching.

---

## 3. Design Principle: Retrieval Features and Payload Features Must Be Separate

The system must distinguish three categories of information.

### 3.1 Bridge Profile

The **bridge profile** is used to match CES respondents to similar ANES respondents.

It should contain variables that are:

- available in both CES and ANES, or meaningfully crosswalkable;
- pre-election or stable background variables;
- not direct 2024 presidential vote choice;
- not post-election outcome variables;
- not TargetSmart validation outcomes.

Examples:

- state / region;
- age or birth-year bucket;
- gender;
- race / ethnicity;
- education;
- voter registration status;
- party ID;
- party lean;
- ideology;
- political attention / campaign interest;
- media use when comparable;
- national economic evaluation;
- household economic situation;
- price / inflation perception;
- presidential or vice-presidential approval;
- immigration attitude;
- abortion attitude;
- gun policy attitude;
- environment / climate attitude;
- foreign policy views, when comparable.

The bridge profile answers:

> Which ANES respondents are similar enough to this CES respondent that their deeper persona information may be useful?

### 3.2 Persona Payload

The **persona payload** is retrieved from matched ANES donors and then summarized into prompt-ready facts.

It should contain ANES information that is valuable for persona construction but is absent from or weaker in CES.

Examples:

- candidate feeling thermometers;
- party feeling thermometers;
- global emotions about the country;
- party identity importance;
- trust in government;
- political efficacy / whether elections make government pay attention;
- open-ended likes/dislikes about Democratic and Republican presidential candidates;
- open-ended likes/dislikes about Democratic and Republican parties.

The payload answers:

> What psychological or narrative texture do similar ANES respondents tend to have?

### 3.3 Forbidden / Leaky Information

The system must exclude direct targets and near-targets from strict persona construction.

Forbidden in strict matching and strict payload:

- post-election turnout;
- post-election presidential vote;
- TargetSmart validated vote or turnout fields;
- direct pre-election presidential vote intention;
- direct pre-election presidential candidate preference;
- two-way presidential vote choice;
- direct pre-election turnout likelihood if the strict condition is meant to predict turnout without poll priors.

These variables may exist in the raw data, but they must not enter strict persona matching or strict prompt context.

---

## 4. Conceptual Pipeline

The final pipeline should be an **offline materialized RAG pipeline**, not a real-time vector-search RAG system.

```text
Raw CES + ANES data
  -> normalized canonical survey profiles
  -> ANES bridge profiles
  -> ANES persona payloads
  -> CES bridge profiles
  -> CES-to-ANES donor retrieval
  -> donor-weighted persona summarization
  -> ANES-derived inferred persona facts for each CES respondent
  -> merge with existing CES strict survey memory
  -> prompt assembly
  -> downstream simulation and aggregation unchanged
```

The retrieval layer should produce persistent intermediate artifacts so that:

- prompts are reproducible;
- leakage can be audited;
- each injected persona fact can be traced back to donor variables;
- low-support matches can be down-weighted or skipped;
- downstream simulation does not need to know how retrieval was performed.

---

## 5. Main Artifacts

The exact file paths can follow the current repository style, but the conceptual artifacts should be as follows.

### 5.1 ANES Bridge Profiles

One row per ANES respondent.

Required conceptual fields:

```text
anes_id
survey_year
state_fips
region
age_bucket
gender_canonical
race_ethnicity_canonical
education_canonical
registration_status
party_id_7
party_id_3
party_lean
ideology_7
political_attention
campaign_interest
economic_evaluation
presidential_approval
vp_approval
issue_immigration
issue_abortion
issue_guns
issue_environment
issue_foreign_policy
missingness_summary
safe_for_matching
```

This artifact must contain only safe variables for matching.

### 5.2 ANES Persona Payloads

One row per ANES respondent, either wide or long.

Conceptual groups:

```text
anes_id
payload_group
payload_name
payload_value
payload_value_canonical
source_variable
safe_policy
```

Recommended payload groups:

```text
affect_candidate
affect_party
emotion_country
identity_strength
trust_efficacy
open_text_candidate_like_dislike
open_text_party_like_dislike
open_text_theme
```

### 5.3 CES Bridge Profiles

One row per CES respondent.

Required conceptual fields mirror the ANES bridge profile as much as possible.

```text
caseid
survey_year
state_fips
region
age_bucket
gender_canonical
race_ethnicity_canonical
education_canonical
registration_status
party_id_7
party_id_3
party_lean
ideology_7
political_attention
media_use
economic_evaluation
household_income_trend
price_change
presidential_approval
vp_approval
issue_immigration
issue_abortion
issue_guns
issue_environment
issue_housing
issue_foreign_policy
missingness_summary
safe_for_matching
```

This profile is the query vector for retrieving ANES donors.

### 5.4 CES-to-ANES Match Table

One row per retrieved donor per CES respondent.

```text
caseid
match_rank
anes_id
distance
donor_weight
block_level
n_observed_bridge_features
n_missing_bridge_features
effective_feature_weight
state_match
region_match
party_distance
ideology_distance
demographic_distance
attitude_distance
support_confidence
matching_policy
created_at
```

The table must make matching transparent and auditable.

### 5.5 CES ANES-Derived Persona Facts

One row per generated persona fact per CES respondent.

```text
memory_fact_id
caseid
fact_role
topic
subtopic
fact_text
fact_priority
fact_strength
source
source_variables_used
donor_anes_ids
donor_weights
support_confidence
safe_as_memory
allowed_memory_policies
leakage_group
created_at
```

Required values:

```text
fact_role = inferred_persona
source = anes_persona_donor
```

The system must clearly distinguish these from observed CES facts.

### 5.6 Enriched Memory Cards

The final prompt input should merge:

```text
Observed CES strict memory facts
+ ANES-derived inferred persona facts
```

The prompt assembler must know which facts are observed and which are inferred.

---

## 6. Recommended Crosswalk Features

The following feature lists are normative at the conceptual level. Concrete source-variable names should be implemented through repository config files, because raw column names and processed column names may differ.

### 6.1 CES Bridge Feature Candidates

High-value CES fields for bridge matching include:

#### Demographics

- `birthyr` -> age bucket;
- `gender4` -> gender;
- `educ` -> education;
- `race`, `hispanic`, `multrace_*` -> race / ethnicity;
- `inputstate` -> state FIPS;
- `region` -> census region.

#### Political identity

- `pid3` -> 3-point party ID;
- `pid7` -> 7-point party ID;
- `votereg` -> registration status;
- ideology self-placement when available;
- candidate / party ideological placement items when available and safe.

#### Media and attention

- `CC24_300_*` -> media use in past 24 hours;
- related social media / political media variables if already processed.

#### Economic evaluations

- `CC24_301` -> national economy over past year;
- `CC24_302` -> household income over past year;
- `CC24_303` -> prices of everyday goods and services.

#### Approval and issue attitudes

Useful source-variable families include:

- `CC24_312*` approval items;
- `CC24_321*` gun / public-safety style items;
- `CC24_323*` immigration items;
- `CC24_324*` abortion items;
- `CC24_326*` environment / climate items;
- `CC24_328*` housing items;
- `CC24_330*` ideological placement / issue placement items.

Do not blindly include every item in these families. Only include variables that are pre-election, interpretable, and non-target.

### 6.2 ANES Bridge Feature Candidates

High-value ANES bridge fields include:

#### Political attention and participation background

- `V241004`: attention to government and politics;
- `V241005`: campaign interest;
- `V241012`: registered to vote.

#### Party and ideology

- ANES party ID recodes such as `V241227x`;
- party registration where available;
- ideology self-placement and issue-placement items when available.

#### Approval and issue views

- `V241134`: presidential job approval;
- `V241138`: vice-presidential approval;
- `V241141`: presidential handling of the economy;
- `V241147`: presidential handling of abortion policy;
- `V241150`: presidential handling of immigration;
- `V241153`: presidential handling of crime;
- comparable issue-position variables.

#### Geography and demographics

Use ANES state, region, age, gender, race / ethnicity, and education variables from the processed profile crosswalk.

### 6.3 ANES Persona Payload Candidates

These are not primarily for matching. They are used after retrieval to generate inferred persona facts.

#### Candidate and party affect

- `V241156`: Democratic presidential candidate feeling thermometer;
- `V241157`: Republican presidential candidate feeling thermometer;
- `V241158`: Joe Biden feeling thermometer;
- `V241165`: Democratic vice-presidential candidate feeling thermometer;
- `V241164`: Republican vice-presidential candidate feeling thermometer;
- `V241166`: Democratic Party feeling thermometer;
- `V241167`: Republican Party feeling thermometer.

Thermometer values should be treated as affective/persona signals, not as direct vote-choice labels.

#### Emotions about the country

- `V241117`: right direction / wrong track;
- `V241118`: hope;
- `V241119`: afraid;
- `V241120`: outrage;
- `V241121`: anger;
- `V241122`: happiness;
- `V241123`: worry;
- `V241124`: pride;
- `V241125`: irritation;
- `V241126`: nervousness.

#### Open-ended candidate evaluations

- `V241110`: likes about Democratic presidential candidate;
- `V241112`: dislikes about Democratic presidential candidate;
- `V241114`: likes about Republican presidential candidate;
- `V241116`: dislikes about Republican presidential candidate.

#### Open-ended party evaluations

- Democratic Party likes/dislikes;
- Republican Party likes/dislikes;
- use the processed ANES questionnaire/codebook to map exact variable names.

#### Trust, efficacy, and identity

- party identity importance;
- trust in the federal government;
- whether elections make government pay attention;
- whether parties care about people like the respondent;
- similar political efficacy variables.

---

## 7. Leakage Policy

The strict persona-enrichment policy should be named something like:

```text
strict_pre_no_vote_with_anes_persona_v1
```

It should inherit all existing CES strict exclusions and add ANES-specific exclusions.

### 7.1 CES Strict Exclusions

Strict CES memory and strict CES-to-ANES matching must exclude:

- post-election turnout;
- post-election vote choice;
- all `TS_*` vote-validation fields;
- direct pre-election turnout intention;
- direct pre-election presidential vote intention;
- direct pre-election presidential candidate preference;
- variables derived from any of the above.

Known CES variables to treat carefully include:

```text
CC24_363      # direct turnout / voting likelihood style item
CC24_364*     # direct presidential preference / vote intention style items
CC24_401      # post-election turnout self-report
CC24_410*     # post-election presidential vote self-report
TS_*          # TargetSmart validation fields
```

### 7.2 ANES Strict Exclusions

Strict ANES bridge profiles and persona payloads must exclude:

```text
V241035       # already voted in the general election
V241038       # voted for president
V241039*      # presidential vote choice if already voted
V241042       # intends to vote for president
V241043*      # intended presidential vote choice
V241045       # presidential candidate preference
V241046*      # preferred presidential candidate
V241049       # two-way presidential vote choice
V241100       # likelihood of voting, if turnout is a strict target
V242067*      # post-election presidential vote
V242068*      # related post-election vote details
```

This list should be expanded whenever a raw or processed ANES variable is recognized as direct vote, turnout, or post-election target information.

### 7.3 Allowed but Sensitive

Candidate and party thermometers are allowed in the recommended persona-affect policy because they are affective attitudes, not direct vote choices. However, they are strong vote predictors. Therefore:

- label them as `inferred_persona`;
- never label them as observed CES facts;
- keep their prompt priority below actual CES respondent facts;
- do not use them in a policy that claims to exclude all candidate-affect information.

---

## 8. Matching Algorithm

Use structured proximity matching, not raw text embedding search, for the first implementation.

The survey variables are mostly categorical, ordinal, or multi-select. A weighted mixed-type distance is more transparent and auditable than dense embeddings.

### 8.1 Distance Function

For a CES respondent \(c\) and an ANES respondent \(a\), define:

\[
D(c,a)=
\frac{
\sum_{j \in J} w_j \cdot m_j(c,a) \cdot d_j(c_j,a_j)
}{
\sum_{j \in J} w_j \cdot m_j(c,a)
}
+
P(c,a).
\]

Where:

- \(J\) is the set of bridge features;
- \(w_j\) is the feature weight;
- \(m_j(c,a)=1\) if both values are observed and comparable, otherwise \(0\);
- \(d_j\) is the feature-specific distance;
- \(P(c,a)\) is an optional penalty for weak support or fallback matching.

Feature distances:

```text
Nominal:
  d = 0 if same category, 1 otherwise.

Ordinal:
  d = |rank_c - rank_a| / (K - 1).

Continuous:
  robust-scale or min-max normalize, then use clipped absolute difference.

Multi-select:
  d = 1 - JaccardSimilarity(set_c, set_a).

Missing:
  excluded from numerator and denominator, but counted in support diagnostics.
```

### 8.2 Feature Weights

The matching weights should reflect predictive and persona relevance.

Recommended starting weights:

```text
party_id_7                         4.0
party_id_3 / party lean             3.0
ideology_7                          3.0
presidential / VP approval          2.0
economic evaluation                 1.5
prices / inflation perception       1.5
immigration                         1.2
abortion                            1.2
guns / public safety                1.0
environment / climate               1.0
foreign policy                      0.8
political attention                 1.0
campaign interest                   1.0
age bucket                          1.0
education                           1.0
race / ethnicity                    1.2
gender                              0.6
state                               0.6
region                              0.5
media use                           0.5
```

Interpretation:

- party and ideology should strongly guide retrieval;
- state and region should matter, but should not force bad matches;
- demographics matter, but should not dominate political attitudes;
- issue variables help distinguish voters within party/ideology strata.

### 8.3 Candidate Generation and Fallback

Because ANES is much smaller than CES, exact state-level matching will often fail.

Use a fallback sequence:

```text
1. Try same state, similar party ID / ideology.
2. If too few donors, relax to same region.
3. If still too few donors, use national pool.
4. Always rank by full weighted distance.
```

Recommended retrieval parameters:

```text
k_retrieve: 20
k_min_support: 5
fallback_order:
  - same_state
  - same_region
  - national
```

State should be a useful contextual signal, not a hard requirement.

### 8.4 Donor Weights

After retrieving top-k ANES donors, convert distances into donor weights:

\[
\tilde{w}_i =
\frac{\exp(-D_i/\tau)}
{\sum_{\ell=1}^{k}\exp(-D_\ell/\tau)}.
\]

Recommended starting value:

```text
tau = 0.15
```

If distances are very compressed or very spread out, tune \(\tau\) so that the top donor matters but does not completely dominate.

### 8.5 Effective Donor Support

Compute effective donor size:

\[
ESS = \frac{1}{\sum_i \tilde{w}_i^2}.
\]

Use ESS and average distance to define support confidence:

```text
high:
  ESS >= 8 and mean distance <= 0.25

medium:
  ESS >= 4 and mean distance <= 0.40

low:
  otherwise
```

Low-confidence matches should produce either fewer persona facts or explicitly cautious persona facts.

---

## 9. Persona Summarization

The system should not inject raw donor records directly.

Instead, it should summarize the donor-weighted payload into a small number of inferred persona facts.

### 9.1 General Rule

Bad prompt style:

```text
The respondent rated Harris 83 and Trump 12.
```

Good prompt style:

```text
Among matched ANES respondents with similar demographics, party identity,
ideology, economic views, and issue views, the inferred affect profile is
warmer toward the Democratic candidate than the Republican candidate.
```

The second style makes clear that the fact is inferred, not directly observed.

### 9.2 Recommended Persona Fact Types

Generate at most a small number of facts per CES respondent.

Recommended fact types:

```text
candidate_affect_summary
party_affect_summary
emotion_summary
identity_strength_summary
trust_efficacy_summary
open_text_theme_summary
```

Do not flood the prompt. The ANES persona section should complement CES facts, not overwhelm them.

### 9.3 Candidate and Party Affect

From thermometer payloads, compute donor-weighted summaries such as:

```text
Democratic candidate affect: cold / neutral / warm
Republican candidate affect: cold / neutral / warm
Candidate affect gap: Dem warmer / Rep warmer / balanced
Democratic Party affect: cold / neutral / warm
Republican Party affect: cold / neutral / warm
Party affect gap: Dem warmer / Rep warmer / balanced
```

Suggested thresholds for 0-100 thermometers:

```text
0-35: cold
36-64: neutral or mixed
65-100: warm
```

For affect gap:

```text
gap = weighted_mean(dem_affect) - weighted_mean(rep_affect)

gap >= +15: warmer toward Democratic side
gap <= -15: warmer toward Republican side
otherwise: mixed or balanced
```

### 9.4 Emotions

From country-emotion payloads, identify the strongest donor-weighted emotions.

Possible summary:

```text
Similar respondents often express high worry and anger about the country's direction.
```

The system should prefer relative summaries:

```text
high worry
high anger
low hope
mixed pride
```

rather than raw numerical scores.

### 9.5 Open-Text Themes

Open-ended ANES text should be converted into themes before injection.

Do not put long raw ANES text into the prompt.

Recommended theme taxonomy:

```text
economy / inflation
jobs / wages
immigration / border
abortion / reproductive rights
guns / crime / public safety
democracy / election legitimacy
candidate character / honesty
candidate competence / experience
age / health / fitness
corruption / establishment distrust
foreign policy / Ukraine / Israel-Gaza
civil rights / identity / race / gender
religion / moral values
party loyalty
anti-MAGA / extremism
anti-left / socialism / progressivism
not enough information
general approval / general dislike
other
```

Each donor open-text answer should be transformed into:

```text
target: democratic_candidate | republican_candidate | democratic_party | republican_party
stance: like | dislike
themes: list[str]
sentiment_strength: low | medium | high
```

Then aggregate themes across donors.

Prompt-ready example:

```text
Common themes among matched ANES donors who dislike the Republican candidate
include candidate character, democracy concerns, and policy extremism.
```

### 9.6 Fact Priority

Observed CES facts should always outrank inferred ANES persona facts.

Recommended priority bands:

```text
Observed CES demographics / profile facts: high
Observed CES strict survey facts: high
ANES-derived inferred persona facts: medium-low
Poll-prior facts: separate policy only, never mixed into strict persona facts
```

If the current repository uses numeric priorities, use lower numbers or lower weights for `inferred_persona` than for direct CES respondent answers.

---

## 10. Prompt Contract

The final prompt should clearly separate observed facts from inferred persona context.

Recommended structure:

```text
You are simulating a real survey respondent in the 2024 U.S. election.

Observed CES respondent facts:
- [direct CES fact]
- [direct CES fact]
- [direct CES fact]

ANES-derived inferred persona context:
The following statements are not this respondent's own survey answers.
They are inferred from matched ANES respondents with similar demographics,
party identity, ideology, economic views, and issue positions. Use them only
as background texture. Do not let them override observed CES facts.
- [inferred persona fact]
- [inferred persona fact]

Task:
Predict whether this respondent voted and how they voted.
Return the required JSON format.
```

The prompt must preserve this distinction:

```text
observed respondent evidence
vs.
inferred donor-based persona context
```

This distinction is important for interpretability and leakage control.

---

## 11. Recommended Memory Policy Names

Use separate memory policies so runs remain interpretable.

### 11.1 Strict CES Only

```text
strict_pre_no_vote_v1
```

Contains:

```text
CES demographics
CES party / ideology
CES non-target pre-election survey facts
```

Excludes:

```text
direct vote intention
direct turnout intention
post-election vote / turnout
TargetSmart fields
```

### 11.2 Strict CES + ANES Persona

```text
strict_pre_no_vote_with_anes_persona_v1
```

Contains:

```text
all strict CES facts
ANES-derived inferred persona facts
```

Still excludes:

```text
direct vote intention
direct turnout intention
post-election vote / turnout
TargetSmart fields
ANES direct vote-choice variables
ANES direct turnout-intention variables
```

### 11.3 Poll-Informed Policy

```text
poll_informed_pre_v1
```

May allow:

```text
direct pre-election vote intention
direct pre-election turnout intention
```

But this is a different information condition and should not be mixed into strict persona enrichment.

---

## 12. Handling Low-Support or Ambiguous Matches

The retrieval system must handle weak matches gracefully.

### 12.1 Low Support

If a CES respondent has weak ANES support:

```text
ESS too low
distance too high
too many missing bridge features
only national fallback available
```

Then:

- generate fewer ANES persona facts;
- lower `fact_strength`;
- lower `fact_priority`;
- optionally skip ANES persona injection entirely;
- never fabricate precise claims.

Prompt-ready low-support language:

```text
The ANES-derived persona context for this respondent has low support and should be used cautiously.
```

### 12.2 Missing Key Features

If party ID or ideology is missing:

- rely more on demographics, geography, economic views, and issue attitudes;
- reduce confidence;
- avoid strong affect summaries unless donors agree strongly.

### 12.3 Conflicting Donors

If matched donors are split:

- do not force a strong persona;
- express ambiguity.

Example:

```text
Matched ANES respondents show mixed affect toward both major candidates,
with no clear candidate-affect direction.
```

---

## 13. Quality Gates

These are not experiments; they are implementation safeguards.

### 13.1 Leakage Gate

Before building prompts, assert:

```text
No forbidden CES variables appear in memory facts.
No forbidden ANES variables appear in bridge features.
No forbidden ANES variables appear in persona payloads.
No forbidden ANES variables appear in source_variables_used.
No raw TargetSmart fields appear in strict prompt context.
No post-election fields appear in strict prompt context.
```

### 13.2 Traceability Gate

Every generated ANES-derived fact must retain:

```text
donor ANES IDs
donor weights
source variable names
matching policy
support confidence
```

This does not need to be shown to the LLM, but it must be present in the artifact.

### 13.3 Prompt Separation Gate

Prompt builder must render ANES facts under a separate heading.

Never merge inferred ANES facts into the same bullet list as observed CES answers without labeling them.

### 13.4 Support Gate

Every CES respondent with ANES persona injection must have:

```text
number of donors
effective donor size
mean distance
fallback level
support confidence
```

If support is below threshold, skip or weaken persona facts.

---

## 14. Recommended System Architecture

The implementation can be organized around four conceptual modules.

### 14.1 Canonical Survey Profile Builder

Responsibilities:

- normalize CES variables into canonical bridge features;
- normalize ANES variables into canonical bridge features;
- handle missing codes;
- harmonize categorical and ordinal scales;
- write bridge profile artifacts.

Input:

```text
processed CES respondents / answers
processed ANES respondents / answers
crosswalk configs
```

Output:

```text
CES bridge profiles
ANES bridge profiles
```

### 14.2 ANES Persona Bank Builder

Responsibilities:

- extract safe ANES payload variables;
- normalize thermometers and emotion scales;
- classify open-ended text into themes;
- create donor persona payloads;
- enforce ANES leakage policy.

Input:

```text
processed ANES answers
ANES payload crosswalk
open-text theme taxonomy
leakage policy
```

Output:

```text
ANES persona payloads
ANES persona cards or payload summaries
```

### 14.3 CES-to-ANES Persona Retriever

Responsibilities:

- retrieve top-k ANES donors for each CES respondent;
- compute weighted mixed-type distances;
- apply fallback blocking;
- compute donor weights;
- compute support diagnostics;
- write match table.

Input:

```text
CES bridge profiles
ANES bridge profiles
matching config
```

Output:

```text
CES-to-ANES match table
support diagnostics
```

### 14.4 Persona Fact Renderer

Responsibilities:

- aggregate donor payloads;
- convert payload aggregates into concise persona facts;
- assign fact roles, priorities, strengths, and leakage groups;
- merge with CES strict memory cards;
- write enriched memory artifacts for existing prompt generation.

Input:

```text
CES-to-ANES match table
ANES persona payloads
CES strict memory facts/cards
```

Output:

```text
ANES-derived inferred persona facts
CES enriched memory facts/cards
```

Downstream simulation code should not need to understand ANES matching internals.

---

## 15. Canonical Missing-Code Rules

Survey data uses many non-substantive response codes. The exact codes differ by survey and variable family, but the canonical system should normalize them early.

Recommended canonical treatment:

```text
Skipped
Not Asked
Refused
Don't know
Don't recognize
Don't know where to rate
Not sure, when substantively ambiguous
```

Map these to:

```text
missing / non-substantive / uncertain
```

depending on context.

For ANES thermometers:

```text
0-100: valid thermometer value
998: don't know where to rate -> missing thermometer
999: don't recognize -> missing thermometer, optional recognition signal
```

For CES-style codes:

```text
8, 9, 98, 99, 55
```

often represent skipped, not asked, don't know, or related non-substantive answers. Confirm by variable family and codebook, then normalize consistently.

---

## 16. Open-Text Processing Rules

Open-ended ANES text is valuable but potentially noisy.

The final system should:

1. keep raw text in local processed artifacts only;
2. classify text into a controlled theme taxonomy;
3. optionally keep a short sanitized excerpt for audit only;
4. inject only theme summaries into LLM prompts;
5. never let a single donor's raw text dominate a CES respondent's prompt.

Recommended text-to-theme artifact:

```text
anes_id
source_variable
target
stance
raw_text_hash
theme_1
theme_1_confidence
theme_2
theme_2_confidence
sentiment_strength
safe_for_prompt
```

Prompt-injected form should be aggregate, not donor-specific.

---

## 17. What the Final Agent Should Receive

A CES respondent enriched with ANES persona should receive three conceptual layers.

### 17.1 Stable Profile

Observed from CES:

```text
state
age
gender
race / ethnicity
education
registration status
party ID
ideology
```

### 17.2 Observed CES Survey Memory

Observed from CES strict pre-election facts:

```text
economic evaluation
price perception
approval / disapproval
issue positions
media / political engagement
other safe pre-election facts
```

### 17.3 Inferred ANES Persona Context

Inferred from matched ANES donors:

```text
candidate affect direction
party affect direction
emotional tone
political identity intensity
trust / efficacy orientation
common open-text explanation themes
```

The final prompt should make clear that layer 3 is inferred.

---

## 18. Non-Goals

Do not implement the following as the main design:

### 18.1 Do Not Merge ANES Into the Electorate

ANES should not become additional simulated voters. Its sample size and design are not suitable for replacing or expanding the CES state-level population base.

### 18.2 Do Not Use Raw Vector RAG Over Survey Text

This is not a question-answering problem. The goal is not to retrieve arbitrary passages.

The first implementation should use structured matching plus summarized persona payloads. Embeddings can be added later for open-text theme classification or donor-card similarity, but they should not replace structured matching.

### 18.3 Do Not Let ANES Override CES

If CES says the respondent is a strong Republican but donor summaries are mixed, the prompt should retain the CES fact as primary. ANES context is supplementary.

### 18.4 Do Not Hide Inferred Status

The LLM must know which context is observed and which context is inferred.

### 18.5 Do Not Mix Strict and Poll-Informed Modes

Direct vote-intention variables belong only in a separate poll-informed condition. They should not leak into strict persona enrichment.

---

## 19. Minimal Final Pipeline Contract

A successful implementation should satisfy the following contract.

### Inputs

```text
processed CES respondent artifacts
processed CES answer artifacts
processed ANES respondent artifacts
processed ANES answer artifacts
CES strict memory facts/cards
crosswalk configs
leakage policy configs
matching config
theme taxonomy
```

### Internal Artifacts

```text
CES bridge profiles
ANES bridge profiles
ANES persona payloads
CES-to-ANES match table
ANES-derived inferred persona facts
enriched CES memory facts/cards
```

### Outputs to Existing Simulator

```text
respondent agents with enriched memory
prompt-ready memory cards
prompt preview preserving observed-vs-inferred separation
```

### Invariants

```text
CES remains the population base.
CES weights remain the aggregation weights.
ANES contributes no population mass.
ANES contributes only inferred persona context.
Strict mode excludes direct vote and turnout targets.
All ANES-derived facts are traceable and labeled.
Downstream simulation and aggregation remain unchanged.
```

---

## 20. Summary

The intended final system is a **structured, offline, auditable persona-RAG layer**:

```text
CES respondent as query
ANES respondents as persona donors
weighted proximity matching
donor-weighted persona summarization
inferred persona facts
strict prompt separation
unchanged CES state aggregation
```

This design uses each dataset where it is strongest:

```text
CES:
  representative respondent base, weights, state coverage, aggregation.

ANES:
  rich political psychology, affect, emotions, open-ended rationales.
```

The result should be a better voter-agent prompt without compromising the statistical role of CES or leaking direct election outcomes into strict prediction.
