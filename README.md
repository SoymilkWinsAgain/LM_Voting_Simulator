# LM Voting Simulator

MVP implementation for a synthetic election simulation pipeline.

The first supported target is a fixture-backed end-to-end run:

```bash
conda run -n jigsaw python -m election_sim.cli run-simulation \
  --run-config configs/runs/first_e2e_2024_pa_fixture.yaml
```

Expected outputs are written under:

```text
data/runs/first_e2e_2024_pa_fixture/
```

Useful commands:

```bash
conda run -n jigsaw python -m election_sim.cli validate-config \
  --config configs/runs/first_e2e_2024_pa_fixture.yaml

conda run -n jigsaw python -m pytest
```

Real ANES 2024 one-agent smoke run:

```bash
conda run -n jigsaw python -m election_sim.cli build-anes \
  --config configs/datasets/anes_2024_real_min.yaml \
  --profile-crosswalk configs/crosswalks/anes_2024_real_min_profile.yaml \
  --question-crosswalk configs/crosswalks/anes_2024_real_min_questions.yaml \
  --out data/processed/anes/2024_real_min

conda run -n jigsaw python -m election_sim.cli build-anes-memory \
  --respondents data/processed/anes/2024_real_min/anes_respondents.parquet \
  --answers data/processed/anes/2024_real_min/anes_answers.parquet \
  --fact-templates configs/fact_templates/anes_2024_real_min_facts.yaml \
  --policy safe_survey_memory_v1 \
  --out data/processed/anes/2024_real_min \
  --max-facts 6

conda run -n jigsaw python -m election_sim.cli run-simulation \
  --run-config configs/runs/real_anes_2024_one_agent_ollama.yaml
```

The real ANES prompt and model response are written to:

```text
data/runs/real_anes_2024_one_agent_ollama/prompt_preview.md
```

## CES 2024 respondent-level mainline

Place the local CES files under:

```text
data/raw/ces/CES_2024.csv
data/raw/ces/CCES24_Common_pre.docx
data/raw/ces/CCES24_Common_post.docx
data/raw/ces/CES_2024_GUIDE_vv.pdf
```

The runtime pipeline reads the static YAML mappings in `configs/`; it does not
parse DOCX/PDF manuals during a run. If your raw CSV has a different name,
change `configs/datasets/ces_2024_real_vv.yaml`.

Build processed CES artifacts:

```bash
conda run -n jigsaw python -m election_sim.cli build-ces \
  --config configs/datasets/ces_2024_real_vv.yaml \
  --profile-crosswalk configs/crosswalks/ces_2024_profile.yaml \
  --question-crosswalk configs/crosswalks/ces_2024_pre_questions.yaml \
  --target-crosswalk configs/crosswalks/ces_2024_targets.yaml \
  --context-crosswalk configs/crosswalks/ces_2024_context.yaml \
  --out data/processed/ces/2024_common_vv
```

Build strict pre-election memory:

```bash
conda run -n jigsaw python -m election_sim.cli build-ces-memory \
  --respondents data/processed/ces/2024_common_vv/ces_respondents.parquet \
  --answers data/processed/ces/2024_common_vv/ces_answers.parquet \
  --fact-templates configs/fact_templates/ces_2024_common_facts.yaml \
  --policy strict_pre_no_vote_v1 \
  --out data/processed/ces/2024_common_vv \
  --max-facts 24
```

Run the deterministic CES smoke:

```bash
conda run -n jigsaw python -m election_sim.cli run-simulation \
  --run-config configs/runs/ces_2024_president_smoke.yaml
```

Expected smoke outputs:

```text
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

For swing-state experiments, build strict memory once as above, then run:

```bash
conda run -n jigsaw python -m election_sim.cli run-simulation \
  --run-config configs/runs/ces_2024_president_swing_strict_pre.yaml
```

For poll-informed prompts, build a separate memory directory and run:

```bash
conda run -n jigsaw python -m election_sim.cli build-ces-memory \
  --respondents data/processed/ces/2024_common_vv/ces_respondents.parquet \
  --answers data/processed/ces/2024_common_vv/ces_answers.parquet \
  --fact-templates configs/fact_templates/ces_2024_common_facts.yaml \
  --policy poll_informed_pre_v1 \
  --out data/processed/ces/2024_common_vv_poll \
  --max-facts 24

conda run -n jigsaw python -m election_sim.cli run-simulation \
  --run-config configs/runs/ces_2024_president_swing_poll_informed.yaml
```

Leakage policies:

- `strict_pre_no_vote_v1`: excludes post-election turnout/vote, all `TS_*`
  validation fields, and direct pre-election vote intention/preference.
- `poll_informed_pre_v1`: still excludes post-election and TargetSmart fields,
  but allows direct pre-election turnout/vote intention as `poll_prior`.
- `post_hoc_explanation_v1`: explanatory-only; report marks it as not a formal
  prediction policy.

CES configs use `weight_common_post` by default because post-election labels are
used for evaluation. If a row has a missing or invalid weight, the agent table
records `weight_missing_reason` and falls back to weight `1.0`.

The default fixture run uses `model.provider: mock` so tests stay deterministic.
Ollama uses the native `/api/chat` endpoint so Qwen thinking can be disabled
and JSON output stays parseable. To run against Windows Ollama from WSL, set
the run config model block to:

```yaml
model:
  provider: ollama
  base_url: http://172.26.48.1:11434
  model_name: qwen3.5:9b-q4_K_M
  temperature: 0.0
```

For DeepSeek or another OpenAI-compatible endpoint:

```yaml
model:
  provider: openai_compatible
  base_url: https://api.deepseek.com/v1
  api_key_env: DEEPSEEK_API_KEY
  model_name: deepseek-chat
```

This repository intentionally does not include real ANES/CES/MIT data.
Place real source files under `data/raw/...` and point the YAML configs to
those files.

Known limitations:

- The default aggregate MIT comparison uses a 7-state fixture until a real MIT
  official CSV is supplied and referenced by config.
- The default swing configs use `100` CES respondents per state for fast smoke
  validation, not final research-scale estimates.
- Mock provider runs are deterministic contract tests; real LLM providers may
  produce parse failures, which are preserved in `responses.parquet` and
  summarized in `eval_report.md`.
