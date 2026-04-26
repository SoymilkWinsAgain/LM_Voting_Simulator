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
