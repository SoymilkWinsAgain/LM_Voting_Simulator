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

The default fixture run uses `model.provider: mock` so tests stay deterministic.
To run against Windows Ollama from WSL, set the run config model block to:

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
