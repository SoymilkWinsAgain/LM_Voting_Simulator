"""Configuration loading and light validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .io import load_yaml


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    year: int
    office: str = "president"
    election_day: str | None = None
    states: list[str]
    level: str = "state"


class PopulationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str = "fixture"
    n_agents_per_state: int = 50
    cell_schema: str = "mvp_state_cell_v1"
    allocation: Literal["deterministic", "multinomial"] = "deterministic"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: str = "mock"
    model_name: str = "mock-voter-v1"
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = 300
    response_format: str = "json"
    cache_enabled: bool = True


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    seed: int = 20260426
    mode: str = "election_simulation"
    scenario: ScenarioConfig
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    baselines: list[str] = Field(default_factory=lambda: ["majority"])
    paths: dict[str, Any] = Field(default_factory=dict)
    gdelt: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    anes_memory: dict[str, Any] = Field(default_factory=lambda: {"memory_policy": "safe_survey_memory_v1"})
    questions: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return Path(self.paths.get("run_dir", f"data/runs/{self.run_id}"))

    @property
    def processed_dir(self) -> Path:
        return Path(self.paths.get("processed_dir", f"data/processed/{self.run_id}"))


def load_run_config(path: str | Path) -> RunConfig:
    return RunConfig.model_validate(load_yaml(path))


def load_cell_schema(path: str | Path) -> dict[str, Any]:
    schema = load_yaml(path)
    required = {"name", "columns"}
    missing = required - set(schema)
    if missing:
        raise ValueError(f"Cell schema missing required keys: {sorted(missing)}")
    return schema


def resolve_config_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value)
