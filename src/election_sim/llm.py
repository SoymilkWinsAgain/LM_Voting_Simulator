"""LLM clients for mock, Ollama, and OpenAI-compatible providers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .config import ModelConfig


class MockLLMClient:
    """Deterministic local client used by default smoke tests."""

    model_name = "mock-voter-v1"

    def complete(self, prompt_text: str, allowed: list[str]) -> str:
        lowered = prompt_text.lower()
        if "party identification: democrat" in lowered:
            answer = "democrat"
            confidence = 0.82
        elif "party identification: republican" in lowered:
            answer = "republican"
            confidence = 0.82
        elif "ideology: conservative" in lowered:
            answer = "republican"
            confidence = 0.65
        elif "ideology: liberal" in lowered:
            answer = "democrat"
            confidence = 0.65
        else:
            answer = allowed[0]
            confidence = 0.45
        if answer not in allowed:
            answer = allowed[0]
        return json.dumps({"answer": answer, "confidence": confidence})


@dataclass
class OpenAICompatibleClient:
    model_name: str
    base_url: str | None
    api_key: str
    temperature: float
    max_tokens: int

    def complete(self, prompt_text: str, allowed: list[str]) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Return only valid JSON. Do not include markdown.",
                },
                {"role": "user", "content": prompt_text},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or "{}"


def _ollama_base_url(base_url: str | None) -> str:
    base = base_url or "http://172.26.48.1:11434"
    return base.rstrip("/") if base.rstrip("/").endswith("/v1") else f"{base.rstrip('/')}/v1"


def build_llm_client(cfg: ModelConfig):
    provider = cfg.provider
    if provider == "mock":
        return MockLLMClient()
    if provider == "ollama":
        return OpenAICompatibleClient(
            model_name=cfg.model_name,
            base_url=_ollama_base_url(cfg.base_url),
            api_key="ollama",
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    if provider in {"openai", "openai_compatible"}:
        env = cfg.api_key_env or ("OPENAI_API_KEY" if provider == "openai" else "DEEPSEEK_API_KEY")
        api_key = os.environ.get(env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {env}")
        return OpenAICompatibleClient(
            model_name=cfg.model_name,
            base_url=cfg.base_url,
            api_key=api_key,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    raise ValueError(f"Unknown model provider: {provider}")
