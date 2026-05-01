"""LLM clients for mock, Ollama, and OpenAI-compatible providers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib import request

from .ces_schema import format_turnout_vote_choice_response
from .config import ModelConfig


class MockLLMClient:
    """Deterministic local client used by default smoke tests."""

    model_name = "mock-voter-v1"

    def complete(self, prompt_text: str, allowed: list[str]) -> str:
        lowered = prompt_text.lower()
        if '"choice": "not_vote|candidate_a|candidate_b"' in lowered:
            if "party identification: republican" in lowered or "ideology: conservative" in lowered:
                return json.dumps({"choice": "candidate_b"})
            if "party identification: democrat" in lowered or "ideology: liberal" in lowered:
                return json.dumps({"choice": "candidate_a"})
            return json.dumps({"choice": "not_vote"})
        if '"choice": "not_vote|democrat|republican"' in lowered or "allowed choices:" in lowered:
            if "party identification: democrat" in lowered:
                choice = "democrat"
            elif "party identification: republican" in lowered:
                choice = "republican"
            elif "ideology: conservative" in lowered:
                choice = "republican"
            elif "ideology: liberal" in lowered:
                choice = "democrat"
            else:
                choice = "not_vote"
            return format_turnout_vote_choice_response(choice)
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


@dataclass
class OllamaClient:
    model_name: str
    base_url: str | None
    temperature: float
    max_tokens: int
    response_format: str

    def complete(self, prompt_text: str, allowed: list[str]) -> str:
        base = (self.base_url or "http://172.26.48.1:11434").rstrip("/")
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only valid JSON. Do not explain. Do not include markdown.",
                },
                {"role": "user", "content": prompt_text},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if self.response_format == "json":
            payload["format"] = "json"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{base}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=120) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
        return parsed.get("message", {}).get("content") or "{}"


def build_llm_client(cfg: ModelConfig):
    provider = cfg.provider
    if provider == "mock":
        return MockLLMClient()
    if provider == "ollama":
        return OllamaClient(
            model_name=cfg.model_name,
            base_url=cfg.base_url,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            response_format=cfg.response_format,
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
