"""LLM backends behind one interface.

"Design for both" means the rest of the system only ever sees `LLMBackend`.
Switch cloud <-> on-device by changing agent.backend in the config; no other
code changes. The local backend speaks the OpenAI chat API, which Ollama,
vLLM, and LM Studio all expose - so the same code targets an edge box.
"""

from __future__ import annotations

import json
from typing import Protocol


class LLMBackend(Protocol):
    def chat(self, system: str, user: str, json_mode: bool = False) -> str: ...


class ClaudeBackend:
    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 1024):
        from anthropic import Anthropic   # reads ANTHROPIC_API_KEY from env
        self.client = Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        if json_mode:
            user += "\n\nRespond with ONLY a valid JSON object, no prose, no code fences."
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if b.type == "text")


class LocalBackend:
    """Any OpenAI-compatible endpoint (Ollama / vLLM / LM Studio)."""

    def __init__(self, base_url: str, model: str, max_tokens: int = 1024):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(
            model=self.model, max_tokens=self.max_tokens, **kwargs,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return resp.choices[0].message.content


def build_backend(cfg) -> LLMBackend:
    a = cfg.agent
    if a.backend == "claude":
        return ClaudeBackend(a.claude_model)
    if a.backend == "local":
        return LocalBackend(a.local_base_url, a.local_model)
    raise ValueError(f"Unknown agent.backend: {a.backend!r}")


def parse_json(text: str) -> dict:
    """Tolerant JSON parse - strips code fences a local model might add."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)
