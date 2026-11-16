"""LLM backends that return structured (Pydantic-validated) output.

Two interchangeable backends:
* AnthropicBackend – the official Anthropic SDK (default when ANTHROPIC_API_KEY is set);
* ClaudeCLIBackend – a locally installed and logged-in Claude Code CLI (`claude -p --json-schema`),
  which lets the tool run on a Claude subscription without an API key.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Type, TypeVar

from pydantic import BaseModel

MODEL = os.environ.get("AI_ANALYST_MODEL", "claude-opus-5-5")
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the model cannot produce a valid structured answer."""


class AnthropicBackend:
    name = "anthropic-api"

    def __init__(self) -> None:
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()

    def structured(self, system: str, user: str, schema: Type[T]) -> T:
        anthropic = self._anthropic
        try:
            response = self.client.messages.parse(
                model=MODEL,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
        except anthropic.AuthenticationError as e:
            raise LLMError("Невірний або відсутній ANTHROPIC_API_KEY") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Перевищено ліміт запитів API, спробуйте пізніше") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Помилка API {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Немає з'єднання з API Anthropic") from e
        if response.stop_reason == "refusal":
            raise LLMError("Модель відмовилася відповідати на запит")
        if response.parsed_output is None:
            raise LLMError(f"Модель не повернула структуровану відповідь (stop_reason={response.stop_reason})")
        return response.parsed_output


def _strict_schema(schema: Type[BaseModel]) -> dict:
    """JSON Schema of a Pydantic model with additionalProperties=false on every object."""
    js = schema.model_json_schema()

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(js)
    return js


class ClaudeCLIBackend:
    name = "claude-code-cli"

    def __init__(self) -> None:
        self.exe = shutil.which("claude")
        if not self.exe:
            raise LLMError("Claude Code CLI не знайдено в PATH")

    def structured(self, system: str, user: str, schema: Type[T]) -> T:
        cmd = [
            self.exe, "-p",
            "--output-format", "json",
            "--json-schema", json.dumps(_strict_schema(schema), ensure_ascii=False),
            "--system-prompt", system,
            "--tools", "",
            "--no-session-persistence",
            "--model", MODEL,
        ]
        try:
            proc = subprocess.run(cmd, input=user, capture_output=True, text=True,
                                  encoding="utf-8", timeout=900)
        except subprocess.TimeoutExpired as e:
            raise LLMError("Claude Code CLI не відповів за 15 хвилин") from e
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise LLMError(f"Неочікувана відповідь CLI: {proc.stdout[:300]} {proc.stderr[:300]}") from e
        if data.get("is_error"):
            raise LLMError(f"Claude Code CLI: {data.get('result')}")
        payload = data.get("structured_output", data.get("result"))
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as e:
                raise LLMError(f"Модель повернула не JSON: {payload[:300]}") from e
        return schema.model_validate(payload)


def get_backend(name: str | None = None):
    """auto: API key -> Anthropic SDK; otherwise a logged-in Claude Code CLI; otherwise the SDK
    (which can still pick up an `ant auth login` profile)."""
    name = (name or os.environ.get("AI_ANALYST_BACKEND") or "auto").lower()
    if name in ("anthropic", "api"):
        return AnthropicBackend()
    if name in ("claude-cli", "cli"):
        return ClaudeCLIBackend()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicBackend()
    if shutil.which("claude"):
        return ClaudeCLIBackend()
    return AnthropicBackend()
