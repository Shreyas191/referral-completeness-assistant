"""Thin LLM wrapper: JSON in, validated Pydantic out, with one repair retry.

Works with Anthropic (Claude) or OpenAI models, chosen by LLM_PROVIDER in .env:

    LLM_PROVIDER=anthropic   LLM_MODEL=claude-haiku-5-5   ANTHROPIC_API_KEY=...
    LLM_PROVIDER=openai      LLM_MODEL=gpt-6-luna         OPENAI_API_KEY=...

Keep one provider and model for every system in an experiment, so comparisons measure the
architecture rather than the model. The harness records provider and model in metrics.json.
"""
import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from referral.config import DEFAULT_MODELS, settings

T = TypeVar("T", bound=BaseModel)
PROVIDERS = ("anthropic", "openai")


class LLM:
    def __init__(self, model: str | None = None, provider: str | None = None):
        self.provider = (provider or settings.llm_provider).lower()
        if self.provider not in PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {PROVIDERS}, got {self.provider!r}")
        if model:
            self.model = model
        elif provider and provider.lower() != settings.llm_provider:
            self.model = DEFAULT_MODELS[self.provider]  # switched provider in code: don't reuse the other's model
        else:
            self.model = settings.llm_model
        self._client = self._make_client()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def _make_client(self):
        if self.provider == "anthropic":
            import anthropic

            return anthropic.Anthropic(timeout=settings.llm_timeout_s, max_retries=settings.llm_max_retries)
        import openai

        return openai.OpenAI(timeout=settings.llm_timeout_s, max_retries=settings.llm_max_retries)

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def complete(self, system: str, user: str, max_tokens: int = 4000, json_mode: bool = False) -> str:
        if self.provider == "anthropic":
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            self._count(msg.usage.input_tokens, msg.usage.output_tokens)
            return "".join(block.text for block in msg.content if block.type == "text")

        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        self._count(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return resp.choices[0].message.content or ""

    def _count(self, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def structured(self, system: str, user: str, schema: type[T], retries: int = 1, max_tokens: int = 4000) -> T:
        prompt = f"{user}\n\nReturn only JSON matching this schema:\n{json.dumps(schema.model_json_schema())}"
        raw = self.complete(system, prompt, max_tokens=max_tokens, json_mode=True)
        for attempt in range(retries + 1):
            try:
                return schema.model_validate_json(_extract_json(raw))
            except (ValidationError, ValueError) as err:
                if attempt == retries:
                    raise
                raw = self.complete(system, f"{prompt}\n\nYour last answer was invalid ({err}). Fix it.",
                                    max_tokens=max_tokens, json_mode=True)
        raise RuntimeError("unreachable")


def _extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in model output")
    return match.group(0)
