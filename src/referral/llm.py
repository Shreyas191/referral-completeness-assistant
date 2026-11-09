"""Thin LLM wrapper: JSON in, validated Pydantic out, with one repair retry."""
import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from referral.config import settings

T = TypeVar("T", bound=BaseModel)


class LLM:
    def __init__(self, model: str | None = None):
        import anthropic

        self.model = model or settings.llm_model
        self._client = anthropic.Anthropic()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(self, system: str, user: str, max_tokens: int = 4000) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        self.calls += 1
        self.input_tokens += msg.usage.input_tokens
        self.output_tokens += msg.usage.output_tokens
        return "".join(block.text for block in msg.content if block.type == "text")

    def structured(self, system: str, user: str, schema: type[T], retries: int = 1) -> T:
        prompt = f"{user}\n\nReturn only JSON matching this schema:\n{json.dumps(schema.model_json_schema())}"
        raw = self.complete(system, prompt)
        for attempt in range(retries + 1):
            try:
                return schema.model_validate_json(_extract_json(raw))
            except (ValidationError, ValueError) as err:
                if attempt == retries:
                    raise
                raw = self.complete(system, f"{prompt}\n\nYour last answer was invalid ({err}). Fix it.")
        raise RuntimeError("unreachable")


def _extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in model output")
    return match.group(0)
