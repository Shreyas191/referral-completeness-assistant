"""The LLM wrapper's parsing, repair retry and both providers, with every API call stubbed out."""
from types import SimpleNamespace as NS

import pytest
from pydantic import ValidationError

from referral.llm import LLM
from referral.pipeline.verifier import VerifierOutput

GOOD = '{"item_id": "echo", "status": "present", "evidence": [], "rationale": "ok"}'


class FakeAnthropic:
    def __init__(self):
        self.sent = []
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.sent.append(kw)
        return NS(content=[NS(type="text", text=GOOD)], usage=NS(input_tokens=11, output_tokens=7))


class FakeOpenAI:
    def __init__(self):
        self.sent = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.sent.append(kw)
        return NS(choices=[NS(message=NS(content=GOOD))], usage=NS(prompt_tokens=13, completion_tokens=5))


@pytest.mark.parametrize("provider, fake, model", [("anthropic", FakeAnthropic, "claude-haiku-5-5"),
                                                   ("openai", FakeOpenAI, "gpt-6-luna")])
def test_both_providers_return_validated_output(monkeypatch, provider, fake, model):
    client = fake()
    monkeypatch.setattr(LLM, "_make_client", lambda self: client)
    llm = LLM(provider=provider, model=model)
    assert llm.structured("sys", "user", VerifierOutput).status == "present"
    assert llm.name == f"{provider}:{model}" and llm.calls == 1 and llm.input_tokens > 0
    sent = client.sent[0]
    assert sent["model"] == model
    if provider == "openai":
        assert sent["messages"][0] == {"role": "system", "content": "sys"}
        assert sent["response_format"] == {"type": "json_object"}
    else:
        assert sent["system"] == "sys"


def test_switching_provider_in_code_uses_that_providers_default_model(monkeypatch):
    monkeypatch.setattr(LLM, "_make_client", lambda self: None)
    other = "openai" if LLM(provider="anthropic").provider == "anthropic" else "anthropic"
    assert LLM(provider=other).model == {"openai": "gpt-6-luna", "anthropic": "claude-haiku-5-5"}[other]


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        LLM(provider="gemini", model="x")


def stub(replies):
    llm = object.__new__(LLM)  # skip __init__, which needs a real client and a key
    llm.prompts = []
    queue = list(replies)

    def complete(system, user, max_tokens=4000, json_mode=False):
        llm.prompts.append(user)
        return queue.pop(0)

    llm.complete = complete
    return llm


def test_parses_json_wrapped_in_prose():
    llm = stub([f"Here is my answer:\n```json\n{GOOD}\n```"])
    assert llm.structured("sys", "user", VerifierOutput).status == "present"
    assert "Return only JSON matching this schema" in llm.prompts[0]


def test_repairs_invalid_output_once():
    llm = stub(['{"item_id": "echo", "status": "maybe"}', GOOD])
    assert llm.structured("sys", "user", VerifierOutput).status == "present"
    assert "was invalid" in llm.prompts[1]


def test_gives_up_after_the_retry():
    llm = stub(["not json", "still not json"])
    with pytest.raises((ValidationError, ValueError)):
        llm.structured("sys", "user", VerifierOutput)


@pytest.mark.parametrize("provider,cls", [("openai", "OpenAI"), ("anthropic", "Anthropic")])
def test_clients_have_a_request_timeout(monkeypatch, provider, cls):
    # Regression: with no timeout, one stalled response blocked a dev run for ~50 minutes (pw-dev-r2).
    # The SDK is stubbed, so this runs without it installed (CI installs neither) and without a key.
    import sys
    import types

    from referral.config import settings

    sdk = types.ModuleType(provider)
    setattr(sdk, cls, lambda **kw: NS(**kw))
    monkeypatch.setitem(sys.modules, provider, sdk)
    client = LLM(provider=provider)._client
    assert client.timeout == settings.llm_timeout_s == 120
    assert client.max_retries == settings.llm_max_retries == 2


@pytest.mark.parametrize("provider, fake, key", [("anthropic", FakeAnthropic, "max_tokens"),
                                                 ("openai", FakeOpenAI, "max_completion_tokens")])
def test_structured_passes_max_tokens_and_defaults_to_4000(monkeypatch, provider, fake, key):
    # The hard-profile generator asks for 20-section documents, which do not fit in 4000 output tokens.
    client = fake()
    monkeypatch.setattr(LLM, "_make_client", lambda self: client)
    llm = LLM(provider=provider)
    llm.structured("sys", "user", VerifierOutput)
    llm.structured("sys", "user", VerifierOutput, max_tokens=14000)
    assert [s[key] for s in client.sent] == [4000, 14000]
