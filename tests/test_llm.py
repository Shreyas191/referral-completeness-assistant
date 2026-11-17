"""The LLM wrapper's parsing and repair retry, with the API call stubbed out."""
import pytest
from pydantic import ValidationError

from referral.llm import LLM
from referral.pipeline.verifier import VerifierOutput

GOOD = '{"item_id": "echo", "status": "present", "evidence": [], "rationale": "ok"}'


def stub(replies):
    llm = object.__new__(LLM)  # skip __init__, which needs the anthropic client and a key
    llm.prompts = []
    queue = list(replies)

    def complete(system, user, max_tokens=4000):
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
