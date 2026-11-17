"""Offline stand-in for referral.llm.LLM, so the whole pipeline can be tested without an API key.

It behaves like a careful reader with no real understanding: an item is present if one of its terms
appears in a passage (quoting that line verbatim), and conditional items are not applicable when the
condition's key word appears nowhere. Knobs let tests simulate the verifier missing evidence.
"""
import re

from referral.pipeline.absence import LongContextOutput
from referral.pipeline.verifier import Evidence, VerifierOutput

PASSAGE_RE = re.compile(r"\[chunk_id=(?P<id>[^ |\]]+)[^\]]*\]\n(?P<text>.*?)(?=\n\n\[chunk_id=|\Z)", re.S)


class FakeLLM:
    def __init__(self, miss_first: set[str] = frozenset(), miss_per_item: set[str] = frozenset()):
        self.miss_first = set(miss_first)        # item ids missed on the first per-item look only
        self.miss_per_item = set(miss_per_item)  # item ids missed on every per-item look (found only by final read)
        self.seen: dict[str, int] = {}
        self.calls = self.input_tokens = self.output_tokens = 0
        self.long_context_calls = 0

    def structured(self, system, user, schema):
        self.calls += 1
        passages = [(m["id"], m["text"]) for m in PASSAGE_RE.finditer(user)]
        if schema is VerifierOutput:
            item_id = re.search(r"Checklist item (\S+):", user)[1]
            description = re.search(r"Checklist item \S+: (.*)", user)[1].split("\nCondition:")[0]
            synonyms = re.search(r"Also known as: (.*)", user)[1]
            condition = re.search(r"Condition: required only if (.*)\.", user)
            self.seen[item_id] = self.seen.get(item_id, 0) + 1
            if item_id in self.miss_per_item or (item_id in self.miss_first and self.seen[item_id] == 1):
                return VerifierOutput(item_id=item_id, status="missing", rationale="fake: not found")
            terms = [description, *[s.strip() for s in synonyms.split(",") if s.strip() and s.strip() != "n/a"]]
            return self._judge(item_id, terms, condition[1] if condition else None, passages)
        if schema is LongContextOutput:
            self.long_context_calls += 1
            out = []
            for line in re.findall(r"^- (\S+): (.*)$", user, re.M):
                item_id, rest = line
                description = rest.split(" (")[0]
                also = re.search(r"\(also: ([^)]*)\)", rest)
                condition = re.search(r"\(required only if ([^)]*)\)", rest)
                terms = [description, *([s.strip() for s in also[1].split(",")] if also else [])]
                out.append(self._judge(item_id, terms, condition[1] if condition else None, passages))
            return LongContextOutput(items=out)
        raise NotImplementedError(schema)

    @staticmethod
    def _judge(item_id, terms, condition, passages):
        if condition:
            key = condition.split()[-1].lower()
            if not any(key in text.lower() for _, text in passages):
                return VerifierOutput(item_id=item_id, status="not_applicable", rationale="fake: condition absent")
        evidence = []
        for chunk_id, text in passages:
            for line in text.splitlines():
                if any(re.search(rf"\b{re.escape(t.lower())}\b", line.lower()) for t in terms if t):
                    evidence.append(Evidence(chunk_id=chunk_id, quote=line.strip()))
                    break
        status = "present" if evidence else "missing"
        return VerifierOutput(item_id=item_id, status=status, evidence=evidence, rationale="fake")
