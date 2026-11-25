"""Per-item record of what the pipeline looked at, for trajectory metrics (#3).

Saved next to each prediction as <case_id>.trace.json. With these, recall@k and error
attribution (retrieval miss vs validator reject vs reasoning error) can be computed after a run.
"""
from pydantic import BaseModel


class ItemTrace(BaseModel):
    item_id: str
    ranked: list[str] = []          # first-pass retrieval, best first, up to TRACE_DEPTH chunk ids
    shown: list[str] = []           # chunk ids actually given to the verifier on the first pass
    expansion: list[str] = []       # extra chunk ids added by value-term expansion, shown with `shown`
    first_status: str = ""          # verifier's first proposal
    absence_extra: list[str] = []   # extra chunk ids shown by the absence re-check
    absence_status: str | None = None
    long_context_status: str | None = None
    final_status: str = ""
    # Quotes the validator rejected: {"chunk_id", "quote", "score"}, score = match against the cited chunk
    # (0-100, accepted at settings.quote_match_threshold), or None if the chunk id does not exist.
    rejected_quotes: list[dict] = []


class CaseTrace(BaseModel):
    case_id: str
    config: dict = {}
    quarantined_docs: list[str] = []
    items: list[ItemTrace] = []

    def item(self, item_id: str) -> ItemTrace:
        return next(t for t in self.items if t.item_id == item_id)


TRACE_DEPTH = 10  # deep enough for recall@3/5/10
