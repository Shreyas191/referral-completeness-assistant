"""B0, lower bound: an item is Present if any synonym appears anywhere in the packet."""
from referral.ingest.chunker import chunk_case
from referral.pipeline.absence import synonym_scan
from referral.schemas import CaseResult, Checklist, Citation, Document, ItemResult, Status


def run_b0(docs: list[Document], checklist: Checklist) -> CaseResult:
    chunks = chunk_case(docs)
    results = []
    for item in checklist.items:
        hits = synonym_scan(item, chunks)
        results.append(
            ItemResult(
                item_id=item.item_id,
                status=Status.PRESENT if hits else Status.MISSING,
                citations=[Citation(chunk_id=c.chunk_id, doc_id=c.doc_id, page=c.page,
                                    quote=c.text[:120], doc_date=c.doc_date) for c in hits[:1]],
            )
        )
    return CaseResult(case_id=docs[0].case_id, checklist_id=checklist.checklist_id, items=results)
