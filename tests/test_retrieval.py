"""CaseRetriever modes (E3) and scoping (E2), with a fake dense index so chromadb is not needed."""
from conftest import make_chunk

from referral.retrieval.hybrid import CaseRetriever, reciprocal_rank_fusion


class FakeDense:
    """Stands in for DenseRetriever: returns a fixed ranking, filtered to case_id like the real where clause."""

    def __init__(self, ranking):
        self.ranking = ranking
        self.case_filters = []

    def search(self, query, k, case_id):
        self.case_filters.append(case_id)
        return [c for c in self.ranking if case_id is None or c.case_id == case_id][:k]


LETTER = make_chunk("Most recent potassium is 5.9 mmol/L.", doc_id="letter")
LABS = make_chunk("BMP: creatinine 1.1 mg/dL, potassium 4.2", doc_id="labs")
NOTE = make_chunk("Patient denies chest pain.", doc_id="note")
OTHER = make_chunk("BMP: creatinine 2.0 mg/dL", doc_id="labs", case_id="c2")


def test_hybrid_recovers_a_chunk_bm25_scores_zero(bmp_item):
    # dev-009: the letter shares no query term with the item, so BM25 drops it; the dense ranking still has it.
    chunks = [LETTER, LABS, NOTE]
    bm25 = CaseRetriever(chunks, mode="bm25")
    assert LETTER not in bm25.search(bmp_item, 5)
    hybrid = CaseRetriever(chunks, dense=FakeDense([LETTER, LABS, NOTE]), mode="hybrid")
    found = hybrid.search(bmp_item, 5)
    assert found[0] == LABS and LETTER in found     # in both rankings beats in one


def test_dense_mode_returns_the_dense_ranking_filtered_to_the_case(bmp_item):
    dense = FakeDense([OTHER, NOTE, LABS])
    found = CaseRetriever([LABS, NOTE], dense=dense, mode="dense").search(bmp_item, 5)
    assert found == [NOTE, LABS] and dense.case_filters == ["c1"]


def test_global_scope_queries_dense_without_a_case_filter(bmp_item):
    dense = FakeDense([OTHER, LABS])
    found = CaseRetriever([LABS], dense=dense, mode="dense", scope="global", global_chunks=[LABS, OTHER])
    assert found.search(bmp_item, 5) == [OTHER, LABS] and dense.case_filters == [None]


def test_without_a_dense_index_every_mode_falls_back_to_bm25(bmp_item):
    # Three chunks: with two, a term in one of them gets BM25 IDF log(1) = 0 and nothing is returned.
    r = CaseRetriever([LABS, NOTE, LETTER], dense=None, mode="hybrid")
    assert r.mode == "bm25" and r.search(bmp_item, 5) == [LABS]


def test_rrf_ranks_by_summed_reciprocal_rank():
    a, b, c = LETTER, LABS, NOTE
    assert reciprocal_rank_fusion([[a, b, c], [b, c, a]], k=3) == [b, a, c]
    assert reciprocal_rank_fusion([[a, b, c], [b, c, a]], k=1) == [b]
