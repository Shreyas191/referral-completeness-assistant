"""Case-scoped hybrid retrieval: BM25 + dense (ChromaDB), fused with reciprocal rank fusion.

Evidence is always filtered to one case_id, so another patient's record can never be cited.
A global index exists only for ablation E2 (scope="global").
"""
import re
from typing import Literal

from rank_bm25 import BM25Okapi

from referral.schemas import ChecklistItem, Chunk


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def item_query(item: ChecklistItem) -> str:
    return " ".join([item.description, *item.synonyms])


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([_tokens(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int) -> list[Chunk]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokens(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:k] if scores[i] > 0]


class DenseRetriever:
    """ChromaDB with its default embedding model. Imported lazily so tests run without it."""

    def __init__(self, persist_dir: str, collection: str = "chunks"):
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(collection)
        self._by_id: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._col.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"case_id": c.case_id, "doc_id": c.doc_id, "page": c.page} for c in chunks],
        )
        self._by_id.update({c.chunk_id: c for c in chunks})

    def search(self, query: str, k: int, case_id: str | None) -> list[Chunk]:
        where = {"case_id": case_id} if case_id else None
        res = self._col.query(query_texts=[query], n_results=k, where=where)
        return [self._by_id[i] for i in res["ids"][0] if i in self._by_id]


def reciprocal_rank_fusion(rankings: list[list[Chunk]], k: int, rrf_k: int = 60) -> list[Chunk]:
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            by_id[chunk.chunk_id] = chunk
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [by_id[i] for i in best]


class CaseRetriever:
    """What the pipeline calls. mode picks the E3 variant; scope picks the E2 variant."""

    def __init__(
        self,
        case_chunks: list[Chunk],
        dense: DenseRetriever | None = None,
        mode: Literal["bm25", "dense", "hybrid"] = "hybrid",
        scope: Literal["case", "global"] = "case",
        global_chunks: list[Chunk] | None = None,
    ):
        self.case_id = case_chunks[0].case_id if case_chunks else None
        pool = global_chunks if scope == "global" and global_chunks else case_chunks
        self.bm25 = BM25Retriever(pool)
        self.dense = dense
        self.mode = mode if dense else "bm25"
        self.scope = scope

    def search(self, item: ChecklistItem, k: int) -> list[Chunk]:
        query = item_query(item)
        case_filter = self.case_id if self.scope == "case" else None
        if self.mode == "bm25":
            return self.bm25.search(query, k)
        dense_hits = self.dense.search(query, k, case_filter)
        if self.mode == "dense":
            return dense_hits
        return reciprocal_rank_fusion([self.bm25.search(query, k), dense_hits], k)
