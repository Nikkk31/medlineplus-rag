from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document

import config
from retrieval.indexing import Indexes, build_or_load_indexes

log = logging.getLogger(__name__)


@dataclass
class ScoredChunk:
    document: Document
    fused_score: float
    dense_rank: int | None
    bm25_rank: int | None
    dense_distance: float | None  # raw FAISS L2 distance, lower = more similar


def _chunk_key(doc: Document) -> str:
    # file_path + first 80 chars is enough to dedupe the same chunk across lists
    return f"{doc.metadata.get('file_path', '')}::{doc.page_content[:80]}"


class HybridRetriever:
    """Dense + BM25 retrieval fused with Reciprocal Rank Fusion."""

    def __init__(self, indexes: Indexes | None = None):
        self.indexes = indexes or build_or_load_indexes()

    # ── individual retrievers, exposed separately for eval/comparison ──

    def dense_search(self, query: str, k: int = config.DENSE_CANDIDATE_K) -> list[tuple[Document, float]]:
        return self.indexes.vectorstore.similarity_search_with_score(query, k=k)

    def bm25_search(self, query: str, k: int = config.BM25_CANDIDATE_K) -> list[Document]:
        self.indexes.bm25.k = k
        return self.indexes.bm25.invoke(query)

    # ── fused search ──

    def search(self, query: str, top_k: int = config.TOP_K, rrf_k: int = config.RRF_K) -> list[ScoredChunk]:
        dense_results = self.dense_search(query)
        bm25_results = self.bm25_search(query)

        dense_rank = {_chunk_key(doc): (i, doc, dist) for i, (doc, dist) in enumerate(dense_results)}
        bm25_rank = {_chunk_key(doc): (i, doc) for i, doc in enumerate(bm25_results)}

        fused: dict[str, ScoredChunk] = {}
        all_keys = set(dense_rank) | set(bm25_rank)
        for key in all_keys:
            d_entry = dense_rank.get(key)
            b_entry = bm25_rank.get(key)
            doc = (d_entry[1] if d_entry else b_entry[1])

            score = 0.0
            d_rank = b_rank = None
            d_dist = None
            if d_entry:
                d_rank, _, d_dist = d_entry
                score += 1.0 / (rrf_k + d_rank + 1)
            if b_entry:
                b_rank, _ = b_entry
                score += 1.0 / (rrf_k + b_rank + 1)

            fused[key] = ScoredChunk(
                document=doc, fused_score=score,
                dense_rank=d_rank, bm25_rank=b_rank, dense_distance=d_dist,
            )

        ranked = sorted(fused.values(), key=lambda sc: sc.fused_score, reverse=True)
        return ranked[:top_k]

    # ── in-scope gate (dense distance is the calibrated signal here, per
    #    the notebook's original threshold sweep) ──

    def is_in_scope(self, query: str, threshold: float = config.SIMILARITY_THRESHOLD) -> tuple[bool, float | None]:
        results = self.dense_search(query, k=1)
        if not results:
            return False, None
        top_score = float(results[0][1])
        return top_score <= threshold, top_score