from __future__ import annotations

from collections import defaultdict

from arklab.embeddings.ark import ArkEmbeddingClient
from arklab.models import DocumentChunk, RetrievalHit
from arklab.text import cosine
from arklab.rag.retrieval import BM25Retriever


class ArkEmbeddingRetriever:
    name = "ark-embedding"

    def __init__(
        self,
        chunks: list[DocumentChunk],
        *,
        client: ArkEmbeddingClient,
        embed_workers: int = 1,
    ) -> None:
        self.chunks = chunks
        self.client = client
        self.vectors = self.client.embed_texts(
            [chunk.text for chunk in chunks],
            max_workers=embed_workers,
        )
        self.cache_stats = self.client.cache_stats()

    def search(self, query: str, *, limit: int = 5, candidate_limit: int = 30) -> list[RetrievalHit]:
        del candidate_limit
        query_vector = self.client.embed_text(query)
        self.cache_stats = self.client.cache_stats()
        scores = [
            (index, cosine(query_vector, vector))
            for index, vector in enumerate(self.vectors)
        ]
        ranked = sorted(scores, key=lambda item: item[1], reverse=True)[:limit]
        return [
            RetrievalHit(
                chunk=self.chunks[index],
                score=score,
                rank=rank,
                bm25_rank=None,
                dense_rank=rank,
            )
            for rank, (index, score) in enumerate(ranked, start=1)
        ]


class ArkHybridRetriever:
    name = "ark-hybrid"

    def __init__(
        self,
        chunks: list[DocumentChunk],
        *,
        client: ArkEmbeddingClient,
        rrf_k: int = 60,
        embed_workers: int = 1,
    ) -> None:
        self.chunks = chunks
        self.client = client
        self.rrf_k = rrf_k
        self.bm25 = BM25Retriever(chunks)
        self.vectors = self.client.embed_texts(
            [chunk.text for chunk in chunks],
            max_workers=embed_workers,
        )
        self.cache_stats = self.client.cache_stats()

    def _dense_search(self, query: str, *, limit: int) -> list[tuple[int, float]]:
        query_vector = self.client.embed_text(query)
        self.cache_stats = self.client.cache_stats()
        scores = [
            (index, cosine(query_vector, vector))
            for index, vector in enumerate(self.vectors)
        ]
        return sorted(scores, key=lambda item: item[1], reverse=True)[:limit]

    def search(self, query: str, *, limit: int = 5, candidate_limit: int = 30) -> list[RetrievalHit]:
        bm25_hits = self.bm25.search(query, limit=candidate_limit)
        dense_hits = self._dense_search(query, limit=candidate_limit)
        scores: dict[int, float] = defaultdict(float)
        bm25_ranks: dict[int, int] = {}
        dense_ranks: dict[int, int] = {}

        for rank, (index, _score) in enumerate(bm25_hits, start=1):
            scores[index] += 1.0 / (self.rrf_k + rank)
            bm25_ranks[index] = rank
        for rank, (index, _score) in enumerate(dense_hits, start=1):
            scores[index] += 1.0 / (self.rrf_k + rank)
            dense_ranks[index] = rank

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [
            RetrievalHit(
                chunk=self.chunks[index],
                score=score,
                rank=rank,
                bm25_rank=bm25_ranks.get(index),
                dense_rank=dense_ranks.get(index),
            )
            for rank, (index, score) in enumerate(ranked, start=1)
        ]
