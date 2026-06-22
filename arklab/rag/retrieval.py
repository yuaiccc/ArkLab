from __future__ import annotations

import math
from collections import Counter, defaultdict

from arklab.models import DocumentChunk, RetrievalHit
from arklab.text import cosine, hashed_vector, tokenize


class BM25Retriever:
    def __init__(self, chunks: list[DocumentChunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.text) for chunk in chunks]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.doc_freq: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.doc_freq[token] += 1

    def search(self, query: str, *, limit: int) -> list[tuple[int, float]]:
        query_tokens = tokenize(query)
        scores: list[tuple[int, float]] = []
        total_docs = max(1, len(self.chunks))
        for doc_index, tokens in enumerate(self.doc_tokens):
            counts = Counter(tokens)
            score = 0.0
            doc_len = self.doc_lengths[doc_index] or 1
            for token in query_tokens:
                freq = counts[token]
                if freq == 0:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                score += idf * (freq * (self.k1 + 1)) / denom
            if score > 0:
                scores.append((doc_index, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:limit]


class DenseHashRetriever:
    def __init__(self, chunks: list[DocumentChunk], *, dims: int = 256) -> None:
        self.chunks = chunks
        self.dims = dims
        self.vectors = [hashed_vector(tokenize(chunk.text), dims=dims) for chunk in chunks]

    def search(self, query: str, *, limit: int) -> list[tuple[int, float]]:
        query_vector = hashed_vector(tokenize(query), dims=self.dims)
        scores = [
            (index, cosine(query_vector, vector))
            for index, vector in enumerate(self.vectors)
        ]
        return [
            item
            for item in sorted(scores, key=lambda item: item[1], reverse=True)
            if item[1] > 0
        ][:limit]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, chunks: list[DocumentChunk], *, rrf_k: int = 60) -> None:
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.bm25 = BM25Retriever(chunks)
        self.dense = DenseHashRetriever(chunks)

    def search(self, query: str, *, limit: int = 5, candidate_limit: int = 30) -> list[RetrievalHit]:
        bm25_hits = self.bm25.search(query, limit=candidate_limit)
        dense_hits = self.dense.search(query, limit=candidate_limit)
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
