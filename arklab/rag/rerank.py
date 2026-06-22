from __future__ import annotations

from arklab.models import RetrievalHit
from arklab.text import tokenize


class LexicalReranker:
    name = "lexical"

    def rerank(self, query: str, hits: list[RetrievalHit], *, limit: int) -> list[RetrievalHit]:
        query_tokens = set(tokenize(query))
        scored: list[tuple[float, RetrievalHit]] = []
        for hit in hits:
            chunk_tokens = set(tokenize(hit.chunk.text))
            overlap = len(query_tokens & chunk_tokens)
            coverage = overlap / max(1, len(query_tokens))
            density = overlap / max(1, len(chunk_tokens))
            rerank_score = coverage * 0.7 + density * 0.2 + hit.score * 0.1
            scored.append((rerank_score, hit))

        ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
        return [
            RetrievalHit(
                chunk=hit.chunk,
                score=hit.score,
                rank=rank,
                bm25_rank=hit.bm25_rank,
                dense_rank=hit.dense_rank,
                rerank_score=score,
            )
            for rank, (score, hit) in enumerate(ranked, start=1)
        ]


class NoopReranker:
    name = "none"

    def rerank(self, query: str, hits: list[RetrievalHit], *, limit: int) -> list[RetrievalHit]:
        del query
        return [
            RetrievalHit(
                chunk=hit.chunk,
                score=hit.score,
                rank=rank,
                bm25_rank=hit.bm25_rank,
                dense_rank=hit.dense_rank,
                rerank_score=hit.rerank_score,
            )
            for rank, hit in enumerate(hits[:limit], start=1)
        ]


def create_reranker(name: str) -> LexicalReranker | NoopReranker:
    if name == "none":
        return NoopReranker()
    if name == "lexical":
        return LexicalReranker()
    raise ValueError(f"unknown reranker: {name}")
