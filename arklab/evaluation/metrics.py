from __future__ import annotations

import math

from arklab.models import RetrievalHit
from arklab.text import tokenize


def recall_at_k(hits: list[RetrievalHit], relevant_ids: set[str], *, k: int) -> float:
    if not relevant_ids:
        return 0.0
    found = {
        hit.chunk.id
        for hit in hits[:k]
        if hit.chunk.id in relevant_ids or hit.chunk.source in relevant_ids
    }
    return len(found) / len(relevant_ids)


def mrr(hits: list[RetrievalHit], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk.id in relevant_ids or hit.chunk.source in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(hits: list[RetrievalHit], relevance: dict[str, float], *, k: int) -> float:
    if not relevance:
        return 0.0

    def gain(hit: RetrievalHit) -> float:
        return relevance.get(hit.chunk.id, relevance.get(hit.chunk.source, 0.0))

    dcg = 0.0
    for index, hit in enumerate(hits[:k], start=1):
        rel = gain(hit)
        dcg += (2**rel - 1) / math.log2(index + 1)

    ideal_relevance = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(index + 1) for index, rel in enumerate(ideal_relevance, start=1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def lexical_faithfulness(answer: str, contexts: list[str]) -> float:
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0
    context_tokens = set(tokenize("\n".join(contexts)))
    if not context_tokens:
        return 0.0
    supported = answer_tokens & context_tokens
    return len(supported) / len(answer_tokens)


def answer_relevancy(answer: str, query: str) -> float:
    answer_tokens = set(tokenize(answer))
    query_tokens = set(tokenize(query))
    if not answer_tokens or not query_tokens:
        return 0.0
    return len(answer_tokens & query_tokens) / len(query_tokens)
