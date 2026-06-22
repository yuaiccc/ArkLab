from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    chunk: DocumentChunk
    score: float
    rank: int
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class ProviderResult:
    content: str
    model: str = "local"
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RagResult:
    query: str
    answer: str
    abstained: bool
    abstain_reason: str | None
    hits: list[RetrievalHit]
    metrics: dict[str, float]
    provider: ProviderResult
