from __future__ import annotations

from typing import Protocol

from arklab.evaluation.metrics import answer_relevancy, lexical_faithfulness
from arklab.models import DocumentChunk, ProviderResult, RagResult
from arklab.providers.base import ModelProvider
from arklab.rag.retrieval import HybridRetriever
from arklab.rag.rerank import NoopReranker
from arklab.trace.writer import TraceWriter


REFUSAL_MARKERS = (
    "无法基于当前知识库回答",
    "无法基于当前上下文回答",
    "上下文不足",
    "无法回答",
)

CONTENT_BLOCK_MARKERS = (
    "content_moderation",
    "SensitiveContentDetected",
    "内容安全",
    "合规系统拦截",
)


class Retriever(Protocol):
    name: str

    def search(self, query: str, *, limit: int = 5, candidate_limit: int = 30):
        ...


class RagPipeline:
    def __init__(
        self,
        *,
        chunks: list[DocumentChunk],
        provider: ModelProvider,
        top_k: int = 5,
        candidate_k: int = 20,
        reranker: NoopReranker | None = None,
        retriever: Retriever | None = None,
        min_retrieval_score: float = 0.01,
        min_faithfulness: float = 0.35,
        min_answer_relevancy: float = 0.0,
        trace_writer: TraceWriter | None = None,
    ) -> None:
        self.chunks = chunks
        self.provider = provider
        self.top_k = top_k
        self.candidate_k = max(top_k, candidate_k)
        self.reranker = reranker or NoopReranker()
        self.min_retrieval_score = min_retrieval_score
        self.min_faithfulness = min_faithfulness
        self.min_answer_relevancy = min_answer_relevancy
        self.trace_writer = trace_writer
        self.retriever = retriever or HybridRetriever(chunks)

    def run(self, query: str) -> RagResult:
        candidates = self.retriever.search(query, limit=self.candidate_k)
        hits = self.reranker.rerank(query, candidates, limit=self.top_k)
        contexts = [hit.chunk.text for hit in hits]
        top_score = hits[0].score if hits else 0.0

        retrieval_abstain = not hits or top_score < self.min_retrieval_score
        if retrieval_abstain:
            provider_result = self._safe_provider_answer(query=query, hits=[])
            answer = "无法基于当前知识库回答这个问题。"
            faithfulness = 0.0
            relevancy = 0.0
            abstained = True
            abstain_reason = (
                "provider_content_block"
                if provider_result.raw.get("error_type") == "provider_content_block"
                else "low_retrieval_confidence"
            )
        else:
            provider_result = self._safe_provider_answer(query=query, hits=hits)
            answer = provider_result.content
            faithfulness = self.provider.judge_faithfulness(answer=answer, contexts=contexts)
            if faithfulness == 0.0:
                faithfulness = lexical_faithfulness(answer, contexts)
            relevancy = answer_relevancy(answer, query)
            provider_refused = any(marker in answer for marker in REFUSAL_MARKERS)
            provider_blocked = provider_result.raw.get("error_type") == "provider_content_block"
            low_relevancy = relevancy < self.min_answer_relevancy
            abstained = (
                provider_blocked
                or provider_refused
                or faithfulness < self.min_faithfulness
                or low_relevancy
            )
            if provider_blocked:
                abstain_reason = "provider_content_block"
            elif provider_refused:
                abstain_reason = "provider_abstain"
            elif abstained:
                abstain_reason = "low_answer_relevancy" if low_relevancy else "low_faithfulness"
            else:
                abstain_reason = None
            if abstained:
                answer = "无法基于当前知识库回答这个问题。"

        metrics = {
            "top_retrieval_score": top_score,
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "abstain": 1.0 if abstained else 0.0,
        }
        result = RagResult(
            query=query,
            answer=answer,
            abstained=abstained,
            abstain_reason=abstain_reason,
            hits=hits,
            metrics=metrics,
            provider=provider_result,
        )
        if self.trace_writer:
            self.trace_writer.write(
                {
                    "event": "rag_query",
                    "provider": provider_result.model,
                    "retriever": self.retriever.name,
                    "reranker": self.reranker.name,
                    "query": query,
                    "answer": answer,
                    "abstained": abstained,
                    "abstain_reason": abstain_reason,
                    "metrics": metrics,
                    "hits": [
                        {
                            "id": hit.chunk.id,
                            "source": hit.chunk.source,
                            "score": hit.score,
                            "rank": hit.rank,
                            "bm25_rank": hit.bm25_rank,
                            "dense_rank": hit.dense_rank,
                            "rerank_score": hit.rerank_score,
                        }
                        for hit in hits
                    ],
                }
            )
        return result

    def _safe_provider_answer(self, *, query: str, hits) -> ProviderResult:
        try:
            return self.provider.answer(query=query, hits=hits)
        except Exception as exc:
            message = str(exc)
            if any(marker in message for marker in CONTENT_BLOCK_MARKERS):
                return ProviderResult(
                    content="无法基于当前知识库回答这个问题。",
                    model=getattr(self.provider, "model", None) or getattr(self.provider, "name", "provider"),
                    usage={},
                    raw={
                        "error_type": "provider_content_block",
                        "error": message,
                    },
                )
            raise
