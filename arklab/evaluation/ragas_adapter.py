from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arklab.embeddings.ark import ArkEmbeddingClient


@dataclass(frozen=True)
class RagasCase:
    query: str
    answer: str
    contexts: list[str]
    reference: str | None = None


class ArkLangchainEmbeddings:
    def __init__(self, client: ArkEmbeddingClient) -> None:
        self.client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.client.embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed_text(text)


def evaluate_with_ragas(
    cases: list[RagasCase],
    *,
    judge_model: str,
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
) -> dict[str, Any]:
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS is not installed. Run `python3.11 -m venv .venv && "
            ". .venv/bin/activate && pip install -e . 'ragas==0.2.15' "
            "'langchain<0.4' 'langchain-core<0.4' 'langchain-community<0.4' "
            "'langchain-openai<0.4'`."
        ) from exc

    rows: list[dict[str, Any]] = []
    has_reference = all(case.reference for case in cases)
    for case in cases:
        row = {
            "user_input": case.query,
            "response": case.answer,
            "retrieved_contexts": case.contexts,
        }
        if has_reference:
            row["reference"] = case.reference
        rows.append(row)

    metrics = [faithfulness, answer_relevancy]
    if has_reference:
        metrics.extend([context_precision, context_recall])

    llm = ChatOpenAI(
        api_key=ArkEmbeddingClient._load_api_key(),
        base_url=base_url,
        model=judge_model,
        temperature=0,
        max_retries=1,
        timeout=60,
        openai_proxy="",
        extra_body={"thinking": {"type": "disabled"}},
    )
    embeddings = ArkLangchainEmbeddings(ArkEmbeddingClient())
    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        show_progress=False,
    )
    frame = result.to_pandas()
    records = frame.to_dict(orient="records")

    metric_names = [metric.name for metric in metrics]
    summary: dict[str, float] = {}
    for name in metric_names:
        values = [
            float(record[name])
            for record in records
            if name in record and record[name] is not None and str(record[name]) != "nan"
        ]
        if values:
            summary[name] = sum(values) / len(values)

    return {
        "summary": summary,
        "cases": records,
        "judge_model": judge_model,
        "metrics": metric_names,
    }
