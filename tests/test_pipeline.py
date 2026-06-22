from arklab.models import DocumentChunk
from arklab.providers.local import LocalHeuristicProvider
from arklab.rag.pipeline import RagPipeline


def test_pipeline_answers_when_retrieval_and_faithfulness_are_supported() -> None:
    chunks = [
        DocumentChunk(
            id="doc.md#0",
            source="doc.md",
            text="ArkLab abstain 机制在检索置信度低或答案缺少上下文支撑时触发。",
        )
    ]
    pipeline = RagPipeline(
        chunks=chunks,
        provider=LocalHeuristicProvider(),
        min_answer_relevancy=0.2,
    )

    result = pipeline.run("ArkLab abstain 机制什么时候触发？")

    assert result.abstained is False
    assert "检索置信度低" in result.answer
    assert result.metrics["faithfulness"] > 0.5


def test_pipeline_abstains_when_retrieval_has_no_hits() -> None:
    chunks = [
        DocumentChunk(
            id="doc.md#0",
            source="doc.md",
            text="ArkLab 评测 RAG 系统的召回率和忠实度。",
        )
    ]
    pipeline = RagPipeline(
        chunks=chunks,
        provider=LocalHeuristicProvider(),
        min_answer_relevancy=0.2,
    )

    result = pipeline.run("火星天气今天多少度？")

    assert result.abstained is True
    assert result.abstain_reason in {"low_retrieval_confidence", "low_answer_relevancy"}
    assert result.answer == "无法基于当前知识库回答这个问题。"
