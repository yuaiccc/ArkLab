import json
from pathlib import Path

from arklab.flywheel import compare_reports, promote_failures_to_eval_set, write_jsonl


def test_promote_failures_dedupes_and_restores_reference(tmp_path: Path) -> None:
    failure_pool = tmp_path / "failure_pool.jsonl"
    source_eval = tmp_path / "eval.jsonl"
    write_jsonl(
        failure_pool,
        [
            {
                "event": "failure_case",
                "ts": "2026-01-01T00:00:00Z",
                "failure_type": "generation_fail_abstain",
                "query": "ArkLab 修什么问题？",
                "answer": "无法基于当前知识库回答这个问题。",
                "abstained": True,
                "metrics": {"faithfulness": 0.0},
            },
            {
                "event": "failure_case",
                "ts": "2026-01-01T00:01:00Z",
                "failure_type": "generation_fail_abstain",
                "query": "ArkLab 修什么问题？",
                "answer": "无法基于当前知识库回答这个问题。",
                "abstained": True,
                "metrics": {"faithfulness": 0.0},
            },
        ],
    )
    write_jsonl(
        source_eval,
        [
            {
                "query": "ArkLab 修什么问题？",
                "answer": "ArkLab 用于诊断 RAG 失败。",
                "relevant_ids": ["doc.md#0"],
            }
        ],
    )

    rows, manifest = promote_failures_to_eval_set(
        failure_pool_path=failure_pool,
        source_eval_set_path=source_eval,
    )

    assert manifest["raw_failures"] == 2
    assert manifest["deduped_duplicates"] == 1
    assert rows == [
        {
            "query": "ArkLab 修什么问题？",
            "relevant_ids": ["doc.md#0"],
            "arklab_flywheel": rows[0]["arklab_flywheel"],
            "answer": "ArkLab 用于诊断 RAG 失败。",
        }
    ]
    assert rows[0]["arklab_flywheel"]["suggested_action"] == "prompt_or_context_answerability"


def test_compare_reports_marks_fixed_abstain_case(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    focus = tmp_path / "focus.jsonl"
    query = "ArkLab 修什么问题？"
    baseline.write_text(
        json.dumps(
            {
                "summary": {"faithfulness": 0.0, "abstain_rate": 1.0},
                "cases": [
                    {
                        "query": query,
                        "recall_at_k": 1.0,
                        "mrr": 1.0,
                        "ndcg_at_k": 1.0,
                        "faithfulness": 0.0,
                        "answer_relevancy": 0.0,
                        "abstained": True,
                        "top_hit": "doc.md#0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            {
                "summary": {"faithfulness": 1.0, "abstain_rate": 0.0},
                "cases": [
                    {
                        "query": query,
                        "recall_at_k": 1.0,
                        "mrr": 1.0,
                        "ndcg_at_k": 1.0,
                        "faithfulness": 1.0,
                        "answer_relevancy": 0.5,
                        "abstained": False,
                        "top_hit": "doc.md#0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(focus, [{"query": query}])

    report = compare_reports(
        baseline_report_path=baseline,
        candidate_report_path=candidate,
        focus_eval_set_path=focus,
    )

    assert report["status_counts"] == {"fixed": 1}
    assert report["focused"]["delta"]["faithfulness"] == 1.0

