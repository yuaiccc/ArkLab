import json
from pathlib import Path

from arklab.analysis import (
    build_compare_summary,
    build_experiment_summary,
    summarize_evidence_coverage,
)
from arklab.cli import main
from arklab.cost import estimate_cost, normalize_usage
from arklab.diagnostics import diagnose_case
from arklab.drilldown import build_compare_drilldown, build_drilldown
from arklab.evaluation.llm_judge import extract_json_object, normalize_judge_payload
from arklab.reporting import export_report, trace_to_html
from arklab.trends import build_trend


def test_diagnostics_classifies_retrieval_failure() -> None:
    diagnosis = diagnose_case(
        {
            "answerable": True,
            "recall_at_k": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "abstained": False,
        }
    )

    assert diagnosis["root_cause"] == "retrieval_failure"
    assert diagnosis["suggested_action"] == "improve_index_chunking_embedding_or_query_rewrite"


def test_usage_and_cost_normalization() -> None:
    usage = normalize_usage({"prompt_tokens": 1000, "completion_tokens": 250})
    cost = estimate_cost(usage, input_price_per_1m=2.0, output_price_per_1m=8.0)

    assert usage == {"input_tokens": 1000, "output_tokens": 250, "total_tokens": 1250}
    assert cost["total_cost"] == 0.004


def test_trace_html_and_export_formats(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "query": "Q",
                "answer": "A",
                "metrics": {"faithfulness": 1.0},
                "hits": [{"id": "doc#0"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    html = tmp_path / "trace.html"
    result = trace_to_html(trace, html)

    assert result["events"] == 1
    assert "ArkLab Trace" in html.read_text(encoding="utf-8")

    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "summary": {"cases": 1, "recall_at_k": 1.0},
                "cases": [
                    {
                        "query": "Q",
                        "answer": "A",
                        "hit_ids": ["doc#0"],
                        "faithfulness": 1.0,
                        "answer_relevancy": 1.0,
                        "recall_at_k": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    deepeval = tmp_path / "deepeval.json"
    phoenix = tmp_path / "phoenix.jsonl"

    assert export_report(report, deepeval, fmt="deepeval-json")["cases"] == 1
    assert json.loads(deepeval.read_text(encoding="utf-8"))[0]["input"] == "Q"
    assert export_report(report, phoenix, fmt="phoenix-jsonl")["cases"] == 1
    assert json.loads(phoenix.read_text(encoding="utf-8").strip())["output"] == "A"


def test_drilldown_writes_case_markdown(tmp_path: Path, capsys) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "query": "Q",
                        "answer": "无法基于当前知识库回答这个问题。",
                        "expected_answer": "A",
                        "expected_relevant_ids": ["doc#0"],
                        "hit_ids": ["doc#1"],
                        "contexts": ["wrong context"],
                        "answerable": True,
                        "abstained": True,
                        "abstain_reason": "provider_abstain",
                        "recall_at_k": 0.0,
                        "mrr": 0.0,
                        "ndcg_at_k": 0.0,
                        "faithfulness": 0.0,
                        "answer_relevancy": 0.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "drilldown"
    result = build_drilldown(report_path=report, output_dir=output_dir, failures_only=True)

    assert result["cases"] == 1
    assert "retrieval_failure" in (output_dir / "index.md").read_text(encoding="utf-8")
    assert "Expected relevant ids" in next(output_dir.glob("001-*.md")).read_text(
        encoding="utf-8"
    )

    exit_code = main(
        [
            "drilldown",
            "--report",
            str(report),
            "--output-dir",
            str(tmp_path / "drilldown-cli"),
            "--failures-only",
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert printed["cases"] == 1


def test_compare_drilldown_writes_fixed_case(tmp_path: Path, capsys) -> None:
    query = "ArkLab 修什么问题？"
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps(
            {
                "summary": {"faithfulness": 0.0, "abstain_rate": 1.0},
                "cases": [
                    {
                        "query": query,
                        "answer": "无法基于当前知识库回答这个问题。",
                        "expected_answer": "ArkLab 诊断 RAG 失败。",
                        "expected_relevant_ids": ["doc#0"],
                        "hit_ids": ["doc#0"],
                        "contexts": ["ArkLab 诊断 RAG 失败。"],
                        "answerable": True,
                        "abstained": True,
                        "abstain_reason": "provider_abstain",
                        "recall_at_k": 1.0,
                        "mrr": 1.0,
                        "ndcg_at_k": 1.0,
                        "faithfulness": 0.0,
                        "answer_relevancy": 0.0,
                        "top_hit": "doc#0",
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
                        "answer": "ArkLab 诊断 RAG 失败。",
                        "expected_answer": "ArkLab 诊断 RAG 失败。",
                        "expected_relevant_ids": ["doc#0"],
                        "hit_ids": ["doc#0"],
                        "contexts": ["ArkLab 诊断 RAG 失败。"],
                        "answerable": True,
                        "abstained": False,
                        "abstain_reason": None,
                        "recall_at_k": 1.0,
                        "mrr": 1.0,
                        "ndcg_at_k": 1.0,
                        "faithfulness": 1.0,
                        "answer_relevancy": 0.6,
                        "top_hit": "doc#0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "compare"
    result = build_compare_drilldown(
        baseline_report_path=baseline,
        candidate_report_path=candidate,
        output_dir=output_dir,
    )

    assert result["cases"] == 1
    assert result["status_counts"] == {"fixed": 1}
    page = next(output_dir.glob("001-fixed-*.md")).read_text(encoding="utf-8")
    assert "Metric Delta" in page
    assert "Baseline" in page
    assert "Candidate" in page

    exit_code = main(
        [
            "compare-drilldown",
            "--baseline-report",
            str(baseline),
            "--candidate-report",
            str(candidate),
            "--output-dir",
            str(tmp_path / "compare-cli"),
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert printed["cases"] == 1


def test_evidence_coverage_and_summary_commands(tmp_path: Path, capsys) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "summary": {
                    "cases": 2,
                    "recall_at_k": 0.5,
                    "faithfulness": 0.5,
                    "answer_relevancy": 0.4,
                    "abstain_rate": 0.5,
                },
                "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                "cases": [
                    {
                        "query": "Q1",
                        "answer": "A1",
                        "expected_relevant_ids": ["doc#0"],
                        "hit_ids": ["doc#0"],
                        "contexts": ["A1"],
                        "answerable": True,
                        "abstained": False,
                        "recall_at_k": 1.0,
                        "mrr": 1.0,
                        "ndcg_at_k": 1.0,
                        "faithfulness": 1.0,
                        "answer_relevancy": 0.8,
                    },
                    {
                        "query": "Q2",
                        "answer": "无法基于当前知识库回答这个问题。",
                        "expected_relevant_ids": ["doc#1"],
                        "hit_ids": ["doc#0"],
                        "contexts": ["A1"],
                        "answerable": True,
                        "abstained": True,
                        "abstain_reason": "provider_abstain",
                        "recall_at_k": 0.0,
                        "mrr": 0.0,
                        "ndcg_at_k": 0.0,
                        "faithfulness": 0.0,
                        "answer_relevancy": 0.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    coverage = summarize_evidence_coverage(json.loads(report.read_text())["cases"])
    assert coverage["average_coverage"] == 0.5
    assert coverage["missing_coverage_cases"] == 1

    markdown, payload = build_experiment_summary(report)
    assert "Evidence Coverage" in markdown
    assert payload["evidence_coverage"]["average_coverage"] == 0.5

    output = tmp_path / "summary.md"
    exit_code = main(["summary", "--report", str(report), "--output", str(output)])
    printed = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output.exists()
    assert Path(printed["json"]).exists()


def test_compare_summary_command(tmp_path: Path, capsys) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    query = "Q"
    baseline.write_text(
        json.dumps(
            {
                "summary": {"faithfulness": 0.0, "abstain_rate": 1.0},
                "cases": [
                    {
                        "query": query,
                        "faithfulness": 0.0,
                        "answer_relevancy": 0.0,
                        "recall_at_k": 1.0,
                        "mrr": 1.0,
                        "ndcg_at_k": 1.0,
                        "abstained": True,
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
                        "faithfulness": 1.0,
                        "answer_relevancy": 0.5,
                        "recall_at_k": 1.0,
                        "mrr": 1.0,
                        "ndcg_at_k": 1.0,
                        "abstained": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    markdown, payload = build_compare_summary(
        baseline_report_path=baseline,
        candidate_report_path=candidate,
    )
    assert "Changed Cases" in markdown
    assert payload["comparison"]["status_counts"] == {"fixed": 1}

    output = tmp_path / "compare-summary.md"
    exit_code = main(
        [
            "summary",
            "--baseline-report",
            str(baseline),
            "--candidate-report",
            str(candidate),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["output"] == str(output)


def test_trend_and_recipe_cli(tmp_path: Path, capsys) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps({"summary": {"cases": 1, "recall_at_k": 0.5}}), encoding="utf-8")
    second.write_text(json.dumps({"summary": {"cases": 1, "recall_at_k": 1.0}}), encoding="utf-8")

    trend = build_trend([str(tmp_path / "*.json")])
    assert [row["recall_at_k"] for row in trend["rows"]] == [0.5, 1.0]

    output = tmp_path / "trend.json"
    exit_code = main(["trend", "--reports", str(tmp_path / "*.json"), "--output", str(output)])
    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["reports"] == 2
    capsys.readouterr()

    exit_code = main(["recipe", "--name", "local-smoke"])
    printed = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert printed["name"] == "local-smoke"


def test_llm_judge_json_extraction_and_clamping() -> None:
    payload = extract_json_object('```json\n{"faithfulness": 2, "root_cause": "passed"}\n```')
    normalized = normalize_judge_payload(payload)

    assert normalized["faithfulness"] == 1.0
    assert normalized["root_cause"] == "passed"
