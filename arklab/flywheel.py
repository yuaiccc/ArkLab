from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ACTION_BY_FAILURE_TYPE = {
    "retrieval_fail": "improve_retrieval_or_query_rewrite",
    "low_confidence_abstain": "prompt_or_context_answerability",
    "generation_fail_abstain": "prompt_or_context_answerability",
    "generation_fail_hallucination": "faithfulness_guardrail",
    "unanswerable_answered": "abstention_guardrail",
}

HIGHER_IS_BETTER = (
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "faithfulness",
    "answer_relevancy",
)
LOWER_IS_BETTER = ("abstain_rate",)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _relevance_from_eval_row(row: dict[str, Any]) -> list[str]:
    if isinstance(row.get("relevance"), dict):
        return [str(key) for key in row["relevance"]]
    relevant_ids = row.get("relevant_ids") or row.get("relevant_doc_ids") or []
    return [str(item) for item in relevant_ids]


def _relevance_from_failure(failure: dict[str, Any]) -> list[str]:
    relevance = failure.get("expected_relevance")
    if isinstance(relevance, dict):
        return [str(key) for key in relevance]
    relevant_ids = failure.get("relevant_ids") or failure.get("relevant_doc_ids") or []
    return [str(item) for item in relevant_ids]


def _source_eval_by_query(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {str(row["query"]): row for row in load_jsonl(path) if row.get("query")}


def promote_failures_to_eval_set(
    *,
    failure_pool_path: Path,
    source_eval_set_path: Path | None = None,
    max_cases: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    failures = [
        row
        for row in load_jsonl(failure_pool_path)
        if row.get("event") == "failure_case" and row.get("query")
    ]
    source_rows = _source_eval_by_query(source_eval_set_path)

    latest_by_query: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for failure in failures:
        query = str(failure["query"])
        if query in latest_by_query:
            duplicates += 1
        latest_by_query[query] = failure

    selected = list(latest_by_query.values())
    selected.sort(key=lambda item: str(item.get("ts", "")))
    if max_cases is not None:
        selected = selected[:max_cases]

    promoted: list[dict[str, Any]] = []
    missing_reference = 0
    action_counts: Counter[str] = Counter()
    for failure in selected:
        query = str(failure["query"])
        source = source_rows.get(query, {})
        relevant_ids = _relevance_from_eval_row(source) or _relevance_from_failure(failure)
        answer = source.get("answer") or source.get("reference")
        if not relevant_ids:
            missing_reference += 1

        failure_type = str(failure.get("failure_type") or "unknown")
        action = ACTION_BY_FAILURE_TYPE.get(failure_type, "manual_diagnosis")
        action_counts[action] += 1

        row: dict[str, Any] = {
            "query": query,
            "relevant_ids": relevant_ids,
            "arklab_flywheel": {
                "source": "failure_pool",
                "failure_type": failure_type,
                "suggested_action": action,
                "promoted_at": _now_iso(),
                "failure_ts": failure.get("ts"),
                "previous_answer": failure.get("answer"),
                "previous_abstained": failure.get("abstained"),
                "previous_abstain_reason": failure.get("abstain_reason"),
                "previous_metrics": failure.get("metrics", {}),
            },
        }
        if source.get("answerable") is not None or failure.get("answerable") is not None:
            row["answerable"] = source.get("answerable", failure.get("answerable"))
        if source.get("expected_behavior") or failure.get("expected_behavior"):
            row["expected_behavior"] = source.get(
                "expected_behavior", failure.get("expected_behavior")
            )
        if source.get("unanswerable_category") or failure.get("unanswerable_category"):
            row["unanswerable_category"] = source.get(
                "unanswerable_category", failure.get("unanswerable_category")
            )
        if answer:
            row["answer"] = answer
        promoted.append(row)

    failure_type_counts = Counter(str(row.get("failure_type") or "unknown") for row in failures)
    promoted_type_counts = Counter(
        str(row["arklab_flywheel"]["failure_type"]) for row in promoted
    )
    manifest = {
        "created_at": _now_iso(),
        "failure_pool": str(failure_pool_path),
        "source_eval_set": str(source_eval_set_path) if source_eval_set_path else None,
        "raw_failures": len(failures),
        "deduped_duplicates": duplicates,
        "promoted_cases": len(promoted),
        "missing_reference_cases": missing_reference,
        "failure_type_counts": dict(failure_type_counts),
        "promoted_failure_type_counts": dict(promoted_type_counts),
        "suggested_action_counts": dict(action_counts),
    }
    return promoted, manifest


def _case_by_query(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case["query"]): case for case in report.get("cases", []) if case.get("query")}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _case_metric(case: dict[str, Any], metric: str) -> float:
    value = case.get(metric)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {"cases": 0}
    return {
        "cases": len(cases),
        "recall_at_k": _mean([_case_metric(case, "recall_at_k") for case in cases]),
        "mrr": _mean([_case_metric(case, "mrr") for case in cases]),
        "ndcg_at_k": _mean([_case_metric(case, "ndcg_at_k") for case in cases]),
        "faithfulness": _mean([_case_metric(case, "faithfulness") for case in cases]),
        "answer_relevancy": _mean([_case_metric(case, "answer_relevancy") for case in cases]),
        "abstain_rate": _mean([1.0 if case.get("abstained") else 0.0 for case in cases]),
    }


def _score(summary: dict[str, Any]) -> float:
    return (
        float(summary.get("recall_at_k", 0.0)) * 0.35
        + float(summary.get("mrr", 0.0)) * 0.20
        + float(summary.get("ndcg_at_k", 0.0)) * 0.20
        + float(summary.get("faithfulness", 0.0)) * 0.20
        + float(summary.get("answer_relevancy", 0.0)) * 0.05
        - float(summary.get("abstain_rate", 0.0)) * 0.15
    )


def _deltas(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for metric in HIGHER_IS_BETTER:
        deltas[metric] = float(candidate.get(metric, 0.0)) - float(baseline.get(metric, 0.0))
    for metric in LOWER_IS_BETTER:
        deltas[metric] = float(baseline.get(metric, 0.0)) - float(candidate.get(metric, 0.0))
    deltas["score"] = _score(candidate) - _score(baseline)
    return deltas


def _case_status(baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    recall_delta = _case_metric(candidate, "recall_at_k") - _case_metric(baseline, "recall_at_k")
    faithfulness_delta = _case_metric(candidate, "faithfulness") - _case_metric(
        baseline, "faithfulness"
    )
    ndcg_delta = _case_metric(candidate, "ndcg_at_k") - _case_metric(baseline, "ndcg_at_k")
    abstain_fixed = bool(baseline.get("abstained")) and not bool(candidate.get("abstained"))
    abstain_regressed = not bool(baseline.get("abstained")) and bool(candidate.get("abstained"))

    if recall_delta < -0.05 or faithfulness_delta < -0.15 or abstain_regressed:
        return "regressed"
    if recall_delta > 0.05 or faithfulness_delta > 0.15 or ndcg_delta > 0.10 or abstain_fixed:
        return "fixed"
    return "unchanged"


def compare_reports(
    *,
    baseline_report_path: Path,
    candidate_report_path: Path,
    focus_eval_set_path: Path | None = None,
) -> dict[str, Any]:
    baseline_report = load_json(baseline_report_path)
    candidate_report = load_json(candidate_report_path)
    baseline_cases = _case_by_query(baseline_report)
    candidate_cases = _case_by_query(candidate_report)

    if focus_eval_set_path:
        focus_queries = [str(row["query"]) for row in load_jsonl(focus_eval_set_path) if row.get("query")]
    else:
        focus_queries = sorted(set(baseline_cases) & set(candidate_cases))

    comparisons: list[dict[str, Any]] = []
    missing_in_baseline: list[str] = []
    missing_in_candidate: list[str] = []
    for query in focus_queries:
        baseline_case = baseline_cases.get(query)
        candidate_case = candidate_cases.get(query)
        if not baseline_case:
            missing_in_baseline.append(query)
            continue
        if not candidate_case:
            missing_in_candidate.append(query)
            continue
        comparisons.append(
            {
                "query": query,
                "status": _case_status(baseline_case, candidate_case),
                "baseline": {
                    "recall_at_k": _case_metric(baseline_case, "recall_at_k"),
                    "mrr": _case_metric(baseline_case, "mrr"),
                    "ndcg_at_k": _case_metric(baseline_case, "ndcg_at_k"),
                    "faithfulness": _case_metric(baseline_case, "faithfulness"),
                    "answer_relevancy": _case_metric(baseline_case, "answer_relevancy"),
                    "abstained": bool(baseline_case.get("abstained")),
                    "top_hit": baseline_case.get("top_hit"),
                },
                "candidate": {
                    "recall_at_k": _case_metric(candidate_case, "recall_at_k"),
                    "mrr": _case_metric(candidate_case, "mrr"),
                    "ndcg_at_k": _case_metric(candidate_case, "ndcg_at_k"),
                    "faithfulness": _case_metric(candidate_case, "faithfulness"),
                    "answer_relevancy": _case_metric(candidate_case, "answer_relevancy"),
                    "abstained": bool(candidate_case.get("abstained")),
                    "top_hit": candidate_case.get("top_hit"),
                },
            }
        )

    baseline_focus_summary = _summarize_cases(
        [baseline_cases[item["query"]] for item in comparisons]
    )
    candidate_focus_summary = _summarize_cases(
        [candidate_cases[item["query"]] for item in comparisons]
    )
    status_counts = Counter(item["status"] for item in comparisons)
    return {
        "created_at": _now_iso(),
        "baseline_report": str(baseline_report_path),
        "candidate_report": str(candidate_report_path),
        "focus_eval_set": str(focus_eval_set_path) if focus_eval_set_path else None,
        "overall": {
            "baseline_summary": baseline_report.get("summary", {}),
            "candidate_summary": candidate_report.get("summary", {}),
            "delta": _deltas(
                baseline_report.get("summary", {}),
                candidate_report.get("summary", {}),
            ),
        },
        "focused": {
            "baseline_summary": baseline_focus_summary,
            "candidate_summary": candidate_focus_summary,
            "delta": _deltas(baseline_focus_summary, candidate_focus_summary),
        },
        "status_counts": dict(status_counts),
        "missing_in_baseline": missing_in_baseline,
        "missing_in_candidate": missing_in_candidate,
        "cases": comparisons,
    }
