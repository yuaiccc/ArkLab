from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arklab.diagnostics import diagnose_case, summarize_diagnostics
from arklab.flywheel import compare_reports


SUMMARY_METRICS = (
    "cases",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "faithfulness",
    "answer_relevancy",
    "abstain_rate",
    "rejection_rate",
    "false_answer_rate",
)


def _clip(value: Any, limit: int = 140) -> str:
    text = "" if value is None else str(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def _id_matches(expected: str, actual: str) -> bool:
    expected_doc, expected_chunk = _split_doc_id(expected)
    actual_doc, actual_chunk = _split_doc_id(actual)
    if expected_doc != actual_doc:
        return False
    if expected_chunk is not None and actual_chunk is not None:
        return expected_chunk == actual_chunk
    return True


def _split_doc_id(value: str) -> tuple[str, str | None]:
    doc_id, separator, chunk_id = value.strip().partition("#")
    if doc_id.endswith(".txt"):
        doc_id = doc_id[:-4]
    return doc_id, chunk_id if separator else None


def _case_hit_key_groups(case: dict[str, Any]) -> list[list[str]]:
    hit_ids = [[str(item)] for item in case.get("hit_ids") or []]
    match_groups = case.get("hit_match_keys") or []
    for index, group in enumerate(match_groups):
        if not isinstance(group, list):
            continue
        if index >= len(hit_ids):
            hit_ids.append([])
        hit_ids[index].extend(str(item) for item in group if item is not None)
    return hit_ids


def evidence_coverage_case(case: dict[str, Any]) -> dict[str, Any]:
    expected_ids = [
        str(item)
        for item in (
            case.get("expected_relevant_ids")
            or case.get("relevant_ids")
            or case.get("relevant_doc_ids")
            or []
        )
    ]
    hit_key_groups = _case_hit_key_groups(case)
    hit_ids = [key for group in hit_key_groups for key in group]
    matched: list[str] = []
    missing: list[str] = []
    for expected in expected_ids:
        if any(_id_matches(expected, hit_id) for hit_id in hit_ids):
            matched.append(expected)
        else:
            missing.append(expected)

    coverage = len(matched) / len(expected_ids) if expected_ids else None
    top_hit_group = hit_key_groups[0] if hit_key_groups else [str(case.get("top_hit") or "")]
    return {
        "query": case.get("query"),
        "expected_count": len(expected_ids),
        "matched_count": len(matched),
        "coverage": coverage,
        "top_hit_relevant": (
            any(
                _id_matches(expected, hit_key)
                for expected in expected_ids
                for hit_key in top_hit_group
            )
            if expected_ids and top_hit_group
            else None
        ),
        "matched_expected_ids": matched,
        "missing_expected_ids": missing,
        "hit_ids": hit_ids,
    }


def summarize_evidence_coverage(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [evidence_coverage_case(case) for case in cases]
    measurable = [row for row in rows if row["coverage"] is not None]
    full = [row for row in measurable if row["coverage"] == 1.0]
    partial = [row for row in measurable if 0.0 < row["coverage"] < 1.0]
    missing = [row for row in measurable if row["coverage"] == 0.0]
    top_hit_known = [row for row in measurable if row["top_hit_relevant"] is not None]
    return {
        "cases": len(rows),
        "measurable_cases": len(measurable),
        "average_coverage": (
            sum(float(row["coverage"]) for row in measurable) / len(measurable)
            if measurable
            else 0.0
        ),
        "full_coverage_cases": len(full),
        "partial_coverage_cases": len(partial),
        "missing_coverage_cases": len(missing),
        "top_hit_relevance_rate": (
            sum(1.0 if row["top_hit_relevant"] else 0.0 for row in top_hit_known)
            / len(top_hit_known)
            if top_hit_known
            else 0.0
        ),
        "cases_detail": rows,
    }


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_experiment_summary(report_path: Path) -> tuple[str, dict[str, Any]]:
    report = load_report(report_path)
    cases = report.get("cases", [])
    diagnostics = report.get("diagnostics") or summarize_diagnostics(cases)
    evidence = summarize_evidence_coverage(cases)
    failing = []
    for index, case in enumerate(cases, start=1):
        diagnosis = diagnose_case(case)
        if diagnosis["root_cause"] in {"passed", "correct_abstention"}:
            continue
        failing.append(
            {
                "case": index,
                "query": case.get("query"),
                "root_cause": diagnosis["root_cause"],
                "suggested_action": diagnosis["suggested_action"],
                "recall_at_k": case.get("recall_at_k"),
                "faithfulness": case.get("faithfulness"),
                "abstained": case.get("abstained"),
            }
        )

    summary = report.get("summary", {})
    usage = report.get("usage", {})
    payload = {
        "report": str(report_path),
        "summary": summary,
        "diagnostics": diagnostics,
        "evidence_coverage": evidence,
        "usage": usage,
        "cost": report.get("cost"),
        "failing_cases": failing,
    }

    lines = [
        "# ArkLab Experiment Summary",
        "",
        f"- Report: `{report_path}`",
        f"- Cases: `{summary.get('cases', len(cases))}`",
        f"- Evidence coverage: `{_metric(evidence.get('average_coverage'))}`",
        f"- Top-hit relevance rate: `{_metric(evidence.get('top_hit_relevance_rate'))}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for metric_name in SUMMARY_METRICS:
        if metric_name in summary:
            lines.append(f"| {metric_name} | {_metric(summary.get(metric_name))} |")
    lines.extend(
        [
            "",
            "## Root Causes",
            "",
            "```json",
            json.dumps(diagnostics.get("root_cause_counts", {}), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Evidence Coverage",
            "",
            "| Segment | Count |",
            "| --- | ---: |",
            f"| measurable_cases | {evidence.get('measurable_cases', 0)} |",
            f"| full_coverage_cases | {evidence.get('full_coverage_cases', 0)} |",
            f"| partial_coverage_cases | {evidence.get('partial_coverage_cases', 0)} |",
            f"| missing_coverage_cases | {evidence.get('missing_coverage_cases', 0)} |",
            "",
            "## Usage",
            "",
            "```json",
            json.dumps(usage, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Failing Cases",
            "",
            "| Case | Root cause | Recall | Faithfulness | Abstained | Query |",
            "| ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in failing[:20]:
        lines.append(
            f"| {row['case']} | `{row['root_cause']}` | {_metric(row.get('recall_at_k'))} | "
            f"{_metric(row.get('faithfulness'))} | `{row.get('abstained')}` | {_clip(row.get('query'))} |"
        )
    return "\n".join(lines).rstrip() + "\n", payload


def build_compare_summary(
    *,
    baseline_report_path: Path,
    candidate_report_path: Path,
    focus_eval_set_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    comparison = compare_reports(
        baseline_report_path=baseline_report_path,
        candidate_report_path=candidate_report_path,
        focus_eval_set_path=focus_eval_set_path,
    )
    payload = {
        "baseline_report": str(baseline_report_path),
        "candidate_report": str(candidate_report_path),
        "focus_eval_set": str(focus_eval_set_path) if focus_eval_set_path else None,
        "comparison": comparison,
    }
    focused = comparison.get("focused", {})
    delta = focused.get("delta", {})
    lines = [
        "# ArkLab Compare Summary",
        "",
        f"- Baseline: `{baseline_report_path}`",
        f"- Candidate: `{candidate_report_path}`",
        f"- Focus eval set: `{focus_eval_set_path}`",
        "",
        "## Status Counts",
        "",
        "```json",
        json.dumps(comparison.get("status_counts", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Focused Delta",
        "",
        "| Metric | Delta |",
        "| --- | ---: |",
    ]
    for metric_name in (
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
        "faithfulness",
        "answer_relevancy",
        "abstain_rate",
        "score",
    ):
        if metric_name in delta:
            lines.append(f"| {metric_name} | {_metric(delta.get(metric_name))} |")

    lines.extend(
        [
            "",
            "## Changed Cases",
            "",
            "| Status | Baseline faith | Candidate faith | Baseline abstain | Candidate abstain | Query |",
            "| --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in comparison.get("cases", []):
        if row.get("status") == "unchanged":
            continue
        baseline = row.get("baseline", {})
        candidate = row.get("candidate", {})
        lines.append(
            f"| `{row.get('status')}` | {_metric(baseline.get('faithfulness'))} | "
            f"{_metric(candidate.get('faithfulness'))} | `{baseline.get('abstained')}` | "
            f"`{candidate.get('abstained')}` | {_clip(row.get('query'))} |"
        )
    return "\n".join(lines).rstrip() + "\n", payload


def write_summary(markdown: str, payload: dict[str, Any], output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output": str(output_path),
        "json": str(json_path),
    }
