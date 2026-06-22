from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from arklab.diagnostics import diagnose_case
from arklab.flywheel import compare_reports


PASSING_ROOT_CAUSES = {"passed", "correct_abstention"}


def _slug(value: str, *, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return value[:80] or fallback


def _clip(value: Any, limit: int = 900) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return "(not provided)"
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def _delta(baseline: Any, candidate: Any) -> str:
    if isinstance(baseline, (int, float)) and isinstance(candidate, (int, float)):
        return f"{candidate - baseline:+.4f}"
    return ""


def _case_status(diagnosis: dict[str, Any]) -> str:
    return "pass" if diagnosis["root_cause"] in PASSING_ROOT_CAUSES else "fail"


def _markdown_case(
    *,
    index: int,
    case: dict[str, Any],
    diagnosis: dict[str, Any],
) -> str:
    provider = case.get("provider") or {}
    llm_judge = case.get("llm_judge") or {}
    lines = [
        f"# Case {index}: {_case_status(diagnosis).upper()}",
        "",
        f"- Root cause: `{diagnosis['root_cause']}`",
        f"- Suggested action: `{diagnosis['suggested_action']}`",
        f"- Abstained: `{case.get('abstained')}`",
        f"- Abstain reason: `{case.get('abstain_reason')}`",
        f"- Top hit: `{case.get('top_hit')}`",
        f"- Provider: `{provider.get('model')}`",
    ]
    if provider.get("error_type"):
        lines.append(f"- Provider error: `{provider.get('error_type')}`")
    lines.extend(
        [
            "",
            "## Query",
            "",
            _clip(case.get("query"), 2000),
            "",
            "## Expected",
            "",
            _clip(case.get("expected_answer") or case.get("reference") or case.get("expected_behavior")),
            "",
            "## Answer",
            "",
            _clip(case.get("answer"), 2000),
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| recall_at_k | {_metric(case.get('recall_at_k'))} |",
            f"| mrr | {_metric(case.get('mrr'))} |",
            f"| ndcg_at_k | {_metric(case.get('ndcg_at_k'))} |",
            f"| faithfulness | {_metric(case.get('faithfulness'))} |",
            f"| answer_relevancy | {_metric(case.get('answer_relevancy'))} |",
            "",
        ]
    )
    if llm_judge:
        lines.extend(
            [
                "## LLM Judge",
                "",
                f"- Root cause: `{llm_judge.get('root_cause')}`",
                f"- Faithfulness: `{_metric(llm_judge.get('faithfulness'))}`",
                f"- Answer relevancy: `{_metric(llm_judge.get('answer_relevancy'))}`",
                f"- Correctness: `{_metric(llm_judge.get('correctness'))}`",
                f"- Reason: {_clip(llm_judge.get('reason'), 1200)}",
                "",
            ]
        )

    expected_ids = case.get("expected_relevant_ids") or case.get("relevant_ids") or []
    hit_ids = case.get("hit_ids") or []
    lines.extend(
        [
            "## Retrieval",
            "",
            f"- Expected relevant ids: `{json.dumps(expected_ids, ensure_ascii=False)}`",
            f"- Hit ids: `{json.dumps(hit_ids, ensure_ascii=False)}`",
            "",
        ]
    )
    for context_index, context in enumerate(case.get("contexts") or [], start=1):
        lines.extend(
            [
                f"### Context {context_index}",
                "",
                "```text",
                _clip(context, 1800),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_drilldown(
    *,
    report_path: Path,
    output_dir: Path,
    failures_only: bool = False,
    max_cases: int | None = None,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = report.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("report cases must be a list")

    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    written = 0
    for case_index, case in enumerate(cases, start=1):
        diagnosis = diagnose_case(case)
        status = _case_status(diagnosis)
        if failures_only and status == "pass":
            continue
        if max_cases is not None and written >= max_cases:
            break
        filename = f"{case_index:03d}-{_slug(str(case.get('query') or ''), fallback='case')}.md"
        path = output_dir / filename
        path.write_text(
            _markdown_case(index=case_index, case=case, diagnosis=diagnosis),
            encoding="utf-8",
        )
        written += 1
        index_rows.append(
            {
                "case": case_index,
                "status": status,
                "root_cause": diagnosis["root_cause"],
                "suggested_action": diagnosis["suggested_action"],
                "query": case.get("query"),
                "file": filename,
            }
        )

    index_lines = [
        "# ArkLab Case Drilldown",
        "",
        f"- Report: `{report_path}`",
        f"- Cases written: `{written}`",
        f"- Failures only: `{failures_only}`",
        "",
        "| Case | Status | Root cause | Suggested action | File |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in index_rows:
        index_lines.append(
            f"| {row['case']} | {row['status']} | `{row['root_cause']}` | "
            f"`{row['suggested_action']}` | [{row['file']}]({row['file']}) |"
        )
    (output_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    (output_dir / "index.json").write_text(
        json.dumps(index_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "report": str(report_path),
        "output_dir": str(output_dir),
        "cases": written,
        "failures_only": failures_only,
        "index": str(output_dir / "index.md"),
    }


def _case_by_query(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case["query"]): case for case in report.get("cases", []) if case.get("query")}


def _root(case: dict[str, Any]) -> str:
    return str(diagnose_case(case)["root_cause"])


def _answer_block(title: str, case: dict[str, Any]) -> list[str]:
    provider = case.get("provider") or {}
    lines = [
        f"## {title}",
        "",
        f"- Root cause: `{_root(case)}`",
        f"- Abstained: `{case.get('abstained')}`",
        f"- Abstain reason: `{case.get('abstain_reason')}`",
        f"- Top hit: `{case.get('top_hit')}`",
        f"- Provider: `{provider.get('model')}`",
    ]
    if provider.get("error_type"):
        lines.append(f"- Provider error: `{provider.get('error_type')}`")
    lines.extend(
        [
            "",
            "### Answer",
            "",
            _clip(case.get("answer"), 2000),
            "",
        ]
    )
    judge = case.get("llm_judge") or {}
    if judge:
        lines.extend(
            [
                "### LLM Judge",
                "",
                f"- Root cause: `{judge.get('root_cause')}`",
                f"- Correctness: `{_metric(judge.get('correctness'))}`",
                f"- Reason: {_clip(judge.get('reason'), 900)}",
                "",
            ]
        )
    return lines


def _comparison_markdown(
    *,
    index: int,
    comparison: dict[str, Any],
    baseline_case: dict[str, Any],
    candidate_case: dict[str, Any],
) -> str:
    query = str(comparison.get("query") or "")
    expected_answer = (
        candidate_case.get("expected_answer")
        or baseline_case.get("expected_answer")
        or candidate_case.get("reference")
        or baseline_case.get("reference")
        or candidate_case.get("expected_behavior")
        or baseline_case.get("expected_behavior")
    )
    expected_ids = (
        candidate_case.get("expected_relevant_ids")
        or baseline_case.get("expected_relevant_ids")
        or candidate_case.get("relevant_ids")
        or baseline_case.get("relevant_ids")
        or []
    )
    metric_names = ("recall_at_k", "mrr", "ndcg_at_k", "faithfulness", "answer_relevancy")
    lines = [
        f"# Comparison {index}: {str(comparison.get('status')).upper()}",
        "",
        f"- Status: `{comparison.get('status')}`",
        f"- Baseline root cause: `{_root(baseline_case)}`",
        f"- Candidate root cause: `{_root(candidate_case)}`",
        "",
        "## Query",
        "",
        _clip(query, 2000),
        "",
        "## Expected",
        "",
        _clip(expected_answer, 2000),
        "",
        "## Metric Delta",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric_name in metric_names:
        baseline_value = baseline_case.get(metric_name)
        candidate_value = candidate_case.get(metric_name)
        lines.append(
            f"| {metric_name} | {_metric(baseline_value)} | "
            f"{_metric(candidate_value)} | {_delta(baseline_value, candidate_value)} |"
        )
    lines.extend(
        [
            f"| abstained | {baseline_case.get('abstained')} | {candidate_case.get('abstained')} |  |",
            "",
        ]
    )
    lines.extend(_answer_block("Baseline", baseline_case))
    lines.extend(_answer_block("Candidate", candidate_case))
    lines.extend(
        [
            "## Retrieval",
            "",
            f"- Expected relevant ids: `{json.dumps(expected_ids, ensure_ascii=False)}`",
            f"- Baseline hit ids: `{json.dumps(baseline_case.get('hit_ids') or [], ensure_ascii=False)}`",
            f"- Candidate hit ids: `{json.dumps(candidate_case.get('hit_ids') or [], ensure_ascii=False)}`",
            "",
            "### Baseline Contexts",
            "",
        ]
    )
    for context_index, context in enumerate(baseline_case.get("contexts") or [], start=1):
        lines.extend([f"#### B{context_index}", "", "```text", _clip(context, 1200), "```", ""])
    lines.extend(["### Candidate Contexts", ""])
    for context_index, context in enumerate(candidate_case.get("contexts") or [], start=1):
        lines.extend([f"#### C{context_index}", "", "```text", _clip(context, 1200), "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def build_compare_drilldown(
    *,
    baseline_report_path: Path,
    candidate_report_path: Path,
    output_dir: Path,
    focus_eval_set_path: Path | None = None,
    statuses: set[str] | None = None,
    max_cases: int | None = None,
) -> dict[str, Any]:
    comparison = compare_reports(
        baseline_report_path=baseline_report_path,
        candidate_report_path=candidate_report_path,
        focus_eval_set_path=focus_eval_set_path,
    )
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    candidate_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    baseline_cases = _case_by_query(baseline_report)
    candidate_cases = _case_by_query(candidate_report)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    written = 0
    for case_index, item in enumerate(comparison.get("cases", []), start=1):
        status = str(item.get("status"))
        if statuses and status not in statuses:
            continue
        if max_cases is not None and written >= max_cases:
            break
        query = str(item.get("query"))
        baseline_case = baseline_cases[query]
        candidate_case = candidate_cases[query]
        filename = f"{case_index:03d}-{status}-{_slug(query, fallback='case')}.md"
        (output_dir / filename).write_text(
            _comparison_markdown(
                index=case_index,
                comparison=item,
                baseline_case=baseline_case,
                candidate_case=candidate_case,
            ),
            encoding="utf-8",
        )
        written += 1
        index_rows.append(
            {
                "case": case_index,
                "status": status,
                "baseline_root_cause": _root(baseline_case),
                "candidate_root_cause": _root(candidate_case),
                "query": query,
                "file": filename,
            }
        )

    index_lines = [
        "# ArkLab Compare Drilldown",
        "",
        f"- Baseline: `{baseline_report_path}`",
        f"- Candidate: `{candidate_report_path}`",
        f"- Focus eval set: `{focus_eval_set_path}`",
        f"- Cases written: `{written}`",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(comparison.get("status_counts", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "| Case | Status | Baseline root | Candidate root | File |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in index_rows:
        index_lines.append(
            f"| {row['case']} | `{row['status']}` | `{row['baseline_root_cause']}` | "
            f"`{row['candidate_root_cause']}` | [{row['file']}]({row['file']}) |"
        )
    (output_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    (output_dir / "index.json").write_text(
        json.dumps(index_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "baseline_report": str(baseline_report_path),
        "candidate_report": str(candidate_report_path),
        "focus_eval_set": str(focus_eval_set_path) if focus_eval_set_path else None,
        "output_dir": str(output_dir),
        "cases": written,
        "statuses": sorted(statuses) if statuses else None,
        "index": str(output_dir / "index.md"),
        "status_counts": comparison.get("status_counts", {}),
    }
