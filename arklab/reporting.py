from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def trace_to_html(trace_path: Path, output_path: Path) -> dict[str, Any]:
    rows = load_jsonl(trace_path)
    items: list[str] = []
    for row in rows:
        title = html.escape(str(row.get("query") or row.get("event") or "trace"))
        answer = html.escape(str(row.get("answer") or ""))
        metrics = html.escape(json.dumps(row.get("metrics", {}), ensure_ascii=False, indent=2))
        hits = html.escape(json.dumps(row.get("hits", []), ensure_ascii=False, indent=2))
        items.append(
            f"<section><h2>{title}</h2><p>{answer}</p>"
            f"<h3>Metrics</h3><pre>{metrics}</pre><h3>Hits</h3><pre>{hits}</pre></section>"
        )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>ArkLab Trace</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.5; }}
    section {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px 0; }}
    pre {{ background: #f6f8fa; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>ArkLab Trace</h1>
  <p>Events: {len(rows)}</p>
  {''.join(items)}
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return {"trace": str(trace_path), "output": str(output_path), "events": len(rows)}


def export_report(report_path: Path, output_path: Path, *, fmt: str) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = report.get("cases", [])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "deepeval-json":
        payload = [
            {
                "input": case.get("query"),
                "actual_output": case.get("answer"),
                "retrieval_context": case.get("contexts") or case.get("hit_ids", []),
                "expected_output": case.get("reference") or case.get("expected_answer"),
                "metrics": {
                    "faithfulness": case.get("faithfulness"),
                    "answer_relevancy": case.get("answer_relevancy"),
                    "recall_at_k": case.get("recall_at_k"),
                },
            }
            for case in cases
        ]
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif fmt == "phoenix-jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(
                    json.dumps(
                        {
                            "input": case.get("query"),
                            "output": case.get("answer"),
                            "retrieval": {
                                "documents": case.get("hit_ids", []),
                                "contexts": case.get("contexts", []),
                            },
                            "evals": {
                                "faithfulness": case.get("faithfulness"),
                                "answer_relevancy": case.get("answer_relevancy"),
                                "recall_at_k": case.get("recall_at_k"),
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    else:
        raise ValueError(f"unknown export format: {fmt}")

    return {"report": str(report_path), "output": str(output_path), "format": fmt, "cases": len(cases)}
