from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any


TREND_KEYS = (
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


def build_trend(patterns: list[str]) -> dict[str, Any]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(item) for item in glob.glob(pattern))
    paths = sorted(set(paths), key=lambda path: str(path))

    rows: list[dict[str, Any]] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        summary = report.get("summary", {})
        row = {"report": str(path)}
        for key in TREND_KEYS:
            if key in summary:
                row[key] = summary[key]
        rows.append(row)

    return {"reports": len(rows), "rows": rows}
