from __future__ import annotations

from pathlib import Path
from typing import Any

from arklab.trace.writer import TraceWriter


class FailurePoolWriter:
    def __init__(self, path: Path) -> None:
        self.writer = TraceWriter(path)

    def write(self, failure: dict[str, Any]) -> None:
        self.writer.write({"event": "failure_case", **failure})


def classify_failure(
    *,
    recall: float | None,
    abstained: bool,
    faithfulness: float,
    answerable: bool = True,
    abstain_reason: str | None = None,
) -> str | None:
    if abstain_reason == "provider_content_block":
        return "provider_content_block"
    if not answerable:
        return None if abstained else "unanswerable_answered"
    if recall is not None and recall == 0.0:
        return "retrieval_fail"
    if abstained and faithfulness > 0.0:
        return "generation_fail_abstain"
    if abstained:
        return "low_confidence_abstain"
    if faithfulness < 0.35:
        return "generation_fail_hallucination"
    return None
