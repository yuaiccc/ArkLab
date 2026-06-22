from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_sha() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def parse_tags(values: list[str] | None) -> dict[str, str]:
    tags: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            tags[value] = "true"
            continue
        key, tag_value = value.split("=", 1)
        tags[key] = tag_value
    return tags


def experiment_id(*, name: str | None, started_at: str) -> str:
    stamp = started_at.replace(":", "").replace("-", "").split(".", 1)[0]
    return f"{stamp}-{name}" if name else stamp


def build_experiment_record(
    *,
    args: Any,
    payload: dict[str, Any],
    started_at: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    name = getattr(args, "experiment_name", None)
    return {
        "experiment_id": experiment_id(name=name, started_at=started_at),
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "git_sha": git_sha(),
        "dataset": {
            "docs": str(getattr(args, "docs", "")),
            "eval_set": str(getattr(args, "eval_set", "")),
        },
        "config": {
            "provider": getattr(args, "provider", None),
            "model": getattr(args, "model", None),
            "retriever": getattr(args, "retriever", None),
            "reranker": getattr(args, "reranker", None),
            "embedding_model": getattr(args, "embedding_model", None),
            "arkcli_prompt_preset": getattr(args, "arkcli_prompt_preset", None),
            "temperature": getattr(args, "temperature", None),
            "top_k": getattr(args, "top_k", None),
            "candidate_k": getattr(args, "candidate_k", None),
            "chunk_tokens": getattr(args, "chunk_tokens", None),
            "chunk_overlap": getattr(args, "chunk_overlap", None),
            "llm_judge": getattr(args, "llm_judge", None),
        },
        "summary": payload.get("summary", {}),
        "diagnostics": {
            "root_cause_counts": payload.get("diagnostics", {}).get("root_cause_counts", {}),
            "suggested_action_counts": payload.get("diagnostics", {}).get(
                "suggested_action_counts",
                {},
            ),
        },
        "usage": payload.get("usage", {}),
        "cost": payload.get("cost"),
        "report": str(getattr(args, "output", "") or ""),
        "trace": str(getattr(args, "trace", "") or ""),
        "failure_pool": str(getattr(args, "failure_pool", "") or ""),
        "tags": parse_tags(getattr(args, "tag", None)),
    }


def append_experiment(registry_path: Path, record: dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_experiments(registry_path: Path) -> list[dict[str, Any]]:
    if not registry_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with registry_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
