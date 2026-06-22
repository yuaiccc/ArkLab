from __future__ import annotations

import subprocess
from typing import Any


RECIPES: dict[str, list[list[str]]] = {
    "local-smoke": [
        [
            "arklab",
            "eval",
            "--docs",
            "examples/docs",
            "--eval-set",
            "examples/evals/qa.jsonl",
            "--output",
            "data/reports/local-smoke.json",
        ]
    ],
    "enterprise-basic10": [
        [
            "arklab",
            "import-enterprise-rag-bench",
            "--sources",
            "github",
            "--question-types",
            "basic",
            "--max-questions",
            "10",
            "--max-docs-per-source",
            "500",
            "--output-dir",
            "benchmarks/enterprise_rag_bench/github_basic_10",
        ],
        [
            "arklab",
            "eval",
            "--docs",
            "benchmarks/enterprise_rag_bench/github_basic_10/docs",
            "--eval-set",
            "benchmarks/enterprise_rag_bench/github_basic_10/eval.jsonl",
            "--output",
            "data/reports/enterprise-basic10-local.json",
        ],
    ],
    "multihop-smoke": [
        [
            "arklab",
            "import-multihop-rag",
            "--max-questions",
            "8",
            "--max-docs",
            "80",
            "--output-dir",
            "benchmarks/multihop_rag/smoke_8",
        ],
        [
            "arklab",
            "eval",
            "--docs",
            "benchmarks/multihop_rag/smoke_8/docs",
            "--eval-set",
            "benchmarks/multihop_rag/smoke_8/eval.jsonl",
            "--output",
            "data/reports/multihop-smoke-local.json",
        ],
    ],
}


def recipe_manifest(name: str) -> dict[str, Any]:
    if name not in RECIPES:
        raise ValueError(f"unknown recipe: {name}")
    return {"name": name, "commands": RECIPES[name]}


def run_recipe(name: str) -> dict[str, Any]:
    manifest = recipe_manifest(name)
    completed: list[dict[str, Any]] = []
    for command in manifest["commands"]:
        proc = subprocess.run(command, check=False, text=True, capture_output=True)
        completed.append(
            {
                "command": command,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            }
        )
        if proc.returncode != 0:
            return {"name": name, "ok": False, "steps": completed}
    return {"name": name, "ok": True, "steps": completed}
