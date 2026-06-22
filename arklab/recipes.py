from __future__ import annotations

import itertools
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPEAT_FLAG_KEYS = {"tag"}


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


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return value or "run"


def _flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def _add_cli_arg(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(_flag_name(key))
        return
    if isinstance(value, list):
        if key in REPEAT_FLAG_KEYS:
            for item in value:
                command.extend([_flag_name(key), str(item)])
            return
        command.append(_flag_name(key))
        command.extend(str(item) for item in value)
        return
    command.extend([_flag_name(key), str(value)])


def _matrix_variants(matrix: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
    if not matrix:
        return [{}]
    keys = list(matrix)
    values = [matrix[key] for key in keys]
    return [dict(zip(keys, variant, strict=False)) for variant in itertools.product(*values)]


def _variant_slug(config: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(config):
        if key in {"docs", "eval_set", "output", "trace", "failure_pool", "embedding_cache"}:
            continue
        value = config[key]
        if value is None:
            continue
        parts.append(f"{key}-{value}")
    return _slug("-".join(parts))


def load_recipe_file(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"recipe must be a YAML mapping: {path}")
    return data


def recipe_file_manifest(path: Path) -> dict[str, Any]:
    data = load_recipe_file(path)
    name = _slug(str(data.get("name") or path.stem))
    eval_config = data.get("eval")
    if not isinstance(eval_config, dict):
        raise ValueError("recipe must contain an eval mapping")
    matrix = data.get("matrix")
    if matrix is not None and not isinstance(matrix, dict):
        raise ValueError("recipe matrix must be a mapping")

    commands: list[list[str]] = []
    for variant in _matrix_variants(matrix):
        config = {**eval_config, **variant}
        variant_name = _variant_slug(variant)
        run_name = _slug(f"{name}-{variant_name}") if variant_name else name
        config.setdefault("output", f"data/reports/{run_name}.json")
        config.setdefault("trace", f"data/traces/{run_name}.jsonl")
        config.setdefault("failure_pool", f"data/failure_pool/{run_name}.jsonl")
        config.setdefault("experiment_name", run_name)

        command = ["arklab", "eval"]
        for key, value in config.items():
            _add_cli_arg(command, key, value)
        commands.append(command)
    return {"name": name, "source": str(path), "commands": commands}


def run_recipe_file(path: Path) -> dict[str, Any]:
    manifest = recipe_file_manifest(path)
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
            return {"name": manifest["name"], "ok": False, "steps": completed}
    return {"name": manifest["name"], "ok": True, "steps": completed}
