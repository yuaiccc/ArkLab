from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any


JUDGE_INSTRUCTIONS = (
    "你是严格的 RAG 评测裁判。只根据给定问题、答案、参考答案和检索上下文评分。"
    "不要因为答案听起来合理就给高分；答案必须被上下文支持。"
    "只输出 JSON，不要输出 Markdown。"
)


def build_judge_prompt(case: dict[str, Any]) -> str:
    contexts = case.get("contexts") or case.get("hit_ids") or []
    return (
        "请评估下面这个 RAG 回答。\n\n"
        f"问题:\n{case.get('query')}\n\n"
        f"模型答案:\n{case.get('answer')}\n\n"
        f"参考答案或期望行为:\n{case.get('reference') or case.get('expected_answer') or case.get('expected_behavior')}\n\n"
        f"检索上下文或命中文档:\n{json.dumps(contexts, ensure_ascii=False, indent=2)}\n\n"
        "请输出如下 JSON：\n"
        "{\n"
        '  "faithfulness": 0 到 1 的数字,\n'
        '  "answer_relevancy": 0 到 1 的数字,\n'
        '  "correctness": 0 到 1 的数字,\n'
        '  "root_cause": "passed/retrieval_failure/unsupported_generation/off_topic_answer/over_abstention/other",\n'
        '  "reason": "一句中文解释"\n'
        "}\n"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _clamp_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


def normalize_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "faithfulness": _clamp_score(payload.get("faithfulness")),
        "answer_relevancy": _clamp_score(payload.get("answer_relevancy")),
        "correctness": _clamp_score(payload.get("correctness")),
        "root_cause": str(payload.get("root_cause") or "other"),
        "reason": str(payload.get("reason") or ""),
    }
    return normalized


def judge_cases_with_arkcli(
    cases: list[dict[str, Any]],
    *,
    model: str | None,
    timeout: int = 120,
) -> list[dict[str, Any]]:
    if shutil.which("arkcli") is None:
        raise RuntimeError("arkcli binary not found")

    judged: list[dict[str, Any]] = []
    for case in cases:
        cmd = [
            "arkcli",
            "+chat",
            "--format",
            "json",
            "--instructions",
            JUDGE_INSTRUCTIONS,
            "--temperature",
            "0",
            "--max-output-tokens",
            "512",
            "--thinking",
            "disabled",
        ]
        if model:
            cmd.extend(["--model", model])
        cmd.append(build_judge_prompt(case))

        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
        raw = json.loads(proc.stdout)
        payload = extract_json_object(str(raw.get("content", "")))
        judged.append(
            {
                "query": case.get("query"),
                "model": raw.get("model") or model or "arkcli-default",
                "usage": raw.get("usage") or {},
                **normalize_judge_payload(payload),
            }
        )
    return judged
