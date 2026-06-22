from __future__ import annotations

import json
import shutil
import subprocess

from arklab.evaluation.metrics import lexical_faithfulness
from arklab.models import ProviderResult, RetrievalHit


BASELINE_INSTRUCTIONS = (
    "你是 ArkLab 的 RAG 回答器。必须优先使用给定上下文回答。"
    "如果上下文中有直接答案，或能从上下文中的句子合理推出答案，就必须回答，不能拒答。"
    "只有在上下文完全没有相关信息时，才回答“无法基于当前知识库回答”。"
    "回答要简洁，不要编造上下文之外的信息。"
)

MULTIHOP_INSTRUCTIONS = (
    "你是 ArkLab 的 RAG 回答器。必须优先使用给定上下文回答。"
    "如果上下文中有直接答案，或能从上下文中的句子合理推出答案，就必须回答，不能拒答。"
    "如果问题需要综合多个上下文片段，请跨片段合成答案。"
    "如果问题是 yes/no 或对比类问题，请先回答 Yes/No，再用一句话说明依据。"
    "不要要求答案必须出现在同一句话里；多个来源分别支持问题中的不同条件时，也应作答。"
    "只有在上下文完全没有相关信息时，才回答“无法基于当前知识库回答”。"
    "回答要简洁，不要编造上下文之外的信息。"
)

INSTRUCTIONS_BY_PRESET = {
    "baseline": BASELINE_INSTRUCTIONS,
    "multihop": MULTIHOP_INSTRUCTIONS,
}


class ArkCliProvider:
    name = "arkcli"

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: int = 120,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        thinking: str = "disabled",
        prompt_preset: str = "multihop",
    ) -> None:
        if shutil.which("arkcli") is None:
            raise RuntimeError("arkcli binary not found")
        if prompt_preset not in INSTRUCTIONS_BY_PRESET:
            raise ValueError(f"unknown arkcli prompt preset: {prompt_preset}")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.thinking = thinking
        self.prompt_preset = prompt_preset

    def answer(self, *, query: str, hits: list[RetrievalHit]) -> ProviderResult:
        contexts = "\n\n".join(
            f"[{index}] source={hit.chunk.source} chunk={hit.chunk.id}\n{hit.chunk.text}"
            for index, hit in enumerate(hits, start=1)
        )
        instructions = INSTRUCTIONS_BY_PRESET[self.prompt_preset]
        prompt = (
            "请根据下面的上下文回答问题。\n\n"
            f"上下文:\n{contexts}\n\n"
            f"问题: {query}\n\n"
            "回答:"
        )
        cmd = [
            "arkcli",
            "+chat",
            "--format",
            "json",
            "--instructions",
            instructions,
            "--temperature",
            str(self.temperature),
            "--max-output-tokens",
            str(self.max_output_tokens),
            "--thinking",
            self.thinking,
        ]
        if self.model:
            cmd.extend(["--model", self.model])
        cmd.append(prompt)

        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())

        data = json.loads(proc.stdout)
        return ProviderResult(
            content=str(data.get("content", "")).strip(),
            model=str(data.get("model") or self.model or "arkcli-default"),
            usage=data.get("usage") or {},
            raw=data,
        )

    def judge_faithfulness(self, *, answer: str, contexts: list[str]) -> float:
        return lexical_faithfulness(answer, contexts)
