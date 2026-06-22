from __future__ import annotations

from typing import Protocol

from arklab.models import ProviderResult, RetrievalHit


class ModelProvider(Protocol):
    name: str

    def answer(self, *, query: str, hits: list[RetrievalHit]) -> ProviderResult:
        ...

    def judge_faithfulness(self, *, answer: str, contexts: list[str]) -> float:
        ...
