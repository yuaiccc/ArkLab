from __future__ import annotations

import re

from arklab.evaluation.metrics import lexical_faithfulness
from arklab.models import ProviderResult, RetrievalHit
from arklab.text import sentence_split, tokenize


class LocalHeuristicProvider:
    name = "local"

    def answer(self, *, query: str, hits: list[RetrievalHit]) -> ProviderResult:
        query_tokens = set(tokenize(query))
        candidates: list[tuple[float, str]] = []
        for hit in hits:
            for sentence in sentence_split(hit.chunk.text):
                sentence_tokens = set(tokenize(sentence))
                if not sentence_tokens:
                    continue
                overlap = len(query_tokens & sentence_tokens)
                candidates.append((overlap + hit.score, sentence))

        chosen = [
            sentence
            for _score, sentence in sorted(candidates, key=lambda item: item[0], reverse=True)[:3]
            if sentence
        ]
        if not chosen:
            return ProviderResult(
                content="无法基于当前知识库回答这个问题。",
                model=self.name,
                usage={"mode": "heuristic"},
            )

        answer = re.sub(r"\s+", " ", " ".join(chosen)).strip()
        return ProviderResult(content=answer, model=self.name, usage={"mode": "heuristic"})

    def judge_faithfulness(self, *, answer: str, contexts: list[str]) -> float:
        return lexical_faithfulness(answer, contexts)
