from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from arklab.models import DocumentChunk, RetrievalHit


class JapaneseVerbMasterError(RuntimeError):
    """Raised when the japanese-verb-master target cannot be queried."""


@dataclass(frozen=True)
class SearchResult:
    hits: list[RetrievalHit]
    degraded: bool = False
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentResult:
    answer: str
    tool_calls: list[dict[str, Any]]
    raw: dict[str, Any] | None = None


class JapaneseVerbMasterClient:
    def __init__(self, base_url: str = "http://localhost:3456", *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def search(self, query: str, *, top_k: int = 5, level: str = "", category: str = "") -> SearchResult:
        payload = self._request_json(
            "GET",
            "/api/knowledge/search",
            params={
                "q": query,
                "topK": top_k,
                "level": level,
                "category": category,
            },
        )
        results = payload.get("results") or []
        hits = [search_item_to_hit(item, rank=index) for index, item in enumerate(results, start=1)]
        return SearchResult(hits=hits, degraded=bool(payload.get("degraded")), raw=payload)

    def agent_run(self, message: str, *, context: dict[str, Any] | None = None) -> AgentResult:
        payload = self._request_json(
            "POST",
            "/api/agent/run",
            json={"message": message, "context": context or {}},
        )
        return AgentResult(
            answer=str(payload.get("answer") or ""),
            tool_calls=list(payload.get("toolCalls") or []),
            raw=payload,
        )

    def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise JapaneseVerbMasterError(f"failed to call japanese-verb-master at {url}: {exc}") from exc
        except ValueError as exc:
            raise JapaneseVerbMasterError(f"invalid JSON response from japanese-verb-master at {url}") from exc
        if not isinstance(data, dict):
            raise JapaneseVerbMasterError(f"unexpected response shape from japanese-verb-master at {url}")
        return data


def search_item_keys(item: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    item_id = item.get("id")
    doc_id = item.get("docId") or item.get("doc_id")
    resource = item.get("resource")
    title = item.get("title")
    for value in (item_id, resource, doc_id, title):
        if value is not None:
            keys.append(str(value))
    if doc_id and title:
        keys.append(f"{doc_id}::{title}")
    return keys


def search_item_to_hit(item: dict[str, Any], *, rank: int) -> RetrievalHit:
    keys = search_item_keys(item)
    chunk_id = keys[0] if keys else f"jvm-hit-{rank}"
    source = str(item.get("resource") or item.get("docId") or item.get("doc_id") or chunk_id)
    title = str(item.get("title") or "")
    content = str(item.get("content") or "")
    text = f"{title}\n{content}".strip()
    score = float(item.get("score") or 0.0)
    chunk = DocumentChunk(
        id=chunk_id,
        source=source,
        text=text,
        metadata={
            "target": "japanese-verb-master",
            "match_keys": keys,
            "doc_id": item.get("docId") or item.get("doc_id"),
            "title": title,
            "level": item.get("level"),
            "category": item.get("category"),
            "raw": item,
        },
    )
    return RetrievalHit(chunk=chunk, score=score, rank=rank)

