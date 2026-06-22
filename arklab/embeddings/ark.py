from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
from typing import Any

import requests

from arklab.embeddings.cache import EmbeddingCache


class ArkEmbeddingClient:
    def __init__(
        self,
        *,
        model: str = "doubao-embedding-vision-251215",
        dimensions: int = 1024,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        timeout: int = 30,
        cache_path: Path | None = None,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.trust_env = False
        self.api_key = self._load_api_key()
        self.cache = EmbeddingCache(cache_path) if cache_path else None

    def embed_text(self, text: str) -> list[float]:
        if self.cache:
            cached = self.cache.get(model=self.model, dimensions=self.dimensions, text=text)
            if cached is not None:
                return cached
        vector = self._embed_text_remote(text)
        if self.cache:
            self.cache.set(model=self.model, dimensions=self.dimensions, text=text, vector=vector)
        return vector

    def embed_texts(self, texts: list[str], *, max_workers: int = 1) -> list[list[float]]:
        if max_workers <= 1 or len(texts) <= 1:
            return [self.embed_text(text) for text in texts]

        vectors: list[list[float] | None] = [None] * len(texts)
        missing: dict[str, str] = {}
        text_by_key: dict[str, str] = {}

        for index, text in enumerate(texts):
            if self.cache:
                cached = self.cache.get(model=self.model, dimensions=self.dimensions, text=text)
                if cached is not None:
                    vectors[index] = cached
                    continue
                cache_key, _text_hash = self.cache.make_key(
                    model=self.model,
                    dimensions=self.dimensions,
                    text=text,
                )
            else:
                cache_key = str(index)
            missing.setdefault(cache_key, text)
            text_by_key[cache_key] = text

        fetched: dict[str, list[float]] = {}
        if missing:
            workers = min(max_workers, len(missing))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._embed_text_remote, text): cache_key
                    for cache_key, text in missing.items()
                }
                for future in as_completed(futures):
                    cache_key = futures[future]
                    vector = future.result()
                    fetched[cache_key] = vector
                    if self.cache:
                        self.cache.set(
                            model=self.model,
                            dimensions=self.dimensions,
                            text=text_by_key[cache_key],
                            vector=vector,
                        )

        for index, text in enumerate(texts):
            if vectors[index] is not None:
                continue
            if self.cache:
                cache_key, _text_hash = self.cache.make_key(
                    model=self.model,
                    dimensions=self.dimensions,
                    text=text,
                )
            else:
                cache_key = str(index)
            vectors[index] = fetched[cache_key]

        if any(vector is None for vector in vectors):
            raise RuntimeError("embedding batch completed with missing vectors")
        return [vector for vector in vectors if vector is not None]

    def _embed_text_remote(self, text: str) -> list[float]:
        payload = {
            "model": self.model,
            "input": [{"type": "text", "text": text}],
            "dimensions": self.dimensions,
            "encoding_format": "float",
        }
        retryable_statuses = {429, 500, 502, 503, 504}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            session = requests.Session()
            session.trust_env = False
            try:
                response = session.post(
                    f"{self.base_url}/embeddings/multimodal",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                data = response.json()
                if response.status_code >= 400:
                    message = data.get("error") if isinstance(data, dict) else data
                    if response.status_code in retryable_statuses and attempt < self.max_retries:
                        time.sleep(0.5 * (2**attempt))
                        continue
                    raise RuntimeError(f"ark embedding request failed: {message}")
                embedding = self._extract_embedding(data)
                if not embedding:
                    raise RuntimeError("ark embedding response did not contain an embedding")
                return [float(value) for value in embedding]
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2**attempt))
                    continue
                raise RuntimeError(f"ark embedding request failed: {exc}") from exc
        if last_error:
            raise RuntimeError(f"ark embedding request failed: {last_error}") from last_error
        raise RuntimeError("ark embedding request failed")

    def cache_stats(self) -> dict[str, int | str] | None:
        return self.cache.stats() if self.cache else None

    @staticmethod
    def _extract_embedding(data: dict[str, Any]) -> list[float]:
        payload = data.get("data")
        if isinstance(payload, dict):
            embedding = payload.get("embedding")
            return embedding if isinstance(embedding, list) else []
        if isinstance(payload, list) and payload:
            embedding = payload[0].get("embedding") if isinstance(payload[0], dict) else None
            return embedding if isinstance(embedding, list) else []
        return []

    @staticmethod
    def _load_api_key() -> str:
        env_key = os.getenv("ARK_API_KEY") or os.getenv("VOLCENGINE_ARK_API_KEY")
        if env_key:
            return env_key

        identity_root = Path.home() / ".arkcli" / "identities"
        for path in sorted(identity_root.glob("*/apikey.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            api_key = data.get("api_key")
            if api_key:
                return str(api_key)

        raise RuntimeError("ARK API key not found; run `arkcli auth apikey` or set ARK_API_KEY")
