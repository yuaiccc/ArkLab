from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                text_hash TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(*, model: str, dimensions: int, text: str) -> tuple[str, str]:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        raw_key = f"{model}\0{dimensions}\0{text_hash}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest(), text_hash

    def get(self, *, model: str, dimensions: int, text: str) -> list[float] | None:
        cache_key, _ = self.make_key(model=model, dimensions=dimensions, text=text)
        row = self.connection.execute(
            "SELECT vector_json FROM embeddings WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if not row:
            self.misses += 1
            return None
        self.hits += 1
        return [float(value) for value in json.loads(row[0])]

    def set(self, *, model: str, dimensions: int, text: str, vector: list[float]) -> None:
        cache_key, text_hash = self.make_key(model=model, dimensions=dimensions, text=text)
        self.connection.execute(
            """
            INSERT OR REPLACE INTO embeddings
              (cache_key, model, dimensions, text_hash, vector_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (cache_key, model, dimensions, text_hash, json.dumps(vector, separators=(",", ":"))),
        )
        self.connection.commit()

    def stats(self) -> dict[str, int | str]:
        return {
            "path": str(self.path),
            "hits": self.hits,
            "misses": self.misses,
        }
