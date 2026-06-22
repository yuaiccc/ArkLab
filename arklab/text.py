from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path

from arklab.models import DocumentChunk

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", re.UNICODE)
DOC_ID_RE = re.compile(r"(dsid_[A-Za-z0-9]+)")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？.!?])\s+|\n{2,}", text.strip())
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str, *, max_tokens: int = 180, overlap: int = 30) -> list[str]:
    sentences = sentence_split(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens: list[str] = []

    def flush() -> None:
        if current:
            chunks.append("\n".join(current).strip())

    for sentence in sentences:
        sentence_tokens = tokenize(sentence)
        if not sentence_tokens:
            continue
        if len(sentence_tokens) > max_tokens:
            flush()
            current = []
            current_tokens = []
            step = max(1, max_tokens - overlap)
            for start in range(0, len(sentence_tokens), step):
                part = sentence_tokens[start : start + max_tokens]
                if part:
                    chunks.append(" ".join(part))
            continue

        would_exceed = current and len(current_tokens) + len(sentence_tokens) > max_tokens
        if would_exceed:
            flush()
            overlap_tokens = current_tokens[-overlap:] if overlap > 0 else []
            current = []
            current_tokens = overlap_tokens[:]

        current.append(sentence)
        current_tokens.extend(sentence_tokens)

    flush()
    return chunks


def load_documents(path: Path, *, max_tokens: int = 180, overlap: int = 30) -> list[DocumentChunk]:
    if path.is_file():
        files = [path]
    else:
        files = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in {".txt", ".md", ".markdown"}
        )

    chunks: list[DocumentChunk] = []
    for file_path in files:
        raw = file_path.read_text(encoding="utf-8")
        doc_id_match = DOC_ID_RE.search(file_path.name)
        doc_id = doc_id_match.group(1) if doc_id_match else file_path.stem
        for index, text in enumerate(chunk_text(raw, max_tokens=max_tokens, overlap=overlap)):
            rel = str(file_path.relative_to(path)) if path.is_dir() else file_path.name
            chunks.append(
                DocumentChunk(
                    id=f"{rel}#{index}",
                    source=rel,
                    text=text,
                    metadata={"chunk_index": index, "path": str(file_path), "doc_id": doc_id},
                )
            )
    return chunks


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def hashed_vector(tokens: list[str], *, dims: int = 256) -> list[float]:
    counts = Counter(tokens)
    vector = [0.0] * dims
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        slot = int.from_bytes(digest, "big") % dims
        vector[slot] += float(count)
    return vector
