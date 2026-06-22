from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CORPUS_URL = "https://media.githubusercontent.com/media/yixuantt/MultiHop-RAG/main/dataset/corpus.json"
QA_URL = "https://media.githubusercontent.com/media/yixuantt/MultiHop-RAG/main/dataset/MultiHopRAG.json"


@dataclass(frozen=True)
class MultiHopImportResult:
    output_dir: Path
    docs_dir: Path
    eval_set: Path
    questions: int
    docs: int
    question_types: list[str]
    downloaded: list[str]


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": "ArkLab/0.1"})


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(_request(url), timeout=180) as response:
        with path.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _doc_key(row: dict[str, Any]) -> str:
    key = str(row.get("url") or row.get("title") or json.dumps(row, sort_keys=True))
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"multihop_{digest}"


def _doc_text(row: dict[str, Any]) -> str:
    fields = [
        ("title", row.get("title")),
        ("source", row.get("source")),
        ("author", row.get("author")),
        ("category", row.get("category")),
        ("published_at", row.get("published_at")),
        ("url", row.get("url")),
    ]
    header = "\n".join(f"{key}: {value}" for key, value in fields if value)
    body = str(row.get("body") or "")
    return f"{header}\n\n{body}".strip()


def _select_questions(
    rows: list[dict[str, Any]],
    *,
    question_types: set[str],
    max_questions: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        question_type = str(row.get("question_type") or "")
        if question_types and question_type not in question_types:
            continue
        selected.append(row)
        if len(selected) >= max_questions:
            break
    return selected


def import_multihop_rag(
    *,
    output_dir: Path,
    cache_dir: Path,
    max_questions: int,
    max_docs: int,
    question_types: list[str],
) -> MultiHopImportResult:
    cache_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = cache_dir / "corpus.json"
    qa_path = cache_dir / "MultiHopRAG.json"
    downloaded: list[str] = []
    if not corpus_path.exists():
        _download(CORPUS_URL, corpus_path)
        downloaded.append(corpus_path.name)
    if not qa_path.exists():
        _download(QA_URL, qa_path)
        downloaded.append(qa_path.name)

    corpus = _load_json(corpus_path)
    qa_rows = _load_json(qa_path)
    type_filter = {item.strip() for item in question_types if item.strip()}
    selected = _select_questions(qa_rows, question_types=type_filter, max_questions=max_questions)
    if not selected:
        raise RuntimeError("no MultiHop-RAG questions matched the requested filters")

    corpus_by_key = {_doc_key(row): row for row in corpus}
    corpus_by_title = {str(row.get("title")): row for row in corpus if row.get("title")}
    corpus_by_url = {str(row.get("url")): row for row in corpus if row.get("url")}

    gold_doc_ids: set[str] = set()
    selected_doc_ids: list[str] = []
    missing_evidence: list[str] = []
    for row in selected:
        for evidence in row.get("evidence_list", []):
            source = corpus_by_url.get(str(evidence.get("url"))) or corpus_by_title.get(
                str(evidence.get("title"))
            )
            if source is None:
                missing_evidence.append(str(evidence.get("title") or evidence.get("url")))
                continue
            doc_id = _doc_key(source)
            gold_doc_ids.add(doc_id)
            if doc_id not in selected_doc_ids:
                selected_doc_ids.append(doc_id)

    for row in corpus:
        if len(selected_doc_ids) >= max(max_docs, len(gold_doc_ids)):
            break
        doc_id = _doc_key(row)
        if doc_id not in selected_doc_ids:
            selected_doc_ids.append(doc_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = output_dir / "docs"
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    for doc_id in selected_doc_ids:
        row = corpus_by_key[doc_id]
        path = docs_dir / f"{doc_id}.txt"
        path.write_text(_doc_text(row), encoding="utf-8")

    eval_set = output_dir / "eval.jsonl"
    with eval_set.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(selected, start=1):
            relevant_ids: list[str] = []
            evidence_facts: list[str] = []
            for evidence in row.get("evidence_list", []):
                source = corpus_by_url.get(str(evidence.get("url"))) or corpus_by_title.get(
                    str(evidence.get("title"))
                )
                if source is None:
                    continue
                doc_id = _doc_key(source)
                if doc_id not in relevant_ids:
                    relevant_ids.append(doc_id)
                if evidence.get("fact"):
                    evidence_facts.append(str(evidence["fact"]))
            out = {
                "query": row["query"],
                "answer": row.get("answer"),
                "answerable": True,
                "relevant_ids": relevant_ids,
                "question_id": f"multihop_{index}",
                "question_type": row.get("question_type"),
                "evidence_facts": evidence_facts,
                "benchmark": "MultiHop-RAG",
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")

    if missing_evidence:
        preview = ", ".join(missing_evidence[:5])
        raise RuntimeError(f"missing {len(missing_evidence)} evidence docs in corpus: {preview}")

    return MultiHopImportResult(
        output_dir=output_dir,
        docs_dir=docs_dir,
        eval_set=eval_set,
        questions=len(selected),
        docs=len(selected_doc_ids),
        question_types=sorted({str(row.get("question_type")) for row in selected}),
        downloaded=downloaded,
    )
