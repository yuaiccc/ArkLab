from __future__ import annotations

import json
import re
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


RELEASE_API_URL = "https://api.github.com/repos/onyx-dot-app/EnterpriseRAG-Bench/releases/latest"
QUESTIONS_URL = "https://raw.githubusercontent.com/onyx-dot-app/EnterpriseRAG-Bench/main/questions.jsonl"
DOC_ID_RE = re.compile(r"(dsid_[A-Za-z0-9]+)")


@dataclass(frozen=True)
class EnterpriseImportResult:
    output_dir: Path
    docs_dir: Path
    eval_set: Path
    questions: int
    docs: int
    sources: list[str]
    downloaded: list[str]


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": "ArkLab/0.1"})


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(_request(url), timeout=120) as response:
        with path.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _read_json_url(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(_request(url), timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _source_match(row: dict[str, Any], sources: set[str]) -> bool:
    row_sources = {str(source) for source in row.get("source_types", [])}
    return not sources or bool(row_sources & sources)


def _type_match(row: dict[str, Any], question_types: set[str]) -> bool:
    return not question_types or str(row.get("question_type")) in question_types


def _select_questions(
    rows: Iterable[dict[str, Any]],
    *,
    sources: set[str],
    question_types: set[str],
    max_questions: int,
    include_unanswerable: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        expected_doc_ids = [str(item) for item in row.get("expected_doc_ids", [])]
        if not include_unanswerable and not expected_doc_ids:
            continue
        if not _source_match(row, sources) or not _type_match(row, question_types):
            continue
        selected.append(row)
        if len(selected) >= max_questions:
            break
    return selected


def _asset_map() -> dict[str, str]:
    release = _read_json_url(RELEASE_API_URL)
    return {
        str(asset["name"]): str(asset["browser_download_url"])
        for asset in release.get("assets", [])
        if "name" in asset and "browser_download_url" in asset
    }


def _assets_for_sources(assets: dict[str, str], sources: Iterable[str]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for source in sources:
        prefix = f"{source}_slice_"
        for name, url in assets.items():
            if name.startswith(prefix) and name.endswith(".zip"):
                selected[name] = url
    return dict(sorted(selected.items()))


def _doc_id_from_zip_name(name: str) -> str | None:
    match = DOC_ID_RE.search(Path(name).name)
    return match.group(1) if match else None


def _extract_docs(
    zip_paths: list[Path],
    docs_dir: Path,
    *,
    gold_doc_ids: set[str],
    max_docs_per_source: int,
) -> int:
    docs_dir.mkdir(parents=True, exist_ok=True)
    extracted_doc_ids: set[str] = set()
    sampled_by_source: dict[str, int] = {}

    for zip_path in zip_paths:
        source = zip_path.name.split("_slice_", 1)[0]
        sampled_by_source.setdefault(source, 0)
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".txt"):
                    continue
                doc_id = _doc_id_from_zip_name(member.filename)
                if not doc_id:
                    continue
                keep_gold = doc_id in gold_doc_ids
                keep_sample = sampled_by_source[source] < max_docs_per_source
                if not keep_gold and not keep_sample:
                    continue

                safe_name = Path(member.filename).name
                target = docs_dir / source / safe_name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(archive.read(member))
                extracted_doc_ids.add(doc_id)
                if keep_sample:
                    sampled_by_source[source] += 1

    missing = gold_doc_ids - extracted_doc_ids
    if missing:
        missing_preview = ", ".join(sorted(missing)[:8])
        raise RuntimeError(
            f"could not find {len(missing)} expected docs in downloaded slices: {missing_preview}"
        )
    return len(extracted_doc_ids)


def import_enterprise_rag_bench(
    *,
    output_dir: Path,
    cache_dir: Path,
    sources: list[str],
    question_types: list[str],
    max_questions: int,
    max_docs_per_source: int,
    include_unanswerable: bool = False,
) -> EnterpriseImportResult:
    normalized_sources = [source.strip().lower() for source in sources if source.strip()]
    source_filter = set(normalized_sources)
    type_filter = {item.strip().lower() for item in question_types if item.strip()}

    cache_dir.mkdir(parents=True, exist_ok=True)
    questions_path = cache_dir / "questions.jsonl"
    if not questions_path.exists():
        _download(QUESTIONS_URL, questions_path)

    questions = _select_questions(
        _read_questions(questions_path),
        sources=source_filter,
        question_types=type_filter,
        max_questions=max_questions,
        include_unanswerable=include_unanswerable,
    )
    if not questions:
        raise RuntimeError("no EnterpriseRAG-Bench questions matched the requested filters")

    effective_sources = sorted(
        {str(source) for row in questions for source in row.get("source_types", [])}
    )
    if not effective_sources:
        effective_sources = normalized_sources
    gold_doc_ids = {
        str(doc_id)
        for row in questions
        for doc_id in row.get("expected_doc_ids", [])
    }

    assets = _asset_map()
    source_assets = _assets_for_sources(assets, effective_sources)
    if not source_assets:
        raise RuntimeError(f"no release slice assets found for sources: {', '.join(effective_sources)}")

    downloaded: list[str] = []
    zip_paths: list[Path] = []
    for name, url in source_assets.items():
        path = cache_dir / name
        if not path.exists():
            _download(url, path)
            downloaded.append(name)
        zip_paths.append(path)

    docs_dir = output_dir / "docs"
    eval_set = output_dir / "eval.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_count = _extract_docs(
        zip_paths,
        docs_dir,
        gold_doc_ids=gold_doc_ids,
        max_docs_per_source=max_docs_per_source,
    )

    with eval_set.open("w", encoding="utf-8") as handle:
        for row in questions:
            out = {
                "query": row["question"],
                "answer": row.get("gold_answer"),
                "relevant_ids": row.get("expected_doc_ids", []),
                "question_id": row.get("question_id"),
                "question_type": row.get("question_type"),
                "source_types": row.get("source_types", []),
                "answer_facts": row.get("answer_facts", []),
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")

    return EnterpriseImportResult(
        output_dir=output_dir,
        docs_dir=docs_dir,
        eval_set=eval_set,
        questions=len(questions),
        docs=docs_count,
        sources=effective_sources,
        downloaded=downloaded,
    )
