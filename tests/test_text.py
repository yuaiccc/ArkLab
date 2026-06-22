from pathlib import Path

from arklab.text import chunk_text, load_documents, tokenize


def test_tokenize_handles_english_digits_and_chinese() -> None:
    assert tokenize("ArkLab 2.0 支持RAG评测") == [
        "arklab",
        "2",
        "0",
        "支",
        "持",
        "rag",
        "评",
        "测",
    ]


def test_chunk_text_preserves_sentence_overlap() -> None:
    chunks = chunk_text(
        "alpha beta. gamma delta. epsilon zeta. eta theta.",
        max_tokens=4,
        overlap=2,
    )

    assert chunks == [
        "alpha beta.\ngamma delta.",
        "gamma delta.\nepsilon zeta.",
        "epsilon zeta.\neta theta.",
    ]


def test_load_documents_extracts_stable_doc_id(tmp_path: Path) -> None:
    docs = tmp_path / "docs" / "github"
    docs.mkdir(parents=True)
    (docs / "dsid_abc123__example.txt").write_text(
        "ArkLab evaluates RAG systems.\n\nIt records failure cases.",
        encoding="utf-8",
    )

    chunks = load_documents(tmp_path / "docs", max_tokens=20, overlap=0)

    assert len(chunks) == 1
    assert chunks[0].source == "github/dsid_abc123__example.txt"
    assert chunks[0].metadata["doc_id"] == "dsid_abc123"

