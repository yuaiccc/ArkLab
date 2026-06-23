import json
from pathlib import Path

from arklab import cli
from arklab.cli import main


def test_import_jvm_golden_converts_expected_and_adversarial(tmp_path: Path, capsys) -> None:
    source = tmp_path / "knowledge-source"
    source.mkdir()
    (source / "golden-set.json").write_text(
        json.dumps(
            {
                "cases": [
                    {"query": "て形怎么变", "expected": "verb-conjugation::て形的变化规则"},
                    {
                        "query": "つもり怎么用",
                        "expected": ["intention::つもり", "contrast::つもり vs 予定"],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (source / "adversarial-set.json").write_text(
        json.dumps({"cases": [{"query": "东京天气怎么样"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    output = tmp_path / "jvm.jsonl"

    exit_code = main(
        [
            "import-jvm-golden",
            "--source-dir",
            str(source),
            "--output",
            str(output),
            "--include-adversarial",
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert exit_code == 0
    assert printed["rows"] == 3
    assert rows[0]["relevant_ids"] == ["verb-conjugation::て形的变化规则"]
    assert rows[1]["relevant_ids"] == ["intention::つもり", "contrast::つもり vs 予定"]
    assert rows[2]["answerable"] is False
    assert rows[2]["expected_behavior"] == "abstain"


def test_eval_jvm_uses_target_api_and_writes_report(tmp_path: Path, monkeypatch, capsys) -> None:
    class FakeJvmClient:
        def __init__(self, base_url: str, *, timeout: float) -> None:
            self.base_url = base_url
            self.timeout = timeout

        def search(self, query: str, *, top_k: int, level: str, category: str):
            from arklab.targets.jvm import SearchResult, search_item_to_hit

            item = {
                "id": 11,
                "docId": "verb-conjugation",
                "resource": "kb://grammar/verb-conjugation",
                "title": "て形的变化规则",
                "content": "五段动词按词尾音变生成て形。",
                "score": 0.9,
            }
            return SearchResult(hits=[search_item_to_hit(item, rank=1)], degraded=False, raw={})

    monkeypatch.setattr(cli, "JapaneseVerbMasterClient", FakeJvmClient)
    eval_set = tmp_path / "eval.jsonl"
    eval_set.write_text(
        json.dumps(
            {
                "query": "て形怎么变",
                "expected": "verb-conjugation::て形的变化规则",
                "relevant_ids": ["verb-conjugation::て形的变化规则"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "eval-jvm",
            "--eval-set",
            str(eval_set),
            "--base-url",
            "http://example.test",
            "--output",
            str(output),
            "--failure-pool",
            "",
            "--experiment-registry",
            "",
            "--trace",
            "",
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert printed["target"]["name"] == "japanese-verb-master"
    assert printed["summary"]["recall_at_k"] == 1.0
    assert printed["summary"]["mrr"] == 1.0
    assert printed["diagnostics"]["root_cause_counts"] == {"passed": 1}
    assert written["cases"][0]["hit_match_keys"][0]

