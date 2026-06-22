import json
from pathlib import Path

from arklab.cli import main


def test_eval_cli_writes_basic_report(tmp_path: Path, capsys) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "arklab.md").write_text(
        "ArkLab 通过 failure pool 把失败样本提升为回归评测集。",
        encoding="utf-8",
    )
    eval_set = tmp_path / "eval.jsonl"
    eval_set.write_text(
        json.dumps(
            {
                "query": "ArkLab 如何使用 failure pool？",
                "answer": "把失败样本提升为回归评测集。",
                "relevant_ids": ["arklab.md#0"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "eval",
            "--docs",
            str(docs),
            "--eval-set",
            str(eval_set),
            "--output",
            str(output),
            "--failure-pool",
            "",
            "--trace",
            "",
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert printed["summary"]["cases"] == 1
    assert written["summary"]["recall_at_k"] == 1.0

