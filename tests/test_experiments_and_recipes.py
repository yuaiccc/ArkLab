import json
from pathlib import Path

from arklab.cli import main
from arklab.experiments import load_experiments
from arklab.models import DocumentChunk
from arklab.rag.pipeline import RagPipeline
from arklab.recipes import recipe_file_manifest


class BlockingProvider:
    name = "blocking"
    model = "blocking-model"

    def answer(self, *, query, hits):
        raise RuntimeError(
            '{"type":"content_moderation","message":"OutputTextSensitiveContentDetected"}'
        )

    def judge_faithfulness(self, *, answer, contexts):
        return 0.0


def test_provider_content_block_becomes_case_result() -> None:
    pipeline = RagPipeline(
        chunks=[DocumentChunk(id="doc#0", source="doc", text="alpha beta")],
        provider=BlockingProvider(),
        min_faithfulness=0.0,
    )

    result = pipeline.run("alpha?")

    assert result.abstained is True
    assert result.abstain_reason == "provider_content_block"
    assert result.provider.raw["error_type"] == "provider_content_block"


def test_eval_appends_experiment_registry(tmp_path: Path, capsys) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc.md").write_text("alpha beta", encoding="utf-8")
    eval_set = tmp_path / "eval.jsonl"
    eval_set.write_text(
        json.dumps({"query": "alpha?", "answer": "alpha", "relevant_ids": ["doc.md#0"]})
        + "\n",
        encoding="utf-8",
    )
    registry = tmp_path / "registry.jsonl"
    report = tmp_path / "report.json"

    exit_code = main(
        [
            "eval",
            "--docs",
            str(docs),
            "--eval-set",
            str(eval_set),
            "--output",
            str(report),
            "--failure-pool",
            "",
            "--trace",
            "",
            "--experiment-name",
            "unit",
            "--experiment-registry",
            str(registry),
            "--tag",
            "suite=unit",
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    rows = load_experiments(registry)
    assert len(rows) == 1
    assert rows[0]["name"] == "unit"
    assert rows[0]["summary"]["cases"] == 1
    assert rows[0]["tags"] == {"suite": "unit"}

    exit_code = main(["experiments", "--registry", str(registry), "--limit", "1"])
    listed = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert listed["experiments"] == 1


def test_yaml_recipe_matrix_expands_eval_commands(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        """
name: tiny
eval:
  docs: examples/docs
  eval_set: examples/evals/qa.jsonl
  provider: local
  failure_pool: ""
  experiment_registry: ""
matrix:
  retriever: [hybrid, ark-embedding]
  reranker: [none]
""".strip(),
        encoding="utf-8",
    )

    manifest = recipe_file_manifest(recipe)

    assert manifest["name"] == "tiny"
    assert len(manifest["commands"]) == 2
    assert manifest["commands"][0][:2] == ["arklab", "eval"]
    assert "--experiment-name" in manifest["commands"][0]
