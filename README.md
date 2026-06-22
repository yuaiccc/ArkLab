# ArkLab

ArkLab is a local-first RAG evaluation workbench for finding out why a retrieval-augmented generation system fails.

It is built around a simple idea: a RAG system should not only return an answer, it should leave enough evidence to explain whether the problem came from retrieval, context construction, answer generation, abstention behavior, or the evaluation set itself.

The current project is intentionally CLI-first. It focuses on the technical core before adding a dashboard or hosted service.

## What ArkLab Does

ArkLab turns a folder of documents and a JSONL evaluation set into a reproducible RAG experiment.

It can:

- run local smoke tests without any model API;
- call real Ark/Doubao models through `arkcli`;
- use BM25, local dense hashing, Ark embedding, or hybrid retrieval;
- evaluate Recall@K, MRR, NDCG, faithfulness, answer relevancy, abstain rate, rejection rate, and false-answer rate;
- write traces and failure cases to JSONL;
- promote failures into a regression set for the next run;
- compare a baseline and a candidate run to see what was fixed or regressed;
- import or generate benchmark-style datasets such as EnterpriseRAG-Bench, MultiHop-RAG, and no-answer cases inspired by UAEval4RAG;
- plug into RAGAS for additional standard RAG metrics.

In plain terms, ArkLab is not just "calling an LLM API". It is an orchestration and diagnosis layer around RAG experiments.

## Why This Exists

Most RAG demos stop at "the answer looks right". Production RAG work usually fails in messier ways:

- the right document is not retrieved;
- the right document is retrieved but ranked too low;
- the model sees the evidence but still refuses to answer;
- the model answers confidently with unsupported content;
- the knowledge base genuinely cannot answer the question;
- a prompt or threshold change fixes one case but breaks another.

ArkLab keeps those cases as structured data, so improvements can be tested again instead of judged by memory or a single lucky prompt.

## Core Workflow

```text
documents + eval set
        |
        v
retrieval + rerank + answer generation
        |
        v
metrics + trace + failure pool
        |
        v
regression eval set
        |
        v
compare baseline vs candidate
```

This is the "data flywheel" in the current project. The flywheel does not fine-tune a model by itself. It makes failures reusable, so changes to retrieval, prompts, thresholds, or models can be evaluated against the exact cases they were supposed to fix.

## Architecture

```text
arklab/
  providers/      model providers: local heuristic, arkcli
  embeddings/     Ark embedding client and SQLite cache
  rag/            retrieval, rerank, RAG pipeline
  evaluation/     basic metrics and RAGAS adapter
  benchmarks/     benchmark import and conversion
  trace/          JSONL trace and failure-pool writers
  flywheel.py     failure promotion and run comparison
  cli.py          command-line interface
```

Runtime artifacts are intentionally kept out of git:

```text
data/             traces, reports, caches, failure pools
benchmarks/       locally converted benchmark samples
```

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run a local offline smoke test:

```bash
arklab eval --docs examples/docs --eval-set examples/evals/qa.jsonl
```

Ask one question against the example knowledge base:

```bash
arklab query --docs examples/docs "ArkLab 的 abstain 机制什么时候触发？"
```

## Running With Real Models

ArkLab can call Ark through the local `arkcli` profile. Model credentials are read from the user's Ark/arkcli environment and are not stored in this repository.

```bash
arklab eval \
  --provider arkcli \
  --model doubao-seed-2-0-mini-260428 \
  --docs examples/docs \
  --eval-set examples/evals/qa.jsonl
```

For retrieval, the strongest current local setup is usually Ark embedding plus a conservative generation prompt:

```bash
arklab eval \
  --retriever ark-embedding \
  --reranker none \
  --provider arkcli \
  --model doubao-seed-2-0-mini-260428 \
  --docs examples/docs \
  --eval-set examples/evals/qa.jsonl
```

Ark embedding results are cached in SQLite by default, so repeated experiments do not pay for the same chunks again.

## Diagnosis Features

ArkLab writes structured failure records when a case shows one of the expected failure modes:

- `retrieval_fail`: relevant evidence was not retrieved;
- `low_confidence_abstain`: retrieval was too weak to answer safely;
- `generation_fail_abstain`: evidence was present but the model still refused;
- `generation_fail_hallucination`: the answer was insufficiently supported;
- `unanswerable_answered`: an unanswerable question was answered anyway.

Those failure records can be promoted into a regression set:

```bash
arklab flywheel-next \
  --failure-pool data/failure_pool/arklab.jsonl \
  --source-eval-set examples/evals/qa.jsonl \
  --output-eval-set data/flywheel/next_eval.jsonl
```

After changing retrieval, prompts, or thresholds, compare the new run against the old one:

```bash
arklab flywheel-compare \
  --baseline-report data/reports/baseline.json \
  --candidate-report data/reports/candidate.json \
  --focus-eval-set data/flywheel/next_eval.jsonl
```

## Evaluation Data Format

ArkLab uses JSON Lines. A minimal answerable case looks like this:

```json
{"query":"Recall@K 衡量什么？","answer":"Recall@K 衡量前 K 个检索结果是否覆盖相关文档。","relevant_ids":["rag_metrics.md"]}
```

An unanswerable case looks like this:

```json
{"query":"今天这家公司的实时股价是多少？","answerable":false,"expected_behavior":"abstain","relevant_ids":[]}
```

`relevant_ids` can point to chunk IDs, source file names, or benchmark document IDs such as `dsid_...`.

## Benchmarks

ArkLab currently supports a pragmatic benchmark strategy:

- start with small local examples for smoke tests;
- import EnterpriseRAG-Bench slices for enterprise-style document QA;
- import MultiHop-RAG to expose cross-document synthesis failures;
- generate no-answer cases to test abstention behavior;
- use RAGAS when a standard evaluator is useful.

The benchmark importers output the same ArkLab format: a `docs/` directory plus an `eval.jsonl` file.

## Guardrails

The pipeline can abstain based on:

- low retrieval confidence;
- provider refusal;
- low faithfulness;
- low answer relevancy.

`--min-answer-relevancy` is optional and defaults to off. It is useful for conservative experiments where a retrieved context is true but not actually relevant to the question.

## Development

Install development dependencies:

```bash
pip install -e '.[dev]'
```

Run the offline test suite:

```bash
pytest
```

The GitHub Actions CI runs the offline suite on Python 3.10, 3.11, and 3.12. It does not require Ark credentials.

## Project Status

ArkLab is an early MVP. The useful center is already working:

- CLI evaluation loop;
- real Ark model integration;
- Ark embedding retrieval;
- failure-pool based regression loop;
- benchmark adapters;
- offline tests and CI.

Likely next steps:

- stronger reranking;
- LLM-as-judge based failure diagnosis;
- automatic prompt/retrieval ablation runs;
- richer trace inspection;
- eventually, a dashboard over the same CLI artifacts.
