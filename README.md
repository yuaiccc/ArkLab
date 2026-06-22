# ArkLab

ArkLab 是一个基于 `arkcli` 的大模型实验与 Agentic RAG 评测平台。当前仓库先落地计划书里的 MVP：本地文本 RAG 评测 CLI，支持知识库查询、Hybrid Retrieval、abstain 判断、基础指标和 JSONL trace。

## 当前能力

- Provider 抽象：`local` 离线启发式 provider，`arkcli` 方舟 CLI provider。
- Retrieval：默认 BM25 + hashed dense + RRF，也支持 `ark-embedding` 真实向量检索，以及 `ark-hybrid` 的 BM25 + 方舟 embedding RRF 融合检索。
- Embedding cache：方舟 embedding 默认写入本地 SQLite 缓存，避免重复给同一批 chunk 调 embedding。
- Rerank：内置轻量 `lexical` reranker，可切换 `--reranker none` 做消融。
- 指标：Recall@K、MRR、NDCG@K、faithfulness、answer relevancy、abstain rate。
- Trace：每次 query 写入 `data/traces/arklab.jsonl`，便于后续 failure_pool 和数据飞轮接入。
- Failure pool：评测时自动把低召回、拒答、低 faithfulness 样本写入 JSONL。
- Flywheel：支持把 failure pool 提升为下一轮回归评测集，并比较 baseline/candidate 报告判断问题是否修复。
- Benchmark 导入：支持把 EnterpriseRAG-Bench 小样本转换成 ArkLab 的 docs/eval.jsonl 格式。
- 示例：内置 `examples/docs` 和 `examples/evals/qa.jsonl`，可直接 smoke test。

## 快速开始

```bash
cd ArkLab
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

如果要启用 RAGAS 标准评测，建议使用 Python 3.11/3.12 的虚拟环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[ragas]'
```

离线查询：

```bash
arklab query --docs examples/docs "ArkLab 的 abstain 机制什么时候触发？"
```

离线评测：

```bash
arklab eval --docs examples/docs --eval-set examples/evals/qa.jsonl
```

保存评测报告和失败样本池：

```bash
arklab eval \
  --docs examples/docs \
  --eval-set examples/evals/qa.jsonl \
  --output data/reports/example-eval.json \
  --failure-pool data/failure_pool/example.jsonl
```

生成冷启动评测集：

```bash
arklab synth-qa \
  --docs examples/docs \
  --output examples/evals/synth.jsonl
```

使用 arkcli provider：

```bash
arklab query \
  --provider arkcli \
  --model doubao-seed-1-6-251015 \
  --docs examples/docs \
  "ArkLab MVP 阶段优先跑通什么链路？"
```

如果当前 arkcli profile 已设置默认文本模型，也可以省略 `--model`。

使用方舟 embedding 做真实向量检索：

```bash
arklab query \
  --retriever ark-embedding \
  --reranker none \
  --provider arkcli \
  --model doubao-seed-2-0-mini-260428 \
  --docs examples/docs \
  "ArkLab 的 abstain 机制什么时候触发？"
```

真实 embedding + 真实模型评测：

```bash
arklab eval \
  --retriever ark-embedding \
  --reranker none \
  --provider arkcli \
  --model doubao-seed-2-0-mini-260428 \
  --docs examples/docs \
  --eval-set examples/evals/qa.jsonl \
  --output data/reports/ark-embedding-mini-eval.json
```

默认 embedding 模型是 `doubao-embedding-vision-251215`。它通过方舟 `/api/v3/embeddings/multimodal` 接口生成 1024 维向量，API Key 从本机 `arkcli` 配置或 `ARK_API_KEY` 环境变量读取，不会写入项目文件。

方舟 embedding 默认缓存到 `data/cache/embeddings/ark.sqlite`。如果要临时关闭缓存，可以传空字符串：

```bash
arklab eval \
  --retriever ark-embedding \
  --embedding-cache "" \
  --docs examples/docs \
  --eval-set examples/evals/qa.jsonl
```

可以提前为知识库预热 embedding 缓存。首次构建时会并发请求方舟 embedding；后续 query/eval 会直接复用本地 SQLite：

```bash
arklab build-embeddings \
  --docs benchmarks/enterprise_rag_bench/github_basic_10/docs \
  --chunk-tokens 900 \
  --chunk-overlap 120 \
  --embedding-workers 8
```

使用真实 embedding + BM25 融合检索：

```bash
arklab eval \
  --retriever ark-hybrid \
  --embedding-workers 8 \
  --reranker none \
  --provider arkcli \
  --model doubao-seed-2-0-mini-260428 \
  --docs examples/docs \
  --eval-set examples/evals/qa.jsonl
```

使用 RAGAS 做标准 RAG 打分：

```bash
arklab eval \
  --evaluator ragas \
  --retriever ark-embedding \
  --reranker none \
  --provider arkcli \
  --model doubao-seed-2-0-mini-260428 \
  --docs examples/docs \
  --eval-set examples/evals/synth.jsonl \
  --output data/reports/ragas-ark-eval.json
```

RAGAS 会额外输出 `faithfulness`、`answer_relevancy`、`context_precision`、`context_recall`。其中 `context_precision/context_recall` 需要评测集里有 `answer` 或 `reference` 标准答案字段；没有标准答案时只跑不依赖 reference 的指标。

## 使用 EnterpriseRAG-Bench

先导入一个可控小样本。默认会从官方 release 下载 `github` 来源切片，抽取 10 个 `basic` 问题、最多 500 篇干扰文档，并保证 gold 文档在本地：

```bash
arklab import-enterprise-rag-bench \
  --sources github \
  --question-types basic \
  --max-questions 10 \
  --max-docs-per-source 500 \
  --output-dir benchmarks/enterprise_rag_bench/github_basic_10
```

离线验证链路：

```bash
arklab eval \
  --docs benchmarks/enterprise_rag_bench/github_basic_10/docs \
  --eval-set benchmarks/enterprise_rag_bench/github_basic_10/eval.jsonl \
  --top-k 5 \
  --output data/reports/enterprise-github-basic10-local.json
```

调用方舟真实模型评测同一批问题：

```bash
arklab eval \
  --provider arkcli \
  --model doubao-seed-2-0-mini-260428 \
  --docs benchmarks/enterprise_rag_bench/github_basic_10/docs \
  --eval-set benchmarks/enterprise_rag_bench/github_basic_10/eval.jsonl \
  --top-k 5 \
  --output data/reports/enterprise-github-basic10-arkcli-hybrid.json
```

EnterpriseRAG-Bench 的 gold relevance 是 `dsid_...` 文档级 ID。ArkLab 会把命中该文档任意 chunk 视为召回该文档，因此这里的 Recall@K 是文档级口径，不是 chunk 级口径。

## 接入更难的 RAG 问题

方舟 embedding 可以作为默认强检索底座。为了继续暴露生产问题，ArkLab 还支持两类更难 benchmark。

MultiHop-RAG 用来测试跨文档多跳问题。原始数据来自 Git LFS，ArkLab 通过 GitHub media URL 下载，不要求本机安装 `git-lfs`：

```bash
arklab import-multihop-rag \
  --max-questions 20 \
  --max-docs 120 \
  --output-dir benchmarks/multihop_rag/sample
```

输出仍然是标准 ArkLab 格式：

```text
benchmarks/multihop_rag/sample/docs
benchmarks/multihop_rag/sample/eval.jsonl
```

UAEval4RAG 用来测试“知识库里答不了时是否会拒答”。上游项目是无答案题生成框架；ArkLab 当前先接入它的分类法，基于本地知识库生成 `answerable=false` 的无答案回归题：

```bash
arklab generate-uaeval4rag \
  --docs benchmarks/enterprise_rag_bench/github_basic_10/docs \
  --max-questions 30 \
  --output-dir benchmarks/uaeval4rag/enterprise_noanswer_30
```

无答案评测集会额外带：

```json
{"answerable": false, "expected_behavior": "abstain", "unanswerable_category": "false_presuppositions"}
```

`arklab eval` 会自动统计：

- `rejection_rate`：无答案题里正确拒答的比例。
- `false_answer_rate`：无答案题里不该答却回答了的比例。

## 自我改进飞轮

ArkLab 的飞轮不是训练模型本身，而是把每轮评测失败沉淀成下一轮必须回归的测试集：

```text
eval -> failure_pool -> flywheel-next -> next eval set -> 改检索/Prompt/模型 -> eval -> flywheel-compare
```

第一步，正常跑评测并保留失败池：

```bash
arklab eval \
  --provider arkcli \
  --model doubao-seed-2-0-mini-260428 \
  --docs benchmarks/enterprise_rag_bench/github_basic_10/docs \
  --eval-set benchmarks/enterprise_rag_bench/github_basic_10/eval.jsonl \
  --output data/reports/enterprise-github-basic10-arkcli-hybrid.json \
  --failure-pool data/failure_pool/arklab.jsonl
```

第二步，把失败池提升为下一轮回归评测集。`--source-eval-set` 用来补回标准答案和文档级 `relevant_ids`：

```bash
arklab flywheel-next \
  --failure-pool data/failure_pool/arklab.jsonl \
  --source-eval-set benchmarks/enterprise_rag_bench/github_basic_10/eval.jsonl \
  --output-eval-set data/flywheel/enterprise-next-eval.jsonl \
  --manifest data/flywheel/enterprise-next-manifest.json
```

第三步，换一个方案重新评测，然后比较修复前后。比如用真实方舟 embedding 检索作为 candidate：

```bash
arklab eval \
  --retriever ark-embedding \
  --reranker none \
  --provider arkcli \
  --model doubao-seed-2-0-mini-260428 \
  --docs benchmarks/enterprise_rag_bench/github_basic_10/docs \
  --eval-set benchmarks/enterprise_rag_bench/github_basic_10/eval.jsonl \
  --output data/reports/enterprise-github-basic10-ark-embedding-arkcli.json
```

只看失败池提升出来的问题是否被修复：

```bash
arklab flywheel-compare \
  --baseline-report data/reports/enterprise-github-basic10-arkcli-hybrid.json \
  --candidate-report data/reports/enterprise-github-basic10-ark-embedding-arkcli.json \
  --focus-eval-set data/flywheel/enterprise-next-eval.jsonl \
  --output data/flywheel/enterprise-compare.json
```

`flywheel-next` 会保留原始失败类型、上一次错误答案、旧指标和建议动作。当前内置动作包括：

- `retrieval_fail` -> 优先改检索、chunking、query rewrite 或 rerank。
- `low_confidence_abstain` / `generation_fail_abstain` -> 优先改 prompt、上下文组织或回答阈值。
- `generation_fail_hallucination` -> 优先加 faithfulness guardrail。

## 评测集格式

评测集使用 JSON Lines，每行至少包含：

```json
{"query":"Recall@K 衡量什么？","relevant_ids":["rag_metrics.md"]}
```

`relevant_ids` 可以写 chunk id，例如 `rag_metrics.md#0`；也可以写源文件名，例如 `rag_metrics.md`。对于 EnterpriseRAG-Bench 这类文件名前缀带 `dsid_...` 的数据，也可以直接写文档 ID。

如果需要 NDCG 的分级相关性，可以使用 `relevance`：

```json
{"query":"Recall@K 衡量什么？","relevance":{"rag_metrics.md":2,"arklab.md":1}}
```

## 目录结构

```text
ArkLab/
  arklab/
    providers/      # local / arkcli provider
    rag/            # retrieval + pipeline
    evaluation/     # metrics
    flywheel.py     # failure pool -> 回归集 + 前后对比
    trace/          # JSONL trace / failure pool writer
    cli.py          # arklab query / eval / synth-qa
  examples/
    docs/           # 示例知识库
    evals/          # 示例评测集
  benchmarks/       # 本地转换后的 benchmark 样本，默认不提交
  data/
    cache/          # 官方 benchmark zip/JSONL 缓存，默认不提交
    traces/         # 运行后生成 JSONL trace
    failure_pool/   # 评测后生成失败样本池
    flywheel/       # 飞轮生成的回归集、manifest、compare 报告
```

## 下一步

- 增加 Cross-Encoder rerank。
- 引入更强的 LLM-as-Judge faithfulness 和失败根因诊断。
- 增加自动 query rewrite / chunking 消融，把候选修复方案批量送进 `flywheel-compare`。
- 增加 Dashboard 展示指标趋势和 trace 明细。
- 接入 finetune 任务管理，形成数据飞轮闭环。
