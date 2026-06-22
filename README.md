# ArkLab

ArkLab 是一个本地优先的 RAG 评测与诊断工具，用来回答一个很具体的问题：

> 一个知识库问答系统答错、拒答或幻觉时，到底是哪里出了问题？

它不是聊天机器人，也不是知识库 SaaS。当前版本刻意先做 CLI，把核心实验链路跑通：文档进入、检索、回答、评测、失败归因、回归测试。

## 项目定位

ArkLab 关注的是 RAG 系统的工程诊断，而不只是“调一次 API 看答案好不好”。

它把一次 RAG 实验拆成几层：

- 检索有没有找回正确材料；
- 正确材料有没有排到足够靠前；
- 模型是否基于材料回答；
- 模型是否过度保守而拒答；
- 模型是否在无答案问题上乱答；
- 改 prompt、embedding、reranker、chunk 参数或模型后，问题是否真的被修复。

因此 ArkLab 更准确的定位是：

```text
RAG 评测编排 + 失败诊断 + 回归飞轮
```

## 能做什么

ArkLab 可以把一个文档目录和一份 JSONL 评测集变成可复现的 RAG 实验。

当前已经支持：

- 本地离线 smoke test，不依赖任何模型 API；
- 通过 `arkcli` 调用真实方舟 / 豆包模型；
- 使用 BM25、本地 dense hashing、方舟 embedding 或混合检索；
- 计算 Recall@K、MRR、NDCG、faithfulness、answer relevancy、abstain rate、rejection rate、false-answer rate；
- 可选用方舟模型做 LLM-as-Judge，输出 faithfulness、relevancy、correctness 和失败原因；
- 统计模型 token usage，并可按输入 / 输出单价估算实验成本；
- 记录 JSONL trace 和 failure pool；
- 把 trace 转成可本地打开的 HTML；
- 把失败样本提升成下一轮回归评测集；
- 对比 baseline 和 candidate，判断哪些问题修复了、哪些退化了；
- 汇总多次实验报告，观察指标趋势；
- 导入或生成 EnterpriseRAG-Bench、MultiHop-RAG、无答案题等 benchmark 风格数据；
- 可选接入 RAGAS，复用标准 RAG 评测指标；
- 导出 DeepEval / Phoenix 形状的数据，方便接入现有评测和可观测生态。

## 为什么需要它

很多 RAG demo 只证明“这个问题看起来答对了”。但真实系统常见的问题更复杂：

- 该找的文档没有找回来；
- 找回来了，但排序太低；
- 证据已经在上下文里，模型还是拒答；
- 模型答得很流畅，但内容没有证据支撑；
- 问题本来就超出知识库范围，模型却硬答；
- 一个 prompt 改动修好了 A 问题，却弄坏了 B 问题。

ArkLab 的目标是把这些失败变成结构化数据，而不是靠印象判断“好像变好了”。

## 核心流程

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

这里的数据飞轮不是自动训练模型，而是把每一轮失败沉淀成下一轮必须复测的问题。后续无论改检索、prompt、阈值还是模型，都可以用同一批失败样本验证是否真的修复。

## 架构

```text
arklab/
  providers/      模型 provider：local heuristic、arkcli
  embeddings/     方舟 embedding client 和 SQLite 缓存
  rag/            检索、rerank、RAG pipeline
  evaluation/     基础指标和 RAGAS adapter
  benchmarks/     benchmark 导入与转换
  trace/          JSONL trace 和 failure pool writer
  flywheel.py     失败样本提升与前后对比
  cli.py          命令行入口
```

运行过程中产生的数据默认不进 git：

```text
data/             trace、报告、缓存、failure pool
benchmarks/       本地转换后的 benchmark 样本
```

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

跑一个离线评测：

```bash
arklab eval --docs examples/docs --eval-set examples/evals/qa.jsonl
```

对示例知识库问一个问题：

```bash
arklab query --docs examples/docs "ArkLab 的 abstain 机制什么时候触发？"
```

## 接真实模型

ArkLab 通过本机 `arkcli` profile 调用方舟模型。模型凭证从用户本机环境读取，不写入仓库。

```bash
arklab eval \
  --provider arkcli \
  --model doubao-seed-2-0-mini-260428 \
  --docs examples/docs \
  --eval-set examples/evals/qa.jsonl
```

如果要使用方舟 embedding 做真实向量检索：

```bash
arklab eval \
  --retriever ark-embedding \
  --reranker none \
  --provider arkcli \
  --model doubao-seed-2-0-mini-260428 \
  --docs examples/docs \
  --eval-set examples/evals/qa.jsonl
```

embedding 结果会默认写入本地 SQLite 缓存，避免重复为同一批 chunk 付费。

## 失败诊断

ArkLab 会把失败样本写入 failure pool，并在报告里给出 root cause：

- `retrieval_fail`：相关证据没有被检索到；
- `low_confidence_abstain`：检索置信度太低，不适合回答；
- `generation_fail_abstain`：证据存在，但模型仍然拒答；
- `generation_fail_hallucination`：回答缺少上下文支撑；
- `unanswerable_answered`：无答案问题被错误回答。

报告里的 `diagnostics` 字段会把这些样本进一步归因成 `retrieval_failure`、`over_abstention`、`unsupported_generation`、`off_topic_answer` 等，并给出下一步建议。需要更强裁判时，可以让方舟模型做 LLM-as-Judge：

```bash
arklab eval \
  --provider arkcli \
  --model doubao-seed-2-0-mini-260428 \
  --llm-judge arkcli \
  --docs examples/docs \
  --eval-set examples/evals/qa.jsonl
```

这些失败样本可以提升成下一轮回归评测集：

```bash
arklab flywheel-next \
  --failure-pool data/failure_pool/arklab.jsonl \
  --source-eval-set examples/evals/qa.jsonl \
  --output-eval-set data/flywheel/next_eval.jsonl
```

然后比较两个方案：

```bash
arklab flywheel-compare \
  --baseline-report data/reports/baseline.json \
  --candidate-report data/reports/candidate.json \
  --focus-eval-set data/flywheel/next_eval.jsonl
```

## 评测集格式

ArkLab 使用 JSON Lines。一个最小的可回答样本：

```json
{"query":"Recall@K 衡量什么？","answer":"Recall@K 衡量前 K 个检索结果是否覆盖相关文档。","relevant_ids":["rag_metrics.md"]}
```

一个无答案样本：

```json
{"query":"今天这家公司的实时股价是多少？","answerable":false,"expected_behavior":"abstain","relevant_ids":[]}
```

`relevant_ids` 可以指向 chunk ID、源文件名，或类似 `dsid_...` 的 benchmark 文档 ID。

## Benchmark 策略

当前项目采用实用路线：

- 用 `examples/` 做最小 smoke test；
- 用 EnterpriseRAG-Bench 做企业文档问答样本；
- 用 MultiHop-RAG 暴露跨文档综合失败；
- 生成无答案题测试拒答能力；
- 需要标准指标时接 RAGAS。

所有 benchmark 最终都会转换成同一套 ArkLab 格式：`docs/` 目录加 `eval.jsonl`。

常用 benchmark 可以通过 recipe 先查看、再运行：

```bash
arklab recipe --name enterprise-basic10
arklab recipe --name enterprise-basic10 --run
```

## Guardrail

pipeline 可以基于以下信号拒答：

- 检索置信度低；
- provider 主动拒答；
- faithfulness 低；
- answer relevancy 低。

`--min-answer-relevancy` 默认关闭，适合做更保守的消融实验，用来拦住“上下文是真的，但回答和问题不相关”的情况。

## 实验资产

一次 `eval` 会产出结构化报告。围绕这份报告，ArkLab 还能继续做几件事：

```bash
arklab trace-html --trace data/traces/arklab.jsonl --output data/reports/trace.html
arklab trend --reports 'data/reports/*.json' --output data/reports/trend.json
arklab export-report --report data/reports/baseline.json --format deepeval-json --output data/reports/deepeval.json
arklab export-report --report data/reports/baseline.json --format phoenix-jsonl --output data/reports/phoenix.jsonl
```

成本估算不直接查询账单，只按报告里的 token usage 和你传入的单价计算：

```bash
arklab eval \
  --docs examples/docs \
  --eval-set examples/evals/qa.jsonl \
  --input-price-per-1m 0.8 \
  --output-price-per-1m 2.0
```

## 开发

安装开发依赖：

```bash
pip install -e '.[dev]'
```

运行离线测试：

```bash
pytest
```

GitHub Actions 会在 Python 3.10、3.11、3.12 上跑离线测试，不需要方舟凭证。

## 当前边界

ArkLab 目前不做这些事：

- 不提供生产知识库服务；
- 不提供前端聊天 UI；
- 不自动训练或微调模型；
- 不保证回答一定正确；
- 不替代 RAGAS / DeepEval 这类完整评测库；
- 不做企业级权限、租户、监控、部署。

它现在最有价值的部分是：让 RAG 失败样本可记录、可归因、可复测。

## 项目状态

当前已经完成：

- CLI 评测主链路；
- 方舟真实模型接入；
- 方舟 embedding 检索；
- 可选 LLM-as-Judge；
- 失败 root cause 诊断；
- trace HTML、成本估算、多次实验趋势；
- failure pool 和回归飞轮；
- benchmark adapter；
- RAGAS / DeepEval / Phoenix 生态出口；
- 离线测试和 CI。

后续可以继续补：

- 更强的 reranker；
- prompt / retrieval / chunking 自动消融；
- 基于 CLI 产物的 dashboard。
