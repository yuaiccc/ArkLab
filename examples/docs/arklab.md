# ArkLab MVP 说明

ArkLab 是一个基于 arkcli 的大模型实验与 Agentic RAG 评测平台。MVP 阶段优先跑通文本 RAG 评测链路，包括 provider 封装、Hybrid Retrieval、abstain 判断、基础指标和 JSON Lines trace。

Provider 层默认支持本地启发式 provider，也预留 arkcli provider。arkcli provider 通过 `arkcli +chat --format json` 调用方舟模型，返回字段中的 `content` 会被解析为助手回答。

检索层采用 BM25 和轻量 hashed dense retrieval 的混合方案，并通过 RRF 进行融合。这个实现不依赖外部向量数据库，适合在本地快速验证评测流程。

abstain 机制会在检索置信度过低或 faithfulness 低于阈值时触发，系统返回结构化的无法回答响应。每次请求都会写入 JSONL trace，包含 query、hit、score、answer、metrics 和 abstain reason。

评测指标第一阶段包含 Recall@K、MRR、faithfulness、answer relevancy 和 abstain rate。后续版本会增加 NDCG、LLM rerank、failure_pool、自动归因和 finetune 任务管理。
