# RAG 评测指标

Recall@K 衡量前 K 个检索结果中是否覆盖相关文档。它主要反映召回能力，需要评测集提供 relevant_ids。

MRR 衡量第一个相关文档出现的位置。相关文档越靠前，MRR 越高。

Faithfulness 衡量回答中的内容是否能被检索上下文支撑。ArkLab MVP 先使用词面重叠近似计算，后续会替换成 LLM-as-Judge。

Abstain rate 是 ArkLab 的一等指标。拒答率过低可能代表系统容易幻觉，拒答率过高可能代表系统过度保守。

JSONL trace 是后续数据飞轮的输入。低 faithfulness、错误召回和过度拒答的案例会进入 failure_pool，并被归因为 retrieval_fail、rerank_fail、generation_fail_hallucination、generation_fail_abstain 或 query_fail。
