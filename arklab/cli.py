from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

from arklab.benchmarks.enterprise_rag import import_enterprise_rag_bench
from arklab.benchmarks.multihop_rag import import_multihop_rag
from arklab.benchmarks.uaeval4rag import UA_CATEGORIES, generate_uaeval4rag
from arklab.cost import estimate_cost, normalize_usage, summarize_usage
from arklab.diagnostics import summarize_diagnostics
from arklab.evaluation.llm_judge import judge_cases_with_arkcli
from arklab.embeddings.ark import ArkEmbeddingClient
from arklab.evaluation.ragas_adapter import RagasCase, evaluate_with_ragas
from arklab.flywheel import compare_reports, promote_failures_to_eval_set, write_jsonl
from arklab.models import RetrievalHit
from arklab.providers.arkcli import INSTRUCTIONS_BY_PRESET
from arklab.providers.factory import create_provider
from arklab.rag.embedding_retrieval import ArkEmbeddingRetriever, ArkHybridRetriever
from arklab.rag.pipeline import RagPipeline
from arklab.rag.retrieval import HybridRetriever
from arklab.rag.rerank import create_reranker
from arklab.recipes import RECIPES, recipe_manifest, run_recipe
from arklab.reporting import export_report, trace_to_html
from arklab.text import load_documents, sentence_split
from arklab.trace.failure_pool import FailurePoolWriter, classify_failure
from arklab.trace.writer import TraceWriter
from arklab.trends import build_trend


MAX_EMBEDDING_WORKERS = 32


def _json_default(value: Any) -> Any:
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _retriever_info(pipeline: RagPipeline) -> dict[str, Any]:
    info: dict[str, Any] = {"name": pipeline.retriever.name}
    cache_stats = getattr(pipeline.retriever, "cache_stats", None)
    if cache_stats:
        info["cache"] = cache_stats
    return info


def _validate_embedding_workers(value: int) -> int:
    if value < 1:
        raise SystemExit("--embedding-workers must be >= 1")
    if value > MAX_EMBEDDING_WORKERS:
        raise SystemExit(
            f"--embedding-workers must be <= {MAX_EMBEDDING_WORKERS}; "
            "higher concurrency is likely to hit provider rate limits"
        )
    return value


def _build_pipeline(args: argparse.Namespace) -> RagPipeline:
    chunks = load_documents(Path(args.docs), max_tokens=args.chunk_tokens, overlap=args.chunk_overlap)
    if not chunks:
        raise SystemExit(f"no .txt/.md documents found under {args.docs}")
    provider_kwargs: dict[str, Any] = {}
    if args.provider == "arkcli":
        provider_kwargs = {
            "temperature": args.temperature,
            "prompt_preset": args.arkcli_prompt_preset,
        }
    provider = create_provider(args.provider, model=args.model, **provider_kwargs)
    reranker = create_reranker(args.reranker)
    if args.retriever in {"ark-embedding", "ark-hybrid"}:
        embedding_client = ArkEmbeddingClient(
            model=args.embedding_model,
            dimensions=args.embedding_dimensions,
            cache_path=Path(args.embedding_cache) if args.embedding_cache else None,
            max_retries=args.embedding_retries,
        )
        if args.retriever == "ark-hybrid":
            retriever = ArkHybridRetriever(
                chunks,
                client=embedding_client,
                embed_workers=_validate_embedding_workers(args.embedding_workers),
            )
        else:
            retriever = ArkEmbeddingRetriever(
                chunks,
                client=embedding_client,
                embed_workers=_validate_embedding_workers(args.embedding_workers),
            )
    else:
        retriever = HybridRetriever(chunks)
    trace_writer = TraceWriter(Path(args.trace)) if args.trace else None
    return RagPipeline(
        chunks=chunks,
        provider=provider,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        reranker=reranker,
        retriever=retriever,
        min_retrieval_score=args.min_retrieval_score,
        min_faithfulness=args.min_faithfulness,
        min_answer_relevancy=args.min_answer_relevancy,
        trace_writer=trace_writer,
    )


def cmd_query(args: argparse.Namespace) -> int:
    pipeline = _build_pipeline(args)
    result = pipeline.run(args.query)
    _print_json(
        {
            "query": result.query,
            "answer": result.answer,
            "abstained": result.abstained,
            "abstain_reason": result.abstain_reason,
            "metrics": result.metrics,
            "hits": [
                {
                    "rank": hit.rank,
                    "id": hit.chunk.id,
                    "source": hit.chunk.source,
                    "score": round(hit.score, 6),
                    "bm25_rank": hit.bm25_rank,
                    "dense_rank": hit.dense_rank,
                    "rerank_score": hit.rerank_score,
                    "preview": hit.chunk.text[:220],
                }
                for hit in result.hits
            ],
            "provider": {
                "model": result.provider.model,
                "usage": normalize_usage(result.provider.usage),
            },
            "retriever": _retriever_info(pipeline),
        }
    )
    return 0


def _load_eval_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def _relevance_from_row(row: dict[str, Any]) -> dict[str, float]:
    if "relevance" in row and isinstance(row["relevance"], dict):
        return {str(key): float(value) for key, value in row["relevance"].items()}
    relevant_ids = row.get("relevant_ids") or row.get("relevant_doc_ids") or []
    return {str(item): 1.0 for item in relevant_ids}


def _is_answerable(row: dict[str, Any]) -> bool:
    return row.get("answerable") is not False and row.get("expected_behavior") != "abstain"


def _mean_metric(cases: list[dict[str, Any]], key: str) -> float:
    values = [case[key] for case in cases if isinstance(case.get(key), (int, float))]
    return mean(values) if values else 0.0


def _expand_relevance(relevance: dict[str, float], pipeline: RagPipeline) -> dict[str, float]:
    expanded: dict[str, float] = {}
    chunk_ids = {chunk.id for chunk in pipeline.chunks}
    for key, value in relevance.items():
        if key in chunk_ids:
            expanded[key] = value
            continue
        matched = False
        for chunk in pipeline.chunks:
            if chunk.source == key or chunk.metadata.get("doc_id") == key:
                expanded[chunk.id] = value
                matched = True
        if not matched:
            expanded[key] = value
    return expanded


def _hit_keys(hit: RetrievalHit) -> list[str]:
    keys = [hit.chunk.id, hit.chunk.source]
    doc_id = hit.chunk.metadata.get("doc_id")
    if doc_id:
        keys.append(str(doc_id))
    return keys


def _matching_relevance_key(hit: RetrievalHit, relevance: dict[str, float], seen: set[str]) -> str | None:
    for key in _hit_keys(hit):
        if key in relevance and key not in seen:
            return key
    return None


def _recall_for_relevance(hits: list[RetrievalHit], relevance: dict[str, float]) -> float:
    if not relevance:
        return 1.0
    matched: set[str] = set()
    for hit in hits:
        for key in _hit_keys(hit):
            if key in relevance:
                matched.add(key)
    return len(matched) / len(relevance)


def _mrr_for_relevance(hits: list[RetrievalHit], relevance: dict[str, float]) -> float:
    for index, hit in enumerate(hits, start=1):
        if any(key in relevance for key in _hit_keys(hit)):
            return 1.0 / index
    return 0.0


def _ndcg_for_relevance(hits: list[RetrievalHit], relevance: dict[str, float]) -> float:
    if not relevance:
        return 1.0
    seen: set[str] = set()
    dcg = 0.0
    for index, hit in enumerate(hits, start=1):
        key = _matching_relevance_key(hit, relevance, seen)
        if key is None:
            continue
        seen.add(key)
        dcg += relevance[key] / math.log2(index + 1)

    ideal_values = sorted(relevance.values(), reverse=True)[: len(hits)]
    ideal = sum(value / math.log2(index + 1) for index, value in enumerate(ideal_values, start=1))
    return dcg / ideal if ideal else 0.0


def cmd_eval(args: argparse.Namespace) -> int:
    pipeline = _build_pipeline(args)
    rows = _load_eval_rows(Path(args.eval_set))
    if not rows:
        raise SystemExit(f"empty eval set: {args.eval_set}")

    per_case: list[dict[str, Any]] = []
    ragas_cases: list[RagasCase] = []
    failure_pool = FailurePoolWriter(Path(args.failure_pool)) if args.failure_pool else None
    for row in rows:
        query = str(row["query"])
        answerable = _is_answerable(row)
        relevance = _relevance_from_row(row)
        expanded_relevance = _expand_relevance(relevance, pipeline) if answerable else {}
        result = pipeline.run(query)
        top_hits = result.hits[: args.top_k]
        recall = _recall_for_relevance(top_hits, relevance) if answerable else None
        faithfulness = result.metrics["faithfulness"]
        case = {
            "query": query,
            "answerable": answerable,
            "expected_behavior": row.get("expected_behavior"),
            "unanswerable_category": row.get("unanswerable_category"),
            "answer": result.answer,
            "abstained": result.abstained,
            "abstain_reason": result.abstain_reason,
            "recall_at_k": recall,
            "mrr": _mrr_for_relevance(result.hits, relevance) if answerable else None,
            "ndcg_at_k": _ndcg_for_relevance(top_hits, relevance) if answerable else None,
            "faithfulness": faithfulness,
            "answer_relevancy": result.metrics["answer_relevancy"],
            "rejection_correct": (not answerable and result.abstained),
            "top_hit": result.hits[0].chunk.id if result.hits else None,
            "hit_ids": [hit.chunk.id for hit in result.hits],
            "provider": {
                "model": result.provider.model,
                "usage": normalize_usage(result.provider.usage),
            },
            "contexts": [hit.chunk.text for hit in result.hits],
        }
        per_case.append(case)
        ragas_cases.append(
            RagasCase(
                query=query,
                answer=result.answer,
                contexts=[hit.chunk.text for hit in result.hits],
                reference=row.get("answer") or row.get("reference"),
            )
        )
        failure_type = classify_failure(
            recall=recall,
            abstained=result.abstained,
            faithfulness=faithfulness,
            answerable=answerable,
        )
        if failure_pool and failure_type:
            failure_pool.write(
                {
                    "failure_type": failure_type,
                    "query": query,
                    "answerable": answerable,
                    "expected_behavior": row.get("expected_behavior"),
                    "unanswerable_category": row.get("unanswerable_category"),
                    "expected_relevance": expanded_relevance,
                    "answer": result.answer,
                    "abstained": result.abstained,
                    "abstain_reason": result.abstain_reason,
                    "metrics": case,
                }
            )

    answerable_cases = [case for case in per_case if case["answerable"]]
    unanswerable_cases = [case for case in per_case if not case["answerable"]]
    summary = {
        "cases": len(per_case),
        "answerable_cases": len(answerable_cases),
        "unanswerable_cases": len(unanswerable_cases),
        "recall_at_k": _mean_metric(answerable_cases, "recall_at_k"),
        "mrr": _mean_metric(answerable_cases, "mrr"),
        "ndcg_at_k": _mean_metric(answerable_cases, "ndcg_at_k"),
        "faithfulness": mean(case["faithfulness"] for case in per_case),
        "answer_relevancy": mean(case["answer_relevancy"] for case in per_case),
        "abstain_rate": mean(1.0 if case["abstained"] else 0.0 for case in per_case),
        "rejection_rate": (
            mean(1.0 if case["abstained"] else 0.0 for case in unanswerable_cases)
            if unanswerable_cases
            else 0.0
        ),
        "false_answer_rate": (
            mean(0.0 if case["abstained"] else 1.0 for case in unanswerable_cases)
            if unanswerable_cases
            else 0.0
        ),
    }
    usage = summarize_usage(per_case)
    payload: dict[str, Any] = {
        "summary": summary,
        "retriever": _retriever_info(pipeline),
        "usage": usage,
        "diagnostics": summarize_diagnostics(per_case),
        "cases": per_case,
    }
    if args.input_price_per_1m or args.output_price_per_1m:
        payload["cost"] = estimate_cost(
            usage,
            input_price_per_1m=args.input_price_per_1m,
            output_price_per_1m=args.output_price_per_1m,
        )
    if args.llm_judge == "arkcli":
        judged = judge_cases_with_arkcli(
            per_case,
            model=args.llm_judge_model or args.model,
            timeout=args.llm_judge_timeout,
        )
        for case, judge in zip(per_case, judged, strict=False):
            case["llm_judge"] = judge
        payload["llm_judge"] = {
            "provider": "arkcli",
            "model": args.llm_judge_model or args.model or "arkcli-default",
            "cases": judged,
            "usage": summarize_usage([{"provider": {"usage": item.get("usage", {})}} for item in judged]),
        }
    if args.evaluator == "ragas":
        ragas_report = evaluate_with_ragas(
            ragas_cases,
            judge_model=args.ragas_judge_model,
        )
        payload["ragas"] = ragas_report
    if args.output:
        _write_json(Path(args.output), payload)
    _print_json(payload)
    return 0


def cmd_trace_html(args: argparse.Namespace) -> int:
    payload = trace_to_html(Path(args.trace), Path(args.output))
    _print_json(payload)
    return 0


def cmd_trend(args: argparse.Namespace) -> int:
    payload = build_trend(args.reports)
    if args.output:
        _write_json(Path(args.output), payload)
    _print_json(payload)
    return 0


def cmd_recipe(args: argparse.Namespace) -> int:
    payload = run_recipe(args.name) if args.run else recipe_manifest(args.name)
    if args.output:
        _write_json(Path(args.output), payload)
    _print_json(payload)
    return 0


def cmd_export_report(args: argparse.Namespace) -> int:
    payload = export_report(Path(args.report), Path(args.output), fmt=args.format)
    _print_json(payload)
    return 0


def cmd_synth_qa(args: argparse.Namespace) -> int:
    chunks = load_documents(Path(args.docs), max_tokens=args.chunk_tokens, overlap=args.chunk_overlap)
    if not chunks:
        raise SystemExit(f"no .txt/.md documents found under {args.docs}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        clean_text = "\n".join(
            line for line in chunk.text.splitlines() if not line.lstrip().startswith("#")
        )
        sentences = [
            sentence.strip("# ").strip()
            for sentence in sentence_split(clean_text)
            if len(sentence.strip()) >= args.min_sentence_chars
        ]
        for sentence in sentences[: args.questions_per_chunk]:
            topic = sentence.split("，", 1)[0].split("。", 1)[0].strip()
            if " 是 " in sentence or "是一个" in sentence or "是一" in sentence:
                subject = topic.split("是", 1)[0].strip()
                query = f"{subject}是什么？" if subject else f"{topic}是什么意思？"
            elif "衡量" in sentence:
                subject = topic.split("衡量", 1)[0].strip()
                query = f"{subject}衡量什么？" if subject else f"{topic}是什么意思？"
            elif "触发" in sentence:
                subject = topic.split("会", 1)[0].strip()
                query = f"{subject}什么时候触发？" if subject else f"{topic}什么时候发生？"
            elif "Provider 层" in sentence and "支持" in sentence:
                query = "Provider 层支持哪些 provider？"
            elif "评测指标第一阶段包含" in sentence or "第一阶段包含" in sentence:
                query = "第一阶段评测指标包含哪些？"
            elif "JSONL trace" in sentence and "数据飞轮" in sentence:
                query = "JSONL trace 在数据飞轮里有什么作用？"
            elif "输入" in sentence or "进入" in sentence:
                query = f"{topic[:40]}有什么作用？"
            else:
                query = f"根据知识库说明，{topic[:40]}？"
            rows.append(
                {
                    "query": query,
                    "answer": sentence,
                    "relevant_ids": [chunk.id],
                    "source": chunk.source,
                }
            )

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _print_json({"output": str(output), "rows": len(rows)})
    return 0


def cmd_import_enterprise_rag_bench(args: argparse.Namespace) -> int:
    result = import_enterprise_rag_bench(
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
        sources=args.sources,
        question_types=args.question_types,
        max_questions=args.max_questions,
        max_docs_per_source=args.max_docs_per_source,
        include_unanswerable=args.include_unanswerable,
    )
    _print_json(
        {
            "output_dir": str(result.output_dir),
            "docs": str(result.docs_dir),
            "eval_set": str(result.eval_set),
            "questions": result.questions,
            "documents": result.docs,
            "sources": result.sources,
            "downloaded": result.downloaded,
        }
    )
    return 0


def cmd_import_multihop_rag(args: argparse.Namespace) -> int:
    result = import_multihop_rag(
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
        max_questions=args.max_questions,
        max_docs=args.max_docs,
        question_types=args.question_types,
    )
    _print_json(
        {
            "output_dir": str(result.output_dir),
            "docs": str(result.docs_dir),
            "eval_set": str(result.eval_set),
            "questions": result.questions,
            "documents": result.docs,
            "question_types": result.question_types,
            "downloaded": result.downloaded,
        }
    )
    return 0


def cmd_generate_uaeval4rag(args: argparse.Namespace) -> int:
    result = generate_uaeval4rag(
        docs=Path(args.docs),
        output_dir=Path(args.output_dir),
        max_questions=args.max_questions,
        categories=args.categories,
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
    )
    _print_json(
        {
            "output_dir": str(result.output_dir),
            "docs": str(result.docs_dir),
            "eval_set": str(result.eval_set),
            "questions": result.questions,
            "documents": result.docs,
            "categories": result.categories,
        }
    )
    return 0


def cmd_build_embeddings(args: argparse.Namespace) -> int:
    chunks = load_documents(Path(args.docs), max_tokens=args.chunk_tokens, overlap=args.chunk_overlap)
    if not chunks:
        raise SystemExit(f"no .txt/.md documents found under {args.docs}")
    embedding_client = ArkEmbeddingClient(
        model=args.embedding_model,
        dimensions=args.embedding_dimensions,
        cache_path=Path(args.embedding_cache) if args.embedding_cache else None,
        max_retries=args.embedding_retries,
    )
    started = time.perf_counter()
    embedding_client.embed_texts(
        [chunk.text for chunk in chunks],
        max_workers=_validate_embedding_workers(args.embedding_workers),
    )
    elapsed_seconds = time.perf_counter() - started
    _print_json(
        {
            "chunks": len(chunks),
            "embedding_model": args.embedding_model,
            "embedding_dimensions": args.embedding_dimensions,
            "embedding_workers": args.embedding_workers,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "cache": embedding_client.cache_stats(),
        }
    )
    return 0


def cmd_flywheel_next(args: argparse.Namespace) -> int:
    source_eval_set = Path(args.source_eval_set) if args.source_eval_set else None
    max_cases = args.max_cases if args.max_cases and args.max_cases > 0 else None
    rows, manifest = promote_failures_to_eval_set(
        failure_pool_path=Path(args.failure_pool),
        source_eval_set_path=source_eval_set,
        max_cases=max_cases,
    )
    output_eval_set = Path(args.output_eval_set)
    write_jsonl(output_eval_set, rows)
    manifest["output_eval_set"] = str(output_eval_set)
    if args.manifest:
        _write_json(Path(args.manifest), manifest)
    _print_json(manifest)
    return 0


def cmd_flywheel_compare(args: argparse.Namespace) -> int:
    focus_eval_set = Path(args.focus_eval_set) if args.focus_eval_set else None
    payload = compare_reports(
        baseline_report_path=Path(args.baseline_report),
        candidate_report_path=Path(args.candidate_report),
        focus_eval_set_path=focus_eval_set,
    )
    if args.output:
        _write_json(Path(args.output), payload)
    _print_json(payload)
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--docs", required=True, help="知识库目录或单个 .txt/.md 文件")
    parser.add_argument("--provider", choices=["local", "arkcli"], default="local")
    parser.add_argument("--model", default=None, help="arkcli 完整模型 ID 或 endpoint ID")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="arkcli provider 采样温度",
    )
    parser.add_argument(
        "--arkcli-prompt-preset",
        choices=sorted(INSTRUCTIONS_BY_PRESET),
        default="multihop",
        help="arkcli provider 的回答 prompt 版本，用于消融",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--retriever", choices=["hybrid", "ark-embedding", "ark-hybrid"], default="hybrid")
    parser.add_argument("--embedding-model", default="doubao-embedding-vision-251215")
    parser.add_argument("--embedding-dimensions", type=int, default=1024)
    parser.add_argument(
        "--embedding-cache",
        default="data/cache/embeddings/ark.sqlite",
        help="方舟 embedding 持久缓存路径；传空字符串可关闭",
    )
    parser.add_argument(
        "--embedding-workers",
        type=int,
        default=4,
        help="首次构建方舟 embedding 缓存时的并发请求数",
    )
    parser.add_argument("--embedding-retries", type=int, default=2, help="embedding 远程请求重试次数")
    parser.add_argument("--reranker", choices=["none", "lexical"], default="lexical")
    parser.add_argument("--chunk-tokens", type=int, default=180)
    parser.add_argument("--chunk-overlap", type=int, default=30)
    parser.add_argument("--min-retrieval-score", type=float, default=0.01)
    parser.add_argument("--min-faithfulness", type=float, default=0.35)
    parser.add_argument(
        "--min-answer-relevancy",
        type=float,
        default=0.0,
        help="答案与问题的最低词面相关度阈值；默认关闭，适合做保守 guardrail 消融",
    )
    parser.add_argument("--trace", default="data/traces/arklab.jsonl", help="JSONL trace 输出路径")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arklab", description="ArkLab RAG evaluation MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query", help="对知识库发起一次 RAG 查询")
    add_common_args(query_parser)
    query_parser.add_argument("query")
    query_parser.set_defaults(func=cmd_query)

    eval_parser = subparsers.add_parser("eval", help="用 JSONL 评测集跑基础指标")
    add_common_args(eval_parser)
    eval_parser.add_argument("--eval-set", required=True, help="JSONL: query + relevant_ids")
    eval_parser.add_argument("--evaluator", choices=["basic", "ragas"], default="basic")
    eval_parser.add_argument("--ragas-judge-model", default="doubao-seed-2-0-mini-260428")
    eval_parser.add_argument(
        "--llm-judge",
        choices=["none", "arkcli"],
        default="none",
        help="可选 LLM-as-Judge；arkcli 会额外调用一次方舟模型做结构化裁判",
    )
    eval_parser.add_argument("--llm-judge-model", default=None, help="LLM-as-Judge 使用的模型；默认复用 --model")
    eval_parser.add_argument("--llm-judge-timeout", type=int, default=120)
    eval_parser.add_argument(
        "--input-price-per-1m",
        type=float,
        default=0.0,
        help="输入 token 单价，用于估算本轮成本；单位：每百万 token",
    )
    eval_parser.add_argument(
        "--output-price-per-1m",
        type=float,
        default=0.0,
        help="输出 token 单价，用于估算本轮成本；单位：每百万 token",
    )
    eval_parser.add_argument("--output", default=None, help="把完整评测报告写入 JSON 文件")
    eval_parser.add_argument(
        "--failure-pool",
        default="data/failure_pool/arklab.jsonl",
        help="失败样本 JSONL；传空字符串可关闭",
    )
    eval_parser.set_defaults(func=cmd_eval)

    synth_parser = subparsers.add_parser("synth-qa", help="从知识库生成冷启动 JSONL 评测集")
    synth_parser.add_argument("--docs", required=True, help="知识库目录或单个 .txt/.md 文件")
    synth_parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    synth_parser.add_argument("--chunk-tokens", type=int, default=180)
    synth_parser.add_argument("--chunk-overlap", type=int, default=30)
    synth_parser.add_argument("--questions-per-chunk", type=int, default=2)
    synth_parser.add_argument("--min-sentence-chars", type=int, default=18)
    synth_parser.set_defaults(func=cmd_synth_qa)

    enterprise_parser = subparsers.add_parser(
        "import-enterprise-rag-bench",
        help="下载并转换 EnterpriseRAG-Bench 小样本",
    )
    enterprise_parser.add_argument(
        "--output-dir",
        default="benchmarks/enterprise_rag_bench",
        help="转换后的 ArkLab 数据目录",
    )
    enterprise_parser.add_argument(
        "--cache-dir",
        default="data/cache/enterprise_rag_bench",
        help="官方问题集和 zip 切片缓存目录",
    )
    enterprise_parser.add_argument(
        "--sources",
        nargs="+",
        default=["github"],
        help="要导入的来源类型，例如 github confluence jira",
    )
    enterprise_parser.add_argument(
        "--question-types",
        nargs="+",
        default=["basic"],
        help="要导入的问题类型，例如 basic semantic constrained",
    )
    enterprise_parser.add_argument("--max-questions", type=int, default=10)
    enterprise_parser.add_argument("--max-docs-per-source", type=int, default=500)
    enterprise_parser.add_argument(
        "--include-unanswerable",
        action="store_true",
        help="也导入没有 expected_doc_ids 的问题",
    )
    enterprise_parser.set_defaults(func=cmd_import_enterprise_rag_bench)

    multihop_parser = subparsers.add_parser(
        "import-multihop-rag",
        help="下载并转换 MultiHop-RAG 多跳问答数据集",
    )
    multihop_parser.add_argument(
        "--output-dir",
        default="benchmarks/multihop_rag/sample",
        help="转换后的 ArkLab 数据目录",
    )
    multihop_parser.add_argument(
        "--cache-dir",
        default="data/cache/multihop_rag",
        help="MultiHop-RAG 原始 JSON 缓存目录",
    )
    multihop_parser.add_argument("--max-questions", type=int, default=20)
    multihop_parser.add_argument("--max-docs", type=int, default=120)
    multihop_parser.add_argument(
        "--question-types",
        nargs="+",
        default=[],
        help="可选过滤，例如 inference_query comparison_query",
    )
    multihop_parser.set_defaults(func=cmd_import_multihop_rag)

    uaeval_parser = subparsers.add_parser(
        "generate-uaeval4rag",
        help="基于 UAEval4RAG 分类为本地知识库生成无答案评测集",
    )
    uaeval_parser.add_argument("--docs", required=True, help="用于生成无答案题的知识库目录")
    uaeval_parser.add_argument(
        "--output-dir",
        default="benchmarks/uaeval4rag/sample",
        help="输出 ArkLab 格式 docs/eval.jsonl",
    )
    uaeval_parser.add_argument("--max-questions", type=int, default=30)
    uaeval_parser.add_argument(
        "--categories",
        nargs="+",
        default=list(UA_CATEGORIES),
        choices=list(UA_CATEGORIES),
    )
    uaeval_parser.add_argument("--chunk-tokens", type=int, default=900)
    uaeval_parser.add_argument("--chunk-overlap", type=int, default=120)
    uaeval_parser.set_defaults(func=cmd_generate_uaeval4rag)

    build_embeddings_parser = subparsers.add_parser(
        "build-embeddings",
        help="为知识库预构建方舟 embedding 缓存",
    )
    build_embeddings_parser.add_argument("--docs", required=True, help="知识库目录或单个 .txt/.md 文件")
    build_embeddings_parser.add_argument("--embedding-model", default="doubao-embedding-vision-251215")
    build_embeddings_parser.add_argument("--embedding-dimensions", type=int, default=1024)
    build_embeddings_parser.add_argument(
        "--embedding-cache",
        default="data/cache/embeddings/ark.sqlite",
        help="方舟 embedding 持久缓存路径；传空字符串可关闭",
    )
    build_embeddings_parser.add_argument("--embedding-workers", type=int, default=4)
    build_embeddings_parser.add_argument("--embedding-retries", type=int, default=2)
    build_embeddings_parser.add_argument("--chunk-tokens", type=int, default=180)
    build_embeddings_parser.add_argument("--chunk-overlap", type=int, default=30)
    build_embeddings_parser.set_defaults(func=cmd_build_embeddings)

    flywheel_next_parser = subparsers.add_parser(
        "flywheel-next",
        help="把 failure pool 提升为下一轮回归评测集",
    )
    flywheel_next_parser.add_argument(
        "--failure-pool",
        default="data/failure_pool/arklab.jsonl",
        help="失败样本 JSONL",
    )
    flywheel_next_parser.add_argument(
        "--source-eval-set",
        default=None,
        help="原始评测集；用于补回标准答案和文档级 relevant_ids",
    )
    flywheel_next_parser.add_argument(
        "--output-eval-set",
        default="data/flywheel/next_eval.jsonl",
        help="输出的下一轮回归评测集 JSONL",
    )
    flywheel_next_parser.add_argument(
        "--manifest",
        default="data/flywheel/next_manifest.json",
        help="输出飞轮生成摘要 JSON；传空字符串可关闭",
    )
    flywheel_next_parser.add_argument("--max-cases", type=int, default=None)
    flywheel_next_parser.set_defaults(func=cmd_flywheel_next)

    flywheel_compare_parser = subparsers.add_parser(
        "flywheel-compare",
        help="比较 baseline/candidate 报告，判断失败样本是否被修复",
    )
    flywheel_compare_parser.add_argument("--baseline-report", required=True)
    flywheel_compare_parser.add_argument("--candidate-report", required=True)
    flywheel_compare_parser.add_argument(
        "--focus-eval-set",
        default=None,
        help="只比较某个飞轮评测集里的 query；不传则比较两个报告共同 query",
    )
    flywheel_compare_parser.add_argument(
        "--output",
        default="data/flywheel/compare.json",
        help="输出对比报告 JSON；传空字符串可关闭",
    )
    flywheel_compare_parser.set_defaults(func=cmd_flywheel_compare)

    trace_html_parser = subparsers.add_parser(
        "trace-html",
        help="把 JSONL trace 转成可本地打开的 HTML",
    )
    trace_html_parser.add_argument("--trace", default="data/traces/arklab.jsonl")
    trace_html_parser.add_argument("--output", default="data/reports/trace.html")
    trace_html_parser.set_defaults(func=cmd_trace_html)

    trend_parser = subparsers.add_parser(
        "trend",
        help="汇总多次 eval 报告，生成趋势 JSON",
    )
    trend_parser.add_argument("--reports", nargs="+", required=True, help="报告路径或 glob，例如 data/reports/*.json")
    trend_parser.add_argument("--output", default=None)
    trend_parser.set_defaults(func=cmd_trend)

    recipe_parser = subparsers.add_parser(
        "recipe",
        help="查看或运行内置 benchmark recipe",
    )
    recipe_parser.add_argument("--name", choices=sorted(RECIPES), required=True)
    recipe_parser.add_argument("--run", action="store_true", help="实际运行 recipe；不传则只打印命令清单")
    recipe_parser.add_argument("--output", default=None)
    recipe_parser.set_defaults(func=cmd_recipe)

    export_parser = subparsers.add_parser(
        "export-report",
        help="把 ArkLab 报告导出成其他评测/观测工具可消费的格式",
    )
    export_parser.add_argument("--report", required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--format", choices=["deepeval-json", "phoenix-jsonl"], required=True)
    export_parser.set_defaults(func=cmd_export_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"arklab: {exc}", file=sys.stderr)
        return 1
