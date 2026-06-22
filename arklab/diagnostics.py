from __future__ import annotations

from collections import Counter
from typing import Any


def diagnose_case(case: dict[str, Any]) -> dict[str, Any]:
    answerable = case.get("answerable") is not False
    recall = float(case.get("recall_at_k") or 0.0)
    faithfulness = float(case.get("faithfulness") or 0.0)
    answer_relevancy = float(case.get("answer_relevancy") or 0.0)
    abstained = bool(case.get("abstained"))
    reason = case.get("abstain_reason")

    if not answerable and abstained:
        root_cause = "correct_abstention"
        action = "keep_abstention_policy"
    elif not answerable:
        root_cause = "unanswerable_answered"
        action = "tighten_abstention_guardrail"
    elif recall <= 0.0:
        root_cause = "retrieval_failure"
        action = "improve_index_chunking_embedding_or_query_rewrite"
    elif abstained and reason == "provider_abstain":
        root_cause = "over_abstention"
        action = "adjust_prompt_context_packaging_or_answerability_threshold"
    elif abstained and reason == "low_answer_relevancy":
        root_cause = "answer_relevancy_guardrail"
        action = "inspect_query_context_mismatch_or_lower_threshold"
    elif abstained:
        root_cause = "low_confidence_or_low_faithfulness"
        action = "inspect_retrieved_context_and_faithfulness_threshold"
    elif faithfulness < 0.35:
        root_cause = "unsupported_generation"
        action = "tighten_prompt_or_add_llm_judge_guardrail"
    elif answer_relevancy < 0.2:
        root_cause = "off_topic_answer"
        action = "improve_prompt_or_answer_relevancy_guardrail"
    else:
        root_cause = "passed"
        action = "no_action"

    return {
        "root_cause": root_cause,
        "suggested_action": action,
        "signals": {
            "answerable": answerable,
            "recall_at_k": recall,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "abstained": abstained,
            "abstain_reason": reason,
            "top_hit": case.get("top_hit"),
        },
    }


def summarize_diagnostics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    diagnosed = [diagnose_case(case) for case in cases]
    root_counts = Counter(item["root_cause"] for item in diagnosed)
    action_counts = Counter(item["suggested_action"] for item in diagnosed)
    return {
        "root_cause_counts": dict(root_counts),
        "suggested_action_counts": dict(action_counts),
        "cases": [
            {
                "query": case.get("query"),
                **diagnosis,
            }
            for case, diagnosis in zip(cases, diagnosed, strict=False)
        ],
    }
