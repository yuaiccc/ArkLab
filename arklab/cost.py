from __future__ import annotations

from typing import Any


INPUT_TOKEN_KEYS = ("input_tokens", "prompt_tokens", "inputTokenCount", "promptTokenCount")
OUTPUT_TOKEN_KEYS = ("output_tokens", "completion_tokens", "outputTokenCount", "completionTokenCount")
TOTAL_TOKEN_KEYS = ("total_tokens", "totalTokenCount")


def _first_number(payload: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    input_tokens = _first_number(usage, INPUT_TOKEN_KEYS)
    output_tokens = _first_number(usage, OUTPUT_TOKEN_KEYS)
    total_tokens = _first_number(usage, TOTAL_TOKEN_KEYS) or input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def summarize_usage(cases: list[dict[str, Any]]) -> dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for case in cases:
        usage = normalize_usage(case.get("provider", {}).get("usage"))
        for key in total:
            total[key] += usage[key]
    return total


def estimate_cost(
    usage: dict[str, int],
    *,
    input_price_per_1m: float = 0.0,
    output_price_per_1m: float = 0.0,
) -> dict[str, float]:
    input_cost = usage["input_tokens"] / 1_000_000 * input_price_per_1m
    output_cost = usage["output_tokens"] / 1_000_000 * output_price_per_1m
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "input_price_per_1m": input_price_per_1m,
        "output_price_per_1m": output_price_per_1m,
    }
