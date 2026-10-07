"""LLM usage helpers for reproducible cost reporting."""

from __future__ import annotations

from typing import Any


def usage_to_dict(usage: Any) -> dict[str, Any]:
    """Convert provider usage objects into plain JSON-friendly dictionaries."""

    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    if hasattr(usage, "model_dump"):
        value = usage.model_dump()
        return dict(value) if isinstance(value, dict) else {}
    return {
        key: getattr(usage, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if getattr(usage, key, None) is not None
    }


def usage_call(
    *,
    stage: str,
    provider: str,
    model: str,
    usage: Any,
    call_type: str = "chat_completion",
) -> dict[str, Any]:
    """Build one normalized LLM usage call record."""

    usage_dict = usage_to_dict(usage)
    prompt_tokens = _int_or_zero(usage_dict.get("prompt_tokens") or usage_dict.get("input_tokens"))
    completion_tokens = _int_or_zero(usage_dict.get("completion_tokens") or usage_dict.get("output_tokens"))
    total_tokens = _int_or_zero(usage_dict.get("total_tokens"))
    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "stage": stage,
        "provider": provider,
        "model": model,
        "call_type": call_type,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "raw_usage": usage_dict,
    }


def summarize_usage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate usage calls for experiment summaries."""

    provider_totals: dict[str, dict[str, Any]] = {}
    model_totals: dict[str, dict[str, Any]] = {}
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    for call in calls:
        prompt = _int_or_zero(call.get("prompt_tokens"))
        completion = _int_or_zero(call.get("completion_tokens"))
        tokens = _int_or_zero(call.get("total_tokens"))
        total_prompt += prompt
        total_completion += completion
        total_tokens += tokens
        _accumulate(provider_totals, str(call.get("provider") or "unknown"), prompt, completion, tokens)
        _accumulate(model_totals, str(call.get("model") or "unknown"), prompt, completion, tokens)
    return {
        "llm_call_count": len(calls),
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "by_provider": provider_totals,
        "by_model": model_totals,
        "calls": calls,
        "estimated_cost": {
            "available": False,
            "reason": "Provider pricing table is not configured; report token totals or apply dated pricing externally.",
        },
    }


def _accumulate(
    bucket: dict[str, dict[str, Any]],
    key: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> None:
    data = bucket.setdefault(
        key,
        {
            "llm_call_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )
    data["llm_call_count"] += 1
    data["prompt_tokens"] += prompt_tokens
    data["completion_tokens"] += completion_tokens
    data["total_tokens"] += total_tokens


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
