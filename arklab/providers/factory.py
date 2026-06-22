from __future__ import annotations

from typing import Any

from arklab.providers.arkcli import ArkCliProvider
from arklab.providers.base import ModelProvider
from arklab.providers.local import LocalHeuristicProvider


def create_provider(name: str, *, model: str | None = None, **kwargs: Any) -> ModelProvider:
    if name == "local":
        return LocalHeuristicProvider()
    if name == "arkcli":
        return ArkCliProvider(model=model, **kwargs)
    raise ValueError(f"unknown provider: {name}")
