"""Query rewrite — переписывание запроса перед retrieval.

Применяется ДО retrieval: переписанный запрос эмбеддится и ищется в FAISS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.text_utils import tokenize

REWRITER_PROVIDERS = ("none", "heuristic", "llm")


@dataclass
class RewriteResult:
    """Результат переписывания запроса."""

    original_query: str
    rewritten_query: str
    method: str  # "none" | "heuristic" | "llm"
    meta: Dict[str, Any] = field(default_factory=dict)


class QueryRewriter:
    """Абстракция переписывания запроса."""

    name: str = "base"

    def rewrite(self, query: str) -> RewriteResult:
        raise NotImplementedError


class NoOpRewriter(QueryRewriter):
    """Возвращает запрос как есть."""

    name = "none"

    def rewrite(self, query: str) -> RewriteResult:
        return RewriteResult(
            original_query=query,
            rewritten_query=query,
            method="none",
            meta={"tokens": len(tokenize(query)), "elapsed_ms": 0},
        )


class HeuristicRewriter(QueryRewriter):
    """Расширяет запрос ключевыми словами (без стоп-слов)."""

    name = "heuristic"

    # Минимальный словарь синонимов (опционально, можно расширять).
    SYNONYMS: Dict[str, str] = {}

    def __init__(self, min_keywords: int = 3):
        self.min_keywords = int(min_keywords)

    def rewrite(self, query: str) -> RewriteResult:
        started = time.time()
        keywords = list(dict.fromkeys(tokenize(query)))  # без стоп-слов, без дублей
        if self.SYNONYMS:
            keywords += [self.SYNONYMS[w] for w in keywords if w in self.SYNONYMS]

        if len(keywords) < self.min_keywords:
            rewritten = query
        else:
            rewritten = f"{query} {' '.join(keywords)}"

        return RewriteResult(
            original_query=query,
            rewritten_query=rewritten,
            method="heuristic",
            meta={"tokens": len(keywords), "elapsed_ms": int((time.time() - started) * 1000)},
        )


class LLMRewriter(QueryRewriter):
    """Переписывает запрос через LLM (DeepSeek)."""

    name = "llm"

    PROMPT = (
        "Перепиши вопрос для поиска в базе документов. Сохрани смысл, "
        "добавь синонимы и ключевые термины. Верни только переписанный вопрос без пояснений."
    )

    def __init__(self, llm, temperature: float = 0.0, max_tokens: int = 256):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

    def rewrite(self, query: str) -> RewriteResult:
        started = time.time()
        response = self.llm.complete(
            [{"role": "system", "content": self.PROMPT}, {"role": "user", "content": query}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        rewritten = (response.text or "").strip() or query
        return RewriteResult(
            original_query=query,
            rewritten_query=rewritten,
            method="llm",
            meta={"tokens": response.completion_tokens, "elapsed_ms": int((time.time() - started) * 1000)},
        )


def make_rewriter(provider: str, config=None, llm=None) -> Optional[QueryRewriter]:
    """Создать переписыватель запроса по имени провайдера."""
    provider = (provider or "none").lower()
    if provider == "none":
        return None
    if provider == "heuristic":
        min_keywords = 3
        if config is not None:
            min_keywords = int(config.get("rewrite.heuristic.min_keywords", 3))
        return HeuristicRewriter(min_keywords=min_keywords)
    if provider == "llm":
        if llm is None:
            raise ValueError("LLMRewriter требует LLMProvider (передайте llm).")
        temperature = 0.0
        max_tokens = 256
        if config is not None:
            temperature = float(config.get("rewrite.llm.temperature", 0.0))
            max_tokens = int(config.get("rewrite.llm.max_tokens", 256))
        return LLMRewriter(llm, temperature=temperature, max_tokens=max_tokens)
    raise ValueError(f"Неизвестный rewriter: {provider!r}. Доступно: {', '.join(REWRITER_PROVIDERS)}")


__all__ = [
    "RewriteResult",
    "QueryRewriter",
    "NoOpRewriter",
    "HeuristicRewriter",
    "LLMRewriter",
    "make_rewriter",
    "REWRITER_PROVIDERS",
]
