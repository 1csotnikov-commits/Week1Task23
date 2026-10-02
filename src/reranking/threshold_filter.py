"""Пороговый фильтр: отсекает чанки с score ниже min_score (с min_keep)."""

from __future__ import annotations

from dataclasses import replace
from typing import List

from src.retrieval.models import RetrievedChunk
from src.reranking.base import Reranker


class ThresholdFilter(Reranker):
    """Фильтрация по порогу косинусной близости. Порядок не меняет."""

    name = "threshold"

    def __init__(self, min_score: float = 0.3, min_keep: int = 1):
        self.min_score = float(min_score)
        self.min_keep = int(min_keep)

    def rerank(self, chunks: List[RetrievedChunk], query: str) -> List[RetrievedChunk]:
        kept = [c for c in chunks if c.score >= self.min_score]
        if len(kept) < self.min_keep and chunks:
            # Оставляем хотя бы min_keep лучших (чанки уже отсортированы по score).
            kept = list(chunks[: self.min_keep])
        return [replace(c, rank=i + 1) for i, c in enumerate(kept)]
