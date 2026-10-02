"""Абстракция реранкера (переранжирования результатов поиска)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.retrieval.models import RetrievedChunk


class Reranker(ABC):
    """Интерфейс переранжирования/фильтрации результатов retrieval.

    rerank(chunks, query) возвращает новый список (возможно, пересортированный
    и отфильтрованный), не изменяя входной.
    """

    name: str = "base"

    @abstractmethod
    def rerank(self, chunks: List[RetrievedChunk], query: str) -> List[RetrievedChunk]:
        raise NotImplementedError
