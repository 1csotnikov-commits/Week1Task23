"""RerankerPipeline: применяет реранкер к результатам retrieval и логирует."""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.retrieval.models import RetrievedChunk
from src.reranking.base import Reranker


class RerankerPipeline:
    """Обёртка над Reranker: фильтрация + пересортировка + метаданные."""

    def __init__(self, reranker: Optional[Reranker] = None):
        self.reranker = reranker

    def apply(
        self,
        chunks: List[RetrievedChunk],
        query: str,
        top_k_after: Optional[int] = None,
    ) -> Tuple[List[RetrievedChunk], dict]:
        before = len(chunks)
        if self.reranker is None:
            after = list(chunks)
            meta = {"provider": "none", "before": before, "after": before, "removed": 0, "threshold": None}
        else:
            after = self.reranker.rerank(chunks, query)
            if top_k_after is not None and len(after) > top_k_after:
                after = after[: top_k_after]
            meta = {
                "provider": self.reranker.name,
                "before": before,
                "after": len(after),
                "removed": before - len(after),
                "threshold": getattr(self.reranker, "min_score", None),
            }
        return after, meta
