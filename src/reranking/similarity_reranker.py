"""Эвристический реранкер: пересчёт score с лексическими признаками."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional

from src.retrieval.models import RetrievedChunk
from src.reranking.base import Reranker
from src.text_utils import phrase_match, query_keyword_overlap

DEFAULT_WEIGHTS = {"cosine": 0.6, "keyword_overlap": 0.3, "phrase_match": 0.1}


class SimilarityReranker(Reranker):
    """Пересчитывает score = w1*cosine + w2*keyword_overlap + w3*phrase_match.

    Сортирует по новому score, отсекает по min_score (по умолчанию 0.0 — не отсекает).
    """

    name = "similarity"

    def __init__(self, weights: Optional[Dict[str, float]] = None, min_score: float = 0.0, min_keep: int = 1):
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.min_score = float(min_score)
        self.min_keep = int(min_keep)

    def rerank(self, chunks: List[RetrievedChunk], query: str) -> List[RetrievedChunk]:
        rescored = []
        for c in chunks:
            kw = query_keyword_overlap(query, c.text)
            ph = phrase_match(query, c.text)
            new_score = (
                self.weights.get("cosine", 0.0) * c.score
                + self.weights.get("keyword_overlap", 0.0) * kw
                + self.weights.get("phrase_match", 0.0) * ph
            )
            rescored.append(replace(c, score=float(new_score)))

        rescored.sort(key=lambda c: c.score, reverse=True)

        kept = [c for c in rescored if c.score >= self.min_score]
        if len(kept) < self.min_keep and rescored:
            kept = rescored[: self.min_keep]
        return [replace(c, rank=i + 1) for i, c in enumerate(kept)]
