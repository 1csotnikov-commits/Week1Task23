"""Reranking — переранжирование и фильтрация результатов поиска (реализация Дня 23)."""

from __future__ import annotations

from typing import Optional

from src.reranking.base import Reranker
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.reranking.pipeline import RerankerPipeline
from src.reranking.similarity_reranker import SimilarityReranker
from src.reranking.threshold_filter import ThresholdFilter

RERANKER_PROVIDERS = ("threshold", "similarity", "cross_encoder", "none")


def make_reranker(provider: str, config=None) -> Optional[Reranker]:
    """Создать реранкер по имени провайдера с параметрами из конфига."""
    provider = (provider or "threshold").lower()
    if provider == "none":
        return None
    if provider == "threshold":
        min_score = 0.3
        min_keep = 1
        if config is not None:
            min_score = float(config.get("reranking.threshold.min_score", 0.3))
            min_keep = int(config.get("reranking.threshold.min_keep", 1))
        return ThresholdFilter(min_score=min_score, min_keep=min_keep)
    if provider == "similarity":
        weights = {"cosine": 0.6, "keyword_overlap": 0.3, "phrase_match": 0.1}
        if config is not None:
            weights = dict(config.get("reranking.similarity.weights", weights) or weights)
        return SimilarityReranker(weights=weights)
    if provider == "cross_encoder":
        model = "BAAI/bge-reranker-v2-m3"
        device = "cpu"
        if config is not None:
            model = config.get("reranking.cross_encoder.model", model)
            device = config.get("reranking.cross_encoder.device", device)
        return CrossEncoderReranker(model_name=model, device=device)
    raise ValueError(f"Неизвестный реранкер: {provider!r}. Доступно: {', '.join(RERANKER_PROVIDERS)}")


__all__ = [
    "Reranker",
    "ThresholdFilter",
    "SimilarityReranker",
    "CrossEncoderReranker",
    "RerankerPipeline",
    "make_reranker",
    "RERANKER_PROVIDERS",
]
