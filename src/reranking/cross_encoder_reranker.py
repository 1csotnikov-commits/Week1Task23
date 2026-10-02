"""Реранкер на отдельной CrossEncoder-модели (опционально, по умолчанию выключен).

Модель BAAI/bge-reranker-v2-m3 (~2.3 ГБ) скачивается при первом использовании.
Работает на CPU: требуется 8–16 GB RAM и ~8 GB диска, обработка медленная.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List

from src.retrieval.models import RetrievedChunk
from src.reranking.base import Reranker

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker(Reranker):
    """Переранжирование через sentence-transformers CrossEncoder."""

    name = "cross_encoder"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        print(
            f"Загрузка cross-encoder модели {self.model_name} (~2.3 ГБ, CPU). "
            "Требуется 8–16 GB RAM; обработка на CPU медленная."
        )
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Пакет sentence-transformers не установлен. Выполните: pip install sentence-transformers"
            ) from exc
        self._model = CrossEncoder(self.model_name, device=self.device)
        print("Cross-encoder загружен.")

    def rerank(self, chunks: List[RetrievedChunk], query: str) -> List[RetrievedChunk]:
        if not chunks:
            return []
        self._ensure_loaded()
        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)
        rescored = [replace(c, score=float(s)) for c, s in zip(chunks, scores)]
        rescored.sort(key=lambda c: c.score, reverse=True)
        return [replace(c, rank=i + 1) for i, c in enumerate(rescored)]
