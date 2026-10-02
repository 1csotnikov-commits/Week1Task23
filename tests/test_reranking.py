"""Тесты reranking (threshold, similarity, pipeline)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import Chunk  # noqa: E402
from src.retrieval.models import RetrievedChunk  # noqa: E402
from src.reranking.pipeline import RerankerPipeline  # noqa: E402
from src.reranking.similarity_reranker import SimilarityReranker  # noqa: E402
from src.reranking.threshold_filter import ThresholdFilter  # noqa: E402


def make_chunks(scores):
    return [
        RetrievedChunk(Chunk(f"c{i}", "a.txt", "a", f"текст {i}", 0, 5, "fixed", ""), s, i + 1)
        for i, s in enumerate(scores)
    ]


class TestReranking(unittest.TestCase):
    def test_threshold_filter(self):
        out = ThresholdFilter(min_score=0.3, min_keep=1).rerank(make_chunks([0.9, 0.5, 0.1, 0.2]), "запрос")
        self.assertEqual([c.score for c in out], [0.9, 0.5])
        self.assertEqual([c.rank for c in out], [1, 2])

    def test_threshold_min_keep(self):
        out = ThresholdFilter(min_score=0.3, min_keep=2).rerank(make_chunks([0.1, 0.05, 0.2]), "запрос")
        self.assertEqual(len(out), 2)

    def test_similarity_keyword_boost(self):
        c_low = RetrievedChunk(Chunk("c0", "a.txt", "a", "текст про другое", 0, 5, "fixed", ""), 0.9, 1)
        c_high = RetrievedChunk(Chunk("c1", "a.txt", "a", "текст про запрос и ещё", 0, 5, "fixed", ""), 0.4, 2)
        r = SimilarityReranker(weights={"cosine": 0.0, "keyword_overlap": 1.0, "phrase_match": 0.0})
        out = r.rerank([c_low, c_high], "запрос")
        self.assertEqual(out[0].chunk_id, "c1")

    def test_pipeline_meta(self):
        p = RerankerPipeline(ThresholdFilter(min_score=0.3, min_keep=1))
        after, meta = p.apply(make_chunks([0.9, 0.5, 0.1, 0.2]), "запрос", top_k_after=3)
        self.assertEqual(meta["before"], 4)
        self.assertEqual(meta["after"], 2)
        self.assertEqual(meta["removed"], 2)
        self.assertEqual(meta["provider"], "threshold")
        self.assertEqual(len(after), 2)


if __name__ == "__main__":
    unittest.main()
