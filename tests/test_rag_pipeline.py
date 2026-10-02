"""Тесты RAG-пайплайна и режима без RAG."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.no_rag_pipeline import NoRAGPipeline  # noqa: E402
from src.rag.pipeline import RAGPipeline  # noqa: E402
from src.retrieval.vector_retriever import VectorRetriever  # noqa: E402
from src.reranking import ThresholdFilter  # noqa: E402
from tests.fakes import FakeEmbedder, FakeLLM, build_temp_index  # noqa: E402


class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _retriever(self) -> VectorRetriever:
        store = build_temp_index(self.root, "idx", n_chunks=4, dim=8)
        return VectorRetriever(store=store, embedder=FakeEmbedder(dim=8), index_name="idx")

    def test_rag_run(self):
        llm = FakeLLM(answer="Ответ с контекстом")
        pipeline = RAGPipeline(self._retriever(), llm)
        result = pipeline.run("Какой вопрос?", top_k_retrieve=2, use_reranker=False)

        self.assertEqual(result.answer, "Ответ с контекстом")
        self.assertEqual(result.mode, "rag")
        self.assertEqual(len(result.used_chunks), 2)
        self.assertIn("Контекст:", result.prompt)
        self.assertIn("Какой вопрос?", result.prompt)
        self.assertIn("текст чанка", result.prompt)
        self.assertEqual(result.meta["top_k_retrieve"], 2)
        self.assertEqual(result.meta["num_chunks"], 2)
        self.assertEqual(result.meta["model"], "fake-model")
        # новые поля
        self.assertEqual(result.original_query, "Какой вопрос?")
        self.assertEqual(len(result.chunks_before_rerank), 2)
        self.assertEqual(len(result.chunks_after_rerank), 2)

    def test_rag_run_with_reranker(self):
        llm = FakeLLM(answer="Ответ")
        pipeline = RAGPipeline(self._retriever(), llm, reranker=ThresholdFilter(min_score=0.0, min_keep=1))
        result = pipeline.run("Вопрос", top_k_retrieve=4, top_k_after_rerank=2, use_reranker=True)
        self.assertLessEqual(len(result.used_chunks), 2)
        self.assertEqual(result.rerank_meta["provider"], "threshold")
        self.assertEqual(result.rerank_meta["before"], 4)
        self.assertEqual(len(result.chunks_before_rerank), 4)

    def test_no_rag_run(self):
        llm = FakeLLM(answer="Ответ без контекста")
        pipeline = NoRAGPipeline(llm)
        result = pipeline.run("Какой вопрос?")

        self.assertEqual(result.mode, "no_rag")
        self.assertEqual(result.used_chunks, [])
        self.assertIn("Какой вопрос?", result.prompt)
        self.assertNotIn("Контекст:", result.prompt)


if __name__ == "__main__":
    unittest.main()
