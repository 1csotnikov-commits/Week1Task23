"""Тесты evaluation (вопросы, эвристики, мультирежимный прогон, отчёты)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.comparator import (  # noqa: E402
    aggregate_modes,
    evaluate_mode,
    is_hit,
    jaccard,
    keyword_overlap,
    source_hit,
    substring_match,
)
from src.evaluation.questions import Question, load_questions, save_questions  # noqa: E402
from src.evaluation.report import render_report  # noqa: E402
from src.evaluation.runner import QuestionResult, build_payload, save_results  # noqa: E402
from src.models import Chunk  # noqa: E402
from src.rag.models import RAGResult  # noqa: E402
from src.retrieval.models import RetrievedChunk  # noqa: E402


def make_chunk(source: str = "a.md", section: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id="c0", source=source, title="a", text="текст", start_char=0, end_char=4, strategy="fixed", section=section),
        score=0.9,
        rank=1,
    )


def make_rag_result(answer: str, chunks, mode: str = "rag") -> RAGResult:
    return RAGResult(
        answer=answer,
        mode=mode,
        used_chunks=chunks,
        chunks_before_rerank=chunks,
        chunks_after_rerank=chunks,
        meta={"elapsed_ms": 10, "tokens_in": 20, "tokens_out": 5},
    )


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_load_save_roundtrip(self):
        qs = [Question(id="q01", question="Вопрос", expected_answer="Ответ", expected_sources=["a.md"], notes="заметка")]
        path = save_questions(qs, self.root / "questions.yaml")
        loaded = load_questions(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].id, "q01")
        self.assertEqual(loaded[0].expected_sources, ["a.md"])

    def test_load_invalid_raises(self):
        p = self.root / "bad.yaml"
        p.write_text("questions:\n  - id: q01\n    question: Без ответа\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_questions(p)

    def test_comparator_functions(self):
        self.assertTrue(substring_match("Это ответ на вопрос", "ответ"))
        self.assertGreater(keyword_overlap("это ответ на вопрос", "ответ"), 0)
        self.assertGreater(jaccard("ответ вопрос", "ответ"), 0)
        self.assertTrue(is_hit({"substring_match": True, "keyword_overlap": 0.0}))

    def test_source_hit(self):
        hit = source_hit([make_chunk("a.md", "Раздел 2")], ["a.md#Раздел 2"])
        self.assertTrue(hit["hit"])
        miss = source_hit([make_chunk("b.md", "")], ["a.md"])
        self.assertFalse(miss["hit"])

    def test_evaluate_and_aggregate(self):
        q = Question("q01", "Вопрос", "ответ", ["a.md"])
        rag = make_rag_result("Это ответ на вопрос", [make_chunk("a.md", "S")])
        no_rag = make_rag_result("Не знаю", [], "no_rag")

        row_rag = evaluate_mode(rag, q)
        self.assertTrue(row_rag["hit"])
        self.assertEqual(row_rag["num_chunks_after"], 1)

        rows = [{"id": "q01", "modes": {"rag": evaluate_mode(rag, q), "no_rag": evaluate_mode(no_rag, q)}}]
        agg = aggregate_modes(rows)
        self.assertEqual(agg["num_questions"], 1)
        self.assertEqual(agg["per_mode"]["rag"]["hit_rate"], 1.0)
        self.assertEqual(agg["per_mode"]["no_rag"]["hit_rate"], 0.0)

    def test_render_formats(self):
        q = Question("q01", "Вопрос", "ответ", ["a.md"])
        rag = make_rag_result("ответ", [make_chunk("a.md")])
        no_rag = make_rag_result("не знаю", [], "no_rag")
        payload = build_payload([QuestionResult(q, {"rag": rag, "no_rag": no_rag})], index_name="idx")
        for fmt in ("console", "markdown", "json"):
            out = render_report(payload, fmt)
            self.assertTrue(out.strip())

    def test_save_results(self):
        q = Question("q01", "Вопрос", "ответ", ["a.md"])
        rag = make_rag_result("ответ", [make_chunk("a.md")])
        no_rag = make_rag_result("не знаю", [], "no_rag")
        json_path, md_path, payload = save_results(
            [QuestionResult(q, {"rag": rag, "no_rag": no_rag})], self.root / "results", "idx"
        )
        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())
        self.assertEqual(payload["index_name"], "idx")
        self.assertEqual(payload["modes"], ["rag", "no_rag"])


if __name__ == "__main__":
    unittest.main()

