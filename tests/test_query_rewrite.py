"""Тесты query rewrite (NoOp, Heuristic, LLM)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.query_rewrite import HeuristicRewriter, LLMRewriter, NoOpRewriter  # noqa: E402
from tests.fakes import FakeLLM  # noqa: E402


class TestQueryRewrite(unittest.TestCase):
    def test_noop(self):
        r = NoOpRewriter().rewrite("запрос")
        self.assertEqual(r.rewritten_query, "запрос")
        self.assertEqual(r.method, "none")

    def test_heuristic(self):
        r = HeuristicRewriter(min_keywords=1).rewrite("что такое ТС ПИоТ")
        self.assertEqual(r.method, "heuristic")
        self.assertIn("тс", r.rewritten_query.lower())

    def test_heuristic_below_min(self):
        r = HeuristicRewriter(min_keywords=10).rewrite("короткий")
        self.assertEqual(r.rewritten_query, "короткий")

    def test_llm(self):
        llm = FakeLLM(answer="переписанный вопрос")
        r = LLMRewriter(llm).rewrite("вопрос")
        self.assertEqual(r.rewritten_query, "переписанный вопрос")
        self.assertEqual(r.method, "llm")


if __name__ == "__main__":
    unittest.main()
