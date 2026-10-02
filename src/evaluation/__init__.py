"""Оценка качества RAG (контрольные вопросы + сравнение режимов)."""

from __future__ import annotations

from src.evaluation.comparator import aggregate_modes, evaluate_mode
from src.evaluation.generator import generate_questions, sample_chunks
from src.evaluation.questions import Question, load_questions, save_questions
from src.evaluation.report import render_report, render_report_file
from src.evaluation.runner import (
    ALL_MODES,
    DEFAULT_MODES,
    QuestionResult,
    build_payload,
    load_payload,
    mode_flags,
    run_evaluation,
    save_results,
)

__all__ = [
    "Question",
    "load_questions",
    "save_questions",
    "generate_questions",
    "sample_chunks",
    "evaluate_mode",
    "aggregate_modes",
    "QuestionResult",
    "run_evaluation",
    "save_results",
    "build_payload",
    "load_payload",
    "render_report",
    "render_report_file",
    "DEFAULT_MODES",
    "ALL_MODES",
    "mode_flags",
]
