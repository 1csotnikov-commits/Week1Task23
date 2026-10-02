"""RAG-пайплайн: (rewrite) -> retrieval -> (rerank) -> prompt -> LLM -> ответ."""

from __future__ import annotations

from typing import Optional

from src.config import Config
from src.llm.base import LLMProvider
from src.rag.models import RAGResult
from src.rag.prompt_builder import build_messages, render_messages
from src.rag.query_rewrite import QueryRewriter
from src.reranking.base import Reranker
from src.reranking.pipeline import RerankerPipeline
from src.retrieval.base import Retriever


class RAGPipeline:
    """RAG с опциональным query rewrite и reranking."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        config: Optional[Config] = None,
        reranker: Optional[Reranker] = None,
        rewriter: Optional[QueryRewriter] = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.config = config
        self.reranker = reranker
        self.rewriter = rewriter
        self._rerank_pipeline = RerankerPipeline(reranker)

    def _cfg(self, dotted: str, default):
        if self.config is not None:
            return self.config.get(dotted, default)
        return default

    def run(
        self,
        question: str,
        top_k_retrieve: Optional[int] = None,
        top_k_after_rerank: Optional[int] = None,
        use_reranker: bool = True,
        use_rewrite: bool = False,
    ) -> RAGResult:
        k_retrieve = int(top_k_retrieve) if top_k_retrieve is not None else int(self._cfg("retrieval.top_k_retrieve", 10))
        k_after = int(top_k_after_rerank) if top_k_after_rerank is not None else int(self._cfg("reranking.top_k_after_rerank", 5))

        original_query = question
        rewritten_query = question
        rewrite_meta = {"method": "none", "tokens": 0, "elapsed_ms": 0}
        if use_rewrite and self.rewriter is not None:
            rr = self.rewriter.rewrite(question)
            rewritten_query = rr.rewritten_query
            rewrite_meta = {
                "method": rr.method,
                "tokens": rr.meta.get("tokens", 0),
                "elapsed_ms": rr.meta.get("elapsed_ms", 0),
            }

        chunks_before = self.retriever.search(rewritten_query, k_retrieve)

        if use_reranker:
            chunks_after, rerank_meta = self._rerank_pipeline.apply(chunks_before, rewritten_query, top_k_after=k_after)
        else:
            chunks_after = list(chunks_before)
            rerank_meta = {
                "provider": "none",
                "before": len(chunks_before),
                "after": len(chunks_after),
                "removed": 0,
                "threshold": None,
            }

        messages = build_messages(original_query, chunks_after)
        prompt_text = render_messages(messages)
        response = self.llm.complete(messages)

        return RAGResult(
            answer=response.text,
            mode="rag",
            used_chunks=chunks_after,
            prompt=prompt_text,
            original_query=original_query,
            rewritten_query=rewritten_query if rewritten_query != original_query else None,
            chunks_before_rerank=chunks_before,
            chunks_after_rerank=chunks_after,
            rewrite_meta=rewrite_meta,
            rerank_meta=rerank_meta,
            meta={
                "model": response.model,
                "tokens_in": response.prompt_tokens,
                "tokens_out": response.completion_tokens,
                "elapsed_ms": response.elapsed_ms,
                "top_k_retrieve": k_retrieve,
                "top_k_after_rerank": k_after,
                "num_chunks": len(chunks_after),
            },
        )

