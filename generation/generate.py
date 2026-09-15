from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache

import config
from retrieval.hybrid import HybridRetriever, ScoredChunk

log = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    question: str
    answer: str
    in_scope: bool
    scope_score: float | None
    sources: list[dict]  # [{title, url, source_name}, ...] deduped, in citation order


def _format_context(chunks: list[ScoredChunk]) -> tuple[str, list[dict]]:
    blocks = []
    sources = []
    seen = set()
    for sc in chunks:
        meta = sc.document.metadata
        title = meta.get("title", "Unknown topic")
        url = meta.get("source_url", "")
        source_name = meta.get("source_name", "Unknown")
        label = f"{source_name}: {title}"

        blocks.append(f"[{label}]\n{sc.document.page_content}")
        if label not in seen:
            seen.add(label)
            sources.append({"title": title, "url": url, "source_name": source_name})

    return "\n\n---\n\n".join(blocks), sources


@lru_cache(maxsize=1)
def _get_local_llm():
    """Load the gguf model once and cache it - this is the expensive part,
    so it must not happen per-request.
    """
    import os

    from llama_cpp import Llama

    if not config.LOCAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Local model not found at {config.LOCAL_MODEL_PATH}. "
            "Download the gguf into medical_rag/models/ or update config.LOCAL_MODEL_PATH."
        )

    log.info("Loading local model from %s (this happens once)", config.LOCAL_MODEL_PATH)
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(config.LOCAL_MODEL_PATH),
        n_ctx=config.LOCAL_MODEL_N_CTX,
        n_gpu_layers=config.LOCAL_MODEL_N_GPU_LAYERS,
        n_threads=os.cpu_count(),
        n_threads_batch=os.cpu_count(),
        n_batch=512,
        verbose=False,
    )
    log.info("Model load took %.1fs", time.perf_counter() - t0)
    return llm


def _generate_local(context: str, question: str) -> str:
    llm = _get_local_llm()
    t0 = time.perf_counter()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {question}",
            },
        ],
        max_tokens=config.GENERATION_MAX_TOKENS,
        temperature=config.GENERATION_TEMPERATURE,
    )
    elapsed = time.perf_counter() - t0
    usage = response.get("usage", {})
    log.info(
        "Generation took %.1fs (prompt_tokens=%s, completion_tokens=%s)",
        elapsed, usage.get("prompt_tokens"), usage.get("completion_tokens"),
    )
    return response["choices"][0]["message"]["content"]


# ── Public entry point ───────────────────────────────────────────────────────

def answer_question(
    question: str,
    retriever: HybridRetriever | None = None,
    top_k: int = config.TOP_K,
) -> RAGAnswer:
    retriever = retriever or HybridRetriever()

    in_scope, scope_score = retriever.is_in_scope(question)
    if not in_scope:
        return RAGAnswer(
            question=question,
            answer=config.OUT_OF_SCOPE_MESSAGE,
            in_scope=False,
            scope_score=scope_score,
            sources=[],
        )

    chunks = retriever.search(question, top_k=top_k)
    context, sources = _format_context(chunks)

    answer_text = _generate_local(context, question)

    return RAGAnswer(
        question=question,
        answer=answer_text,
        in_scope=True,
        scope_score=scope_score,
        sources=sources,
    )