"""
Build (or load from cache) both retrieval indexes over the chunked markdown
corpus:
  - dense:  FAISS + sentence-transformer embeddings   (semantic similarity)
  - sparse: BM25 over tokenized chunk text             (lexical / keyword match)

Both are fingerprinted on the corpus content hash, same pattern as the
original notebook's FAISS cache, so re-running only rebuilds when the
markdown corpus actually changed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import config
from retrieval.loaders import load_and_chunk

log = logging.getLogger(__name__)


@dataclass
class Indexes:
    vectorstore: FAISS
    bm25: BM25Retriever
    chunks: list[Document]


def _fingerprint(chunks: list[Document]) -> str:
    hashes = sorted(
        c.metadata.get("content_hash", hashlib.md5(c.page_content.encode()).hexdigest())
        + c.page_content[:50]  # chunk-level differentiation, not just parent doc hash
        for c in chunks
    )
    return hashlib.md5("".join(hashes).encode()).hexdigest()


def _fingerprint_path() -> Path:
    return config.STORAGE_DIR / "fingerprint.json"


def _faiss_dir() -> Path:
    return config.STORAGE_DIR / "faiss_index"


def _bm25_path() -> Path:
    return config.STORAGE_DIR / "bm25.pkl"


def _load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"batch_size": config.EMBEDDING_BATCH_SIZE, "normalize_embeddings": True},
    )


def build_or_load_indexes(force_rebuild: bool = False) -> Indexes:
    chunks = load_and_chunk()
    fingerprint = _fingerprint(chunks)
    fp_path = _fingerprint_path()

    if fp_path.exists() and not force_rebuild:
        old = json.loads(fp_path.read_text()).get("fingerprint")
        if old == fingerprint and _faiss_dir().exists() and _bm25_path().exists():
            log.info("Corpus unchanged - loading cached FAISS + BM25 indexes")
            embeddings = _load_embeddings()
            vectorstore = FAISS.load_local(
                str(_faiss_dir()), embeddings, allow_dangerous_deserialization=True
            )
            bm25 = pickle.loads(_bm25_path().read_bytes())
            return Indexes(vectorstore=vectorstore, bm25=bm25, chunks=chunks)
        log.info("Corpus changed - rebuilding indexes")

    # ── Dense (FAISS) ──
    embeddings = _load_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(_faiss_dir()))

    # ── Sparse (BM25) ──
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = config.BM25_CANDIDATE_K
    _bm25_path().write_bytes(pickle.dumps(bm25))

    fp_path.write_text(
        json.dumps({"fingerprint": fingerprint, "built_at": datetime.now(timezone.utc).isoformat(),
                    "num_chunks": len(chunks)}, indent=2)
    )
    log.info("Built and cached indexes over %d chunks", len(chunks))
    return Indexes(vectorstore=vectorstore, bm25=bm25, chunks=chunks)