from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import config

log = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Parse the hand-rolled frontmatter written by ingest/to_markdown.py.
    Returns (metadata_dict, body_markdown).
    """
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw

    fm_block, body = m.group(1), m.group(2)
    meta: dict = {}
    for line in fm_block.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value.startswith("[") or value.startswith("{"):
            try:
                meta[key] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        if value == "null":
            meta[key] = None
        else:
            meta[key] = value.strip('"')
    return meta, body


def load_corpus_documents(corpus_dir: Path | None = None) -> list[Document]:
    """One Document per markdown file, metadata pulled from frontmatter.
    (Not chunked yet - see chunk_documents.)
    """
    corpus_dir = corpus_dir or config.CORPUS_DIR
    md_files = sorted(corpus_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            f"No markdown files in {corpus_dir} - run the ingest pipeline first "
            "(python -m ingest.fetch_medlineplus && python -m ingest.to_markdown)."
        )

    docs = []
    for path in md_files:
        meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        meta["file_path"] = str(path)
        docs.append(Document(page_content=body, metadata=meta))

    log.info("Loaded %d documents from %s", len(docs), corpus_dir)
    return docs


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Two-stage split: first on markdown headers (so a chunk never straddles
    an unrelated section like "Symptoms" and "Prevention"), then on size
    within each section.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=config.MARKDOWN_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.CHUNK_SEPARATORS,
    )

    all_chunks: list[Document] = []
    for doc in documents:
        header_sections = header_splitter.split_text(doc.page_content)
        for section in header_sections:
            section.metadata = {**doc.metadata, **section.metadata}
        sized = size_splitter.split_documents(header_sections)
        all_chunks.extend(sized)

    log.info("Chunked %d documents into %d chunks", len(documents), len(all_chunks))
    return all_chunks


def load_and_chunk(corpus_dir: Path | None = None) -> list[Document]:
    return chunk_documents(load_corpus_documents(corpus_dir))