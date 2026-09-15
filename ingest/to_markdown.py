from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from markdownify import markdownify

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(title: str) -> str:
    return _SLUG_RE.sub("-", title.lower()).strip("-")


def html_to_markdown(html: str) -> str:
    md = markdownify(html, heading_style="ATX")
    # collapse >2 blank lines left over from the HTML
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md


def record_to_markdown_file(record: dict) -> Path:
    body_md = html_to_markdown(record["summary_html"])
    content_hash = hashlib.md5(body_md.encode()).hexdigest()

    frontmatter = {
        "title": record["title"],
        "source_name": config.CORPUS_SOURCE_NAME,
        "source_url": record["url"],
        "topic_id": record["topic_id"],
        "mesh_headings": record["mesh_headings"],
        "also_called": record["also_called"],
        "primary_institute": record["primary_institute"],
        "content_hash": content_hash,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

    # Simple, dependency-free YAML frontmatter (values are all
    # strings/lists/None, so hand-rolling this avoids a pyyaml dependency).
    fm_lines = ["---"]
    for k, v in frontmatter.items():
        if isinstance(v, list):
            fm_lines.append(f"{k}: {json.dumps(v)}")
        elif v is None:
            fm_lines.append(f"{k}: null")
        else:
            fm_lines.append(f'{k}: "{str(v).replace(chr(34), chr(39))}"')
    fm_lines.append("---\n")

    file_text = "\n".join(fm_lines) + f"# {record['title']}\n\n{body_md}\n"

    out_path = config.CORPUS_DIR / f"{slugify(record['title'])}.md"
    out_path.write_text(file_text, encoding="utf-8")
    return out_path


def build_corpus(filtered_json: Path | None = None) -> list[Path]:
    filtered_json = filtered_json or (config.RAW_CACHE_DIR / "filtered_topics.json")
    if not filtered_json.exists():
        raise FileNotFoundError(
            f"{filtered_json} not found - run `python -m ingest.fetch_medlineplus` first."
        )

    records = json.loads(filtered_json.read_text())
    written = [record_to_markdown_file(r) for r in records]
    log.info("Wrote %d markdown documents to %s", len(written), config.CORPUS_DIR)
    return written


if __name__ == "__main__":
    build_corpus()