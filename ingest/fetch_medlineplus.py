from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


@dataclass
class HealthTopicRecord:
    topic_id: str
    title: str
    url: str
    language: str
    date_created: str
    also_called: list[str]
    summary_html: str
    mesh_headings: list[str]
    primary_institute: str | None


def _candidate_dates(n: int = 6) -> list[str]:
    """MedlinePlus files are only generated Tue-Sat; try the last n days."""
    today = datetime.utcnow().date()
    return [(today - timedelta(days=i)).isoformat() for i in range(n)]


def download_topics_xml(date: str | None = None, force_refresh: bool = False) -> Path:
    """Download the MedlinePlus health-topic XML for `date` (YYYY-MM-DD),
    or walk backwards from today until one is found. Returns the local path.
    """
    dates_to_try = [date] if date else _candidate_dates()

    for d in dates_to_try:
        local_path = config.RAW_CACHE_DIR / f"mplus_topics_{d}.xml"
        if local_path.exists() and not force_refresh:
            log.info("Using cached MedlinePlus dump for %s", d)
            return local_path

        url = config.MEDLINEPLUS_XML_URL_TEMPLATE.format(date=d)
        log.info("Trying %s", url)
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200 and resp.content.startswith(b"<?xml"):
                local_path.write_bytes(resp.content)
                log.info("Saved %s (%d KB)", local_path, len(resp.content) // 1024)
                return local_path
        except requests.exceptions.RequestException as e:
            log.warning("Fetch failed for %s: %s", d, e)

    raise RuntimeError(
        f"Could not fetch a MedlinePlus health-topic XML dump for any of {dates_to_try}. "
        "Pass --date explicitly for a known-good day (files are published Tue-Sat)."
    )


def _text_or_none(el) -> str | None:
    return el.text.strip() if el is not None and el.text else None


def parse_and_filter(xml_path: Path, keywords: list[str] | None = None) -> list[HealthTopicRecord]:
    """Parse the MedlinePlus health-topic XML and keep only English topics
    whose title or 'also-called' vocabulary matches one of `keywords`.
    """
    keywords = [k.lower() for k in (keywords or config.TOPIC_KEYWORDS)]
    tree = ET.parse(xml_path)
    root = tree.getroot()

    kept: list[HealthTopicRecord] = []
    for topic in root.findall("health-topic"):
        language = topic.get("language", "English")
        if language != "English":
            continue

        title = topic.get("title", "")
        also_called = [ac.text.strip() for ac in topic.findall("also-called") if ac.text]
        searchable = " ".join([title] + also_called).lower()

        if not any(kw in searchable for kw in keywords):
            continue

        summary_el = topic.find("full-summary")
        summary_html = summary_el.text or "" if summary_el is not None else ""
        if not summary_html or len(summary_html.strip()) < config.MIN_CONTENT_CHARS:
            continue

        mesh_headings = [m.text.strip() for m in topic.findall("mesh-heading/descriptor") if m.text]
        institute_el = topic.find("primary-institute")

        kept.append(
            HealthTopicRecord(
                topic_id=topic.get("id", ""),
                title=title,
                url=topic.get("url", ""),
                language=language,
                date_created=topic.get("date-created", ""),
                also_called=also_called,
                summary_html=summary_html.strip(),
                mesh_headings=mesh_headings,
                primary_institute=_text_or_none(institute_el),
            )
        )

    log.info("Kept %d/%d English topics matching %s", len(kept), len(root.findall("health-topic")), keywords)
    return kept


def save_filtered_records(records: list[HealthTopicRecord], out_name: str = "filtered_topics.json") -> Path:
    out_path = config.RAW_CACHE_DIR / out_name
    out_path.write_text(json.dumps([asdict(r) for r in records], indent=2))
    log.info("Wrote %d filtered records to %s", len(records), out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="YYYY-MM-DD; defaults to walking back from today")
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    xml_path = download_topics_xml(date=args.date, force_refresh=args.force_refresh)
    records = parse_and_filter(xml_path)
    save_filtered_records(records)


if __name__ == "__main__":
    main()