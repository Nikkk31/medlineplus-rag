from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def cmd_ingest(args):
    from ingest.fetch_medlineplus import download_topics_xml, parse_and_filter, save_filtered_records
    from ingest.to_markdown import build_corpus

    xml_path = download_topics_xml(date=args.date, force_refresh=args.force_refresh)
    records = parse_and_filter(xml_path)
    save_filtered_records(records)
    build_corpus()


def cmd_build_index(args):
    from retrieval.indexing import build_or_load_indexes

    build_or_load_indexes(force_rebuild=args.force_rebuild)


def cmd_ask(args):
    from generation.generate import answer_question

    result = answer_question(args.question)
    print(f"\nIn scope: {result.in_scope} (score={result.scope_score})\n")
    print(result.answer)
    if result.sources:
        print("\nSources:")
        for s in result.sources:
            print(f"  - {s['title']} ({s['source_name']}) {s['url']}")


def cmd_eval(args):
    from eval.eval_retrieval import evaluate, print_report
    from retrieval.hybrid import HybridRetriever

    print_report(evaluate(HybridRetriever()))


def cmd_repl(args):
    """Interactive mode: load embeddings + indexes + local model ONCE,
    then answer questions in a loop instead of paying full startup cost
    (HF cache checks, FAISS/BM25 load, gguf load) on every single question.
    """
    from generation.generate import answer_question
    from retrieval.hybrid import HybridRetriever

    print("Loading indexes and model (one-time)...")
    retriever = HybridRetriever()  # builds/loads FAISS + BM25 once

    print("Ready. Type a question, or 'exit' to quit.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("exit", "quit"):
            break

        # Reuses the already-loaded retriever; the local gguf model is cached
        # via @lru_cache inside generation/generate.py, so it also only loads
        # once across this whole loop, not once per question.
        result = answer_question(question, retriever=retriever)
        print(f"\nIn scope: {result.in_scope} (score={result.scope_score})\n")
        print(result.answer)
        if result.sources:
            print("\nSources:")
            for s in result.sources:
                print(f"  - {s['title']} ({s['source_name']}) {s['url']}")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Fetch + filter MedlinePlus data, write markdown corpus")
    p_ingest.add_argument("--date", default=None)
    p_ingest.add_argument("--force-refresh", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_index = sub.add_parser("build-index", help="Build/refresh FAISS + BM25 indexes from the markdown corpus")
    p_index.add_argument("--force-rebuild", action="store_true")
    p_index.set_defaults(func=cmd_build_index)

    p_ask = sub.add_parser("ask", help="Ask a question against the RAG pipeline")
    p_ask.add_argument("question")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="Run the retrieval eval harness (dense vs BM25 vs hybrid)")
    p_eval.set_defaults(func=cmd_eval)

    p_repl = sub.add_parser("repl", help="Interactive mode - loads everything once, then answer questions in a loop")
    p_repl.set_defaults(func=cmd_repl)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()