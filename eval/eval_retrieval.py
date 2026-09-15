from __future__ import annotations

import logging
from pathlib import Path

import config
from eval.eval_set import EVAL_SET
from retrieval.hybrid import HybridRetriever

logging.basicConfig(level=logging.WARNING)  # keep eval output clean


def _topic_slug(doc) -> str:
    file_path = doc.metadata.get("file_path", "")
    return Path(file_path).stem if file_path else "unknown"


def _hit(expected: list[str] | None, retrieved_topics: list[str]) -> bool | None:
    if expected is None:
        return None  # not yet verified - see diagnostic output instead
    if not expected:
        return None  # out-of-scope control, nothing to match against
    return bool(set(expected) & set(retrieved_topics))


def evaluate(retriever: HybridRetriever, k: int = config.TOP_K) -> dict:
    rows = []
    for item in EVAL_SET:
        q = item["question"]
        expected = item.get("expected_topics")
        out_of_scope = item.get("out_of_scope", False)

        dense = retriever.dense_search(q, k=k)
        dense_topics = [_topic_slug(d) for d, _ in dense]

        bm25 = retriever.bm25_search(q, k=k)
        bm25_topics = [_topic_slug(d) for d in bm25]

        fused = retriever.search(q, top_k=k)
        fused_topics = [_topic_slug(sc.document) for sc in fused]

        in_scope, scope_score = retriever.is_in_scope(q)

        rows.append({
            "question": q,
            "expected": expected,
            "needs_review": expected is None and not out_of_scope,
            "out_of_scope": out_of_scope,
            "dense_topics": dense_topics,
            "bm25_topics": bm25_topics,
            "fused_topics": fused_topics,
            "dense_hit": _hit(expected, dense_topics),
            "bm25_hit": _hit(expected, bm25_topics),
            "hybrid_hit": _hit(expected, fused_topics),
            "in_scope_predicted": in_scope,
            "in_scope_expected": not out_of_scope,
            "scope_score": scope_score,
        })
    return {"rows": rows}


def print_report(results: dict) -> None:
    rows = results["rows"]
    scored = [r for r in rows if not r["out_of_scope"] and not r["needs_review"]]
    review = [r for r in rows if r["needs_review"]]
    unscoped = [r for r in rows if r["out_of_scope"]]

    def acc(key):
        hits = sum(1 for r in scored if r[key])
        return hits, len(scored), 100 * hits / len(scored) if scored else 0.0

    print("Retrieval accuracy by method (top-k = %d, topic-level ground truth)\n" % config.TOP_K)
    for key, label in [("dense_hit", "Dense (FAISS)"), ("bm25_hit", "BM25"), ("hybrid_hit", "Hybrid (RRF)")]:
        hits, total, pct = acc(key)
        print(f"  {label:<16} {hits}/{total}  ({pct:.0f}%)")

    print("\nPer-question detail (scored):")
    for r in scored:
        flag = "✅" if r["hybrid_hit"] else "❌"
        methods = f"dense={'✓' if r['dense_hit'] else '✗'} bm25={'✓' if r['bm25_hit'] else '✗'}"
        score = f"{r['scope_score']:.3f}" if r["scope_score"] is not None else "n/a"
        print(f"  {flag} {r['question']}  expected={r['expected']}  [{methods}]  scope_score={score}")
        if not r["hybrid_hit"]:
            print(f"      dense retrieved:  {r['dense_topics']}")
            print(f"      bm25 retrieved:   {r['bm25_topics']}")
            print(f"      hybrid retrieved: {r['fused_topics']}")

    if review:
        print("\nNeeds manual review (expected_topics=None in eval_set.py) - "
              "fill in the real answer based on what's actually retrieved:")
        for r in review:
            print(f"\n  ❓ {r['question']}")
            print(f"      dense retrieved:  {r['dense_topics']}")
            print(f"      bm25 retrieved:   {r['bm25_topics']}")
            print(f"      hybrid retrieved: {r['fused_topics']}")

    gate_correct = sum(1 for r in rows if r["in_scope_predicted"] == r["in_scope_expected"])
    print(f"\nIn-scope gate: {gate_correct}/{len(rows)} correct (threshold={config.SIMILARITY_THRESHOLD})")

    if scored:
        in_scope_scores = [r["scope_score"] for r in scored if r["scope_score"] is not None]
        if in_scope_scores:
            print(f"\nIn-scope score range: min={min(in_scope_scores):.3f}  max={max(in_scope_scores):.3f}")
    if unscoped:
        out_scores = [r["scope_score"] for r in unscoped if r["scope_score"] is not None]
        if out_scores:
            print(f"Out-of-scope score range: min={min(out_scores):.3f}  max={max(out_scores):.3f}")
        print("Out-of-scope controls:")
        for r in unscoped:
            score = f"{r['scope_score']:.3f}" if r["scope_score"] is not None else "n/a"
            print(f"  {r['question']} -> score={score}, flagged_in_scope={r['in_scope_predicted']}")


def main():
    retriever = HybridRetriever()
    results = evaluate(retriever)
    print_report(results)


if __name__ == "__main__":
    main()