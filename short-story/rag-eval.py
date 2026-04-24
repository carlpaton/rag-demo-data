from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_pages(story_dir: Path) -> dict[str, str]:
    pages: dict[str, str] = {}
    for path in sorted(story_dir.glob("pg*.md"), key=lambda p: int(p.stem[2:])):
        pages[path.name] = path.read_text(encoding="utf-8")
    if not pages:
        raise FileNotFoundError(f"No pg*.md files found in {story_dir}")
    return pages


def load_eval(eval_path: Path) -> list[dict]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    return payload["items"]


def build_tfidf(texts: dict[str, str]):
    tokenized = {k: tokenize(v) for k, v in texts.items()}
    n_docs = len(tokenized)

    df = Counter()
    for toks in tokenized.values():
        df.update(set(toks))

    vocab = {tok: i for i, tok in enumerate(sorted(df.keys()))}
    idf = [0.0] * len(vocab)
    for tok, idx in vocab.items():
        idf[idx] = math.log((n_docs + 1) / (df[tok] + 1)) + 1.0

    vectors: dict[str, list[float]] = {}
    for key, toks in tokenized.items():
        tf = Counter(toks)
        vec = [0.0] * len(vocab)
        total = len(toks) or 1
        for tok, count in tf.items():
            idx = vocab.get(tok)
            if idx is None:
                continue
            vec[idx] = (count / total) * idf[idx]
        vectors[key] = vec

    return vocab, idf, vectors


def text_to_tfidf(text: str, vocab: dict[str, int], idf: list[float]) -> list[float]:
    toks = tokenize(text)
    tf = Counter(toks)
    total = len(toks) or 1

    vec = [0.0] * len(vocab)
    for tok, count in tf.items():
        idx = vocab.get(tok)
        if idx is None:
            continue
        vec[idx] = (count / total) * idf[idx]
    return vec


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    na = norm(a)
    nb = norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def rank_docs(
    query_vec: list[float],
    doc_vectors: dict[str, list[float]],
    score_fn: Callable[[list[float], list[float]], float],
    reverse: bool,
) -> list[tuple[str, float]]:
    scored = [(doc, score_fn(query_vec, vec)) for doc, vec in doc_vectors.items()]
    return sorted(scored, key=lambda x: x[1], reverse=reverse)


def evaluate_metric(
    eval_items: list[dict],
    doc_vectors: dict[str, list[float]],
    vocab: dict[str, int],
    idf: list[float],
    score_fn: Callable[[list[float], list[float]], float],
    reverse: bool,
    ks: tuple[int, ...],
):
    hits = {k: 0 for k in ks}
    recalls = {k: 0.0 for k in ks}
    precision = {k: 0.0 for k in ks}
    reciprocal_rank_sum = 0.0

    for item in eval_items:
        query_vec = text_to_tfidf(item["question"], vocab, idf)
        ranked = rank_docs(query_vec, doc_vectors, score_fn, reverse)
        ranked_docs = [doc for doc, _ in ranked]

        relevant = set(item["support_docs"])

        for k in ks:
            top_k = ranked_docs[:k]
            retrieved_relevant = len(relevant.intersection(top_k))
            if retrieved_relevant > 0:
                hits[k] += 1
            recalls[k] += retrieved_relevant / len(relevant)
            precision[k] += retrieved_relevant / k

        rr = 0.0
        for idx, doc in enumerate(ranked_docs, start=1):
            if doc in relevant:
                rr = 1.0 / idx
                break
        reciprocal_rank_sum += rr

    n = len(eval_items)
    return {
        "hit_rate": {f"@{k}": hits[k] / n for k in ks},
        "recall": {f"@{k}": recalls[k] / n for k in ks},
        "precision": {f"@{k}": precision[k] / n for k in ks},
        "mrr": reciprocal_rank_sum / n,
    }


_SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Compare retrieval metrics on story pages.")
    parser.add_argument("--story-dir", default=str(_SCRIPT_DIR), help="Folder with pg*.md files")
    parser.add_argument(
        "--eval-json",
        default=str(_SCRIPT_DIR / "rag-eval.json"),
        help="Evaluation JSON containing questions and support_docs",
    )
    parser.add_argument(
        "--ks",
        default="3,5",
        help="Comma-separated K values for hit/recall/precision (default: 3,5)",
    )
    args = parser.parse_args()

    ks = tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())

    story_dir = Path(args.story_dir)
    eval_path = Path(args.eval_json)

    pages = load_pages(story_dir)
    eval_items = load_eval(eval_path)

    vocab, idf, doc_vectors = build_tfidf(pages)

    results = {
        "cosine_similarity": evaluate_metric(
            eval_items,
            doc_vectors,
            vocab,
            idf,
            cosine_similarity,
            True,
            ks,
        ),
        "euclidean_distance": evaluate_metric(
            eval_items,
            doc_vectors,
            vocab,
            idf,
            euclidean_distance,
            False,
            ks,
        ),
        "dot_product": evaluate_metric(
            eval_items,
            doc_vectors,
            vocab,
            idf,
            dot,
            True,
            ks,
        ),
    }

    print(json.dumps(results, indent=2))

    best_metric = max(results.items(), key=lambda x: x[1]["mrr"])
    print(f"\nBest by MRR: {best_metric[0]} ({best_metric[1]['mrr']:.4f})")


if __name__ == "__main__":
    main()
