# Retrieval Technique Evaluation

Compares three vector similarity/distance functions for page-level retrieval on the short story dataset.

## Setup

- **Pages:** `short-story/pg1.md` – `pg20.md` (20 files, loaded into memory as strings)
- **Vectors:** TF-IDF computed in pure Python — no external library, no vector database
- **Ground truth:** `short-story/rag-eval.json` — 10 questions, each with a `support_docs` list of expected relevant pages
- **Script:** `evaluate_retrieval.py`

## How It Works

1. All pages are tokenized and converted to TF-IDF vectors (plain Python lists of floats)
2. Each of the 10 questions from `rag-eval.json` is also converted to a TF-IDF vector at query time
3. Each retrieval function scores every page against the query vector and returns a ranked list
4. The ranked list is compared against the `support_docs` labels to compute the metrics below

## Retrieval Functions Tested

| Function | Direction | Description |
|---|---|---|
| Cosine Similarity | Higher = better | Angle between vectors; length-normalised |
| Euclidean Distance | Lower = better | Straight-line distance in vector space |
| Dot Product | Higher = better | Raw inner product; sensitive to vector magnitude |

## Metrics Explained

| Metric | What it measures |
|---|---|
| HitRate@k | Did any relevant page appear in the top k? (binary per question, averaged) |
| Recall@k | Fraction of all relevant pages that appeared in the top k |
| Precision@k | Fraction of the top k results that were actually relevant |
| MRR | Mean Reciprocal Rank — how high was the first relevant page on average (1.0 = always ranked first) |

K values used: **3** and **5**.

## Results

### Cosine Similarity

| Metric | @3 | @5 |
|---|---|---|
| HitRate | 0.90 | 0.90 |
| Recall | 0.2876 | 0.3919 |
| Precision | 0.4333 | 0.3800 |

**MRR: 0.7476**

### Euclidean Distance

| Metric | @3 | @5 |
|---|---|---|
| HitRate | 0.90 | **1.00** |
| Recall | 0.2826 | 0.4002 |
| Precision | 0.4333 | 0.3800 |

**MRR: 0.7583**

### Dot Product

| Metric | @3 | @5 |
|---|---|---|
| HitRate | 0.90 | 0.90 |
| Recall | **0.3019** | 0.3719 |
| Precision | **0.4667** | 0.3600 |

**MRR: 0.7611** ← best overall

## Summary

All three functions performed closely on this dataset. Dot Product edges out the others on MRR and Precision@3, meaning it ranked the first relevant page highest most often and had the least noise in its top-3 results. Euclidean Distance achieved perfect HitRate@5 (every question had at least one relevant page in the top 5) and the best Recall@5.

| Winner | Metric |
|---|---|
| Dot Product | MRR, Precision@3, Recall@3 |
| Euclidean Distance | HitRate@5, Recall@5 |
| Cosine Similarity | No outright wins; competitive across all |

## Scalability Note

This approach works well for small corpora (tens to low hundreds of documents). It does **not** scale to large datasets because:

- All vectors are held in memory as Python lists
- Scoring is O(n × d) per query (n = docs, d = vocab size), with no indexing
- TF-IDF vocab grows with corpus size and becomes sparse and high-dimensional

For larger-scale use, the same three similarity functions can be applied inside a vector database (e.g. pgvector, Qdrant, Chroma) or an approximate nearest-neighbour index (e.g. FAISS, HNSW), and paired with neural sentence embeddings instead of TF-IDF vectors for better semantic coverage.

## Glossary

| Term | Full Name | Definition |
|---|---|---|
| TF-IDF | Term Frequency–Inverse Document Frequency | A numerical statistic that reflects how important a word is to a document relative to a collection. High TF-IDF means the word appears often in this document but rarely across others. |
| MRR | Mean Reciprocal Rank | The average of 1/rank across all queries, where rank is the position of the first relevant result. A score of 1.0 means the first relevant page was always ranked first. |
| RAG | Retrieval-Augmented Generation | A pattern where a language model's answer is grounded by first retrieving relevant source documents and passing them as context. |
| FAISS | Facebook AI Similarity Search | An open-source library from Meta for efficient approximate nearest-neighbour search over dense vectors. Suitable for large-scale retrieval. |
| HNSW | Hierarchical Navigable Small World | A graph-based approximate nearest-neighbour index algorithm. Used inside many vector databases for fast, scalable vector search. |
| ANN | Approximate Nearest Neighbour | A class of algorithms that find vectors close to a query vector without exhaustively comparing every entry, trading a small accuracy loss for significant speed gains. |
| pgvector | PostgreSQL Vector Extension | A PostgreSQL extension that adds a vector column type and supports cosine, euclidean, and dot product similarity queries directly in SQL. |
| O(n × d) | Big-O notation | A measure of computational complexity. Here n = number of documents and d = vocabulary/vector dimension. Cost grows linearly with both, with no shortcut index. |
