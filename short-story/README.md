# Short Story Dataset

A short-story dataset inspired by The 100, prepared for retrieval-augmented generation (RAG) testing.

## Overview

The dataset is organized as a single story split into page files, plus summary and evaluation assets.

## Dataset Structure

| File | Description |
|---|---|
| `pg1.md` – `pg20.md` | Story pages — intentionally small to simulate chunk-level retrieval |
| `_summary.md` | Story summary, entities, and 10 RAG test questions with expected answers |
| `rag-eval.json` | Machine-readable evaluation set for automated testing |
| `rag-eval.py` | Retrieval evaluation script (TF-IDF, no external dependencies) |
| `rag-eval.md` | [Evaluation results and methodology](rag-eval.md) |

## Intended Use

Use this data to test:

- Retrieval quality across multi-file narrative content
- Answer faithfulness to source pages
- Citation quality using supporting document references
- End-to-end RAG evaluation workflows

## Notes

- The page files are intentionally small to simulate chunk-level retrieval.
- The questions in `_summary.md` and `rag-eval.json` are designed to require cross-page synthesis.

---

## Running `rag-eval.py`

The script requires **Python 3.8+** and uses only the standard library — no `pip install` needed.

### Run from the repository root

```bash
python short-story/rag-eval.py
```

Example output

```json
{
  "cosine_similarity": {
    "hit_rate": {
      "@3": 0.9,
      "@5": 0.9
    },
    "recall": {
      "@3": 0.2876190476190476,
      "@5": 0.39190476190476187
    },
    "precision": {
      "@3": 0.4333333333333334,
      "@5": 0.38
    },
    "mrr": 0.7476190476190476
  },
  "euclidean_distance": {
    "hit_rate": {
      "@3": 0.9,
      "@5": 1.0
    },
    "recall": {
      "@3": 0.2826190476190476,
      "@5": 0.4002380952380952
    },
    "precision": {
      "@3": 0.4333333333333333,
      "@5": 0.38
    },
    "mrr": 0.7583333333333333
  },
  "dot_product": {
    "hit_rate": {
      "@3": 0.9,
      "@5": 0.9
    },
    "recall": {
      "@3": 0.30190476190476184,
      "@5": 0.37190476190476185
    },
    "precision": {
      "@3": 0.4666666666666666,
      "@5": 0.36
    },
    "mrr": 0.7611111111111111
  }
}
```

### What the script does

1. Loads all `pg*.md` page files and builds TF-IDF vectors in pure Python.
2. Loads the 10 questions from `rag-eval.json`, each with a `support_docs` list of expected relevant pages.
3. Runs three retrieval functions — **Cosine Similarity**, **Euclidean Distance**, and **Dot Product** — against every question.
4. Reports HitRate, Recall, Precision (at k=3 and k=5), and MRR for each function.

See [rag-eval.md](rag-eval.md) for full results and methodology.
