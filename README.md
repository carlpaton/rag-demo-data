# RAG Demo Data

This repository contains a short-story dataset inspired by The 100, prepared for retrieval-augmented generation (RAG) testing.

## Overview

The dataset is organized as a single story split into page files, plus summary and evaluation assets.

## Repository Structure

- short-story/: Main dataset folder
- short-story/pg1.md to short-story/pg20.md: Story pages
- short-story/_summary.md: Story summary, entities, and 10 RAG test questions with expected answers
- short-story/rag-eval.json: Machine-readable evaluation set for automated testing

## Intended Use

Use this data to test:

- Retrieval quality across multi-file narrative content
- Answer faithfulness to source pages
- Citation quality using supporting document references
- End-to-end RAG evaluation workflows

## Notes

- The page files are intentionally small to simulate chunk-level retrieval.
- The questions in _summary.md and rag-eval.json are designed to require cross-page synthesis.
