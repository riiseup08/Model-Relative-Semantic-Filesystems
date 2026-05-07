# pymrsf — Novelty-Aware RAG Chunk Scoring

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-33%20passing-brightgreen)]()

**Stop wasting context window on information your LLM already knows.**

`pymrsf` scores RAG chunks by measuring **information gain** — not just relevance. It uses the model's own predictive surprise to detect which chunks contain genuinely new information.

## The Problem

Standard RAG retrieves chunks by *relevance* (cosine similarity). But a chunk can be highly relevant while containing *only facts the model already memorized during training*. You waste precious context window on redundant information.

Also, if the LLM already *knows the answer* to the query, even novel chunks are less useful. And if two chunks say the same thing, you don't need both.

## The Solution

`pymrsf` introduces **multi-factor novelty-aware scoring**:

| Factor | What It Measures | Weight |
|--------|-----------------|--------|
| **Novelty** | How much *new* information does this chunk contain? | 40% |
| **Relevance** | How related is this chunk to the query? | 40% |
| **Query Ignorance** | Does the model *not* know the answer to your question? | 20% |
| **Diversity** | Does a better chunk already cover this content? | Dedup |

## Quick Start

```python
from pymrsf.rag import filter_chunks

# Your retrieved chunks
chunks = [
    "Backpropagation computes gradients using the chain rule.",
    "Neural networks are inspired by the human brain.",
    "The sky is blue because of Rayleigh scattering.",
]

# Filter to only useful chunks
query = "How does backpropagation work?"
useful = filter_chunks(chunks, query, min_rag_score=50, verbose=True)

# → Pass only useful chunks to your LLM
answer = llm.complete(query, context=useful)
```

## Installation

```bash
pip install llama-cpp-python faiss-cpu msgpack tiktoken
git clone https://github.com/riiseup08/mrsf.git
cd mrsf
pip install -e .
```

## Features

### 🎯 RAG Chunk Scoring (Core Feature)

```python
from pymrsf.rag import score_chunk, score_chunks, score_chunks_batch

# Single chunk scoring
result = score_chunk(
    "Backpropagation computes gradients using the chain rule.",
    query="How does backpropagation work?",
    verbose=True
)
print(result["rag_score"])    # 72/100
print(result["verdict"])      # "good"
print(result["query_knowledge"])  # how much model knows the query

# Batch scoring (3-5x faster for many chunks)
results = score_chunks_batch(chunks, query)

# Custom weights (adjust the formula)
weights = {"novelty": 0.5, "relevance": 0.3, "query_ignorance": 0.2}
result = score_chunk(chunk, query, weights=weights)
```

### 🔍 Knowledge Probing

```python
from pymrsf import probe

result = probe("To be or not to be, that is the question.")
print(f"Knowledge: {result['knowledge_score']}/100 ({result['label']})")
# → Knowledge: 92/100 (memorized) — Shakespeare is well-known

result = probe("My proprietary algorithm uses a novel attention mechanism.")
print(f"Knowledge: {result['knowledge_score']}/100 ({result['label']})")
# → Knowledge: 15/100 (unknown) — novel content!
```

### 🔧 RAG Pipeline Filter

```python
from pymrsf.rag import filter_chunks

chunks = retriever.get(query, top_k=20)   # your retriever

# Only keep chunks worth sending to the LLM
good = filter_chunks(
    chunks,
    query,
    min_rag_score=50,      # skip low-value chunks
    top_k=5,                # limit context window usage
    diversity_threshold=0.85,  # dedup similar chunks
    verbose=True,
)

answer = llm.complete(query, context=good)
```

### 📦 Delta Compression (Experimental)

Store text efficiently using LLM surprises:

```python
from pymrsf import mrsf_write, mrsf_read, save_index

# Write (stores only surprise tokens = ~40% compression)
mrsf_write("The Eiffel Tower is in Paris.")
save_index()

# Read (reconstructs from delta + model)
results = mrsf_read("famous landmark in France")
```

## Configuration

Create a `.env` file:

```bash
PYMRSF_PROVIDER=local
PYMRSF_MODEL_PATH=./models/mistral-7b-v0.1.Q4_K_M.gguf
```

## Scoring Concepts

### RAG Score Formula
```
rag_score = novelty × 0.40 + relevance × 0.40 + query_ignorance × 0.20
```

### What the Scores Mean

| Score | Verdict | Action |
|-------|---------|--------|
| 80-100 | Excellent | Prioritize this chunk |
| 60-79 | Good | Include in context |
| 40-59 | Moderate | Include if space allows |
| 20-39 | Weak | Skip if better chunks exist |
| 0-19 | Skip | Model already knows this |

## Project Structure

```
pymrsf/
├── __init__.py     # Public API exports
├── core.py         # Provider routing (local + openai), lazy model loading
├── embeddings.py   # Ollama embedding API client
├── probe.py        # Knowledge probing (how well does model know a text?)
├── rag.py          # RAG chunk scoring with novelty + relevance + diversity
├── storage.py      # Delta compression storage (experimental)
├── inspect.py      # Token-level visualization tools
└── benchmark.py    # Compression/latency benchmarks
```

## Project Status

**Alpha** — The RAG novelty scoring works and solves a real problem. The delta compression/storage system is experimental.

## License

MIT
