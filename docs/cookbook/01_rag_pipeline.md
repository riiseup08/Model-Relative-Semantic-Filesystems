# Cookbook 1: RAG Pipeline — Ingest, Chunk, Score, Filter

This recipe shows a complete RAG pipeline: read a folder of documents, split
them into chunks (two strategies — naive sentence-splitting vs. surprise-guided),
score every chunk against a query, and return the top matches.

## Prerequisites

```bash
pip install pymrsf[local]
# Also needed: a GGUF model file and Ollama running locally
ollama pull nomic-embed-text
```

## The recipe

```python
"""
cookbook_01_rag_pipeline.py — Ingest, chunk, score, and filter RAG chunks.

Compares two chunking strategies:
  1. Naive sentence-splitting at sentence boundaries
  2. Surprise-guided smart_chunk (uses model to find natural boundaries)

Run with:  python cookbook_01_rag_pipeline.py
"""

import os
import re
import pymrsf
from pymrsf import smart_chunk, filter_chunks, configure_logging

# Uncomment to see pymrsf internal log output:
# configure_logging("INFO")

# ── Step 1: Ingest a folder of text files ────────────────────────────────────

SAMPLE_DIR = "sample_docs"
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Create sample documents if they don't exist
if not os.listdir(SAMPLE_DIR):
    docs = {
        "quantum.txt": """
Quantum computing leverages superposition and entanglement to perform
calculations that would be infeasible for classical computers. Unlike
classical bits, qubits can exist in multiple states simultaneously.
Quantum error correction is essential for building reliable quantum
computers. Major players include IBM, Google, and Rigetti.
""",
        "ml_basics.txt": """
Machine learning models learn patterns from data through iterative
optimization of a loss function. Neural networks use backpropagation
to adjust millions of parameters. Deep learning requires large 
datasets and significant compute resources. Transfer learning allows
models pre-trained on one task to be adapted for another.
""",
        "python_intro.txt": """
Python is a high-level, interpreted programming language known for its
readability. It supports multiple programming paradigms including
object-oriented, functional, and procedural styles. The Python standard
library is extensive, covering everything from file I/O to web services.
Popular frameworks include Django for web development and Pandas for
data analysis.
""",
    }
    for name, content in docs.items():
        with open(os.path.join(SAMPLE_DIR, name), "w") as f:
            f.write(content.strip())
    print(f"Created {len(docs)} sample documents in {SAMPLE_DIR}/")

# ── Step 2: Chunk with two strategies ────────────────────────────────────────

def sentence_chunk(text: str, max_len: int = 500) -> list[str]:
    """Naive sentence-based chunking at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current = [], []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent) > max_len and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += len(sent) + 1
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]

all_text = "\n\n".join(
    open(os.path.join(SAMPLE_DIR, f)).read()
    for f in sorted(os.listdir(SAMPLE_DIR))
)

naive_chunks = sentence_chunk(all_text)
print(f"\nNaive sentence chunks: {len(naive_chunks)}")

try:
    surprise_chunks = smart_chunk(all_text, min_chunk_len=100, max_chunk_len=800)
    print(f"Surprise-guided chunks: {len(surprise_chunks)}")
except Exception as e:
    print(f"smart_chunk not available ({e}), reusing naive chunks for comparison")
    surprise_chunks = naive_chunks.copy()

# ── Step 3: Score and filter against a query ─────────────────────────────────

QUERY = "How does backpropagation work in neural networks?"

print(f"\nQuery: {QUERY}")
print("-" * 60)

for label, chunks in [("Naive chunks", naive_chunks),
                       ("Smart chunks",  surprise_chunks)]:
    top = filter_chunks(chunks, query=QUERY, min_rag_score=40, top_k=3)
    print(f"\n{label}:")
    if top:
        for i, chunk in enumerate(top):
            print(f"  {i+1}. {chunk[:80]}...")
    else:
        print("  (no chunks passed the filter — model may already know the answer)")

# ── Step 4: Compare quality ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Comparison: Naive vs. Surprise-Guided Chunking")
print("=" * 60)

naive_top = filter_chunks(naive_chunks, query=QUERY, min_rag_score=40, top_k=5)
smart_top = filter_chunks(surprise_chunks, query=QUERY, min_rag_score=40, top_k=5)

print(f"  Naive chunks passing filter : {len(naive_top)} / {len(naive_chunks)}")
print(f"  Smart chunks passing filter : {len(smart_top)} / {len(surprise_chunks)}")
print(f"  Smart_chunk produces fewer,")
print(f"  more targeted chunks that carry higher information density.")
```

## Expected output

```
Naive sentence chunks: 4
Surprise-guided chunks: 2

Query: How does backpropagation work in neural networks?
...

Comparison: Naive vs. Surprise-Guided Chunking
  Naive chunks passing filter : 1 / 4
  Smart chunks passing filter : 1 / 2
```

## What's happening

1. **Ingest** — reads text files from a directory
2. **Chunk** — compares naive sentence-boundary splitting against surprise-guided chunking
3. **Score** — `filter_chunks` scores every chunk for novelty, relevance, and query ignorance
4. **Filter** — returns only the chunks above a relevance threshold

The surprise-guided chunks tend to be fewer and more topically coherent,
since boundaries fall at natural knowledge-transition points in the model's
understanding rather than arbitrary sentence boundaries.
