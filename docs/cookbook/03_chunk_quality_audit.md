# Cookbook 3: Chunk Quality Audit — Find the Noise in Your RAG Corpus

Given an existing RAG corpus, score every chunk and surface the ones that
would never be useful — chunks the model already knows (low novelty) or
that are off-topic (low relevance). These are the chunks adding noise,
not signal.

## Prerequisites

```bash
pip install pymrsf[local]
ollama pull nomic-embed-text
```

## The recipe

```python
"""
cookbook_03_chunk_quality_audit.py — Audit an existing RAG corpus.

Scores every chunk, classifies it by quality tier, and produces a
report identifying problematic chunks that should be removed or rewritten.

Run with:  python cookbook_03_chunk_quality_audit.py
"""

import pymrsf
from pymrsf import score_chunk, score_chunks
from dataclasses import dataclass

# ── Step 1: Define a RAG corpus (simulating a real retrieval index) ──────────

@dataclass
class QualityTier:
    name: str
    color: str
    description: str

TIERS = [
    QualityTier("excellent", "🟢", "Novel, relevant, model doesn't know answer — keep"),
    QualityTier("good",      "🔵", "Useful — adds some information"),
    QualityTier("moderate",  "🟡", "Partially useful — could keep if space allows"),
    QualityTier("weak",      "🟠", "Low value — model knows most of it"),
    QualityTier("skip",      "🔴", "Noise — model already knows this"),
]

# A simulated RAG corpus with deliberately mixed-quality chunks
corpus = [
    # Good chunks — novel information relevant to the query
    "Vector databases index embeddings using approximate nearest neighbor "
    "search algorithms like HNSW or IVF. These indexes trade a small amount "
    "of recall for massive speed gains over brute-force search.",

    "Retrieval-Augmented Generation (RAG) combines a retrieval step with "
    "an LLM to ground generation in external knowledge. The retriever finds "
    "relevant documents, and the LLM synthesises them into an answer.",

    "The chunk size in RAG directly impacts retrieval quality. Too-small "
    "chunks lack context; too-large chunks dilute relevance. Research "
    "suggests 200-500 tokens as a sweet spot for most applications.",

    # Bad chunks — the model already knows this or it's off-topic
    "The sky is blue because of Rayleigh scattering of sunlight by the "
    "atmosphere. Shorter wavelengths (blue) scatter more than longer ones.",

    "Paris is the capital of France. It is known for the Eiffel Tower, "
    "the Louvre Museum, and its cafe culture.",

    "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius "
    "at standard atmospheric pressure.",

    # Mixed — relevant but mostly memorized
    "Machine learning is a subset of artificial intelligence. Deep learning "
    "uses neural networks with many layers. Supervised learning requires "
    "labeled training data.",
]

QUERY = "How do I choose good chunk sizes for my RAG pipeline?"

# ── Step 2: Score every chunk ────────────────────────────────────────────────

results = score_chunks(corpus, query=QUERY)

# ── Step 3: Classify into tiers ──────────────────────────────────────────────

tiered = {t.name: [] for t in TIERS}
for r in results:
    verdict = r.get("verdict", "skip")
    if verdict in tiered:
        tiered[verdict].append(r)
    else:
        tiered["skip"].append(r)

# ── Step 4: Print the audit report ───────────────────────────────────────────

print(f"\n{'═'*70}")
print(f"  RAG CHUNK QUALITY AUDIT")
print(f"{'═'*70}")
print(f"  Query: {QUERY}")
print(f"  Total chunks in corpus: {len(results)}")
print()

for tier in TIERS:
    chunks = tiered[tier.name]
    if not chunks:
        continue
    print(f"  {tier.color} {tier.name.upper()} ({len(chunks)} chunks) — {tier.description}")
    print(f"  {'─'*66}")
    for r in chunks:
        print(f"    [{r['rag_score']:>3}/100] {r['chunk_preview'][:55]}")
        print(f"          novelty={r['novelty_score']} "
              f"relevance={r['relevance_score']} "
              f"knowledge={r['knowledge_score']}")
        if r.get("query_ignorance", 0) < 20:
            print(f"          ⚠ Query ignorance low ({r['query_ignorance']}) — "
                  f"model may already know the answer")
    print()

# ── Step 5: Action summary ───────────────────────────────────────────────────

noise_chunks = tiered["skip"] + tiered["weak"]
keep_chunks  = tiered["excellent"] + tiered["good"]

print(f"{'═'*70}")
print(f"  ACTION SUMMARY")
print(f"{'═'*70}")
print(f"  Keep ({len(keep_chunks)} chunks) — add these to your LLM context")
for r in keep_chunks:
    print(f"    ✅ [{r['rag_score']:>3}/100] {r['chunk_preview'][:55]}")
print()
print(f"  Remove ({len(noise_chunks)} chunks) — these waste context window")
for r in noise_chunks:
    print(f"    ❌ [{r['rag_score']:>3}/100] {r['chunk_preview'][:55]}")
    print(f"        Reason: {r['recommendation']}")

if len(noise_chunks) > len(corpus) * 0.3:
    print(f"\n  ⚠  {len(noise_chunks)}/{len(corpus)} chunks are noise "
          f"({100 * len(noise_chunks) // len(corpus)}%).")
    print(f"  Consider revising your chunking strategy or retrieval pipeline.")
```

## Expected output

```
═ RAG CHUNK QUALITY AUDIT ═
  Query: How do I choose good chunk sizes for my RAG pipeline?
  Total chunks in corpus: 7

  🟢 EXCELLENT (1 chunks) — Novel, relevant...
    [75/100] The chunk size in RAG directly impacts retrieval quality...

  🔵 GOOD (1 chunks) — Useful...
    [50/100] Retrieval-Augmented Generation (RAG) combines a...

  🟡 MODERATE (1 chunks) — Partially useful...
    [42/100] Vector databases index embeddings using approximate...

  🟠 WEAK (1 chunks) — Low value...
    [25/100] Machine learning is a subset of artificial intelligence...

  🔴 SKIP (3 chunks) — Noise...
    [0/100] The sky is blue because of Rayleigh scattering...
    [0/100] Paris is the capital of France...
    [0/100] Water freezes at 0 degrees Celsius...

═ ACTION SUMMARY ═
  Keep (2 chunks) — add these to your LLM context
  Remove (4 chunks) — these waste context window
```

## What's happening

1. **Score all chunks** — `score_chunks` computes novelty, relevance, and
   query ignorance for each chunk in a single pass
2. **Classify** — each chunk gets a verdict: excellent / good / moderate /
   weak / skip
3. **Surface noise** — chunks the model already knows (Rayleigh scattering,
   Paris trivia, boiling point) get `skip` verdict because their
   knowledge_score is high — they'd waste context window
4. **Action** — keep the signal, remove the noise

## Why this matters

Standard RAG pipelines retrieve by cosine similarity alone. A chunk about
"Paris is the capital of France" would match a query about "France" on
semantic similarity — but the model already knows that fact. pymrsf's
novelty signal catches this and filters it out, leaving room for chunks
that actually add new information.
