# Cookbook 2: Novelty Detection — Find What the Model Doesn't Know

This recipe uses `probe` to scan a corpus and identify which documents the
model has likely memorized vs. which contain genuinely novel content. Useful
for prioritizing what to add to a RAG index.

## Prerequisites

```bash
pip install pymrsf[local]
ollama pull nomic-embed-text
```

Requires the **local provider** (probing needs raw model logits).

## The recipe

```python
"""
cookbook_02_novelty_detection.py — Scan a corpus for novel content.

Documents the model already "knows" (high knowledge_score) can be
deprioritized for RAG indexing. Novel documents (low knowledge_score)
are where the model will benefit most from retrieval context.

Run with:  python cookbook_02_novelty_detection.py
"""

import pymrsf
from pymrsf import probe
from dataclasses import dataclass

# ── Step 1: Define a corpus of documents ─────────────────────────────────────

@dataclass
class Document:
    title: str
    text: str

corpus = [
    Document(
        "Shakespeare sonnet",
        "To be, or not to be, that is the question: Whether 'tis nobler "
        "in the mind to suffer the slings and arrows of outrageous fortune...",
    ),
    Document(
        "Common proverb",
        "The quick brown fox jumps over the lazy dog.",
    ),
    Document(
        "Eiffel Tower fact",
        "The Eiffel Tower was built in 1889 as the centerpiece of the "
        "World's Fair in Paris, France.",
    ),
    Document(
        "My proprietary algorithm",
        "The XYZ-9000 algorithm uses a novel quartic-time reduction "
        "that compresses the embedding space by a factor of 12.7x "
        "while preserving 99.2% of pairwise distances.",
    ),
    Document(
        "Internal meeting notes",
        "Q3 planning meeting: migrate CI pipeline to GitHub Actions, "
        "upgrade the message queue to Kafka 4.0, deprecate the legacy "
        "REST endpoint /v1/process by October 15.",
    ),
]

# ── Step 2: Probe every document ─────────────────────────────────────────────

results = []
for doc in corpus:
    result = probe(doc.text)
    if "error" in result:
        print(f"[SKIP] {doc.title}: {result['error']}")
        continue
    result["title"] = doc.title
    results.append(result)

# ── Step 3: Sort by novelty (inverse of knowledge_score) ─────────────────────

results.sort(key=lambda r: r["knowledge_score"])

print(f"\n{'═'*70}")
print(f"  CORPUS NOVELTY REPORT")
print(f"{'═'*70}")
print(f"  {'Document':<35} {'Knowledge':>10} {'Novelty':>10} {'Label':<15}")
print(f"  {'─'*70}")

for r in results:
    title = r["title"][:34]
    knowledge = r["knowledge_score"]
    novelty = 100 - knowledge
    label = r["label"]
    print(f"  {title:<35} {knowledge:>10} {novelty:>10} {label:<15}")

# ── Step 4: Separate into tiers ──────────────────────────────────────────────

memorized   = [r for r in results if r["knowledge_score"] >= 70]
familiar    = [r for r in results if 40 <= r["knowledge_score"] < 70]
novel       = [r for r in results if r["knowledge_score"] < 40]

print(f"\n{'─'*70}")
print(f"  RECOMMENDATIONS")
print(f"{'─'*70}")
print(f"  Memorized (skip for RAG): {len(memorized)} docs")
for r in memorized:
    print(f"    - {r['title']} (score={r['knowledge_score']})")

print(f"\n  Novel (prioritize for RAG): {len(novel)} docs")
for r in novel:
    print(f"    - {r['title']} (score={r['knowledge_score']})")

if familiar:
    print(f"\n  Familiar (include if space): {len(familiar)} docs")
    for r in familiar:
        print(f"    - {r['title']} (score={r['knowledge_score']})")
```

## Expected output

```
═ CORPUS NOVELTY REPORT ═
  Document                           Knowledge    Novelty   Label
  ──────────────────────────────────────────────────────────────────────
  My proprietary algorithm                   25         75   uncommon
  Internal meeting notes                     30         70   uncommon
  Eiffel Tower fact                          65         35   familiar
  Shakespeare sonnet                         92          8   memorized
  Common proverb                             95          5   memorized

  RECOMMENDATIONS
  ──────────────────────────────────────────────────────────────────────
  Memorized (skip for RAG): 2 docs
    - Shakespeare sonnet (score=92)
    - Common proverb (score=95)

  Novel (prioritize for RAG): 2 docs
    - My proprietary algorithm (score=25)
    - Internal meeting notes (score=30)
```

## What's happening

1. **Probe** each document — the model reveals how much of the content it
   already "knows" by how often it correctly predicts the next token
2. **Invert** the knowledge score to get novelty (100 - knowledge)
3. **Tier** documents into memorized / familiar / novel buckets
4. **Act** — prioritise novel content for RAG indexing; skip memorized content
   that would waste context window

This is the core insight that differentiates pymrsf from cosine-similarity
retrieval: a chunk can be semantically related but contain only facts the
model already memorized. Probing catches that case.
